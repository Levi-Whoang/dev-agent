import importlib
import json
import sys
from types import ModuleType
from unittest.mock import MagicMock

import anyio
import httpx
import pytest


class MemorySessionManager:
    def __init__(self):
        self.sessions = {}

    def create(self, session_id):
        if session_id in self.sessions:
            raise ValueError("duplicate")
        self.sessions[session_id] = []

    def exists(self, session_id):
        return session_id in self.sessions

    def get_messages(self, session_id):
        return list(self.sessions.get(session_id, []))

    def clear(self, session_id):
        self.sessions[session_id] = []

    def delete(self, session_id):
        self.sessions.pop(session_id, None)

    def list_sessions(self):
        return [{"session_id": key} for key in self.sessions]


class ASGIFakeAgent:
    def __init__(self):
        self.session_manager = MemorySessionManager()
        self.calls = []

    def run(self, message, session_id):
        self.calls.append(("json", message, session_id))
        return f"完整回答：{message}"

    def run_stream(self, message, session_id):
        self.calls.append(("stream", message, session_id))
        yield "第一段"
        yield "第二段"


@pytest.fixture
def asgi_app(monkeypatch):
    fake_agent = ASGIFakeAgent()
    fake_main = ModuleType("main")
    fake_main.agent = fake_agent
    fake_logging = ModuleType("logging_config")
    fake_logging.logger = MagicMock()
    monkeypatch.setitem(sys.modules, "main", fake_main)
    monkeypatch.setitem(sys.modules, "logging_config", fake_logging)
    sys.modules.pop("app", None)
    app_module = importlib.import_module("app")

    try:
        yield app_module.app, fake_agent
    finally:
        sys.modules.pop("app", None)


def test_asgi_health_and_sse_chat_without_credentials(asgi_app):
    app, fake_agent = asgi_app

    async def make_requests():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            health = await client.get("/health")
            chat = await client.post(
                "/chat",
                json={"message": "测试流式", "session_id": "asgi-1"},
            )
            return health, chat

    health, chat = anyio.run(make_requests)

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert chat.status_code == 200
    assert chat.headers["content-type"].startswith("text/event-stream")
    events = [
        json.loads(line.removeprefix("data: "))
        for line in chat.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events == [
        {"type": "delta", "content": "第一段"},
        {"type": "delta", "content": "第二段"},
        {"type": "done", "session_id": "asgi-1"},
    ]
    assert fake_agent.calls == [("stream", "测试流式", "asgi-1")]


def test_asgi_preserves_json_chat_mode_and_validates_requests(asgi_app):
    app, fake_agent = asgi_app

    async def make_requests():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            chat = await client.post(
                "/chat",
                json={
                    "message": "兼容请求",
                    "session_id": "asgi-2",
                    "stream": False,
                },
            )
            invalid = await client.post("/chat", json={"message": ""})
            return chat, invalid

    chat, invalid = anyio.run(make_requests)

    assert chat.status_code == 200
    assert chat.json()["data"]["answer"] == "完整回答：兼容请求"
    assert invalid.status_code == 422
    assert invalid.json()["code"] == 422
    assert fake_agent.calls == [("json", "兼容请求", "asgi-2")]


def test_asgi_session_routes_use_injected_memory_store(asgi_app):
    app, _ = asgi_app

    async def make_requests():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            created = await client.post("/sessions", json={"session_id": "asgi-3"})
            listed = await client.get("/sessions")
            deleted = await client.delete("/sessions/asgi-3")
            missing = await client.get("/sessions/asgi-3/messages")
            return created, listed, deleted, missing

    created, listed, deleted, missing = anyio.run(make_requests)

    assert created.status_code == 200
    assert listed.json()["data"] == [{"session_id": "asgi-3"}]
    assert deleted.status_code == 200
    assert missing.status_code == 404
