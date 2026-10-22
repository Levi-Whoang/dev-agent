import importlib
import hashlib
import json
import sys
from types import ModuleType
from unittest.mock import MagicMock

import anyio
import httpx
import pytest

ALICE_HEADERS = {}
BOB_HEADERS = {}


def stored_session_id(user_id, session_id):
    owner_key = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:24]
    return f"{owner_key}:{session_id}"


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

    def list_sessions(self, prefix):
        return [
            {"session_id": key}
            for key in self.sessions
            if key.startswith(prefix)
        ]


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
    from auth.passwords import hash_password

    monkeypatch.setenv(
        "AUTH_USERS_JSON",
        json.dumps({
            "alice": hash_password("alice-password", iterations=100_000),
            "bob": hash_password("bob-password", iterations=100_000),
        }),
    )
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-with-more-than-32-characters")
    monkeypatch.setenv("JWT_EXPIRE_MINUTES", "30")
    fake_agent = ASGIFakeAgent()
    fake_main = ModuleType("main")
    fake_main.agent = fake_agent
    fake_logging = ModuleType("logging_config")
    fake_logging.logger = MagicMock()
    monkeypatch.setitem(sys.modules, "main", fake_main)
    monkeypatch.setitem(sys.modules, "logging_config", fake_logging)
    sys.modules.pop("app", None)
    app_module = importlib.import_module("app")
    global ALICE_HEADERS, BOB_HEADERS
    alice_token, _ = app_module.create_access_token("alice")
    bob_token, _ = app_module.create_access_token("bob")
    ALICE_HEADERS = {"Authorization": f"Bearer {alice_token}"}
    BOB_HEADERS = {"Authorization": f"Bearer {bob_token}"}

    try:
        yield app_module.app, fake_agent
    finally:
        sys.modules.pop("app", None)


def test_asgi_health_and_sse_chat_with_credentials(asgi_app):
    app, fake_agent = asgi_app
    fake_agent.session_manager.sessions[stored_session_id("alice", "asgi-1")] = [
        {"role": "user", "content": "查询办公地点"},
        {
            "role": "tool",
            "content": json.dumps({
                "citations": [
                    {
                        "source": "company.txt",
                        "content": "办公室位于深圳市南山区。",
                        "score": 0.91,
                        "dense_score": 0.88,
                        "keyword_score": 1.2,
                    }
                ]
            }, ensure_ascii=False),
        },
    ]

    async def make_requests():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers=ALICE_HEADERS,
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
        {
            "type": "done",
            "session_id": "asgi-1",
            "citations": [
                {
                    "source": "company.txt",
                    "content": "办公室位于深圳市南山区。",
                    "score": 0.91,
                    "dense_score": 0.88,
                    "keyword_score": 1.2,
                }
            ],
        },
    ]
    assert fake_agent.calls == [
        ("stream", "测试流式", stored_session_id("alice", "asgi-1"))
    ]


def test_asgi_json_chat_returns_citations_from_latest_turn(asgi_app):
    app, fake_agent = asgi_app
    citation = {
        "source": "faq.txt",
        "content": "办公地点位于深圳市南山区。",
        "score": 0.82,
        "dense_score": 0.82,
        "keyword_score": 0.0,
    }
    fake_agent.session_manager.sessions[
        stored_session_id("alice", "asgi-citation")
    ] = [
        {"role": "user", "content": "公司在哪里？"},
        {"role": "tool", "content": json.dumps({"citations": [citation]})},
    ]

    async def make_request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers=ALICE_HEADERS,
        ) as client:
            return await client.post(
                "/chat",
                json={
                    "message": "公司在哪里？",
                    "session_id": "asgi-citation",
                    "stream": False,
                },
            )

    response = anyio.run(make_request)
    assert response.status_code == 200
    assert response.json()["data"]["citations"] == [citation]


def test_asgi_preserves_json_chat_mode_and_validates_requests(asgi_app):
    app, fake_agent = asgi_app

    async def make_requests():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers=ALICE_HEADERS,
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
    assert fake_agent.calls == [
        ("json", "兼容请求", stored_session_id("alice", "asgi-2"))
    ]


def test_asgi_session_routes_use_injected_memory_store(asgi_app):
    app, _ = asgi_app

    async def make_requests():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers=ALICE_HEADERS,
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


def test_asgi_requires_authentication_except_health(asgi_app):
    app, _ = asgi_app

    async def make_requests():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            health = await client.get("/health")
            sessions = await client.get("/sessions")
            bad_token = await client.get(
                "/sessions",
                headers={"Authorization": "Bearer wrong-token"},
            )
            return health, sessions, bad_token

    health, sessions, bad_token = anyio.run(make_requests)
    assert health.status_code == 200
    assert sessions.status_code == 401
    assert sessions.headers["www-authenticate"] == "Bearer"
    assert bad_token.status_code == 401


def test_asgi_login_issues_jwt_and_rejects_invalid_credentials(asgi_app):
    app, _ = asgi_app

    async def make_requests():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            success = await client.post(
                "/auth/token",
                json={"username": "alice", "password": "alice-password"},
            )
            failure = await client.post(
                "/auth/token",
                json={"username": "alice", "password": "wrong-password"},
            )
            return success, failure

    success, failure = anyio.run(make_requests)
    assert success.status_code == 200
    assert success.json()["data"]["token_type"] == "bearer"
    assert success.json()["data"]["expires_in"] == 1800
    assert success.json()["data"]["access_token"]
    assert failure.status_code == 401


def test_asgi_session_ownership_and_list_are_scoped_per_user(asgi_app):
    app, fake_agent = asgi_app

    async def make_requests():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            created = await client.post(
                "/sessions",
                json={"session_id": "alice-private"},
                headers=ALICE_HEADERS,
            )
            alice_list = await client.get("/sessions", headers=ALICE_HEADERS)
            bob_list = await client.get("/sessions", headers=BOB_HEADERS)
            bob_read = await client.get(
                "/sessions/alice-private/messages",
                headers=BOB_HEADERS,
            )
            bob_clear = await client.post(
                "/sessions/alice-private/clear",
                headers=BOB_HEADERS,
            )
            bob_delete = await client.delete(
                "/sessions/alice-private",
                headers=BOB_HEADERS,
            )
            return created, alice_list, bob_list, bob_read, bob_clear, bob_delete

    created, alice_list, bob_list, bob_read, bob_clear, bob_delete = anyio.run(
        make_requests
    )
    assert created.status_code == 200
    assert [item["session_id"] for item in alice_list.json()["data"]] == [
        "alice-private"
    ]
    assert bob_list.json()["data"] == []
    assert bob_read.status_code == 404
    assert bob_clear.status_code == 404
    assert bob_delete.status_code == 404
    assert len(fake_agent.session_manager.sessions) == 1


def test_asgi_chat_without_session_id_creates_unique_user_sessions(asgi_app):
    app, fake_agent = asgi_app

    async def make_requests():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            alice = await client.post(
                "/chat",
                json={"message": "Alice 的新对话", "stream": False},
                headers=ALICE_HEADERS,
            )
            bob = await client.post(
                "/chat",
                json={"message": "Bob 的新对话", "stream": False},
                headers=BOB_HEADERS,
            )
            return alice, bob

    alice, bob = anyio.run(make_requests)
    alice_session_id = alice.json()["data"]["session_id"]
    bob_session_id = bob.json()["data"]["session_id"]
    assert alice.status_code == 200
    assert bob.status_code == 200
    assert alice_session_id != bob_session_id
    assert len(fake_agent.session_manager.sessions) == 2
    assert all(call[2].count(":") == 1 for call in fake_agent.calls)
