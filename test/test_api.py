import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from api.api_schemas import ChatRequest, CreateSessionRequest, SessionTitleRequest
from api.chat_helpers import chat_event_stream
from api.session_helpers import stored_session_id


class FakeSessionManager:
    def __init__(self):
        self.sessions = {}
        self.titles = {}

    def create(self, session_id, owner_user_id=None, title="新对话"):
        if session_id in self.sessions:
            raise ValueError("duplicate session")
        self.sessions[session_id] = []
        self.titles[session_id] = title

    def exists(self, session_id):
        return session_id in self.sessions

    def get_messages(self, session_id):
        return list(self.sessions.get(session_id, []))

    def clear(self, session_id):
        if session_id in self.sessions:
            self.sessions[session_id] = []

    def delete(self, session_id):
        self.sessions.pop(session_id, None)
        self.titles.pop(session_id, None)

    def rename(self, session_id, owner_user_id, title):
        if session_id not in self.sessions:
            return False
        self.titles[session_id] = title
        return True

    def list_sessions(self, owner_user_id):
        return [
            {"session_id": key, "title": self.titles.get(key, "新对话")}
            for key in self.sessions
        ]


class FakeAgent:
    def __init__(self):
        self.session_manager = FakeSessionManager()
        self.calls = []
        self.stream_calls = []
        self.error = None

    def run(self, message, session_id):
        self.calls.append((message, session_id))
        if self.error is not None:
            raise self.error
        return f"回答：{message}"

    def run_stream(self, message, session_id):
        self.stream_calls.append((message, session_id))
        yield f"回答：{message[:1]}"
        yield message[1:]


@pytest.fixture
def api_module(monkeypatch):
    import auth.jwt_auth as jwt_auth

    class MemoryAuthRepository:
        def find_user(self, username):
            return {"id": 1, "username": username}

    monkeypatch.setattr(jwt_auth, "auth_repository", MemoryAuthRepository())
    fake_agent = FakeAgent()
    fake_runtime = ModuleType("agent.runtime")
    fake_runtime.agent = fake_agent
    fake_logs = ModuleType("logs")
    fake_logs.__path__ = []
    fake_logging = ModuleType("logs.logging_config")
    fake_logging.logger = MagicMock()
    monkeypatch.setitem(sys.modules, "logs", fake_logs)
    monkeypatch.setitem(sys.modules, "logs.logging_config", fake_logging)
    monkeypatch.setitem(sys.modules, "agent.runtime", fake_runtime)

    # 防止导入应用时创建真实 OpenAI 客户端或写入日志。
    for module_name in (
        "app", "app.main", "agent.runtime", "api.routes", "api.routes.sessions",
        "api.routes.users", "api.routes.documents", "app.errors",
    ):
        sys.modules.pop(module_name, None)
    module = importlib.import_module("api.routes.sessions")
    try:
        yield module, fake_agent, fake_logging.logger
    finally:
        for module_name in (
            "app", "app.main", "agent.runtime", "api.routes", "api.routes.sessions",
            "api.routes.users", "api.routes.documents", "app.errors",
        ):
            sys.modules.pop(module_name, None)


def test_chat_route_streams_sse_deltas_and_completion_event(api_module):
    routes, agent, logger = api_module
    user_id = "test-user"
    internal_session_id = stored_session_id(user_id, "stream-1")
    response = routes.chat(
        ChatRequest(message="你好", session_id="stream-1"),
        current_user=user_id,
    )
    events = list(
        chat_event_stream(agent, logger, "你好", "stream-1", internal_session_id)
    )

    assert response.media_type == "text/event-stream"
    assert response.headers["cache-control"] == "no-cache"
    body = "".join(events)

    assert '"type": "delta", "content": "回答：你"' in body
    assert '"type": "delta", "content": "好"' in body
    assert '"type": "done", "session_id": "stream-1"' in body
    assert agent.stream_calls == [("你好", internal_session_id)]


@pytest.mark.parametrize(
    "message",
    ["", "x" * 4001],
    ids=["empty", "too-long"],
)
def test_chat_request_rejects_invalid_message_lengths(message):
    with pytest.raises(ValidationError):
        ChatRequest(message=message)


def test_chat_request_has_no_shared_default_session():
    assert ChatRequest(message="新对话").session_id is None


def test_session_routes_cover_create_list_read_clear_delete_and_missing(api_module):
    routes, _, _ = api_module
    user_id = "test-user"
    routes.create_session(CreateSessionRequest(session_id="session-1"), current_user=user_id)
    assert routes.list_sessions(current_user=user_id)["data"] == [
        {"session_id": "session-1", "title": "新对话"}
    ]
    assert routes.get_session_messages("session-1", current_user=user_id)["data"]["messages"] == []

    assert routes.clear_session("session-1", current_user=user_id)["message"] == "会话已清空"
    assert routes.delete_session("session-1", current_user=user_id)["message"] == "会话删除成功"

    with pytest.raises(HTTPException) as error:
        routes.get_session_messages("session-1", current_user=user_id)
    assert error.value.status_code == 404


def test_create_duplicate_session_raises_conflict(api_module):
    routes, _, _ = api_module
    request = CreateSessionRequest(session_id="duplicate")
    routes.create_session(request, current_user="test-user")

    with pytest.raises(HTTPException) as error:
        routes.create_session(request, current_user="test-user")

    assert error.value.status_code == 409


def test_rename_session_updates_the_saved_title(api_module):
    routes, _, _ = api_module
    user_id = "test-user"
    routes.create_session(CreateSessionRequest(session_id="rename-me"), current_user=user_id)

    response = routes.rename_session(
        "rename-me",
        SessionTitleRequest(title="  新的   会话名称  "),
        current_user=user_id,
    )

    assert response["data"]["title"] == "新的 会话名称"
    assert routes.list_sessions(current_user=user_id)["data"][0]["title"] == "新的 会话名称"
