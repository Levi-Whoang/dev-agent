import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from api.api_schemas import ChatRequest, CreateSessionRequest


class FakeSessionManager:
    def __init__(self):
        self.sessions = {}

    def create(self, session_id):
        if session_id in self.sessions:
            raise ValueError("duplicate session")
        self.sessions[session_id] = []

    def exists(self, session_id):
        return session_id in self.sessions

    def get_messages(self, session_id):
        return list(self.sessions.get(session_id, []))

    def clear(self, session_id):
        if session_id in self.sessions:
            self.sessions[session_id] = []

    def delete(self, session_id):
        self.sessions.pop(session_id, None)

    def list_sessions(self, prefix):
        return [
            {"session_id": key}
            for key in self.sessions
            if key.startswith(prefix)
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
    fake_agent = FakeAgent()
    fake_main = ModuleType("main")
    fake_main.agent = fake_agent
    fake_logging = ModuleType("logging_config")
    fake_logging.logger = MagicMock()
    monkeypatch.setitem(sys.modules, "main", fake_main)
    monkeypatch.setitem(sys.modules, "logging_config", fake_logging)

    # 防止导入应用时创建真实 OpenAI 客户端或写入日志。
    sys.modules.pop("app", None)
    module = importlib.import_module("app")
    try:
        yield module, fake_agent
    finally:
        sys.modules.pop("app", None)


def test_chat_route_calls_agent_and_returns_standard_response(api_module):
    app, agent = api_module
    user_id = "test-user"
    stored_session_id = app._stored_session_id(user_id, "chat-1")

    response = app.chat(
        ChatRequest(message="你好", session_id="chat-1", stream=False),
        current_user=user_id,
    )

    assert response == {
        "code": 0,
        "message": "success",
        "data": {
            "answer": "回答：你好",
            "session_id": "chat-1",
            "citations": [],
        },
    }
    assert agent.calls == [("你好", stored_session_id)]


def test_chat_route_streams_sse_deltas_and_completion_event(api_module):
    app, agent = api_module
    user_id = "test-user"
    stored_session_id = app._stored_session_id(user_id, "stream-1")
    response = app.chat(
        ChatRequest(message="你好", session_id="stream-1"),
        current_user=user_id,
    )
    events = list(app.chat_event_stream("你好", "stream-1", stored_session_id))

    assert response.media_type == "text/event-stream"
    assert response.headers["cache-control"] == "no-cache"
    body = "".join(events)

    assert '"type": "delta", "content": "回答：你"' in body
    assert '"type": "delta", "content": "好"' in body
    assert '"type": "done", "session_id": "stream-1"' in body
    assert agent.stream_calls == [("你好", stored_session_id)]


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
    app, _ = api_module
    user_id = "test-user"
    app.create_session(CreateSessionRequest(session_id="session-1"), current_user=user_id)
    assert app.list_sessions(current_user=user_id)["data"] == [{"session_id": "session-1"}]
    assert app.get_session_messages("session-1", current_user=user_id)["data"]["messages"] == []

    assert app.clear_session("session-1", current_user=user_id)["message"] == "会话已清空"
    assert app.delete_session("session-1", current_user=user_id)["message"] == "会话删除成功"

    with pytest.raises(HTTPException) as error:
        app.get_session_messages("session-1", current_user=user_id)
    assert error.value.status_code == 404


def test_create_duplicate_session_raises_conflict(api_module):
    app, _ = api_module
    request = CreateSessionRequest(session_id="duplicate")
    app.create_session(request, current_user="test-user")

    with pytest.raises(HTTPException) as error:
        app.create_session(request, current_user="test-user")

    assert error.value.status_code == 409
