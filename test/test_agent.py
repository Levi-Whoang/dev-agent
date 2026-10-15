
import json
from unittest.mock import MagicMock

from agent.agent import Agent


class FakeSessionManager:
    """测试用会话管理器，不连接真实 MySQL。"""

    def __init__(self):
        self.sessions = {}

    def get_messages(self, session_id):
        return list(self.sessions.get(session_id, []))

    def save(self, session_id, messages):
        self.sessions[session_id] = list(messages)


def test_agent_tool_call(monkeypatch):
    def calculate(a: int, b: int):
        return a + b

    # 替换真实 SessionManager，避免测试连接 MySQL
    monkeypatch.setattr(
        "agent.agent.SessionManager",
        FakeSessionManager,
    )

    arguments = json.dumps({"a": 10, "b": 20})

    tool_call = MagicMock()
    tool_call.id = "call_123"
    tool_call.function.name = "calculate"
    tool_call.function.arguments = arguments

    first_message = MagicMock()
    first_message.tool_calls = [tool_call]
    first_message.model_dump.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "calculate",
                    "arguments": arguments,
                },
            }
        ],
    }

    final_message = MagicMock()
    final_message.tool_calls = None
    final_message.content = "计算结果是 30"
    final_message.model_dump.return_value = {
        "role": "assistant",
        "content": "计算结果是 30",
    }

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=first_message)]),
        MagicMock(choices=[MagicMock(message=final_message)]),
    ]

    agent = Agent(
        client=client,
        tools=[],
        tool_registry={"calculate": calculate},
    )

    result = agent.run("计算 10 + 20", session_id="test_user")

    assert result == "计算结果是 30"
    assert client.chat.completions.create.call_count == 2

    messages = agent.session_manager.get_messages("test_user")
    assert any(message.get("role") == "tool" for message in messages)
    assert messages[-1]["content"] == "计算结果是 30"
