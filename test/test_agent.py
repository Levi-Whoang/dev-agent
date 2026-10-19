import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import agent.agent as agent_module
from agent.agent import Agent


def calculate(a: int, b: int):
    return a + b


def explode(value: int):
    raise RuntimeError("boom")


class FakeSessionManager:
    """内存会话存储，确保测试不连接 MySQL。"""

    def __init__(self):
        self.sessions = {}

    def get_messages(self, session_id):
        return list(self.sessions.get(session_id, []))

    def save(self, session_id, messages):
        self.sessions[session_id] = list(messages)


def make_tool_call(name, arguments, call_id="call_123"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def make_message(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)

    def model_dump(exclude_none=True):
        result = {"role": "assistant"}
        if content is not None or not exclude_none:
            result["content"] = content
        if tool_calls:
            result["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls
            ]
        elif exclude_none:
            result = {key: value for key, value in result.items() if value is not None}
        return result

    message.model_dump = model_dump
    return message


def llm_response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_agent(monkeypatch, *messages, tool_registry=None):
    monkeypatch.setattr(agent_module, "SessionManager", FakeSessionManager)
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        llm_response(message) for message in messages
    ]
    agent = Agent(
        client=client,
        tools=[{"type": "function", "function": {"name": "stub"}}],
        tool_registry=tool_registry if tool_registry is not None else {},
    )
    return agent, client


def get_tool_message(agent, session_id):
    messages = agent.session_manager.get_messages(session_id)
    return next(message for message in messages if message.get("role") == "tool")


def test_agent_executes_tool_and_persists_final_answer(monkeypatch):
    call = make_tool_call("calculate", json.dumps({"a": 10, "b": 20}))
    agent, client = make_agent(
        monkeypatch,
        make_message(tool_calls=[call]),
        make_message(content="计算结果是 30"),
        tool_registry={"calculate": calculate},
    )

    result = agent.run("计算 10 + 20", session_id="test_user")

    assert result == "计算结果是 30"
    assert client.chat.completions.create.call_count == 2
    first_request = client.chat.completions.create.call_args_list[0].kwargs
    assert first_request["tool_choice"] == "auto"
    assert first_request["messages"][0]["role"] == "system"

    messages = agent.session_manager.get_messages("test_user")
    assert any(message.get("role") == "tool" for message in messages)
    assert messages[-1]["content"] == "计算结果是 30"
    assert json.loads(get_tool_message(agent, "test_user")["content"]) == 30


@pytest.mark.parametrize(
    ("tool_name", "arguments", "registry", "expected_error"),
    [
        (
            "calculate",
            '{"a": 1',
            {},
            "工具参数格式错误",
        ),
        (
            "missing_tool",
            "{}",
            {},
            "工具不存在",
        ),
        (
            "calculate",
            json.dumps({"a": 1}),
            {"calculate": calculate},
            "工具参数校验失败",
        ),
        (
            "explode",
            json.dumps({"value": 1}),
            {"explode": explode},
            "工具执行失败: boom",
        ),
        (
            "untyped",
            json.dumps({"value": 1}),
            {"untyped": lambda value: value},
            "工具执行前发生错误",
        ),
    ],
    ids=[
        "malformed-json",
        "unknown-tool",
        "invalid-arguments",
        "tool-exception",
        "invalid-tool-schema",
    ],
)
def test_agent_returns_tool_errors_to_the_model(
    monkeypatch,
    tool_name,
    arguments,
    registry,
    expected_error,
):
    call = make_tool_call(tool_name, arguments)
    agent, client = make_agent(
        monkeypatch,
        make_message(tool_calls=[call]),
        make_message(content="处理完毕"),
        tool_registry=registry,
    )

    assert agent.run("执行工具", session_id="error_case") == "处理完毕"

    tool_result = json.loads(get_tool_message(agent, "error_case")["content"])
    assert expected_error in json.dumps(tool_result, ensure_ascii=False)
    assert client.chat.completions.create.call_count == 2
    second_request = client.chat.completions.create.call_args_list[1].kwargs
    assert any(message.get("role") == "tool" for message in second_request["messages"])


def test_agent_stops_after_maximum_tool_iterations(monkeypatch):
    def increment(value: int):
        return value + 1

    responses = [
        make_message(
            tool_calls=[
                make_tool_call(
                    "increment",
                    json.dumps({"value": index}),
                    call_id=f"call_{index}",
                )]
        )
        for index in range(10)
    ]
    agent, client = make_agent(
        monkeypatch,
        *responses,
        tool_registry={"increment": increment},
    )

    with pytest.raises(RuntimeError, match="超过最大决策轮数"):
        agent.run("一直调用工具", session_id="iteration_limit")

    assert client.chat.completions.create.call_count == 10
    messages = agent.session_manager.get_messages("iteration_limit")
    assert sum(message.get("role") == "tool" for message in messages) == 10
