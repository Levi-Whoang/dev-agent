
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


# 未知工具处理
def test_agent_unknown_tool(monkeypatch):
    import json
    from unittest.mock import MagicMock

    from agent.agent import Agent

    # 使用假的会话管理器，不连接真实数据库
    monkeypatch.setattr(
        "agent.agent.SessionManager",
        FakeSessionManager,
    )

    # 模拟大模型请求调用一个不存在的工具
    tool_call = MagicMock()
    tool_call.id = "call_unknown"
    tool_call.function.name = "unknown_tool"
    tool_call.function.arguments = "{}"

    first_message = MagicMock()
    first_message.tool_calls = [tool_call]
    first_message.model_dump.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_unknown",
                "type": "function",
                "function": {
                    "name": "unknown_tool",
                    "arguments": "{}",
                },
            }
        ],
    }

    # 模拟大模型收到错误结果后，生成最终回答
    final_message = MagicMock()
    final_message.tool_calls = None
    final_message.content = "抱歉，该工具不存在。"
    final_message.model_dump.return_value = {
        "role": "assistant",
        "content": "抱歉，该工具不存在。",
    }

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=first_message)]),
        MagicMock(choices=[MagicMock(message=final_message)]),
    ]

    agent = Agent(
        client=client,
        tools=[],
        tool_registry={},
    )

    result = agent.run("请调用一个不存在的工具", session_id="test_unknown")

    # 断言 Agent 最终返回正常结果
    assert result == "抱歉，该工具不存在。"

    # 断言大模型被调用了两次
    assert client.chat.completions.create.call_count == 2

    # 检查 Agent 是否将未知工具错误反馈给大模型
    messages = agent.session_manager.get_messages("test_unknown")
    tool_messages = [
        message for message in messages
        if message.get("role") == "tool"
    ]

    assert len(tool_messages) == 1

    error_result = json.loads(tool_messages[0]["content"])
    assert "error" in error_result
    assert "工具不存在" in error_result["error"]

# 非法 JSON 工具参数
def test_agent_invalid_json_arguments(monkeypatch):
    import json
    from unittest.mock import MagicMock

    from agent.agent import Agent

    monkeypatch.setattr(
        "agent.agent.SessionManager",
        FakeSessionManager,
    )

    # 模拟大模型生成不合法的 JSON 参数
    tool_call = MagicMock()
    tool_call.id = "call_invalid_json"
    tool_call.function.name = "calculate"
    tool_call.function.arguments = '{"a": 10'

    first_message = MagicMock()
    first_message.tool_calls = [tool_call]
    first_message.model_dump.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_invalid_json",
                "type": "function",
                "function": {
                    "name": "calculate",
                    "arguments": '{"a": 10',
                },
            }
        ],
    }

    # 模拟大模型收到错误反馈后给出最终回答
    final_message = MagicMock()
    final_message.tool_calls = None
    final_message.content = "工具参数格式不正确。"
    final_message.model_dump.return_value = {
        "role": "assistant",
        "content": "工具参数格式不正确。",
    }

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=first_message)]),
        MagicMock(choices=[MagicMock(message=final_message)]),
    ]

    # 如果参数解析失败，calculate 不应该被执行
    def calculate(a: int):
        raise AssertionError("非法 JSON 不应该执行工具")

    agent = Agent(
        client=client,
        tools=[],
        tool_registry={"calculate": calculate},
    )

    result = agent.run(
        "请计算 10",
        session_id="test_invalid_json",
    )

    assert result == "工具参数格式不正确。"
    assert client.chat.completions.create.call_count == 2

    messages = agent.session_manager.get_messages("test_invalid_json")
    tool_messages = [
        message for message in messages
        if message.get("role") == "tool"
    ]

    assert len(tool_messages) == 1

    error_result = json.loads(tool_messages[0]["content"])
    assert "error" in error_result
    assert "工具参数格式错误" in error_result["error"]

# 参数校验失败
def test_agent_invalid_tool_arguments(monkeypatch):
    import json
    from unittest.mock import MagicMock

    from agent.agent import Agent

    monkeypatch.setattr(
        "agent.agent.SessionManager",
        FakeSessionManager,
    )

    # JSON 格式正确，但 a 的类型不符合工具要求
    arguments = '{"a": "hello"}'

    tool_call = MagicMock()
    tool_call.id = "call_invalid_args"
    tool_call.function.name = "calculate"
    tool_call.function.arguments = arguments

    first_message = MagicMock()
    first_message.tool_calls = [tool_call]
    first_message.model_dump.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_invalid_args",
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
    final_message.content = "参数类型不正确。"
    final_message.model_dump.return_value = {
        "role": "assistant",
        "content": "参数类型不正确。",
    }

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=first_message)]),
        MagicMock(choices=[MagicMock(message=final_message)]),
    ]

    # 如果校验失败，工具不应该执行
    def calculate(a: int):
        raise AssertionError("参数校验失败时不应该执行工具")

    agent = Agent(
        client=client,
        tools=[],
        tool_registry={"calculate": calculate},
    )

    result = agent.run(
        "请计算一个数",
        session_id="test_invalid_args",
    )

    assert result == "参数类型不正确。"
    assert client.chat.completions.create.call_count == 2

    messages = agent.session_manager.get_messages("test_invalid_args")
    tool_messages = [
        message for message in messages
        if message.get("role") == "tool"
    ]

    assert len(tool_messages) == 1

    error_result = json.loads(tool_messages[0]["content"])
    assert error_result["error"] == "工具参数校验失败"
    assert "details" in error_result

# 工具执行时抛出异常
def test_agent_tool_execution_error(monkeypatch):
    import json
    from unittest.mock import MagicMock

    from agent.agent import Agent

    monkeypatch.setattr(
        "agent.agent.SessionManager",
        FakeSessionManager,
    )

    arguments = '{"a": 10}'

    tool_call = MagicMock()
    tool_call.id = "call_tool_error"
    tool_call.function.name = "calculate"
    tool_call.function.arguments = arguments

    first_message = MagicMock()
    first_message.tool_calls = [tool_call]
    first_message.model_dump.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_tool_error",
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
    final_message.content = "工具执行失败，暂时无法计算。"
    final_message.model_dump.return_value = {
        "role": "assistant",
        "content": "工具执行失败，暂时无法计算。",
    }

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=first_message)]),
        MagicMock(choices=[MagicMock(message=final_message)]),
    ]

    # 模拟工具内部出现运行时异常
    def calculate(a: int):
        raise RuntimeError("模拟数据库连接失败")

    agent = Agent(
        client=client,
        tools=[],
        tool_registry={"calculate": calculate},
    )

    result = agent.run(
        "请计算 10",
        session_id="test_tool_error",
    )

    assert result == "工具执行失败，暂时无法计算。"
    assert client.chat.completions.create.call_count == 2

    messages = agent.session_manager.get_messages("test_tool_error")
    tool_messages = [
        message for message in messages
        if message.get("role") == "tool"
    ]

    assert len(tool_messages) == 1

    error_result = json.loads(tool_messages[0]["content"])
    assert "error" in error_result
    assert "工具执行失败" in error_result["error"]
    assert "模拟数据库连接失败" in error_result["error"]

# Agent 超过最大决策轮数
def test_agent_stops_after_max_iterations(monkeypatch):
    from unittest.mock import MagicMock

    import pytest

    from agent.agent import Agent

    monkeypatch.setattr(
        "agent.agent.SessionManager",
        FakeSessionManager,
    )

    # 每次都返回一个工具调用，让 Agent 无法正常结束
    tool_call = MagicMock()
    tool_call.id = "call_loop"
    tool_call.function.name = "calculate"
    tool_call.function.arguments = '{"a": 1}'

    message = MagicMock()
    message.tool_calls = [tool_call]
    message.model_dump.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_loop",
                "type": "function",
                "function": {
                    "name": "calculate",
                    "arguments": '{"a": 1}',
                },
            }
        ],
    }

    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=message)]
    )

    def calculate(a: int):
        return a + 1

    agent = Agent(
        client=client,
        tools=[],
        tool_registry={"calculate": calculate},
    )

    with pytest.raises(
        RuntimeError,
        match="超过最大决策轮数",
    ):
        agent.run(
            "请不断调用工具",
            session_id="test_max_iterations",
        )

    # 最大决策轮数为 10，因此大模型最多请求 10 次
    assert client.chat.completions.create.call_count == 10

# 会话历史隔离
def test_agent_session_history_is_isolated(monkeypatch):
    from unittest.mock import MagicMock

    from agent.agent import Agent

    monkeypatch.setattr(
        "agent.agent.SessionManager",
        FakeSessionManager,
    )

    client = MagicMock()

    # 模拟每次请求都直接返回最终回答
    answers = [
        "这是用户 A 的回答",
        "这是用户 B 的回答",
    ]

    messages = []

    for answer in answers:
        message = MagicMock()
        message.tool_calls = None
        message.content = answer
        message.model_dump.return_value = {
            "role": "assistant",
            "content": answer,
        }
        messages.append(
            MagicMock(choices=[MagicMock(message=message)])
        )

    client.chat.completions.create.side_effect = messages

    agent = Agent(
        client=client,
        tools=[],
        tool_registry={},
    )

    # 两个用户分别使用不同的 session_id
    result_a = agent.run("我是用户 A", session_id="user_a")
    result_b = agent.run("我是用户 B", session_id="user_b")

    assert result_a == "这是用户 A 的回答"
    assert result_b == "这是用户 B 的回答"

    history_a = agent.session_manager.get_messages("user_a")
    history_b = agent.session_manager.get_messages("user_b")

    # A 的历史中应有 A 的问题，不应出现 B 的问题
    assert any(
        m.get("content") == "我是用户 A"
        for m in history_a
    )
    assert not any(
        m.get("content") == "我是用户 B"
        for m in history_a
    )

    # B 的历史中应有 B 的问题，不应出现 A 的问题
    assert any(
        m.get("content") == "我是用户 B"
        for m in history_b
    )
    assert not any(
        m.get("content") == "我是用户 A"
        for m in history_b
    )

