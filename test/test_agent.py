import json
from unittest.mock import MagicMock

from agent.agent import Agent


def test_agent_tool_call(tmp_path):
    # 模拟一个计算器工具
    def calculate(a: int, b: int):
        return a + b

    # 模拟 LLM 第一次请求调用工具
    tool_call = MagicMock()
    tool_call.id = "call_123"
    tool_call.function.name = "calculate"
    tool_call.function.arguments = json.dumps({
        "a": 10,
        "b": 20
    })

    first_message = MagicMock()
    first_message.tool_calls = [tool_call]
    first_message.model_dump.return_value = {
        "role": "assistant",
        "tool_calls": []
    }

    # 模拟 LLM 第二次返回最终回答
    final_message = MagicMock()
    final_message.tool_calls = None
    final_message.content = "计算结果是 30"
    final_message.model_dump.return_value = {
        "role": "assistant",
        "content": "计算结果是 30"
    }

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=first_message)]),
        MagicMock(choices=[MagicMock(message=final_message)])
    ]

    # 使用临时文件保存 Session，避免污染真实历史
    agent = Agent(
        client=client,
        tools=[],
        tool_registry={"calculate": calculate}
    )
    agent.session_manager.file_path = tmp_path / "sessions.json"

    result = agent.run("计算 10 + 20", session_id="test_user")

    assert result == "计算结果是 30"
    assert client.chat.completions.create.call_count == 2
