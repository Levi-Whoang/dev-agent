
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import agent.agent as agent_module
import tools.knowledge as knowledge_module
from agent.agent import Agent
from tools.knowledge import knowledge_search_context


class FakeSessionManager:
    """用内存模拟会话管理，避免连接 MySQL。"""

    def __init__(self):
        self.sessions = {}

    def get_messages(self, session_id):
        return list(self.sessions.get(session_id, []))

    def save(self, session_id, messages):
        self.sessions[session_id] = list(messages)


def make_llm_message(content=None, tool_calls=None):
    """构造模拟 LLM 返回的消息。"""

    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls

    if tool_calls:
        message.model_dump.return_value = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls
            ],
        }
    else:
        message.model_dump.return_value = {
            "role": "assistant",
            "content": content,
        }

    return message


@pytest.mark.parametrize(
    ("tool_result", "final_answer", "expected"),
    [
        (
            "来源：company.txt\n证据：公司位于深圳市南山区。",
            "公司位于深圳市南山区。",
            "深圳市南山区",
        ),
        (
            "NO_RELEVANT_EVIDENCE",
            "抱歉，当前知识库无法确认公司老板的姓氏。",
            "无法确认",
        ),
    ],
)
def test_rag_agent_end_to_end(
    monkeypatch,
    tool_result,
    final_answer,
    expected,
):
    # 1. 隔离数据库
    monkeypatch.setattr(
        agent_module,
        "SessionManager",
        FakeSessionManager,
    )

    # 2. 模拟向量检索结果，但保留真实的知识检索上下文整理函数
    if tool_result == "NO_RELEVANT_EVIDENCE":
        monkeypatch.setattr(
            knowledge_module,
            "knowledge_search",
            lambda query: [],
        )
    else:
        monkeypatch.setattr(
            knowledge_module,
            "knowledge_search",
            lambda query: [
                {
                    "content": "公司位于深圳市南山区。",
                    "source": "company.txt",
                    "score": 0.9,
                }
            ],
        )

    # 3. 模拟 LLM 第一次请求知识检索工具
    tool_call = SimpleNamespace(
        id="call_123",
        function=SimpleNamespace(
            name="knowledge_search_context",
            arguments=json.dumps(
                {"query": "公司在哪里？"},
                ensure_ascii=False,
            ),
        ),
    )

    first_message = make_llm_message(
        tool_calls=[tool_call],
    )
    final_message = make_llm_message(
        content=final_answer,
    )

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        SimpleNamespace(
            choices=[SimpleNamespace(message=first_message)]
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(message=final_message)]
        ),
    ]

    agent = Agent(
        client=client,
        tools=[],
        tool_registry={
            "knowledge_search_context": knowledge_search_context,
        },
    )

    # 4. 执行完整 Agent 工具调用循环
    query = (
        "公司老板姓什么？"
        if tool_result == "NO_RELEVANT_EVIDENCE"
        else "公司在哪里？"
    )
    result = agent.run(query, session_id="rag_test")

    # 5. 验证最终回答和调用链
    assert expected in result
    assert client.chat.completions.create.call_count == 2

    second_call_messages = (
        client.chat.completions.create.call_args_list[1]
        .kwargs["messages"]
    )

    tool_messages = [
        message
        for message in second_call_messages
        if message.get("role") == "tool"
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_123"
    assert expected in tool_messages[0]["content"] or (
        tool_result == "NO_RELEVANT_EVIDENCE"
        and "NO_RELEVANT_EVIDENCE" in tool_messages[0]["content"]
    )
