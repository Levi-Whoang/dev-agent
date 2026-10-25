"""RAG + Agent 的端到端链路（工具返回什么，模型就看到什么）。

不连真实模型、不连真实知识库：只验证“检索工具的返回值如何进入
第二轮的模型请求”，以及无证据时的拒答路径。
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.agent import Agent
from test_agent_stream import FakeChatClient, FakeStore, chunk, tool_delta

EVIDENCE = "来源：company.txt\n证据：公司位于深圳市南山区。\n语义相似度：0.91"


async def run_with_evidence(query: str, evidence: str, final_answer: str):
    async def knowledge_search_context(query: str) -> str:
        return evidence

    client = FakeChatClient(
        [
            [chunk(tool_calls=[tool_delta(0, "call_1", "knowledge_search_context", json.dumps({"query": query}, ensure_ascii=False))])],
            [chunk(final_answer)],
        ]
    )

    store = FakeStore()
    agent = Agent(
        client=client,
        store=store,
        session_id="rag-1",
        tools=[{"type": "function", "function": {"name": "knowledge_search_context"}}],
        tool_registry={"knowledge_search_context": knowledge_search_context},
        model="fake-model",
    )

    events = [event async for event in agent.stream(query)]
    return agent, client, store, events


async def test_evidence_is_passed_to_model():
    _, client, store, events = await run_with_evidence(
        "公司在哪里？", EVIDENCE, "公司位于深圳市南山区（来源：company.txt）。"
    )

    assert events[-1].event == "done"
    assert "深圳市南山区" in events[-1].data["answer"]

    # 第二轮请求里必须带着 tool 消息，否则模型等于没查到东西
    second_call = client.calls[1]
    tool_messages = [m for m in second_call["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert "company.txt" in tool_messages[0]["content"]

    assert [m["role"] for m in store.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


async def test_refusal_path_when_no_evidence():
    _, _, store, events = await run_with_evidence(
        "公司老板姓什么？",
        "NO_RELEVANT_EVIDENCE",
        "抱歉，当前知识库无法确认公司老板的姓氏。",
    )

    assert "无法确认" in events[-1].data["answer"]

    tool_message = next(m for m in store.messages if m["role"] == "tool")
    assert tool_message["content"] == "NO_RELEVANT_EVIDENCE"
