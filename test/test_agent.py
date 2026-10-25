"""Agent 决策循环（非流式）：用假客户端验证工具调用链路，不打真实模型。"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.agent import Agent


class FakeStore:
    """内存版会话存储，替掉数据库。"""

    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def history(self, session_id: str, limit: int = 40) -> list[dict]:
        return list(self.messages[-limit:])

    async def append(
        self, session_id: str, role: str, content: str, payload: dict | None = None
    ) -> None:
        message = {"role": role, "content": content}
        if payload:
            message.update(payload)
        self.messages.append(message)


def make_tool_call(name: str, arguments: str, call_id: str = "call_1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def make_response(content: str | None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


async def test_agent_calls_tool_then_answers():
    calls: list[tuple] = []

    async def calculate(a: float, b: float, operation: str) -> dict:
        calls.append((a, b, operation))
        return {"result": a + b}

    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            make_response(
                None,
                [make_tool_call("calculate", json.dumps({"a": 10, "b": 20, "operation": "add"}))],
            ),
            make_response("计算结果是 30"),
        ]
    )

    store = FakeStore()
    agent = Agent(
        client=client,
        store=store,
        session_id="s1",
        tools=[{"type": "function", "function": {"name": "calculate"}}],
        tool_registry={"calculate": calculate},
        model="fake-model",
    )

    answer = await agent.run("计算 10 + 20")

    assert answer == "计算结果是 30"
    assert calls == [(10.0, 20.0, "add")]
    assert client.chat.completions.create.await_count == 2

    roles = [m["role"] for m in store.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    # 工具结果必须原样回灌给模型，否则第二轮模型看不到数据
    assert json.loads(store.messages[2]["content"])["result"] == 30.0
    assert store.messages[2]["tool_call_id"] == "call_1"


async def test_agent_reports_unknown_tool_to_model():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            make_response(None, [make_tool_call("delete_everything", "{}")]),
            make_response("这个工具不存在"),
        ]
    )

    store = FakeStore()
    agent = Agent(
        client=client,
        store=store,
        session_id="s2",
        tool_registry={},
        model="fake-model",
    )

    await agent.run("删库")

    tool_message = next(m for m in store.messages if m["role"] == "tool")
    assert "工具不存在" in tool_message["content"]


async def test_agent_rejects_bad_arguments():
    async def calculate(a: float, b: float, operation: str) -> dict:
        return {"result": a + b}

    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            make_response(
                None,
                [make_tool_call("calculate", json.dumps({"a": "not-a-number"}))],
            ),
            make_response("参数有问题"),
        ]
    )

    store = FakeStore()
    agent = Agent(
        client=client,
        store=store,
        session_id="s3",
        tool_registry={"calculate": calculate},
        model="fake-model",
    )

    await agent.run("算一下")

    tool_message = next(m for m in store.messages if m["role"] == "tool")
    # 参数不合法时不能执行工具，要把校验错误交给模型
    assert "校验失败" in tool_message["content"]


async def test_agent_stops_after_max_iterations():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=make_response(None, [make_tool_call("noop", "{}")])
    )

    async def noop() -> dict:
        return {"ok": True}

    agent = Agent(
        client=client,
        store=FakeStore(),
        session_id="s4",
        tool_registry={"noop": noop},
        model="fake-model",
        max_iterations=3,
    )

    with pytest.raises(RuntimeError):
        await agent.run("死循环")
