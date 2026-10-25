"""流式输出：验证增量事件顺序、工具调用过程中的事件、以及落库内容。

假客户端自己实现 `create()`：因为 `stream=True` 时真实 SDK 返回的是
**异步迭代器**，普通 AsyncMock 返回值对不上（await 一次就变成函数了）。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from agent.agent import Agent
from agent.events import DELTA, DONE, ERROR, META, TOOL_RESULT, TOOL_START, AgentEvent


class FakeStore:
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


def chunk(content: str | None = None, tool_calls=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=tool_calls)
            )
        ]
    )


def tool_delta(index: int, call_id=None, name=None, arguments=None):
    function = None
    if name is not None or arguments is not None:
        function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=call_id, function=function)


def agen(chunks):
    async def generator():
        for item in chunks:
            yield item

    return generator()


class FakeChatClient:
    """按顺序吐出预设的流（或抛异常）。"""

    def __init__(self, streams: list) -> None:
        self._streams = list(streams)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._streams.pop(0)
        if isinstance(item, Exception):
            raise item
        return agen(item)


async def collect(agent: Agent, question: str) -> list:
    return [event async for event in agent.stream(question)]


async def test_stream_emits_tool_events_then_answer():
    async def calculate(a: float, b: float, operation: str) -> dict:
        return {"result": a + b}

    client = FakeChatClient(
        [
            # 第一轮：一次工具调用被拆成三个分片
            [
                chunk(tool_calls=[tool_delta(0, "call_1", "calc", "")]),
                chunk(tool_calls=[tool_delta(0, None, "ulate", '{"a": 10, ')]),
                chunk(
                    tool_calls=[
                        tool_delta(0, None, None, '"b": 20, "operation": "add"}')
                    ]
                ),
            ],
            # 第二轮：正常吐字，也拆成三段
            [chunk("答案"), chunk("是 "), chunk("30")],
        ]
    )

    store = FakeStore()
    agent = Agent(
        client=client,
        store=store,
        session_id="stream-1",
        tools=[{"type": "function", "function": {"name": "calculate"}}],
        tool_registry={"calculate": calculate},
        model="fake-model",
    )

    events = await collect(agent, "计算 10 + 20")
    names = [event.event for event in events]

    assert names[0] == META
    assert TOOL_START in names
    assert TOOL_RESULT in names
    assert names[-1] == DONE

    # 工具名被拆成 "calc" + "ulate"，必须拼回 calculate
    tool_start = next(e for e in events if e.event == TOOL_START)
    assert tool_start.data["name"] == "calculate"

    text = "".join(e.data["text"] for e in events if e.event == DELTA)
    assert text == "答案是 30"
    assert events[-1].data["answer"] == "答案是 30"

    # 落库顺序：user / assistant(带 tool_calls) / tool / assistant
    roles = [m["role"] for m in store.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert json.loads(store.messages[2]["content"])["result"] == 30.0
    assert store.messages[1]["tool_calls"][0]["function"]["name"] == "calculate"


async def test_stream_without_tools_ends_with_done():
    client = FakeChatClient([[chunk("你好"), chunk("，我是助手")]])

    agent = Agent(
        client=client,
        store=FakeStore(),
        session_id="stream-2",
        model="fake-model",
    )

    events = await collect(agent, "你好")

    assert [e.event for e in events] == [META, DELTA, DELTA, DONE]
    assert events[-1].data["answer"] == "你好，我是助手"
    # 没工具时不该给模型传空 tools 列表
    assert client.calls[0]["tools"] is None


async def test_stream_reports_error_as_event():
    client = FakeChatClient([RuntimeError("模型超时")])

    agent = Agent(
        client=client,
        store=FakeStore(),
        session_id="stream-3",
        model="fake-model",
    )

    events = await collect(agent, "你好")

    # 流已经开始了，不能再改 HTTP 状态码，只能用 error 事件告知
    assert events[-1].event == ERROR
    assert "模型超时" in events[-1].data["message"]


async def test_sse_format_is_parseable():
    sse = AgentEvent(DELTA, {"text": "你好\n世界"}).to_sse()

    assert sse.startswith("event: delta\n")
    assert sse.endswith("\n\n")
    assert '"text": "你好\\n世界"' in sse
