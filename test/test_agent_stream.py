"""流式输出：验证增量事件顺序、工具调用过程中的事件、以及落库内容。

假客户端自己实现 `create()`：因为 `stream=True` 时真实 SDK 返回的是
**异步迭代器**，普通 AsyncMock 返回值对不上（await 一次就变成函数了）。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from agent.agent import Agent
from agent.events import (
    DELTA,
    DONE,
    ERROR,
    META,
    THOUGHT,
    TOOL_RESULT,
    TOOL_START,
    AgentEvent,
)


class FakeStore:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def message_rows(self, session_id: str, limit: int = 500):
        rows = []
        for index, message in enumerate(self.messages[-limit:], start=1):
            extra = {
                k: v for k, v in message.items() if k not in ("role", "content")
            }
            rows.append(
                SimpleNamespace(
                    id=index,
                    role=message["role"],
                    content=message.get("content") or "",
                    payload=extra or None,
                )
            )
        return rows

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
    names = [event.event for event in events]

    # 流已经开始了，不能再改 HTTP 状态码，只能用 error 事件告知
    error = next(e for e in events if e.event == ERROR)
    assert "模型超时" in error.data["message"]

    # 但事件流必须有一个统一的终态：异常也要走 done 收尾，
    # 否则客户端只能靠"连接断了"来猜，拿不到终止原因和推理轨迹。
    assert names[-1] == DONE
    assert events[-1].data["termination"] == "error"
    assert events[-1].data["answer"] == ""


async def test_stream_emits_thought_before_action():
    """ReAct 的 Thought：模型先说"我要查什么"，再发起调用。

    正文增量已经通过 delta 吐过了，thought 事件是为了让不渲染增量的
    客户端（脚本、日志回放）也能拿到这一步的推理。
    """
    client = FakeChatClient(
        [
            [
                chunk("我先算一下这个加法。"),
                chunk(tool_calls=[tool_delta(0, "call_1", "calculate", '{"a":1,"b":2}')]),
            ],
            [chunk("等于 3")],
        ]
    )

    async def calculate(a: float, b: float) -> dict:
        return {"result": a + b}

    agent = Agent(
        client=client,
        store=FakeStore(),
        session_id="stream-thought",
        tools=[{"type": "function", "function": {"name": "calculate"}}],
        tool_registry={"calculate": calculate},
        model="fake-model",
    )

    events = await collect(agent, "1+2 是多少")
    names = [event.event for event in events]

    thought = next(e for e in events if e.event == THOUGHT)
    assert thought.data["text"] == "我先算一下这个加法。"
    assert thought.data["step"] == 1

    # 顺序：思考 → 动作 → 观测；思考必须在对外的动作之前
    assert names.index(THOUGHT) < names.index(TOOL_START) < names.index(TOOL_RESULT)
    # 收尾那一步没有动作，不该再产生 thought
    assert sum(1 for n in names if n == THOUGHT) == 1

    done = events[-1]
    assert done.data["termination"] == "answer"
    assert done.data["trace"][0]["thought"] == "我先算一下这个加法。"


async def test_stream_skips_thought_when_model_stays_silent():
    """模型不解释直接调工具时，不能伪造一句思考。"""
    client = FakeChatClient(
        [
            [chunk(tool_calls=[tool_delta(0, "call_1", "noop", "{}")])],
            [chunk("好了")],
        ]
    )

    async def noop() -> dict:
        return {"ok": True}

    agent = Agent(
        client=client,
        store=FakeStore(),
        session_id="stream-silent",
        tools=[{"type": "function", "function": {"name": "noop"}}],
        tool_registry={"noop": noop},
        model="fake-model",
    )

    events = await collect(agent, "做点什么")
    assert THOUGHT not in [e.event for e in events]


async def test_stream_aborts_on_repeated_identical_tool_call():
    """同一个工具、同一份参数连续调用三次 → 中断，别把步数预算烧光。"""
    same = [chunk(tool_calls=[tool_delta(0, "call_x", "search", '{"q":"制度"}')])]
    client = FakeChatClient([same, same, same, [chunk("不该走到这里")]])

    async def search(q: str) -> dict:
        return {"error": "索引未就绪"}

    agent = Agent(
        client=client,
        store=FakeStore(),
        session_id="stream-loop",
        tools=[{"type": "function", "function": {"name": "search"}}],
        tool_registry={"search": search},
        model="fake-model",
    )

    events = await collect(agent, "查制度")
    names = [event.event for event in events]

    assert names[-1] == DONE
    assert events[-1].data["termination"] == "loop"
    # 第三次重复直接掐断，所以只真正执行了两次
    assert names.count(TOOL_START) == 2
    # 第二次执行前会贴一句提醒
    hint = next(e for e in events if e.event == TOOL_START and e.data["step"] == 2)
    assert hint.data["step"] == 2


async def test_loop_warning_is_injected_into_observation():
    """提醒贴在观测结果前面，而不是插一条额外的 system 消息。"""
    same = [chunk(tool_calls=[tool_delta(0, "call_x", "search", '{"q":"制度"}')])]
    client = FakeChatClient([same, same, same, [chunk("放弃")]])

    store = FakeStore()

    async def search(q: str) -> dict:
        return {"error": "索引未就绪"}

    agent = Agent(
        client=client,
        store=store,
        session_id="stream-hint",
        tools=[{"type": "function", "function": {"name": "search"}}],
        tool_registry={"search": search},
        model="fake-model",
    )

    await collect(agent, "查制度")

    tool_rows = [m for m in store.messages if m["role"] == "tool"]
    assert not tool_rows[0]["content"].startswith("[系统提示]")
    assert tool_rows[1]["content"].startswith("[系统提示]")


async def test_stream_stops_at_max_iterations():
    """每轮都换参数、但永远不给答案 → 到上限收尾，termination=max_steps。"""

    def stream_for(step: int):
        return [
            chunk(
                tool_calls=[
                    tool_delta(0, f"call_{step}", "search", f'{{"q":"第{step}次"}}')
                ]
            )
        ]

    client = FakeChatClient([stream_for(i) for i in range(1, 5)])

    async def search(q: str) -> dict:
        return {"hits": []}

    agent = Agent(
        client=client,
        store=FakeStore(),
        session_id="stream-budget",
        tools=[{"type": "function", "function": {"name": "search"}}],
        tool_registry={"search": search},
        model="fake-model",
        max_iterations=3,
    )

    events = await collect(agent, "找出答案")
    names = [event.event for event in events]

    assert names[-1] == DONE
    assert events[-1].data["termination"] == "max_steps"
    assert names.count(TOOL_START) == 3
    assert client.calls[0]["tools"] is not None


async def test_sse_format_is_parseable():
    sse = AgentEvent(DELTA, {"text": "你好\n世界"}).to_sse()

    assert sse.startswith("event: delta\n")
    assert sse.endswith("\n\n")
    assert '"text": "你好\\n世界"' in sse
