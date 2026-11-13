from test.async_helpers import AsyncStream, collect
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock

import pytest


import agent.agent as agent_module
from agent.agent import Agent
from agent.history import SUMMARY_PREFIX, count_tokens


def calculate(a: int, b: int):
    return a + b


def explode(value: int):
    raise RuntimeError("boom")


def echo(text: str):
    return {"text": text * 2000}


class FakeSessionManager:
    """内存会话存储，确保测试不连接 MySQL。"""

    def __init__(self):
        self.sessions = {}
        self.summaries = {}
        self.save_calls = 0

    async def get_messages(self, session_id):
        return list(self.sessions.get(session_id, []))

    async def save(self, session_id, messages):
        self.save_calls += 1
        self.sessions[session_id] = list(messages)

    async def append_message(self, session_id, message):
        self.sessions.setdefault(session_id, []).append(dict(message))

    async def get_summary(self, session_id):
        return self.summaries.get(session_id)

    async def save_summary(self, session_id, content, covered_messages):
        self.summaries[session_id] = {
            "content": content,
            "covered_messages": covered_messages,
        }

    async def clear_summary(self, session_id):
        self.summaries.pop(session_id, None)


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


def stream_chunk(content=None, tool_calls=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def stream_tool_call_delta(
    index=0,
    call_id=None,
    name=None,
    arguments=None,
):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=function,
    )


def make_agent(monkeypatch, *messages, tool_registry=None):
    monkeypatch.setattr(agent_module, "SessionManager", FakeSessionManager)
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    client.chat.completions.create.side_effect = [
        llm_response(message) for message in messages
    ]
    agent = Agent(
        client=client,
        tools=[{"type": "function", "function": {"name": "stub"}}],
        tool_registry=tool_registry if tool_registry is not None else {},
    )
    return agent, client


async def get_tool_message(agent, session_id):
    messages = await agent.session_manager.get_messages(session_id)
    return next(message for message in messages if message.get("role") == "tool")


def record_requests(client):
    """在请求发出时保存消息快照，避免后续 append 修改断言对象。"""
    snapshots = []
    original = client.chat.completions.create

    async def create(**kwargs):
        snapshots.append([dict(message) for message in kwargs["messages"]])
        return await original(**kwargs)

    client.chat.completions.create = AsyncMock(side_effect=create)
    return snapshots


async def test_agent_summarizes_old_turns_and_persists_full_history(monkeypatch):
    monkeypatch.setenv("HISTORY_RECENT_TURNS", "1")
    # 第一次调用用于生成摘要，第二次才是本轮决策。
    agent, client = make_agent(
        monkeypatch,
        make_message(content="旧对话摘要"),
        make_message(content="最新回答"),
    )
    snapshots = record_requests(client)
    history = [{"role": "system", "content": "旧系统提示"}]
    for index in range(3):
        history.append({"role": "user", "content": f"旧问题{index}"})
        history.append({"role": "assistant", "content": f"旧回答{index}"})
    await agent.session_manager.save("summary-case", history)

    result = await agent.run("新问题", session_id="summary-case")

    assert result == "最新回答"
    request_messages = snapshots[-1]
    assert request_messages[0]["role"] == "system"
    assert sum(
        message["content"].startswith(SUMMARY_PREFIX) for message in request_messages
    ) == 1
    # 更早的三轮进入摘要，最近一轮保留原文。
    assert request_messages[-1] == {"role": "user", "content": "新问题"}
    assert not any(message["content"] == "旧问题2" for message in request_messages)

    summary = await agent.session_manager.get_summary("summary-case")
    assert summary["covered_messages"] == 6
    assert summary["content"] == "旧对话摘要"

    # 持久化的仍是完整历史，摘要只影响发给模型的上下文。
    stored = await agent.session_manager.get_messages("summary-case")
    assert [message["content"] for message in stored if message["role"] == "user"] == [
        "旧问题0",
        "旧问题1",
        "旧问题2",
        "新问题",
    ]
    assert stored[-1] == {"role": "assistant", "content": "最新回答"}
    assert count_tokens(request_messages) <= 12000


async def test_agent_appends_turns_without_rewriting_whole_history(monkeypatch):
    agent, _ = make_agent(
        monkeypatch,
        make_message(content="第一轮回答"),
        make_message(content="第二轮回答"),
    )

    assert (await agent.run("第一问", session_id="append-case")) == "第一轮回答"
    assert (await agent.run("第二问", session_id="append-case")) == "第二轮回答"

    stored = await agent.session_manager.get_messages("append-case")
    assert [message["role"] for message in stored] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [message["content"] for message in stored] == [
        "第一问",
        "第一轮回答",
        "第二问",
        "第二轮回答",
    ]
    # 逐条追加，不再整段覆盖历史；system 提示每轮重建，不入库。
    assert agent.session_manager.save_calls == 0


async def test_agent_enforces_context_budget_on_every_model_call(monkeypatch):
    monkeypatch.setenv("HISTORY_RECENT_TURNS", "10")
    monkeypatch.setenv("HISTORY_TOKEN_BUDGET", "1000")
    tool_call = make_tool_call("echo", json.dumps({"text": "x"}))
    agent, client = make_agent(
        monkeypatch,
        make_message(tool_calls=[tool_call]),
        make_message(content="完成"),
        tool_registry={"echo": echo},
    )
    snapshots = record_requests(client)

    assert (await agent.run("请处理", session_id="budget-case")) == "完成"

    assert len(snapshots) == 2
    for request_messages in snapshots:
        assert count_tokens(request_messages) <= 1000

    stored = await agent.session_manager.get_messages("budget-case")
    # 工具结果在完整历史中不被截断。
    tool_message = next(message for message in stored if message["role"] == "tool")
    assert len(json.loads(tool_message["content"])["text"]) == 2000


async def test_agent_executes_tool_and_persists_final_answer(monkeypatch):
    call = make_tool_call("calculate", json.dumps({"a": 10, "b": 20}))
    agent, client = make_agent(
        monkeypatch,
        make_message(tool_calls=[call]),
        make_message(content="计算结果是 30"),
        tool_registry={"calculate": calculate},
    )

    result = await agent.run("计算 10 + 20", session_id="test_user")

    assert result == "计算结果是 30"
    assert client.chat.completions.create.call_count == 2
    first_request = client.chat.completions.create.call_args_list[0].kwargs
    assert first_request["tool_choice"] == "auto"
    assert first_request["messages"][0]["role"] == "system"

    messages = await agent.session_manager.get_messages("test_user")
    assert any(message.get("role") == "tool" for message in messages)
    assert messages[-1]["content"] == "计算结果是 30"
    assert json.loads(((await get_tool_message(agent, "test_user")))["content"]) == 30


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
            "工具执行失败",
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
async def test_agent_returns_tool_errors_to_the_model(
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

    assert (await agent.run("执行工具", session_id="error_case")) == "处理完毕"

    tool_result = json.loads(((await get_tool_message(agent, "error_case")))["content"])
    assert expected_error in json.dumps(tool_result, ensure_ascii=False)
    assert client.chat.completions.create.call_count == 2
    second_request = client.chat.completions.create.call_args_list[1].kwargs
    assert any(message.get("role") == "tool" for message in second_request["messages"])


async def test_agent_stops_after_maximum_tool_iterations(monkeypatch):
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
        await agent.run("一直调用工具", session_id="iteration_limit")

    assert client.chat.completions.create.call_count == 10
    messages = await agent.session_manager.get_messages("iteration_limit")
    assert sum(message.get("role") == "tool" for message in messages) == 10


async def test_agent_streams_final_answer_and_persists_complete_message(monkeypatch):
    monkeypatch.setattr(agent_module, "SessionManager", FakeSessionManager)
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    client.chat.completions.create.side_effect = [llm_response(make_message(content="决策草稿")), AsyncStream([
        stream_chunk(content="你好"),
        stream_chunk(content="，世界！"),
    ])]
    agent = Agent(client, tools=[], tool_registry={})

    chunks = await collect(agent.run_stream("打招呼", session_id="stream-1"))

    assert chunks == ["你好", "，世界！"]
    assert client.chat.completions.create.call_args.kwargs["stream"] is True
    assert ((await agent.session_manager.get_messages("stream-1")))[-1] == {
        "role": "assistant",
        "content": "你好，世界！",
    }


async def test_agent_streams_tool_call_round_then_streams_final_answer(monkeypatch):
    monkeypatch.setattr(agent_module, "SessionManager", FakeSessionManager)
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    client.chat.completions.create.side_effect = [
        llm_response(make_message(tool_calls=[make_tool_call("calculate", '{"a":10,"b":20}', "call_stream")])),
        llm_response(make_message(content="决策结束")),
        AsyncStream([stream_chunk(content="结果是 30")]),
    ]
    agent = Agent(client, tools=[], tool_registry={"calculate": calculate})

    chunks = await collect(agent.run_stream("计算 10 + 20", session_id="stream-tool"))

    assert chunks == ["结果是 30"]
    assert client.chat.completions.create.call_count == 3
    messages = await agent.session_manager.get_messages("stream-tool")
    assert messages[-1] == {"role": "assistant", "content": "结果是 30"}
    tool_message = next(message for message in messages if message["role"] == "tool")
    assert tool_message["tool_call_id"] == "call_stream"
    assert json.loads(tool_message["content"]) == 30


# 在消费完模型流之前即可拿到正文，停止后关闭流并保存已发送文本。
async def test_real_incremental_and_stop(monkeypatch):
    from threading import Event
    monkeypatch.setattr(agent_module, "SessionManager", FakeSessionManager)
    consumed = []
    def chunks():
        for text in ["<thi", "nk>隐藏</think>你好", "世界"]:
            consumed.append(text)
            yield stream_chunk(content=text)
    response = AsyncStream(chunks())
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    client.chat.completions.create.side_effect = [llm_response(make_message(content="草稿")), response]
    agent = Agent(client, [], {})
    event = Event()
    stream = agent.run_stream("你好", "stop", event)
    assert (await anext(stream)) == "你好"
    assert len(consumed) == 2
    event.set()
    assert (await collect(stream)) == []
    assert response.closed
    assert ((await agent.session_manager.get_messages("stop")))[-1]["content"] == "你好"
    assert "tools" not in client.chat.completions.create.call_args.kwargs


# 工具超时返回可识别错误且不无限等待。
async def test_tool_timeout():
    from agent.control import execute_tool
    from threading import Event
    release = Event()
    def slow():
        release.wait(1)
    try:
        result = await execute_tool(slow, {}, 0.02)
        assert result["code"] == "tool_timeout"
    finally:
        release.set()


# RAG 工具完成后直接流式回答，不能再请求一次无用的完整答案。
async def test_rag_skips_duplicate_answer(monkeypatch):
    monkeypatch.setattr(agent_module, "SessionManager", FakeSessionManager)
    def knowledge_search_context(query: str):
        return {"evidence":"公开证据"}
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    client.chat.completions.create.side_effect = [
        llm_response(make_message(tool_calls=[make_tool_call("knowledge_search_context", '{"query":"问题"}')])),
        AsyncStream([stream_chunk(content="第一段"),stream_chunk(content="第二段")]),
    ]
    agent = Agent(client, [], {"knowledge_search_context":knowledge_search_context})
    assert (await collect(agent.run_stream("问题", "fast-rag"))) == ["第一段", "第二段"]
    assert client.chat.completions.create.call_count == 2


# 提前关闭异步生成器时关闭模型连接，并保存已经发送的正文。
async def test_close_stream_closes_and_saves(monkeypatch):
    monkeypatch.setattr(agent_module, "SessionManager", FakeSessionManager)
    response = AsyncStream([stream_chunk(content="已发送"), stream_chunk(content="尚未发送")])
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    client.chat.completions.create.side_effect = [llm_response(make_message(content="决策")), response]
    agent = Agent(client, [], {})
    stream = agent.run_stream("测试", "disconnect")
    assert (await anext(stream)) == "已发送"
    await stream.aclose()
    assert response.closed
    assert (await agent.session_manager.get_messages("disconnect"))[-1]["content"] == "已发送"


# 停止信号取消尚未返回的模型请求，避免等待网络超时。
async def test_stop_cancels_pending_model(monkeypatch):
    import asyncio
    from threading import Event
    from agent.control import GenerationStopped
    monkeypatch.setattr(agent_module, "SessionManager", FakeSessionManager)
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    # 用可取消等待模拟迟迟未返回的网络请求。
    async def request(**kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=request)
    agent = Agent(client, [], {})
    event = Event()
    stream = agent.run_stream("问题", "pending-stop", event)
    task = asyncio.create_task(anext(stream))
    await asyncio.wait_for(entered.wait(), timeout=1)
    event.set()
    with pytest.raises(GenerationStopped):
        await asyncio.wait_for(task, timeout=1)
    assert cancelled.is_set()
    await stream.aclose()


# 网络等待期间事件循环仍可执行其他任务。
async def test_network_wait_does_not_block_loop():
    import asyncio
    from threading import Event
    from agent.control import await_or_stop
    ready = asyncio.Event()

    # 等待另一个任务发出信号，模拟网络响应就绪。
    async def network():
        await ready.wait()
        return "ok"

    task = asyncio.create_task(await_or_stop(network(), Event()))
    await asyncio.sleep(0)
    assert not task.done()
    ready.set()
    assert await asyncio.wait_for(task, timeout=1) == "ok"


# 取消分片读取后关闭模型连接，并保存此前已发送的正文。
async def test_cancel_stream_closes_and_saves(monkeypatch):
    import asyncio
    monkeypatch.setattr(agent_module, "SessionManager", FakeSessionManager)
    entered = asyncio.Event()

    class WaitingStream(AsyncStream):
        # 首片段之后持续等待，用于模拟浏览器断开连接。
        async def __anext__(self):
            if not hasattr(self, "sent"):
                self.sent = True
                return stream_chunk(content="已发送")
            entered.set()
            await asyncio.Event().wait()

    response = WaitingStream([])
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=[llm_response(make_message(content="决策")), response])
    agent = Agent(client, [], {})
    stream = agent.run_stream("测试", "cancelled-disconnect")
    assert await anext(stream) == "已发送"
    reading = asyncio.create_task(anext(stream))
    await asyncio.wait_for(entered.wait(), timeout=1)
    reading.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reading
    assert response.closed
    assert (await agent.session_manager.get_messages("cancelled-disconnect"))[-1]["content"] == "已发送"




pytestmark = pytest.mark.anyio

# 测试统一使用项目的asyncio运行时。
@pytest.fixture
def anyio_backend():
    return "asyncio"
