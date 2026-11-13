"""整体超时预算、剩余时间传递与可重试错误退避。"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import agent.agent as agent_module
from agent.agent import Agent
from agent.timeouts import (
    AgentTimeoutError,
    Deadline,
    TimeoutConfig,
    is_retryable,
    run_with_retry,
)
from api.chat_helpers import chat_event_stream
from test.async_helpers import AsyncStream, collect
from test.test_agent import (
    FakeSessionManager,
    llm_response,
    make_agent,
    make_message,
    make_tool_call,
    stream_chunk,
)


class RetryableError(Exception):
    status_code = 503


class FatalError(Exception):
    status_code = 400


def test_deadline_reports_remaining_time_and_expires(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("agent.timeouts.time.monotonic", lambda: now[0])
    deadline = Deadline.after(10)
    assert deadline.remaining() == 10
    now[0] = 109.9
    assert deadline.ensure() > 0
    now[0] = 110.0
    with pytest.raises(AgentTimeoutError):
        deadline.ensure()


def test_timeout_config_reads_environment(monkeypatch):
    monkeypatch.setenv("CHAT_TOTAL_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "8")
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_SECONDS", "0.2")
    monkeypatch.setenv("LLM_MAX_RETRY_BACKOFF_SECONDS", "2")
    config = TimeoutConfig.from_env()
    assert config.total_seconds == 60
    assert config.request_seconds == 8
    assert config.max_retries == 3
    assert config.retry_backoff_seconds == 0.2
    assert config.max_retry_backoff_seconds == 2


def test_timeout_config_falls_back_on_invalid_values(monkeypatch):
    monkeypatch.setenv("CHAT_TOTAL_TIMEOUT_SECONDS", "not-a-number")
    monkeypatch.setenv("LLM_MAX_RETRIES", "-1")
    config = TimeoutConfig.from_env()
    assert config.total_seconds == 120
    assert config.max_retries == 0


def test_retryable_error_classification():
    assert is_retryable(RetryableError())
    assert is_retryable(TimeoutError())
    assert not is_retryable(FatalError())
    assert not is_retryable(ValueError("bad request"))


async def test_run_with_retry_retries_then_succeeds():
    attempts = []
    config = TimeoutConfig(max_retries=2, retry_backoff_seconds=0)

    async def operation():
        attempts.append(1)
        if len(attempts) < 3:
            raise RetryableError("busy")
        return "ok"

    result = await run_with_retry(operation, Deadline.after(5), config)
    assert result == "ok"
    assert len(attempts) == 3


async def test_run_with_retry_stops_after_limit():
    async def operation():
        raise RetryableError("busy")

    config = TimeoutConfig(max_retries=1, retry_backoff_seconds=0)
    with pytest.raises(RetryableError):
        await run_with_retry(operation, Deadline.after(5), config)


async def test_agent_passes_remaining_time_to_model_and_tool(monkeypatch):
    monkeypatch.setenv("CHAT_TOTAL_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "5")
    tool_call = make_tool_call("calculate", '{"a": 1, "b": 2}')
    agent, client = make_agent(
        monkeypatch,
        make_message(tool_calls=[tool_call]),
        make_message(content="完成"),
        tool_registry={"calculate": lambda a, b: a + b},
    )
    calls = []
    original = client.chat.completions.create

    async def create(**kwargs):
        calls.append(kwargs)
        return await original(**kwargs)

    client.chat.completions.create = AsyncMock(side_effect=create)
    tool_timeouts = []

    async def execute_tool(registry, call, timeout=None, stop_event=None):
        tool_timeouts.append(timeout)
        return {"value": 3}

    monkeypatch.setattr(agent_module, "execute_tool_call", execute_tool)

    assert (await agent.run("计算", session_id="timeout-case")) == "完成"

    assert len(calls) == 2
    assert all(0 < call["timeout"] <= 5 for call in calls)
    assert len(tool_timeouts) == 1
    assert 0 < tool_timeouts[0] <= 15


async def test_agent_retries_retryable_model_error(monkeypatch):
    class RetryAgent(Agent):
        pass

    monkeypatch.setattr(agent_module, "SessionManager", FakeSessionManager)
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[RetryableError("busy"), llm_response(make_message(content="恢复"))]
    )
    agent = RetryAgent(client, [], {})
    agent.timeouts = TimeoutConfig(
        total_seconds=5,
        request_seconds=2,
        max_retries=2,
        retry_backoff_seconds=0,
    )

    assert (await agent.run("重试", session_id="retry-case")) == "恢复"
    assert client.chat.completions.create.call_count == 2


async def test_agent_does_not_retry_non_retryable_error(monkeypatch):
    monkeypatch.setattr(agent_module, "SessionManager", FakeSessionManager)
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[FatalError("bad"), llm_response(make_message(content="不应到达"))]
    )
    agent = Agent(client, [], {})
    agent.timeouts = TimeoutConfig(
        total_seconds=5,
        request_seconds=2,
        max_retries=2,
        retry_backoff_seconds=0,
    )

    with pytest.raises(FatalError):
        await agent.run("失败", session_id="fatal-case")
    assert client.chat.completions.create.call_count == 1


async def test_agent_raises_timeout_when_total_budget_is_exhausted(monkeypatch):
    monkeypatch.setattr(agent_module, "SessionManager", FakeSessionManager)

    async def slow(**kwargs):
        await asyncio.sleep(0.1)
        return llm_response(make_message(content="太慢"))

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=slow)
    agent = Agent(client, [], {})
    agent.timeouts = TimeoutConfig(
        total_seconds=0.01,
        request_seconds=20,
        max_retries=0,
    )

    with pytest.raises(AgentTimeoutError) as error:
        await agent.run("会超时", session_id="timeout-budget")
    assert error.value.code == "agent_timeout"
    assert "超时" in str(error.value)


async def test_stream_passes_remaining_time_to_final_model_call(monkeypatch):
    monkeypatch.setattr(agent_module, "SessionManager", FakeSessionManager)
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[llm_response(make_message(content="决策"))]
    )
    calls = []
    original = client.chat.completions.create

    async def create(**kwargs):
        calls.append(kwargs)
        if kwargs.get("stream"):
            return AsyncStream([stream_chunk(content="流式回答")])
        return await original(**kwargs)

    client.chat.completions.create = AsyncMock(side_effect=create)
    agent = Agent(client, [], {})
    agent.timeouts = TimeoutConfig(
        total_seconds=30, request_seconds=5, max_retries=0
    )

    chunks = await collect(agent.run_stream("流式", "stream-timeout"))

    assert chunks == ["流式回答"]
    assert len(calls) == 2
    assert all(0 < call["timeout"] <= 5 for call in calls)


async def test_chat_event_stream_returns_timeout_code():
    class TimeoutAgent:
        async def run_stream(self, message, session_id):
            raise AgentTimeoutError()
            yield ""  # pragma: no cover - 保持异步生成器形态

    events = await collect(
        chat_event_stream(TimeoutAgent(), MagicMock(), "你好", "s1", "internal-1")
    )
    payload = json.loads(events[0][6:])
    assert payload["type"] == "error"
    assert payload["code"] == "agent_timeout"
    assert "超时" in payload["message"]


async def test_chat_event_stream_returns_generation_failed_code():
    class FailingAgent:
        async def run_stream(self, message, session_id):
            raise RuntimeError("boom")
            yield ""  # pragma: no cover - 保持异步生成器形态

    events = await collect(
        chat_event_stream(FailingAgent(), MagicMock(), "你好", "s2", "internal-2")
    )
    payload = json.loads(events[0][6:])
    assert payload["type"] == "error"
    assert payload["code"] == "generation_failed"


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"
