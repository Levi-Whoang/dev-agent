"""历史摘要与 token 预算管理测试。"""

import asyncio
import time
from types import SimpleNamespace

import pytest

from agent.history import (
    CLIP_MARKER,
    SUMMARY_PREFIX,
    HistoryConfig,
    HistoryManager,
    clip_text,
    count_tokens,
    estimate_tokens,
    message_tokens,
    split_history,
    truncate_to_tokens,
    _drop_oldest_turn,
)


class FakeSessionManager:
    """内存摘要存储，避免测试连接数据库。"""

    def __init__(self, summaries=None):
        self.summaries = dict(summaries or {})
        self.saved = []
        self.cleared = []

    async def get_summary(self, session_id):
        return self.summaries.get(session_id)

    async def save_summary(self, session_id, content, covered_messages):
        self.saved.append((session_id, content, covered_messages))
        self.summaries[session_id] = {
            "content": content,
            "covered_messages": covered_messages,
        }

    async def clear_summary(self, session_id):
        self.cleared.append(session_id)
        self.summaries.pop(session_id, None)


class FakeClient:
    """按顺序返回摘要结果；异常项会被抛出。"""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=result))]
        )


class SlowClient:
    """摘要请求会一直等待，直到测试显式放行。"""

    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        self.started.set()
        await self.release.wait()
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="后台模型摘要"))
            ]
        )


def make_client(*results):
    client = FakeClient(*results)
    return SimpleNamespace(chat=SimpleNamespace(completions=client)), client


def make_turn(index):
    return [
        {"role": "user", "content": f"问题{index}"},
        {"role": "assistant", "content": f"回答{index}"},
    ]


def make_history(turns, system="系统提示"):
    messages = [{"role": "system", "content": system}]
    for turn in turns:
        messages.extend(turn)
    return messages


@pytest.fixture(autouse=True)
def reset_history_env(monkeypatch):
    for name in (
        "HISTORY_RECENT_TURNS",
        "HISTORY_TOKEN_BUDGET",
        "HISTORY_SUMMARY_MAX_TOKENS",
        "HISTORY_SUMMARY_TIMEOUT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_history_config_reads_env_and_falls_back(monkeypatch):
    defaults = HistoryConfig.from_env()
    assert defaults.recent_turns == 10
    assert defaults.token_budget == 12000
    assert defaults.summary_max_tokens == 600
    assert defaults.summary_timeout == 20.0
    assert defaults.summary_max_chars == 1800

    monkeypatch.setenv("HISTORY_RECENT_TURNS", "3")
    monkeypatch.setenv("HISTORY_TOKEN_BUDGET", "2000")
    monkeypatch.setenv("HISTORY_SUMMARY_MAX_TOKENS", "200")
    monkeypatch.setenv("HISTORY_SUMMARY_TIMEOUT", "5.5")
    config = HistoryConfig.from_env()
    assert (config.recent_turns, config.token_budget) == (3, 2000)
    assert (config.summary_max_tokens, config.summary_timeout) == (200, 5.5)

    # 低于下限时向安全方向收敛，非法字符串回落到默认值。
    monkeypatch.setenv("HISTORY_RECENT_TURNS", "0")
    monkeypatch.setenv("HISTORY_TOKEN_BUDGET", "abc")
    config = HistoryConfig.from_env()
    assert config.recent_turns == 1
    assert config.token_budget == 12000


def test_estimate_tokens_handles_cjk_ascii_and_emoji():
    assert estimate_tokens(None) == 0
    assert estimate_tokens("") == 0
    assert estimate_tokens("你好") == 2
    assert estimate_tokens("，。") == 2
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens("😀") == 1


def test_clip_text_keeps_head_and_tail():
    text = "a" * 100
    clipped = clip_text(text, 20)
    assert len(clipped) == 20
    assert CLIP_MARKER in clipped
    assert clipped.startswith("a")
    assert clipped.endswith("a")
    assert clip_text("短文本", 20) == "短文本"
    assert clip_text("abcdef", 2) == "ab"
    assert clip_text("abcdef", 0) == ""


def test_truncate_to_tokens_respects_budget():
    text = "测" * 200
    result = truncate_to_tokens(text, 50)
    assert estimate_tokens(result) <= 50
    assert result.endswith("…")
    assert truncate_to_tokens("短", 50) == "短"
    assert truncate_to_tokens(text, 0) == ""


def test_message_and_count_tokens_include_tool_calls():
    plain = {"role": "user", "content": "你好"}
    assert message_tokens(plain) == 4 + 1 + 2

    with_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "function": {"name": "search", "arguments": "{\"q\": \"x\"}"},
            }
        ],
    }
    call_tokens = message_tokens(with_call)
    assert call_tokens > message_tokens({"role": "assistant", "content": None})
    assert count_tokens([plain, with_call]) == message_tokens(plain) + call_tokens


def test_split_history_groups_turns_and_extracts_system():
    history = make_history([make_turn(0), make_turn(1)])
    system, turns = split_history(history)
    assert [message["content"] for message in system] == ["系统提示"]
    assert len(turns) == 2
    assert turns[1][0]["content"] == "问题1"

    # 首条不是 user 时也单独成轮，避免消息丢失。
    system, turns = split_history(
        [{"role": "assistant", "content": "开场"}, {"role": "user", "content": "问题"}]
    )
    assert system == []
    assert len(turns) == 2


def test_drop_oldest_turn_keeps_tool_pairing():
    messages = [
        {"role": "user", "content": "一"},
        {"role": "assistant", "content": "答一"},
        {"role": "user", "content": "二"},
        {"role": "assistant", "content": "答二"},
    ]
    assert _drop_oldest_turn(messages) is True
    assert messages[0]["content"] == "二"
    assert _drop_oldest_turn(messages) is False


@pytest.mark.anyio
async def test_prepare_keeps_recent_turns_and_summarizes_older(monkeypatch):
    monkeypatch.setenv("HISTORY_RECENT_TURNS", "2")
    manager = FakeSessionManager()
    history_manager = HistoryManager(client=None, session_manager=manager)

    context = await history_manager.prepare(
        "s1", make_history([make_turn(i) for i in range(4)])
    )

    assert context[0] == {"role": "system", "content": "系统提示"}
    assert context[1]["role"] == "system"
    assert context[1]["content"].startswith(SUMMARY_PREFIX)
    assert "问题0" in context[1]["content"]
    assert "问题1" in context[1]["content"]
    assert [message["content"] for message in context[2:]] == [
        "问题2",
        "回答2",
        "问题3",
        "回答3",
    ]
    assert manager.saved[0][2] == 4
    assert manager.summaries["s1"]["covered_messages"] == 4


@pytest.mark.anyio
async def test_prepare_only_summarizes_new_older_turns(monkeypatch):
    monkeypatch.setenv("HISTORY_RECENT_TURNS", "2")
    manager = FakeSessionManager()
    history_manager = HistoryManager(client=None, session_manager=manager)

    await history_manager.prepare("s1", make_history([make_turn(i) for i in range(4)]))
    history = make_history([make_turn(i) for i in range(6)])
    context = await history_manager.prepare("s1", history)

    assert len(manager.saved) == 2
    assert manager.summaries["s1"]["covered_messages"] == 8
    summary = next(
        message["content"]
        for message in context
        if message["content"].startswith(SUMMARY_PREFIX)
    )
    assert "问题0" in summary
    assert "问题2" in summary
    # 最近两轮仍保留原文，不重复写进摘要。
    assert "问题4" not in summary

    # 没有新的更早消息时不再调用摘要，也不会产生重复摘要消息。
    before = len(manager.saved)
    context = await history_manager.prepare("s1", history)
    assert len(manager.saved) == before
    assert (
        sum(message["content"].startswith(SUMMARY_PREFIX) for message in context) == 1
    )


@pytest.mark.anyio
async def test_prepare_reuses_stored_summary_verbatim(monkeypatch):
    monkeypatch.setenv("HISTORY_RECENT_TURNS", "2")
    manager = FakeSessionManager(
        {"s1": {"content": "旧摘要", "covered_messages": 4}}
    )
    history_manager = HistoryManager(client=None, session_manager=manager)

    context = await history_manager.prepare(
        "s1", make_history([make_turn(i) for i in range(4)])
    )

    assert manager.saved == []
    assert context[1]["content"] == f"{SUMMARY_PREFIX}\n旧摘要"


@pytest.mark.anyio
async def test_prepare_discards_summary_after_history_shrinks(monkeypatch):
    monkeypatch.setenv("HISTORY_RECENT_TURNS", "2")
    manager = FakeSessionManager(
        {"s1": {"content": "过期摘要", "covered_messages": 99}}
    )
    history_manager = HistoryManager(client=None, session_manager=manager)

    context = await history_manager.prepare("s1", make_history([make_turn(0)]))

    assert manager.cleared == ["s1"]
    assert all(SUMMARY_PREFIX not in message["content"] for message in context)
    assert [message["content"] for message in context[1:]] == ["问题0", "回答0"]


@pytest.mark.anyio
async def test_prepare_rebuilds_summary_when_covered_exceeds_older_window(monkeypatch):
    # 保留 4 轮时，前 2 轮（4 条消息）属于“较早历史”。
    monkeypatch.setenv("HISTORY_RECENT_TURNS", "4")
    manager = FakeSessionManager(
        {"s1": {"content": "覆盖过多的旧摘要", "covered_messages": 8}}
    )
    history_manager = HistoryManager(client=None, session_manager=manager)

    context = await history_manager.prepare(
        "s1", make_history([make_turn(i) for i in range(6)])
    )

    # 旧摘要覆盖了保留窗口，不能再直接使用；按当前较早原文重建。
    assert manager.cleared == ["s1"]
    assert manager.summaries["s1"]["covered_messages"] == 4
    summary = next(
        message["content"]
        for message in context
        if message["content"].startswith(SUMMARY_PREFIX)
    )
    assert "问题0" in summary and "问题1" in summary
    assert "问题2" not in summary
    assert [message["content"] for message in context if message["role"] == "user"] == [
        "问题2",
        "问题3",
        "问题4",
        "问题5",
    ]


@pytest.mark.anyio
async def test_prepare_uses_client_summary_and_falls_back_on_error(monkeypatch):
    monkeypatch.setenv("HISTORY_RECENT_TURNS", "1")
    monkeypatch.setenv("HISTORY_SUMMARY_MAX_TOKENS", "100")

    client, calls = make_client("模型生成的摘要")
    manager = FakeSessionManager()
    history_manager = HistoryManager(client=client, session_manager=manager)
    context = await history_manager.prepare(
        "s1", make_history([make_turn(0), make_turn(1)])
    )

    assert len(calls.calls) == 1
    assert calls.calls[0]["max_tokens"] == 100
    assert calls.calls[0]["timeout"] == 20.0
    assert context[1]["content"].endswith("模型生成的摘要")

    client, _ = make_client(RuntimeError("模型超时"))
    manager = FakeSessionManager()
    history_manager = HistoryManager(client=client, session_manager=manager)
    context = await history_manager.prepare(
        "s2", make_history([make_turn(0), make_turn(1)])
    )
    # 失败时退回本地摘要，问答不中断。
    assert "问题0" in context[1]["content"]

    client, _ = make_client("")
    manager = FakeSessionManager()
    history_manager = HistoryManager(client=client, session_manager=manager)
    context = await history_manager.prepare(
        "s3", make_history([make_turn(0), make_turn(1)])
    )
    assert "问题0" in context[1]["content"]


@pytest.mark.anyio
async def test_prepare_does_not_block_on_slow_summary(monkeypatch):
    monkeypatch.setenv("HISTORY_RECENT_TURNS", "1")
    client = SlowClient()
    manager = FakeSessionManager()
    history_manager = HistoryManager(
        client=SimpleNamespace(chat=SimpleNamespace(completions=client)),
        session_manager=manager,
    )

    started_at = time.perf_counter()
    context = await history_manager.prepare(
        "s1", make_history([make_turn(0), make_turn(1), make_turn(2)])
    )
    elapsed = time.perf_counter() - started_at

    # 摘要模型未返回时，首包只等很短的宽限时间，先使用本地摘要。
    assert elapsed < 0.5
    assert "问题0" in context[1]["content"]
    assert manager.summaries["s1"]["covered_messages"] == 4
    await asyncio.wait_for(client.started.wait(), timeout=1)

    client.release.set()
    await history_manager.wait_for_summaries()
    stored = manager.summaries["s1"]
    assert stored["content"] == "后台模型摘要"
    assert stored["covered_messages"] == 4


@pytest.mark.anyio
async def test_prepare_tolerates_legacy_session_manager(monkeypatch):
    monkeypatch.setenv("HISTORY_RECENT_TURNS", "1")

    class LegacyManager:
        pass

    history_manager = HistoryManager(client=None, session_manager=LegacyManager())
    context = await history_manager.prepare(
        "s1", make_history([make_turn(0), make_turn(1)])
    )
    # 没有摘要接口时仍返回最近轮次。
    assert [
        message["content"]
        for message in context
        if message["role"] != "system"
    ] == ["问题1", "回答1"]


def test_enforce_budget_noop_returns_copy():
    messages = [
        {"role": "system", "content": "系统"},
        {"role": "user", "content": "你好"},
    ]
    result = HistoryManager().enforce_budget(messages, HistoryConfig())
    assert result == messages
    assert result is not messages
    assert result[0] is not messages[0]


def test_enforce_budget_clips_tool_result_first():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "问题"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "function": {"name": "search", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "x" * 3000},
    ]
    config = HistoryConfig(token_budget=600, tool_result_max_chars=100)

    result = HistoryManager().enforce_budget(messages, config)

    assert count_tokens(result) <= config.token_budget
    assert len(result[3]["content"]) == 100
    assert CLIP_MARKER in result[3]["content"]
    # 原始消息不被修改。
    assert len(messages[3]["content"]) == 3000


def test_enforce_budget_drops_whole_old_turns():
    turns = [make_turn(i) for i in range(6)]
    messages = make_history(turns)
    config = HistoryConfig(token_budget=60, recent_turns=10)

    result = HistoryManager().enforce_budget(messages, config)

    assert count_tokens(result) <= config.token_budget
    users = [message for message in result if message.get("role") == "user"]
    assert users[-1]["content"] == "问题5"
    # 保留的消息必须从一整轮开始，不能留下孤立的 assistant/tool。
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user"


def test_enforce_budget_keeps_tool_pairs_when_dropping():
    turns = []
    for index in range(4):
        turns.append(
            [
                {"role": "user", "content": f"问题{index}"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "function": {"name": "search", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call_{index}",
                    "content": "结果" * 100,
                },
                {"role": "assistant", "content": f"回答{index}"},
            ]
        )
    messages = make_history(turns)
    config = HistoryConfig(token_budget=400)

    result = HistoryManager().enforce_budget(messages, config)

    call_ids = {
        call["id"]
        for message in result
        for call in (message.get("tool_calls") or [])
    }
    tool_ids = {
        message["tool_call_id"]
        for message in result
        if message.get("role") == "tool"
    }
    assert tool_ids <= call_ids
    assert count_tokens(result) <= config.token_budget


def test_enforce_budget_drops_summary_before_current_question():
    messages = [
        {"role": "system", "content": "主提示"},
        {"role": "system", "content": f"{SUMMARY_PREFIX}\n" + "摘" * 200},
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "回答"},
    ]
    config = HistoryConfig(token_budget=100)

    result = HistoryManager().enforce_budget(messages, config)

    assert count_tokens(result) <= config.token_budget
    assert all(SUMMARY_PREFIX not in message["content"] for message in result)
    assert result[-1]["content"] == "回答"


def test_enforce_budget_truncates_current_question_as_last_resort():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "问" * 1000},
    ]
    config = HistoryConfig(token_budget=100)

    result = HistoryManager().enforce_budget(messages, config)

    assert count_tokens(result) <= config.token_budget
    assert len(result) == 2
    assert result[1]["content"].endswith("…（问题过长，已按上下文预算截断）")


def test_enforce_budget_extreme_fallback_fits_budget():
    messages = [{"role": "system", "content": "系" * 500}]
    config = HistoryConfig(token_budget=60)

    result = HistoryManager().enforce_budget(messages, config)

    assert count_tokens(result) <= config.token_budget
    assert len(result) == 1
    assert result[0]["content"].endswith("…（系统提示已按预算截断）")


@pytest.mark.anyio
async def test_summarize_strips_whitespace_and_passes_previous_summary():
    client, calls = make_client("  摘要正文 \n")
    manager = HistoryManager(client=client)
    config = HistoryConfig(summary_max_tokens=120)

    result = await manager._summarize("旧摘要", "用户：新内容", config, "s1")

    assert result == "摘要正文"
    assert calls.calls[0]["messages"][0]["role"] == "system"
    assert "旧摘要" in calls.calls[0]["messages"][1]["content"]
    assert "新内容" in calls.calls[0]["messages"][1]["content"]


# 测试统一使用项目的 asyncio 运行时。
@pytest.fixture
def anyio_backend():
    return "asyncio"
