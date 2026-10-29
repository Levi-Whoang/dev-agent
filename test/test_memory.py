"""记忆层测试。

分两类：
1. 纯函数（窗口切分、token 估算、压缩）—— 不碰数据库，跑得快，可以参数化扫大量边界；
2. 需要数据库的（摘要落库、长期记忆、remember 工具）。

第一类里最重要的那条是 `test_no_orphan_tool_at_any_boundary`：
它把"窗口边界不能切断工具调用配对"钉死。之前按行数裁剪的写法，
在这组参数下会有一半的用例报出孤立 tool。
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from agent.memory import longterm
from agent.memory.summarizer import render_transcript, summarize
from agent.memory.tokens import estimate_tokens
from agent.memory.window import (
    MemoryMessage,
    drop_orphan_tools,
    fit_turns,
    shrink_turn,
    split_turns,
)


# ---------- 构造工具 ----------
def build_history(turn_count: int, tools_per_turn: int = 1) -> list[MemoryMessage]:
    """造一段"真实形状"的历史：每轮 = user → (assistant 调工具 + tool) × n → assistant 回答。"""
    messages: list[MemoryMessage] = []
    next_id = 1

    def add(role: str, content: str, payload: dict | None = None) -> None:
        nonlocal next_id
        messages.append(
            MemoryMessage(id=next_id, role=role, content=content, payload=payload)
        )
        next_id += 1

    for turn in range(turn_count):
        add("user", f"第 {turn} 个问题")
        for i in range(tools_per_turn):
            call_id = f"call_{turn}_{i}"
            add(
                "assistant",
                "",
                {
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "ks", "arguments": "{}"},
                        }
                    ]
                },
            )
            add("tool", f"第 {turn} 轮第 {i} 次检索结果", {"tool_call_id": call_id, "name": "ks"})
        add("assistant", f"第 {turn} 个回答")

    return messages


def find_orphan_tools(flat: list[dict]) -> list[int]:
    """OpenAI 协议：每条 tool 前面必须是带 tool_calls 的 assistant。"""
    bad = []
    for index, message in enumerate(flat):
        if message.get("role") != "tool":
            continue
        prev = flat[index - 1] if index else None
        if not prev or prev.get("role") != "assistant" or not prev.get("tool_calls"):
            bad.append(index)
    return bad


def find_unanswered_calls(flat: list[dict]) -> list[str]:
    """反过来查：assistant 声明了 tool_calls，后面必须跟着对应的 tool 结果。"""
    missing = []
    for index, message in enumerate(flat):
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        declared = {c["id"] for c in message["tool_calls"]}
        # 往后收集连续的 tool 消息
        for follow in flat[index + 1 :]:
            if follow.get("role") != "tool":
                break
            declared.discard(follow.get("tool_call_id"))
        missing.extend(sorted(declared))
    return missing


# ---------- 窗口：钉死 bug ----------
@pytest.mark.parametrize("turn_count", range(1, 10))
@pytest.mark.parametrize("max_turns", [1, 2, 3, 5, 12])
@pytest.mark.parametrize("tools_per_turn", [1, 3])
def test_no_orphan_tool_at_any_boundary(turn_count, max_turns, tools_per_turn):
    """不管在哪个位置切窗口，都不能出现孤立 tool，也不能有没回应的 tool_calls。"""
    messages = build_history(turn_count, tools_per_turn)
    turns = split_turns(messages)

    kept, dropped = fit_turns(turns, budget_tokens=10_000, max_turns=max_turns)
    flat = [m.to_llm() for turn in kept for m in turn.messages]

    assert find_orphan_tools(flat) == []
    assert find_unanswered_calls(flat) == []
    # 保留 + 挤出去的轮，必须正好等于总轮数，一条都不能凭空消失
    assert len(kept) + len(dropped) == turn_count


def test_turn_boundary_is_user_message():
    messages = build_history(3, tools_per_turn=2)
    turns = split_turns(messages)

    assert len(turns) == 3
    for turn in turns:
        assert turn.messages[0].role == "user"
        # 一轮里 assistant 和它的 tool 都在里面
        assert any(m.role == "tool" for m in turn.messages)


def test_leading_non_user_messages_are_dropped():
    """首条不是 user 时（数据被手工改过），不能建"影子轮"，否则又可能带出孤立 tool。"""
    messages = [
        MemoryMessage(id=1, role="tool", content="孤儿", payload={"tool_call_id": "x"}),
        MemoryMessage(id=2, role="user", content="你好"),
        MemoryMessage(id=3, role="assistant", content="你好"),
    ]
    turns = split_turns(messages)

    assert len(turns) == 1
    assert turns[0].messages[0].id == 2


def test_fit_respects_token_budget():
    messages = build_history(10, tools_per_turn=1)
    turns = split_turns(messages)

    kept, dropped = fit_turns(turns, budget_tokens=60, max_turns=100)

    assert sum(t.tokens for t in kept) <= 60
    assert len(dropped) > 0
    # 保留的一定是最近的几轮
    assert kept == turns[len(turns) - len(kept) :]


def test_fit_keeps_at_least_one_turn():
    """预算小到装不下任何东西时，也得留最后一轮 —— 否则等于把上下文清空了。"""
    messages = build_history(2, tools_per_turn=1)
    turns = split_turns(messages)

    kept, _ = fit_turns(turns, budget_tokens=1, max_turns=10)

    assert len(kept) == 1
    assert kept[0] is turns[-1]


def test_shrink_turn_keeps_pairing_intact():
    """单轮超预算时压缩这一轮，同样不能留下没有 tool 回应的 tool_calls。"""
    messages = build_history(1, tools_per_turn=4)
    turn = split_turns(messages)[0]

    shrunk = shrink_turn(turn)
    flat = [m.to_llm() for m in shrunk.messages]

    assert find_orphan_tools(flat) == []
    assert find_unanswered_calls(flat) == []
    # 压缩后只剩提问和最终回答
    assert [m["role"] for m in flat] == ["user", "assistant"]


def test_drop_orphan_tools_removes_only_the_orphan():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "tool", "content": "无主", "tool_call_id": "a"},
        {"role": "assistant", "content": "a"},
    ]
    cleaned = drop_orphan_tools(messages)

    assert [m["role"] for m in cleaned] == ["user", "assistant"]


# ---------- token 估算 ----------
def test_estimate_tokens_basic():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0
    # 中文按字算
    assert estimate_tokens("公司在哪里") == 5
    # 英文 4 字符 1 token，向上取整
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


# ---------- 摘要 ----------
def test_render_transcript_skips_empty_and_truncates_tool():
    messages = build_history(1, tools_per_turn=1)
    turn = split_turns(messages)[0]

    # 塞一条超长的工具结果，确认会被截断
    turn.messages[2].content = "结" * 500
    text = render_transcript([turn])

    assert "用户：" in text
    assert len(text) < 400  # 500 字的工具结果被截到 160


async def test_summarize_falls_back_on_error():
    """摘要失败必须返回旧摘要，不能把异常抛给聊天主流程。"""

    class BoomClient:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    raise RuntimeError("模型挂了")

    messages = build_history(1, tools_per_turn=1)
    text = await summarize(BoomClient(), "m", split_turns(messages), "旧摘要内容")

    assert text == "旧摘要内容"


# ---------- 长期记忆（需要数据库） ----------
async def make_user(db, prefix: str = "mem") -> int:
    from auth.passwords import hash_password
    from database.models import User

    user = User(username=f"{prefix}_{uuid.uuid4().hex[:8]}", password_hash=hash_password("x" * 10))
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user.id


async def test_longterm_remember_dedup(db_ready):
    from database.base import get_session_factory

    async with get_session_factory()() as db:
        user_id = await make_user(db)

        item, created = await longterm.remember(db, user_id, "用户在深圳")
        assert created is True

        # 完全重复 → 不新增
        _, created_again = await longterm.remember(db, user_id, "用户在深圳")
        assert created_again is False

        # 包含关系也算重复（标点空白差异也会被归一化掉）
        _, created_third = await longterm.remember(db, user_id, "用户在深圳，")
        assert created_third is False

        assert len(await longterm.list_all(db, user_id)) == 1


async def test_longterm_select_prefers_relevant(db_ready):
    from database.base import get_session_factory

    async with get_session_factory()() as db:
        user_id = await make_user(db)
        await longterm.remember(db, user_id, "用户是后端开发，主要写 Python")
        await longterm.remember(db, user_id, "用户住在深圳南山区")
        await longterm.remember(db, user_id, "用户偏好简洁的回答")

        memories = await longterm.list_all(db, user_id)
        picked = longterm.select_relevant(memories, "Python 岗位要求")

        # 相关的那条排在最前
        assert "Python" in picked[0].content

        block = longterm.render_block(picked)
        assert "已知信息" in block
        assert "Python" in block


async def test_longterm_forget(db_ready):
    from database.base import get_session_factory

    async with get_session_factory()() as db:
        user_id = await make_user(db)
        await longterm.remember(db, user_id, "用户喜欢猫")
        await longterm.remember(db, user_id, "用户在深圳")

        removed = await longterm.forget(db, user_id, "猫")

        assert removed == 1
        remaining = [m.content for m in await longterm.list_all(db, user_id)]
        assert remaining == ["用户在深圳"]


# ---------- 门面：一次 prepare 组装三层 ----------
async def test_prepare_injects_longterm_memory(db_ready):
    from agent.memory import MemoryManager
    from agent.store import SessionStore
    from database.base import get_session_factory

    async with get_session_factory()() as db:
        user_id = await make_user(db)
        await longterm.remember(db, user_id, "用户是后端开发，只看 Python 岗位", kind="profile")

        store = SessionStore(db)
        session_id = f"s-{uuid.uuid4().hex[:8]}"
        await store.get_or_create(session_id, user_id)

        manager = MemoryManager(
            store=store,
            session_id=session_id,
            user_id=user_id,
            summary_enabled=False,
        )
        context = await manager.prepare("你是助手", "有什么岗位推荐")

        assert "Python" in context.system_prompt
        assert context.stats["memory_block"] is True
        assert context.history == []


async def test_prepare_windows_history_by_turns(db_ready):
    from agent.memory import MemoryManager
    from agent.memory.window import MemoryMessage
    from agent.store import SessionStore
    from database.base import get_session_factory

    async with get_session_factory()() as db:
        user_id = await make_user(db)
        store = SessionStore(db)
        session_id = f"s-{uuid.uuid4().hex[:8]}"
        await store.get_or_create(session_id, user_id)

        for i in range(6):
            await store.append(session_id, "user", f"问题 {i}")
            await store.append(session_id, "assistant", f"回答 {i}")

        manager = MemoryManager(
            store=store,
            session_id=session_id,
            user_id=user_id,
            max_turns=3,
            summary_enabled=False,
        )
        context = await manager.prepare("你是助手", "新问题")

        assert context.stats["turns_kept"] == 3
        assert context.stats["turns_dropped"] == 3
        # 保留的是最近三轮
        assert context.history[0]["content"] == "问题 3"
        assert find_orphan_tools(context.history) == []


async def test_summary_is_generated_and_reused(db_ready):
    """挤出窗口的对话要被压成摘要，下一次 prepare 能直接拿到。"""
    from agent.memory import MemoryManager
    from agent.store import SessionStore
    from database.base import get_session_factory

    class FakeSummaryClient:
        def __init__(self) -> None:
            self.calls = 0
            outer = self

            class _Completions:
                @staticmethod
                async def create(**kwargs):
                    outer.calls += 1
                    return SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content="- 用户问过公司地址")
                            )
                        ]
                    )

            self.chat = SimpleNamespace(completions=_Completions())

    async with get_session_factory()() as db:
        user_id = await make_user(db)
        store = SessionStore(db)
        session_id = f"s-{uuid.uuid4().hex[:8]}"
        await store.get_or_create(session_id, user_id)

        for i in range(8):
            await store.append(session_id, "user", f"问题 {i} " + "补" * 40)
            await store.append(session_id, "assistant", f"回答 {i} " + "充" * 40)

        client = FakeSummaryClient()
        manager = MemoryManager(
            store=store,
            session_id=session_id,
            user_id=user_id,
            client=client,
            max_turns=2,
            summary_enabled=True,
            summary_trigger_tokens=10,  # 强制立刻触发
        )

        first = await manager.prepare("你是助手", "新问题")
        assert client.calls == 1
        assert "早期对话摘要" in first.system_prompt

        session = await store.get_session(session_id)
        assert session.summary
        assert session.summary_upto_id > 0

        # 第二次：没有任何新挤出去的轮，不该再调模型
        second = await manager.prepare("你是助手", "又一个问题")
        assert client.calls == 1
        assert "早期对话摘要" in second.system_prompt


# ---------- remember 工具 ----------
async def test_remember_tool_writes_memory(db_ready):
    from database.base import get_session_factory
    from tools.memory_tool import make_remember_tool

    async with get_session_factory()() as db:
        user_id = await make_user(db)
        remember = make_remember_tool(db, user_id, "s-test")

        result = await remember("用户是后端开发，主要写 Python", kind="profile")

        assert result["saved"] is True
        assert result["created"] is True
        stored = await longterm.list_all(db, user_id)
        assert stored[0].kind == "profile"
        assert stored[0].source_session_id == "s-test"


async def test_remember_tool_rejects_empty(db_ready):
    from database.base import get_session_factory
    from tools.memory_tool import make_remember_tool

    async with get_session_factory()() as db:
        user_id = await make_user(db)
        remember = make_remember_tool(db, user_id)

        result = await remember("   ")

        assert result["saved"] is False
