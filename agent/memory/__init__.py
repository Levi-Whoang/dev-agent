"""记忆管理：一次 prepare() 组装出本次请求该带上的全部上下文。

三层各管一段，失败互不影响：

    ┌ 长期记忆（user_memories 表）  跨会话，"这个人是谁"
    ├ 会话摘要（chat_sessions.summary）  会话级，"之前聊到哪了"
    └ 最近几轮原文（chat_messages）      精确，"刚才说了什么"

为什么都塞进一条 system 消息里，而不是插多条 system：
不同服务商对"多条 system"的处理不一致，有的直接 400，有的只认第一条。
折成一条最稳，代价只是 token 上没区别。

失败策略是刻意设计成"降级而不是报错"的：
摘要挂了 → 用旧摘要；长期记忆查不到 → 跳过。
记忆是增强项，不能因为它出问题就让用户发不出消息。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agent.memory import longterm
from agent.memory.summarizer import summarize
from agent.memory.tokens import estimate_messages_tokens, estimate_tokens
from agent.memory.window import (
    MemoryMessage,
    drop_orphan_tools,
    fit_turns,
    shrink_turn,
    split_turns,
)
from agent.store import SessionStore

logger = logging.getLogger("dev-agent.memory")


@dataclass(slots=True)
class MemoryContext:
    """拼装好的上下文。"""

    system_prompt: str
    history: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


class MemoryManager:
    def __init__(
        self,
        store: SessionStore,
        session_id: str,
        user_id: int | None = None,
        client: Any = None,
        model: str = "gpt-4o-mini",
        window_tokens: int = 3000,
        max_turns: int = 12,
        summary_enabled: bool = True,
        summary_trigger_tokens: int = 800,
        longterm_enabled: bool = True,
    ) -> None:
        self.store = store
        self.session_id = session_id
        self.user_id = user_id
        self.client = client
        self.model = model
        self.window_tokens = window_tokens
        self.max_turns = max_turns
        self.summary_enabled = summary_enabled and client is not None
        self.summary_trigger_tokens = summary_trigger_tokens
        self.longterm_enabled = longterm_enabled and user_id is not None

    async def prepare(
        self, system_prompt: str, user_input: str
    ) -> MemoryContext:
        """组装 system prompt 与历史消息（不含本次的 user_input）。"""
        rows = await self.store.message_rows(self.session_id)
        # store 返回 ORM 行，这里换成记忆层自己的类型：
        # 一来 window 是纯函数、不依赖 SQLAlchemy，测试时不用造数据库；
        # 二来 id 要跟着走，摘要靠它记"覆盖到哪条"。
        messages = [
            MemoryMessage(
                id=row.id,
                role=row.role,
                content=row.content or "",
                payload=row.payload,
            )
            for row in rows
        ]
        turns = split_turns(messages)

        kept, dropped = fit_turns(turns, self.window_tokens, self.max_turns)

        # 单轮就超预算时，把这一轮压成"提问 + 最终回答"
        if kept and len(kept) == 1 and kept[0].tokens > self.window_tokens:
            shrunk = shrink_turn(kept[0])
            if shrunk.messages:
                kept = [shrunk]
                logger.info("单轮超出窗口预算，已压缩为提问+回答")

        summary = await self._refresh_summary(dropped)

        blocks = [system_prompt.rstrip()]
        memory_block = await self._longterm_block(user_input)
        if memory_block:
            blocks.append(memory_block)
        if summary:
            blocks.append(f"【早期对话摘要】\n{summary}")

        history: list[dict[str, Any]] = []
        for turn in kept:
            history.extend(turn.to_llm())

        history = drop_orphan_tools(history)

        return MemoryContext(
            system_prompt="\n\n".join(blocks),
            history=history,
            stats={
                "turns_kept": len(kept),
                "turns_dropped": len(dropped),
                "history_messages": len(history),
                "history_tokens": estimate_messages_tokens(history),
                "system_tokens": estimate_tokens("\n\n".join(blocks)),
                "has_summary": bool(summary),
                "memory_block": bool(memory_block),
                # 本次请求的 user_input 也要占预算，一起报出来方便排查
                "user_input_tokens": estimate_tokens(user_input),
            },
        )

    # ---------- 摘要 ----------
    async def _refresh_summary(self, dropped: list) -> str:
        """把新挤出去的轮次压进摘要。只压"尚未被摘要覆盖"的部分。"""
        # 先短路：关掉摘要时连 session 都不用查。
        # 这不只是省一次查询 —— 它让"记忆层"对 store 的依赖降到最低，
        # 测试和评估脚本里的假 store 只需要实现 message_rows 就能跑。
        if not self.summary_enabled:
            return ""

        session = await self.store.get_session(self.session_id)
        if session is None:
            return ""

        previous = session.summary or ""
        covered_id = session.summary_upto_id or 0

        if not dropped:
            return previous

        fresh = [t for t in dropped if t.last_id > covered_id]
        if not fresh:
            return previous

        pending = sum(t.tokens for t in fresh)
        if pending < self.summary_trigger_tokens:
            # 攒着，等挤出去的内容够多了再压一次，避免每轮都多打一次模型
            return previous

        merged = await summarize(self.client, self.model, fresh, previous)
        if not merged:
            return previous

        await self.store.save_summary(self.session_id, merged, fresh[-1].last_id)
        logger.info(
            "会话摘要已更新：覆盖到消息 #%s，压缩了 %d 轮",
            fresh[-1].last_id,
            len(fresh),
        )
        return merged

    # ---------- 长期记忆 ----------
    async def _longterm_block(self, query: str) -> str:
        if not self.longterm_enabled:
            return ""

        try:
            memories = await longterm.list_all(self.store.db, self.user_id)
        except Exception:
            logger.exception("读取长期记忆失败，本次跳过")
            return ""

        picked = longterm.select_relevant(memories, query)
        if not picked:
            return ""

        await self.store.touch_memories([m.id for m in picked])
        return longterm.render_block(picked)


def build_manager(store: SessionStore, session_id: str, **kwargs: Any) -> MemoryManager:
    """给 factory 用的薄封装，参数名集中在这里，方便以后加开关。"""
    return MemoryManager(store=store, session_id=session_id, **kwargs)


__all__ = [
    "MemoryContext",
    "MemoryManager",
    "MemoryMessage",
    "build_manager",
    "longterm",
    "split_turns",
]
