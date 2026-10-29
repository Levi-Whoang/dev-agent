"""上下文窗口：按「轮」裁剪，不按「行」裁剪。

这是整个 memory 里最容易写错、也最容易在测试里蒙混过关的一块。

原来的写法是 `ORDER BY id DESC LIMIT 40` —— 取最近 40 **行**。问题在于
OpenAI 的对话协议有个硬约束：

    任何 role="tool" 的消息，前面必须紧跟一条带 tool_calls 的 assistant。

而一次工具调用会写入两行（assistant + tool）。40 是偶数，历史行数也是变化的，
于是窗口边界有可能正好落在一行 tool 上 —— 送出去的对话开头是一条孤立的 tool
消息，模型直接返回 400。

实测（logs/probe_memory.py）：41/43/45 行正常，42/44/46 行全部报错。
也就是说这个 bug 有一半的概率会触发，而且在开发时很难复现。

修法不是"少取一条"或者"跳过 tool"，而是换最小单位：
    一轮 = 一条 user 消息 + 它之后直到下一条 user 之前的所有消息
按轮裁剪，assistant 和它的 tool 结果天然在同一边，不可能被切开。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.memory.tokens import estimate_message_tokens


@dataclass(slots=True)
class MemoryMessage:
    """历史里的一条消息。带上数据库 id，才知道摘要覆盖到哪。"""

    id: int
    role: str
    content: str = ""
    payload: dict[str, Any] | None = None

    @property
    def tokens(self) -> int:
        return estimate_message_tokens(self.to_llm())

    def to_llm(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.payload:
            message.update(self.payload)
        # assistant 只调工具时 content 必须是 None，空字符串会被部分服务商拒绝
        if message["role"] == "assistant" and message.get("tool_calls"):
            message["content"] = message["content"] or None
        return message


@dataclass(slots=True)
class Turn:
    messages: list[MemoryMessage] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(m.tokens for m in self.messages)

    @property
    def last_id(self) -> int:
        return self.messages[-1].id if self.messages else 0

    def to_llm(self) -> list[dict[str, Any]]:
        return [m.to_llm() for m in self.messages]


def split_turns(messages: list[MemoryMessage]) -> list[Turn]:
    """按 user 消息切轮。

    开头那些不属于任何 user 的消息（正常不会出现，除非数据被手工改过）
    会被丢掉 —— 它们无法保证工具配对完整，留着比丢了危险。
    """
    turns: list[Turn] = []
    current: Turn | None = None

    for message in messages:
        if message.role == "user":
            current = Turn()
            turns.append(current)
        if current is None:
            # 首条不是 user：跳过，不建立"影子轮"
            continue
        current.messages.append(message)

    return turns


def fit_turns(
    turns: list[Turn], budget_tokens: int, max_turns: int
) -> tuple[list[Turn], list[Turn]]:
    """从最近的一轮往回装，直到装不下。

    返回 (保留的轮, 挤出去的轮)。挤出去的轮交给摘要器压缩，
    这样早期信息以摘要形式留存，而不是直接消失。
    """
    if not turns:
        return [], []

    kept: list[Turn] = []
    used = 0

    for turn in reversed(turns):
        if len(kept) >= max_turns:
            break
        cost = turn.tokens
        # 至少保留最后一轮：哪怕它自己就超预算，也比发一个空的上下文强
        if kept and used + cost > budget_tokens:
            break
        kept.append(turn)
        used += cost

    kept.reverse()
    return kept, turns[: len(turns) - len(kept)]


def shrink_turn(turn: Turn) -> Turn:
    """单轮就超预算时的兜底：只留提问和最终回答。

    注意不能简单地"删掉 tool 消息" —— assistant 里还带着 tool_calls，
    删了 tool 又变成孤立调用，一样报 400。所以这里连中间那些
    带 tool_calls 的 assistant 一起丢掉，只保留能独立成立的两条。
    """
    if len(turn.messages) <= 2:
        return turn

    user_part = [m for m in turn.messages if m.role == "user"][:1]
    finals = [m for m in turn.messages if m.role == "assistant" and not m.payload]

    if not finals:
        # 这一轮还没产出最终回答（比如上一轮报错了），只剩提问
        return Turn(user_part)

    return Turn(user_part + finals[-1:])


def drop_orphan_tools(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """最后一道防线：清掉没有对应 tool_calls 的 tool 消息。

    split_turns 理论上已经保证了配对，这个函数是给"外部传进来的消息"
    兜底的（比如以后有人自己拼 history）。成本很低，留着更安心。
    """
    cleaned: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "tool":
            prev = cleaned[-1] if cleaned else None
            if not prev or prev.get("role") != "assistant" or not prev.get("tool_calls"):
                continue
        cleaned.append(message)
    return cleaned
