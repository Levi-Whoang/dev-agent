"""历史摘要与上下文 token 预算管理。

完整消息始终保存在会话表中；本模块只在调用模型前构造一份上下文：
较早的轮次合并为摘要，最近若干轮保留原文，并在超过预算时逐级降级。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from agent.timeouts import Deadline

logger = logging.getLogger(__name__)

# 摘要消息使用独立的 system 消息，避免覆盖 Agent 的主系统提示。
SUMMARY_PREFIX = (
    "以下是更早对话的摘要（按时间顺序压缩；若与最近对话冲突，以最近对话为准）："
)
SUMMARY_SYSTEM_PROMPT = (
    "你是对话历史压缩助手。请把“已有摘要”和“新增对话”合并成一段供后续对话使用的中文摘要。"
    "必须保留：用户目标与偏好、已经确认的事实和结论、关键数字、文件路径与代码位置、"
    "未完成事项和待确认问题。删除寒暄、重复内容和工具调用细节。"
    "不得编造原文没有的信息。只输出摘要正文，不要标题或解释。"
)
CLIP_MARKER = "…（中间内容已省略）"
SUMMARY_CHUNK_TOKENS = 6000


@dataclass(frozen=True)
class HistoryConfig:
    """历史上下文相关配置，均可通过环境变量覆盖。"""

    recent_turns: int = 10
    token_budget: int = 12000
    summary_max_tokens: int = 600
    summary_timeout: float = 20.0
    # 单条工具结果在降级时允许保留的字符数。
    tool_result_max_chars: int = 2000
    # 单条普通消息在降级时允许保留的字符数。
    message_max_chars: int = 6000
    # 缩小保留轮数时的下限（仍可能继续降到 1 轮）。
    min_recent_turns: int = 2

    @property
    def summary_max_chars(self) -> int:
        """按中英混合文本估算摘要的最大字符数。"""
        return self.summary_max_tokens * 3

    @classmethod
    def from_env(cls) -> "HistoryConfig":
        def read_int(name: str, default: int, minimum: int) -> int:
            raw = os.getenv(name)
            if raw is None or not raw.strip():
                return default
            try:
                return max(minimum, int(raw))
            except ValueError:
                logger.warning("history_config_invalid name=%s value=%r", name, raw)
                return default

        def read_float(name: str, default: float, minimum: float) -> float:
            raw = os.getenv(name)
            if raw is None or not raw.strip():
                return default
            try:
                return max(minimum, float(raw))
            except ValueError:
                logger.warning("history_config_invalid name=%s value=%r", name, raw)
                return default

        return cls(
            recent_turns=read_int("HISTORY_RECENT_TURNS", 10, 1),
            token_budget=read_int("HISTORY_TOKEN_BUDGET", 12000, 1000),
            summary_max_tokens=read_int("HISTORY_SUMMARY_MAX_TOKENS", 600, 100),
            summary_timeout=read_float("HISTORY_SUMMARY_TIMEOUT", 20.0, 1.0),
        )


def _is_cjk(code: int) -> bool:
    return (
        0x3400 <= code <= 0x4DBF  # 扩展 A
        or 0x4E00 <= code <= 0x9FFF  # 基本汉字
        or 0xF900 <= code <= 0xFAFF  # 兼容汉字
        or 0x3040 <= code <= 0x30FF  # 日文假名
        or 0xAC00 <= code <= 0xD7AF  # 韩文
        or 0x3000 <= code <= 0x303F  # 中文标点
        or 0xFF00 <= code <= 0xFFEF  # 全角字符
    )


def estimate_tokens(text: str | None) -> int:
    """无第三方分词器时的保守 token 估算。

    中日韩字符及全角符号按 1 token 计，其余字符按约 4 字符 1 token 计。
    只用于预算控制，不要求与真实模型分词完全一致。
    """
    if not text:
        return 0
    cjk = 0
    narrow = 0
    for char in text:
        code = ord(char)
        if _is_cjk(code):
            cjk += 1
        elif code < 128:
            narrow += 1
        else:
            # emoji 等其他字符按 1 token 估算，避免低估。
            cjk += 1
    return cjk + (narrow + 3) // 4


def clip_text(text: str, limit: int) -> str:
    """按字符数截断文本，保留开头和结尾，便于模型看到首尾信息。"""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= len(CLIP_MARKER):
        return text[:limit]
    head = int((limit - len(CLIP_MARKER)) * 0.7)
    tail = limit - len(CLIP_MARKER) - head
    suffix = text[-tail:] if tail else ""
    return text[:head] + CLIP_MARKER + suffix


def truncate_to_tokens(text: str, max_tokens: int, suffix: str = "…") -> str:
    """按估算 token 数截断文本；预算不足时返回空字符串。"""
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    suffix_tokens = estimate_tokens(suffix)
    target = max(0, max_tokens - suffix_tokens)
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(text[:middle]) <= target:
            low = middle
        else:
            high = middle - 1
    return text[:low] + suffix if low else suffix


def message_tokens(message: dict) -> int:
    """估算单条消息的 token，包含角色、正文和工具调用参数。"""
    total = 4 + estimate_tokens(str(message.get("role") or ""))
    content = message.get("content")
    if isinstance(content, str):
        total += estimate_tokens(content)
    elif content is not None:
        total += estimate_tokens(
            json.dumps(content, ensure_ascii=False, default=str)
        )
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict):
            function = call.get("function") or {}
            name = function.get("name", "") if isinstance(function, dict) else ""
            arguments = (
                function.get("arguments", "") if isinstance(function, dict) else ""
            )
        else:
            function = getattr(call, "function", None)
            name = getattr(function, "name", "") or ""
            arguments = getattr(function, "arguments", "") or ""
        total += 8 + estimate_tokens(str(name)) + estimate_tokens(str(arguments))
    return total


def count_tokens(messages: list[dict]) -> int:
    """估算一组消息的总 token。"""
    return sum(message_tokens(message) for message in messages)


def split_history(messages: list[dict]) -> tuple[list[dict], list[list[dict]]]:
    """拆出 system 消息，并把其余消息按用户提问切分为轮次。"""
    remaining = list(messages)
    system: list[dict] = []
    while remaining and remaining[0].get("role") == "system":
        system.append(remaining.pop(0))
    turns: list[list[dict]] = []
    for message in remaining:
        if message.get("role") == "user" or not turns:
            turns.append([])
        turns[-1].append(message)
    return system, turns


def _flatten(turns: list[list[dict]]) -> list[dict]:
    return [message for turn in turns for message in turn]


def _content_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, default=str)


def _last_user_index(messages: list[dict]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return None


def _drop_oldest_turn(messages: list[dict]) -> bool:
    """删除最旧的一整轮；返回是否实际删除。"""
    user_indices = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user"
    ]
    if len(user_indices) < 2:
        return False
    start, end = user_indices[0], user_indices[1]
    del messages[start:end]
    return True


class HistoryManager:
    """维护“历史摘要 + 最近轮次”，并在超出 token 预算时降级。"""

    def __init__(self, client=None, session_manager=None):
        self.client = client
        self.session_manager = session_manager

    async def prepare(
        self,
        session_id: str,
        history: list[dict],
        deadline: Deadline | None = None,
    ) -> list[dict]:
        """构造本轮发给模型的上下文。"""
        config = HistoryConfig.from_env()
        system, turns = split_history(history)
        total_messages = sum(len(turn) for turn in turns)

        stored = await self._load_summary(session_id)
        if stored is not None and stored["covered_messages"] > total_messages:
            # 历史被清空或缩短后，旧摘要不再可信。
            logger.info(
                "history_summary_reset session_id=%s covered=%s total=%s",
                session_id,
                stored["covered_messages"],
                total_messages,
            )
            await self._delete_summary(session_id)
            stored = None

        keep_from = max(0, len(turns) - config.recent_turns)
        older_count = sum(len(turn) for turn in turns[:keep_from])
        summary = ""
        covered = 0
        if stored is not None:
            covered = stored["covered_messages"]
            # 摘要覆盖范围已经进入保留窗口时先不使用，避免与原文重复。
            if covered <= older_count:
                summary = stored["content"]
            else:
                # 调大保留轮数后，旧摘要可能覆盖了部分保留窗口；直接忽略
                # 会丢失窗口之前的消息，因此按当前原文重建摘要。
                logger.info(
                    "history_summary_rebuild session_id=%s covered=%s older=%s",
                    session_id,
                    covered,
                    older_count,
                )
                await self._delete_summary(session_id)
                covered = 0

        if older_count > covered:
            delta = _flatten(turns[:keep_from])[covered:older_count]
            summary = await self._merge_summary(
                summary, delta, config, session_id, deadline
            )
            covered = older_count
            await self._save_summary(session_id, summary, covered)

        context = list(system)
        if summary:
            context.append(
                {"role": "system", "content": f"{SUMMARY_PREFIX}\n{summary}"}
            )
        context.extend(_flatten(turns[keep_from:]))
        return self.enforce_budget(context, config, session_id)

    def enforce_budget(
        self,
        messages: list[dict],
        config: HistoryConfig | None = None,
        session_id: str = "",
    ) -> list[dict]:
        """在不破坏工具调用配对的前提下把上下文压到预算内。

        降级顺序：压缩工具结果 → 缩小保留轮数 → 截断普通消息 →
        丢弃摘要 → 只保留最近一轮 → 截断当前问题。
        """
        config = config or HistoryConfig.from_env()
        work = [dict(message) for message in messages]
        before = count_tokens(work)
        if before <= config.token_budget:
            return work

        level = 0
        changed = False

        # 1. 工具结果通常最大，优先保留首尾。
        for message in work:
            if message.get("role") != "tool":
                continue
            content = _content_text(message)
            if len(content) > config.tool_result_max_chars:
                message["content"] = clip_text(
                    content, config.tool_result_max_chars
                )
                changed = True
        if changed:
            level = 1
        if count_tokens(work) <= config.token_budget:
            return self._log_degrade(session_id, config, level, before, work)

        # 2. 缩小保留轮数，但先保留 min_recent_turns 轮。
        dropped = False
        while (
            count_tokens(work) > config.token_budget
            and self._user_turn_count(work) > config.min_recent_turns
        ):
            if not _drop_oldest_turn(work):
                break
            dropped = True
        if dropped:
            level = 2
        if count_tokens(work) <= config.token_budget:
            return self._log_degrade(session_id, config, level, before, work)

        # 3. 截断保留窗口内的长消息，当前问题暂不截断。
        last_user = _last_user_index(work)
        clipped = False
        for index, message in enumerate(work):
            if message.get("role") == "system" or index == last_user:
                continue
            content = _content_text(message)
            if len(content) > config.message_max_chars:
                message["content"] = clip_text(
                    content, config.message_max_chars
                )
                clipped = True
        if clipped:
            level = 3
        if count_tokens(work) <= config.token_budget:
            return self._log_degrade(session_id, config, level, before, work)

        # 4. 摘要也占预算，必要时先丢弃摘要。
        kept = [
            message
            for message in work
            if not (
                message.get("role") == "system"
                and _content_text(message).startswith(SUMMARY_PREFIX)
            )
        ]
        if len(kept) != len(work):
            work = kept
            level = 4
        if count_tokens(work) <= config.token_budget:
            return self._log_degrade(session_id, config, level, before, work)

        # 5. 只保留最近一轮。
        while (
            count_tokens(work) > config.token_budget
            and self._user_turn_count(work) > 1
        ):
            if not _drop_oldest_turn(work):
                break
            level = 5
        if count_tokens(work) <= config.token_budget:
            return self._log_degrade(session_id, config, level, before, work)

        # 6. 进一步压缩除当前问题外的正文。
        last_user = _last_user_index(work)
        for index, message in enumerate(work):
            if message.get("role") == "system" or index == last_user:
                continue
            limit = 300 if message.get("role") == "tool" else 800
            content = _content_text(message)
            if len(content) > limit:
                message["content"] = clip_text(content, limit)
                level = 6
        if count_tokens(work) <= config.token_budget:
            return self._log_degrade(session_id, config, level, before, work)

        # 7. 最后才截断当前问题本身，保证上下文不超预算。
        last_user = _last_user_index(work)
        if last_user is not None:
            other_tokens = count_tokens(
                [
                    message
                    for index, message in enumerate(work)
                    if index != last_user
                ]
            )
            # 消息本身还有角色与结构开销，正文预算要扣掉这部分。
            overhead = message_tokens(
                {"role": work[last_user].get("role"), "content": ""}
            )
            remaining = config.token_budget - other_tokens - overhead
            work[last_user]["content"] = truncate_to_tokens(
                _content_text(work[last_user]),
                max(0, remaining),
                "…（问题过长，已按上下文预算截断）",
            )
            level = 7

        if count_tokens(work) > config.token_budget:
            # 极端情况下只保留主系统提示，防止预算被彻底突破。
            system = next(
                (message for message in work if message.get("role") == "system"),
                None,
            )
            work = [system] if system is not None else work[:1]
            if work:
                overhead = message_tokens({"role": work[0].get("role"), "content": ""})
                work[0]["content"] = truncate_to_tokens(
                    _content_text(work[0]),
                    max(0, config.token_budget - overhead),
                    "…（系统提示已按预算截断）",
                )
        return self._log_degrade(session_id, config, level, before, work)

    @staticmethod
    def _user_turn_count(messages: list[dict]) -> int:
        return sum(1 for message in messages if message.get("role") == "user")

    @staticmethod
    def _log_degrade(
        session_id: str,
        config: HistoryConfig,
        level: int,
        before: int,
        messages: list[dict],
    ) -> list[dict]:
        if level:
            logger.info(
                "history_budget_degraded session_id=%s level=%s "
                "before=%s after=%s budget=%s",
                session_id,
                level,
                before,
                count_tokens(messages),
                config.token_budget,
            )
        return messages

    async def _load_summary(self, session_id: str) -> dict | None:
        if self.session_manager is None:
            return None
        loader = getattr(self.session_manager, "get_summary", None)
        if loader is None:
            return None
        data = await loader(session_id)
        if not isinstance(data, dict):
            return None
        content = data.get("content")
        if not isinstance(content, str) or not content.strip():
            return None
        try:
            covered = int(data.get("covered_messages", 0))
        except (TypeError, ValueError):
            return None
        return {"content": content.strip(), "covered_messages": max(0, covered)}

    async def _save_summary(
        self, session_id: str, content: str, covered_messages: int
    ) -> None:
        if self.session_manager is None:
            return
        saver = getattr(self.session_manager, "save_summary", None)
        if saver is None:
            return
        try:
            await saver(session_id, content, covered_messages)
        except Exception as exc:  # 摘要保存失败不应中断本轮问答。
            logger.warning(
                "history_summary_save_failed session_id=%s error=%s",
                session_id,
                exc,
            )

    async def _delete_summary(self, session_id: str) -> None:
        if self.session_manager is None:
            return
        remover = getattr(self.session_manager, "clear_summary", None)
        if remover is None:
            return
        try:
            await remover(session_id)
        except Exception as exc:
            logger.warning(
                "history_summary_delete_failed session_id=%s error=%s",
                session_id,
                exc,
            )

    async def _merge_summary(
        self,
        previous: str,
        messages: list[dict],
        config: HistoryConfig,
        session_id: str,
        deadline: Deadline | None = None,
    ) -> str:
        summary = previous.strip()
        for chunk in self._chunk_messages(messages):
            summary = await self._summarize(
                summary, chunk, config, session_id, deadline
            )
            summary = truncate_to_tokens(
                summary,
                config.summary_max_tokens,
                "\n…（摘要已按预算截断）",
            )
        return summary

    @staticmethod
    def _chunk_messages(messages: list[dict]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        tokens = 0
        for message in messages:
            line = HistoryManager._format_message(message)
            line_tokens = estimate_tokens(line)
            if current and tokens + line_tokens > SUMMARY_CHUNK_TOKENS:
                chunks.append("\n".join(current))
                current = []
                tokens = 0
            current.append(line)
            tokens += line_tokens
        if current:
            chunks.append("\n".join(current))
        return chunks

    @staticmethod
    def _format_message(message: dict) -> str:
        role = {
            "user": "用户",
            "assistant": "助手",
            "tool": "工具",
        }.get(message.get("role"), str(message.get("role")))
        content = clip_text(_content_text(message), 800)
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            names = []
            for call in tool_calls:
                if isinstance(call, dict):
                    function = call.get("function") or {}
                    names.append(function.get("name", "") if isinstance(function, dict) else "")
                else:
                    function = getattr(call, "function", None)
                    names.append(getattr(function, "name", "") or "")
            content += f"（请求工具：{', '.join(name for name in names if name)}）"
        return f"{role}：{content}"

    async def _summarize(
        self,
        previous: str,
        transcript: str,
        config: HistoryConfig,
        session_id: str,
        deadline: Deadline | None = None,
    ) -> str:
        if self.client is None:
            return self._fallback_summary(previous, transcript, config)
        # 摘要属于可降级步骤：整体预算不足时直接保留原话要点，不占用回答时间。
        timeout = config.summary_timeout
        if deadline is not None:
            remaining = deadline.remaining()
            if remaining <= 0.05:
                logger.info(
                    "history_summary_skipped session_id=%s reason=deadline",
                    session_id,
                )
                return self._fallback_summary(previous, transcript, config)
            timeout = min(timeout, remaining)
        user_content = (
            f"已有摘要：\n{previous or '（无）'}\n\n"
            f"需要合并的新对话：\n{transcript}\n\n"
            "请输出合并后的摘要。"
        )
        try:
            response = await self.client.chat.completions.create(
                model=os.getenv("LLM_MODEL"),
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=config.summary_max_tokens,
                timeout=timeout,
            )
            content = (response.choices[0].message.content or "").strip()
            if not content:
                raise ValueError("模型返回空摘要")
            logger.info(
                "history_summary session_id=%s input_tokens=%s output_tokens=%s",
                session_id,
                estimate_tokens(user_content),
                estimate_tokens(content),
            )
            return content
        except Exception as exc:
            logger.warning(
                "history_summary_failed session_id=%s error=%s", session_id, exc
            )
            return self._fallback_summary(previous, transcript, config)

    @staticmethod
    def _fallback_summary(
        previous: str, transcript: str, config: HistoryConfig
    ) -> str:
        """模型摘要失败时保留用户与助手原话要点，保证对话可以继续。"""
        new_lines = [
            line
            for line in transcript.splitlines()
            if line.startswith(("用户：", "助手："))
        ]
        new_text = "\n".join(new_lines) or transcript
        if not previous:
            return truncate_to_tokens(
                new_text, config.summary_max_tokens, "\n…（摘要已截断）"
            )
        old_limit = max(1, int(config.summary_max_tokens * 0.6))
        new_limit = max(1, config.summary_max_tokens - old_limit)
        old_text = truncate_to_tokens(previous, old_limit, "\n…")
        new_text = truncate_to_tokens(new_text, new_limit, "\n…（摘要已截断）")
        return f"{old_text}\n{new_text}"
