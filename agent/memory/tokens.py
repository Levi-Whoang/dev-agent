"""Token 估算：不引 tiktoken。

为什么不用 tiktoken：
- 它要下载编码表（几 MB），而且对国产模型的 tokenizer 并不准；
- 我们只需要一个"别把上下文撑爆"的阈值判断，误差 10~20% 完全可接受。

误差方向是**故意估高**的：
中文 1 字算 1 token（gpt 系实际约 0.7~1.5），英文 4 字符算 1 token。
估高了最多是少留一点历史；估低了会直接撞上模型的上下文上限报错。
宁可保守。
"""

from __future__ import annotations

import json
from typing import Any

# 每条消息的结构开销（role、分隔符等），OpenAI 文档给的是 3~4
MESSAGE_OVERHEAD = 4


def _is_wide(ch: str) -> bool:
    """CJK 汉字、全角标点、中文标点 —— 这些在分词器里基本 1 字 1 token。"""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF  # 常用汉字
        or 0x3400 <= code <= 0x4DBF  # 扩展 A
        or 0x3000 <= code <= 0x303F  # 中文标点
        or 0xFF00 <= code <= 0xFFEF  # 全角字符
        or 0x3040 <= code <= 0x30FF  # 日文假名（混排时也会出现）
    )


def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0

    wide = 0
    narrow = 0
    for ch in text:
        if _is_wide(ch):
            wide += 1
        else:
            narrow += 1

    # 向上取整，(narrow + 3) // 4 就是 ceil(narrow / 4)
    return wide + (narrow + 3) // 4


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """单条消息的估算，含 tool_calls 里那一坨 JSON 参数。"""
    total = MESSAGE_OVERHEAD + estimate_tokens(message.get("content"))

    tool_calls = message.get("tool_calls")
    if tool_calls:
        # 参数是 JSON 字符串，直接按长度估；ensure_ascii=False 才对得上实际字节
        total += estimate_tokens(json.dumps(tool_calls, ensure_ascii=False))

    return total


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)
