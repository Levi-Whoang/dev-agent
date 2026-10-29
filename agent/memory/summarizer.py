"""滚动摘要：把挤出窗口的早期对话压成一段话。

为什么需要它：
窗口一裁，被裁掉的内容就永久丢了。用户第 3 轮说的"我们公司主要做跨境电商"，
到第 40 轮模型已经不知道了 —— 而这类信息恰恰是后面所有回答的前提。

摘要是一种有损压缩：细节会掉，但主线（用户是谁、聊过什么、
确认过哪些事实、哪些问题还没解决）能留下来，成本又远低于把原文全带上。

两个刻意的设计：
1. **增量**：每次只压缩"新挤出去的那几轮"，拼到旧摘要后面。
   全量重压会随着对话变长越来越慢、越来越贵。
2. **失败不致命**：摘要调用失败就沿用旧摘要，聊天继续。
   记忆是增强功能，不能因为它挂了就让用户发不出消息。
"""

from __future__ import annotations

import logging
from typing import Any

from agent.memory.window import Turn

logger = logging.getLogger("dev-agent.memory.summary")

SUMMARY_PROMPT = (
    "你在压缩一段对话历史，压缩结果会被后续对话当作背景资料使用。\n"
    "请输出纯文本要点，不要任何开场白或客套话。必须保留：\n"
    "1. 用户的身份、角色、已有的偏好或约束；\n"
    "2. 用户明确确认过的事实与数据（公司名、地点、岗位要求等具体信息）；\n"
    "3. 已经给出的结论和它的依据来源；\n"
    "4. 尚未解决的问题或用户表达过但没被满足的需求。\n"
    "丢弃：寒暄、重复表述、工具调用的原始 JSON。\n"
    "控制在 300 字以内，用「-」开头的短句罗列。"
)

# 工具返回的是原始 JSON，压进摘要里纯属浪费 token，截断即可
TOOL_CONTENT_LIMIT = 160
# 一次要压缩的原文上限，防止某轮塞了一个巨大检索结果把请求撑爆
TRANSCRIPT_LIMIT = 12000

_ROLE_LABEL = {
    "user": "用户",
    "assistant": "助手",
    "tool": "工具结果",
}


def render_transcript(turns: list[Turn]) -> str:
    lines: list[str] = []

    for turn in turns:
        for message in turn.messages:
            text = (message.content or "").strip()
            if not text:
                # 只调工具的 assistant 没有正文，跳过但保留配对关系不影响摘要
                continue
            if message.role == "tool":
                text = text[:TOOL_CONTENT_LIMIT]
            lines.append(f"{_ROLE_LABEL.get(message.role, message.role)}：{text}")

    transcript = "\n".join(lines)
    if len(transcript) > TRANSCRIPT_LIMIT:
        # 只留尾部：越靠后的内容越接近当前话题
        transcript = "……（前文略）\n" + transcript[-TRANSCRIPT_LIMIT:]

    return transcript


async def summarize(
    client: Any,
    model: str,
    turns: list[Turn],
    previous_summary: str | None = None,
) -> str:
    """把 turns 压成一段摘要，并接在 previous_summary 后面。

    任何异常都返回旧摘要（可能为 None），绝不向上抛 —— 调用方不该
    因为"记不住"而无法回答。
    """
    if not turns:
        return previous_summary or ""

    transcript = render_transcript(turns)
    if not transcript.strip():
        return previous_summary or ""

    user_content = transcript
    if previous_summary:
        user_content = (
            f"【已有摘要】\n{previous_summary}\n\n"
            f"【需要合并进来的新对话】\n{transcript}\n\n"
            "请把两者合并成一份新的摘要。"
        )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
        )
        text = (response.choices[0].message.content or "").strip()
        return text or (previous_summary or "")

    except Exception:
        # 摘要失败只是"记得少一点"，不是致命错误
        logger.exception("生成对话摘要失败，沿用旧摘要")
        return previous_summary or ""
