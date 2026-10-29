"""长期记忆：关于「这个人」的事实，跨会话保留。

和滚动摘要的区别（两者经常被混为一谈）：

    摘要   = 这段对话讲了什么   → 会话级，随对话推进不断重写
    长期记忆 = 这个人是谁、要什么 → 用户级，一条条累积，很少变

为什么不能只靠摘要：换个会话，摘要就没了。用户第一句话说过"我是后端开发，
只看 Python 相关的岗位"，开新会话后模型又会推荐一堆 Java 岗位。

存储上有个取舍：**不做向量检索，用字面重合度排序**。
理由是这个阶段记忆条数是几十条量级（一个人能被记住的事实本来就不多），
为几十条数据加载一个 embedding 模型，冷启动就要几百毫秒，
收益远小于成本。等真到几千条再换。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from agent.memory.tokens import estimate_tokens
from database.models import UserMemory

logger = logging.getLogger("dev-agent.memory.longterm")

# 注入到 system prompt 里的记忆块，上限（token 估算值）
BLOCK_BUDGET = 320

# 单条记忆的长度上限，防止模型把一整段对话塞进来
CONTENT_LIMIT = 120

# 只影响注入时的分组标题，不做强校验
KIND_LABEL = {
    "profile": "身份",
    "preference": "偏好",
    "fact": "事实",
}

_PUNCT = re.compile(r"[\s，。、；：！？,.;:!?\"'“”‘’()（）\[\]【】\-—_/]+")


def normalize(text: str) -> str:
    """判重用的归一化：去空白和标点、统一小写。"""
    return _PUNCT.sub("", text).lower()


def _bigrams(text: str) -> set[str]:
    """中文没有空格，二元组是最省事的"分词"方式；英文按词切。"""
    if re.search(r"[a-zA-Z]", text):
        words = {w for w in re.split(r"[^a-z0-9]+", text.lower()) if len(w) > 1}
    else:
        words = set()

    compact = normalize(text)
    for i in range(len(compact) - 1):
        words.add(compact[i : i + 2])
    return words


def _overlap_score(query: str, content: str) -> float:
    """query 与记忆的字面重合度，0~1。"""
    q = _bigrams(query)
    if not q:
        return 0.0
    c = _bigrams(content)
    if not c:
        return 0.0
    return len(q & c) / len(q)


async def remember(
    db: AsyncSession,
    user_id: int,
    content: str,
    kind: str = "fact",
    source_session_id: str | None = None,
) -> tuple[UserMemory, bool]:
    """记一条。返回 (对象, 是否新增)。

    判重规则：归一化后完全相同，或一方包含另一方，就认为是同一条，
    更新内容而不是新增一条。不这么做的话，"我在深圳"会被记成十几条。
    """
    content = content.strip()[:CONTENT_LIMIT]
    if not content:
        raise ValueError("记忆内容不能为空")

    stmt = select(UserMemory).where(UserMemory.user_id == user_id)
    existing = (await db.execute(stmt)).scalars().all()

    key = normalize(content)
    for item in existing:
        old = normalize(item.content)
        if old == key or (len(key) >= 4 and key in old) or (len(old) >= 4 and old in key):
            item.content = content
            item.kind = kind or item.kind
            await db.commit()
            await db.refresh(item)
            return item, False

    item = UserMemory(
        user_id=user_id,
        content=content,
        kind=kind or "fact",
        source_session_id=source_session_id,
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)
    logger.info("记住了一条用户信息（%s）：%s", kind, content)
    return item, True


async def list_all(db: AsyncSession, user_id: int) -> list[UserMemory]:
    stmt = (
        select(UserMemory)
        .where(UserMemory.user_id == user_id)
        .order_by(UserMemory.id.desc())
    )
    return list((await db.execute(stmt)).scalars().all())


async def forget(db: AsyncSession, user_id: int, keyword: str) -> int:
    """按关键词删记忆，返回删除条数。"""
    key = normalize(keyword)
    if not key:
        return 0

    removed = 0
    for item in await list_all(db, user_id):
        if key in normalize(item.content):
            await db.execute(delete(UserMemory).where(UserMemory.id == item.id))
            removed += 1
    if removed:
        await db.commit()
    return removed


def select_relevant(
    memories: list[UserMemory], query: str, limit: int = 12
) -> list[UserMemory]:
    """按相关度挑几条，并且总量不超预算。

    相关度为 0 的也会保留（按新到旧补位）—— 因为"用户叫 hewier"
    这种身份信息跟当前问题往往没有任何字面重合，但它永远该在场。
    """
    if not memories:
        return []

    scored = sorted(
        memories,
        key=lambda m: (_overlap_score(query, m.content), m.id),
        reverse=True,
    )

    picked: list[UserMemory] = []
    used = 0
    for item in scored:
        if len(picked) >= limit:
            break
        cost = estimate_tokens(item.content) + 4
        if picked and used + cost > BLOCK_BUDGET:
            break
        picked.append(item)
        used += cost

    # 按 id 还原时间顺序，注入时读起来更自然
    picked.sort(key=lambda m: m.id)
    return picked


def render_block(memories: list[UserMemory]) -> str:
    """渲染成可以直接拼进 system prompt 的一段文字。"""
    if not memories:
        return ""

    lines = ["以下是关于当前用户的已知信息，回答时请据此调整（不必逐条复述）："]
    for item in memories:
        label = KIND_LABEL.get(item.kind, "事实")
        lines.append(f"- [{label}] {item.content}")

    return "\n".join(lines)


def to_dict(item: UserMemory) -> dict[str, Any]:
    return {
        "id": item.id,
        "content": item.content,
        "kind": item.kind,
        "created_at": item.created_at.isoformat() if item.created_at else None,
    }
