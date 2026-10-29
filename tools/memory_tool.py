"""长期记忆工具：让 Agent 自己决定"这条信息值得记下来"。

为什么用闭包而不是普通函数：
工具的参数是**模型来填的**。如果把 `db` / `user_id` 写进签名，
它们会出现在 schema 里，模型有可能试图自己编一个 user_id 传进来 ——
这既是安全问题，也会让模型困惑。用闭包把上下文焊死，
模型只看到 `content` 和 `kind`，拿不到也改不了身份。

为什么不让 Agent 自动记忆（每轮都抽取一次）：
每轮多一次模型调用，成本和延迟都翻倍，而且抽取质量不稳定。
交给模型主动判断更省，也更符合"工具"这个抽象 ——
它知道自己刚听到的是一句寒暄还是一条可用于未来回答的事实。
"""

from __future__ import annotations

import logging
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("dev-agent.tools.memory")

# 单个用户最多存多少条，防止被模型刷爆
MAX_MEMORIES = 60


def make_remember_tool(
    db: AsyncSession, user_id: int, session_id: str | None = None
):
    """生成一个绑定了当前用户和数据库会话的 remember 工具。"""

    async def remember(
        content: str,
        kind: Literal["profile", "preference", "fact"] = "fact",
    ) -> dict:
        """记下一条关于用户的长期信息，跨会话长期有效。

        什么时候该用：
        - 用户介绍自己（职业、技术栈、所在城市、公司） → kind="profile"
        - 用户表达稳定的偏好或要求（"以后只推荐 Python 岗位"） → kind="preference"
        - 用户提供了知识库里查不到的、以后还会用到的事实 → kind="fact"

        什么时候不要用：
        - 寒暄、一次性的提问、临时的上下文
        - 知识库里本来就有的公司信息（那些走检索，不要重复记）
        - 用户的临时状态（"我现在很急"）

        content 写成一句完整、脱离上下文也能看懂的陈述句，
        例如「用户是后端开发，只关注 Python 和 FastAPI 相关岗位」。
        不要写成「他喜欢 Python」这种没有主语的句子。
        """
        # 延迟导入：agent.memory 会牵动 agent.store -> database，
        # 在模块顶层导入容易和 agent/__init__.py 形成循环
        from agent.memory import longterm

        content = (content or "").strip()
        if not content:
            return {"saved": False, "reason": "内容为空，没有记录"}

        try:
            existing = await longterm.list_all(db, user_id)
            if len(existing) >= MAX_MEMORIES:
                return {
                    "saved": False,
                    "reason": (
                        f"记忆条数已达上限 {MAX_MEMORIES} 条，"
                        "如需记录请先让用户删除一些旧记忆"
                    ),
                }

            item, created = await longterm.remember(
                db,
                user_id,
                content,
                kind=kind,
                source_session_id=session_id,
            )
        except Exception as exc:
            logger.exception("写入长期记忆失败")
            return {"saved": False, "reason": f"写入失败：{exc}"}

        return {
            "saved": True,
            "created": created,
            "content": item.content,
            "kind": item.kind,
            "note": "已记住，后续对话会自动带上这条信息"
            if created
            else "已有相似记忆，已更新为本次内容",
        }

    return remember
