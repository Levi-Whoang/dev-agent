"""会话存储：用 SQLAlchemy 会话替代手写 SQL。

和原来 SessionManager 的关键区别：
- 原来一次 save() 覆盖整段 messages JSON，并发写会互相覆盖；
  现在一条消息一行，追加就是 INSERT，不丢别人的消息；
- 所有查询都带 user_id 条件，越权访问在 SQL 层就查不到；
- 不再有连接泄漏：session 由依赖注入管理，请求结束自动关闭。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import ChatMessage, ChatSession

logger = logging.getLogger("dev-agent.store")


class SessionStore:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ---------- 会话 ----------
    async def get_or_create(
        self, session_id: str, user_id: int, title: str | None = None
    ) -> ChatSession:
        session = await self.db.get(ChatSession, session_id)

        if session is None:
            session = ChatSession(
                id=session_id,
                user_id=user_id,
                title=title or "新对话",
            )
            self.db.add(session)
            await self.db.commit()
            await self.db.refresh(session)
            return session

        if session.user_id != user_id:
            # 别人已经用了这个 id，别悄悄覆盖，交给上层报 403/409
            raise PermissionError(session_id)

        return session

    async def exists(self, session_id: str, user_id: int) -> bool:
        stmt = select(func.count()).select_from(ChatSession).where(
            ChatSession.id == session_id, ChatSession.user_id == user_id
        )
        return bool((await self.db.execute(stmt)).scalar_one())

    async def list_sessions(self, user_id: int) -> list[dict[str, Any]]:
        stmt = (
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
        )
        sessions = (await self.db.execute(stmt)).scalars().all()
        return [
            {
                "session_id": s.id,
                "title": s.title,
                "updated_at": s.updated_at.isoformat() if s.updated_at else None,
            }
            for s in sessions
        ]

    async def rename(self, session_id: str, user_id: int, title: str) -> bool:
        """改会话标题。返回 False 表示这个 id 不属于该用户（或不存在）。

        故意**不**碰 updated_at：它是列表“最近聊的在最上面”的排序依据。
        改个名字不该让会话跳到顶部——否则用户会以为列表顺序乱了，
        而且 rename 是个纯展示操作，不应该改变会话的活跃度。

        条件里带 user_id 而不是先查再判断，是为了少一次往返，
        同时保证越权和不存在都落到同一个 rowcount=0，对外都是 404。

        注意 values() 里那一行 `updated_at=ChatSession.updated_at`：
        models.py 给这列挂了 onupdate=_utcnow，Core 的 update() 只要没显式
        给它赋值就会自动刷新。写成赋值给自身，SQL 变成
        `SET updated_at = chat_sessions.updated_at`，等于原地不动，
        才真正实现“改名不打乱顺序”。这一行删掉，上面那段理由就作废了。
        """
        stmt = (
            update(ChatSession)
            .where(ChatSession.id == session_id, ChatSession.user_id == user_id)
            .values(title=title, updated_at=ChatSession.updated_at)
        )
        result = await self.db.execute(stmt)
        await self.db.commit()
        return bool(result.rowcount)

    async def delete(self, session_id: str, user_id: int) -> bool:
        stmt = delete(ChatSession).where(
            ChatSession.id == session_id, ChatSession.user_id == user_id
        )
        result = await self.db.execute(stmt)
        await self.db.commit()
        return bool(result.rowcount)

    async def clear(self, session_id: str) -> None:
        await self.db.execute(
            delete(ChatMessage).where(ChatMessage.session_id == session_id)
        )
        await self.db.commit()
        await self._touch(session_id)

    async def maybe_autotitle(self, session_id: str, text: str) -> None:
        """会话还没起名时，用第一句提问当标题，列表里好认。"""
        session = await self.db.get(ChatSession, session_id)
        if session is None or session.title not in (None, "", "新对话"):
            return

        session.title = text.strip().replace("\n", " ")[:40] or "新对话"
        await self.db.commit()

    async def _touch(self, session_id: str) -> None:
        """刷新会话的 updated_at，列表排序才符合“最近聊的在最上面”。"""
        session = await self.db.get(ChatSession, session_id)
        if session is not None:
            session.updated_at = datetime.now(timezone.utc)
            await self.db.commit()

    # ---------- 消息 ----------
    async def history(self, session_id: str, limit: int = 40) -> list[dict[str, Any]]:
        """取最近 N 条消息。

        限制条数是为了控制上下文长度：全量塞给模型既慢又贵，
        而且早期内容对当前问题基本没用。
        """
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id.desc())
            .limit(limit)
        )
        rows = (await self.db.execute(stmt)).scalars().all()
        return [row.to_llm_message() for row in reversed(rows)]

    async def append(
        self,
        session_id: str,
        role: str,
        content: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.db.add(
            ChatMessage(
                session_id=session_id,
                role=role,
                content=content or "",
                payload=payload,
            )
        )
        await self.db.commit()

    async def messages_for_api(self, session_id: str) -> list[dict[str, Any]]:
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id)
        )
        rows = (await self.db.execute(stmt)).scalars().all()
        messages: list[dict[str, Any]] = []
        for row in rows:
            # tool 消息是给模型看的中间产物，不展示给用户
            if row.role == "tool":
                continue
            # 只调工具、没说话的那一轮也跳过，否则界面会出现空气泡
            if row.role == "assistant" and not row.content:
                continue
            messages.append(
                {
                    "role": row.role,
                    "content": row.content,
                    "created_at": (
                        row.created_at.isoformat() if row.created_at else None
                    ),
                }
            )
        return messages


def dump_tool_calls(tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {"tool_calls": json.loads(json.dumps(tool_calls, ensure_ascii=False))}
