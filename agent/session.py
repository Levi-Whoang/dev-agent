"""聊天会话数据访问层。"""

import json
from datetime import datetime

from sqlalchemy import select

from modules.database import session_scope
from modules.models import ChatSession


class SessionManager:
    """通过 SQLAlchemy 管理聊天历史和会话元数据。"""

    def get_messages(self, session_id: str) -> list[dict]:
        with session_scope() as db:
            record = db.get(ChatSession, session_id)
            if record is None:
                return []
            messages = record.messages

        # 兼容旧数据中以字符串形式保存的 JSON 内容。
        return json.loads(messages) if isinstance(messages, str) else messages

    def exists(self, session_id: str) -> bool:
        with session_scope() as db:
            return db.scalar(
                select(ChatSession.session_id).where(
                    ChatSession.session_id == session_id
                )
            ) is not None

    def save(self, session_id: str, messages: list[dict]) -> None:
        with session_scope() as db:
            record = db.get(ChatSession, session_id)
            if record is None:
                db.add(ChatSession(session_id=session_id, messages=messages))
            else:
                record.messages = messages

    def clear(self, session_id: str) -> None:
        """清空聊天记录，但保留会话及其标题。"""
        with session_scope() as db:
            record = db.get(ChatSession, session_id)
            if record is not None:
                record.messages = []

    def delete(self, session_id: str) -> None:
        """删除整个会话。"""
        with session_scope() as db:
            record = db.get(ChatSession, session_id)
            if record is not None:
                db.delete(record)

    def create(
        self,
        session_id: str,
        owner_user_id: int | None = None,
        title: str = "新对话",
    ) -> None:
        with session_scope() as db:
            db.add(
                ChatSession(
                    session_id=session_id,
                    messages=[],
                    owner_user_id=owner_user_id,
                    title=title,
                )
            )

    def rename(self, session_id: str, owner_user_id: int, title: str) -> bool:
        with session_scope() as db:
            record = db.scalar(
                select(ChatSession).where(
                    ChatSession.session_id == session_id,
                    ChatSession.owner_user_id == owner_user_id,
                )
            )
            if record is None:
                return False
            record.title = title
            return True

    def list_sessions(self, owner_user_id: int) -> list[dict]:
        with session_scope() as db:
            records = db.scalars(
                select(ChatSession)
                .where(ChatSession.owner_user_id == owner_user_id)
                .order_by(ChatSession.updated_at.desc())
            ).all()
            return [
                {
                    "session_id": record.session_id,
                    "title": record.title,
                    "updated_at": (
                        record.updated_at.isoformat()
                        if isinstance(record.updated_at, datetime)
                        else None
                    ),
                }
                for record in records
            ]
