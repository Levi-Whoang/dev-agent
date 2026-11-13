"""聊天会话数据访问层。"""

import json
import logging
from datetime import datetime

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError

from modules.database import session_scope
from modules.models import ChatMessage, ChatSession, ChatSummary

logger = logging.getLogger(__name__)


# 把模型消息拆成 session_messages 的列；正文保持原样，非字符串时转成 JSON。
def _message_values(message: dict) -> dict:
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return {
        "role": str(message.get("role") or ""),
        "content": content,
        "tool_calls": message.get("tool_calls"),
        "tool_call_id": message.get("tool_call_id"),
        "name": message.get("name"),
    }


# 把数据库行还原成模型可用的消息结构，省略空的可选字段。
def _row_to_message(row: ChatMessage) -> dict:
    message = {"role": row.role}
    if row.content is not None:
        message["content"] = row.content
    if row.tool_calls is not None:
        message["tool_calls"] = row.tool_calls
    if row.tool_call_id is not None:
        message["tool_call_id"] = row.tool_call_id
    if row.name is not None:
        message["name"] = row.name
    return message


# 解析旧版 sessions.messages JSON，异常数据按空历史处理。
def _legacy_messages(raw) -> list[dict]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            logger.warning("legacy_messages_invalid")
            return []
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


class SessionManager:
    """通过 SQLAlchemy 管理聊天历史和会话元数据。"""

    async def get_messages(self, session_id: str) -> list[dict]:
        """按写入顺序读取独立消息表；旧 JSON 历史首次读取时自动迁移。"""
        async with session_scope() as db:
            rows = await self._select_messages(db, session_id)
            if rows:
                return [_row_to_message(row) for row in rows]
            migrated = await self._migrate_legacy(db, session_id)
            return migrated or []

    async def append_message(self, session_id: str, message: dict) -> None:
        """追加一条消息，不覆盖其他请求已经写入的内容。"""
        await self.append_messages(session_id, [message])

    async def append_messages(self, session_id: str, messages: list[dict]) -> None:
        """在一个事务里追加多条消息，用于工具调用等需要连续写入的场景。"""
        if not messages:
            return
        async with session_scope() as db:
            await self._ensure_session(db, session_id)
            await self._migrate_legacy(db, session_id)
            for message in messages:
                db.add(ChatMessage(session_id=session_id, **_message_values(message)))
            await self._touch(db, session_id)

    async def exists(self, session_id: str) -> bool:
        async with session_scope() as db:
            return (await db.scalar(
                select(ChatSession.session_id).where(
                    ChatSession.session_id == session_id
                )
            )) is not None

    async def save(self, session_id: str, messages: list[dict]) -> None:
        """整段覆盖历史，仅用于兼容旧调用与测试；Agent 不再使用。"""
        async with session_scope() as db:
            record = await db.get(ChatSession, session_id)
            if record is None:
                db.add(ChatSession(session_id=session_id, messages=[]))
                await db.flush()
            else:
                record.messages = []
            await db.execute(
                delete(ChatMessage).where(ChatMessage.session_id == session_id)
            )
            for message in messages:
                db.add(ChatMessage(session_id=session_id, **_message_values(message)))
            await self._touch(db, session_id)

    async def get_summary(self, session_id: str) -> dict | None:
        """读取已保存的历史摘要；没有摘要时返回 None。"""
        async with session_scope() as db:
            record = await db.get(ChatSummary, session_id)
            if record is None:
                return None
            return {
                "content": record.content or "",
                "covered_messages": int(record.covered_messages or 0),
            }

    async def save_summary(
        self, session_id: str, content: str, covered_messages: int
    ) -> None:
        """写入或更新历史摘要。"""
        async with session_scope() as db:
            if await db.get(ChatSession, session_id) is None:
                db.add(ChatSession(session_id=session_id, messages=[]))
                await db.flush()
            record = await db.get(ChatSummary, session_id)
            if record is None:
                db.add(
                    ChatSummary(
                        session_id=session_id,
                        content=content,
                        covered_messages=covered_messages,
                    )
                )
            else:
                record.content = content
                record.covered_messages = covered_messages

    async def clear_summary(self, session_id: str) -> None:
        """删除摘要，使后续请求从原始消息重新生成。"""
        async with session_scope() as db:
            record = await db.get(ChatSummary, session_id)
            if record is not None:
                await db.delete(record)

    async def clear(self, session_id: str) -> None:
        """清空聊天记录，但保留会话及其标题。"""
        async with session_scope() as db:
            record = await db.get(ChatSession, session_id)
            if record is not None:
                record.messages = []
            await db.execute(
                delete(ChatMessage).where(ChatMessage.session_id == session_id)
            )
            summary = await db.get(ChatSummary, session_id)
            if summary is not None:
                await db.delete(summary)

    async def delete(self, session_id: str) -> None:
        """删除整个会话及其消息、摘要。"""
        async with session_scope() as db:
            await db.execute(
                delete(ChatMessage).where(ChatMessage.session_id == session_id)
            )
            summary = await db.get(ChatSummary, session_id)
            if summary is not None:
                await db.delete(summary)
            record = await db.get(ChatSession, session_id)
            if record is not None:
                await db.delete(record)

    async def create(
        self,
        session_id: str,
        owner_user_id: int | None = None,
        title: str = "新对话",
    ) -> None:
        async with session_scope() as db:
            db.add(
                ChatSession(
                    session_id=session_id,
                    messages=[],
                    owner_user_id=owner_user_id,
                    title=title,
                )
            )

    async def rename(self, session_id: str, owner_user_id: int, title: str) -> bool:
        async with session_scope() as db:
            record = await db.scalar(
                select(ChatSession).where(
                    ChatSession.session_id == session_id,
                    ChatSession.owner_user_id == owner_user_id,
                )
            )
            if record is None:
                return False
            record.title = title
            return True

    async def list_sessions(self, owner_user_id: int) -> list[dict]:
        async with session_scope() as db:
            records = (await db.scalars(
                select(ChatSession)
                .where(ChatSession.owner_user_id == owner_user_id)
                .order_by(ChatSession.updated_at.desc())
            )).all()
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

    @staticmethod
    async def _select_messages(db, session_id: str) -> list[ChatMessage]:
        return (await db.scalars(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id)
        )).all()

    # 会话不存在时创建；并发下唯一键冲突说明其他请求已创建，忽略即可。
    @staticmethod
    async def _ensure_session(db, session_id: str) -> None:
        if await db.get(ChatSession, session_id) is not None:
            return
        try:
            async with db.begin_nested():
                db.add(ChatSession(session_id=session_id, messages=[]))
                await db.flush()
        except IntegrityError:
            pass

    # 迁移旧版整段 JSON 历史：锁会话行并复查，保证只迁移一次。
    @staticmethod
    async def _migrate_legacy(db, session_id: str) -> list[dict] | None:
        raw = await db.scalar(
            select(ChatSession.messages)
            .where(ChatSession.session_id == session_id)
            .with_for_update()
        )
        if raw is None:
            return None
        legacy = _legacy_messages(raw)
        existing = (await db.scalars(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id)
            .with_for_update()
        )).all()
        if existing:
            # 新表已有权威数据，清掉遗留 JSON 防止下次重复迁移。
            if legacy:
                await db.execute(
                    update(ChatSession)
                    .where(ChatSession.session_id == session_id)
                    .values(messages=[])
                )
            return [_row_to_message(row) for row in existing]
        if not legacy:
            return None
        for message in legacy:
            db.add(ChatMessage(session_id=session_id, **_message_values(message)))
        await db.execute(
            update(ChatSession)
            .where(ChatSession.session_id == session_id)
            .values(messages=[])
        )
        return legacy

    # 追加消息后刷新会话活跃时间，供会话列表排序。
    @staticmethod
    async def _touch(db, session_id: str) -> None:
        await db.execute(
            update(ChatSession)
            .where(ChatSession.session_id == session_id)
            .values(updated_at=func.current_timestamp())
        )
