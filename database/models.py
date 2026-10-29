"""ORM 模型：用户、会话、消息。

为什么不再把整个 messages 数组塞进一个 JSON 字段：
- 数据库能按 role / 时间查询，也能分页，不用每次读写整段历史；
- 两条请求同时写同一会话时，是插入新行而不是整段覆盖，不会互相丢消息；
- 消息和用户通过外键绑起来，越权读别人的会话在 SQL 层就查不到。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    email: Mapped[str | None] = mapped_column(String(255), unique=True, default=None)
    password_hash: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    sessions: Mapped[list["ChatSession"]] = relationship(
        back_populates="owner",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    # 沿用原来的字符串 session_id，前端 / 旧接口都不用改
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(200), default="新对话")

    # ---------- 会话记忆 ----------
    # 滚动摘要：把已经挤出上下文窗口的早期对话压成一段话。
    # summary_upto_id 是"摘要已经覆盖到哪条消息"，用它避免重复摘要同一段内容。
    summary: Mapped[str | None] = mapped_column(Text, default=None)
    summary_upto_id: Mapped[int | None] = mapped_column(default=None)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        server_default=func.now(),
    )

    owner: Mapped[User] = relationship(back_populates="sessions")
    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ChatMessage.id",
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text, default="")
    # 存 tool_calls / tool_call_id 这类 OpenAI 协议字段
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    session: Mapped[ChatSession] = relationship(back_populates="messages")

    __table_args__ = (Index("ix_chat_messages_session_order", "session_id", "id"),)

    def to_llm_message(self) -> dict[str, Any]:
        """还原成可以直接丢给 OpenAI SDK 的 message 字典。"""
        message: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.payload:
            message.update(self.payload)
        # assistant 只调工具时 content 必须是 None，不能是空字符串
        if message["role"] == "assistant" and message.get("tool_calls"):
            message["content"] = message["content"] or None
        return message


class UserMemory(Base):
    """跨会话的长期记忆：关于"这个人"的事实，而不是某次对话的内容。

    为什么要和 chat_messages 分开：
    - 消息是流水，会越积越多，也随时可能被清空；
    - 而这些事实（"用户是后端开发"、"公司在南山科技园"）要活很久，
      而且要能跨会话注入到每次对话里。
    两者生命周期完全不同，混在一张表里迟早要么被误删，要么检索时互相干扰。

    user_id 上不建 (user_id, content) 唯一索引：同一句话在不同语境下
    可能表述不同（"我在深圳"vs"我常驻深圳"），靠 SQL 判重会漏；
    判重放在应用层做归一化比较，见 agent/memory/longterm.py。
    """

    __tablename__ = "user_memories"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    content: Mapped[str] = mapped_column(Text)
    # fact=客观事实  preference=偏好  profile=身份信息，目前只影响注入时的分组
    kind: Mapped[str] = mapped_column(String(20), default="fact")
    # 从哪个会话里学到的，方便追溯，也方便删会话时判断要不要留
    source_session_id: Mapped[str | None] = mapped_column(String(100), default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )

    __table_args__ = (Index("ix_user_memories_user", "user_id", "id"),)
