"""项目数据库 ORM 模型。"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.mysql import BIGINT as MySQLBigInteger
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """所有 ORM 模型共用的元数据基类。"""


USER_ID_TYPE = MySQLBigInteger(unsigned=True).with_variant(Integer, "sqlite")


class AuthUser(Base):
    """注册用户及其密码哈希。"""

    __tablename__ = "auth_users"
    __table_args__ = (
        UniqueConstraint("username", name="uq_auth_users_username"),
    )

    id: Mapped[int] = mapped_column(USER_ID_TYPE, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )

    tokens: Mapped[list["AuthToken"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    sessions: Mapped[list["ChatSession"]] = relationship(back_populates="owner")


class AuthToken(Base):
    """JWT 登记信息；数据库不保存令牌原文。"""

    __tablename__ = "auth_tokens"
    __table_args__ = (
        Index("idx_auth_tokens_user_expiry", "user_id", "expires_at"),
    )

    jti: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        USER_ID_TYPE,
        ForeignKey("auth_users.id", name="fk_auth_tokens_user", ondelete="CASCADE"),
        nullable=False,
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    user: Mapped[AuthUser] = relationship(back_populates="tokens")


class AuthLoginFailure(Base):
    """用于按账号及客户端来源限制连续登录失败。"""

    __tablename__ = "auth_login_failures"
    __table_args__ = (Index("idx_auth_login_failures_attempted", "attempted_at"),)

    id: Mapped[int] = mapped_column(USER_ID_TYPE, primary_key=True, autoincrement=True)
    identity_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    attempted_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class AuthAuditEvent(Base):
    """记录认证结果，不保存密码、JWT 或原始 IP 地址。"""

    __tablename__ = "auth_audit_events"
    __table_args__ = (Index("idx_auth_audit_created", "created_at"),)

    id: Mapped[int] = mapped_column(USER_ID_TYPE, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(40), nullable=False)
    username: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    succeeded: Mapped[bool] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ChatSession(Base):
    """聊天历史、所有者及标题。"""

    __tablename__ = "sessions"
    __table_args__ = (
        Index("idx_sessions_owner_updated", "owner_user_id", "updated_at"),
    )

    session_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    messages: Mapped[list[dict]] = mapped_column(JSON, nullable=False, default=list)
    owner_user_id: Mapped[int | None] = mapped_column(
        USER_ID_TYPE,
        ForeignKey("auth_users.id", name="fk_sessions_owner", ondelete="CASCADE"),
        nullable=True,
    )
    title: Mapped[str] = mapped_column(
        String(120), nullable=False, default="新对话", server_default="新对话"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )

    owner: Mapped[AuthUser | None] = relationship(back_populates="sessions")
