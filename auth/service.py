"""用户相关的业务逻辑，只跟数据库打交道，不碰 HTTP。

路由层拿到的异常是业务异常（用户名重复 / 密码错），翻译成状态码是路由层的事，
这样同一份逻辑既能在 API 里用，也能在脚本或测试里直接用。
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from auth.passwords import hash_password, verify_password
from auth.schemas import RegisterRequest
from auth.tokens import create_access_token
from database.models import User


class UsernameTaken(Exception):
    pass


class InvalidCredentials(Exception):
    pass


class InactiveUser(Exception):
    pass


async def get_user_by_id(db: AsyncSession, user_id: int) -> User | None:
    return await db.get(User, user_id)


async def get_user_by_username(db: AsyncSession, username: str) -> User | None:
    result = await db.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()


async def register_user(db: AsyncSession, data: RegisterRequest) -> User:
    if await get_user_by_username(db, data.username) is not None:
        raise UsernameTaken(data.username)

    user = User(
        username=data.username,
        email=data.email,
        password_hash=hash_password(data.password),
    )
    db.add(user)

    try:
        await db.commit()
    except IntegrityError as exc:
        # 并发注册同一个名字：唯一索引兜底，不能只靠上面的查询
        await db.rollback()
        raise UsernameTaken(data.username) from exc

    await db.refresh(user)
    return user


async def authenticate_user(
    db: AsyncSession, username: str, password: str
) -> User:
    user = await get_user_by_username(db, username)

    # 用户不存在时也要走一次哈希校验，避免用响应时间探测哪些用户名存在
    stored = user.password_hash if user else (
        "pbkdf2_sha256$260000$AAAAAAAAAAAAAAAAAAAAAA==$"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
    )

    if not verify_password(password, stored) or user is None:
        raise InvalidCredentials

    if not user.is_active:
        raise InactiveUser

    return user


def issue_token(user: User) -> tuple[str, int]:
    return create_access_token(
        subject=str(user.id),
        extra_claims={"username": user.username},
    )
