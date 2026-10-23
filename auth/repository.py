"""用户和 JWT 登记的数据访问层。"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError

from modules.database import session_scope
from modules.models import AuthToken, AuthUser


class DuplicateUsernameError(Exception):
    """用户名已存在。"""


class AuthRepository:
    """封装用户与令牌操作，业务逻辑不直接依赖数据库连接细节。"""

    def create_user(self, username: str, password_hash: str) -> int:
        try:
            with session_scope() as db:
                user = AuthUser(username=username, password_hash=password_hash)
                db.add(user)
                db.flush()
                return int(user.id)
        except IntegrityError as error:
            original_message = str(getattr(error, "orig", error)).lower()
            if "username" in original_message or "1062" in original_message:
                raise DuplicateUsernameError from error
            raise

    def find_user(self, username: str) -> dict | None:
        with session_scope() as db:
            user = db.scalar(
                select(AuthUser).where(AuthUser.username == username)
            )
            if user is None:
                return None
            return {
                "id": user.id,
                "username": user.username,
                "password_hash": user.password_hash,
            }

    def create_token(self, jti: str, user_id: int, expires_at: datetime) -> None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        expiry_utc = expires_at.astimezone(timezone.utc).replace(tzinfo=None)
        with session_scope() as db:
            db.execute(
                delete(AuthToken).where(
                    (AuthToken.expires_at <= now)
                    | (AuthToken.revoked_at < now - timedelta(days=1))
                )
            )
            db.add(
                AuthToken(
                    jti=jti,
                    user_id=user_id,
                    expires_at=expiry_utc,
                    created_at=now,
                )
            )

    def is_token_active(self, jti: str, username: str) -> bool:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with session_scope() as db:
            return db.scalar(
                select(AuthToken.jti)
                .join(AuthUser, AuthUser.id == AuthToken.user_id)
                .where(
                    AuthToken.jti == jti,
                    AuthUser.username == username,
                    AuthToken.revoked_at.is_(None),
                    AuthToken.expires_at > now,
                )
            ) is not None

    def revoke_token(self, jti: str, username: str) -> None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with session_scope() as db:
            db.execute(
                update(AuthToken)
                .where(
                    AuthToken.jti == jti,
                    AuthToken.user_id
                    == select(AuthUser.id).where(AuthUser.username == username).scalar_subquery(),
                    AuthToken.revoked_at.is_(None),
                )
                .values(revoked_at=now)
            )


auth_repository = AuthRepository()
