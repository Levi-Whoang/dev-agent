"""用户和 JWT 登记的数据访问层。"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError

from modules.database import session_scope
from modules.models import AuthAuditEvent, AuthLoginFailure, AuthToken, AuthUser


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

    def user_exists(self, username: str) -> bool:
        with session_scope() as db:
            return db.scalar(select(AuthUser.id).where(AuthUser.username == username)) is not None

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

    def create_tokens(self, tokens: list[dict]) -> None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with session_scope() as db:
            db.execute(
                delete(AuthToken).where(
                    (AuthToken.expires_at <= now)
                    | (AuthToken.revoked_at < now - timedelta(days=1))
                )
            )
            db.add_all([
                AuthToken(
                    jti=item["jti"],
                    user_id=item["user_id"],
                    expires_at=item["expires_at"].astimezone(timezone.utc).replace(tzinfo=None),
                    created_at=now,
                )
                for item in tokens
            ])

    def rotate_tokens(self, old_refresh_jti: str, old_access_jti: str, username: str, tokens: list[dict]) -> bool:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with session_scope() as db:
            active = db.scalar(
                select(AuthToken.jti)
                .join(AuthUser, AuthUser.id == AuthToken.user_id)
                .where(
                    AuthToken.jti == old_refresh_jti,
                    AuthUser.username == username,
                    AuthToken.revoked_at.is_(None),
                    AuthToken.expires_at > now,
                )
                .with_for_update()
            )
            if active is None:
                return False
            db.execute(
                update(AuthToken)
                .where(AuthToken.jti.in_([old_refresh_jti, old_access_jti]))
                .values(revoked_at=now)
            )
            db.add_all([
                AuthToken(
                    jti=item["jti"], user_id=item["user_id"],
                    expires_at=item["expires_at"].astimezone(timezone.utc).replace(tzinfo=None),
                    created_at=now,
                )
                for item in tokens
            ])
        return True

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

    def revoke_token(self, jti: str, username: str, related_jti: str | None = None) -> None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with session_scope() as db:
            db.execute(
                update(AuthToken)
                .where(
                    AuthToken.jti.in_([jti, related_jti]) if related_jti else AuthToken.jti == jti,
                    AuthToken.user_id
                    == select(AuthUser.id).where(AuthUser.username == username).scalar_subquery(),
                    AuthToken.revoked_at.is_(None),
                )
                .values(revoked_at=now)
            )

    def login_failure_count(self, identity_hash: str, since: datetime) -> int:
        with session_scope() as db:
            db.execute(
                delete(AuthLoginFailure).where(
                    AuthLoginFailure.attempted_at < since - timedelta(days=1)
                )
            )
            return int(db.scalar(
                select(func.count(AuthLoginFailure.id)).where(
                    AuthLoginFailure.identity_hash == identity_hash,
                    AuthLoginFailure.attempted_at >= since,
                )
            ) or 0)

    def record_login_failure(self, identity_hash: str, attempted_at: datetime) -> None:
        with session_scope() as db:
            db.add(AuthLoginFailure(identity_hash=identity_hash, attempted_at=attempted_at))

    def clear_login_failures(self, identity_hash: str) -> None:
        with session_scope() as db:
            db.execute(delete(AuthLoginFailure).where(AuthLoginFailure.identity_hash == identity_hash))

    def audit(self, event_type: str, username: str | None, source_hash: str | None, succeeded: bool) -> None:
        with session_scope() as db:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            db.execute(
                delete(AuthAuditEvent).where(
                    AuthAuditEvent.created_at < now - timedelta(days=180)
                )
            )
            db.add(AuthAuditEvent(
                event_type=event_type,
                username=username[:64] if username else None,
                source_hash=source_hash,
                succeeded=succeeded,
                created_at=now,
            ))


auth_repository = AuthRepository()
