"""使用项目 MySQL 数据库保存用户和 JWT 会话登记。"""

from datetime import datetime, timezone
import os
from threading import Lock

from dotenv import load_dotenv

load_dotenv()


class DuplicateUsernameError(Exception):
    """用户名已存在。"""


class AuthRepository:
    def __init__(self):
        self._initialized = False
        self._init_lock = Lock()

    def _connect(self):
        import pymysql

        return pymysql.connect(
            host=os.getenv("MYSQL_HOST", "127.0.0.1"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DATABASE", "agent_db"),
            charset="utf8mb4",
            connect_timeout=5,
            cursorclass=pymysql.cursors.DictCursor,
        )

    def _ensure_schema(self, connection):
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            with connection.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS auth_users (
                        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                        username VARCHAR(64) NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE KEY uq_auth_users_username (username)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS auth_tokens (
                        jti CHAR(32) NOT NULL PRIMARY KEY,
                        user_id BIGINT UNSIGNED NOT NULL,
                        expires_at DATETIME NOT NULL,
                        created_at DATETIME NOT NULL,
                        revoked_at DATETIME NULL,
                        KEY idx_auth_tokens_user_expiry (user_id, expires_at),
                        CONSTRAINT fk_auth_tokens_user
                            FOREIGN KEY (user_id) REFERENCES auth_users(id)
                            ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
            connection.commit()
            self._initialized = True

    def _open(self):
        connection = self._connect()
        try:
            self._ensure_schema(connection)
        except Exception:
            connection.close()
            raise
        return connection

    def create_user(self, username: str, password_hash: str) -> int:
        connection = self._open()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO auth_users (username, password_hash) VALUES (%s, %s)",
                    (username, password_hash),
                )
                user_id = cursor.lastrowid
            connection.commit()
            return int(user_id)
        except Exception as error:
            connection.rollback()
            if getattr(error, "args", (None,))[0] == 1062:
                raise DuplicateUsernameError from error
            raise
        finally:
            connection.close()

    def find_user(self, username: str) -> dict | None:
        connection = self._open()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT id, username, password_hash FROM auth_users WHERE username = %s",
                    (username,),
                )
                return cursor.fetchone()
        finally:
            connection.close()

    def create_token(self, jti: str, user_id: int, expires_at: datetime) -> None:
        connection = self._open()
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        expiry_utc = expires_at.astimezone(timezone.utc).replace(tzinfo=None)
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """DELETE FROM auth_tokens
                       WHERE expires_at <= UTC_TIMESTAMP()
                          OR revoked_at < DATE_SUB(UTC_TIMESTAMP(), INTERVAL 1 DAY)"""
                )
                cursor.execute(
                    """INSERT INTO auth_tokens
                       (jti, user_id, expires_at, created_at)
                       VALUES (%s, %s, %s, %s)""",
                    (jti, user_id, expiry_utc, now),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def is_token_active(self, jti: str, username: str) -> bool:
        connection = self._open()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT 1
                       FROM auth_tokens AS token
                       JOIN auth_users AS account ON account.id = token.user_id
                       WHERE token.jti = %s
                         AND account.username = %s
                         AND token.revoked_at IS NULL
                         AND token.expires_at > UTC_TIMESTAMP()""",
                    (jti, username),
                )
                return cursor.fetchone() is not None
        finally:
            connection.close()

    def revoke_token(self, jti: str, username: str) -> None:
        connection = self._open()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """UPDATE auth_tokens AS token
                       JOIN auth_users AS account ON account.id = token.user_id
                       SET token.revoked_at = UTC_TIMESTAMP()
                       WHERE token.jti = %s AND account.username = %s
                         AND token.revoked_at IS NULL""",
                    (jti, username),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


auth_repository = AuthRepository()
