
import json
import hashlib
import os
from threading import Lock

from dotenv import load_dotenv


load_dotenv()


class SessionManager:
    def __init__(self):
        self.db_config = {
            "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
            "port": int(os.getenv("MYSQL_PORT", "3306")),
            "user": os.getenv("MYSQL_USER", "root"),
            "password": os.getenv("MYSQL_PASSWORD"),
            "database": os.getenv("MYSQL_DATABASE", "agent_db"),
            "charset": "utf8mb4",
            "connect_timeout": 5,
        }
        self._initialized = False
        self._init_lock = Lock()

    def _connect(self):
        # 只在第一次实际访问会话数据时加载驱动并连接数据库。
        import pymysql

        conn = pymysql.connect(**self.db_config)

        try:
            if not self._initialized:
                with self._init_lock:
                    if not self._initialized:
                        with conn.cursor() as cursor:
                            cursor.execute("""
                                CREATE TABLE IF NOT EXISTS sessions (
                                    session_id VARCHAR(100) PRIMARY KEY,
                                    messages JSON NOT NULL,
                                    owner_user_id BIGINT UNSIGNED NULL,
                                    title VARCHAR(120) NOT NULL DEFAULT '新对话',
                                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                                        ON UPDATE CURRENT_TIMESTAMP,
                                    KEY idx_sessions_owner_updated (owner_user_id, updated_at),
                                    CONSTRAINT fk_sessions_owner
                                        FOREIGN KEY (owner_user_id) REFERENCES auth_users(id)
                                        ON DELETE CASCADE
                                )
                            """)
                            try:
                                cursor.execute(
                                    "ALTER TABLE sessions ADD COLUMN owner_user_id BIGINT UNSIGNED NULL"
                                )
                            except Exception as error:
                                if getattr(error, "args", (None,))[0] != 1060:
                                    raise

                            try:
                                cursor.execute(
                                    "ALTER TABLE sessions ADD COLUMN title VARCHAR(120) NOT NULL DEFAULT '新对话'"
                                )
                            except Exception as error:
                                if getattr(error, "args", (None,))[0] != 1060:
                                    raise

                            try:
                                cursor.execute(
                                    "CREATE INDEX idx_sessions_owner_updated ON sessions (owner_user_id, updated_at)"
                                )
                            except Exception as error:
                                if getattr(error, "args", (None,))[0] != 1061:
                                    raise
                            cursor.execute(
                                "SELECT id, username FROM auth_users"
                            )
                            for user in cursor.fetchall():
                                owner_prefix = hashlib.sha256(
                                    user[1].encode("utf-8")
                                ).hexdigest()[:24] + ":%"
                                cursor.execute(
                                    "UPDATE sessions SET owner_user_id = %s "
                                    "WHERE owner_user_id IS NULL AND session_id LIKE %s",
                                    (user[0], owner_prefix),
                                )
                            try:
                                cursor.execute(
                                    "ALTER TABLE sessions ADD CONSTRAINT fk_sessions_owner "
                                    "FOREIGN KEY (owner_user_id) REFERENCES auth_users(id) ON DELETE CASCADE"
                                )
                            except Exception as error:
                                if getattr(error, "args", (None,))[0] != 1826:
                                    raise
                        conn.commit()
                        self._initialized = True
        except Exception:
            conn.close()
            raise

        return conn

    def get_messages(self, session_id):
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT messages FROM sessions WHERE session_id = %s",
                    (session_id,)
                )
                row = cursor.fetchone()

        if row is None:
            return []

        messages = row[0]

        # 兼容驱动返回 JSON 字符串或已解析的 Python 对象
        if isinstance(messages, str):
            return json.loads(messages)

        return messages

    def exists(self, session_id: str) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM sessions WHERE session_id = %s",
                    (session_id,)
                )
                row = cursor.fetchone()
        return row is not None

    def save(self, session_id, messages):
        messages_json = json.dumps(
            messages,
            ensure_ascii=False
        )

        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO sessions (session_id, messages)
                    VALUES (%s, %s)
                    ON DUPLICATE KEY UPDATE
                        messages = VALUES(messages)
                    """,
                    (session_id, messages_json)
                )
            conn.commit()

    def clear(self, session_id: str):
        """清空聊天记录，但保留会话"""
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE sessions SET messages = %s WHERE session_id = %s",
                    (json.dumps([], ensure_ascii=False), session_id)
                )
            conn.commit()

    def delete(self, session_id: str):
        """删除整个会话"""
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM sessions WHERE session_id = %s",
                    (session_id,)
                )
            conn.commit()

    def create(
        self,
        session_id: str,
        owner_user_id: int | None = None,
        title: str = "新对话",
    ):
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO sessions (session_id, messages, owner_user_id, title)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (session_id, json.dumps([], ensure_ascii=False), owner_user_id, title)
                )
            conn.commit()

    def rename(self, session_id: str, owner_user_id: int, title: str) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM sessions WHERE session_id = %s AND owner_user_id = %s",
                    (session_id, owner_user_id),
                )
                if cursor.fetchone() is None:
                    return False
                cursor.execute(
                    "UPDATE sessions SET title = %s "
                    "WHERE session_id = %s AND owner_user_id = %s",
                    (title, session_id, owner_user_id),
                )
                updated = cursor.rowcount > 0
            conn.commit()
        return updated

    def list_sessions(self, owner_user_id: int):
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT session_id, title, updated_at
                    FROM sessions
                    WHERE owner_user_id = %s
                    ORDER BY updated_at DESC
                    """,
                    (owner_user_id,),
                )
                rows = cursor.fetchall()

        return [
            {
                "session_id": row[0],
                "title": row[1],
                "updated_at": row[2].isoformat() if row[2] else None,
            }
            for row in rows
        ]




