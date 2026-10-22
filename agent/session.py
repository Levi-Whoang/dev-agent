
import json
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
                                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                                        ON UPDATE CURRENT_TIMESTAMP
                                )
                            """)
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

    def create(self, session_id: str):
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO sessions (session_id, messages)
                    VALUES (%s, %s)
                    """,
                    (session_id, json.dumps([], ensure_ascii=False))
                )
            conn.commit()


    def list_sessions(self, prefix: str):
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT session_id, updated_at
                    FROM sessions
                    WHERE session_id LIKE %s
                    ORDER BY updated_at DESC
                    """,
                    (f"{prefix}%",),
                )
                rows = cursor.fetchall()

        return [
            {
                "session_id": row[0],
                "updated_at": row[1].isoformat() if row[1] else None,
            }
            for row in rows
        ]




