
import json
import os

import pymysql
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

        # 启动时检查数据库连接，并确保会话表存在
        with self._connect() as conn:
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

    def _connect(self):
        return pymysql.connect(**self.db_config)

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


    def list_sessions(self):
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT session_id, updated_at
                    FROM sessions
                    ORDER BY updated_at DESC
                    """
                )
                rows = cursor.fetchall()

        return [
            {
                "session_id": row[0],
                "updated_at": row[1].isoformat() if row[1] else None,
            }
            for row in rows
        ]



        return row is not None
