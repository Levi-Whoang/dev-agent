"""SQLAlchemy engine、会话工厂和兼容旧表的初始化逻辑。"""

from contextlib import contextmanager
import hashlib
import os
from threading import Lock
from typing import Iterator

from dotenv import load_dotenv
from sqlalchemy import Engine, URL, create_engine, inspect, update
from sqlalchemy.orm import Session, sessionmaker

from modules.models import AuthUser, Base, ChatSession

load_dotenv()

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None
_initialized = False
_engine_lock = Lock()
_initialization_lock = Lock()


def get_engine() -> Engine:
    """按环境配置惰性创建数据库引擎。"""
    global _engine, _session_factory
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                database_url = os.getenv("DATABASE_URL")
                if database_url:
                    url = database_url
                else:
                    url = URL.create(
                        "mysql+pymysql",
                        username=os.getenv("MYSQL_USER", "root"),
                        password=os.getenv("MYSQL_PASSWORD"),
                        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
                        port=int(os.getenv("MYSQL_PORT", "3306")),
                        database=os.getenv("MYSQL_DATABASE", "agent_db"),
                        query={"charset": "utf8mb4"},
                    )
                _engine = create_engine(
                    url,
                    pool_pre_ping=True,
                    connect_args={"connect_timeout": 5}
                    if str(url).startswith("mysql")
                    else {},
                )
                _session_factory = sessionmaker(
                    bind=_engine, expire_on_commit=False
                )
    return _engine


def _migrate_legacy_sessions(engine: Engine) -> None:
    """为已有 sessions 表补充所有者/标题字段并回填可识别的旧会话。"""
    inspector = inspect(engine)
    if "sessions" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("sessions")}
    owner_column_type = (
        "BIGINT UNSIGNED" if engine.dialect.name == "mysql" else "INTEGER"
    )
    with engine.begin() as connection:
        if "owner_user_id" not in columns:
            connection.exec_driver_sql(
                "ALTER TABLE sessions ADD COLUMN owner_user_id "
                f"{owner_column_type} NULL"
            )
        if "title" not in columns:
            connection.exec_driver_sql(
                "ALTER TABLE sessions ADD COLUMN title VARCHAR(120) NOT NULL DEFAULT '新对话'"
            )

    inspector = inspect(engine)
    indexes = {index["name"] for index in inspector.get_indexes("sessions")}
    if "idx_sessions_owner_updated" not in indexes:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE INDEX idx_sessions_owner_updated "
                "ON sessions (owner_user_id, updated_at)"
            )

    # 旧会话 ID 的前缀为用户名 SHA-256 摘要前 24 位，只有匹配到已有账号时才回填。
    with _new_session(engine) as db:
        users = db.query(AuthUser.id, AuthUser.username).all()
        for user_id, username in users:
            prefix = hashlib.sha256(username.encode("utf-8")).hexdigest()[:24] + ":%"
            db.execute(
                update(ChatSession)
                .where(
                    ChatSession.owner_user_id.is_(None),
                    ChatSession.session_id.like(prefix),
                )
                .values(owner_user_id=user_id)
            )

    # ORM 建新表时会创建外键。这里为旧表补外键；非 MySQL 方言由其迁移工具管理。
    if engine.dialect.name == "mysql":
        foreign_keys = {
            foreign_key.get("name")
            for foreign_key in inspect(engine).get_foreign_keys("sessions")
        }
        if "fk_sessions_owner" not in foreign_keys:
            with engine.begin() as connection:
                connection.exec_driver_sql(
                    "ALTER TABLE sessions ADD CONSTRAINT fk_sessions_owner "
                    "FOREIGN KEY (owner_user_id) REFERENCES auth_users(id) ON DELETE CASCADE"
                )


@contextmanager
def _new_session(engine: Engine) -> Iterator[Session]:
    """初始化迁移期间使用的短生命周期 ORM Session。"""
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    db = factory()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def initialize_database() -> Engine:
    """创建缺失表，并为历史 sessions 表执行向前兼容迁移。"""
    global _initialized
    engine = get_engine()
    if _initialized:
        return engine
    with _initialization_lock:
        if not _initialized:
            Base.metadata.create_all(engine)
            _migrate_legacy_sessions(engine)
            _initialized = True
    return engine


@contextmanager
def session_scope() -> Iterator[Session]:
    """提供带自动提交、回滚和关闭的 ORM Session。"""
    global _session_factory
    engine = initialize_database()
    assert _session_factory is not None
    db = _session_factory(bind=engine)
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
