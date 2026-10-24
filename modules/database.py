"""SQLAlchemy engine、Session 工厂和 ORM 表初始化。"""

from contextlib import contextmanager
import os
from threading import Lock
from typing import Iterator

from dotenv import load_dotenv
from sqlalchemy import Engine, URL, create_engine
from sqlalchemy.orm import Session, sessionmaker

from modules.models import Base

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


def initialize_database() -> Engine:
    """按 ORM 模型创建缺失表，不自动修改已有表结构。"""
    global _initialized
    engine = get_engine()
    if _initialized:
        return engine
    with _initialization_lock:
        if not _initialized:
            Base.metadata.create_all(engine)
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
