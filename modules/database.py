"""SQLAlchemy 异步引擎、独立 Session 与事务管理。"""
from contextlib import asynccontextmanager
import asyncio
import os
from typing import AsyncIterator

from dotenv import load_dotenv
from sqlalchemy import URL, make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from modules.models import Base

load_dotenv()
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_initialized = False
_initialization_lock = asyncio.Lock()


# 根据配置构造异步驱动URL，兼容原有同步MySQL或SQLite连接串。
def database_url():
    configured = os.getenv("DATABASE_URL")
    url = make_url(configured) if configured else URL.create(
        "mysql+aiomysql", username=os.getenv("MYSQL_USER", "root"), password=os.getenv("MYSQL_PASSWORD"),
        host=os.getenv("MYSQL_HOST", "127.0.0.1"), port=int(os.getenv("MYSQL_PORT", "3306")),
        database=os.getenv("MYSQL_DATABASE", "agent_db"), query={"charset": "utf8mb4"})
    if url.drivername in {"mysql", "mysql+pymysql"}:
        url = url.set(drivername="mysql+aiomysql")
    elif url.drivername in {"sqlite", "sqlite+pysqlite"}:
        url = url.set(drivername="sqlite+aiosqlite")
    if url.drivername not in {"mysql+aiomysql", "sqlite+aiosqlite"}:
        raise ValueError("DATABASE_URL 必须使用 mysql+aiomysql 或 sqlite+aiosqlite")
    return url


# 惰性创建异步引擎，不在导入时连接数据库。
def get_engine() -> AsyncEngine:
    global _engine, _session_factory
    if _engine is None:
        url = database_url()
        _engine = create_async_engine(url, pool_pre_ping=True,
            connect_args={"connect_timeout": 5} if url.get_backend_name() == "mysql" else {})
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


# 在事务连接上创建缺失表，不自动迁移已有结构。
async def initialize_database() -> AsyncEngine:
    global _initialized
    engine = get_engine()
    async with _initialization_lock:
        if not _initialized:
            async with engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)
            _initialized = True
    return engine


# 每次调用创建独立Session，自动提交、异常及取消回滚并关闭。
@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    await initialize_database()
    assert _session_factory is not None
    async with _session_factory() as db:
        async with db.begin():
            yield db


# 应用退出时释放连接池并重置初始化状态。
async def dispose_database():
    global _engine, _session_factory, _initialized, _initialization_lock
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
    _initialized = False
    _initialization_lock = asyncio.Lock()
