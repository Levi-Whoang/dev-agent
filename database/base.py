"""异步数据库连接：engine / Session 工厂 / FastAPI 依赖。

和原来 pymysql 的区别：
- `with conn.cursor()` 每次手动开关连接 → 现在由连接池管理，请求结束自动归还；
- 手动 commit → 现在 `async with get_db()` 里异常自动回滚；
- 线程阻塞 → `await` 期间事件循环可以处理别的请求。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from config import get_settings


class Base(DeclarativeBase):
    """所有 ORM 模型的基类，建表时靠它收集元数据。"""


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """延迟创建 engine。

    放在函数里而不是模块级变量，测试里换 DATABASE_URL / 覆盖依赖时不会被
    模块导入时的旧配置锁死。
    """
    global _engine
    if _engine is None:
        settings = get_settings()
        connect_args = {}
        if settings.database_url.startswith("sqlite"):
            # SQLite 默认禁止跨线程使用连接，异步驱动需要放开
            connect_args["check_same_thread"] = False
        _engine = create_async_engine(
            settings.database_url,
            echo=settings.db_echo,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,  # commit 后还能读对象属性，避免多余查询
        )
    return _session_factory


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI 依赖：一个请求一个会话，出错回滚。"""
    async with get_session_factory()() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """建表（开发用）。

    生产环境应该换成 Alembic 迁移，这里只是为了 `uvicorn app:app` 能直接跑起来。
    """
    from database import models  # noqa: F401  确保模型已注册到 Base.metadata

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_db() -> None:
    """关闭连接池，应用退出时调用，避免资源泄漏。"""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
