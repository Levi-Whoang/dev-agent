"""异步数据库连接：engine / Session 工厂 / FastAPI 依赖。

和原来 pymysql 的区别：
- `with conn.cursor()` 每次手动开关连接 → 现在由连接池管理，请求结束自动归还；
- 手动 commit → 现在 `async with get_db()` 里异常自动回滚；
- 线程阻塞 → `await` 期间事件循环可以处理别的请求。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from config import get_settings

logger = logging.getLogger("dev-agent.database")

# 老库需要补的列：(表名, 列名, 列定义)
# create_all() 只会建**新表**，对已存在的表不会加列 —— 所以升级过模型之后，
# 已有的 dev_agent.db 会缺字段，查询时会直接报 no such column。
# 项目没上 Alembic（对单人开发是过度设计），这里用一个幂等的补列逻辑兜住。
_ADDED_COLUMNS = (
    ("chat_sessions", "summary", "TEXT"),
    ("chat_sessions", "summary_upto_id", "INTEGER"),
)


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
    """建表 + 补列（开发用）。

    生产环境应该换成 Alembic 迁移，这里只是为了 `uvicorn app:app` 能直接跑起来。
    """
    from database import models  # noqa: F401  确保模型已注册到 Base.metadata

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_add_missing_columns)


def _add_missing_columns(conn) -> None:
    """给已存在的表补上模型里新增的列。幂等，重复跑不报错。"""
    inspector = inspect(conn)
    tables = {name for name, _, _ in _ADDED_COLUMNS}

    for table in tables:
        if not inspector.has_table(table):
            continue
        existing = {col["name"] for col in inspector.get_columns(table)}

        for target_table, column, ddl in _ADDED_COLUMNS:
            if target_table != table or column in existing:
                continue
            # 表名列名都来自上面的常量，不接受外部输入，不存在注入面
            conn.exec_driver_sql(
                f"ALTER TABLE {target_table} ADD COLUMN {column} {ddl}"
            )
            logger.info("数据库升级：%s 表补上 %s 列", target_table, column)


async def dispose_db() -> None:
    """关闭连接池，应用退出时调用，避免资源泄漏。"""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
