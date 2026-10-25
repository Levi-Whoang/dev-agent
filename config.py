"""全局配置：所有环境变量只在这里读一次。

为什么单独一个文件：
- 之前每个模块各自 `load_dotenv()` + `os.getenv()`，换个变量名要全局搜索；
- 测试时想改数据库地址，只能去改业务代码。
现在业务代码只依赖 `get_settings()`，测试里覆盖它一个点就够了。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# 只加载项目根目录的 .env，不受启动时工作目录影响
load_dotenv(BASE_DIR / ".env")

logger = logging.getLogger("dev-agent.config")


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # ---------- 大模型 ----------
    llm_api_key: str
    llm_base_url: str | None
    llm_model: str
    llm_timeout: float

    # ---------- 数据库 ----------
    database_url: str
    db_echo: bool

    # ---------- 鉴权 ----------
    jwt_secret: str
    jwt_algorithm: str
    jwt_expire_minutes: int

    # ---------- 检索 ----------
    knowledge_dir: Path
    embedding_model: str
    top_k: int
    dense_threshold: float
    dense_floor: float
    rrf_k: int
    candidate_pool: int

    @property
    def is_default_jwt_secret(self) -> bool:
        return self.jwt_secret.startswith("dev-only-")


def _default_database_url() -> str:
    """默认用 SQLite：开箱即跑，不要求本机装 MySQL。

    优先级：
    1. `DATABASE_URL`（显式配置，最高优先级）；
    2. `MYSQL_HOST`（老配置）+ 已安装 aiomysql；
    3. 项目根目录的 SQLite 文件。
    """
    explicit = os.getenv("DATABASE_URL")
    if explicit:
        return explicit

    mysql_host = os.getenv("MYSQL_HOST")
    if mysql_host:
        try:
            import aiomysql  # noqa: F401
        except ImportError:
            # 配了 MySQL 却没装异步驱动，如果直接建 engine 会启动失败；
            # 这里降级到 SQLite 并明确提示，别让人对着 traceback 猜
            logger.warning(
                "检测到 MYSQL_HOST 但未安装 aiomysql，本次改用 SQLite。"
                "需要 MySQL 请执行：pip install aiomysql"
            )
        else:
            user = os.getenv("MYSQL_USER", "root")
            password = os.getenv("MYSQL_PASSWORD", "")
            port = os.getenv("MYSQL_PORT", "3306")
            database = os.getenv("MYSQL_DATABASE", "agent_db")
            return (
                f"mysql+aiomysql://{user}:{password}"
                f"@{mysql_host}:{port}/{database}?charset=utf8mb4"
            )

    return f"sqlite+aiosqlite:///{(BASE_DIR / 'dev_agent.db').as_posix()}"


def _resolve_knowledge_dir() -> Path:
    """相对路径按项目根解析，避免从别的目录启动时找不到知识库。"""
    raw = os.getenv("KNOWLEDGE_DIR")
    default = BASE_DIR / "knowledge_base"

    if not raw:
        return default

    path = Path(raw)
    if not path.is_absolute():
        path = BASE_DIR / path

    if path.exists():
        return path

    if default.exists():
        # 配置写错了也不至于静默变成“永远检索不到东西”
        logger.warning(
            "KNOWLEDGE_DIR 指向的目录不存在：%s，改用 %s", path, default
        )
        return default

    return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_base_url=os.getenv("LLM_BASE_URL") or None,
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        llm_timeout=float(os.getenv("LLM_TIMEOUT", "60")),
        database_url=_default_database_url(),
        db_echo=_as_bool(os.getenv("DB_ECHO")),
        # 生产环境务必替换；程序启动时会打印警告
        jwt_secret=os.getenv("JWT_SECRET", "dev-only-change-me"),
        jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
        jwt_expire_minutes=int(os.getenv("JWT_EXPIRE_MINUTES", "720")),
        knowledge_dir=_resolve_knowledge_dir(),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
        ),
        top_k=int(os.getenv("RAG_TOP_K", "3")),
        # 稠密检索的“能不能答”门槛，低于它不认为有证据
        dense_threshold=float(os.getenv("RAG_DENSE_THRESHOLD", "0.55")),
        # 只有关键词命中的片段，语义分至少要到这里才允许入选
        dense_floor=float(os.getenv("RAG_DENSE_FLOOR", "0.45")),
        rrf_k=int(os.getenv("RAG_RRF_K", "60")),
        candidate_pool=int(os.getenv("RAG_CANDIDATE_POOL", "10")),
    )
