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

    # ---------- 记忆 ----------
    # 最近几轮原文的 token 预算（估算值，不是精确值）
    memory_window_tokens: int
    memory_max_turns: int
    # 会话摘要：超出窗口的早期对话压成一段话
    memory_summary_enabled: bool
    # 挤出去的内容攒到多少 token 才压一次，太小会每轮都多打一次模型
    memory_summary_trigger_tokens: int
    # 跨会话的长期记忆（user_memories 表）
    memory_longterm_enabled: bool

    # ---------- ReAct 循环 ----------
    # 一次提问最多走几轮「思考→动作→观测」
    react_max_iterations: int
    # 同一工具 + 同一参数重复到第几次开始提醒/中断
    react_loop_threshold: int
    # 单条观测结果回灌前的截断长度（字符）
    react_observation_limit: int

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
        # 记忆：窗口 3000 token 大约能装 8~12 轮正常对话
        memory_window_tokens=int(os.getenv("MEMORY_WINDOW_TOKENS", "3000")),
        memory_max_turns=int(os.getenv("MEMORY_MAX_TURNS", "12")),
        memory_summary_enabled=_as_bool(
            os.getenv("MEMORY_SUMMARY_ENABLED"), default=True
        ),
        memory_summary_trigger_tokens=int(
            os.getenv("MEMORY_SUMMARY_TRIGGER_TOKENS", "800")
        ),
        memory_longterm_enabled=_as_bool(
            os.getenv("MEMORY_LONGTERM_ENABLED"), default=True
        ),
        # ReAct：10 步足够覆盖"检索→追问角度→组织答案"，
        # 再往上多是模型在原地打转，只会拖长等待时间
        react_max_iterations=int(os.getenv("REACT_MAX_ITERATIONS", "10")),
        react_loop_threshold=int(os.getenv("REACT_LOOP_THRESHOLD", "2")),
        react_observation_limit=int(os.getenv("REACT_OBSERVATION_LIMIT", "1200")),
    )
