"""装配工厂：把「客户端 / 工具表 / 数据库会话」拼成一个 Agent。

为什么不在 main.py 里建全局 agent：
AsyncSession 是“每个请求一个”的，进程级单例会让两个用户的对话共用一个连接，
高并发下直接报错。所以客户端和工具表是单例，Agent 和 store 按请求创建。
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession

from agent.agent import Agent
from agent.memory import MemoryManager
from agent.store import SessionStore
from config import get_settings
from tools.memory_tool import make_remember_tool
from tools.registry import TOOLS
from tools.schema import function_to_schema, tools_to_schema

logger = logging.getLogger("dev-agent.factory")


@lru_cache(maxsize=1)
def get_llm_client():
    from openai import AsyncOpenAI

    settings = get_settings()
    if not settings.llm_api_key:
        logger.warning("LLM_API_KEY 未配置，聊天接口会在调用模型时失败")

    return AsyncOpenAI(
        api_key=settings.llm_api_key or "not-configured",
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout,
    )


@lru_cache(maxsize=1)
def get_tool_schemas() -> list[dict]:
    return tools_to_schema(TOOLS)


def build_agent(db: AsyncSession, session_id: str, user_id: int | None = None) -> Agent:
    """装配一个 Agent。

    user_id 是给长期记忆用的：摘要只关心"这次对话"，而"用户是谁"
    必须跨会话，所以记忆层需要知道当前登录的是谁。
    不传的话长期记忆自动关闭，其余功能不受影响。
    """
    settings = get_settings()
    store = SessionStore(db)

    memory = MemoryManager(
        store=store,
        session_id=session_id,
        user_id=user_id,
        client=get_llm_client(),
        model=settings.llm_model,
        window_tokens=settings.memory_window_tokens,
        max_turns=settings.memory_max_turns,
        summary_enabled=settings.memory_summary_enabled,
        summary_trigger_tokens=settings.memory_summary_trigger_tokens,
        longterm_enabled=settings.memory_longterm_enabled,
    )

    # 工具表按请求构造：remember 需要绑定当前用户，不能放进进程级单例
    registry = dict(TOOLS)
    schemas = get_tool_schemas()

    if user_id is not None and settings.memory_longterm_enabled:
        remember = make_remember_tool(db, user_id, session_id)
        registry["remember"] = remember
        # get_tool_schemas() 返回的是缓存列表，用 + 拼新列表，别原地 append
        schemas = schemas + [function_to_schema(remember)]

    return Agent(
        client=get_llm_client(),
        store=store,
        session_id=session_id,
        tools=schemas,
        tool_registry=registry,
        model=settings.llm_model,
        memory=memory,
        max_iterations=settings.react_max_iterations,
        loop_threshold=settings.react_loop_threshold,
        observation_limit=settings.react_observation_limit,
    )
