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
from agent.store import SessionStore
from config import get_settings
from tools.registry import TOOLS
from tools.schema import tools_to_schema

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


def build_agent(db: AsyncSession, session_id: str) -> Agent:
    settings = get_settings()
    return Agent(
        client=get_llm_client(),
        store=SessionStore(db),
        session_id=session_id,
        tools=get_tool_schemas(),
        tool_registry=TOOLS,
        model=settings.llm_model,
    )
