"""知识库检索工具（Async 版本）。

工具函数本身保持很薄：真正的检索逻辑在 `rag/` 里。
这样换检索策略（比如以后加 rerank）不用改 Agent 和工具注册表。
"""

from __future__ import annotations

from rag.retriever import get_retriever

# 保留旧名字，外部脚本/测试还能 import 到常量
__all__ = ["knowledge_search", "knowledge_search_context"]


async def knowledge_search(query: str) -> list[dict]:
    """在企业知识库中做混合检索（语义 + 关键词），返回片段、来源和相似度。"""
    results = await get_retriever().search(query)
    return [item.to_dict() for item in results]


async def knowledge_search_context(query: str) -> str:
    """检索企业知识库并整理成带来源的证据文本；没有相关证据时返回 NO_RELEVANT_EVIDENCE。"""
    return await get_retriever().search_context(query)
