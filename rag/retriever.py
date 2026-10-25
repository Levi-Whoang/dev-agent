"""混合检索入口：向量召回 + BM25 召回 → RRF 融合 → 门槛过滤。

一次 `search()` 的完整链路：
1. 稠密：查询向量和每个片段向量做点积（已归一化，等价余弦），取前 N；
2. 稀疏：BM25 算关键词分数，取前 N（分数为 0 的直接丢）；
3. RRF：两份名次表融合成一份排序；
4. 门槛：语义分太低的不许进——防止“只因为出现‘公司’两个字”
   就把不相关段落塞进证据里，把拒答能力搞坏。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass

import numpy as np

from config import Settings, get_settings
from rag.bm25 import BM25Index
from rag.embedder import Embedder
from rag.hybrid import rank_indices, reciprocal_rank_fusion
from rag.loader import Chunk, dir_signature, load_chunks

logger = logging.getLogger("dev-agent.rag")


@dataclass(slots=True)
class RetrievedChunk:
    content: str
    source: str
    score: float          # 稠密余弦相似度，保留原来的字段含义
    rank_score: float     # RRF 融合分，仅用于排序参考
    dense_rank: int | None
    sparse_rank: int | None
    matched_by: str       # dense / sparse / both

    def to_dict(self) -> dict:
        return asdict(self)


class KnowledgeRetriever:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._embedder = Embedder(self.settings.embedding_model)
        self._chunks: tuple[Chunk, ...] = ()
        self._vectors: np.ndarray | None = None
        self._bm25: BM25Index | None = None
        self._signature: float | None = None
        self._lock = asyncio.Lock()

    async def _ensure_index(self) -> None:
        """建索引：只在知识库变化时重建。"""
        signature = dir_signature(self.settings.knowledge_dir)

        if self._signature == signature and self._vectors is not None:
            return

        async with self._lock:
            if self._signature == signature and self._vectors is not None:
                return

            chunks = load_chunks(self.settings.knowledge_dir)
            self._chunks = chunks
            self._bm25 = (
                BM25Index([c.content for c in chunks]) if chunks else None
            )
            self._vectors = (
                await self._embedder.encode([c.content for c in chunks])
                if chunks
                else None
            )
            self._signature = signature
            logger.info("检索索引已构建：%d 个片段", len(chunks))

    async def search(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            return []

        await self._ensure_index()

        if not self._chunks or self._vectors is None or self._bm25 is None:
            return []

        settings = self.settings
        top_k = top_k or settings.top_k
        pool = min(settings.candidate_pool, len(self._chunks))

        query_vector = await self._embedder.encode_one(query)
        dense_scores = self._vectors @ query_vector

        dense_order = rank_indices(dense_scores, pool)
        sparse_pairs = self._bm25.top_n(query, pool)
        sparse_order = [idx for idx, _ in sparse_pairs]
        sparse_scores = dict(sparse_pairs)

        fused = reciprocal_rank_fusion(
            [dense_order, sparse_order], k=settings.rrf_k
        )

        dense_rank_of = {idx: r for r, idx in enumerate(dense_order, start=1)}
        sparse_rank_of = {idx: r for r, idx in enumerate(sparse_order, start=1)}

        results: list[RetrievedChunk] = []
        seen: set[str] = set()

        for chunk_index, fused_score in fused:
            dense_score = float(dense_scores[chunk_index])
            sparse_score = sparse_scores.get(chunk_index, 0.0)

            keyword_only_ok = (
                sparse_score > 0 and dense_score >= settings.dense_floor
            )
            if dense_score < settings.dense_threshold and not keyword_only_ok:
                continue

            chunk = self._chunks[chunk_index]
            if chunk.content in seen:
                continue
            seen.add(chunk.content)

            in_dense = chunk_index in dense_rank_of
            in_sparse = chunk_index in sparse_rank_of
            matched_by = (
                "both" if in_dense and in_sparse
                else "dense" if in_dense
                else "sparse"
            )

            results.append(
                RetrievedChunk(
                    content=chunk.content,
                    source=chunk.source,
                    score=round(dense_score, 4),
                    rank_score=round(fused_score, 4),
                    dense_rank=dense_rank_of.get(chunk_index),
                    sparse_rank=sparse_rank_of.get(chunk_index),
                    matched_by=matched_by,
                )
            )

            if len(results) >= top_k:
                break

        return results

    async def search_context(self, query: str) -> str:
        results = await self.search(query)

        if not results:
            return "NO_RELEVANT_EVIDENCE"

        blocks = []
        for item in results:
            blocks.append(
                f"来源：{item.source}\n"
                f"证据：{item.content}\n"
                f"语义相似度：{item.score}\n"
                f"命中方式：{item.matched_by}"
            )

        return "\n\n".join(blocks)


_retriever: KnowledgeRetriever | None = None


def get_retriever() -> KnowledgeRetriever:
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeRetriever()
    return _retriever


def reset_retriever() -> None:
    """测试或热更新知识库后重置单例。"""
    global _retriever
    _retriever = None
