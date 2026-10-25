"""混合检索的融合策略：RRF（Reciprocal Rank Fusion）。

为什么不直接把“余弦相似度”和“BM25 分数”相加：
两套分数的量纲完全不一样——余弦在 0~1，BM25 可以是 0~30，
直接相加等于让 BM25 单方面决定排序。

RRF 只用“名次”：score = Σ 1 / (k + rank)。
好处是不需要归一化、不需要调权重，两个检索器各自排第几就贡献多少分，
对量纲差异天然免疫。k 一般取 60，作用是压一压头部名次的绝对优势。
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """输入若干个有序的文档 id 列表，返回融合后的 (id, score) 降序列表。"""
    if k <= 0:
        raise ValueError("k 必须大于 0")

    fused: dict[int, float] = {}

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


def rank_indices(scores: Sequence[float], limit: int) -> list[int]:
    """按分数降序取前 limit 个下标。"""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return order[:limit]
