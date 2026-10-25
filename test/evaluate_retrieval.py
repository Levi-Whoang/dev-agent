"""检索参数评估：命中率 vs 误召回率。

用法：
    python test/evaluate_retrieval.py

会真的加载向量模型（只跑一次索引），然后对每组参数重新算分。
注意：改 chunk 大小需要重建索引，所以这里只扫阈值和 top_k。
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_settings  # noqa: E402
from rag.retriever import KnowledgeRetriever  # noqa: E402
from retrieval_dataset import EVALUATION_CASES  # noqa: E402


async def evaluate_config(retriever: KnowledgeRetriever, **overrides) -> dict:
    retriever.settings = replace(get_settings(), **overrides)

    positive_total = positive_hits = 0
    negative_total = false_positives = 0
    missing: list[str] = []

    for case in EVALUATION_CASES:
        results = await retriever.search(case["query"])
        expected = case["expected"]

        if expected is None:
            negative_total += 1
            if results:
                false_positives += 1
                missing.append(f"误召回：{case['query']}")
            continue

        positive_total += 1
        text = "\n".join(item.content for item in results)
        absent = [keyword for keyword in expected if keyword not in text]
        if absent:
            missing.append(f"{case['query']} 缺 {'/'.join(absent)}")
        else:
            positive_hits += 1

    return {
        "hit_rate": positive_hits / positive_total if positive_total else 0.0,
        "false_positive_rate": (
            false_positives / negative_total if negative_total else 0.0
        ),
        "positive_hits": positive_hits,
        "positive_total": positive_total,
        "false_positives": false_positives,
        "negative_total": negative_total,
        "missing": missing,
    }


async def main() -> None:
    settings = get_settings()
    if not settings.knowledge_dir.exists():
        raise SystemExit(
            "知识库目录不存在，先运行 python scripts/generate_knowladge_base.py"
        )

    retriever = KnowledgeRetriever()
    await retriever.search("预热")  # 先建索引，避免每组参数都重新编码

    configs = [
        {"dense_threshold": 0.50, "dense_floor": 0.40, "top_k": 3},
        {"dense_threshold": 0.55, "dense_floor": 0.45, "top_k": 3},
        {"dense_threshold": 0.60, "dense_floor": 0.50, "top_k": 3},
        {"dense_threshold": 0.55, "dense_floor": 0.45, "top_k": 5},
        {"dense_threshold": 0.65, "dense_floor": 0.55, "top_k": 3},
    ]

    print("\nRAG 检索参数评估（混合检索 + RRF）")
    print("=" * 78)

    for config in configs:
        result = await evaluate_config(retriever, **config)
        print(
            f"threshold={config['dense_threshold']:.2f} "
            f"floor={config['dense_floor']:.2f} "
            f"top_k={config['top_k']} | "
            f"命中率={result['hit_rate']:.0%} "
            f"({result['positive_hits']}/{result['positive_total']}) | "
            f"误召回率={result['false_positive_rate']:.0%} "
            f"({result['false_positives']}/{result['negative_total']})"
        )
        for note in result["missing"]:
            print(f"    - {note}")

    print("=" * 78)
    print("样例很少，只用来观察趋势；真正调参需要更大的标注集。")


if __name__ == "__main__":
    asyncio.run(main())
