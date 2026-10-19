# 评估 Recall@K、正确来源的 MRR 和无答案误召回率。


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval_dataset import EVALUATION_CASES
import tools.knowledge as knowledge


def evaluate_config(
    chunk_size: int,
    overlap: int,
    threshold: float,
    top_k: int,
) -> dict:
    """评估一组 RAG 检索参数。"""

    # 更新当前实验参数
    knowledge.CHUNK_SIZE = chunk_size
    knowledge.CHUNK_OVERLAP = overlap
    knowledge.SIMILARITY_THRESHOLD = threshold
    knowledge.TOP_K = top_k

    # 清除旧文本片段、向量和关键词索引缓存
    knowledge.load_knowledge.cache_clear()
    knowledge.get_knowledge_vectors.cache_clear()
    knowledge._get_keyword_index.cache_clear()

    positive_total = 0
    positive_hits = 0
    reciprocal_rank_total = 0.0
    negative_total = 0
    false_positives = 0
    missed_queries = []
    false_positive_queries = []
    category_counts = {}

    for case in EVALUATION_CASES:
        results = knowledge.knowledge_search(case["query"])

        expected = case["expected"]
        category = case.get("category", "uncategorized")
        category_counts.setdefault(category, {"total": 0, "hits": 0})

        if expected is None:
            negative_total += 1

            if results:
                false_positives += 1
                false_positive_queries.append(case["query"])

            continue

        positive_total += 1
        category_counts[category]["total"] += 1

        expected_sources = set(case.get("sources", []))
        source_rank = next(
            (
                rank
                for rank, item in enumerate(results, start=1)
                if not expected_sources or item["source"] in expected_sources
            ),
            None,
        )
        if source_rank is not None:
            reciprocal_rank_total += 1 / source_rank

        retrieved_text = "\n".join(
            item["content"]
            for item in results
            if not expected_sources or item["source"] in expected_sources
        )

        if all(keyword in retrieved_text for keyword in expected):
            positive_hits += 1
            category_counts[category]["hits"] += 1
        else:
            missed_queries.append(case["query"])

    hit_rate = (
        positive_hits / positive_total
        if positive_total else 0
    )

    false_positive_rate = (
        false_positives / negative_total
        if negative_total else 0
    )

    return {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "threshold": threshold,
        "top_k": top_k,
        "hit_rate": hit_rate,
        "mrr": reciprocal_rank_total / positive_total if positive_total else 0,
        "false_positive_rate": false_positive_rate,
        "positive_hits": positive_hits,
        "positive_total": positive_total,
        "false_positives": false_positives,
        "negative_total": negative_total,
        "missed_queries": missed_queries,
        "false_positive_queries": false_positive_queries,
        "category_counts": category_counts,
    }


def main():
    configurations = [
        (100, 20, 0.45, 3),
        (100, 20, 0.55, 3),
        (300, 50, 0.45, 3),
        (300, 50, 0.55, 3),
        (300, 50, 0.60, 3),
        (300, 50, 0.61, 3),
        (300, 50, 0.65, 3),
        (500, 50, 0.55, 3),
    ]

    print("\nRAG 检索参数评估")
    print("=" * 85)

    results = []

    for config in configurations:
        result = evaluate_config(*config)
        results.append(result)

        print(
            f"chunk={result['chunk_size']:>3}, "
            f"overlap={result['overlap']:>2}, "
            f"threshold={result['threshold']:.2f}, "
            f"top_k={result['top_k']} | "
            f"命中率={result['hit_rate']:.0%} "
            f"({result['positive_hits']}/{result['positive_total']}) | "
            f"MRR={result['mrr']:.3f} | "
            f"误召回率={result['false_positive_rate']:.0%} "
            f"({result['false_positives']}/{result['negative_total']})"
        )

    print("=" * 85)
    print("未命中问题（最后一组配置）：")
    for query in results[-1]["missed_queries"]:
        print(f"  - {query}")
    print("无答案问题误召回（最后一组配置）：")
    for query in results[-1]["false_positive_queries"]:
        print(f"  - {query}")
    print("按问题类型的 Recall@K（最后一组配置）：")
    for category, counts in results[-1]["category_counts"].items():
        if counts["total"]:
            rate = counts["hits"] / counts["total"]
            print(f"  - {category}: {rate:.0%} ({counts['hits']}/{counts['total']})")
    print("评估完成。命中率为带标准证据词和来源约束的 Recall@K；MRR 按首个允许来源的排名计算。")


if __name__ == "__main__":
    main()
