#评估检索命中率、无答案误召回等指标


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

    # 清除旧文本片段和旧向量缓存
    knowledge.load_knowledge.cache_clear()
    knowledge.get_knowledge_vectors.cache_clear()

    positive_total = 0
    positive_hits = 0
    negative_total = 0
    false_positives = 0

    for case in EVALUATION_CASES:
        results = knowledge.knowledge_search(case["query"])

        expected = case["expected"]

        if expected is None:
            negative_total += 1

            if results:
                false_positives += 1

            continue

        positive_total += 1

        retrieved_text = "\n".join(
            item["content"] for item in results
        )

        # 所有预期关键词都出现在召回内容中才算命中
        if all(keyword in retrieved_text for keyword in expected):
            positive_hits += 1

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
        "false_positive_rate": false_positive_rate,
        "positive_hits": positive_hits,
        "positive_total": positive_total,
        "false_positives": false_positives,
        "negative_total": negative_total,
    }


def main():
    configurations = [
        (100, 20, 0.45, 3),
        (100, 20, 0.55, 3),
        (300, 50, 0.45, 3),
        (300, 50, 0.55, 3),
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
            f"误召回率={result['false_positive_rate']:.0%} "
            f"({result['false_positives']}/{result['negative_total']})"
        )

    print("=" * 85)
    print("评估完成。请结合命中率和误召回率选择参数。")


if __name__ == "__main__":
    main()