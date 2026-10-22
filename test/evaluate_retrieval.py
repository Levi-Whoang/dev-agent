# 评估 Recall@K、正确来源的 MRR 和无答案误召回率。


import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval_dataset import EVALUATION_CASES, EVALUATION_SET_VERSION
import tools.knowledge as knowledge


CONFIGURATIONS = [
    (100, 20, 0.45, 3),
    (100, 20, 0.55, 3),
    (300, 50, 0.45, 3),
    (300, 50, 0.55, 3),
    (300, 50, 0.60, 3),
    (300, 50, 0.61, 3),
    (300, 50, 0.65, 3),
    (500, 50, 0.55, 3),
]


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
    failure_cases = []
    category_counts = {}

    for index, case in enumerate(EVALUATION_CASES, start=1):
        results = knowledge.knowledge_search(case["query"])

        expected = case["expected"]
        category = case.get("category", "uncategorized")
        counts = category_counts.setdefault(category, {"total": 0, "correct": 0})
        counts["total"] += 1
        result_summary = [
            {
                "source": item["source"],
                "score": item.get("score"),
                "dense_score": item.get("dense_score"),
                "keyword_score": item.get("keyword_score"),
                "content_sha256": hashlib.sha256(
                    item["content"].encode("utf-8")
                ).hexdigest(),
            }
            for item in results
        ]

        if expected is None:
            negative_total += 1

            if results:
                false_positives += 1
                false_positive_queries.append(case["query"])
                failure_cases.append({
                    "case_id": f"R{index:03d}",
                    "failure_type": "false_positive",
                    "category": category,
                    "query": case["query"],
                    "expected": None,
                    "retrieved": result_summary,
                })
            else:
                counts["correct"] += 1

            continue

        positive_total += 1
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
            counts["correct"] += 1
        else:
            missed_queries.append(case["query"])
            failure_cases.append({
                "case_id": f"R{index:03d}",
                "failure_type": "missed_answerable",
                "category": category,
                "query": case["query"],
                "expected_evidence": expected,
                "allowed_sources": sorted(expected_sources),
                "retrieved": result_summary,
            })

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
        "failure_cases": failure_cases,
    }


def build_report(results: list[dict]) -> dict:
    """組裝可追溯版本、配置和失敗案例的評估報告。"""
    dataset_payload = json.dumps(
        EVALUATION_CASES,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    distribution = Counter(
        case.get("category", "uncategorized") for case in EVALUATION_CASES
    )
    return {
        "report_schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": {
            "version": EVALUATION_SET_VERSION,
            "sha256": hashlib.sha256(dataset_payload.encode("utf-8")).hexdigest(),
            "case_count": len(EVALUATION_CASES),
            "category_counts": dict(sorted(distribution.items())),
        },
        "retriever_settings": {
            "embedding_model": "BAAI/bge-small-zh-v1.5",
            "dense_weight": knowledge.DENSE_WEIGHT,
            "keyword_weight": knowledge.KEYWORD_WEIGHT,
            "bm25_k1": knowledge.BM25_K1,
            "bm25_b": knowledge.BM25_B,
            "keyword_min_coverage": knowledge.KEYWORD_MIN_COVERAGE,
        },
        "configurations": [
            {
                "chunk_size": item[0],
                "overlap": item[1],
                "threshold": item[2],
                "top_k": item[3],
            }
            for item in CONFIGURATIONS
        ],
        "results": results,
    }


def save_report(report: dict, output_dir: Path | None = None) -> Path:
    """将 JSON 报告写入版本化评估结果目录。"""
    output_dir = output_dir or PROJECT_ROOT / "test" / "evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f_%z")
    path = output_dir / f"rag_retrieval_{timestamp}.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def main():
    print("\nRAG 检索参数评估")
    print("=" * 85)
    print(f"评估集版本：{EVALUATION_SET_VERSION}（{len(EVALUATION_CASES)} 条）")

    results = []

    for config in CONFIGURATIONS:
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
    print("按问题类型的正确率（最后一组配置）：")
    for category, counts in results[-1]["category_counts"].items():
        if counts["total"]:
            rate = counts["correct"] / counts["total"]
            print(f"  - {category}: {rate:.0%} ({counts['correct']}/{counts['total']})")
    print("评估完成。命中率为带标准证据词和来源约束的 Recall@K；MRR 按首个允许来源的排名计算。")
    report_path = save_report(build_report(results))
    print(f"完整评估配置、失败案例和数据集指纹已保存：{report_path}")


if __name__ == "__main__":
    main()
