"""独立于模型和数据库的检索评估指标。"""
import hashlib
from collections import defaultdict

# 用来源约束和标准证据词评价一次检索，不把相关来源等同于正确证据。
def score_case(case, results):
    expected = case["expected"]
    allowed = set(case.get("sources", []))
    relevant = [r for r in results if not allowed or r["source"] in allowed]
    if expected is None:
        return {"correct": not results, "false_positive": bool(results), "source_rr": 0, "evidence_rr": 0}
    joined = "\n".join(r["content"] for r in relevant)
    source_rank = next((i for i,r in enumerate(results, 1) if not allowed or r["source"] in allowed), None)
    evidence_rank = next((i for i,r in enumerate(results, 1) if (not allowed or r["source"] in allowed) and all(word in r["content"] for word in expected)), None)
    return {"correct": all(word in joined for word in expected), "false_positive": False,
            "source_rr": 1/source_rank if source_rank else 0, "evidence_rr": 1/evidence_rank if evidence_rank else 0}

# 汇总正例证据覆盖命中、来源/单片段证据排名、无答案误召回及分类正确率。
def summarize(rows):
    positives = [r for r in rows if r["answerable"]]
    negatives = [r for r in rows if not r["answerable"]]
    categories = defaultdict(lambda: {"total": 0, "correct": 0})
    for row in rows:
        categories[row["category"]]["total"] += 1
        categories[row["category"]]["correct"] += int(row["correct"])
    return {"positive_total": len(positives), "negative_total": len(negatives),
            "evidence_hit_at_k": sum(r["correct"] for r in positives)/len(positives) if positives else None,
            "source_mrr": sum(r["source_rr"] for r in positives)/len(positives) if positives else None,
            "single_chunk_evidence_mrr": sum(r["evidence_rr"] for r in positives)/len(positives) if positives else None,
            "false_positive_rate": sum(r["false_positive"] for r in negatives)/len(negatives) if negatives else None,
            "categories": dict(categories)}

# 记录来源、分数与证据摘要哈希，报告中不复制完整文档正文。
def result_digest(results):
    return [{"source": r["source"], "score": r.get("score"), "dense_score": r.get("dense_score"),
             "keyword_score": r.get("keyword_score"),
             "content_sha256": hashlib.sha256(r["content"].encode()).hexdigest()} for r in results]

# 同一问题成对对照，保留提升、退化及排序变化。
def compare(cases, search):
    rows = {"dense": [], "hybrid": []}
    differences = []
    for index, case in enumerate(cases, 1):
        pair = {}
        for mode in rows:
            results = search(case["query"], mode)
            row = {"case_id": f"R{index:03}", "query": case["query"], "category": case.get("category", "unknown"),
                   "answerable": case["expected"] is not None, **score_case(case, results), "retrieved": result_digest(results)}
            rows[mode].append(row)
            pair[mode] = row
        if pair["dense"]["retrieved"] != pair["hybrid"]["retrieved"]:
            differences.append({"case_id": f"R{index:03}", "query": case["query"],
                                "change": "improved" if pair["hybrid"]["correct"] and not pair["dense"]["correct"] else
                                "regressed" if pair["dense"]["correct"] and not pair["hybrid"]["correct"] else "ranking_or_score_only"})
    return {"metrics": {mode: summarize(items) for mode, items in rows.items()}, "cases": rows, "differences": differences}
