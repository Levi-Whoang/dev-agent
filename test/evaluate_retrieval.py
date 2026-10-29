"""检索消融评测：三种检索方式对比 + 门槛贡献分析。

用法：
    python test/evaluate_retrieval.py
    python test/evaluate_retrieval.py --json logs/eval.json

为什么不只跑生产配置：
只说"我做了混合检索"是没有信息量的 —— 得回答"混合比单路好多少"、
"如果只看关键词会漏掉什么"。所以这里把同一份标注集跑四遍：

    仅向量 / 仅 BM25 / 混合 RRF（生产）/ 混合但关掉门槛

指标定义（都写在输出里，避免各说各话）：

    Recall@3   前 3 个**不同文档**里是否命中期望文档
    MRR@10     第一个相关文档排在第几位的倒数，没命中记 0
    拒答准确率 负样本（知识库确实没答案）中返回空结果的比例
    误拒率     正样本中返回空结果的比例 —— 必须和拒答率一起看，
               否则只要把阈值调到无限高，拒答率就是 100%

三个前置约束：
1. 只测线上真正跑的那条路径（mode 是 retriever 自己的开关，不是脚本里另写一份）；
2. 取 top_k=10 一次拿够，Recall@3 从里面截，保证两个指标同源；
3. 文档级去重 —— 一篇文档可能切成好几个片段，不能让它一个人占满前 3 名。
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_settings  # noqa: E402
from rag.loader import load_chunks  # noqa: E402
from rag.retriever import KnowledgeRetriever  # noqa: E402
from retrieval_eval_set import CASES, negatives, positives  # noqa: E402

# 一次拿 10 条，Recall@3 与 MRR@10 都用这一份结果，避免两次检索的随机性
FETCH_K = 10
RECALL_K = 3

CONFIGS: list[tuple[str, dict]] = [
    ("仅向量 dense", {"mode": "dense", "apply_gate": True}),
    ("仅关键词 BM25", {"mode": "sparse", "apply_gate": True}),
    ("混合 RRF（生产）", {"mode": "hybrid", "apply_gate": True}),
    ("混合 · 关门槛", {"mode": "hybrid", "apply_gate": False}),
]


def ranked_sources(results) -> list[str]:
    """按名次排好的**不同文档**列表。"""
    seen: list[str] = []
    for item in results:
        if item.source not in seen:
            seen.append(item.source)
    return seen


def evaluate_case(results, case: dict) -> dict:
    if not case["docs"]:
        # 负样本多记一项 first_matched_by：
        # "sparse" 意味着这条结果**只**靠关键词兜底通道进来了，
        # 它是误召的主要来源，单独统计才能解释数字。
        return {
            "refused": len(results) == 0,
            "hit": None,
            "mrr": None,
            "first_matched_by": results[0].matched_by if results else None,
        }

    docs = ranked_sources(results)
    top = docs[:RECALL_K]
    first_rank = next(
        (rank for rank, doc in enumerate(docs, start=1) if doc in case["docs"]),
        None,
    )
    return {
        "refused": len(results) == 0,
        "hit": any(doc in top for doc in case["docs"]),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "first_rank": first_rank,
        "ranked": docs,
    }


async def run_config(
    retriever: KnowledgeRetriever,
    label: str,
    options: dict,
    cases: list[dict] | None = None,
) -> dict:
    cases = cases or CASES
    mode = options["mode"]
    gate = options["apply_gate"]

    records: list[dict] = []
    for case in cases:
        results = await retriever.search(
            case["query"], top_k=FETCH_K, mode=mode, apply_gate=gate
        )
        scored = evaluate_case(results, case)
        records.append({"case": case, **scored})

    pos = [r for r in records if r["case"]["docs"]]
    neg = [r for r in records if not r["case"]["docs"]]

    recall = sum(1 for r in pos if r["hit"]) / len(pos)
    mrr = sum(r["mrr"] for r in pos) / len(pos)
    refusal = sum(1 for r in neg if r["refused"]) / len(neg)
    over_refusal = sum(1 for r in pos if r["refused"]) / len(pos)

    per_difficulty = {}
    for level in ("easy", "medium", "hard"):
        subset = [r for r in pos if r["case"]["difficulty"] == level]
        if subset:
            per_difficulty[level] = sum(1 for r in subset if r["hit"]) / len(subset)

    return {
        "label": label,
        "mode": mode,
        "apply_gate": gate,
        "recall@3": recall,
        "mrr@10": mrr,
        "refusal_accuracy": refusal,
        "over_refusal": over_refusal,
        "positive_hits": sum(1 for r in pos if r["hit"]),
        "positive_total": len(pos),
        "false_positives": sum(1 for r in neg if not r["refused"]),
        "false_positive_via_sparse": sum(
            1 for r in neg if not r["refused"] and r.get("first_matched_by") == "sparse"
        ),
        "negative_total": len(neg),
        "per_difficulty": per_difficulty,
        # 逐条结果留着，才能做配对比较。
        # 光看聚合百分比会把"1 条用例"说成"1.2 个百分点"，
        # 在 83 条样本上这就是噪音，必须看"谁赢谁输"。
        "per_case": {
            r["case"]["id"]: {"hit": r["hit"], "rank": r.get("first_rank")}
            for r in pos
        },
        "misses": [
            {
                "id": r["case"]["id"],
                "query": r["case"]["query"],
                "difficulty": r["case"]["difficulty"],
                "expected": r["case"]["docs"],
                "got": r.get("ranked", [])[:RECALL_K],
            }
            for r in pos
            if not r["hit"]
        ],
        "false_positive_cases": [
            {"id": r["case"]["id"], "query": r["case"]["query"]}
            for r in neg
            if not r["refused"]
        ],
    }


def split_cases(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    """按难度**分层**切分，约 2/3 开发集 + 1/3 留出集。

    两个坑都踩过：
    1. 在同一份数据上调阈值再报同一份数据的成绩，等于自己给自己判卷；
    2. 纯随机切分会让负样本分布跑偏 —— 第一次跑出来留出集只剩 4 条负样本，
       拒答率"100%"其实只是 4 条里蒙对 4 条，毫无意义。
       所以按 difficulty 分层，保证每个难度在两边都有大致相同的比例。

    用 crc32 而不是 hash()：Python 的 hash 每个进程都加盐，换个进程切分就变了。
    """
    dev, test = [], []
    seen: dict[str, int] = {}
    for case in cases:
        level = case["difficulty"]
        position = seen.get(level, 0)
        seen[level] = position + 1
        (test if position % 3 == 0 else dev).append(case)
    return dev, test


async def sweep(
    retriever: KnowledgeRetriever, dev: list[dict]
) -> list[dict]:
    """在开发集上扫阈值组合。

    目标函数用 balanced accuracy = (召回 + 拒答) / 2。
    这只是个产品取舍：如果业务更怕"乱答"，就该给拒答更高权重。
    这里取等权，是为了不预设偏好，把两个方向都摆出来。
    """
    base = retriever.settings
    rows: list[dict] = []

    for threshold, floor in itertools.product(
        [0.45, 0.50, 0.55, 0.60, 0.65, 0.70], [0.35, 0.40, 0.45, 0.50, 0.55]
    ):
        if floor >= threshold:
            continue  # 门槛比兜底还高，等于兜底不起作用，没意义
        retriever.settings = replace(
            base, dense_threshold=threshold, dense_floor=floor
        )
        result = await run_config(
            retriever,
            f"t={threshold} f={floor}",
            {"mode": "hybrid", "apply_gate": True},
            dev,
        )
        result["threshold"] = threshold
        result["floor"] = floor
        result["balanced"] = (
            result["recall@3"] + result["refusal_accuracy"]
        ) / 2
        rows.append(result)

    retriever.settings = base
    return sorted(rows, key=lambda r: r["balanced"], reverse=True)


def print_pairing(results: list[dict]) -> None:
    """配对比较：混合检索到底赢了几条、输了几条。

    为什么必须看这个：
    83 条正样本里 1 条 = 1.2 个百分点。只看 96.4% → 97.6% 很像"有提升"，
    拆开看可能只是"赢了 2 条、输了 1 条"。这种量级没有统计意义，
    不能写进结论，更不能写进简历。
    """
    hybrid = next(r for r in results if "生产" in r["label"])
    print("配对比较（每条用例逐个比，而不是比总分）")
    print("-" * 82)
    print(f"{'对比':<24}{'混合赢':>8}{'混合输':>8}{'打平':>8}{'净差':>8}")
    for other in results:
        if other is hybrid:
            continue
        win = lose = 0
        for case_id, mine in hybrid["per_case"].items():
            theirs = other["per_case"].get(case_id)
            if theirs is None:
                continue
            if mine["hit"] and not theirs["hit"]:
                win += 1
            elif theirs["hit"] and not mine["hit"]:
                lose += 1
        tie = hybrid["positive_total"] - win - lose
        print(
            f"{'混合 vs ' + other['label']:<24}"
            f"{win:>8}{lose:>8}{tie:>8}{win - lose:>+8}"
        )
    print()

    # 有多少题是"怎么检都中"或"怎么检都不中"—— 这两类不提供任何信息
    always = sum(
        1
        for case_id, mine in hybrid["per_case"].items()
        if all(r["per_case"][case_id]["hit"] for r in results)
    )
    never = sum(
        1
        for case_id in hybrid["per_case"]
        if not any(r["per_case"][case_id]["hit"] for r in results)
    )
    total = hybrid["positive_total"]
    print(
        f"  全部四种配置都命中：{always}/{total}"
        f"    全部都没命中：{never}/{total}"
        f"    真正能区分优劣的：{total - always - never}/{total}"
    )
    print()


def print_environment(retriever: KnowledgeRetriever, chunks) -> None:
    s = retriever.settings
    docs = len({c.source for c in chunks})
    print("运行环境")
    print("-" * 82)
    print(f"  知识库      {s.knowledge_dir}")
    print(f"  文档 / 片段  {docs} 篇 → {len(chunks)} 个片段")
    print(f"  向量模型    {s.embedding_model}")
    print(f"  生产参数    top_k={s.top_k} threshold={s.dense_threshold} "
          f"floor={s.dense_floor} rrf_k={s.rrf_k} pool={s.candidate_pool}")
    print(f"  评测集      {len(CASES)} 条（可回答 {len(positives())} / "
          f"不可回答 {len(negatives())}）")
    print()


def print_table(results: list[dict]) -> None:
    header = (
        f"{'配置':<18}{'Recall@3':>10}{'MRR@10':>9}"
        f"{'拒答准确':>10}{'误拒率':>9}{'误召数':>8}"
    )
    print(header)
    print("-" * 82)
    for r in results:
        print(
            f"{r['label']:<18}"
            f"{r['recall@3']:>9.1%} "
            f"{r['mrr@10']:>8.3f} "
            f"{r['refusal_accuracy']:>9.1%} "
            f"{r['over_refusal']:>8.1%} "
            f"{r['false_positives']:>7}/{r['negative_total']}"
        )
    print()
    print("明细（命中数 / 可回答总数）：")
    for r in results:
        print(f"  {r['label']:<18} {r['positive_hits']}/{r['positive_total']}")
    print()


def print_difficulty(results: list[dict]) -> None:
    print("按难度拆开看 Recall@3（easy 复用原词 / hard 刻意避开原词）")
    print("-" * 82)
    print(f"{'配置':<18}{'easy':>8}{'medium':>9}{'hard':>8}")
    for r in results:
        d = r["per_difficulty"]
        print(
            f"{r['label']:<18}"
            f"{d.get('easy', 0):>7.0%} {d.get('medium', 0):>8.0%} "
            f"{d.get('hard', 0):>7.0%}"
        )
    print()


def print_failures(production: dict) -> None:
    print(f"生产配置的漏检（{len(production['misses'])} 条）")
    print("-" * 82)
    for miss in production["misses"]:
        print(f"  [{miss['difficulty']}] {miss['query']}")
        print(f"        期望 {miss['expected']}")
        print(f"        实际 {miss['got'] or '（空结果）'}")
    print()

    if production["false_positive_cases"]:
        print(f"生产配置的误召回（{len(production['false_positive_cases'])} 条）")
        print("-" * 82)
        for item in production["false_positive_cases"]:
            print(f"  - {item['query']}")
        via_sparse = production["false_positive_via_sparse"]
        print(
            f"  其中 {via_sparse}/{production['false_positives']} 条**只**靠关键词兜底"
            f"通道进来（dense 低于门槛、但 BM25 命中）"
        )
        print()


async def main() -> None:
    parser = argparse.ArgumentParser(description="检索消融评测")
    parser.add_argument("--json", help="把结果写成 JSON（便于追踪历史）")
    parser.add_argument(
        "--sweep", action="store_true", help="额外跑阈值扫描（开发集调参 + 留出集验证）"
    )
    args = parser.parse_args()

    settings = get_settings()
    if not settings.knowledge_dir.exists():
        raise SystemExit(
            "知识库目录不存在，先运行 python scripts/generate_knowladge_base.py"
        )

    retriever = KnowledgeRetriever()
    await retriever.search("预热", top_k=1)  # 先建索引，别把建索引时间算进评测

    chunks = load_chunks(settings.knowledge_dir)
    print()
    print("=" * 82)
    print("RAG 检索消融评测")
    print("=" * 82)
    print_environment(retriever, chunks)

    results = []
    for label, options in CONFIGS:
        results.append(await run_config(retriever, label, options))

    print("总体指标")
    print("-" * 82)
    print_table(results)
    print_difficulty(results)
    print_pairing(results)

    production = next(r for r in results if "生产" in r["label"])
    print_failures(production)

    sweep_payload = None
    if args.sweep:
        dev, test = split_cases(CASES)
        dev_neg = sum(1 for c in dev if not c["docs"])
        test_neg = sum(1 for c in test if not c["docs"])
        print("=" * 82)
        print(
            f"阈值扫描：开发集 {len(dev)} 条（含负样本 {dev_neg}）调参"
            f" → 留出集 {len(test)} 条（含负样本 {test_neg}）验证"
        )
        print("=" * 82)

        rows = await sweep(retriever, dev)
        print("开发集 Top 8（目标 = balanced accuracy =（召回 + 拒答）/ 2）")
        print("-" * 82)
        print(f"{'threshold':>10}{'floor':>8}{'Recall@3':>11}"
              f"{'拒答准确':>11}{'balanced':>10}")
        for row in rows[:8]:
            print(
                f"{row['threshold']:>10.2f}{row['floor']:>8.2f}"
                f"{row['recall@3']:>10.1%} {row['refusal_accuracy']:>10.1%} "
                f"{row['balanced']:>9.1%}"
            )

        best = rows[0]
        base = retriever.settings
        comparison = []
        for label, threshold, floor in (
            ("当前生产参数", base.dense_threshold, base.dense_floor),
            ("扫描最优参数", best["threshold"], best["floor"]),
        ):
            retriever.settings = replace(
                base, dense_threshold=threshold, dense_floor=floor
            )
            r = await run_config(
                retriever,
                label,
                {"mode": "hybrid", "apply_gate": True},
                test,
            )
            comparison.append({**r, "threshold": threshold, "floor": floor})
        retriever.settings = base

        print()
        print(f"留出集验证（{len(test)} 条，调参时没见过）")
        print("-" * 82)
        print(f"{'参数':<14}{'threshold':>10}{'floor':>8}"
              f"{'Recall@3':>11}{'拒答准确':>11}{'balanced':>10}")
        for r in comparison:
            balanced = (r["recall@3"] + r["refusal_accuracy"]) / 2
            print(
                f"{r['label']:<14}{r['threshold']:>10.2f}{r['floor']:>8.2f}"
                f"{r['recall@3']:>10.1%} {r['refusal_accuracy']:>10.1%} "
                f"{balanced:>9.1%}"
            )
        print()
        print("注：留出集只有统计意义，样本量小时 1 条用例就是 3 个百分点。")
        print()

        sweep_payload = {
            "dev_size": len(dev),
            "test_size": len(test),
            "dev_top": [
                {
                    "threshold": r["threshold"],
                    "floor": r["floor"],
                    "recall@3": r["recall@3"],
                    "refusal_accuracy": r["refusal_accuracy"],
                    "balanced": r["balanced"],
                }
                for r in rows[:8]
            ],
            "heldout": comparison,
        }

    print("=" * 82)
    print("注：Recall@3 按**不同文档**去重计算；负样本 20 条中后 10 条是与知识库"
          "有词汇重叠的强干扰。")
    print("=" * 82)

    if args.json:
        payload = {
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "env": {
                "knowledge_dir": str(settings.knowledge_dir),
                "chunks": len(chunks),
                "embedding_model": settings.embedding_model,
                "top_k": settings.top_k,
                "dense_threshold": settings.dense_threshold,
                "dense_floor": settings.dense_floor,
                "rrf_k": settings.rrf_k,
                "candidate_pool": settings.candidate_pool,
                "cases": len(CASES),
            },
            "results": results,
            "sweep": sweep_payload,
        }
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
        print(f"\n已写入 {out}")


if __name__ == "__main__":
    asyncio.run(main())
