"""运行同语料、同参数的纯向量与混合检索对照实验。"""
import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

from evaluation.metrics import compare
from test.retrieval_dataset import EVALUATION_CASES, EVALUATION_SET_VERSION
from tools import knowledge

ROOT = Path(__file__).resolve().parent.parent

# 为配置、语料和问题集生成可追溯指纹。
def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

# 根据报告生成中文 TXT，明确指标口径、差异和适用边界。
def render_summary(report):
    lines = ["RAG 检索模式对照评估", "生成时间：" + report["generated_at"],
             "比较对象：dense=历史纯向量策略重建；hybrid=当前向量+BM25加权融合。", 
             "两者复用当前相同的切分、模型、阈值、Top-K、去重；不代表任一历史提交的全版本性能。",
             f"问题集：{report['dataset']['version']}，共 {report['dataset']['count']} 条。",
             f"公共语料文件数：{len(report['corpus'])}；SHA256：{report['corpus_sha256']}",
             "配置：" + json.dumps(report["config"], ensure_ascii=False),
             "证据覆盖命中率=允许来源的 Top-K 文本合并后覆盖全部标准词；不是严格片段 Recall。",
             "source_mrr=首个允许来源倒数排名；single_chunk_evidence_mrr 要求单个片段包含全部标准词。",
             "误召回率=无答案问题中返回任意片段的比例；不是最终回答幻觉率。", ""]
    for mode, metrics in report["comparison"]["metrics"].items():
        lines.append(f"{mode}：正例 {metrics['positive_total']}，负例 {metrics['negative_total']}")
        for key in ["evidence_hit_at_k", "source_mrr", "single_chunk_evidence_mrr", "false_positive_rate"]:
            lines.append(f"  {key}: {metrics[key]:.4f}" if metrics[key] is not None else f"  {key}: N/A")
        for category, counts in metrics["categories"].items():
            lines.append(f"  {category}: {counts['correct']}/{counts['total']}")
    counts = Counter(row["change"] for row in report["comparison"]["differences"])
    lines += ["", "混合检索相对纯向量：" + json.dumps(dict(counts), ensure_ascii=False)]
    dense, hybrid = (report["comparison"]["metrics"][mode] for mode in ["dense", "hybrid"])
    for key in ["evidence_hit_at_k", "false_positive_rate"]:
        if dense[key] is not None and hybrid[key] is not None:
            lines.append(f"{key} 变化：{(hybrid[key]-dense[key])*100:+.2f} 个百分点")
    lines += ["", "正例标注在完整公共语料内仍缺少标准词的项目（未偷偷排除，可能影响绝对分数）："]
    lines += [f"  {item['case_id']}: {item['missing_terms']}" for item in report["gold_audit"]] or ["  无"]
    lines += ["", "发生差异的问题："]
    lines += [f"  {r['case_id']} {r['change']}: {r['query']}" for r in report["comparison"]["differences"]]
    lines += ["", "说明：混合检索允许低向量分但关键词覆盖达标的片段入选，可能补回精确关键词，也可能增加误召回；以本次成对结果判断。",
              "本次只评估检索；没有调用生成模型，不能代表答案正确率或引用支持性。问题集是开发集，不是独立留出集。",
              "耗时包含各次查询编码与排序，模型/文档编码预热单独记录；不作为严格性能基准。",
              f"预热耗时 {report['warmup_seconds']:.3f}s；对照总耗时 {report['comparison_seconds']:.3f}s。",
              "完整逐题结果、失败案例（correct=false）、配置和环境见同名 JSON。"]
    return "\n".join(lines) + "\n"

# 加载固定公共语料并运行对照，将 JSON 和 TXT 同时保存。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "demo_data/knowledge_base")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/results")
    args = parser.parse_args()
    corpus = args.corpus.resolve()
    knowledge.KNOWLEDGE_DIR = corpus
    knowledge.refresh_knowledge_cache()
    files = sorted(corpus.glob("*.txt"))
    if not files:
        raise RuntimeError("语料目录没有公共 TXT 文件")
    manifest = [{"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in files]
    chunks = knowledge.load_knowledge()
    audit = []
    for i, case in enumerate(EVALUATION_CASES, 1):
        if case["expected"] is None:
            continue
        text = "\n".join(r["content"] for r in chunks if r["source"] in case.get("sources", []))
        missing = [word for word in case["expected"] if word not in text]
        if missing:
            audit.append({"case_id": f"R{i:03}", "missing_terms": missing})
    started = time.perf_counter()
    model = knowledge.get_embedding_model()
    knowledge.get_knowledge_vectors()
    warmup = time.perf_counter()-started
    started = time.perf_counter()
    comparison = compare(EVALUATION_CASES, knowledge.search_knowledge)
    duration = time.perf_counter()-started
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip() != ""
    now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8)))
    report = {"schema_version": 1, "generated_at": now.isoformat(),
              "dataset": {"version": EVALUATION_SET_VERSION, "sha256": fingerprint(EVALUATION_CASES), "count": len(EVALUATION_CASES)},
              "corpus": manifest, "corpus_sha256": fingerprint(manifest), "gold_audit": audit,
              "config": {name: getattr(knowledge, name) for name in ["CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "SIMILARITY_THRESHOLD", "DENSE_WEIGHT", "KEYWORD_WEIGHT", "KEYWORD_MIN_COVERAGE", "BM25_K1", "BM25_B"]},
              "environment": {"python": platform.python_version(), "commit": commit, "dirty": dirty,
                              "packages": {name: importlib.metadata.version(name) for name in ["sentence-transformers", "torch", "scikit-learn"]}},
              "model": {"name": "BAAI/bge-small-zh-v1.5", "revision": getattr(model[0].auto_model.config, "_commit_hash", None)},
              "warmup_seconds": warmup, "comparison_seconds": duration, "comparison": comparison,
              "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__), ROOT / "evaluation/metrics.py", ROOT / "tools/knowledge.py", ROOT / "tools/text_splitter.py"]},
              "failure_cases": {mode: [row for row in rows if not row["correct"]] for mode, rows in comparison["cases"].items()}}
    args.output.mkdir(parents=True, exist_ok=True)
    name = "retrieval_comparison_" + now.strftime("%Y%m%d_%H%M%S")
    (args.output / (name + ".json")).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = render_summary(report)
    (args.output / (name + ".txt")).write_text(summary, encoding="utf-8")
    (ROOT / "TXT/RAG检索对比评估结果.txt").write_text(summary, encoding="utf-8")
    print(summary, flush=True)

if __name__ == "__main__":
    main()
