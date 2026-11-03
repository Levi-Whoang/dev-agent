"""检索评估口径与成对差异测试。"""
from evaluation.metrics import score_case, summarize, compare

# 来源匹配不等于证据覆盖；多个片段可覆盖标准词，但单片段证据排名为零。
def test_evidence_and_source_metrics():
    case = {"expected": ["Python", "RAG"], "sources": ["a.txt"]}
    result = score_case(case, [{"source": "a.txt", "content": "Python"}, {"source": "a.txt", "content": "RAG"}])
    assert result["correct"]
    assert result["source_rr"] == 1
    assert result["evidence_rr"] == 0
    assert not score_case(case, [{"source": "wrong.txt", "content": "Python RAG"}])["correct"]

# 无答案误召回分母只包含负例；缺失一类时返回空值而非伪造百分比。
def test_negative_denominator():
    positive = score_case({"expected": ["x"]}, [{"source": "a", "content": "x"}])
    negative = score_case({"expected": None}, [{"source": "a", "content": "x"}])
    rows = [{**positive, "answerable": True, "category": "p"}, {**negative, "answerable": False, "category": "n"}]
    metrics = summarize(rows)
    assert metrics["evidence_hit_at_k"] == 1
    assert metrics["false_positive_rate"] == 1
    assert summarize([])["false_positive_rate"] is None

# 同一批样例同时识别混合检索的正例补回及负例退化。
def test_paired_changes():
    cases = [{"query": "yes", "expected": ["x"]}, {"query": "no", "expected": None}]
    def search(query, mode):
        return [{"source": "a", "content": "x"}] if mode == "hybrid" else []
    report = compare(cases, search)
    assert [r["change"] for r in report["differences"]] == ["improved", "regressed"]
    assert all("content" not in r for row in report["cases"]["hybrid"] for r in row["retrieved"])
