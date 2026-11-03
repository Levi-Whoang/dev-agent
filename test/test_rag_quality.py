"""五项指标的证据口径和裁判结果校验测试。"""
import pytest
from evaluation.quality import retrieval_metrics, answer_metrics, aggregate

# Recall 按全部标注单元计算，Precision 固定以 K 为分母。
def test_evidence_metrics():
    case = {"evidence":[{"id":"e1","source":"a","quote":"事实一"},{"id":"e2","source":"b","quote":"事实二"}]}
    result = retrieval_metrics(case,[{"source":"wrong","content":"事实一"},{"source":"a","content":"事实一"}],3)
    assert result["recall_at_k"] == 0.5
    assert result["mrr"] == 0.5
    assert result["precision_at_k"] == 1/3

# 不能把出现拒答词但编造事实的答案算作正确拒答。
def test_refusal_with_unsupported_claim():
    result = answer_metrics({"refused":True,"answered_question":True,"claims":[{"text":"老板姓张","supported":False}]})
    assert result["faithfulness"] == 0
    assert not result["correct_refusal"]

# 纯拒答无事实时忠实度无分母，不能记为满分。
def test_empty_refusal():
    result = answer_metrics({"refused":True,"answered_question":False,"claims":[]})
    assert result["faithfulness"] is None
    assert result["correct_refusal"]
    with pytest.raises(ValueError):
        answer_metrics({"refused":False,"answered_question":True,"claims":[]})

# 生成未运行时准确率留空；生成调用失败在负例拒答分母中计失败。
def test_generation_failure_denominator():
    row={"answerable":False,"retrieved":[],"precision_at_k":0}
    assert aggregate([row])["no_answer_accuracy"] is None
    assert aggregate([{**row,"generation":{"error":"Timeout"}}])["no_answer_accuracy"] == 0
