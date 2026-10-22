import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_retrieval import build_report, save_report
from retrieval_dataset import EVALUATION_CASES, EVALUATION_SET_VERSION


def test_retrieval_dataset_has_versioned_answerable_and_distractor_cases():
    answerable = [case for case in EVALUATION_CASES if case["expected"] is not None]
    unanswerable = [case for case in EVALUATION_CASES if case["expected"] is None]
    similar_distractors = [
        case for case in unanswerable
        if case.get("category") == "similar_distractor"
    ]

    assert EVALUATION_SET_VERSION == "rag-eval-2.0"
    assert len(EVALUATION_CASES) == 80
    assert len(answerable) == 50
    assert len(unanswerable) == 30
    assert len(similar_distractors) == 20
    assert all(case.get("sources") for case in answerable)
    assert len({case["query"] for case in EVALUATION_CASES}) == len(EVALUATION_CASES)


def test_report_records_dataset_fingerprint_configs_and_failures(tmp_path):
    failure = {
        "case_id": "R001",
        "failure_type": "false_positive",
        "query": "未被知识库支持的问题",
        "retrieved": [],
    }
    result = {
        "chunk_size": 300,
        "overlap": 50,
        "threshold": 0.6,
        "top_k": 3,
        "failure_cases": [failure],
    }

    report = build_report([result])
    saved_path = save_report(report, tmp_path)
    restored = json.loads(saved_path.read_text(encoding="utf-8"))

    assert restored["dataset"]["version"] == EVALUATION_SET_VERSION
    assert len(restored["dataset"]["sha256"]) == 64
    assert restored["dataset"]["case_count"] == 80
    assert restored["retriever_settings"]["embedding_model"] == "BAAI/bge-small-zh-v1.5"
    assert restored["configurations"]
    assert restored["results"][0]["chunk_size"] == 300
    assert restored["results"][0]["failure_cases"] == [failure]
