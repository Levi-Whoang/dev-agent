from pathlib import Path

from test.retrieval_dataset import EVALUATION_CASES


KNOWLEDGE_BASE = Path(__file__).resolve().parent.parent / "knowledge_base"


def test_retrieval_evaluation_cases_have_valid_evidence_annotations():
    assert len(EVALUATION_CASES) >= 40
    assert sum(case["expected"] is None for case in EVALUATION_CASES) >= 8

    for case in EVALUATION_CASES:
        if case["expected"] is None:
            assert case.get("category") == "no_answer"
            continue

        sources = case.get("sources")
        assert sources, f"正例必须标注来源文件：{case['query']}"
        source_text = "\n".join(
            (KNOWLEDGE_BASE / source).read_text(encoding="utf-8")
            for source in sources
        )
        assert all(
            term in source_text for term in case["expected"]
        ), f"标准证据词必须存在于标注来源：{case['query']}"

