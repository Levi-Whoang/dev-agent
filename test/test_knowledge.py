"""混合检索的行为测试。

这些测试依赖真实向量模型（首次运行会加载本地缓存的 bge-small-zh）。
知识库目录不存在时自动跳过——knowledge_base/ 在 .gitignore 里，
换台机器没跑过 `scripts/generate_knowladge_base.py` 就没有数据。
"""

import pytest

from rag.bm25 import BM25Index, tokenize
from rag.hybrid import rank_indices, reciprocal_rank_fusion


def test_tokenize_mixed_language():
    tokens = tokenize("Python 类型标注")

    assert "python" in tokens
    assert "类型" in tokens
    assert "标注" in tokens


def test_tokenize_produces_bigrams():
    # 4 个汉字 → 3 个二元组
    assert tokenize("深圳市南") == ["深圳", "圳市", "市南"]


def test_bm25_ranks_matching_document_first():
    docs = ["订单 10086 已支付", "今天天气不错", "公司地址在深圳"]
    index = BM25Index(docs)

    scores = index.get_scores("订单 10086")

    assert scores[0] == max(scores)
    assert index.top_n("订单 10086", 1)[0][0] == 0


def test_bm25_no_overlap_gives_zero():
    index = BM25Index(["完全无关的内容"])
    assert index.get_scores("量子力学") == [0.0]


def test_rrf_prefers_document_ranked_high_by_both():
    dense = [3, 1, 2]
    sparse = [3, 2, 5]

    fused = dict(reciprocal_rank_fusion([dense, sparse], k=60))

    # 3 号被两个检索器都排第一，融合分必须最高
    assert max(fused, key=fused.get) == 3


def test_rrf_orders_descending():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1]], k=60)
    scores = [score for _, score in fused]

    assert scores == sorted(scores, reverse=True)


def test_rank_indices_respects_limit():
    assert rank_indices([0.1, 0.9, 0.5], limit=2) == [1, 2]


# ---------- 端到端检索 ----------
@pytest.fixture(scope="module")
def retriever(knowledge_dir):
    if not knowledge_dir.exists() or not any(knowledge_dir.glob("*.txt")):
        pytest.skip("knowledge_base 为空，先运行 scripts/generate_knowladge_base.py")

    from rag.retriever import KnowledgeRetriever

    return KnowledgeRetriever()


async def test_retrieve_job_requirements(retriever):
    results = await retriever.search("这个实习岗位需要掌握哪些技术？")

    assert results
    assert any("Python" in item.content for item in results)


async def test_retrieve_company_location(retriever):
    results = await retriever.search("公司在哪里？")

    assert results
    assert any("深圳市南山区" in item.content for item in results)


async def test_result_format(retriever):
    results = await retriever.search("岗位要求是什么？")

    assert results
    for item in results:
        payload = item.to_dict()
        assert isinstance(payload["content"], str)
        assert isinstance(payload["score"], float)
        assert 0.0 <= payload["score"] <= 1.0
        assert payload["matched_by"] in {"dense", "sparse", "both"}


async def test_unrelated_question_returns_nothing(retriever):
    """知识库里没有的话题必须返回空，否则拒答逻辑就失效了。"""
    results = await retriever.search("今天股市大盘怎么样？")

    assert results == []


async def test_search_context_refusal_protocol(retriever):
    context = await retriever.search_context("今天股市大盘怎么样？")
    assert context == "NO_RELEVANT_EVIDENCE"


async def test_search_context_contains_source(retriever):
    context = await retriever.search_context("公司在哪里？")

    assert "来源：" in context
    assert "命中方式：" in context


# ---------- 消融开关（mode / apply_gate） ----------
# 这两个参数是评测用的，但它们直接改生产检索路径，所以必须有测试兜住：
# 一旦 mode 拼错却没报错，评测脚本会静默测出假的"混合检索"成绩。
async def test_dense_mode_only_uses_vectors(retriever):
    results = await retriever.search("公司在哪里？", mode="dense")

    assert results
    # 只跑向量这一路，结果不可能被标成稀疏命中
    assert all(item.matched_by == "dense" for item in results)
    assert all(item.sparse_rank is None for item in results)


async def test_sparse_mode_only_uses_keywords(retriever):
    results = await retriever.search("Python 类型标注", mode="sparse")

    assert results
    assert all(item.matched_by == "sparse" for item in results)
    assert all(item.dense_rank is None for item in results)


async def test_hybrid_mode_marks_matches(retriever):
    results = await retriever.search("公司在哪里？", mode="hybrid")

    assert results
    assert {item.matched_by for item in results} <= {"dense", "sparse", "both"}


async def test_unknown_mode_is_rejected(retriever):
    """拼错模式名必须炸，不能悄悄退化成某一种检索。"""
    with pytest.raises(ValueError):
        await retriever.search("公司在哪里？", mode="hybird")


async def test_gate_off_returns_more_for_unanswerable(retriever):
    """关掉门槛后，本应拒答的问题不该再返回空 —— 这是消融实验的前提。"""
    gated = await retriever.search("报销流程是什么？", top_k=10)
    ungated = await retriever.search("报销流程是什么？", top_k=10, apply_gate=False)

    assert len(ungated) >= len(gated)
