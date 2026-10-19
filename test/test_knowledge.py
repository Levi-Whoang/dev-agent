import sys
from types import ModuleType

import numpy as np
import pytest

import tools.knowledge as knowledge


@pytest.fixture(autouse=True)
def clear_model_and_document_caches():
    knowledge.get_embedding_model.cache_clear()
    knowledge.load_knowledge.cache_clear()
    knowledge.get_knowledge_vectors.cache_clear()
    yield
    knowledge.get_embedding_model.cache_clear()
    knowledge.load_knowledge.cache_clear()
    knowledge.get_knowledge_vectors.cache_clear()


def test_load_knowledge_reads_txt_files_and_preserves_sources(tmp_path, monkeypatch):
    (tmp_path / "b.txt").write_text("  第二份知识  ", encoding="utf-8")
    (tmp_path / "a.txt").write_text("第一份知识", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")
    (tmp_path / "ignored.md").write_text("忽略此文件", encoding="utf-8")
    monkeypatch.setattr(knowledge, "KNOWLEDGE_DIR", tmp_path)

    chunks = knowledge.load_knowledge()

    assert chunks == (
        {"content": "第一份知识", "source": "a.txt"},
        {"content": "第二份知识", "source": "b.txt"},
    )


def test_missing_knowledge_directory_raises_clear_error(tmp_path, monkeypatch):
    missing_dir = tmp_path / "not-created"
    monkeypatch.setattr(knowledge, "KNOWLEDGE_DIR", missing_dir)

    with pytest.raises(FileNotFoundError, match="知识库目录不存在"):
        knowledge.load_knowledge()


def test_empty_knowledge_returns_without_loading_embedding_model(tmp_path, monkeypatch):
    monkeypatch.setattr(knowledge, "KNOWLEDGE_DIR", tmp_path)

    def unexpected_model_load():
        raise AssertionError("空知识库不应加载 embedding 模型")

    monkeypatch.setattr(knowledge, "get_embedding_model", unexpected_model_load)

    assert knowledge.get_knowledge_vectors() is None
    assert knowledge.knowledge_search("任意问题") == []


def test_embedding_model_is_loaded_once_on_first_request(monkeypatch):
    constructor_calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_name):
            constructor_calls.append(model_name)

    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    assert constructor_calls == []
    first = knowledge.get_embedding_model()
    second = knowledge.get_embedding_model()

    assert first is second
    assert len(constructor_calls) == 1
    assert constructor_calls[0] == "BAAI/bge-small-zh-v1.5"


def test_search_ranks_results_filters_low_scores_deduplicates_and_caps(
    monkeypatch,
):
    items = (
        {"content": "最佳匹配", "source": "best.txt"},
        {"content": "次佳匹配", "source": "near.txt"},
        {"content": "最佳匹配", "source": "duplicate.txt"},
        {"content": "低相关片段", "source": "weak.txt"},
    )
    vectors = np.array([
        [1.0, 0.0],
        [0.8, 0.6],
        [0.9, 0.1],
        [0.0, 1.0],
    ])

    class FakeEmbeddingModel:
        def encode(self, texts, normalize_embeddings):
            assert texts == ["测试查询"]
            assert normalize_embeddings is True
            return np.array([[1.0, 0.0]])

    monkeypatch.setattr(knowledge, "load_knowledge", lambda: items)
    monkeypatch.setattr(knowledge, "get_knowledge_vectors", lambda: vectors)
    monkeypatch.setattr(knowledge, "get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr(knowledge, "SIMILARITY_THRESHOLD", 0.5)
    monkeypatch.setattr(knowledge, "TOP_K", 3)

    results = knowledge.knowledge_search("测试查询")

    assert [item["content"] for item in results] == ["最佳匹配", "次佳匹配"]
    assert results[0]["source"] == "best.txt"
    assert results[0]["score"] == 1.0
    assert 0.5 <= results[1]["score"] < results[0]["score"]
    assert len(results) <= 3


def test_search_context_formats_evidence_and_empty_result(monkeypatch):
    monkeypatch.setattr(
        knowledge,
        "knowledge_search",
        lambda query: [
            {"content": "公司在深圳。", "source": "company.txt", "score": 0.91}
        ],
    )

    context = knowledge.knowledge_search_context("公司在哪里？")

    assert "来源：company.txt" in context
    assert "证据：公司在深圳。" in context
    assert "相似度：0.91" in context

    monkeypatch.setattr(knowledge, "knowledge_search", lambda query: [])
    assert knowledge.knowledge_search_context("没有答案的问题") == "NO_RELEVANT_EVIDENCE"
