"""分块策略测试：段落 → 句子 → 词，三级递归。

核心断言不是"切成了几块"，而是：
**切点不能落在句子中间**。块数变了只是参数问题，句子被劈开才是语义损坏。
"""

import re

import pytest

from rag.splitter import split_document, split_text


def test_empty_text():
    assert split_document("") == []
    assert split_document("   \n\n  ") == []
    assert split_text("") == []


def test_short_text_kept_whole():
    assert split_document("你好，Agent！", min_chars=0) == ["你好，Agent！"]


def test_paragraphs_are_kept_separate():
    text = "第一段内容。\n\n第二段内容。"

    chunks = split_document(text, max_chars=100, min_chars=0)

    assert chunks == ["第一段内容。", "第二段内容。"]


def test_short_paragraph_attaches_to_next():
    """标题这类短块不能单独成 chunk，否则向量里几乎没有信息。"""
    text = "公司介绍\n\n这是一段足够长的正文内容，用来验证短块会被吸附到下一段里。"

    chunks = split_document(text, max_chars=300, min_chars=60)

    assert len(chunks) == 1
    assert chunks[0].startswith("公司介绍")
    assert "足够长的正文" in chunks[0]


def test_long_paragraph_splits_at_sentence_end():
    sentences = [f"这是第{i}句话，用来把段落撑长，避免被当成短文本处理。" for i in range(1, 8)]
    text = "".join(sentences)

    chunks = split_document(text, max_chars=80, overlap=20, min_chars=0)

    assert len(chunks) > 1
    assert all(len(chunk) <= 80 for chunk in chunks)
    # 每一块都必须以句末标点收尾 —— 这就是"没有从句子中间切开"
    for chunk in chunks:
        assert chunk.rstrip()[-1] in "。！？；…", chunk


def test_no_content_lost_when_splitting_sentences():
    sentences = [f"第{i}项要求必须写清楚。" for i in range(1, 20)]
    text = "".join(sentences)

    chunks = split_document(text, max_chars=60, overlap=0, min_chars=0)

    rebuilt = "".join(chunks)
    assert rebuilt == text


def test_english_sentences_keep_word_boundaries():
    text = (
        "FastAPI is a modern web framework. It supports async endpoints out of the box. "
        "SQLAlchemy provides an async engine as well. Pydantic validates request bodies."
    )

    chunks = split_document(text, max_chars=90, overlap=0, min_chars=0)

    for chunk in chunks:
        # 块首尾不能是半个单词
        assert not chunk.startswith(" ")
        assert not re.search(r"[A-Za-z]$", chunk) or chunk.endswith((".", "!", "?"))


def test_very_long_sentence_falls_back_to_tokens():
    """没有句号的一整段话（如错误日志）最后一级按词/字兜底。"""
    text = "错误码" + "ABCDEFGHIJ" * 40

    chunks = split_document(text, max_chars=100, overlap=20, min_chars=0)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_token_split_keeps_english_words_intact():
    sentence = "timeout while calling " + " ".join(["endpoint"] * 30)

    chunks = split_document(sentence, max_chars=60, overlap=10, min_chars=0)

    for chunk in chunks:
        for word in chunk.split():
            assert word in {"timeout", "while", "calling", "endpoint"}, chunk


def test_adjacent_chunks_overlap_by_sentences():
    sentences = [f"第{i}条：这是一句用于测试重叠的说明文字。" for i in range(1, 10)]

    chunks = split_document("".join(sentences), max_chars=60, overlap=25, min_chars=0)

    assert len(chunks) > 2
    # 重叠按整句回退：必然有某一句同时出现在相邻两块里
    shared = [s for s in sentences if s in chunks[0] and s in chunks[1]]
    assert shared


def test_no_infinite_loop_with_zero_overlap():
    text = "字" * 2000
    chunks = split_document(text, max_chars=100, overlap=0, min_chars=0)

    assert 15 <= len(chunks) <= 25
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        split_document("hello", max_chars=0)

    with pytest.raises(ValueError):
        split_document("hello", max_chars=100, overlap=100)

    with pytest.raises(ValueError):
        split_document("hello", max_chars=100, min_chars=-1)


def test_split_text_still_works_for_compatibility():
    text = "A" * 700
    chunks = split_text(text, chunk_size=300, overlap=50)

    assert len(chunks) == 3
    assert chunks[0][-50:] == chunks[1][:50]
