import pytest

from tools.text_splitter import split_text


def test_empty_text():
    assert split_text("") == []
    assert split_text("   ") == []


def test_short_text():
    assert split_text("你好，Agent！") == ["你好，Agent！"]


def test_long_text_is_split():
    text = "A" * 700

    chunks = split_text(
        text,
        chunk_size=300,
        overlap=50,
    )

    assert len(chunks) == 3
    assert all(len(chunk) <= 300 for chunk in chunks)


def test_adjacent_chunks_overlap():
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10

    chunks = split_text(
        text,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) > 1
    assert chunks[0][-20:] == chunks[1][:20]


def test_prefers_sentence_boundaries_when_splitting_chinese_text():
    text = "第一句结束。第二句结束。第三句结束。"

    chunks = split_text(text, chunk_size=10, overlap=0)

    assert "".join(chunks) == text
    assert all(len(chunk) <= 10 for chunk in chunks)
    assert all(chunk.endswith("。") for chunk in chunks[:-1])


def test_prefers_whitespace_without_splitting_english_words():
    chunks = split_text("alpha beta gamma delta", chunk_size=12, overlap=0)

    assert chunks == ["alpha beta", "gamma delta"]


def test_long_sentence_falls_back_to_bounded_character_chunks():
    chunks = split_text("甲" * 35, chunk_size=15, overlap=3)

    assert len(chunks) == 3
    assert all(0 < len(chunk) <= 15 for chunk in chunks)
    assert chunks[0][-3:] == chunks[1][:3]


def test_invalid_parameters():
    with pytest.raises(ValueError):
        split_text("hello", chunk_size=0)

    with pytest.raises(ValueError):
        split_text("hello", chunk_size=100, overlap=100)
