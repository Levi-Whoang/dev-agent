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


def test_invalid_parameters():
    with pytest.raises(ValueError):
        split_text("hello", chunk_size=0)

    with pytest.raises(ValueError):
        split_text("hello", chunk_size=100, overlap=100)
