
from tools.text_splitter import split_text


def test_splitter_manual():
    text = (
        "第一句话说明 Agent 的基本概念。"
        "第二句话介绍工具调用。"
        "第三句话介绍知识检索。"
    ) * 5

    chunk_size = 30
    overlap = 5

    chunks = split_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    print("原文长度：", len(text))
    print("Chunk 数量：", len(chunks))
    print("原文：", repr(text))
    print("第一个 Chunk：", repr(chunks[0]))
    print("第二个 Chunk：", repr(chunks[1]))

    assert chunks, "切分结果不应为空"

    for i, chunk in enumerate(chunks, start=1):
        print(f"Chunk {i}（{len(chunk)} 字符）: {chunk}")
        assert len(chunk) <= chunk_size, (
            f"Chunk {i} 超长：{len(chunk)} > {chunk_size}"
        )

    # 检查每个 Chunk 是否都来自原文，允许 overlap 重复
    # 逐个检查 Chunk 内容是否能在原文中找到
    search_start = 0

    for i, chunk in enumerate(chunks, start=1):
        # 允许从上一个 Chunk 的末尾向前搜索 overlap
        pos = text.find(chunk, max(0, search_start - overlap))

        assert pos != -1, (
            f"Chunk {i} 无法在原文中找到，可能包含意外拼接的内容：{chunk!r}"
        )

        search_start = max(search_start, pos + len(chunk) - overlap)

    print("Chunk 原文来源检查通过")
    # 检查相邻 Chunk 是否存在重叠
    has_overlap = any(
        chunks[i][-overlap:] in chunks[i + 1]
        for i in range(len(chunks) - 1)
    )

    assert has_overlap, "没有检测到预期的 overlap"

    print("原文保留检查通过")
    print("Overlap 检查通过")

    print("基础长度测试通过")


if __name__ == "__main__":
    test_splitter_manual()



# 短段落	短段落不应该被不必要地拆开
def test_short_paragraph_is_preserved():
    text = "第一段内容。\n\n第二段内容。"

    chunks = split_text(text, chunk_size=30, overlap=5)

    assert chunks == [text]

# 超长句	每块不超长，字符级 overlap 不导致内容缺失
def test_long_sentence_respects_chunk_size():
    text = "A" * 100

    chunks = split_text(text, chunk_size=30, overlap=5)

    assert chunks
    assert all(len(chunk) <= 30 for chunk in chunks)
    assert "".join(
        chunk if i == 0 else chunk[5:]
        for i, chunk in enumerate(chunks)
    ) == text

 # 空文本	空字符串和纯空白输入应返回空列表
def test_empty_text():
    assert split_text("") == []
    assert split_text("   ") == []


# 检查长段落是否丢失内容
def test_long_paragraph_preserves_all_sentences():
    sentences = [
        f"第{i}句介绍 Agent 的功能。"
        for i in range(1, 21)
    ]
    text = "".join(sentences)

    chunks = split_text(text, chunk_size=50, overlap=10)

    assert chunks
    assert all(len(chunk) <= 50 for chunk in chunks)

    # 每个原始句子都必须至少在某个 Chunk 中完整出现
    for sentence in sentences:
        assert any(sentence in chunk for chunk in chunks), (
            f"切分后找不到完整句子：{sentence}"
        )


# 检查 overlap 是否有效
def test_long_sentence_has_expected_overlap():
    text = "A" * 100

    chunks = split_text(text, chunk_size=30, overlap=5)

    assert len(chunks) > 1

    for previous, current in zip(chunks, chunks[1:]):
        assert previous[-5:] == current[:5], (
            "相邻 Chunk 没有保留预期的 5 个字符重叠"
        )



def test_chinese_english_mixed_sentence_boundaries():
    text = (
        "Agent 可以调用 Tool. "
        "RAG retrieves relevant documents. "
        "智能体根据结果生成 Answer！"
    )

    chunks = split_text(text, chunk_size=40, overlap=5)

    assert chunks
    assert all(len(chunk) <= 40 for chunk in chunks)

    # 每个句子都必须完整保留在某个 Chunk 中
    sentences = [
        "Agent 可以调用 Tool. ",
        "RAG retrieves relevant documents. ",
        "智能体根据结果生成 Answer！",
    ]

    for sentence in sentences:
        assert any(sentence in chunk for chunk in chunks), (
            f"句子或其末尾空格被拆散：{sentence!r}"
        )



def test_punctuation_is_preserved():
    text = "你好！Hello?这是测试。End."

    chunks = split_text(text, chunk_size=10, overlap=2)

    assert chunks
    assert all(len(chunk) <= 10 for chunk in chunks)

    # 每个原文标点都必须至少出现在某个 Chunk 中
    punctuation_positions = [
        i for i, char in enumerate(text)
        if char in "！?。."
    ]

    for position in punctuation_positions:
        mark = text[position]

        assert any(mark in chunk for chunk in chunks), (
            f"标点 {mark!r} 没有保留在任何 Chunk 中"
        )
