"""把文本切成适合检索的片段（RAG 的最小单元）。

三级递归切分，优先级从粗到细：

    段落 (\\n\\n)  →  句子 (。！？；!?)  →  词/字 (标点 + 定长窗口)

为什么不能只按 300 字硬切：
硬切会从句子中间劈开，"团队规模约五十人，采用远程与现场" 和
"结合的办公方式" 变成两个片段，两边都不完整，向量也变糊。

为什么不是所有段落一刀切完就算：
一段 800 字的文档如果整段进 chunk，会带入大量无关内容，
命中率反而下降——所以超长段落还要往细里拆。

为什么要合并短片段：
"公司介绍（虚构示例）" 这种标题单独成块时，向量几乎不含信息，
检索时既召回不了也干扰排序。短块会被吸附到相邻段落。

最后一级"按词"：中文按词切需要分词词典（jieba 那种几十 MB 的依赖），
收益很小——切点落在词中间时，重叠窗口会把上下文补回来。
所以这一级用「英文按单词、中文按单字」的 token 流 + 定长窗口兜底，
至少保证英文单词不会被切成两半。
"""

from __future__ import annotations

import re

# 句子边界：中文标点之后（且不是连续标点）、英文句点 + 空白、硬换行
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[。！？；…!?])(?![。！？；…!?])"
    r"|(?<=\.)\s+"
    r"|\n+"
)

# 分句边界：逗号、顿号、冒号这类"次级停顿"
_CLAUSE_BOUNDARY = re.compile(r"(?<=[，,、：:；])")

# 词语级切分：英文单词/数字整体、中文单字、空白、其余标点各自成 token。
# 空白必须单独成 token，否则拼回去时英文单词会黏成一串。
_TOKEN = re.compile(r"[A-Za-z0-9_.+\-/]+|[\u4e00-\u9fff]|\s+|[^\sA-Za-z0-9]")


def split_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """定长窗口切分（最底层兜底，也保留给老代码调用）。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须大于等于 0 且小于 chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def split_document(
    text: str,
    max_chars: int = 300,
    overlap: int = 50,
    min_chars: int = 60,
) -> list[str]:
    """文档级三级切分入口。

    max_chars: 单个片段的目标上限
    overlap:   片段之间保留的上下文预算（按整句回退，不会切半句）
    min_chars: 小于这个长度的块尝试并入相邻段，避免标题独立成块
    """
    if max_chars <= 0:
        raise ValueError("max_chars 必须大于 0")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap 必须大于等于 0 且小于 max_chars")
    if min_chars < 0:
        raise ValueError("min_chars 不能为负")

    text = text.strip()
    if not text:
        return []

    paragraphs = _split_paragraphs(text)
    blocks = _attach_short_blocks(paragraphs, min_chars=min_chars, max_chars=max_chars)

    chunks: list[str] = []
    for block in blocks:
        chunks.extend(_split_block(block, max_chars=max_chars, overlap=overlap))

    # 不做全局去重：知识库里重复出现的表格行、固定条款都是有效内容，
    # 按内容去重会把它们误删
    return [chunk.strip() for chunk in chunks if chunk.strip()]


# --------------------------------------------------------------------------
# 一级：段落
# --------------------------------------------------------------------------
def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs or [text.strip()]


def _attach_short_blocks(
    paragraphs: list[str], min_chars: int, max_chars: int
) -> list[str]:
    """把过短的段落（标题、单句）并到相邻段，避免产生无信息量的碎片。"""
    blocks: list[str] = []
    pending: str | None = None

    for paragraph in paragraphs:
        if pending is not None:
            # 短块在前，正文在后；用换行连接保持可读
            blocks.append(f"{pending}\n{paragraph}")
            pending = None
            continue

        if len(paragraph) < min_chars:
            pending = paragraph
            continue

        blocks.append(paragraph)

    if pending is not None:
        if blocks and len(blocks[-1]) + len(pending) + 1 <= max_chars:
            blocks[-1] = f"{blocks[-1]}\n{pending}"
        else:
            blocks.append(pending)

    return blocks


# --------------------------------------------------------------------------
# 二级：句子
# --------------------------------------------------------------------------
def _split_block(block: str, max_chars: int, overlap: int) -> list[str]:
    if len(block) <= max_chars:
        return [block]

    units: list[str] = []
    for sentence in _split_sentences(block):
        if len(sentence) <= max_chars:
            units.append(sentence)
        else:
            # 三级：单句还是太长 → 分句 → 词语
            units.extend(_split_long_sentence(sentence, max_chars, overlap))

    return _merge_units(units, max_chars=max_chars, overlap=overlap)


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_BOUNDARY.split(text)
    return [part.strip() for part in parts if part and part.strip()]


# --------------------------------------------------------------------------
# 三级：分句 → 词语
# --------------------------------------------------------------------------
def _split_long_sentence(sentence: str, max_chars: int, overlap: int) -> list[str]:
    units: list[str] = []

    for clause in _CLAUSE_BOUNDARY.split(sentence):
        clause = clause.strip()
        if not clause:
            continue
        if len(clause) <= max_chars:
            units.append(clause)
        else:
            units.extend(_split_tokens(clause, max_chars, overlap))

    return units or [sentence]


def _split_tokens(text: str, max_chars: int, overlap: int) -> list[str]:
    """词语级兜底：英文整词、中文单字，按窗口累积并保留尾部重叠。"""
    tokens = _TOKEN.findall(text)
    if not tokens:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for token in tokens:
        # 单个 token 就超过上限（比如一长串没有空格和标点的 ID），
        # 只能在它内部硬切，否则这一个 token 会撑爆整个 chunk
        if len(token) > max_chars:
            if current:
                chunks.append("".join(current).strip())
                current = []
                length = 0
            chunks.extend(split_text(token, chunk_size=max_chars, overlap=overlap))
            continue

        if current and length + len(token) > max_chars:
            chunks.append("".join(current).strip())

            # 回退尾部若干 token 作为下一篇的上下文
            keep: list[str] = []
            kept = 0
            for prev in reversed(current):
                if kept + len(prev) > overlap:
                    break
                keep.insert(0, prev)
                kept += len(prev)

            current = keep
            length = kept

        current.append(token)
        length += len(token)

    if current:
        chunks.append("".join(current).strip())

    return [chunk for chunk in chunks if chunk]


# --------------------------------------------------------------------------
# 合并：把短句攒成接近 max_chars 的块，并按整句回退做重叠
# --------------------------------------------------------------------------
def _merge_units(units: list[str], max_chars: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    index = 0

    while index < len(units):
        current: list[str] = []
        length = 0
        cursor = index

        while cursor < len(units):
            piece_len = len(units[cursor])
            if current and length + piece_len > max_chars:
                break
            current.append(units[cursor])
            length += piece_len
            cursor += 1

        if not current:
            break

        chunks.append(_smart_join(current))

        if cursor >= len(units):
            break

        # 回退整句，且保证至少前进 1 个 unit，否则死循环
        back = 0
        kept = 0
        for prev in reversed(current):
            if kept + len(prev) > overlap:
                break
            kept += len(prev)
            back += 1

        next_index = cursor - back
        index = next_index if next_index > index else index + 1

    return chunks


def _smart_join(parts: list[str]) -> str:
    """拼接片段：中文直接接，英文单词之间补一个空格。"""
    result = ""
    for part in parts:
        if not part:
            continue
        if result and _needs_space(result[-1], part[0]):
            result += " "
        result += part
    return result.strip()


def _needs_space(left: str, right: str) -> bool:
    return left.isascii() and left.isalnum() and right.isascii() and right.isalnum()
