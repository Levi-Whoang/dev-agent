
import re


def _split_sentences_with_spans(
    text: str,
) -> list[tuple[int, int]]:
    """识别句子在原文中的起止位置，返回左闭右开区间。"""
    spans = []
    start = 0

    # 常见中英文句末标点；英文句号后需要空白或文本结束。
    pattern = re.compile(r"[。！？!?；;]+|[.](?=\s|$)")

    for match in pattern.finditer(text):
        end = match.end()

        # 英文句号后吸收紧随的空白，避免切分后丢失空格。
        while end < len(text) and text[end].isspace():
            end += 1

        if text[start:end].strip():
            spans.append((start, end))

        start = end

    if start < len(text) and text[start:].strip():
        spans.append((start, len(text)))

    return spans


def _split_long_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """对超长文本按原文字符位置切片，保留 overlap。"""
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def _split_long_paragraph(
    paragraph: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """长段落优先按句子切分，超长句按字符切分。"""
    if len(paragraph) <= chunk_size:
        return [paragraph]

    spans = _split_sentences_with_spans(paragraph)
    chunks = []
    current_start = None
    current_end = None

    for sentence_start, sentence_end in spans:
        sentence = paragraph[sentence_start:sentence_end]

        if len(sentence) > chunk_size:
            if current_start is not None:
                chunks.append(paragraph[current_start:current_end])
                current_start = None
                current_end = None

            # 超长句独立切片，不与前一个句子拼接。
            chunks.extend(
                _split_long_text(sentence, chunk_size, overlap)
            )
            continue

        if current_start is None:
            current_start = sentence_start
            current_end = sentence_end
            continue

        candidate_length = sentence_end - current_start

        if candidate_length <= chunk_size:
            current_end = sentence_end
        else:
            chunks.append(paragraph[current_start:current_end])

            # 从上一片原文中选取重叠区域，但只在完整句子边界开始。
            target_start = max(
                current_start,
                current_end - overlap,
            )
            next_start = sentence_start

            if next_start - target_start <= chunk_size:
                current_start = target_start
                current_end = sentence_end
            else:
                current_start = sentence_start
                current_end = sentence_end

    if current_start is not None:
        chunks.append(paragraph[current_start:current_end])

    return chunks


def split_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[str]:
    """
    优先保留完整段落；长段落按句子切分；
    超长句按字符切分，避免拼接时引入额外字符。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须大于等于 0 且小于 chunk_size")

    text = text.strip()

    if not text:
        return []

    # 先按空行划分段落，并保留每个段落在原文中的位置。
    paragraph_spans = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|$)", text, re.DOTALL):
        paragraph_spans.append((match.start(), match.end()))

    chunks = []
    current_start = None
    current_end = None

    for paragraph_start, paragraph_end in paragraph_spans:
        paragraph = text[paragraph_start:paragraph_end]

        if len(paragraph) > chunk_size:
            if current_start is not None:
                chunks.append(text[current_start:current_end])
                current_start = None
                current_end = None

            chunks.extend(
                _split_long_paragraph(paragraph, chunk_size, overlap)
            )
            continue

        if current_start is None:
            current_start = paragraph_start
            current_end = paragraph_end
            continue

        # 两段之间保留原文分隔符，不能额外制造字符。
        candidate_length = paragraph_end - current_start

        if candidate_length <= chunk_size:
            current_end = paragraph_end
        else:
            chunks.append(text[current_start:current_end])
            current_start = paragraph_start
            current_end = paragraph_end

    if current_start is not None:
        chunks.append(text[current_start:current_end])

    # 最终安全检查：不允许任何 Chunk 超出长度限制。
    result = []
    for chunk in chunks:
        if len(chunk) <= chunk_size and chunk.strip():
            result.append(chunk)
        elif chunk.strip():
            result.extend(_split_long_text(chunk, chunk_size, overlap))

    return result
