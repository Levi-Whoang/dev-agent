import re


_NATURAL_BOUNDARY = re.compile(r"\n\s*\n|[。！？!?；;]+[”’\"')）\]]*|\n")
_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n")


def _natural_boundaries(text: str, start: int, end: int) -> list[int]:
    return [
        match.end()
        for match in _NATURAL_BOUNDARY.finditer(text, start, end)
        if start < match.end() <= end
    ]


def _choose_end(text: str, start: int, hard_end: int, chunk_size: int) -> int:
    """优先在句子/换行边界切开，避免过早切分导致片段太短。"""
    earliest = start + max(1, int(chunk_size * 0.55))
    boundaries = _natural_boundaries(text, earliest, hard_end)

    if boundaries:
        latest = max(boundaries)
        paragraph_boundaries = [
            match.end()
            for match in _PARAGRAPH_BOUNDARY.finditer(text, earliest, hard_end)
        ]
        if paragraph_boundaries:
            paragraph_end = max(paragraph_boundaries)
            if latest - paragraph_end <= max(20, int(chunk_size * 0.1)):
                return paragraph_end
        return latest

    # 长句或无标点文本优先在空白处断开，中文连续文本最终按字符硬切。
    whitespace_boundaries = [
        earliest + match.end()
        for match in re.finditer(r"\s+", text[earliest:hard_end])
    ]
    return max(whitespace_boundaries, default=hard_end)


def _choose_overlap_start(
    text: str,
    previous_start: int,
    end: int,
    overlap: int,
) -> int:
    target = max(previous_start + 1, end - overlap)
    if overlap == 0:
        return end

    # 重叠起点尽量落在前一句结束处，重叠长度可随句长略有变化。
    lower_bound = max(previous_start + 1, end - overlap * 2)
    candidates = [
        boundary
        for boundary in _natural_boundaries(text, lower_bound, end - 1)
        if lower_bound <= boundary < end
    ]
    if not candidates:
        return target

    return min(candidates, key=lambda boundary: abs(boundary - target))


def split_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[str]:
    """按句子、段落边界切分文本，并在相邻片段间保留上下文重叠。"""
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
        hard_end = min(start + chunk_size, len(text))
        if hard_end == len(text):
            end = hard_end
        else:
            end = _choose_end(text, start, hard_end, chunk_size)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        next_start = _choose_overlap_start(text, start, end, overlap)
        start = max(start + 1, next_start)

    return chunks
