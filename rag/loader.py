"""知识库加载：读 TXT → 分段 → 缓存。

索引只建一次，之后走内存。文档更新后调用 `reload_chunks()` 让缓存失效。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from rag.splitter import split_document

logger = logging.getLogger("dev-agent.rag")

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
MIN_CHUNK_SIZE = 60


@dataclass(frozen=True, slots=True)
class Chunk:
    index: int
    content: str
    source: str


_cache: dict[str, tuple[float, tuple[Chunk, ...]]] = {}


def dir_signature(directory: Path) -> float:
    """目录里最新的修改时间，用来判断缓存是否失效。"""
    latest = 0.0
    for path in directory.glob("*.txt"):
        latest = max(latest, path.stat().st_mtime)
    return latest


def load_chunks(directory: Path | str) -> tuple[Chunk, ...]:
    directory = Path(directory)

    if not directory.exists():
        logger.warning("知识库目录不存在：%s", directory)
        return ()

    signature = dir_signature(directory)
    cached = _cache.get(str(directory))
    if cached and cached[0] == signature:
        return cached[1]

    chunks: list[Chunk] = []
    for file_path in sorted(directory.glob("*.txt")):
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # 单个坏文件不该让整个知识库起不来
            logger.warning("跳过非 UTF-8 文件：%s", file_path.name)
            continue

        for piece in split_document(
            content,
            max_chars=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            min_chars=MIN_CHUNK_SIZE,
        ):
            chunks.append(
                Chunk(index=len(chunks), content=piece, source=file_path.name)
            )

    result = tuple(chunks)
    _cache[str(directory)] = (signature, result)
    logger.info("知识库加载完成：%d 个片段，来自 %s", len(result), directory)
    return result


def reload_chunks(directory: Path | str) -> tuple[Chunk, ...]:
    _cache.pop(str(Path(directory)), None)
    return load_chunks(directory)
