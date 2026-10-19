
from pathlib import Path
from functools import lru_cache

from tools.text_splitter import split_text


CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
SIMILARITY_THRESHOLD = 0.55
TOP_K = 3

KNOWLEDGE_DIR = (
    Path(__file__).resolve().parent.parent / "knowledge_base"
)

@lru_cache(maxsize=1)
def get_embedding_model():
    """首次执行向量编码时再加载模型，并在进程内复用。"""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("BAAI/bge-small-zh-v1.5")


@lru_cache(maxsize=1)
def load_knowledge() -> tuple[dict, ...]:
    """读取所有 TXT 文件，并将长文本切分成知识片段。"""
    if not KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(
            f"知识库目录不存在：{KNOWLEDGE_DIR}"
        )

    chunks = []

    for file_path in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8")

        # 将标题和正文保留在同一段中，避免标题独立成为知识块
        content = content.strip()

        if not content:
            continue

        text_chunks = split_text(
            content,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        for text_chunk in text_chunks:
            chunks.append({
                "content": text_chunk,
                "source": file_path.name,
            })

    return tuple(chunks)

@lru_cache(maxsize=1)
def get_knowledge_vectors():
    """计算并缓存知识片段的向量。"""
    knowledge = load_knowledge()

    if not knowledge:
        return None

    texts = [item["content"] for item in knowledge]

    return get_embedding_model().encode(
        texts,
        normalize_embeddings=True,
    )


def knowledge_search(query: str) -> list[dict]:
    """检索知识库，返回文本、来源和相似度分数。"""
    knowledge = load_knowledge()
    knowledge_vectors = get_knowledge_vectors()

    if not knowledge or knowledge_vectors is None:
        return []

    query_vector = get_embedding_model().encode(
        [query],
        normalize_embeddings=True,
    )

    from sklearn.metrics.pairwise import cosine_similarity

    scores = cosine_similarity(
        query_vector,
        knowledge_vectors,
    )[0]

    threshold = SIMILARITY_THRESHOLD
    ranked_indices = scores.argsort()[::-1]

    results = []
    seen_contents = set()

    for index in ranked_indices:
        score = float(scores[index])

        if score < threshold:
            break

        content = knowledge[index]["content"].strip()

        if content in seen_contents:
            continue

        seen_contents.add(content)

        results.append({
            "content": content,
            "source": knowledge[index]["source"],
            "score": round(score, 4),
        })

        if len(results) >= TOP_K:
            break

    return results


def knowledge_search_context(query: str) -> str:
    """检索知识库并整理为供 LLM 使用的证据。"""
    results = knowledge_search(query)

    if not results:
        return "NO_RELEVANT_EVIDENCE"

    evidence = []

    for item in results:
        evidence.append(
            f"来源：{item['source']}\n"
            f"证据：{item['content']}\n"
            f"相似度：{item['score']}"
        )

    return "\n\n".join(evidence)
