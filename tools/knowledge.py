
from pathlib import Path
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_FILE = (
    Path(__file__).resolve().parent.parent
    / "knowledge_base"
    / "company.txt"
)

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")


@lru_cache(maxsize=1)
def load_knowledge() -> tuple[str, ...]:
    """加载并缓存知识库文本。"""
    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(f"知识库文件不存在：{KNOWLEDGE_FILE}")

    content = KNOWLEDGE_FILE.read_text(encoding="utf-8")
    chunks = tuple(
        item.strip()
        for item in content.split("\n\n")
        if item.strip()
    )
    return chunks


@lru_cache(maxsize=1)
def get_knowledge_vectors():
    """计算并缓存知识库向量。"""
    knowledge = load_knowledge()
    if not knowledge:
        return None

    return model.encode(
        list(knowledge),
        normalize_embeddings=True,
    )


def knowledge_search(query: str) -> list[str]:
    """通过向量相似度检索相关知识。"""
    knowledge = load_knowledge()
    knowledge_vectors = get_knowledge_vectors()

    if not knowledge or knowledge_vectors is None:
        return []

    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
    )

    scores = cosine_similarity(
        query_vector,
        knowledge_vectors,
    )[0]

    top_k = min(3, len(knowledge))
    ranked_indices = scores.argsort()[::-1][:top_k]

    return [knowledge[i] for i in ranked_indices]
