
from pathlib import Path
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tools.text_splitter import split_text

from rank_bm25 import BM25Okapi


CHUNK_SIZE = 300 # 每个知识片目标长度
CHUNK_OVERLAP = 50 # 相邻知识片保留重叠部分
SIMILARITY_THRESHOLD = 0.55 # 相似度阈值
TOP_K = 3 # 向量结果返回数量

BM25_TOP_K = 5 # BM25返回结果数量
BM25_K1 = 1.5 # K1 控制 BM25 对关键词出现频率的敏感程度
BM25_B = 0.75 # B 控制 BM25 对文档长度的归一化程度。

RRF_K = 60 # RRF 的平滑常数
DENSE_CANDIDATE_K = 10 # BGE 最多提供 10 个候选片段。
BM25_CANDIDATE_K = 10 # BM25 最多提供 10 个候选片段。


KNOWLEDGE_DIR = (
    Path(__file__).resolve().parent.parent / "knowledge_base"
)

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")



# RRF融合函数
def reciprocal_rank_fusion(ranked_lists: list[list[int]],k: int = RRF_K,) -> dict[int, float]:
    """根据多个检索结果的排名计算 RRF 分数。"""
    fused_scores = {}

    for ranked_indices in ranked_lists:
        for rank, index in enumerate(ranked_indices, start=1):
            fused_scores[index] = (
                fused_scores.get(index, 0.0)
                + 1.0 / (k + rank)
            )

    return fused_scores

# 将文本转化为二元字符
def tokenize_text(text: str) -> list[str]:
    """将文本转换为连续的二元字符 token。"""
    text = text.lower().strip()

    if len(text) < 2:
        return [text] if text else []

    return [
        text[i:i + 2]
        for i in range(len(text) - 1)
    ]

# 读取txt文本，并切割成知识片（缓存后，每次只要运行一次该函数）
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

# 计算并缓存知识片段的向量
@lru_cache(maxsize=1)
def get_knowledge_vectors():
    """计算并缓存知识片段的向量。"""
    knowledge = load_knowledge()

    if not knowledge:
        return None

    texts = [item["content"] for item in knowledge]

    return model.encode(
        texts,
        normalize_embeddings=True,
    )


# 建立并缓存 BM25 索引
@lru_cache(maxsize=1)
def get_bm25_index():
    """建立并缓存 BM25 索引。"""
    knowledge = load_knowledge()

    if not knowledge:
        return None

    tokenized_texts = [
        tokenize_text(item["content"])
        for item in knowledge
    ]

    return BM25Okapi(
        tokenized_texts,
        k1=BM25_K1,
        b=BM25_B,
    )

# 使用 BM25 检索相关知识片段
def bm25_search(query: str,top_k: int = BM25_TOP_K,) -> list[dict]:
    """使用 BM25 检索相关知识片段。"""
    knowledge = load_knowledge()
    bm25 = get_bm25_index()

    if not knowledge or bm25 is None:
        return []

    query_tokens = tokenize_text(query)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    ranked_indices = scores.argsort()[::-1]# 排序

    results = []
    seen_contents = set()

    for index in ranked_indices:
        score = float(scores[index])

        # 没有有效匹配时停止
        if score <= 0:
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

        if len(results) >= top_k:
            break

    return results


# 使用 BGE + BM25 + RRF 进行混合检索。
def knowledge_search(query: str) -> list[dict]:
    """使用 BGE + BM25 + RRF 进行混合检索。"""
    knowledge = load_knowledge()
    knowledge_vectors = get_knowledge_vectors()
    bm25 = get_bm25_index()

    if not knowledge:
        return []

    # ---------- 1. BGE 向量检索 ----------
    dense_scores = {}

    if knowledge_vectors is not None:
        query_vector = model.encode(
            [query],
            normalize_embeddings=True,
        )

        scores = cosine_similarity(
            query_vector,
            knowledge_vectors,
        )[0]

        dense_ranked = scores.argsort()[::-1][
            :DENSE_CANDIDATE_K
        ].tolist()

        for index, score in enumerate(scores):
            dense_scores[index] = float(score)
    else:
        dense_ranked = []

    # ---------- 2. BM25 词法检索 ----------
    bm25_scores = {}

    if bm25 is not None:
        query_tokens = tokenize_text(query)

        if query_tokens:
            scores = bm25.get_scores(query_tokens)

            bm25_ranked = scores.argsort()[::-1][
                :BM25_CANDIDATE_K
            ].tolist()

            for index, score in enumerate(scores):
                bm25_scores[index] = float(score)

            bm25_ranked = [
                index
                for index in bm25_ranked
                if bm25_scores[index] > 0
            ]
        else:
            bm25_ranked = []
    else:
        bm25_ranked = []

    # ---------- 3. RRF 排名融合 ----------
    fused_scores = reciprocal_rank_fusion(
        [dense_ranked, bm25_ranked]
    )

    # ---------- 4. 候选过滤与结果整理 ----------
    results = []
    seen_contents = set()

    ranked_indices = sorted(
        fused_scores,
        key=lambda index: fused_scores[index],
        reverse=True,
    )

    for index in ranked_indices:
        dense_score = dense_scores.get(index, 0.0)
        bm25_score = bm25_scores.get(index, 0.0)

        # 暂时保留原有的向量阈值；
        # BM25 阈值是初始实验参数，后续需要根据实际结果调整。
        if dense_score < SIMILARITY_THRESHOLD:

            continue

        content = knowledge[index]["content"].strip()

        if content in seen_contents:
            continue

        seen_contents.add(content)

        results.append({
            "content": content,
            "source": knowledge[index]["source"],
            "score": round(fused_scores[index], 6),
            "dense_score": round(dense_score, 4),
            "bm25_score": round(bm25_score, 4),
        })

        if len(results) >= TOP_K:
            break

    return results

# 检索知识库并整理为供 LLM 使用的证据
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
