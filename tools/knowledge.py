
import math
import json
import os
import re
from contextlib import contextmanager
from collections import Counter
from contextvars import ContextVar
from pathlib import Path
from functools import lru_cache

from tools.text_splitter import split_text


CHUNK_SIZE = 300 # 每个知识片段的目标长度
CHUNK_OVERLAP = 50 # 相邻片段的重叠长度
SIMILARITY_THRESHOLD = 0.60 # 向量相似度阈值
TOP_K = 3 # 最终返回数量上限
DENSE_WEIGHT = 0.7 # 向量分数权重
KEYWORD_WEIGHT = 0.3 # 关键词分数权重
BM25_K1 = 1.5 # BM25 参数
BM25_B = 0.75 # BM25 参数
KEYWORD_MIN_COVERAGE = 0.4 # 关键词覆盖率门槛

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.IGNORECASE)
_KEYWORD_STOP_WORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "how", "in",
    "is", "of", "or", "the", "this", "to", "what", "when", "where", "which",
    "who", "why", "公司", "企业", "岗位", "实习", "具体", "目前", "这个", "哪些",
    "如何", "怎么", "怎样", "什么", "哪里", "在哪", "哪儿", "多少", "是否", "需要",
    "应该", "可以", "吗", "呢", "的", "有", "是", "和", "及", "员工", "老板",
    "姓氏", "范围", "今年", "每周", "几天",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_knowledge_dir = Path(os.getenv("KNOWLEDGE_DIR", PROJECT_ROOT / "knowledge_base"))
KNOWLEDGE_DIR = (
    _knowledge_dir if _knowledge_dir.is_absolute() else PROJECT_ROOT / _knowledge_dir
)
_CURRENT_KNOWLEDGE_OWNER: ContextVar[int | None] = ContextVar(
    "current_knowledge_owner", default=None
)


# 设置当前聊天的知识库文档所有者范围。
@contextmanager
def knowledge_owner_scope(owner_user_id: int | None):
    """在一次聊天调用期间限制文档检索的用户范围。"""
    token = _CURRENT_KNOWLEDGE_OWNER.set(owner_user_id)
    try:
        yield
    finally:
        _CURRENT_KNOWLEDGE_OWNER.reset(token)

# 延迟加载并复用文本向量模型。
@lru_cache(maxsize=1)
def get_embedding_model():
    """首次执行向量编码时再加载模型，并在进程内复用。"""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("BAAI/bge-small-zh-v1.5")


# 加载公共语料和指定用户文档并切分文本。
@lru_cache(maxsize=128)
def load_knowledge(owner_user_id: int | None = None) -> tuple[dict, ...]:
    """读取公共语料和当前用户的上传文档，并切分成知识片段。"""
    if not KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(
            f"知识库目录不存在：{KNOWLEDGE_DIR}"
        )

    chunks = []

    file_paths = list(KNOWLEDGE_DIR.glob("*.txt"))
    if owner_user_id is not None:
        file_paths.extend(
            (KNOWLEDGE_DIR / "documents" / str(owner_user_id)).glob("*.txt")
        )
    for file_path in sorted(file_paths):
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
                "source": (
                    file_path.stem.split("--", 1)[1]
                    if file_path.parent.parent.name == "documents" and "--" in file_path.stem
                    else file_path.name
                ),
            })

    return tuple(chunks)

# 获取知识片段的缓存向量。
@lru_cache(maxsize=128)
def get_knowledge_vectors(owner_user_id: int | None = None):
    """计算并缓存知识片段的向量。"""
    knowledge = (
        load_knowledge()
        if owner_user_id is None
        else load_knowledge(owner_user_id)
    )

    if not knowledge:
        return None

    texts = [item["content"] for item in knowledge]

    return get_embedding_model().encode(
        texts,
        normalize_embeddings=True,
    )


# 将英文和中文文本转换为关键词片段。
def _tokenize_for_keyword(text: str) -> list[str]:
    """保留英文词，并将连续中文拆成重叠的二元/三元字片段。"""
    tokens = []

    for match in _TOKEN_PATTERN.finditer(text.lower()):
        segment = match.group(0)

        if segment[0].isascii():
            tokens.append(segment)
            continue

        if len(segment) == 1:
            tokens.append(segment)
            continue

        for size in (2, 3):
            tokens.extend(
                segment[index:index + size]
                for index in range(len(segment) - size + 1)
            )

    return tokens


# 构建并缓存关键词 BM25 索引。
@lru_cache(maxsize=128)
def _get_keyword_index(owner_user_id: int | None = None):
    """缓存 BM25 所需的分词结果、文档频率和平均文档长度。"""
    knowledge = (
        load_knowledge()
        if owner_user_id is None
        else load_knowledge(owner_user_id)
    )
    document_terms = tuple(
        Counter(_tokenize_for_keyword(item["content"]))
        for item in knowledge
    )

    document_frequency = Counter()
    for terms in document_terms:
        document_frequency.update(terms.keys())

    average_length = (
        sum(sum(terms.values()) for terms in document_terms) / len(document_terms)
        if document_terms
        else 0.0
    )

    return document_terms, document_frequency, average_length


# 提取查询中的有效关键词并移除通用词。
def _keyword_query_terms(query: str) -> set[str]:
    """移除通用词和疑问词，避免它们单独触发关键词召回。"""
    return {
        term
        for term in _tokenize_for_keyword(query)
        if term not in _KEYWORD_STOP_WORDS
    }


# 计算每个知识片段对查询关键词的覆盖率。
def _keyword_coverages(query: str, owner_user_id: int | None = None) -> list[float]:
    """计算每个片段覆盖了多少个有效查询词。"""
    document_terms, _, _ = (
        _get_keyword_index()
        if owner_user_id is None
        else _get_keyword_index(owner_user_id)
    )
    query_terms = _keyword_query_terms(query)

    if not query_terms:
        return [0.0] * len(document_terms)

    return [
        sum(term in terms for term in query_terms) / len(query_terms)
        for terms in document_terms
    ]


# 计算查询词对知识片段的 BM25 分数。
def _bm25_scores(query: str, owner_user_id: int | None = None) -> list[float]:
    """为每个知识片段计算 BM25 关键词相关度。"""
    document_terms, document_frequency, average_length = (
        _get_keyword_index()
        if owner_user_id is None
        else _get_keyword_index(owner_user_id)
    )

    if not document_terms or average_length == 0:
        return []

    query_terms = _keyword_query_terms(query)
    if not query_terms:
        return [0.0] * len(document_terms)

    document_count = len(document_terms)
    scores = []

    for terms in document_terms:
        document_length = sum(terms.values())
        score = 0.0

        for term in query_terms:
            term_frequency = terms.get(term, 0)
            if not term_frequency:
                continue

            frequency = document_frequency[term]
            inverse_document_frequency = math.log(
                1 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            length_normalizer = BM25_K1 * (
                1 - BM25_B + BM25_B * document_length / average_length
            )
            score += inverse_document_frequency * (
                term_frequency * (BM25_K1 + 1)
                / (term_frequency + length_normalizer)
            )

        scores.append(score)

    return scores


# 融合向量分和关键词分检索相关知识证据。
def knowledge_search(query: str) -> list[dict]:
    """融合向量相似度和 BM25 关键词分，返回排序后的证据片段。"""
    owner_user_id = _CURRENT_KNOWLEDGE_OWNER.get()
    if owner_user_id is None:
        knowledge = load_knowledge()
        knowledge_vectors = get_knowledge_vectors()
    else:
        knowledge = load_knowledge(owner_user_id)
        knowledge_vectors = get_knowledge_vectors(owner_user_id)

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

    keyword_scores = _bm25_scores(query, owner_user_id)
    keyword_coverages = _keyword_coverages(query, owner_user_id)
    qualified_keyword_indices = {
        index
        for index, coverage in enumerate(keyword_coverages)
        if coverage >= KEYWORD_MIN_COVERAGE and keyword_scores[index] > 0
    }
    max_keyword_score = max(
        (keyword_scores[index] for index in qualified_keyword_indices),
        default=0.0,
    )

    ranked_results = []
    for index, dense_score_value in enumerate(scores):
        dense_score = float(dense_score_value)
        keyword_score = keyword_scores[index] if keyword_scores else 0.0

        # 向量通道保留原有阈值；关键词通道可补回语义分偏低但词面明确的结果。
        has_keyword_evidence = index in qualified_keyword_indices
        if dense_score < SIMILARITY_THRESHOLD and not has_keyword_evidence:
            continue

        dense_component = max(0.0, min(1.0, dense_score))
        if has_keyword_evidence and max_keyword_score > 0:
            keyword_component = keyword_score / max_keyword_score
            combined_score = (
                DENSE_WEIGHT * dense_component
                + KEYWORD_WEIGHT * keyword_component
            )
        else:
            # 没有词面匹配时保持旧的纯向量分，避免无关关键词通道压低分数。
            combined_score = dense_component

        ranked_results.append((index, dense_score, keyword_score, combined_score))

    ranked_results.sort(key=lambda item: item[3], reverse=True)

    results = []
    seen_contents = set()

    for index, dense_score, keyword_score, combined_score in ranked_results:
        content = knowledge[index]["content"].strip()

        if content in seen_contents:
            continue

        seen_contents.add(content)

        results.append({
            "content": content,
            "source": knowledge[index]["source"],
            "score": round(combined_score, 4),
            "dense_score": round(dense_score, 4),
            "keyword_score": round(keyword_score, 4),
        })

        if len(results) >= TOP_K:
            break

    return results


# 将检索证据和引用来源整理为 JSON 文本。
def knowledge_search_context(query: str) -> str:
    """以结构化 JSON 返回模型证据和 API 可展示的引用。"""
    results = knowledge_search(query)

    if not results:
        return json.dumps(
            {
                "status": "no_relevant_evidence",
                "evidence": "NO_RELEVANT_EVIDENCE",
                "citations": [],
            },
            ensure_ascii=False,
        )

    evidence = []

    for item in results:
        evidence.append(
            f"来源：{item['source']}\n"
            f"证据：{item['content']}\n"
            f"综合分：{item['score']}"
            + (
                f"（向量：{item['dense_score']}，关键词：{item['keyword_score']}）"
                if "dense_score" in item and "keyword_score" in item
                else ""
            )
        )

    return json.dumps(
        {
            "status": "ok",
            "evidence": "\n\n".join(evidence),
            "citations": [
                {
                    "source": item["source"],
                    "content": item["content"],
                    "score": item["score"],
                    "dense_score": item.get("dense_score"),
                    "keyword_score": item.get("keyword_score"),
                }
                for item in results
            ],
        },
        ensure_ascii=False,
    )


# 清除文档变更后需重建的检索缓存。
def refresh_knowledge_cache() -> None:
    """清理文档变更后需要重建的知识文本、向量和关键词缓存。"""
    load_knowledge.cache_clear()
    get_knowledge_vectors.cache_clear()
    _get_keyword_index.cache_clear()
