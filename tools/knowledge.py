
import math
import json
import os
import re
from collections import Counter
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


@lru_cache(maxsize=1)
def _get_keyword_index():
    """缓存 BM25 所需的分词结果、文档频率和平均文档长度。"""
    knowledge = load_knowledge()
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


def _keyword_query_terms(query: str) -> set[str]:
    """移除通用词和疑问词，避免它们单独触发关键词召回。"""
    return {
        term
        for term in _tokenize_for_keyword(query)
        if term not in _KEYWORD_STOP_WORDS
    }


def _keyword_coverages(query: str) -> list[float]:
    """计算每个片段覆盖了多少个有效查询词。"""
    document_terms, _, _ = _get_keyword_index()
    query_terms = _keyword_query_terms(query)

    if not query_terms:
        return [0.0] * len(document_terms)

    return [
        sum(term in terms for term in query_terms) / len(query_terms)
        for terms in document_terms
    ]


def _bm25_scores(query: str) -> list[float]:
    """为每个知识片段计算 BM25 关键词相关度。"""
    document_terms, document_frequency, average_length = _get_keyword_index()

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


def knowledge_search(query: str) -> list[dict]:
    """融合向量相似度和 BM25 关键词分，返回排序后的证据片段。"""
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

    keyword_scores = _bm25_scores(query)
    keyword_coverages = _keyword_coverages(query)
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
