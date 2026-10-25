"""BM25 关键词检索（稀疏检索）。

为什么需要它：
向量检索擅长“意思相近”，但会把专有名词、错误码、人名、型号这些
“必须原样命中”的东西糊掉。BM25 靠词频和逆文档频率，正好补这块。

为什么自己做而不装 rank_bm25：
- 依赖可以少一个，BM25 公式本身只有二十行；
- 中文分词不想引入 jieba 这种几十 MB 的词典包。

中文分词的取舍：按“连续汉字切二元组”（bigram）处理。
“深圳市南山区” → 深圳市 / 圳市南 / 市南山 / 南山区。
这是 ES 的 cjk 分析器也在用的做法，不需要词典，召回够用。
"""

from __future__ import annotations

import math
import re
from collections import Counter

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_WORD = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []

    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))

    for word in _ASCII_WORD.findall(text):
        tokens.append(word.lower())

    return tokens


class BM25Index:
    """标准 BM25Okapi（k1 控制词频饱和，b 控制文档长度惩罚）。"""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(doc) for doc in documents]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0

        self.doc_freqs: list[Counter[str]] = []
        df: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            counts = Counter(tokens)
            self.doc_freqs.append(counts)
            df.update(counts.keys())

        total = len(self.doc_tokens)
        # +1 的平滑让 idf 始终为正，避免高频词把分数压成负数
        self.idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def get_scores(self, query: str) -> list[float]:
        query_terms = tokenize(query)
        scores = [0.0] * len(self.doc_tokens)

        if not query_terms or not self.doc_tokens:
            return scores

        for i, counts in enumerate(self.doc_freqs):
            length_norm = 1 - self.b + self.b * (self.doc_len[i] / (self.avg_len or 1))
            score = 0.0
            for term in query_terms:
                freq = counts.get(term, 0)
                if freq == 0:
                    continue
                score += (
                    self.idf.get(term, 0.0)
                    * freq
                    * (self.k1 + 1)
                    / (freq + self.k1 * length_norm)
                )
            scores[i] = score

        return scores

    def top_n(self, query: str, n: int) -> list[tuple[int, float]]:
        scores = self.get_scores(query)
        ranked = sorted(
            ((i, s) for i, s in enumerate(scores) if s > 0),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:n]
