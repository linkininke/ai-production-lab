"""用当前片段重建的内存 BM25 索引。

索引不落盘。Chroma 仍是唯一正文来源，避免关键词索引和向量库各写一份后对不上。
分数是 Okapi BM25，越大越相关，不能和余弦距离相加。

IDF 使用 Lucene 的正值公式 ln(1 + (N - df + 0.5) / (df + 0.5))。
常见实现里的 ln((N - df + 0.5) / (df + 0.5)) 在词只出现于一半片段时为 0，
两篇文档的知识库会把精确关键词全部丢掉。
"""

from __future__ import annotations

import math
from collections import Counter

from app.chunking.models import Chunk
from app.retrieval.tokenizer import Tokenizer

_K1 = 1.5
_B = 0.75


class BM25Index:
    def __init__(self, chunks: list[Chunk], tokenizer: Tokenizer) -> None:
        self._chunks: list[Chunk] = []
        self._counts: list[Counter[str]] = []
        document_frequency: Counter[str] = Counter()
        lengths: list[int] = []
        for chunk in chunks:
            tokens = tokenizer.tokenize(chunk.text)
            if not tokens:
                continue
            counts = Counter(tokens)
            self._chunks.append(chunk)
            self._counts.append(counts)
            lengths.append(sum(counts.values()))
            document_frequency.update(counts.keys())
        total = len(self._counts)
        self._avgdl = sum(lengths) / total if total else 0.0
        self._idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def search(self, query_tokens: list[str], top_k: int) -> list[tuple[Chunk, float]]:
        if not self._counts or not query_tokens or top_k < 1:
            return []
        ranked = [
            (index, score)
            for index, counts in enumerate(self._counts)
            if (score := self._score(counts, query_tokens)) > 0
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return [(self._chunks[index], score) for index, score in ranked[:top_k]]

    def _score(self, counts: Counter[str], query_tokens: list[str]) -> float:
        doc_len = sum(counts.values())
        normalizer = _K1 * (1 - _B + _B * doc_len / self._avgdl)
        score = 0.0
        for term in query_tokens:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            numerator = tf * (_K1 + 1)
            score += self._idf[term] * numerator / (tf + normalizer)
        return score
