"""用关键词检索片段。

每次提问都从向量库读出全部片段并重建索引。上传或删除之后，下一次提问能看到新内容。
返回的 score 是 BM25 相关度，score_kind 为 bm25。它不是距离，也不能拿来和向量分数相加。
"""

from __future__ import annotations

import logging
from time import perf_counter

from app.core.exceptions import RetrievalError
from app.retrieval.bm25_index import BM25Index
from app.retrieval.models import RetrievalResult
from app.retrieval.tokenizer import Tokenizer
from app.retrieval.trace import note_bm25
from app.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


class BM25Retriever:
    name = "bm25"

    def __init__(self, store: VectorStore, tokenizer: Tokenizer) -> None:
        self._store = store
        self._tokenizer = tokenizer

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not query.strip():
            raise RetrievalError("问题不能为空")
        if top_k < 1:
            raise RetrievalError("top_k 必须大于 0")
        started = perf_counter()
        index = BM25Index(self._store.list_chunks(), self._tokenizer)
        query_tokens = self._tokenizer.tokenize(query.strip())
        matches = index.search(query_tokens, top_k)
        hits = [
            RetrievalResult(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                score=score,
                score_kind="bm25",
                retriever=self.name,
                metadata=dict(chunk.metadata),
            )
            for chunk, score in matches
        ]
        note_bm25(hits, latency_ms=(perf_counter() - started) * 1000)
        logger.info(
            "retrieval_completed retriever=%s query_length=%s top_k=%s "
            "result_count=%s latency_ms=%.1f",
            self.name,
            len(query.strip()),
            top_k,
            len(hits),
            (perf_counter() - started) * 1000,
        )
        return hits
