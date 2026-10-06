"""BM25 检索。分数是相关度，越大越相关，不是向量距离。"""

from __future__ import annotations

import logging

import pytest

from app.chunking.models import Chunk
from app.core.exceptions import RetrievalError
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.factory import RetrieverFactory
from app.retrieval.tokenizer import JiebaTokenizer
from tests.fake_embedding import HashEmbedding


class MemoryStore:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def list_chunks(self) -> list[Chunk]:
        return list(self.chunks)


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=chunk_id,
        text=text,
        chunk_index=0,
        metadata={
            "filename": f"{chunk_id}.md",
            "file_type": "markdown",
            "chunk_index": 0,
            "document_id": chunk_id,
        },
    )


def _retriever(chunks: list[Chunk]) -> tuple[BM25Retriever, MemoryStore]:
    store = MemoryStore(chunks)
    return BM25Retriever(store, JiebaTokenizer()), store


def test_exact_keyword_ranks_the_matching_chunk() -> None:
    retriever, _store = _retriever(
        [
            _chunk("tx", "Spring transaction rollback marks the transaction rollback-only."),
            _chunk("pool", "连接池在空闲时回收连接。"),
        ]
    )
    hits = retriever.retrieve("transaction", top_k=5)
    assert [hit.chunk_id for hit in hits] == ["tx"]
    assert hits[0].score_kind == "bm25"
    assert hits[0].retriever == "bm25"
    assert hits[0].score > 0


def test_technical_identifier_matches_the_whole_token() -> None:
    retriever, _store = _retriever(
        [
            _chunk("overflow", "连接池参数 max_overflow 控制溢出连接数。"),
            _chunk("proxy", "事务通过代理生效。"),
        ]
    )
    hits = retriever.retrieve("max_overflow", top_k=5)
    assert [hit.chunk_id for hit in hits] == ["overflow"]


def test_chinese_query_finds_the_spring_note() -> None:
    retriever, _store = _retriever(
        [
            _chunk("spring", "Spring事务失效通常是因为自调用没有走代理。"),
            _chunk("http", "HTTP 客户端要设置超时。"),
        ]
    )
    hits = retriever.retrieve("事务失效", top_k=5)
    assert hits[0].chunk_id == "spring"
    assert hits[0].score_kind == "bm25"


def test_blank_query_is_rejected() -> None:
    retriever, _store = _retriever([_chunk("spring", "Spring事务失效通常是因为自调用没有走代理。")])
    with pytest.raises(RetrievalError, match="不能为空"):
        retriever.retrieve("  ")


def test_unknown_keyword_returns_no_hits(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    retriever, _store = _retriever(
        [_chunk("spring", "Spring事务失效通常是因为自调用没有走代理。")]
    )
    hits = retriever.retrieve("kubernetes", top_k=5)
    assert hits == []
    assert "retriever=bm25" in caplog.text
    assert "result_count=0" in caplog.text
    assert "kubernetes" not in caplog.text


def test_index_rebuilds_after_the_store_changes() -> None:
    store = MemoryStore([_chunk("http", "HTTP 客户端要设置超时。")])
    retriever = BM25Retriever(store, JiebaTokenizer())
    assert retriever.retrieve("事务失效") == []
    store.chunks.append(_chunk("spring", "Spring事务失效通常是因为自调用没有走代理。"))
    hits = retriever.retrieve("事务失效")
    assert [hit.chunk_id for hit in hits] == ["spring"]


def test_shared_term_still_scores_in_a_two_chunk_index() -> None:
    retriever, _store = _retriever(
        [
            _chunk("a", "connection pool borrows a connection"),
            _chunk("b", "the pool returns the connection"),
        ]
    )
    hits = retriever.retrieve("pool")
    assert {hit.chunk_id for hit in hits} == {"a", "b"}
    assert all(hit.score > 0 for hit in hits)


def test_factory_bm25_does_not_embed() -> None:
    embedder = HashEmbedding()
    store = MemoryStore([_chunk("tx", "Spring transaction rollback uses a proxy.")])
    retriever = RetrieverFactory(embedder, store).create("bm25")
    hits = retriever.retrieve("transaction")
    assert hits[0].chunk_id == "tx"
    assert hits[0].score_kind == "bm25"
    assert embedder.document_calls == 0
