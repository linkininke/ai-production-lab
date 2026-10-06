"""重排只重排已有候选。Mock 不编造分数，真实适配器只使用服务返回的相关度。"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from app.core.exceptions import ConfigurationError, RerankerError, RetrievalError
from app.retrieval.factory import RetrieverFactory
from app.retrieval.models import RetrievalResult
from app.retrieval.openai_reranker import OpenAICompatibleReranker, rerank_url
from app.retrieval.reranker import MockReranker, RerankingRetriever
from tests.fake_embedding import HashEmbedding


class ScriptedRetriever:
    def __init__(self, hits: list[RetrievalResult]) -> None:
        self.hits = hits
        self.calls: list[int] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        self.calls.append(top_k)
        return self.hits[:top_k]


def _hit(chunk_id: str, score: float = 0.4) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        text=f"正文{chunk_id}",
        score=score,
        score_kind="distance",
        retriever="vector",
        metadata={"filename": f"{chunk_id}.md"},
    )


def test_mock_reorders_without_inventing_scores() -> None:
    documents = [_hit("A", 0.2), _hit("B", 0.5), _hit("C", 0.8)]
    hits = MockReranker(["C", "A"]).rerank("事务失效", documents, top_k=3)
    assert [hit.chunk_id for hit in hits] == ["C", "A", "B"]
    assert [hit.score for hit in hits] == [0.8, 0.2, 0.5]
    assert [hit.score_kind for hit in hits] == ["distance", "distance", "distance"]
    assert [hit.reranker for hit in hits] == ["mock", "mock", "mock"]


def test_mock_does_not_expand_the_candidate_pool() -> None:
    hits = MockReranker().rerank("事务", [_hit("A"), _hit("B")], top_k=5)
    assert [hit.chunk_id for hit in hits] == ["A", "B"]


def test_mock_rejects_a_blank_query_and_an_empty_pool() -> None:
    with pytest.raises(RetrievalError, match="不能为空"):
        MockReranker().rerank("  ", [_hit("A")], top_k=1)
    assert MockReranker(["A"]).rerank("事务", [], top_k=3) == []


def test_reranking_retriever_asks_for_more_candidates_than_it_returns() -> None:
    documents = [_hit(chunk_id) for chunk_id in ("A", "B", "C", "D")]
    retriever = ScriptedRetriever(documents)
    wrapped = RerankingRetriever(
        retriever,
        MockReranker(["D", "B", "C", "A"]),
        candidate_top_k=20,
        final_top_k=2,
    )
    hits = wrapped.retrieve("事务失效")
    assert retriever.calls == [20]
    assert [hit.chunk_id for hit in hits] == ["D", "B"]
    assert hits[0].score == documents[3].score
    assert hits[0].reranker == "mock"


def test_empty_retrieval_does_not_call_the_reranker() -> None:
    class BoomReranker:
        def rerank(
            self,
            query: str,
            documents: list[RetrievalResult],
            top_k: int,
        ) -> list[RetrievalResult]:
            raise AssertionError(query)

    wrapped = RerankingRetriever(ScriptedRetriever([]), BoomReranker())
    assert wrapped.retrieve("事务") == []


def test_factory_refuses_to_use_the_mock_as_production() -> None:
    factory = RetrieverFactory(HashEmbedding(), _Store(), reranker=MockReranker())
    with pytest.raises(ConfigurationError, match="MockReranker 只用于测试"):
        factory.create("hybrid_rerank")


def test_rerank_url_appends_once() -> None:
    assert rerank_url("https://rerank.example/v1") == "https://rerank.example/v1/rerank"
    assert rerank_url("https://rerank.example/v1/rerank") == "https://rerank.example/v1/rerank"


def test_real_reranker_uses_only_the_returned_relevance(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    documents = [_hit("A"), _hit("B"), _hit("C")]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert body["model"] == "bge-reranker"
        assert body["documents"] == ["正文A", "正文B", "正文C"]
        assert body["top_n"] == 2
        assert request.headers["authorization"] == "Bearer sk-rerank-secret"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "index": 2,
                        "relevance_score": 0.2,
                        "document": {"text": "服务端伪造正文"},
                    },
                    {"index": 0, "relevance_score": 0.9},
                ]
            },
        )

    reranker = OpenAICompatibleReranker(
        base_url="https://rerank.example/v1",
        api_key="sk-rerank-secret",
        model="bge-reranker",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    hits = reranker.rerank("机密问题唯一标记", documents, top_k=2)
    assert [hit.chunk_id for hit in hits] == ["A", "C"]
    assert [hit.score for hit in hits] == [0.9, 0.2]
    assert hits[0].text == "正文A"
    assert hits[0].score_kind == "rerank"
    assert hits[0].reranker == "openai_compatible"
    assert "机密问题唯一标记" not in caplog.text
    assert "正文A" not in caplog.text
    assert "sk-rerank-secret" not in caplog.text
    assert "rerank_completed" in caplog.text


def test_real_reranker_does_not_invent_a_score_for_a_bad_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": 0}]})

    reranker = OpenAICompatibleReranker(
        base_url="https://rerank.example/v1",
        api_key="sk-rerank-secret",
        model="bge-reranker",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(RerankerError, match="缺少相关度"):
        reranker.rerank("事务", [_hit("A")], top_k=1)


def test_real_reranker_hides_the_api_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="rejected sk-rerank-secret")

    reranker = OpenAICompatibleReranker(
        base_url="https://rerank.example/v1",
        api_key="sk-rerank-secret",
        model="bge-reranker",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(RerankerError, match="400") as exc_info:
        reranker.rerank("事务", [_hit("A")], top_k=1)
    assert "sk-rerank-secret" not in str(exc_info.value)


def test_empty_documents_do_not_call_the_rerank_api() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(request.url)

    reranker = OpenAICompatibleReranker(
        base_url="https://rerank.example/v1",
        api_key="sk-rerank-secret",
        model="bge-reranker",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert reranker.rerank("事务", [], top_k=3) == []


class _Store:
    embedding_model = "fake-hash"
    dimension = 8
