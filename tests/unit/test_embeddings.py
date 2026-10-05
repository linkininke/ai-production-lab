"""OpenAI 兼容 Embedding 客户端。使用假 HTTP 响应，不访问外网。"""

from __future__ import annotations

import json

import httpx
import pytest

from app.core.exceptions import ConfigurationError, EmbeddingError
from app.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
    embeddings_url,
)


def _provider(
    handler,
    *,
    dimension: int = 3,
    batch_size: int = 64,
    max_retries: int = 2,
    sleeps: list[float] | None = None,
) -> OpenAICompatibleEmbeddingProvider:
    def sleeper(seconds: float) -> None:
        if sleeps is not None:
            sleeps.append(seconds)

    return OpenAICompatibleEmbeddingProvider(
        base_url="https://embed.example/v1",
        api_key="sk-test-secret-value",
        model="embed-model",
        dimension=dimension,
        batch_size=batch_size,
        max_retries=max_retries,
        sleeper=sleeper,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _ok(vectors: list[list[float]], *, reverse: bool = False) -> dict[str, object]:
    rows = [
        {"object": "embedding", "index": index, "embedding": vector}
        for index, vector in enumerate(vectors)
    ]
    if reverse:
        rows.reverse()
    return {"data": rows, "usage": {"prompt_tokens": 5, "total_tokens": 5}}


def test_embeddings_url_appends_once() -> None:
    assert embeddings_url("https://embed.example/v1") == "https://embed.example/v1/embeddings"
    assert embeddings_url("https://embed.example/v1/") == "https://embed.example/v1/embeddings"
    assert (
        embeddings_url("https://embed.example/v1/embeddings")
        == "https://embed.example/v1/embeddings"
    )


def test_sorts_vectors_by_index_and_checks_shape() -> None:
    seen: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        seen.append(body["input"])
        assert request.headers["authorization"] == "Bearer sk-test-secret-value"
        assert request.url.path == "/v1/embeddings"
        vectors = [[float(index), 0.0, 1.0] for index, _ in enumerate(body["input"])]
        return httpx.Response(200, json=_ok(vectors, reverse=True))

    provider = _provider(handler, batch_size=2)
    result = provider.embed_documents(["甲", "乙", "丙"])
    assert seen == [["甲", "乙"], ["丙"]]
    assert result == [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    assert provider.embed_query("甲") == [0.0, 0.0, 1.0]


def test_empty_input_does_not_call_the_api() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(request.url)

    provider = _provider(handler)
    assert provider.embed_documents([]) == []
    with pytest.raises(EmbeddingError, match="空文本"):
        provider.embed_documents(["有内容", "  "])


def test_retries_rate_limit_then_succeeds() -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json=_ok([[0.1, 0.2, 0.3]]))

    provider = _provider(handler, sleeps=sleeps)
    assert provider.embed_query("问题") == [0.1, 0.2, 0.3]
    assert calls == 2
    assert sleeps


def test_client_error_is_not_retried_and_hides_api_key() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="rejected sk-test-secret-value")

    provider = _provider(handler)
    with pytest.raises(EmbeddingError, match="400") as exc_info:
        provider.embed_query("问题")
    assert calls == 1
    assert "sk-test-secret-value" not in str(exc_info.value)


def test_timeout_becomes_embedding_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    provider = _provider(handler, max_retries=0)
    with pytest.raises(EmbeddingError, match="超时"):
        provider.embed_query("问题")


def test_dimension_mismatch_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok([[0.1, 0.2]]))

    provider = _provider(handler, dimension=3)
    with pytest.raises(EmbeddingError, match="维度"):
        provider.embed_query("问题")


def test_missing_settings_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="EMBEDDING_API_KEY"):
        OpenAICompatibleEmbeddingProvider(
            base_url="https://embed.example/v1",
            api_key="",
            model="embed-model",
            dimension=3,
        )
