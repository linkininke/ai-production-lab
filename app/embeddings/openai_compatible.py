"""OpenAI 兼容的 Embedding 客户端。

请求发到 `{base}/embeddings`。base 通常以 `/v1` 结尾。
响应里的 data 按 index 排序后再使用，因为有的服务商不保证返回顺序。
空文本直接拒绝。用空格冒充向量会把无意义的点写进索引。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import httpx

from app.config import Settings
from app.core.exceptions import ConfigurationError, EmbeddingError

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dimension: int,
        timeout: float = 60,
        batch_size: int = 64,
        max_retries: int = 2,
        sleeper: Callable[[float], None] | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not base_url.strip():
            raise ConfigurationError("缺少 EMBEDDING_BASE_URL")
        if not api_key.strip():
            raise ConfigurationError("缺少 EMBEDDING_API_KEY")
        if not model.strip():
            raise ConfigurationError("缺少 EMBEDDING_MODEL")
        if dimension < 1:
            raise ConfigurationError("EMBEDDING_DIMENSION 必须是正整数")
        if batch_size < 1:
            raise ConfigurationError("EMBEDDING_BATCH_SIZE 必须大于 0")
        self._base_url = base_url.strip()
        self._api_key = api_key
        self._model = model.strip()
        self._dimension = dimension
        self._timeout = timeout
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._sleeper = sleeper or time.sleep
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=timeout)

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAICompatibleEmbeddingProvider:
        if settings.embedding_dimension is None:
            raise ConfigurationError("缺少 EMBEDDING_DIMENSION")
        return cls(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key.get_secret_value(),
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
            timeout=settings.embedding_timeout,
            batch_size=settings.embedding_batch_size,
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        _reject_blank(texts)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        started = time.perf_counter()
        body = self._post_embeddings(texts)
        rows = body.get("data")
        if not isinstance(rows, list):
            raise EmbeddingError("Embedding 响应缺少 data")
        if all(isinstance(row, dict) and "index" in row for row in rows):
            rows = sorted(rows, key=lambda row: row["index"])
        vectors: list[list[float]] = []
        for row in rows:
            if not isinstance(row, dict) or "embedding" not in row:
                raise EmbeddingError("Embedding 响应缺少向量")
            embedding = row["embedding"]
            if not isinstance(embedding, list):
                raise EmbeddingError("Embedding 响应里的向量格式不正确")
            vectors.append([float(value) for value in embedding])
        _validate_vectors(vectors, expected_count=len(texts), dimension=self._dimension)
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        logger.info(
            "embedding_completed model=%s count=%s duration_ms=%.1f prompt_tokens=%s",
            self._model,
            len(texts),
            (time.perf_counter() - started) * 1000,
            usage.get("prompt_tokens"),
        )
        return vectors

    def _post_embeddings(self, texts: list[str]) -> dict[str, object]:
        url = embeddings_url(self._base_url)
        payload = {"model": self._model, "input": texts}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        attempts = self._max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self._client.post(url, headers=headers, json=payload)
            except httpx.TimeoutException as exc:
                last_error = exc
                if self._retry(attempt, attempts):
                    continue
                raise EmbeddingError("Embedding 服务超时") from exc
            except httpx.HTTPError as exc:
                last_error = exc
                if self._retry(attempt, attempts):
                    continue
                raise EmbeddingError("Embedding 服务网络错误") from exc
            if response.status_code in _RETRYABLE_STATUS and self._retry(attempt, attempts):
                continue
            if response.status_code >= 400:
                raise EmbeddingError(self._safe_message(response.status_code, response.text))
            try:
                body = response.json()
            except ValueError as exc:
                raise EmbeddingError("Embedding 响应不是 JSON") from exc
            if not isinstance(body, dict):
                raise EmbeddingError("Embedding 响应格式不正确")
            return body
        raise EmbeddingError("Embedding 服务请求失败") from last_error

    def _retry(self, attempt: int, attempts: int) -> bool:
        if attempt >= attempts - 1:
            return False
        self._sleeper(min(2.0, 0.2 * (2**attempt)))
        return True

    def _safe_message(self, status: int, body: str) -> str:
        redacted = body
        if len(self._api_key) >= 8:
            redacted = redacted.replace(self._api_key, "***")
        snippet = " ".join(redacted.split())[:200]
        return f"Embedding 服务返回 {status}：{snippet}"


def embeddings_url(base_url: str) -> str:
    trimmed = base_url.strip().rstrip("/")
    if trimmed.endswith("/embeddings"):
        return trimmed
    return trimmed + "/embeddings"


def _reject_blank(texts: list[str]) -> None:
    if any(not text.strip() for text in texts):
        raise EmbeddingError("存在空文本，无法向量化")


def _validate_vectors(
    vectors: list[list[float]],
    *,
    expected_count: int,
    dimension: int,
) -> None:
    if len(vectors) != expected_count:
        raise EmbeddingError(
            f"Embedding 返回 {len(vectors)} 条向量，输入有 {expected_count} 条文本"
        )
    for vector in vectors:
        if len(vector) != dimension:
            raise EmbeddingError(f"向量维度是 {len(vector)}，配置要求 {dimension}")
