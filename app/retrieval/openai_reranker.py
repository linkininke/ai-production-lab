"""兼容 POST {base}/rerank 的重排客户端。

请求体是 model、query、documents 和 top_n。
响应使用 results[].index 和 results[].relevance_score。
分数只来自服务返回值。缺字段、序号越界或空结果会失败，不会补一条假分数。
日志只记模型、候选数量和耗时，不记问题、正文和密钥。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import httpx

from app.config import Settings
from app.core.exceptions import ConfigurationError, RerankerError
from app.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class OpenAICompatibleReranker:
    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60,
        max_retries: int = 2,
        sleeper: Callable[[float], None] | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not base_url.strip():
            raise ConfigurationError("缺少 RERANKER_BASE_URL")
        if not api_key.strip():
            raise ConfigurationError("缺少 RERANKER_API_KEY")
        if not model.strip():
            raise ConfigurationError("缺少 RERANKER_MODEL")
        if timeout <= 0:
            raise ConfigurationError("RERANKER_TIMEOUT 必须大于 0")
        self._base_url = base_url.strip()
        self._api_key = api_key
        self._model = model.strip()
        self._timeout = timeout
        self._max_retries = max_retries
        self._sleeper = sleeper or time.sleep
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=timeout)

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAICompatibleReranker:
        return cls(
            base_url=settings.reranker_base_url,
            api_key=settings.reranker_api_key.get_secret_value(),
            model=settings.reranker_model,
            timeout=settings.reranker_timeout,
            max_retries=settings.provider_max_retries,
        )

    @property
    def model_name(self) -> str:
        return self._model

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def rerank(
        self,
        query: str,
        documents: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise RerankerError("问题不能为空")
        if top_k < 1:
            raise RerankerError("top_k 必须大于 0")
        if not documents:
            return []
        if any(not hit.text.strip() for hit in documents):
            raise RerankerError("重排候选缺少正文")
        started = time.perf_counter()
        body = self._post_rerank(query, [hit.text for hit in documents], min(top_k, len(documents)))
        selected = _apply_ranking(documents, body)
        logger.info(
            "rerank_completed model=%s candidate_count=%s result_count=%s latency_ms=%.1f",
            self._model,
            len(documents),
            len(selected),
            (time.perf_counter() - started) * 1000,
        )
        return selected[:top_k]

    def _post_rerank(self, query: str, documents: list[str], top_n: int) -> dict[str, object]:
        url = rerank_url(self._base_url)
        payload = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }
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
                raise RerankerError("重排服务超时") from exc
            except httpx.HTTPError as exc:
                last_error = exc
                if self._retry(attempt, attempts):
                    continue
                raise RerankerError("重排服务网络错误") from exc
            if response.status_code in _RETRYABLE_STATUS and self._retry(attempt, attempts):
                continue
            if response.status_code >= 400:
                raise RerankerError(self._safe_message(response.status_code, response.text))
            try:
                body = response.json()
            except ValueError as exc:
                raise RerankerError("重排响应不是 JSON") from exc
            if not isinstance(body, dict):
                raise RerankerError("重排响应格式不正确")
            return body
        raise RerankerError("重排服务请求失败") from last_error

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
        return f"重排服务返回 {status}：{snippet}"


def rerank_url(base_url: str) -> str:
    trimmed = base_url.strip().rstrip("/")
    if trimmed.endswith("/rerank"):
        return trimmed
    return trimmed + "/rerank"


def _apply_ranking(
    documents: list[RetrievalResult],
    body: dict[str, object],
) -> list[RetrievalResult]:
    rows = body.get("results")
    if not isinstance(rows, list) or not rows:
        raise RerankerError("重排响应没有结果")
    ranked: list[tuple[float, int, RetrievalResult]] = []
    seen: set[int] = set()
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RerankerError("重排响应格式不正确")
        index = _as_index(row.get("index"))
        if index < 0 or index >= len(documents) or index in seen:
            raise RerankerError("重排响应的序号无效")
        seen.add(index)
        score = _as_score(row.get("relevance_score"))
        hit = documents[index].model_copy(
            update={
                "score": score,
                "score_kind": "rerank",
                "reranker": "openai_compatible",
            }
        )
        ranked.append((score, position, hit))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [hit for _, _, hit in ranked]


def _as_index(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RerankerError("重排响应缺少序号")
    return value


def _as_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RerankerError("重排响应缺少相关度")
    return float(value)
