"""OpenAI 兼容的对话客户端。

请求发到 `{base}/chat/completions`。base 通常以 `/v1` 结尾。
系统提示词和用户消息分开发送。参考资料只放在用户消息里，避免被当成系统规则。
日志只记模型、耗时和 Token，不记完整提示词。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import httpx

from app.config import Settings
from app.core.exceptions import ConfigurationError, LLMError
from app.llm.models import LLMResult

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class OpenAICompatibleLLMProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        timeout: float = 60,
        max_retries: int = 2,
        sleeper: Callable[[float], None] | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not base_url.strip():
            raise ConfigurationError("缺少 LLM_BASE_URL")
        if not api_key.strip():
            raise ConfigurationError("缺少 LLM_API_KEY")
        if not model.strip():
            raise ConfigurationError("缺少 LLM_MODEL")
        if timeout <= 0:
            raise ConfigurationError("LLM_TIMEOUT 必须大于 0")
        self._base_url = base_url.strip()
        self._api_key = api_key
        self._model = model.strip()
        self._temperature = temperature
        self._timeout = timeout
        self._max_retries = max_retries
        self._sleeper = sleeper or time.sleep
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=timeout)

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAICompatibleLLMProvider:
        return cls(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout,
        )

    @property
    def model_name(self) -> str:
        return self._model

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResult:
        if not system_prompt.strip() or not user_prompt.strip():
            raise LLMError("系统提示词和用户消息都不能为空")
        started = time.perf_counter()
        body = self._post_chat(system_prompt, user_prompt)
        text = _message_text(body)
        if not text.strip():
            raise LLMError("模型返回了空回答")
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        prompt_tokens = _optional_int(usage.get("prompt_tokens"))
        completion_tokens = _optional_int(usage.get("completion_tokens"))
        logger.info(
            "llm_completed model=%s duration_ms=%.1f prompt_tokens=%s completion_tokens=%s",
            self._model,
            (time.perf_counter() - started) * 1000,
            prompt_tokens,
            completion_tokens,
        )
        return LLMResult(
            text=text,
            model=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _post_chat(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        url = chat_completions_url(self._base_url)
        payload = {
            "model": self._model,
            "temperature": self._temperature,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
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
                raise LLMError("大模型服务超时") from exc
            except httpx.HTTPError as exc:
                last_error = exc
                if self._retry(attempt, attempts):
                    continue
                raise LLMError("大模型服务网络错误") from exc
            if response.status_code in _RETRYABLE_STATUS and self._retry(attempt, attempts):
                continue
            if response.status_code >= 400:
                raise LLMError(self._safe_message(response.status_code, response.text))
            try:
                body = response.json()
            except ValueError as exc:
                raise LLMError("大模型响应不是 JSON") from exc
            if not isinstance(body, dict):
                raise LLMError("大模型响应格式不正确")
            return body
        raise LLMError("大模型服务请求失败") from last_error

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
        return f"大模型服务返回 {status}：{snippet}"


def chat_completions_url(base_url: str) -> str:
    trimmed = base_url.strip().rstrip("/")
    if trimmed.endswith("/chat/completions"):
        return trimmed
    return trimmed + "/chat/completions"


def _message_text(body: dict[str, object]) -> str:
    rows = body.get("choices")
    if not isinstance(rows, list) or not rows:
        raise LLMError("大模型响应缺少 choices")
    if all(isinstance(row, dict) and "index" in row for row in rows):
        rows = sorted(rows, key=lambda row: row["index"])
    first = rows[0]
    if not isinstance(first, dict):
        raise LLMError("大模型响应缺少消息")
    message = first.get("message")
    if not isinstance(message, dict):
        raise LLMError("大模型响应缺少消息")
    return _content_text(message.get("content"))


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    raise LLMError("大模型响应缺少文本")


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
