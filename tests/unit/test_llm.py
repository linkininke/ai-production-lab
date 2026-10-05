"""OpenAI 兼容对话客户端。使用假 HTTP 响应，不访问外网。"""

from __future__ import annotations

import json

import httpx
import pytest

from app.core.exceptions import ConfigurationError, LLMError
from app.llm.openai_compatible import OpenAICompatibleLLMProvider, chat_completions_url


def _provider(
    handler,
    *,
    max_retries: int = 2,
    sleeps: list[float] | None = None,
) -> OpenAICompatibleLLMProvider:
    def sleeper(seconds: float) -> None:
        if sleeps is not None:
            sleeps.append(seconds)

    return OpenAICompatibleLLMProvider(
        base_url="https://llm.example/v1",
        api_key="sk-llm-secret-value",
        model="chat-model",
        temperature=0.2,
        max_retries=max_retries,
        sleeper=sleeper,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_chat_url_appends_once() -> None:
    assert chat_completions_url("https://llm.example/v1") == (
        "https://llm.example/v1/chat/completions"
    )
    assert chat_completions_url("https://llm.example/v1/chat/completions") == (
        "https://llm.example/v1/chat/completions"
    )


def test_splits_system_and_user_messages_and_reads_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert request.headers["authorization"] == "Bearer sk-llm-secret-value"
        assert request.url.path == "/v1/chat/completions"
        assert body["model"] == "chat-model"
        assert body["temperature"] == 0.2
        assert body["stream"] is False
        assert body["messages"][0] == {"role": "system", "content": "规则"}
        assert body["messages"][1]["role"] == "user"
        assert "事务" in body["messages"][1]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 1,
                        "message": {"role": "assistant", "content": "后写的"},
                    },
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "事务通过代理。"},
                                {"type": "text", "text": "[C1]"},
                            ],
                        },
                    },
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 4},
            },
        )

    result = _provider(handler).generate("规则", "问题：事务")
    assert result.text == "事务通过代理。[C1]"
    assert result.prompt_tokens == 9
    assert result.completion_tokens == 4
    assert result.model == "chat-model"


def test_retries_rate_limit_but_not_bad_request() -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(
            200,
            json={"choices": [{"index": 0, "message": {"content": "好的"}}]},
        )

    result = _provider(handler, sleeps=sleeps).generate("规则", "问题")
    assert result.text == "好的"
    assert calls == 2
    assert sleeps

    calls = 0

    def rejected(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="rejected sk-llm-secret-value")

    with pytest.raises(LLMError, match="400") as exc_info:
        _provider(rejected).generate("规则", "问题")
    assert calls == 1
    assert "sk-llm-secret-value" not in str(exc_info.value)


def test_timeout_and_empty_answer_become_llm_errors() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(LLMError, match="超时"):
        _provider(timeout, max_retries=0).generate("规则", "问题")

    def empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"index": 0, "message": {"content": "  "}}]},
        )

    with pytest.raises(LLMError, match="空回答"):
        _provider(empty).generate("规则", "问题")


def test_missing_api_key_is_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="LLM_API_KEY"):
        OpenAICompatibleLLMProvider(
            base_url="https://llm.example/v1",
            api_key="",
            model="chat-model",
        )
