"""领域异常与 HTTP 响应分离。"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.exception_handlers import register_exception_handlers
from app.core.exceptions import (
    ConfigurationError,
    ContextOverflowError,
    DocumentIngestionError,
    DocumentValidationError,
    EmbeddingError,
    EvaluationError,
    LLMError,
    RetrievalError,
    VectorStoreError,
)


@pytest.mark.parametrize(
    ("exc_type", "code", "status_code"),
    [
        (DocumentValidationError, "document_validation_error", 400),
        (DocumentIngestionError, "document_ingestion_error", 500),
        (EmbeddingError, "embedding_error", 502),
        (VectorStoreError, "vector_store_error", 500),
        (RetrievalError, "retrieval_error", 500),
        (LLMError, "llm_error", 502),
        (ContextOverflowError, "context_overflow_error", 400),
        (ConfigurationError, "configuration_error", 500),
        (EvaluationError, "evaluation_error", 400),
    ],
)
def test_error_catalog(exc_type: type[Exception], code: str, status_code: int) -> None:
    error = exc_type("示例错误")
    assert error.code == code
    assert error.status_code == status_code
    assert error.message == "示例错误"


def _app_with_handlers() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/validation-error")
    def validation_error(required_value: int) -> dict[str, int]:
        return {"required_value": required_value}

    @app.get("/app-error")
    def app_error() -> None:
        raise DocumentValidationError("文件类型不支持")

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("secret traceback detail")

    return app


def test_app_error_response_hides_traceback() -> None:
    client = TestClient(_app_with_handlers(), raise_server_exceptions=False)
    response = client.get("/app-error")
    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "document_validation_error",
            "message": "文件类型不支持",
        }
    }
    assert "Traceback" not in response.text


def test_unexpected_error_hides_internal_detail() -> None:
    client = TestClient(_app_with_handlers(), raise_server_exceptions=False)
    response = client.get("/boom")
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "internal_error"
    assert body["error"]["message"] == "服务内部错误"
    assert "Traceback" not in response.text
    assert "secret traceback detail" not in response.text
    assert "RuntimeError" not in response.text


def test_validation_error_uses_unified_shape() -> None:
    client = TestClient(_app_with_handlers(), raise_server_exceptions=False)
    response = client.get("/validation-error")
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert "required_value" in body["error"]["message"]


def test_not_found_uses_unified_shape() -> None:
    client = TestClient(_app_with_handlers(), raise_server_exceptions=False)
    response = client.get("/missing")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert "Traceback" not in response.text
