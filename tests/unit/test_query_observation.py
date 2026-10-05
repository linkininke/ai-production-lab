"""错误分类把超时和检索逻辑分开。"""

from __future__ import annotations

import pytest

from app.core.exceptions import (
    ConfigurationError,
    ContextOverflowError,
    EmbeddingError,
    LLMError,
    QuestionValidationError,
    RetrievalError,
    VectorStoreError,
)
from app.observability.records import classify_failure


@pytest.mark.parametrize(
    ("exc", "stage", "category"),
    [
        (EmbeddingError("Embedding 服务超时"), "interface", "timeout"),
        (EmbeddingError("Embedding 服务网络错误"), "interface", "upstream"),
        (EmbeddingError("Embedding 响应不是 JSON"), "interface", "upstream"),
        (EmbeddingError("模拟向量化失败"), "retrieval", "retrieval"),
        (RetrievalError("检索失败"), "retrieval", "retrieval"),
        (VectorStoreError("向量库读写失败"), "retrieval", "retrieval"),
        (LLMError("大模型服务超时"), "interface", "timeout"),
        (LLMError("大模型服务返回 500：bad"), "interface", "upstream"),
        (LLMError("大模型响应不是 JSON"), "interface", "upstream"),
        (LLMError("模型返回了空回答"), "generation", "generation"),
        (ContextOverflowError("上下文过长"), "generation", "generation"),
        (QuestionValidationError("问题不能为空"), "validation", "validation"),
        (ConfigurationError("缺少配置"), "interface", "configuration"),
        (RuntimeError("意外"), "interface", "unexpected"),
    ],
)
def test_classify_failure(exc: Exception, stage: str, category: str) -> None:
    assert classify_failure(exc) == (stage, category)
