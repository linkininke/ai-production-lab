"""领域异常。

这些异常描述失败原因，不负责 HTTP 状态码的序列化。
路由层把它们转换成统一错误响应，避免把 Python 堆栈返回给调用方。
"""

from __future__ import annotations


class AppError(Exception):
    """应用内部可预期错误的基类。"""

    code = "app_error"
    status_code = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class DocumentValidationError(AppError):
    """文件类型、大小或编码不合法。"""

    code = "document_validation_error"
    status_code = 400


class DocumentIngestionError(AppError):
    """文档导入过程失败。"""

    code = "document_ingestion_error"
    status_code = 500


class DocumentNotFoundError(AppError):
    """要删除的文档不在索引中。"""

    code = "document_not_found"
    status_code = 404


class EmbeddingError(AppError):
    """向量化服务调用失败。"""

    code = "embedding_error"
    status_code = 502


class VectorStoreError(AppError):
    """向量库读写失败。"""

    code = "vector_store_error"
    status_code = 500


class QuestionValidationError(AppError):
    """问题为空、过长，或 top_k 超出允许范围。"""

    code = "question_validation_error"
    status_code = 422


class RetrievalError(AppError):
    """检索流程失败。"""

    code = "retrieval_error"
    status_code = 500


class LLMError(AppError):
    """大模型调用失败。"""

    code = "llm_error"
    status_code = 502


class ContextOverflowError(AppError):
    """组装后的上下文超过允许长度。"""

    code = "context_overflow_error"
    status_code = 400


class ConfigurationError(AppError):
    """启动配置缺失或相互矛盾。"""

    code = "configuration_error"
    status_code = 500


class EvaluationError(AppError):
    """评测集、语料或报告文件无法使用。"""

    code = "evaluation_error"
    status_code = 400

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
