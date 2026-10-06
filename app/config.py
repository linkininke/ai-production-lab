"""应用配置。

业务代码只通过 Settings 读取环境，不直接访问 os.environ。
这样测试可以注入配置，密钥也不会散落到各个模块。

相对路径相对于项目根目录，而不是进程启动时的工作目录。
LLM 与 Embedding 分开配置，因为它们可以来自不同服务商，向量维度也不能混用。
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.exceptions import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_ENV_LABELS = {
    "app_port": "APP_PORT",
    "app_host": "APP_HOST",
    "llm_temperature": "LLM_TEMPERATURE",
    "llm_timeout": "LLM_TIMEOUT",
    "embedding_dimension": "EMBEDDING_DIMENSION",
    "embedding_timeout": "EMBEDDING_TIMEOUT",
    "embedding_batch_size": "EMBEDDING_BATCH_SIZE",
    "chunk_size": "CHUNK_SIZE",
    "chunk_overlap": "CHUNK_OVERLAP",
    "default_top_k": "DEFAULT_TOP_K",
    "max_top_k": "MAX_TOP_K",
    "max_question_chars": "MAX_QUESTION_CHARS",
    "max_context_chars": "MAX_CONTEXT_CHARS",
    "retrieval_max_distance": "RETRIEVAL_MAX_DISTANCE",
    "hybrid_vector_top_k": "HYBRID_VECTOR_TOP_K",
    "hybrid_bm25_top_k": "HYBRID_BM25_TOP_K",
    "hybrid_final_top_k": "HYBRID_FINAL_TOP_K",
    "rrf_k": "RRF_K",
    "reranker_timeout": "RERANKER_TIMEOUT",
    "reranker_candidate_top_k": "RERANKER_CANDIDATE_TOP_K",
    "reranker_final_top_k": "RERANKER_FINAL_TOP_K",
    "llm_input_price_per_1m": "LLM_INPUT_PRICE_PER_1M",
    "llm_output_price_per_1m": "LLM_OUTPUT_PRICE_PER_1M",
    "embedding_price_per_1m": "EMBEDDING_PRICE_PER_1M",
    "reranker_price_per_1k": "RERANKER_PRICE_PER_1K",
    "max_upload_size_mb": "MAX_UPLOAD_SIZE_MB",
    "chroma_collection": "CHROMA_COLLECTION",
    "evaluation_dataset_path": "EVALUATION_DATASET_PATH",
    "evaluation_corpus_dir": "EVALUATION_CORPUS_DIR",
    "evaluation_reports_dir": "EVALUATION_REPORTS_DIR",
}


def resolve_project_path(value: str) -> Path:
    """把配置中的路径解析为绝对路径。绝对路径保持不变。"""
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


class Settings(BaseSettings):
    """进程启动时一次性读入的配置。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8000, ge=1, le=65535)

    # 关闭时允许不填写模型密钥，便于本地启动和单元测试。
    enable_external_models: bool = False

    llm_base_url: str = ""
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""
    llm_temperature: float = Field(default=0.2, ge=0, le=2)
    llm_timeout: float = Field(default=60, gt=0)

    embedding_base_url: str = ""
    embedding_api_key: SecretStr = SecretStr("")
    embedding_model: str = ""
    embedding_dimension: int | None = None
    embedding_timeout: float = Field(default=60, gt=0)
    embedding_batch_size: int = Field(default=64, ge=1, le=256)

    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "knowledge_base"

    chunk_size: int = Field(default=800, ge=1)
    chunk_overlap: int = Field(default=120, ge=0)

    default_top_k: int = Field(default=5, ge=1)
    max_top_k: int = Field(default=20, ge=1)
    max_question_chars: int = Field(default=2000, ge=1)
    max_context_chars: int = Field(default=12000, ge=1)
    # 空值表示不按距离过滤。这是余弦距离上限，不是相似度下限。
    retrieval_max_distance: float | None = Field(default=None, ge=0)
    # 混合检索的候选数量可以大于最终条数。RRF 只使用名次。
    hybrid_vector_top_k: int = Field(default=20, ge=1)
    hybrid_bm25_top_k: int = Field(default=20, ge=1)
    hybrid_final_top_k: int = Field(default=10, ge=1)
    rrf_k: int = Field(default=60, ge=0)
    # 默认关闭。开启后才创建重排客户端，且不使用 MockReranker。
    reranker_enabled: bool = False
    reranker_base_url: str = ""
    reranker_api_key: SecretStr = SecretStr("")
    reranker_model: str = ""
    reranker_timeout: float = Field(default=60, gt=0)
    reranker_candidate_top_k: int = Field(default=20, ge=1)
    reranker_final_top_k: int = Field(default=5, ge=1)
    # 留空表示不估算费用。没有 Token 用量时结果仍是 null，不用价格表编造成本。
    llm_input_price_per_1m: float | None = Field(default=None, ge=0)
    llm_output_price_per_1m: float | None = Field(default=None, ge=0)
    embedding_price_per_1m: float | None = Field(default=None, ge=0)
    reranker_price_per_1k: float | None = Field(default=None, ge=0)
    max_upload_size_mb: int = Field(default=10, ge=1)

    evaluation_dataset_path: str = "./data/evaluation/questions.json"
    evaluation_corpus_dir: str = "./data/evaluation/corpus"
    evaluation_reports_dir: str = "./data/evaluation/reports"
    # mock 不调用模型，也不产生正确性分数。llm 才调用对话模型，并且仍不是标准答案。
    judge_mode: str = "mock"
    judge_max_retries: int = Field(default=1, ge=0, le=2)
    # 三个外部客户端共用。0 表示不重试。上限避免无限重试。
    provider_max_retries: int = Field(default=2, ge=0, le=3)
    evaluation_max_cases: int | None = Field(default=None, ge=1)
    evaluation_max_cost: float | None = Field(default=None, ge=0)
    budget_prompt_tokens: int | None = Field(default=None, ge=0)
    budget_completion_tokens: int | None = Field(default=None, ge=0)
    budget_embedding_tokens: int | None = Field(default=None, ge=0)
    quality_gate_recall_at_5: float | None = Field(default=None, ge=0, le=1)
    quality_gate_citation_validity: float | None = Field(default=None, ge=0, le=1)
    quality_gate_groundedness: float | None = Field(default=None, ge=0, le=4)
    quality_gate_p95_ms: float | None = Field(default=None, ge=0)
    slo_success_rate: float | None = Field(default=None, ge=0, le=1)
    slo_p95_latency_ms: float | None = Field(default=None, ge=0)
    slo_citation_validity: float | None = Field(default=None, ge=0, le=1)
    # 健康卡警告。留空表示不因为这项把状态降为 WARNING。不要在页面里写死比例。
    health_p95_increase_ratio: float | None = Field(default=None, ge=0)
    health_cost_increase_ratio: float | None = Field(default=None, ge=0)
    health_quality_drop: float | None = Field(default=None, ge=0, le=1)
    health_failure_rate_maximum: float | None = Field(default=None, ge=0, le=1)

    @field_validator(
        "app_host",
        "llm_base_url",
        "llm_model",
        "embedding_base_url",
        "embedding_model",
        "reranker_base_url",
        "reranker_model",
        "chroma_persist_dir",
        "evaluation_dataset_path",
        "evaluation_corpus_dir",
        "evaluation_reports_dir",
        "judge_mode",
        mode="before",
    )
    @classmethod
    def strip_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("app_host")
    @classmethod
    def host_not_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("APP_HOST 不能为空")
        return value

    @field_validator(
        "embedding_dimension",
        "retrieval_max_distance",
        "llm_input_price_per_1m",
        "llm_output_price_per_1m",
        "embedding_price_per_1m",
        "reranker_price_per_1k",
        "evaluation_max_cases",
        "evaluation_max_cost",
        "budget_prompt_tokens",
        "budget_completion_tokens",
        "budget_embedding_tokens",
        "quality_gate_recall_at_5",
        "quality_gate_citation_validity",
        "quality_gate_groundedness",
        "quality_gate_p95_ms",
        "slo_success_rate",
        "slo_p95_latency_ms",
        "slo_citation_validity",
        "health_p95_increase_ratio",
        "health_cost_increase_ratio",
        "health_quality_drop",
        "health_failure_rate_maximum",
        mode="before",
    )
    @classmethod
    def blank_optional_number_is_none(cls, value: object) -> object:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return None
        return value

    @field_validator("embedding_dimension")
    @classmethod
    def dimension_must_be_positive(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value < 1:
            raise ValueError("EMBEDDING_DIMENSION 必须是正整数")
        return value

    @field_validator(
        "evaluation_dataset_path",
        "evaluation_corpus_dir",
        "evaluation_reports_dir",
    )
    @classmethod
    def evaluation_path_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("评测路径不能为空")
        return value

    @field_validator("judge_mode")
    @classmethod
    def judge_mode_is_known(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned not in {"mock", "llm"}:
            raise ValueError("JUDGE_MODE 只能是 mock 或 llm")
        return cleaned

    @field_validator("chroma_collection")
    @classmethod
    def collection_name_is_valid(cls, value: str) -> str:
        stripped = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{1,510}[A-Za-z0-9]", stripped):
            raise ValueError(
                "CHROMA_COLLECTION 须为 3 到 512 个字符，只能包含字母、数字、点、下划线和短横线，"
                "并且以字母或数字开头和结尾"
            )
        return stripped

    @model_validator(mode="after")
    def check_cross_fields(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "CHUNK_OVERLAP 必须小于 CHUNK_SIZE"
                f"（当前 CHUNK_OVERLAP={self.chunk_overlap}，CHUNK_SIZE={self.chunk_size}）"
            )
        if self.default_top_k > self.max_top_k:
            raise ValueError(
                "DEFAULT_TOP_K 不能大于 MAX_TOP_K"
                f"（当前 DEFAULT_TOP_K={self.default_top_k}，MAX_TOP_K={self.max_top_k}）"
            )
        if self.max_context_chars < self.chunk_size:
            raise ValueError(
                "MAX_CONTEXT_CHARS 不能小于 CHUNK_SIZE"
                f"（当前 MAX_CONTEXT_CHARS={self.max_context_chars}，"
                f"CHUNK_SIZE={self.chunk_size}）"
            )
        if self.enable_external_models:
            required: dict[str, object] = {
                "LLM_BASE_URL": self.llm_base_url,
                "LLM_API_KEY": self.llm_api_key.get_secret_value(),
                "LLM_MODEL": self.llm_model,
                "EMBEDDING_BASE_URL": self.embedding_base_url,
                "EMBEDDING_API_KEY": self.embedding_api_key.get_secret_value(),
                "EMBEDDING_MODEL": self.embedding_model,
                "EMBEDDING_DIMENSION": self.embedding_dimension,
            }
            missing = [name for name, item in required.items() if item in (None, "")]
            if missing:
                raise ValueError("ENABLE_EXTERNAL_MODELS=true 时缺少配置：" + ", ".join(missing))
        if self.reranker_candidate_top_k < self.reranker_final_top_k:
            raise ValueError(
                "RERANKER_CANDIDATE_TOP_K 不能小于 RERANKER_FINAL_TOP_K"
                f"（当前 RERANKER_CANDIDATE_TOP_K={self.reranker_candidate_top_k}，"
                f"RERANKER_FINAL_TOP_K={self.reranker_final_top_k}）"
            )
        if self.reranker_enabled:
            reranker_required: dict[str, object] = {
                "RERANKER_BASE_URL": self.reranker_base_url,
                "RERANKER_API_KEY": self.reranker_api_key.get_secret_value(),
                "RERANKER_MODEL": self.reranker_model,
            }
            missing_reranker = [
                name for name, item in reranker_required.items() if item in (None, "")
            ]
            if missing_reranker:
                raise ValueError(
                    "RERANKER_ENABLED=true 时缺少配置：" + ", ".join(missing_reranker)
                )
        if self.judge_mode == "llm" and not self.enable_external_models:
            raise ValueError("JUDGE_MODE=llm 时必须打开 ENABLE_EXTERNAL_MODELS")
        return self

    @property
    def chroma_path(self) -> Path:
        return resolve_project_path(self.chroma_persist_dir).resolve()

    @property
    def evaluation_dataset(self) -> Path:
        return resolve_project_path(self.evaluation_dataset_path)

    @property
    def evaluation_corpus(self) -> Path:
        return resolve_project_path(self.evaluation_corpus_dir)

    @property
    def evaluation_reports(self) -> Path:
        return resolve_project_path(self.evaluation_reports_dir)


def _format_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error["loc"])
        label = _ENV_LABELS.get(location, location or "配置")
        parts.append(f"{label}: {error['msg']}")
    detail = "；".join(parts) if parts else str(exc)
    return f"配置无效，请检查环境变量或 .env 文件。{detail}"


@lru_cache
def get_settings() -> Settings:
    """读取并缓存配置。校验失败时抛出可读的 ConfigurationError。"""
    try:
        return Settings()
    except ValidationError as exc:
        raise ConfigurationError(_format_validation_error(exc)) from exc
