"""配置校验。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import PROJECT_ROOT, get_settings
from app.core.exceptions import ConfigurationError
from tests.helpers import make_settings


def test_defaults_do_not_require_api_keys() -> None:
    settings = make_settings()
    assert settings.enable_external_models is False
    assert settings.llm_api_key.get_secret_value() == ""
    assert settings.embedding_api_key.get_secret_value() == ""
    assert settings.chunk_overlap < settings.chunk_size
    assert settings.hybrid_vector_top_k == 20
    assert settings.hybrid_bm25_top_k == 20
    assert settings.hybrid_final_top_k == 10
    assert settings.rrf_k == 60
    assert settings.reranker_enabled is False
    assert settings.reranker_candidate_top_k == 20
    assert settings.reranker_final_top_k == 5


def test_relative_chroma_path_uses_project_root() -> None:
    settings = make_settings(chroma_persist_dir="./data/chroma")
    assert settings.chroma_path == (PROJECT_ROOT / "data" / "chroma").resolve()


def test_settings_repr_hides_api_key() -> None:
    settings = make_settings(
        llm_api_key="sk-super-secret-value",
        reranker_api_key="sk-rerank-secret",
    )
    assert "sk-super-secret-value" not in repr(settings)
    assert "sk-rerank-secret" not in repr(settings)


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError):
        make_settings(chunk_size=100, chunk_overlap=100)


def test_enabled_reranker_requires_its_own_settings() -> None:
    with pytest.raises(ValidationError) as exc_info:
        make_settings(reranker_enabled=True)
    message = str(exc_info.value)
    assert "RERANKER_BASE_URL" in message
    assert "RERANKER_API_KEY" in message
    assert "RERANKER_MODEL" in message


def test_rerank_candidate_pool_cannot_be_smaller_than_final() -> None:
    with pytest.raises(ValidationError, match="RERANKER_CANDIDATE_TOP_K"):
        make_settings(reranker_candidate_top_k=4, reranker_final_top_k=5)


def test_overlap_error_is_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "120")
    monkeypatch.setenv("ENABLE_EXTERNAL_MODELS", "false")
    get_settings.cache_clear()
    with pytest.raises(ConfigurationError) as exc_info:
        get_settings()
    message = str(exc_info.value)
    assert "CHUNK_OVERLAP" in message
    assert "CHUNK_SIZE" in message


def test_default_top_k_cannot_exceed_max() -> None:
    with pytest.raises(ValidationError) as exc_info:
        make_settings(default_top_k=21, max_top_k=20)
    assert "DEFAULT_TOP_K" in str(exc_info.value)


def test_external_models_require_separate_llm_and_embedding_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_EXTERNAL_MODELS", "true")
    monkeypatch.setenv("LLM_BASE_URL", "")
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_MODEL", "")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "")
    monkeypatch.setenv("EMBEDDING_API_KEY", "")
    monkeypatch.setenv("EMBEDDING_MODEL", "")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "")
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("CHUNK_OVERLAP", "120")
    get_settings.cache_clear()
    with pytest.raises(ConfigurationError) as exc_info:
        get_settings()
    message = str(exc_info.value)
    assert "LLM_API_KEY" in message
    assert "EMBEDDING_MODEL" in message
    assert "EMBEDDING_DIMENSION" in message


def test_llm_and_embedding_config_stay_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_EXTERNAL_MODELS", "true")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-llm-key-123456")
    monkeypatch.setenv("LLM_MODEL", "chat-model")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://emb.example/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-emb-key-123456")
    monkeypatch.setenv("EMBEDDING_MODEL", "embed-model")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1536")
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("CHUNK_OVERLAP", "120")
    monkeypatch.setenv("DEFAULT_TOP_K", "5")
    monkeypatch.setenv("MAX_TOP_K", "20")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.llm_model == "chat-model"
    assert settings.embedding_model == "embed-model"
    assert settings.llm_base_url != settings.embedding_base_url
    assert settings.embedding_dimension == 1536


def test_blank_embedding_dimension_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_EXTERNAL_MODELS", "false")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "")
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("CHUNK_OVERLAP", "120")
    get_settings.cache_clear()
    assert get_settings().embedding_dimension is None


def test_collection_name_must_match_chroma_rules() -> None:
    with pytest.raises(ValidationError, match="CHROMA_COLLECTION"):
        make_settings(chroma_collection="kb")


def test_blank_retrieval_distance_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_EXTERNAL_MODELS", "false")
    monkeypatch.setenv("RETRIEVAL_MAX_DISTANCE", "  ")
    get_settings.cache_clear()
    try:
        assert get_settings().retrieval_max_distance is None
    finally:
        get_settings.cache_clear()


def test_context_limit_must_cover_one_chunk() -> None:
    with pytest.raises(ValidationError, match="MAX_CONTEXT_CHARS"):
        make_settings(chunk_size=100, chunk_overlap=10, max_context_chars=50)


def test_invalid_port_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_PORT", "0")
    monkeypatch.setenv("ENABLE_EXTERNAL_MODELS", "false")
    get_settings.cache_clear()
    with pytest.raises(ConfigurationError) as exc_info:
        get_settings()
    assert "APP_PORT" in str(exc_info.value)
