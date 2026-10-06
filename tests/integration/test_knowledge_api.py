"""文档与问答接口。使用本地 Chroma 和假模型，不访问外网。"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.container import AppContainer
from app.main import create_app
from app.vectorstore.chroma_store import ChromaVectorStore
from tests.fake_embedding import HashEmbedding
from tests.fake_llm import ScriptedLLM
from tests.helpers import make_settings

_SPRING = "事务通过代理生效。自调用不会开启事务。"
_POOL = "数据库连接池的大小需要单独配置。"


def _client(tmp_path: Path) -> tuple[TestClient, ChromaVectorStore]:
    settings = make_settings(
        chroma_persist_dir=str(tmp_path / "chroma"),
        chunk_size=200,
        chunk_overlap=20,
        default_top_k=2,
        max_top_k=5,
        max_question_chars=200,
        max_upload_size_mb=1,
    )
    store = ChromaVectorStore(
        persist_dir=tmp_path / "chroma",
        collection_name="kb_local",
        embedding_model="fake-hash",
        embedding_dimension=8,
    )
    container = AppContainer(
        settings=settings,
        embedder=HashEmbedding(),
        vector_store=store,
        llm=ScriptedLLM("事务通过代理生效。[C1] [C99]"),
    )
    return TestClient(create_app(settings, container)), store


def test_upload_list_chat_and_delete_round_trip(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    client, store = _client(tmp_path)
    secret = "正文机密标记ABC"
    try:
        with client:
            created = client.post(
                "/api/v1/documents/upload",
                files={"file": ("spring.md", f"{_SPRING}{secret}".encode(), "text/markdown")},
            )
            assert created.status_code == 200
            body = created.json()
            assert body["filename"] == "spring.md"
            assert body["status"] == "completed"
            assert body["chunk_count"] == 1
            document_id = body["document_id"]

            again = client.post(
                "/api/v1/documents/upload",
                files={"file": ("spring.md", f"{_SPRING}{secret}".encode(), "text/markdown")},
            )
            assert again.status_code == 200
            assert again.json()["status"] == "skipped"
            assert again.json()["document_id"] == document_id

            other = client.post(
                "/api/v1/documents/upload",
                files={"file": ("pool.txt", _POOL.encode(), "text/plain")},
            )
            assert other.status_code == 200

            listing = client.get("/api/v1/documents")
            assert listing.status_code == 200
            listed = listing.json()["documents"]
            assert {item["document_id"] for item in listed} == {
                document_id,
                other.json()["document_id"],
            }
            spring = next(item for item in listed if item["document_id"] == document_id)
            assert spring["filename"] == "spring.md"
            assert spring["chunk_count"] == 1
            assert spring["status"] == "completed"
            assert spring["created_at"]
            assert secret not in listing.text

            answer = client.post(
                "/api/v1/chat",
                json={"question": f"{_SPRING}{secret}", "top_k": 1},
            )
            assert answer.status_code == 200
            payload = answer.json()
            assert payload["citations"][0]["citation_id"] == "C1"
            assert payload["citations"][0]["filename"] == "spring.md"
            assert payload["citations"][0]["document_id"] == document_id
            assert "C99" not in {item["citation_id"] for item in payload["citations"]}
            assert payload["retrieved_chunks"][0]["score_kind"] == "distance"
            assert payload["retrieved_chunks"][0]["score"] < 1e-5
            assert payload["metrics"]["prompt_tokens"] == 11
            assert payload["metrics"]["retrieval_mode"] == "vector"
            assert payload["debug"] is None
            assert payload["metrics"]["reranker_enabled"] is False
            assert "提问机密标记DEF" not in caplog.text

            removed = client.delete(f"/api/v1/documents/{document_id}")
            assert removed.status_code == 200
            assert removed.json()["deleted_chunks"] == 1
            missing = client.delete(f"/api/v1/documents/{document_id}")
            assert missing.status_code == 404
            assert missing.json()["error"]["code"] == "document_not_found"
            remaining = client.get("/api/v1/documents").json()["documents"]
            assert document_id not in {item["document_id"] for item in remaining}
            assert secret not in caplog.text
    finally:
        store.close()


def test_upload_and_chat_reject_invalid_input(tmp_path: Path) -> None:
    client, store = _client(tmp_path)
    try:
        with client:
            rejected = client.post(
                "/api/v1/documents/upload",
                files={"file": ("notes.pdf", b"not supported", "application/pdf")},
            )
            assert rejected.status_code == 400
            assert rejected.json()["error"]["code"] == "document_validation_error"

            nested = client.post(
                "/api/v1/documents/upload",
                files={"file": ("folder/notes.md", "正文".encode(), "text/markdown")},
            )
            assert nested.status_code == 400

            empty = client.post("/api/v1/chat", json={"question": "   ", "top_k": 1})
            assert empty.status_code == 422
            assert empty.json()["error"]["code"] == "question_validation_error"

            huge = client.post("/api/v1/chat", json={"question": "问", "top_k": 9})
            assert huge.status_code == 422

            missing_file = client.post("/api/v1/documents/upload")
            assert missing_file.status_code == 422
            assert missing_file.json()["error"]["code"] == "validation_error"
    finally:
        store.close()


def test_oversized_upload_is_rejected(tmp_path: Path) -> None:
    client, store = _client(tmp_path)
    try:
        with client:
            too_big = client.post(
                "/api/v1/documents/upload",
                files={"file": ("notes.md", b"x" * (1024 * 1024 + 1), "text/markdown")},
            )
            assert too_big.status_code == 400
            assert "大小" in too_big.json()["error"]["message"]
    finally:
        store.close()


def test_routes_without_models_return_configuration_error(tmp_path: Path) -> None:
    settings = make_settings(chroma_persist_dir=str(tmp_path / "chroma"))
    with TestClient(create_app(settings)) as client:
        listed = client.get("/api/v1/documents")
        assert listed.status_code == 500
        assert listed.json()["error"]["code"] == "configuration_error"
        uploaded = client.post(
            "/api/v1/documents/upload",
            files={"file": ("spring.md", _SPRING.encode(), "text/markdown")},
        )
        assert uploaded.status_code == 500
        asked = client.post("/api/v1/chat", json={"question": "事务为什么失效"})
        assert asked.status_code == 500
        assert asked.json()["error"]["code"] == "configuration_error"
        health = client.get("/api/v1/health")
        assert health.status_code == 200
