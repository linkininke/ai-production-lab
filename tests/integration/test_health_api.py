"""健康检查接口。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.routes.health import vector_store_status
from app.main import create_app
from tests.helpers import make_settings


def test_vector_store_status_ok(tmp_path) -> None:
    assert vector_store_status(tmp_path / "chroma") == "ok"
    assert not (tmp_path / "chroma" / ".health_probe").exists()


def test_vector_store_status_error_when_path_is_file(tmp_path) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")
    assert vector_store_status(blocker) == "error"


def test_health_ok(tmp_path) -> None:
    settings = make_settings(chroma_persist_dir=str(tmp_path / "chroma"))
    response = TestClient(create_app(settings)).get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "services": {"api": "ok", "vector_store": "ok"},
    }
    assert response.headers["x-request-id"].startswith("req_")


def test_health_degraded_when_vector_path_blocked(tmp_path) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    settings = make_settings(chroma_persist_dir=str(blocker))
    response = TestClient(create_app(settings)).get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["services"]["api"] == "ok"
    assert body["services"]["vector_store"] == "error"


def test_health_does_not_echo_secrets(tmp_path) -> None:
    settings = make_settings(
        chroma_persist_dir=str(tmp_path / "chroma"),
        llm_api_key="sk-health-should-not-leak",
        embedding_api_key="sk-embed-should-not-leak",
    )
    response = TestClient(create_app(settings)).get("/api/v1/health")
    assert "sk-health-should-not-leak" not in response.text
    assert "sk-embed-should-not-leak" not in response.text


def test_client_request_id_is_preserved(tmp_path) -> None:
    settings = make_settings(chroma_persist_dir=str(tmp_path / "chroma"))
    response = TestClient(create_app(settings)).get(
        "/api/v1/health",
        headers={"X-Request-ID": "req_from_client"},
    )
    assert response.headers["x-request-id"] == "req_from_client"
