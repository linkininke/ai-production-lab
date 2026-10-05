"""评测执行器。使用本地 Chroma 和假模型，不访问外网。"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.container import AppContainer
from app.core.exceptions import LLMError
from app.ingestion.identity import document_id_for
from app.llm.models import LLMResult
from app.main import create_app
from app.vectorstore.chroma_store import ChromaVectorStore
from tests.fake_embedding import HashEmbedding
from tests.helpers import make_settings

_SPRING = "自调用绕过了代理。"
_SECRET = "问题机密标记不在库里"


class SelectiveLLM:
    @property
    def model_name(self) -> str:
        return "fake-llm"

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResult:
        del system_prompt
        if "触发生成失败标记" in user_prompt:
            raise LLMError("模拟生成失败")
        if "没有检索到参考资料" in user_prompt:
            return LLMResult(
                text="知识库中缺少相关信息。",
                model="fake-llm",
                prompt_tokens=3,
                completion_tokens=2,
            )
        return LLMResult(
            text="自调用绕过了代理。[C1]",
            model="fake-llm",
            prompt_tokens=5,
            completion_tokens=4,
        )


def _prepare(tmp_path: Path) -> tuple[TestClient, ChromaVectorStore, Path]:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "spring.md").write_text(_SPRING, encoding="utf-8")
    questions = tmp_path / "questions.json"
    document_id = document_id_for("eval/spring.md")
    payload = {
        "version": "v1",
        "questions": [
            {
                "id": "q001",
                "question": _SPRING,
                "expected_keywords": ["自调用", "代理"],
                "expected_document_ids": [document_id],
                "expects_abstention": False,
            },
            {
                "id": "q002",
                "question": _SECRET,
                "expected_keywords": [],
                "expected_document_ids": [],
                "expects_abstention": True,
            },
            {
                "id": "q003",
                "question": "触发生成失败标记",
                "expected_keywords": ["不会出现"],
                "expected_document_ids": [document_id],
                "expects_abstention": False,
            },
        ],
    }
    questions.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    reports = tmp_path / "reports"
    settings = make_settings(
        chroma_persist_dir=str(tmp_path / "chroma"),
        chroma_collection="kb_local",
        chunk_size=200,
        chunk_overlap=20,
        default_top_k=2,
        max_top_k=5,
        retrieval_max_distance=1e-4,
        evaluation_dataset_path=str(questions),
        evaluation_corpus_dir=str(corpus),
        evaluation_reports_dir=str(reports),
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
        llm=SelectiveLLM(),
    )
    return TestClient(create_app(settings, container)), store, questions


def test_run_saves_report_and_keeps_the_dataset(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    client, store, questions = _prepare(tmp_path)
    original = questions.read_text(encoding="utf-8")
    try:
        with client:
            first = client.post("/api/v1/evaluation/run", json={"top_k": 1})
            assert first.status_code == 200
            body = first.json()
            assert body["question_count"] == 3
            assert body["recall_at_k"] == pytest.approx(1.0)
            assert body["mrr_at_k"] == pytest.approx(1.0)
            assert body["keyword_coverage"] == pytest.approx(1.0)
            assert body["abstention_rate"] == pytest.approx(1.0)
            assert body["abstention_method"] == "phrase_rule"
            assert body["failure_count"] == 1
            assert body["failures"][0]["id"] == "q003"
            assert body["failures"][0]["failure_stage"] == "generation"
            assert body["embedding_model"] == "fake-hash"
            assert body["llm_model"] == "fake-llm"
            assert body["corpus_documents"][0]["status"] == "completed"
            by_id = {item["id"]: item for item in body["questions"]}
            assert by_id["q001"]["retrieved"][0]["score_kind"] == "distance"
            assert by_id["q001"]["retrieved"][0]["score"] < 1e-5
            assert by_id["q002"]["abstained"] is True
            assert by_id["q002"]["retrieved"] == []
            assert questions.read_text(encoding="utf-8") == original
            assert _SECRET not in caplog.text
            saved = tmp_path / "reports" / body["filename"]
            assert _SECRET in saved.read_text(encoding="utf-8")

            second = client.post("/api/v1/evaluation/run", json={"top_k": 2})
            assert second.status_code == 200
            assert second.json()["corpus_documents"][0]["status"] == "skipped"
            listed = client.get("/api/v1/evaluation/reports")
            assert listed.status_code == 200
            names = [item["filename"] for item in listed.json()]
            assert body["filename"] in names
            assert second.json()["filename"] in names

            compared = client.post(
                "/api/v1/evaluation/compare",
                json={"left": body["filename"], "right": second.json()["filename"]},
            )
            assert compared.status_code == 200
            differences = compared.json()["config_differences"]
            assert any("Top-K" in item for item in differences)
            recall = next(
                item for item in compared.json()["metrics"] if item["name"] == "recall_at_k"
            )
            assert recall["delta"] == pytest.approx(0.0)

            missing = client.get(
                "/api/v1/evaluation/reports/eval_20261005T000000000000Z_k5.json"
            )
            assert missing.status_code == 404
            invalid = client.get("/api/v1/evaluation/reports/not-a-report.json")
            assert invalid.status_code == 400
            too_wide = client.post("/api/v1/evaluation/run", json={"top_k": 9})
            assert too_wide.status_code == 422
    finally:
        store.close()


def test_evaluation_without_models_returns_configuration_error(tmp_path: Path) -> None:
    settings = make_settings(
        chroma_persist_dir=str(tmp_path / "chroma"),
        evaluation_reports_dir=str(tmp_path / "reports"),
    )
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/evaluation/run", json={})
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "configuration_error"
        health = client.get("/api/v1/health")
        assert health.status_code == 200
