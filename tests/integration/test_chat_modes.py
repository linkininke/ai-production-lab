"""问答可以切换检索模式。不传模式时仍是向量检索。调试结果不写进日志。"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from tests.integration.test_knowledge_api import _client

_SECRET = "片段正文不应出现在日志里"


def test_bm25_and_hybrid_debug_keep_text_out_of_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    client, store = _client(tmp_path)
    try:
        with client:
            body = f"事务通过代理生效。{_SECRET}".encode()
            created = client.post(
                "/api/v1/documents/upload",
                files={"file": ("spring.md", body, "text/markdown")},
            )
            assert created.status_code == 200
            bm25 = client.post(
                "/api/v1/chat",
                json={
                    "question": "代理",
                    "top_k": 2,
                    "retrieval_mode": "bm25",
                    "debug": True,
                },
            )
            assert bm25.status_code == 200
            bm25_body = bm25.json()
            assert bm25_body["metrics"]["retrieval_mode"] == "bm25"
            assert bm25_body["retrieved_chunks"][0]["score_kind"] == "bm25"
            assert [item["stage"] for item in bm25_body["debug"]["stages"]] == ["bm25"]
            assert [item["name"] for item in bm25_body["debug"]["spans"]] == [
                "query_validation",
                "bm25_retrieval",
                "context_builder",
                "prompt_builder",
                "llm_generation",
                "citation_validation",
            ]
            assert bm25_body["debug"]["trace_status"] == "warning"
            assert bm25_body["debug"]["spans"][-1]["status"] == "warning"
            span_text = str(bm25_body["debug"]["spans"])
            assert _SECRET not in span_text
            assert bm25_body["metrics"]["bm25_candidate_count"] == 1
            assert bm25_body["metrics"]["vector_candidate_count"] is None
            assert bm25_body["metrics"]["reranker_enabled"] is False

            hybrid = client.post(
                "/api/v1/chat",
                json={"question": "代理", "top_k": 2, "retrieval_mode": "hybrid", "debug": True},
            )
            assert hybrid.status_code == 200
            hybrid_body = hybrid.json()
            assert [item["stage"] for item in hybrid_body["debug"]["stages"]] == [
                "vector",
                "bm25",
                "rrf",
            ]
            assert hybrid_body["retrieved_chunks"][0]["score_kind"] == "rrf"
            assert hybrid_body["metrics"]["hybrid_candidate_count"] >= 1
            assert hybrid_body["metrics"]["final_result_count"] >= 1
            assert _SECRET in hybrid.text
            assert _SECRET not in caplog.text
    finally:
        store.close()


def test_unknown_mode_and_unconfigured_rerank_are_rejected(tmp_path: Path) -> None:
    client, store = _client(tmp_path)
    try:
        with client:
            unknown = client.post(
                "/api/v1/chat",
                json={"question": "代理", "retrieval_mode": "graph"},
            )
            assert unknown.status_code == 422
            rerank = client.post(
                "/api/v1/chat",
                json={"question": "代理", "retrieval_mode": "hybrid_rerank"},
            )
            assert rerank.status_code == 500
            assert rerank.json()["error"]["code"] == "configuration_error"
            assert "未启用重排" in rerank.json()["error"]["message"]
    finally:
        store.close()
