"""检索对比写到临时目录，不调用外部模型和重排服务。"""

from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.dataset import load_dataset
from app.evaluation.experiment import run_retrieval_experiment
from tests.fake_embedding import HashEmbedding
from tests.helpers import make_settings


def test_retrieval_experiment_skips_rerank_and_leaves_cost_empty(tmp_path: Path) -> None:
    settings = make_settings(reranker_enabled=False)
    dataset = load_dataset(settings.evaluation_dataset)
    output = tmp_path / "results"
    report = run_retrieval_experiment(
        settings,
        dataset,
        HashEmbedding(),
        output_dir=output,
        semantic_embedding=False,
        note="本次使用哈希向量，不是语义模型。",
        reranker=None,
    )
    assert report.question_count >= 30
    assert report.semantic_embedding is False
    assert [item.retriever for item in report.retrievers] == ["vector", "bm25", "hybrid"]
    assert any("RERANKER_ENABLED=false" in item for item in report.skipped)
    for item in report.retrievers:
        assert item.estimated_cost_usd is None
        assert item.recall_at_5 is not None
        assert 0.0 <= item.recall_at_5 <= 1.0
        assert item.precision_at_5 is not None
        assert 0.0 <= item.precision_at_5 <= 1.0
        assert item.mrr_at_5 is not None
        assert 0.0 <= item.mrr_at_5 <= 1.0
    payload = json.loads((output / "latest.json").read_text(encoding="utf-8"))
    assert payload["embedding_model"] == "fake-hash"
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert "哈希向量" in summary
    assert "null" in summary
    assert "0.7300" not in summary
