"""检索对比的命令行入口。

外部模型关闭时使用哈希向量，并在报告里写明这不是语义效果。
外部模型开启时使用配置里的向量服务，写入临时集合 kb_eval，不改用户知识库。
"""

from __future__ import annotations

import argparse
import sys

from app.config import PROJECT_ROOT, get_settings
from app.core.exceptions import AppError
from app.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider
from app.evaluation.dataset import load_dataset
from app.evaluation.experiment import run_retrieval_experiment
from app.evaluation.reliability import plan_budget, predicted_case_cost_usd
from app.retrieval.openai_reranker import OpenAICompatibleReranker

_HASH_NOTE = (
    "本次使用哈希向量，不是语义模型。"
    "Vector 和 Hybrid 的数字不能用来证明混合检索优于向量检索。"
    "BM25 不依赖向量，是这次语料上的关键词结果。"
    "Hybrid+Reranker 在未配置重排时被跳过。estimated_cost_usd 为 null。"
)
_SEMANTIC_NOTE = (
    "本次使用配置中的向量模型，结果来自这次检索。"
    "没有 Token 用量，estimated_cost_usd 为 null。"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="运行检索对比。设置预算后，无法估算或超出预算会在调用前停止。"
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-cost", type=float, default=None)
    args = parser.parse_args(argv)
    settings = get_settings()
    dataset = load_dataset(settings.evaluation_dataset)
    decision = plan_budget(
        question_count=len(dataset.questions),
        max_cases=args.max_cases,
        max_cost=args.max_cost,
        predicted_cost_per_case=predicted_case_cost_usd(
            prompt_tokens=settings.budget_prompt_tokens,
            completion_tokens=settings.budget_completion_tokens,
            embedding_tokens=settings.budget_embedding_tokens,
            llm_input_price_per_1m=settings.llm_input_price_per_1m,
            llm_output_price_per_1m=settings.llm_output_price_per_1m,
            embedding_price_per_1m=settings.embedding_price_per_1m,
        ),
    )
    if decision.aborted:
        print(decision.reason, file=sys.stderr)
        return 1
    if decision.selected_cases < len(dataset.questions):
        dataset = dataset.model_copy(
            update={"questions": dataset.questions[: decision.selected_cases]}
        )
    output_dir = PROJECT_ROOT / "evaluation_results"
    embedder = None
    reranker = None
    try:
        if settings.enable_external_models:
            embedder = OpenAICompatibleEmbeddingProvider.from_settings(settings)
            if settings.reranker_enabled:
                reranker = OpenAICompatibleReranker.from_settings(settings)
            semantic = True
            note = _SEMANTIC_NOTE
        else:
            from tests.fake_embedding import HashEmbedding

            embedder = HashEmbedding()
            semantic = False
            note = _HASH_NOTE
        report = run_retrieval_experiment(
            settings,
            dataset,
            embedder,
            output_dir=output_dir,
            semantic_embedding=semantic,
            note=note,
            reranker=reranker,
        )
    except AppError as exc:
        print(exc.message, file=sys.stderr)
        return 1
    finally:
        for resource in (reranker, embedder):
            close = getattr(resource, "close", None)
            if close is not None:
                close()
    print(output_dir / "summary.md")
    print(f"question_count={report.question_count} semantic_embedding={report.semantic_embedding}")
    return 0
