"""运行评测集，并读取已保存的 JSON 报告。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.schemas.evaluation import CompareRequest, EvaluationRunRequest
from app.core.container import AppContainer
from app.core.dependencies import get_container
from app.evaluation.experiment import (
    ExperimentReport,
    latest_experiment_path,
    load_saved_experiment,
)
from app.evaluation.health import (
    HealthCard,
    HealthWarningRules,
    build_health_card,
    gates_for_saved_report,
    load_health_context,
)
from app.evaluation.models import EvaluationReport, ReportComparison, ReportSummary
from app.evaluation.runner import EvaluationRunner

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/run", response_model=EvaluationReport)
def run_evaluation(
    body: EvaluationRunRequest,
    container: AppContainer = Depends(get_container),
) -> EvaluationReport:
    return EvaluationRunner.from_container(container).run(
        top_k=body.top_k,
        max_cases=body.max_cases,
        max_cost=body.max_cost,
    )


@router.get("/retrieval-experiment", response_model=ExperimentReport)
def read_retrieval_experiment() -> ExperimentReport:
    return load_saved_experiment(latest_experiment_path())


@router.get("/health", response_model=HealthCard)
def read_health_card(
    report: str,
    baseline: str | None = None,
    container: AppContainer = Depends(get_container),
) -> HealthCard:
    runner = EvaluationRunner.from_container(container)
    current = runner.load_report(report)
    previous = runner.load_report(baseline) if baseline else None
    settings = container.settings
    regression, experiment, note = load_health_context(settings.evaluation_dataset)
    rules = HealthWarningRules(
        p95_increase_ratio=settings.health_p95_increase_ratio,
        cost_increase_ratio=settings.health_cost_increase_ratio,
        quality_drop=settings.health_quality_drop,
        failure_rate_maximum=settings.health_failure_rate_maximum,
    )
    live_gate, live_slo = gates_for_saved_report(current, settings)
    return build_health_card(
        current,
        baseline=previous,
        regression=regression,
        experiment=experiment,
        warning=rules,
        context_note=note,
        live_gate=live_gate,
        live_slo=live_slo,
    )


@router.get("/reports", response_model=list[ReportSummary])
def list_reports(container: AppContainer = Depends(get_container)) -> list[ReportSummary]:
    return EvaluationRunner.from_container(container).list_reports()


@router.get("/reports/{filename}", response_model=EvaluationReport)
def read_report(
    filename: str,
    container: AppContainer = Depends(get_container),
) -> EvaluationReport:
    return EvaluationRunner.from_container(container).load_report(filename)


@router.post("/compare", response_model=ReportComparison)
def compare_saved_reports(
    body: CompareRequest,
    container: AppContainer = Depends(get_container),
) -> ReportComparison:
    return EvaluationRunner.from_container(container).compare(body.left, body.right)
