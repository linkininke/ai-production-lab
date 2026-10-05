"""运行评测集，并读取已保存的 JSON 报告。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.schemas.evaluation import CompareRequest, EvaluationRunRequest
from app.core.container import AppContainer
from app.core.dependencies import get_container
from app.evaluation.models import EvaluationReport, ReportComparison, ReportSummary
from app.evaluation.runner import EvaluationRunner

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/run", response_model=EvaluationReport)
def run_evaluation(
    body: EvaluationRunRequest,
    container: AppContainer = Depends(get_container),
) -> EvaluationReport:
    return EvaluationRunner.from_container(container).run(top_k=body.top_k)


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
