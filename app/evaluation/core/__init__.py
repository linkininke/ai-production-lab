"""评测核心类型。指标函数仍在 app.evaluation.metrics。"""

from app.evaluation.core.case import EvaluationCase, QuestionCategory
from app.evaluation.core.result import EvaluationResult, measure_retrieval

__all__ = [
    "EvaluationCase",
    "EvaluationResult",
    "QuestionCategory",
    "measure_retrieval",
]
