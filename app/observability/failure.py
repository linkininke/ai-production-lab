"""失败分类。

这里区分的是失败发生在哪一层，不是答案正确率。
正确性仍留给 Judge。没有标注、也没有异常时，不猜测生成失败。
记录里只有类型、严重程度、数量和编号，不写问题原文、异常原文和片段正文。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.observability.records import ErrorCategory, ErrorStage

FailureType = Literal[
    "QUERY_FAILURE",
    "RETRIEVAL_FAILURE",
    "RERANK_FAILURE",
    "CONTEXT_FAILURE",
    "PROMPT_FAILURE",
    "GENERATION_FAILURE",
    "CITATION_FAILURE",
    "EVALUATION_FAILURE",
    "TIMEOUT",
    "PROVIDER_ERROR",
    "CONFIGURATION_ERROR",
]
Severity = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]

_STAGE_ORDER = (
    "query_validation",
    "retrieval",
    "reranker",
    "context_builder",
    "prompt_builder",
    "llm_generation",
    "citation_validation",
    "evaluation",
    "provider",
)


class FailureRecord(BaseModel):
    """一条可查询的失败。description 使用固定说明，不复述用户内容。"""

    model_config = ConfigDict(extra="forbid")

    case_id: str = ""
    trace_id: str = ""
    failure_type: FailureType
    severity: Severity
    stage: str
    description: str
    evidence: dict[str, str | int | float | bool] = Field(default_factory=dict)
    suggested_action: str


class FailureSignals(BaseModel):
    """分类器能看到的数量和状态。不接收问题或回答原文。"""

    model_config = ConfigDict(extra="forbid")

    case_id: str = ""
    trace_id: str = ""
    error_type: str | None = None
    error_stage: ErrorStage | None = None
    error_category: ErrorCategory | None = None
    expects_abstention: bool = False
    expected_document_count: int = 0
    retrieved_expected_document_count: int = 0
    expected_chunk_count: int = 0
    retrieved_expected_chunk_count: int = 0
    reranker_executed: bool = False
    has_expected_keywords: bool = False
    keyword_coverage: float | None = None
    has_expected_points: bool = False
    answer_completeness: float | None = None
    invalid_citations: list[str] = Field(default_factory=list)
    citation_validity: float | None = None
    abstention_correct: bool | None = None


def classify_failures(signals: FailureSignals) -> list[FailureRecord]:
    """按阶段给出失败记录。一次请求可以同时有检索失败和引用失败。"""
    if signals.error_type is not None:
        return [_from_error(signals)]
    records = [
        *_retrieval_failure(signals),
        *_generation_failure(signals),
        *_citation_failure(signals),
    ]
    return sorted(records, key=lambda item: _STAGE_ORDER.index(item.stage))


def _from_error(signals: FailureSignals) -> FailureRecord:
    failure_type, severity, stage = _error_kind(signals)
    return _record(
        signals,
        failure_type=failure_type,
        severity=severity,
        stage=stage,
        description=_ERROR_TEXT[failure_type],
        suggested_action=_ACTION[failure_type],
        evidence={
            "error_type": signals.error_type or "",
            "error_category": signals.error_category or "",
        },
    )


def _error_kind(signals: FailureSignals) -> tuple[FailureType, Severity, str]:
    if signals.error_category == "timeout":
        return "TIMEOUT", "HIGH", "provider"
    if signals.error_category == "configuration":
        return "CONFIGURATION_ERROR", "CRITICAL", "provider"
    if signals.error_category == "upstream":
        return "PROVIDER_ERROR", "HIGH", "provider"
    if signals.error_category == "validation" or signals.error_type == "QuestionValidationError":
        return "QUERY_FAILURE", "MEDIUM", "query_validation"
    if signals.error_type == "RerankerError":
        return "RERANK_FAILURE", "HIGH", "reranker"
    if signals.error_type == "ContextOverflowError":
        return "CONTEXT_FAILURE", "HIGH", "context_builder"
    if signals.error_stage == "retrieval":
        return "RETRIEVAL_FAILURE", "HIGH", "retrieval"
    if signals.error_stage == "generation":
        return "GENERATION_FAILURE", "HIGH", "llm_generation"
    if signals.error_type == "EvaluationError":
        return "EVALUATION_FAILURE", "HIGH", "evaluation"
    return "EVALUATION_FAILURE", "HIGH", "evaluation"


def _retrieval_failure(signals: FailureSignals) -> list[FailureRecord]:
    if signals.expects_abstention:
        return []
    document_gap = _gap(signals.expected_document_count, signals.retrieved_expected_document_count)
    chunk_gap = _gap(signals.expected_chunk_count, signals.retrieved_expected_chunk_count)
    if document_gap is None and chunk_gap is None:
        return []
    missed_all = document_gap == "all" or chunk_gap == "all"
    return [
        _record(
            signals,
            failure_type="RETRIEVAL_FAILURE",
            severity="HIGH" if missed_all else "MEDIUM",
            stage="retrieval",
            description="相关文档没有全部出现在检索结果里。",
            suggested_action=_ACTION["RETRIEVAL_FAILURE"],
            evidence={
                "expected_document_count": signals.expected_document_count,
                "retrieved_expected_document_count": signals.retrieved_expected_document_count,
                "expected_chunk_count": signals.expected_chunk_count,
                "retrieved_expected_chunk_count": signals.retrieved_expected_chunk_count,
                "reranker_executed": signals.reranker_executed,
            },
        )
    ]


def _gap(expected: int, found: int) -> str | None:
    if expected < 1 or found >= expected:
        return None
    if found == 0:
        return "all"
    return "partial"


def _generation_failure(signals: FailureSignals) -> list[FailureRecord]:
    if signals.abstention_correct is False and signals.expects_abstention:
        return [_generation_record(signals, "回答没有按要求拒答。")]
    expected = signals.expected_document_count
    found = signals.retrieved_expected_document_count
    chunks_ready = (
        signals.expected_chunk_count < 1
        or signals.retrieved_expected_chunk_count >= signals.expected_chunk_count
    )
    if expected < 1 or found < expected or not chunks_ready:
        return []
    missed_keywords = signals.has_expected_keywords and signals.keyword_coverage == 0
    missed_points = signals.has_expected_points and signals.answer_completeness == 0
    abstained_anyway = signals.abstention_correct is False
    if not missed_keywords and not missed_points and not abstained_anyway:
        return []
    return [_generation_record(signals, "相关文档已经进入上下文，但回答没有覆盖要求的内容。")]


def _generation_record(signals: FailureSignals, description: str) -> FailureRecord:
    evidence: dict[str, str | int | float | bool] = {
        "retrieved_expected_document_count": signals.retrieved_expected_document_count,
    }
    if signals.keyword_coverage is not None:
        evidence["keyword_coverage"] = signals.keyword_coverage
    if signals.answer_completeness is not None:
        evidence["answer_completeness"] = signals.answer_completeness
    return _record(
        signals,
        failure_type="GENERATION_FAILURE",
        severity="MEDIUM",
        stage="llm_generation",
        description=description,
        suggested_action=_ACTION["GENERATION_FAILURE"],
        evidence=evidence,
    )


def _citation_failure(signals: FailureSignals) -> list[FailureRecord]:
    if not signals.invalid_citations:
        return []
    validity = signals.citation_validity
    return [
        _record(
            signals,
            failure_type="CITATION_FAILURE",
            severity="HIGH" if validity == 0 else "MEDIUM",
            stage="citation_validation",
            description="回答使用了上下文中不存在的引用编号。",
            suggested_action=_ACTION["CITATION_FAILURE"],
            evidence={
                "invalid_count": len(signals.invalid_citations),
                "invalid_citations": ",".join(signals.invalid_citations),
            },
        )
    ]


def _record(
    signals: FailureSignals,
    *,
    failure_type: FailureType,
    severity: Severity,
    stage: str,
    description: str,
    suggested_action: str,
    evidence: dict[str, str | int | float | bool],
) -> FailureRecord:
    return FailureRecord(
        case_id=signals.case_id,
        trace_id=signals.trace_id,
        failure_type=failure_type,
        severity=severity,
        stage=stage,
        description=description,
        evidence=evidence,
        suggested_action=suggested_action,
    )


_ACTION: dict[FailureType, str] = {
    "QUERY_FAILURE": "缩短问题，或把 top_k 调回允许范围。",
    "RETRIEVAL_FAILURE": "增大候选数量，并检查分块和混合检索是否漏掉相关文档。",
    "RERANK_FAILURE": "检查重排服务是否启用，以及候选数量是否有效。",
    "CONTEXT_FAILURE": "缩短片段，或提高上下文长度上限。",
    "PROMPT_FAILURE": "检查提示词组装是否抛错。",
    "GENERATION_FAILURE": "相关内容已在上下文中。收紧提示词，或更换模型后再评。",
    "CITATION_FAILURE": "提示词只允许使用上下文里给出的引用编号。",
    "EVALUATION_FAILURE": "检查评测运行过程。这不是模型回答质量。",
    "TIMEOUT": "提高超时时间，或缩短单次输入后再试。",
    "PROVIDER_ERROR": "检查模型服务、网络和本机代理是否可用。",
    "CONFIGURATION_ERROR": "补齐对应服务的地址、模型和密钥。",
}

_ERROR_TEXT: dict[FailureType, str] = {
    "QUERY_FAILURE": "问题或检索参数没有通过校验。",
    "RETRIEVAL_FAILURE": "检索过程失败，没有完成召回。",
    "RERANK_FAILURE": "重排过程失败。",
    "CONTEXT_FAILURE": "上下文无法组装。",
    "PROMPT_FAILURE": "提示词无法组装。",
    "GENERATION_FAILURE": "模型没有完成回答。",
    "CITATION_FAILURE": "引用检查失败。",
    "EVALUATION_FAILURE": "评测或请求过程出现未分类错误。",
    "TIMEOUT": "调用外部服务超时。",
    "PROVIDER_ERROR": "外部服务返回错误或无法连接。",
    "CONFIGURATION_ERROR": "服务配置缺失或不一致。",
}
