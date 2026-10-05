"""读取评测集。

文件保存标注，运行时只读，不会为了提高分数改预期文档或关键词。
拒答题必须显式标记。空的文档列表不能反推成拒答，避免把漏标当成无答案。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.core.exceptions import EvaluationError

_QUESTION_ID = re.compile(r"^[a-z0-9_-]{1,32}$")
_DOCUMENT_ID = re.compile(r"^doc_[0-9a-f]{16}$")
_VERSION = re.compile(r"^[A-Za-z0-9._-]{1,32}$")


class EvalQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    expected_keywords: list[str] = Field(default_factory=list)
    expected_document_ids: list[str] = Field(default_factory=list)
    expects_abstention: bool = False

    @field_validator("id")
    @classmethod
    def question_id_is_safe(cls, value: str) -> str:
        if not _QUESTION_ID.fullmatch(value):
            raise ValueError("题目 ID 只能使用小写字母、数字、下划线和短横线")
        return value

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("问题不能为空")
        return stripped

    @field_validator("expected_keywords")
    @classmethod
    def keywords_are_unique(cls, value: list[str]) -> list[str]:
        cleaned = [_keyword(item) for item in value]
        folded = [item.casefold() for item in cleaned]
        if len(folded) != len(set(folded)):
            raise ValueError("关键词不能重复")
        return cleaned

    @field_validator("expected_document_ids")
    @classmethod
    def document_ids_are_real(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in value:
            if not _DOCUMENT_ID.fullmatch(item):
                raise ValueError("文档 ID 必须是导入后的 doc_ 加 16 位哈希")
            cleaned.append(item)
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("预期文档不能重复")
        return cleaned

    @model_validator(mode="after")
    def abstention_does_not_expect_documents(self) -> EvalQuestion:
        if self.expects_abstention and self.expected_document_ids:
            raise ValueError("拒答题不能同时标注相关文档")
        if not self.expects_abstention and not self.expected_document_ids:
            raise ValueError("非拒答题必须标注相关文档")
        return self


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    questions: list[EvalQuestion]

    @field_validator("version")
    @classmethod
    def version_is_stable(cls, value: str) -> str:
        if not _VERSION.fullmatch(value):
            raise ValueError("数据集版本只能包含字母、数字、点、下划线和短横线")
        return value

    @model_validator(mode="after")
    def questions_are_present_and_unique(self) -> EvaluationDataset:
        if not self.questions:
            raise ValueError("评测集至少要有一道题")
        ids = [item.id for item in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("题目 ID 不能重复")
        return self


def load_dataset(path: Path) -> EvaluationDataset:
    """从 JSON 读取评测集。格式错误时给出字段位置，不回显整份文件。"""
    if not path.is_file():
        raise EvaluationError(f"评测集不存在：{path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationError("评测集不是合法 JSON") from exc
    except OSError as exc:
        raise EvaluationError("读取评测集失败") from exc
    try:
        return EvaluationDataset.model_validate(payload)
    except ValidationError as exc:
        raise EvaluationError(_format_validation_error(exc)) from exc


def _keyword(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("关键词不能为空")
    return stripped


def _format_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors()[:8]:
        location = ".".join(str(item) for item in error["loc"])
        parts.append(f"{location}: {error['msg']}")
    detail = "；".join(parts) if parts else "字段无效"
    return f"评测集格式无效。{detail}"
