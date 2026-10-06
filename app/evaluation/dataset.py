"""读取评测集。

文件保存标注，运行时只读，不会为了提高分数改预期文档或关键词。
拒答题必须显式标记。空的文档列表不能反推成拒答，避免把漏标当成无答案。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from app.core.exceptions import EvaluationError
from app.evaluation.core.case import EvaluationCase

_VERSION = re.compile(r"^[A-Za-z0-9._-]{1,32}$")

# 旧名称保留，类型就是 EvaluationCase。
EvalQuestion = EvaluationCase


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    questions: list[EvaluationCase]

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


def _format_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors()[:8]:
        location = ".".join(str(item) for item in error["loc"])
        parts.append(f"{location}: {error['msg']}")
    detail = "；".join(parts) if parts else "字段无效"
    return f"评测集格式无效。{detail}"
