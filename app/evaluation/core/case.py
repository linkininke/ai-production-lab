"""一道评测题的标注。

v2 题集的字段继续有效。答案要点默认为空，不要求每道题都写标准答案。
能否回答由 expects_abstention 决定，不再保存一个含义相反的字段。
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_QUESTION_ID = re.compile(r"^[a-z0-9_-]{1,32}$")
_DOCUMENT_ID = re.compile(r"^doc_[0-9a-f]{16}$")
_CHUNK_ID = re.compile(r"^chunk_[0-9a-f]{20}$")
QuestionCategory = Literal[
    "semantic",
    "exact_keyword",
    "technical_identifier",
    "multi_document",
    "unanswerable",
    "reasoning",
    "comparison",
    "procedural",
]


class EvaluationCase(BaseModel):
    """一题的输入和预期属性。运行时只读，不根据检索结果回写标注。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    category: QuestionCategory = "semantic"
    expected_keywords: list[str] = Field(default_factory=list)
    expected_answer_points: list[str] = Field(default_factory=list)
    expected_document_ids: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expects_abstention: bool = False

    @property
    def answerable(self) -> bool:
        return not self.expects_abstention

    @property
    def relevant_chunk_ids(self) -> list[str]:
        return list(self.expected_chunk_ids)

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
        return _unique_texts(value, empty="关键词不能为空", duplicate="关键词不能重复")

    @field_validator("expected_answer_points")
    @classmethod
    def answer_points_are_unique(cls, value: list[str]) -> list[str]:
        return _unique_texts(value, empty="答案要点不能为空", duplicate="答案要点不能重复")

    @field_validator("expected_document_ids")
    @classmethod
    def document_ids_are_real(cls, value: list[str]) -> list[str]:
        return _unique_ids(
            value,
            _DOCUMENT_ID,
            "文档 ID 必须是导入后的 doc_ 加 16 位哈希",
            "预期文档不能重复",
        )

    @field_validator("expected_chunk_ids")
    @classmethod
    def chunk_ids_are_real(cls, value: list[str]) -> list[str]:
        return _unique_ids(
            value,
            _CHUNK_ID,
            "片段 ID 必须是 chunk_ 加 20 位哈希",
            "预期片段不能重复",
        )

    @model_validator(mode="after")
    def abstention_does_not_expect_documents(self) -> EvaluationCase:
        if self.expects_abstention and (self.expected_document_ids or self.expected_chunk_ids):
            raise ValueError("拒答题不能同时标注相关文档")
        if not self.expects_abstention and not self.expected_document_ids:
            raise ValueError("非拒答题必须标注相关文档")
        return self


def _unique_texts(value: list[str], *, empty: str, duplicate: str) -> list[str]:
    cleaned: list[str] = []
    for item in value:
        stripped = item.strip()
        if not stripped:
            raise ValueError(empty)
        cleaned.append(stripped)
    folded = [item.casefold() for item in cleaned]
    if len(folded) != len(set(folded)):
        raise ValueError(duplicate)
    return cleaned


def _unique_ids(
    value: list[str],
    pattern: re.Pattern[str],
    invalid: str,
    duplicate: str,
) -> list[str]:
    cleaned: list[str] = []
    for item in value:
        if not pattern.fullmatch(item):
            raise ValueError(invalid)
        cleaned.append(item)
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(duplicate)
    return cleaned
