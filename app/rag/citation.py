"""从回答里提取引用标记，并只保留上下文里真实存在的来源。

模型写出的 [C9]，如果这次没有 C9，就丢掉。
文件名、片段 ID 和原文都来自检索结果，不解析模型自己编的出处。
"""

from __future__ import annotations

import re

from app.rag.models import Citation

_CITATION = re.compile(r"\[C(\d+)\]", re.IGNORECASE)


def map_citations(answer: str, available: list[Citation]) -> list[Citation]:
    by_id = {item.citation_id.upper(): item for item in available}
    selected: list[Citation] = []
    seen: set[str] = set()
    for citation_id in citation_markers(answer):
        if citation_id in seen or citation_id not in by_id:
            continue
        seen.add(citation_id)
        selected.append(by_id[citation_id])
    return selected


def audit_citations(answer: str, available_ids: list[str]) -> tuple[list[str], list[str]]:
    """区分回答里真实存在的引用和不存在的引用。不存在的编号必须留下来，不能当成有效引用。"""
    available = {item.upper() for item in available_ids}
    valid: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for citation_id in citation_markers(answer):
        if citation_id in seen:
            continue
        seen.add(citation_id)
        if citation_id in available:
            valid.append(citation_id)
        else:
            invalid.append(citation_id)
    return valid, invalid


def citation_markers(answer: str) -> list[str]:
    return [f"C{int(match.group(1))}" for match in _CITATION.finditer(answer)]
