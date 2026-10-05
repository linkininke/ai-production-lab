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
    for match in _CITATION.finditer(answer):
        citation_id = f"C{int(match.group(1))}"
        if citation_id in seen or citation_id not in by_id:
            continue
        seen.add(citation_id)
        selected.append(by_id[citation_id])
    return selected
