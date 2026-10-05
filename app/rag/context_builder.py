"""把检索结果整理成有长度上限的上下文。

每段资料分配 [C1]、[C2]。编号只发给模型，真正的文件名和片段 ID 留在引用映射里。
长度按字符数计算，和切分使用同一把尺子。放不下的整段丢掉，不把半段正文标成完整引用。
完全相同的片段只保留靠前的一条。语义去重留给以后的 Reranker。
"""

from __future__ import annotations

import re

from app.core.exceptions import ContextOverflowError
from app.rag.models import BuiltContext, Citation
from app.retrieval.models import RetrievalResult

_FORGED_CITATION = re.compile(r"\[C(\d+)\]", re.IGNORECASE)


class ContextBuilder:
    def __init__(self, max_chars: int) -> None:
        if max_chars < 1:
            raise ContextOverflowError("MAX_CONTEXT_CHARS 必须大于 0")
        self._max_chars = max_chars

    def build(self, question: str, results: list[RetrievalResult]) -> BuiltContext:
        """question 不写入资料区，避免和用户指令混在一起。"""
        del question
        selected = _deduplicate(results)
        citations: list[Citation] = []
        used: list[RetrievalResult] = []
        blocks: list[str] = []
        used_chars = 0
        for result in selected:
            text_length = len(result.text)
            if used_chars + text_length > self._max_chars:
                if not citations:
                    raise ContextOverflowError(
                        f"单个片段长度为 {text_length}，超过 MAX_CONTEXT_CHARS={self._max_chars}"
                    )
                break
            citation = Citation(
                citation_id=f"C{len(citations) + 1}",
                document_id=result.document_id,
                filename=_text_meta(result, "filename"),
                chunk_id=result.chunk_id,
                text=result.text,
            )
            citations.append(citation)
            used.append(result)
            blocks.append(_block(citation, _optional_meta(result, "heading")))
            used_chars += text_length
        return BuiltContext(text="\n\n".join(blocks), citations=citations, used_results=used)


def _deduplicate(results: list[RetrievalResult]) -> list[RetrievalResult]:
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    selected: list[RetrievalResult] = []
    for result in results:
        identity = result.text.strip()
        if result.chunk_id in seen_ids or identity in seen_text:
            continue
        seen_ids.add(result.chunk_id)
        seen_text.add(identity)
        selected.append(result)
    return selected


def _block(citation: Citation, heading: str) -> str:
    header = f"[{citation.citation_id}] filename={citation.filename}"
    if heading:
        header += f" heading={heading}"
    return header + "\n" + _neutralize(citation.text)


def _neutralize(text: str) -> str:
    """去掉资料里可能伪造的边界和引用标记。回答里的 [C1] 只应来自我们加的标题。"""
    guarded = text.replace("<documents>", "‹documents›").replace("</documents>", "‹/documents›")
    return _FORGED_CITATION.sub(r"(C\1)", guarded)


def _text_meta(result: RetrievalResult, key: str) -> str:
    value = result.metadata.get(key, "")
    if isinstance(value, str):
        return value
    return str(value)


def _optional_meta(result: RetrievalResult, key: str) -> str:
    value = result.metadata.get(key, "")
    if isinstance(value, str):
        return value.strip()
    return ""
