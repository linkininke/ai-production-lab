"""文档级检索指标、关键词覆盖率和拒答短语检查。

Recall@K 看的是标注文档有没有出现在前 K 条片段所属的文档里，不是片段级相关性。
MRR@K 只看第一条命中的名次。关键词覆盖率是诊断信号，不能当成答案正确率。
拒答检查只匹配事先写明的短语，避免把任意否定句都算成拒答。
"""

from __future__ import annotations

# 与系统提示词中的拒答要求对齐。增加短语时要同步测试，不能靠模糊包含“没有”来判断。
ABSTENTION_PHRASES: tuple[str, ...] = (
    "知识库中缺少相关信息",
    "缺少相关信息",
    "没有相关信息",
    "知识库中没有",
    "无法从知识库",
)

ABSTENTION_METHOD = "phrase_rule"


def document_recall_at_k(
    retrieved_document_ids: list[str],
    expected_document_ids: list[str],
) -> float:
    """命中的标注文档数除以标注文档数。重复命中同一文档只算一次。"""
    return _recall(retrieved_document_ids, expected_document_ids, "标注文档")


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """前若干条里命中的标注数除以全部标注数。只命中其中一个时不是 100%。"""
    return _recall(retrieved_ids, relevant_ids, "标注")


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """前 K 个位置里相关结果的数量除以 K。没检索满 K 条时，空位不算相关。"""
    if k < 1:
        raise ValueError("K 必须大于 0")
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("计算 Precision 时必须有标注")
    hits = sum(1 for item in retrieved_ids[:k] if item in relevant)
    return hits / k


def reciprocal_rank_at_k(
    retrieved_document_ids: list[str],
    expected_document_ids: list[str],
) -> float:
    """第一条相关文档的名次倒数。前 K 条里没有相关文档时为 0。名次从 1 开始。"""
    return _mrr(retrieved_document_ids, expected_document_ids, "标注文档")


def mrr_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """只在前 K 条里找第一条相关结果。名次从 1 开始，没有命中时为 0。"""
    if k < 1:
        raise ValueError("K 必须大于 0")
    return _mrr(retrieved_ids[:k], relevant_ids, "标注")


def _recall(retrieved_ids: list[str], relevant_ids: list[str], label: str) -> float:
    relevant = list(dict.fromkeys(relevant_ids))
    if not relevant:
        raise ValueError(f"计算 Recall 时必须有{label}")
    found = set(retrieved_ids) & set(relevant)
    return len(found) / len(relevant)


def _mrr(retrieved_ids: list[str], relevant_ids: list[str], label: str) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError(f"计算 MRR 时必须有{label}")
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def keyword_coverage(answer: str, keywords: list[str]) -> float | None:
    """答案里出现的预设关键词比例。没有关键词时返回 None，不把空列表算成满分或零分。"""
    if not keywords:
        return None
    folded = answer.casefold()
    hits = sum(1 for keyword in keywords if keyword.casefold() in folded)
    return hits / len(keywords)


def matched_keywords(answer: str, keywords: list[str]) -> list[str]:
    folded = answer.casefold()
    return [keyword for keyword in keywords if keyword.casefold() in folded]


def abstention_phrase(answer: str) -> str | None:
    """返回命中的拒答短语。没有命中时返回 None，留给人工判断。"""
    folded = answer.casefold()
    for phrase in ABSTENTION_PHRASES:
        if phrase.casefold() in folded:
            return phrase
    return None


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
