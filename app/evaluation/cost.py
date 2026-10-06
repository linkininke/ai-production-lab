"""根据 Token 用量和单价估算一次调用的美元成本。

缺用量或缺单价时返回 None。
只给一部分价格时也不做半截估算。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UsageCost:
    """一种用量和它对应的单价。tokens 是这次实际计数，price 是每单位美元。"""

    tokens: int | None
    price_per_unit: float | None
    unit: int


def estimate_cost_usd(*usages: UsageCost) -> float | None:
    """把已给出的用量乘上单价。任何一项只有用量没有单价，或全部用量都是空，都返回 None。"""
    present = [item for item in usages if item.tokens is not None]
    if not present:
        return None
    if any(item.price_per_unit is None for item in present):
        return None
    total = 0.0
    for item in present:
        assert item.tokens is not None
        assert item.price_per_unit is not None
        total += item.tokens * item.price_per_unit / item.unit
    return total
