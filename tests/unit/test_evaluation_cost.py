"""费用估算缺价格或缺用量时必须是空，不能写半截数字。"""

from __future__ import annotations

from app.evaluation.cost import UsageCost, estimate_cost_usd


def test_cost_is_none_when_usage_or_price_is_missing() -> None:
    assert estimate_cost_usd(UsageCost(tokens=None, price_per_unit=2.0, unit=1_000_000)) is None
    assert estimate_cost_usd(UsageCost(tokens=1000, price_per_unit=None, unit=1_000_000)) is None
    assert (
        estimate_cost_usd(
            UsageCost(tokens=1_000_000, price_per_unit=2.0, unit=1_000_000),
            UsageCost(tokens=500, price_per_unit=None, unit=1_000),
        )
        is None
    )


def test_cost_multiplies_complete_usage_by_its_price() -> None:
    cost = estimate_cost_usd(
        UsageCost(tokens=1_000_000, price_per_unit=2.0, unit=1_000_000),
        UsageCost(tokens=2_000, price_per_unit=1.0, unit=1_000),
    )
    assert cost == 4.0
