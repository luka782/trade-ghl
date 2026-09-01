from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


RegimeEntryMode = Literal["legacy_all", "confirmation_count"]


@dataclass(frozen=True, slots=True)
class RegimeEntryDecision:
    candidate: bool
    regime_allowed: bool
    price_allowed: bool
    rsi_confirmation: bool
    bollinger_confirmation: bool
    factor_confirmation: bool
    confirmation_count: int
    confirmation_required: int
    passed: bool


def regime_entry_decision(
    *,
    mode: RegimeEntryMode,
    candidate: bool,
    regime_allowed: bool,
    price_allowed: bool,
    rsi_confirmation: bool,
    bollinger_confirmation: bool,
    factor_confirmation: bool,
    confirmation_required: int,
) -> RegimeEntryDecision:
    """组合趋势反转的纯决策函数，便于审计每一层过滤条件。"""
    confirmations = (
        int(rsi_confirmation)
        + int(bollinger_confirmation)
        + int(factor_confirmation)
    )
    if mode == "legacy_all":
        passed = (
            candidate
            and regime_allowed
            and price_allowed
            and rsi_confirmation
            and bollinger_confirmation
            and factor_confirmation
        )
        required = 3
    else:
        required = confirmation_required
        passed = (
            candidate
            and regime_allowed
            and price_allowed
            and confirmations >= required
        )
    return RegimeEntryDecision(
        candidate=candidate,
        regime_allowed=regime_allowed,
        price_allowed=price_allowed,
        rsi_confirmation=rsi_confirmation,
        bollinger_confirmation=bollinger_confirmation,
        factor_confirmation=factor_confirmation,
        confirmation_count=confirmations,
        confirmation_required=required,
        passed=passed,
    )


def donchian_entry(
    *,
    close: float,
    upper: float,
    trend_filter_passed: bool,
) -> bool:
    return close > upper and trend_filter_passed


def donchian_exit(*, close: float, lower: float) -> bool:
    return close < lower


def moving_average_entry(
    *,
    previous_fast: float,
    previous_slow: float,
    fast: float,
    slow: float,
    slow_slope: float,
) -> bool:
    return (
        previous_fast <= previous_slow
        and fast > slow
        and slow_slope > 0
    )


def moving_average_exit(
    *,
    previous_fast: float,
    previous_slow: float,
    fast: float,
    slow: float,
) -> bool:
    return previous_fast >= previous_slow and fast < slow


__all__ = [
    "RegimeEntryDecision",
    "RegimeEntryMode",
    "donchian_entry",
    "donchian_exit",
    "moving_average_entry",
    "moving_average_exit",
    "regime_entry_decision",
]
