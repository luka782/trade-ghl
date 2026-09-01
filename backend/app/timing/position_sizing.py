from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Literal

import numpy as np


PositionSizingMode = Literal["full", "fixed", "atr_risk"]


@dataclass(frozen=True, slots=True)
class PositionSize:
    shares: int
    target_notional: float
    risk_cash: float | None
    stop_distance: float | None
    reason: str | None = None


def calculate_position_size(
    *,
    cash: float,
    equity: float,
    execution_price: float,
    adjusted_price: float,
    atr: float,
    lot_size: int,
    mode: PositionSizingMode,
    fixed_fraction: float,
    risk_per_trade: float,
    max_fraction: float,
    atr_stop_multiple: float,
) -> PositionSize:
    """按独立风险预算计算整手目标仓位，不处理手续费。

    手续费和现金精确约束仍由撮合层二次截断；该函数只负责仓位政策，使策略
    信号、仓位管理和成交规则保持分离。
    """
    if (
        cash <= 0
        or equity <= 0
        or execution_price <= 0
        or adjusted_price <= 0
        or lot_size < 1
    ):
        return PositionSize(0, 0.0, None, None, "invalid_position_inputs")

    if mode == "full":
        target_notional = min(cash, equity)
        shares = int(
            floor(target_notional / execution_price / lot_size) * lot_size
        )
        return PositionSize(shares, target_notional, None, None)

    fraction = fixed_fraction if mode == "fixed" else max_fraction
    target_notional = min(cash, equity * fraction)
    budget_shares = int(
        floor(target_notional / execution_price / lot_size) * lot_size
    )
    if mode == "fixed":
        return PositionSize(budget_shares, target_notional, None, None)

    if not np.isfinite(atr) or atr <= 0:
        return PositionSize(0, target_notional, None, None, "missing_atr")
    raw_atr = execution_price * atr / adjusted_price
    stop_distance = raw_atr * atr_stop_multiple
    if not np.isfinite(stop_distance) or stop_distance <= 0:
        return PositionSize(
            0, target_notional, None, None, "invalid_stop_distance"
        )
    risk_cash = equity * risk_per_trade
    risk_shares = int(
        floor(risk_cash / stop_distance / lot_size) * lot_size
    )
    return PositionSize(
        min(budget_shares, risk_shares),
        target_notional,
        risk_cash,
        stop_distance,
    )


__all__ = [
    "PositionSize",
    "PositionSizingMode",
    "calculate_position_size",
]
