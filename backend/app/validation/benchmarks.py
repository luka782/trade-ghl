from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .diagnostics import sharpe_ratio


def return_metrics(
    returns: Iterable[Any],
    *,
    periods_per_year: int = 252,
) -> dict[str, float | int]:
    """Dependency-light metrics shared by strategy and baseline returns."""

    values = pd.Series(list(returns), dtype=float).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if values.empty:
        return {
            "observations": 0,
            "total_return": float("nan"),
            "annualized_return": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "volatility": float("nan"),
        }
    wealth = (1.0 + values).cumprod()
    total_return = float(wealth.iloc[-1] - 1.0)
    annualized = (
        float((1.0 + total_return) ** (periods_per_year / len(values)) - 1.0)
        if total_return > -1.0
        else -1.0
    )
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "observations": len(values),
        "total_return": total_return,
        "annualized_return": annualized,
        "sharpe": sharpe_ratio(values, periods_per_year=periods_per_year),
        "max_drawdown": float(drawdown.min()),
        "volatility": (
            float(values.std(ddof=1) * np.sqrt(periods_per_year))
            if len(values) > 1
            else 0.0
        ),
    }


def buy_and_hold_returns(prices: Iterable[Any]) -> pd.Series:
    values = pd.Series(list(prices), dtype=float).replace(
        [np.inf, -np.inf], np.nan
    )
    if values.dropna().empty or (values.dropna() <= 0).any():
        raise ValueError("prices must contain positive finite observations")
    return values.pct_change(fill_method=None).fillna(0.0)


def equal_weight_returns(
    returns: pd.DataFrame,
    *,
    rebalance: str | None = None,
) -> pd.Series:
    """Equal-weight available assets, optionally resetting weights periodically."""

    numeric = returns.apply(pd.to_numeric, errors="coerce")
    if numeric.empty or numeric.shape[1] < 1:
        raise ValueError("returns must contain at least one asset")
    if rebalance is None:
        return numeric.mean(axis=1, skipna=True).fillna(0.0)
    if not isinstance(numeric.index, pd.DatetimeIndex):
        raise ValueError("A DatetimeIndex is required for periodic rebalancing")

    # Drift weights between rebalance dates; only information through t-1 is used.
    result = pd.Series(0.0, index=numeric.index, dtype=float)
    weights = pd.Series(0.0, index=numeric.columns)
    period = numeric.index.to_period(rebalance)
    previous_period: Any = None
    for position, (current_period, (_, row)) in enumerate(zip(period, numeric.iterrows())):
        available = row.notna()
        if position == 0 or current_period != previous_period:
            weights[:] = 0.0
            if available.any():
                weights.loc[available] = 1.0 / int(available.sum())
        result.iloc[position] = float((weights * row.fillna(0.0)).sum())
        gross = weights * (1.0 + row.fillna(0.0))
        if gross.sum() > 0:
            weights = gross / gross.sum()
        previous_period = current_period
    return result


def causal_moving_average_returns(
    prices: Iterable[Any],
    *,
    window: int = 200,
) -> pd.Series:
    """Long/cash trend baseline; today's position uses yesterday's close history."""

    if window < 2:
        raise ValueError("window must be at least 2")
    values = pd.Series(list(prices), dtype=float)
    asset_returns = values.pct_change(fill_method=None).fillna(0.0)
    average = values.rolling(window, min_periods=window).mean()
    position = values.gt(average).shift(1).fillna(False).astype(float)
    return asset_returns * position


def baseline_metric_helpers(
    prices: pd.DataFrame | Mapping[str, Iterable[Any]],
    *,
    periods_per_year: int = 252,
    moving_average_window: int = 200,
) -> dict[str, Any]:
    """Metrics for cash, per-symbol buy/hold, equal weight, and causal MA."""

    frame = (
        prices.copy()
        if isinstance(prices, pd.DataFrame)
        else pd.DataFrame({name: list(values) for name, values in prices.items()})
    )
    if frame.empty:
        raise ValueError("prices cannot be empty")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    asset_returns = frame.pct_change(fill_method=None).fillna(0.0)
    buy_hold = {
        str(name): return_metrics(
            asset_returns[name], periods_per_year=periods_per_year
        )
        for name in frame
    }
    equal_weight = equal_weight_returns(asset_returns)
    moving_average = pd.concat(
        [
            causal_moving_average_returns(
                frame[name], window=moving_average_window
            ).rename(name)
            for name in frame
        ],
        axis=1,
    ).mean(axis=1)
    return {
        "cash": return_metrics(
            np.zeros(len(frame)), periods_per_year=periods_per_year
        ),
        "buy_and_hold": buy_hold,
        "equal_weight_buy_and_hold": return_metrics(
            equal_weight, periods_per_year=periods_per_year
        ),
        "equal_weight_moving_average": return_metrics(
            moving_average, periods_per_year=periods_per_year
        ),
    }


def relative_baseline_metrics(
    strategy_returns: Iterable[Any],
    baseline_returns: Iterable[Any],
    *,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    strategy = pd.Series(list(strategy_returns), dtype=float)
    baseline = pd.Series(list(baseline_returns), dtype=float)
    aligned = pd.concat(
        [strategy.rename("strategy"), baseline.rename("baseline")],
        axis=1,
    ).dropna()
    if aligned.empty:
        raise ValueError("strategy and baseline have no aligned finite returns")
    excess = aligned["strategy"] - aligned["baseline"]
    tracking_error = float(excess.std(ddof=1))
    return {
        "strategy": return_metrics(
            aligned["strategy"], periods_per_year=periods_per_year
        ),
        "baseline": return_metrics(
            aligned["baseline"], periods_per_year=periods_per_year
        ),
        "excess_total_return": float(
            (1.0 + aligned["strategy"]).prod()
            - (1.0 + aligned["baseline"]).prod()
        ),
        "information_ratio": (
            float(excess.mean() / tracking_error * np.sqrt(periods_per_year))
            if tracking_error > 0
            else float("nan")
        ),
    }


__all__ = [
    "baseline_metric_helpers",
    "buy_and_hold_returns",
    "causal_moving_average_returns",
    "equal_weight_returns",
    "relative_baseline_metrics",
    "return_metrics",
]
