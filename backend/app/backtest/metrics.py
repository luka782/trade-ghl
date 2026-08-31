from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..json_utils import json_safe


def _annualized_return(total_return: float, dates: pd.Series) -> float:
    """按实际自然日跨度折算年化收益，避免把非交易日误当成收益期。"""
    if len(dates) < 2:
        return float("nan")
    elapsed_days = (pd.Timestamp(dates.iloc[-1]) - pd.Timestamp(dates.iloc[0])).days
    if elapsed_days <= 0 or 1.0 + total_return <= 0:
        return float("nan")
    return (1.0 + total_return) ** (365.25 / elapsed_days) - 1.0


def calculate_metrics(
    equity_curve: pd.DataFrame,
    benchmark_bars: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """从每日权益曲线计算收益、风险、回撤和同口径基准对照。

    指标只描述已模拟的净值路径，不对策略的统计显著性作结论；后者由验证模块
    的 Walk-Forward、DSR/PBO 等诊断完成。
    """
    if equity_curve.empty:
        raise ValueError("Equity curve is empty")

    curve = equity_curve.sort_values("date").copy()
    curve["date"] = pd.to_datetime(curve["date"])
    curve["equity"] = pd.to_numeric(curve["equity"], errors="coerce")
    curve = curve.dropna(subset=["equity"])
    if curve.empty:
        raise ValueError("Equity curve has no finite values")

    initial_equity = float(curve["equity"].iloc[0])
    total_return = (
        float(curve["equity"].iloc[-1] / initial_equity - 1.0)
        if initial_equity > 0
        else float("nan")
    )
    daily_returns = curve["equity"].pct_change(fill_method=None).dropna()
    return_standard_deviation = (
        float(daily_returns.std(ddof=1)) if len(daily_returns) > 1 else 0.0
    )
    volatility = return_standard_deviation * np.sqrt(252.0)
    sharpe = (
        float(daily_returns.mean()) / return_standard_deviation * np.sqrt(252.0)
        if return_standard_deviation > 0
        else float("nan")
    )

    running_peak = curve["equity"].cummax()
    # 回撤相对于历史最高权益计算，因此最大回撤永远为 0 或负数。
    curve["drawdown"] = curve["equity"] / running_peak - 1.0
    maximum_drawdown = float(curve["drawdown"].min())

    annual_rows: list[dict[str, Any]] = []
    for year, group in curve.groupby(curve["date"].dt.year, sort=True):
        first = float(group["equity"].iloc[0])
        last = float(group["equity"].iloc[-1])
        annual_rows.append(
            {
                "year": int(year),
                "return": last / first - 1.0 if first > 0 else float("nan"),
            }
        )

    benchmark_total = float("nan")
    benchmark_annualized = float("nan")
    benchmark_curve: list[dict[str, Any]] = []
    if benchmark_bars is not None and not benchmark_bars.empty:
        # 以策略可评价日期对齐基准。允许基准偶尔缺行前向填充，但不会为策略
        # 缺失行情编造净值。
        benchmark = benchmark_bars.loc[:, ["date", "close"]].copy()
        benchmark["date"] = pd.to_datetime(benchmark["date"])
        benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
        benchmark = (
            benchmark.dropna()
            .sort_values("date")
            .drop_duplicates("date", keep="last")
            .set_index("date")
        )
        aligned = benchmark["close"].reindex(
            pd.DatetimeIndex(curve["date"]), method="ffill"
        )
        aligned = aligned.dropna()
        if len(aligned) >= 1 and float(aligned.iloc[0]) > 0:
            normalized = aligned / float(aligned.iloc[0])
            benchmark_total = float(normalized.iloc[-1] - 1.0)
            benchmark_dates = pd.Series(aligned.index)
            benchmark_annualized = _annualized_return(
                benchmark_total, benchmark_dates
            )
            benchmark_curve = [
                {"date": index, "net_value": value}
                for index, value in normalized.items()
            ]

    result = {
        "total_return": total_return,
        "annualized_return": _annualized_return(total_return, curve["date"]),
        "benchmark_return": benchmark_total,
        "benchmark_annualized_return": benchmark_annualized,
        "excess_return": (
            total_return - benchmark_total
            if np.isfinite(benchmark_total)
            else float("nan")
        ),
        "sharpe": sharpe,
        "max_drawdown": maximum_drawdown,
        "volatility": volatility,
        "drawdown": [
            {"date": row.date, "drawdown": row.drawdown}
            for row in curve[["date", "drawdown"]].itertuples(index=False)
        ],
        "annual_returns": annual_rows,
        "benchmark_curve": benchmark_curve,
    }
    return json_safe(result)
