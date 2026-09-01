from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..data.base import normalize_bars
from ..factors.base import Factor, assert_factor_is_causal
from ..json_utils import json_safe
from .metrics import calculate_metrics


Rebalance = Literal["D", "W", "M"]


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """横截面 Top-N 组合回测的全部可复现参数。

    因子排序、调仓频率和摩擦成本都写入同一配置对象，避免结果只记录收益却无法
    回答“以什么成交规则得到该收益”的问题。
    """
    start_date: date
    end_date: date
    top_n: int = 10
    rebalance: Rebalance = "W"
    commission_rate: float = 0.0003
    minimum_commission: float = 5.0
    minimum_trade_notional: float = 1_000.0
    rebalance_tolerance: float = 0.001
    stamp_duty_rate: float = 0.0005
    historical_stamp_duty: bool = True
    slippage_rate: float = 0.0005
    initial_capital: float = 1_000_000.0
    max_stale_sessions: int = 20

    def __post_init__(self) -> None:
        if self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")
        if self.top_n < 1:
            raise ValueError("top_n must be at least 1")
        if self.rebalance not in {"D", "W", "M"}:
            raise ValueError("rebalance must be D, W, or M")
        for name in ("commission_rate", "stamp_duty_rate", "slippage_rate"):
            value = getattr(self, name)
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.minimum_commission < 0:
            raise ValueError("minimum_commission must be non-negative")
        if self.minimum_trade_notional < 0:
            raise ValueError("minimum_trade_notional must be non-negative")
        if not 0 <= self.rebalance_tolerance < 1:
            raise ValueError("rebalance_tolerance must be in [0, 1)")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.max_stale_sessions < 1:
            raise ValueError("max_stale_sessions must be at least 1")


def _scheduled_dates(dates: list[pd.Timestamp], frequency: Rebalance) -> set[pd.Timestamp]:
    """返回每个调仓周期的最后交易日，而非自然周/月的固定日期。

    A 股节假日和停牌会使自然日期不存在；使用实际可交易日期可保证信号日有效。
    """
    if frequency == "D":
        return set(dates)
    calendar = pd.DataFrame({"date": dates})
    if frequency == "W":
        keys = calendar["date"].dt.to_period("W-FRI")
    else:
        keys = calendar["date"].dt.to_period("M")
    return set(calendar.groupby(keys, sort=True)["date"].max())


def _price_limit(symbol: str, is_st: bool, trading_date: date) -> float:
    """按证券代码、ST 状态与制度变更日期推断日涨跌幅限制。"""
    if symbol.startswith(("300", "301")):
        return 0.20 if trading_date >= date(2020, 8, 24) else 0.10
    if symbol.startswith(("688", "689")):
        return 0.20
    if symbol.startswith(("4", "8", "9")):
        return 0.30
    if is_st:
        return 0.05
    return 0.10


def _row_number(row: pd.Series, column: str) -> float:
    execution_column = f"trade_{column}"
    key = execution_column if execution_column in row.index else column
    return float(
        pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
    )


def _row_flag(row: pd.Series, column: str, default: bool = False) -> bool:
    execution_column = f"trade_{column}"
    key = execution_column if execution_column in row.index else column
    value = row.get(key, default)
    return default if pd.isna(value) else bool(value)


def _reference_close(row: pd.Series) -> float:
    value = _row_number(row, "reference_close")
    return (
        value
        if np.isfinite(value) and value > 0
        else _row_number(row, "prev_close")
    )


def _rounded_limit_price(
    previous_close: float,
    limit: float,
    side: Literal["buy", "sell"],
) -> float:
    multiplier = Decimal("1") + (
        Decimal(str(limit)) if side == "buy" else -Decimal(str(limit))
    )
    return float(
        (Decimal(str(previous_close)) * multiplier).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
    )


def _blocked_reason(
    symbol: str,
    row: pd.Series | None,
    side: Literal["buy", "sell"],
    trading_date: pd.Timestamp,
) -> str | None:
    """判断订单是否因缺行情、停牌或开盘封板而无法模拟成交。

    日线模型在集合竞价开盘成交；缺少委托簿时，开盘价等于涨跌停价被保守
    视为不可成交，以避免假设订单一定能排在封板队列前方。
    """
    if row is None:
        return "missing_bar"
    open_price = _row_number(row, "open")
    volume = _row_number(row, "volume")
    if not np.isfinite(open_price) or open_price <= 0:
        return "missing_open"
    if not np.isfinite(volume) or volume <= 0:
        return "suspension"

    previous_close = _reference_close(row)
    if np.isfinite(previous_close) and previous_close > 0:
        is_st = _row_flag(row, "is_st") if _row_flag(row, "is_st_known") else False
        limit = _price_limit(symbol, is_st, trading_date.date())
        limit_price = _rounded_limit_price(previous_close, limit, side)
        if side == "buy" and open_price >= limit_price - 1e-8:
            return "sealed_limit_up"
        if side == "sell" and open_price <= limit_price + 1e-8:
            return "sealed_limit_down"
    return None


def execution_block_reason(
    symbol: str,
    row: pd.Series | None,
    side: Literal["buy", "sell"],
    trading_date: pd.Timestamp,
) -> str | None:
    """Return the shared suspension/price-limit execution block reason."""

    return _blocked_reason(symbol, row, side, trading_date)


def _commission(notional: float, config: BacktestConfig) -> float:
    if notional <= 0 or config.commission_rate <= 0:
        return 0.0
    return max(notional * config.commission_rate, config.minimum_commission)


def _meaningful_rebalance(
    current_quantity: float,
    desired_quantity: float,
    price: float,
    equity: float,
    config: BacktestConfig,
) -> bool:
    """过滤金额过小或权益占比过低的调仓，防止手续费主导回测结果。"""
    if not np.isfinite(price) or price <= 0 or equity <= 0:
        return False
    notional = abs(desired_quantity - current_quantity) * price
    return (
        notional >= config.minimum_trade_notional
        and notional / equity >= config.rebalance_tolerance
    )


def _stamp_duty_rate(config: BacktestConfig, trading_date: pd.Timestamp) -> float:
    """根据历史政策日期计算卖出印花税，或按用户选择使用固定费率。"""
    if config.stamp_duty_rate <= 0:
        return 0.0
    if not config.historical_stamp_duty:
        return config.stamp_duty_rate
    if trading_date.date() < date(2008, 9, 19):
        raise ValueError(
            "Historical stamp-duty modeling before 2008-09-19 is unsupported."
        )
    return 0.001 if trading_date.date() < date(2023, 8, 28) else 0.0005


def _simulated_open_time(value: pd.Timestamp) -> str:
    return f"{pd.Timestamp(value).date().isoformat()}T09:30:00+08:00"


def _simulated_close_time(value: pd.Timestamp) -> str:
    return f"{pd.Timestamp(value).date().isoformat()}T15:00:00+08:00"


def run_backtest(
    bars: pd.DataFrame,
    factor: Factor,
    config: BacktestConfig,
    benchmark_bars: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run an equal-weight Top-N simulation with T+1-open execution.

    A signal is calculated after close on T and traded at the next market day's
    open. The position therefore earns that session's open-to-close return.
    Orders are netted by symbol, and a stock bought on one session cannot be sold
    until the following trading session.
    """

    if bars.empty:
        raise ValueError("No bars are available for backtesting")
    panel = normalize_bars(bars).sort_values(["date", "symbol"]).reset_index(drop=True)
    factor.validate(panel)
    factor_values = factor.compute(panel)
    assert_factor_is_causal(factor, panel, factor_values)
    panel["factor"] = pd.to_numeric(factor_values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    panel["rank_score"] = panel["factor"] * factor.metadata.direction

    simulation = panel[
        (panel["date"] >= pd.Timestamp(config.start_date))
        & (panel["date"] <= pd.Timestamp(config.end_date))
    ].copy()
    dates = list(pd.DatetimeIndex(simulation["date"].drop_duplicates().sort_values()))
    if not dates:
        raise ValueError("No trading dates exist in the requested backtest period")

    model_warnings: list[str] = []
    st_known_column = (
        "trade_is_st_known"
        if "trade_is_st_known" in simulation.columns
        else "is_st_known"
    )
    if (
        st_known_column not in simulation.columns
        or not simulation[st_known_column].fillna(False).astype(bool).all()
    ):
        model_warnings.append(
            "Historical ST status is unavailable for part or all of the sample; "
            "standard board limits were used where status was unknown."
        )
    if "trade_close" not in simulation.columns:
        model_warnings.append(
            "No separate unadjusted execution fields were supplied; input prices "
            "were assumed to be unadjusted for suspension and price-limit checks."
        )
    model_warnings.extend(
        [
            "Without order-book data, an open at the rounded daily price limit is "
            "conservatively treated as unfillable.",
            "IPO and relisting no-limit windows are not modeled because listing-event "
            "history is unavailable.",
            "Portfolio accounting permits fractional adjusted units; A-share "
            "100-share buy-lot rounding is not modeled.",
        ]
    )

    schedule = _scheduled_dates(dates, config.rebalance)
    next_date = {dates[index]: dates[index + 1] for index in range(len(dates) - 1)}
    rows_by_date: dict[pd.Timestamp, dict[str, pd.Series]] = {}
    for trading_date, group in simulation.groupby("date", sort=True):
        rows_by_date[pd.Timestamp(trading_date)] = {
            str(row["symbol"]): row
            for _, row in group.sort_values("symbol").iterrows()
        }

    pending: dict[pd.Timestamp, dict[str, Any]] = {}
    holdings: dict[str, float] = {}
    last_prices: dict[str, float] = {}
    last_buy_dates: dict[str, pd.Timestamp] = {}
    stale_sessions: dict[str, int] = {}
    cash = float(config.initial_capital)
    equity_rows: list[dict[str, Any]] = []
    holding_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    blocked_trades: list[dict[str, Any]] = []
    total_commission = 0.0
    total_stamp_duty = 0.0
    total_slippage = 0.0

    for trading_date in dates:
        current_rows = rows_by_date.get(trading_date, {})
        # 开盘撮合前只能使用当日开盘价和此前收盘估值，不能用尚未发生的当日收盘价
        # 决定目标数量。开盘成交后，日终权益再使用当日收盘价，因此买入日自然
        # 包含从 T+1 开盘到收盘的收益。
        pre_trade_equity = cash + sum(
            quantity
            * (
                float(current_rows[symbol]["open"])
                if symbol in current_rows
                and np.isfinite(float(current_rows[symbol].get("open", float("nan"))))
                and float(current_rows[symbol].get("open", 0.0)) > 0
                else last_prices.get(symbol, 0.0)
            )
            for symbol, quantity in holdings.items()
        )
        daily_turnover_notional = 0.0
        order = pending.pop(trading_date, None)
        if order is not None and pre_trade_equity > 0:
            targets: list[str] = order["targets"]
            target_value = pre_trade_equity / len(targets) if targets else 0.0
            all_symbols = set(holdings) | set(targets)
            desired_quantity: dict[str, float] = {}
            rebalance_prices: dict[str, float] = {}
            for symbol in all_symbols:
                row = current_rows.get(symbol)
                open_price = (
                    float(row["open"])
                    if row is not None and np.isfinite(float(row["open"]))
                    else last_prices.get(symbol, float("nan"))
                )
                rebalance_prices[symbol] = open_price
                desired_quantity[symbol] = (
                    target_value / open_price
                    if symbol in targets
                    and np.isfinite(open_price)
                    and open_price > 0
                    else 0.0
                )

            sell_symbols = sorted(
                symbol
                for symbol in all_symbols
                if holdings.get(symbol, 0.0)
                > desired_quantity.get(symbol, 0.0) + 1e-12
                and _meaningful_rebalance(
                    holdings.get(symbol, 0.0),
                    desired_quantity.get(symbol, 0.0),
                    rebalance_prices.get(symbol, float("nan")),
                    pre_trade_equity,
                    config,
                )
            )
            buy_symbols = sorted(
                symbol
                for symbol in all_symbols
                if desired_quantity.get(symbol, 0.0)
                > holdings.get(symbol, 0.0) + 1e-12
                and _meaningful_rebalance(
                    holdings.get(symbol, 0.0),
                    desired_quantity.get(symbol, 0.0),
                    rebalance_prices.get(symbol, float("nan")),
                    pre_trade_equity,
                    config,
                )
            )

            for symbol in sell_symbols:
                row = current_rows.get(symbol)
                reason = (
                    "t_plus_one"
                    if last_buy_dates.get(symbol) == trading_date
                    else _blocked_reason(symbol, row, "sell", trading_date)
                )
                quantity = holdings.get(symbol, 0.0) - desired_quantity[symbol]
                if reason is not None:
                    blocked_trades.append(
                        {
                            "signal_date": order["signal_date"],
                            "signal_time": _simulated_close_time(
                                order["signal_date"]
                            ),
                            "date": trading_date,
                            "execution_time": _simulated_open_time(trading_date),
                            "symbol": symbol,
                            "side": "sell",
                            "reason": reason,
                            "requested_quantity": quantity,
                        }
                    )
                    continue
                open_price = float(row["open"])  # type: ignore[index]
                market_open = _row_number(row, "open")  # type: ignore[arg-type]
                execution_price = open_price * (1.0 - config.slippage_rate)
                market_execution_price = market_open * (
                    1.0 - config.slippage_rate
                )
                notional = quantity * execution_price
                estimated_market_shares = (
                    notional / market_execution_price
                    if market_execution_price > 0
                    else float("nan")
                )
                commission = _commission(notional, config)
                applied_stamp_duty_rate = _stamp_duty_rate(config, trading_date)
                stamp_duty = notional * applied_stamp_duty_rate
                slippage_cost = quantity * open_price * config.slippage_rate
                cash += notional - commission - stamp_duty
                remaining = holdings.get(symbol, 0.0) - quantity
                if remaining <= 1e-12:
                    holdings.pop(symbol, None)
                    stale_sessions.pop(symbol, None)
                    last_buy_dates.pop(symbol, None)
                else:
                    holdings[symbol] = remaining
                total_commission += commission
                total_stamp_duty += stamp_duty
                total_slippage += slippage_cost
                daily_turnover_notional += quantity * open_price
                trades.append(
                    {
                        "signal_date": order["signal_date"],
                        "signal_time": _simulated_close_time(order["signal_date"]),
                        "date": trading_date,
                        "execution_time": _simulated_open_time(trading_date),
                        "execution_session": "T+1 open",
                        "symbol": symbol,
                        "side": "sell",
                        "quantity": quantity,
                        "accounting_quantity": quantity,
                        "estimated_market_shares": estimated_market_shares,
                        "estimated_market_lots": estimated_market_shares / 100.0,
                        "open": open_price,
                        "market_open": market_open,
                        "market_reference_close": _reference_close(row),
                        "execution_price": execution_price,
                        "market_execution_price": market_execution_price,
                        "notional": notional,
                        "commission": commission,
                        "stamp_duty_rate": applied_stamp_duty_rate,
                        "stamp_duty": stamp_duty,
                        "slippage_cost": slippage_cost,
                    }
                )

            for symbol in buy_symbols:
                row = current_rows.get(symbol)
                reason = _blocked_reason(symbol, row, "buy", trading_date)
                requested_quantity = (
                    desired_quantity[symbol] - holdings.get(symbol, 0.0)
                )
                if reason is not None:
                    blocked_trades.append(
                        {
                            "signal_date": order["signal_date"],
                            "signal_time": _simulated_close_time(
                                order["signal_date"]
                            ),
                            "date": trading_date,
                            "execution_time": _simulated_open_time(trading_date),
                            "symbol": symbol,
                            "side": "buy",
                            "reason": reason,
                            "requested_quantity": requested_quantity,
                        }
                    )
                    continue
                open_price = float(row["open"])  # type: ignore[index]
                market_open = _row_number(row, "open")  # type: ignore[arg-type]
                execution_price = open_price * (1.0 + config.slippage_rate)
                if config.commission_rate > 0:
                    affordable = min(
                        max(
                            (cash - config.minimum_commission) / execution_price,
                            0.0,
                        ),
                        cash
                        / (
                            execution_price
                            * (1.0 + config.commission_rate)
                        ),
                    )
                else:
                    affordable = cash / execution_price
                quantity = min(requested_quantity, max(affordable, 0.0))
                if quantity <= 1e-12:
                    blocked_trades.append(
                        {
                            "signal_date": order["signal_date"],
                            "signal_time": _simulated_close_time(
                                order["signal_date"]
                            ),
                            "date": trading_date,
                            "execution_time": _simulated_open_time(trading_date),
                            "symbol": symbol,
                            "side": "buy",
                            "reason": "insufficient_cash",
                            "requested_quantity": requested_quantity,
                        }
                    )
                    continue
                notional = quantity * execution_price
                market_execution_price = market_open * (
                    1.0 + config.slippage_rate
                )
                estimated_market_shares = (
                    notional / market_execution_price
                    if market_execution_price > 0
                    else float("nan")
                )
                commission = _commission(notional, config)
                slippage_cost = quantity * open_price * config.slippage_rate
                cash -= notional + commission
                holdings[symbol] = holdings.get(symbol, 0.0) + quantity
                last_buy_dates[symbol] = trading_date
                stale_sessions[symbol] = 0
                total_commission += commission
                total_slippage += slippage_cost
                daily_turnover_notional += quantity * open_price
                trades.append(
                    {
                        "signal_date": order["signal_date"],
                        "signal_time": _simulated_close_time(order["signal_date"]),
                        "date": trading_date,
                        "execution_time": _simulated_open_time(trading_date),
                        "execution_session": "T+1 open",
                        "symbol": symbol,
                        "side": "buy",
                        "quantity": quantity,
                        "accounting_quantity": quantity,
                        "estimated_market_shares": estimated_market_shares,
                        "estimated_market_lots": estimated_market_shares / 100.0,
                        "open": open_price,
                        "market_open": market_open,
                        "market_reference_close": _reference_close(row),
                        "execution_price": execution_price,
                        "market_execution_price": market_execution_price,
                        "notional": notional,
                        "commission": commission,
                        "stamp_duty_rate": 0.0,
                        "stamp_duty": 0.0,
                        "slippage_cost": slippage_cost,
                    }
                )

        # 收盘后更新持仓估值。新仓从本日开盘成交，因此本日开盘到收盘的价格变化
        # 已反映在当日权益和收益中。
        for symbol, row in current_rows.items():
            close = float(row["close"])
            if np.isfinite(close) and close > 0:
                last_prices[symbol] = close
        for symbol in holdings:
            row = current_rows.get(symbol)
            has_valuation = (
                row is not None
                and np.isfinite(float(row.get("close", float("nan"))))
                and float(row.get("close", 0.0)) > 0
            )
            stale_sessions[symbol] = 0 if has_valuation else stale_sessions.get(
                symbol, 0
            ) + 1
            if stale_sessions[symbol] > config.max_stale_sessions:
                raise ValueError(
                    f"Held symbol {symbol} has no valuation bar for "
                    f"{stale_sessions[symbol]} market sessions as of "
                    f"{trading_date.date()}; the backtest was stopped instead of "
                    "carrying a stale price indefinitely."
                )

        equity = cash + sum(
            quantity * last_prices.get(symbol, 0.0)
            for symbol, quantity in holdings.items()
        )
        turnover = (
            daily_turnover_notional / pre_trade_equity
            if pre_trade_equity > 0
            else 0.0
        )
        equity_rows.append(
            {
                "date": trading_date,
                "equity": equity,
                "net_value": equity / config.initial_capital,
                "cash": cash,
                "turnover": turnover,
            }
        )
        for symbol, quantity in sorted(holdings.items()):
            close = last_prices.get(symbol, float("nan"))
            market_value = quantity * close
            current_row = current_rows.get(symbol)
            holding_rows.append(
                {
                    "date": trading_date,
                    "symbol": symbol,
                    "quantity": quantity,
                    "close": close,
                    "market_close": (
                        _row_number(current_row, "close")
                        if current_row is not None
                        else float("nan")
                    ),
                    "market_value": market_value,
                    "weight": market_value / equity if equity > 0 else float("nan"),
                    "stale_sessions": stale_sessions.get(symbol, 0),
                    "factor_value": (
                        current_row.get("factor")
                        if current_row is not None
                        else float("nan")
                    ),
                    "rank_score": (
                        current_row.get("rank_score")
                        if current_row is not None
                        else float("nan")
                    ),
                }
            )

        if trading_date in schedule and trading_date in next_date:
            signal_rows = simulation[
                (simulation["date"] == trading_date)
                & simulation["factor"].notna()
            ].sort_values(["rank_score", "symbol"], ascending=[False, True])
            targets = (
                signal_rows["symbol"].astype(str).head(config.top_n).tolist()
            )
            pending[next_date[trading_date]] = {
                "signal_date": trading_date,
                "targets": targets,
            }

    equity_curve = pd.DataFrame(equity_rows)
    equity_curve["daily_return"] = equity_curve["equity"].pct_change(
        fill_method=None
    )
    metrics = calculate_metrics(equity_curve, benchmark_bars)
    total_cost = total_commission + total_stamp_duty + total_slippage
    max_observed_stale = max(
        (int(row["stale_sessions"]) for row in holding_rows),
        default=0,
    )
    if max_observed_stale:
        model_warnings.append(
            f"At least one holding used a stale valuation for up to "
            f"{max_observed_stale} market session(s)."
        )
    summary = {
        key: metrics[key]
        for key in (
            "total_return",
            "annualized_return",
            "benchmark_return",
            "benchmark_annualized_return",
            "excess_return",
            "sharpe",
            "max_drawdown",
            "volatility",
        )
    }
    summary.update(
        {
            "turnover": float(equity_curve["turnover"].sum()),
            "trade_count": len(trades),
            "blocked_trade_count": len(blocked_trades),
            "total_cost": total_cost,
            "commission": total_commission,
            "minimum_commission": config.minimum_commission,
            "minimum_trade_notional": config.minimum_trade_notional,
            "rebalance_tolerance": config.rebalance_tolerance,
            "stamp_duty": total_stamp_duty,
            "stamp_duty_mode": (
                "historical" if config.historical_stamp_duty else "fixed"
            ),
            "slippage_cost": total_slippage,
            "max_stale_sessions_observed": max_observed_stale,
        }
    )

    benchmark_by_date = {
        pd.Timestamp(item["date"]).normalize(): item["net_value"]
        for item in metrics["benchmark_curve"]
    }
    output_curve = equity_curve.copy()
    output_curve["strategy"] = output_curve["net_value"]
    output_curve["benchmark"] = output_curve["date"].map(benchmark_by_date)
    annual_returns = [
        {**item, "strategy": item["return"]} for item in metrics["annual_returns"]
    ]
    result = {
        "factor_name": factor.metadata.name,
        "direction": factor.metadata.direction,
        "direction_label": factor.metadata.direction_label,
        "execution_policy": (
            "Signals use T close. Net orders execute at the next market day's open "
            "(T+1); returns begin at that open. Unadjusted market data controls "
            "suspension and rounded opening price-limit checks, while the selected "
            "price series controls return accounting. A stock cannot be sold until "
            "the trading session after its buy session."
        ),
        "model_warnings": list(dict.fromkeys(model_warnings)),
        "warnings": list(dict.fromkeys(model_warnings)),
        "summary": summary,
        "metrics": summary,
        "equity_curve": output_curve.to_dict(orient="records"),
        "drawdown": metrics["drawdown"],
        "annual_returns": annual_returns,
        "benchmark_curve": metrics["benchmark_curve"],
        "trades": trades,
        "holdings": holding_rows,
        "blocked_trades": blocked_trades,
    }
    return json_safe(result)
