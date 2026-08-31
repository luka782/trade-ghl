from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.backtest.engine import (
    BacktestConfig,
    _blocked_reason,
    _price_limit,
    _stamp_duty_rate,
    run_backtest,
)
from app.backtest.metrics import calculate_metrics
from app.data.base import normalize_bars
from app.factors.base import Factor, FactorMetadata


class SignalFactor(Factor):
    metadata = FactorMetadata(
        name="test_signal",
        description="Deterministic test signal.",
        lookback=0,
        required_columns=("signal",),
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        return pd.to_numeric(bars["signal"], errors="coerce")


class NegativeSignalFactor(Factor):
    metadata = FactorMetadata(
        name="negative_test_signal",
        description="Lower raw values rank higher.",
        lookback=0,
        required_columns=("signal",),
        direction=-1,
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        return pd.to_numeric(bars["signal"], errors="coerce")


def _panel(
    closes: dict[str, list[float]],
    signals: dict[str, list[float]],
    volumes: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=len(next(iter(closes.values()))))
    frames = []
    for symbol, values in closes.items():
        frame = pd.DataFrame(
            {
                "symbol": symbol,
                "date": dates,
                "open": values,
                "high": values,
                "low": values,
                "close": values,
                "volume": (
                    volumes[symbol]
                    if volumes and symbol in volumes
                    else [1000.0] * len(dates)
                ),
                "signal": signals[symbol],
                "is_st": False,
            }
        )
        frames.append(frame)
    return normalize_bars(pd.concat(frames, ignore_index=True))


def test_signal_from_t_executes_at_t_plus_one_close() -> None:
    bars = _panel(
        {"600001": [10.0, 10.8, 11.664, 11.664]},
        {"600001": [1.0, 1.0, 1.0, 1.0]},
    )
    dates = sorted(bars["date"].unique())
    result = run_backtest(
        bars,
        SignalFactor(),
        BacktestConfig(
            start_date=pd.Timestamp(dates[0]).date(),
            end_date=pd.Timestamp(dates[-1]).date(),
            top_n=1,
            rebalance="D",
            commission_rate=0,
            stamp_duty_rate=0,
            slippage_rate=0,
        ),
    )

    curve = result["equity_curve"]
    assert curve[0]["net_value"] == pytest.approx(1.0)
    assert curve[1]["net_value"] == pytest.approx(1.0)
    assert curve[2]["net_value"] == pytest.approx(1.08)
    first_trade = result["trades"][0]
    assert first_trade["signal_date"] == "2024-01-02T00:00:00"
    assert first_trade["date"] == "2024-01-03T00:00:00"
    assert first_trade["signal_time"] == "2024-01-02T15:00:00+08:00"
    assert first_trade["execution_time"] == "2024-01-03T15:00:00+08:00"
    assert first_trade["execution_session"] == "T+1 close"
    assert first_trade["estimated_market_shares"] > 0
    assert first_trade["estimated_market_lots"] > 0
    assert first_trade["execution_price"] == pytest.approx(10.8)


def test_negative_direction_ranks_low_raw_value_first_and_exposes_score() -> None:
    bars = _panel(
        {
            "600001": [10.0, 10.0, 10.0],
            "600002": [10.0, 10.0, 10.0],
        },
        {
            "600001": [1.0, 1.0, 1.0],
            "600002": [5.0, 5.0, 5.0],
        },
    )
    dates = sorted(bars["date"].unique())

    result = run_backtest(
        bars,
        NegativeSignalFactor(),
        BacktestConfig(
            start_date=pd.Timestamp(dates[0]).date(),
            end_date=pd.Timestamp(dates[-1]).date(),
            top_n=1,
            rebalance="D",
            commission_rate=0,
            stamp_duty_rate=0,
            slippage_rate=0,
        ),
    )

    assert result["direction"] == -1
    assert result["trades"][0]["side"] == "buy"
    assert result["trades"][0]["symbol"] == "600001"
    first_holding = result["holdings"][0]
    assert first_holding["symbol"] == "600001"
    assert first_holding["factor_value"] == pytest.approx(1.0)
    assert first_holding["rank_score"] == pytest.approx(-1.0)


def test_daily_rebalance_never_sells_a_position_on_its_buy_date() -> None:
    bars = _panel(
        {
            "600001": [10.0, 10.0, 10.0, 10.0],
            "600002": [10.0, 10.0, 10.0, 10.0],
        },
        {
            "600001": [2.0, 1.0, 1.0, 1.0],
            "600002": [1.0, 2.0, 2.0, 2.0],
        },
    )
    dates = sorted(bars["date"].unique())
    result = run_backtest(
        bars,
        SignalFactor(),
        BacktestConfig(
            start_date=pd.Timestamp(dates[0]).date(),
            end_date=pd.Timestamp(dates[-1]).date(),
            top_n=1,
            rebalance="D",
            commission_rate=0,
            stamp_duty_rate=0,
            slippage_rate=0,
        ),
    )
    buy_dates: dict[str, pd.Timestamp] = {}
    for trade in result["trades"]:
        trade_date = pd.Timestamp(trade["date"])
        if trade["side"] == "buy":
            buy_dates[trade["symbol"]] = trade_date
        else:
            assert trade_date > buy_dates[trade["symbol"]]


def test_single_asset_daily_rebalance_ignores_micro_orders() -> None:
    closes = [10.0 * (1.001**index) for index in range(80)]
    bars = _panel(
        {"515080": closes},
        {"515080": [1.0] * len(closes)},
    )
    dates = sorted(bars["date"].unique())
    result = run_backtest(
        bars,
        SignalFactor(),
        BacktestConfig(
            start_date=pd.Timestamp(dates[0]).date(),
            end_date=pd.Timestamp(dates[-1]).date(),
            top_n=1,
            rebalance="D",
            commission_rate=0.0003,
            minimum_commission=5.0,
            stamp_duty_rate=0,
            slippage_rate=0.0005,
        ),
    )
    assert [trade["side"] for trade in result["trades"]] == ["buy"]
    assert result["summary"]["commission"] > 0
    assert result["summary"]["trade_count"] == 1


def test_costs_stamp_duty_and_suspension_block_are_applied() -> None:
    bars = _panel(
        {
            "600001": [10.0, 10.2, 10.4, 10.5],
            "600002": [10.0, 10.1, 10.2, 10.3],
        },
        {
            "600001": [2.0, 1.0, 1.0, 1.0],
            "600002": [1.0, 2.0, 2.0, 2.0],
        },
        {
            "600001": [1000.0, 1000.0, 1000.0, 1000.0],
            "600002": [1000.0, 1000.0, 0.0, 1000.0],
        },
    )
    dates = sorted(bars["date"].unique())
    result = run_backtest(
        bars,
        SignalFactor(),
        BacktestConfig(
            start_date=pd.Timestamp(dates[0]).date(),
            end_date=pd.Timestamp(dates[-1]).date(),
            top_n=1,
            rebalance="D",
            commission_rate=0.001,
            stamp_duty_rate=0.001,
            slippage_rate=0.001,
        ),
    )

    assert result["summary"]["commission"] > 0
    assert result["summary"]["stamp_duty"] > 0
    assert result["summary"]["slippage_cost"] > 0
    assert result["summary"]["total_cost"] > 0
    assert any(
        item["symbol"] == "600002"
        and item["side"] == "buy"
        and item["reason"] == "suspension"
        for item in result["blocked_trades"]
    )
    sides_per_day_symbol: dict[tuple[str, str], set[str]] = {}
    for trade in result["trades"]:
        sides_per_day_symbol.setdefault(
            (trade["date"], trade["symbol"]), set()
        ).add(trade["side"])
    assert all(len(sides) == 1 for sides in sides_per_day_symbol.values())


def test_sealed_limit_up_and_down_block_trades() -> None:
    bars = _panel(
        {
            "600001": [10.0, 10.0, 9.0, 9.0],
            "600002": [10.0, 10.0, 11.0, 11.0],
        },
        {
            "600001": [2.0, 1.0, 1.0, 1.0],
            "600002": [1.0, 2.0, 2.0, 2.0],
        },
    )
    dates = sorted(bars["date"].unique())
    result = run_backtest(
        bars,
        SignalFactor(),
        BacktestConfig(
            start_date=pd.Timestamp(dates[0]).date(),
            end_date=pd.Timestamp(dates[-1]).date(),
            top_n=1,
            rebalance="D",
            commission_rate=0,
            stamp_duty_rate=0,
            slippage_rate=0,
        ),
    )
    reasons = {(item["symbol"], item["side"], item["reason"]) for item in result["blocked_trades"]}
    assert ("600001", "sell", "sealed_limit_down") in reasons
    assert ("600002", "buy", "sealed_limit_up") in reasons


def test_price_limits_use_raw_prices_rounding_and_historical_rules() -> None:
    row = pd.Series(
        {
            "close": 5.50,
            "prev_close": 5.00,
            "volume": 1_000,
            "trade_close": 10.50,
            "trade_prev_close": 10.00,
            "trade_volume": 100_000,
            "trade_is_st": False,
            "trade_is_st_known": True,
        }
    )
    trading_date = pd.Timestamp("2024-01-03")
    assert _blocked_reason("600001", row, "buy", trading_date) is None
    row["trade_volume"] = 0
    assert _blocked_reason("600001", row, "buy", trading_date) == "suspension"
    row["trade_volume"] = 100_000

    row["trade_prev_close"] = 10.03
    row["trade_close"] = 11.03
    assert (
        _blocked_reason("600001", row, "buy", trading_date)
        == "sealed_limit_up"
    )
    row["trade_prev_close"] = 10.00
    row["trade_reference_close"] = 8.50
    row["trade_close"] = 8.60
    assert _blocked_reason("600001", row, "sell", trading_date) is None
    assert _price_limit("300001", False, date(2020, 8, 21)) == 0.10
    assert _price_limit("300001", False, date(2020, 8, 24)) == 0.20


def test_unknown_st_status_is_not_silently_treated_as_known() -> None:
    row = pd.Series(
        {
            "close": 10.50,
            "prev_close": 10.00,
            "volume": 100_000,
            "is_st": True,
            "is_st_known": False,
        }
    )
    trading_date = pd.Timestamp("2024-01-03")
    assert _blocked_reason("600001", row, "buy", trading_date) is None
    row["is_st_known"] = True
    assert (
        _blocked_reason("600001", row, "buy", trading_date)
        == "sealed_limit_up"
    )


def test_minimum_commission_is_applied_per_order() -> None:
    bars = _panel(
        {"600001": [10.0, 10.0]},
        {"600001": [1.0, 1.0]},
    )
    dates = sorted(bars["date"].unique())
    result = run_backtest(
        bars,
        SignalFactor(),
        BacktestConfig(
            start_date=pd.Timestamp(dates[0]).date(),
            end_date=pd.Timestamp(dates[-1]).date(),
            top_n=1,
            rebalance="D",
            commission_rate=0.001,
            minimum_commission=5.0,
            stamp_duty_rate=0,
            slippage_rate=0,
            initial_capital=1_000,
        ),
    )
    assert result["summary"]["commission"] == pytest.approx(5.0)


def test_historical_stamp_duty_switches_on_policy_date() -> None:
    config = BacktestConfig(
        start_date=date(2023, 1, 1),
        end_date=date(2024, 1, 1),
        stamp_duty_rate=0.0005,
        historical_stamp_duty=True,
    )
    assert _stamp_duty_rate(config, pd.Timestamp("2023-08-25")) == 0.001
    assert _stamp_duty_rate(config, pd.Timestamp("2023-08-28")) == 0.0005
    fixed = BacktestConfig(
        start_date=date(2023, 1, 1),
        end_date=date(2024, 1, 1),
        stamp_duty_rate=0.0008,
        historical_stamp_duty=False,
    )
    assert _stamp_duty_rate(fixed, pd.Timestamp("2023-08-25")) == 0.0008


def test_long_stale_holding_stops_instead_of_using_last_price_forever() -> None:
    dates = pd.bdate_range("2024-01-02", periods=7)
    held = pd.DataFrame(
        {
            "symbol": "600001",
            "date": dates[:2],
            "open": [10.0, 10.0],
            "high": [10.0, 10.0],
            "low": [10.0, 10.0],
            "close": [10.0, 10.0],
            "volume": [1000.0, 1000.0],
            "signal": [2.0, 2.0],
        }
    )
    calendar_anchor = pd.DataFrame(
        {
            "symbol": "600002",
            "date": dates,
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 1000.0,
            "signal": 1.0,
        }
    )
    bars = normalize_bars(pd.concat([held, calendar_anchor], ignore_index=True))
    with pytest.raises(ValueError, match="stale price indefinitely"):
        run_backtest(
            bars,
            SignalFactor(),
            BacktestConfig(
                start_date=dates[0].date(),
                end_date=dates[-1].date(),
                top_n=1,
                rebalance="D",
                commission_rate=0,
                stamp_duty_rate=0,
                slippage_rate=0,
                max_stale_sessions=2,
            ),
        )


def test_metrics_include_drawdown_volatility_and_null_benchmark() -> None:
    curve = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=3),
            "equity": [100.0, 110.0, 99.0],
        }
    )
    metrics = calculate_metrics(curve)
    assert metrics["total_return"] == pytest.approx(-0.01)
    assert metrics["max_drawdown"] == pytest.approx(-0.10)
    assert metrics["volatility"] > 0
    assert metrics["benchmark_return"] is None
    assert len(metrics["drawdown"]) == 3
    assert metrics["annual_returns"]
