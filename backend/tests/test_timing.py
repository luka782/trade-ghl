from __future__ import annotations

import pandas as pd
import pytest

from app.timing import TimingConfig, run_timing
from app.timing.position_sizing import calculate_position_size


def _frame(
    scores: list[float],
    *,
    adjusted: list[float] | None = None,
    raw: list[float] | None = None,
    adjusted_opens: list[float] | None = None,
    raw_opens: list[float] | None = None,
    trends: list[float] | None = None,
    start: str = "2024-01-02",
    volume: list[float] | None = None,
    price_positions: list[float] | None = None,
    entry_scores: list[float] | None = None,
    exit_scores: list[float] | None = None,
) -> pd.DataFrame:
    size = len(scores)
    adjusted = adjusted or [10.0] * size
    raw = raw or list(adjusted)
    adjusted_opens = adjusted_opens or list(adjusted)
    raw_opens = raw_opens or list(raw)
    trends = trends or [1.0] * size
    volume = volume or [100_000.0] * size
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range(start, periods=size),
            "open": adjusted_opens,
            "high": adjusted,
            "low": adjusted,
            "close": adjusted,
            "trade_open": raw_opens,
            "trade_high": raw,
            "trade_low": raw,
            "trade_close": raw,
            "trade_volume": volume,
            "trade_reference_close": raw,
            "composite_score": scores,
            "trend_score": trends,
        }
    )
    if price_positions is not None:
        frame["timing_price_position_60"] = price_positions
    if entry_scores is not None:
        frame["entry_score"] = entry_scores
    if exit_scores is not None:
        frame["exit_score"] = exit_scores
    return frame


def _zero_cost(**overrides: object) -> TimingConfig:
    values: dict[str, object] = {
        "commission_rate": 0.0,
        "minimum_commission": 0.0,
        "slippage": 0.0,
        "position_sizing": "full",
        "minimum_holding_sessions": 0,
        "cooldown_sessions": 0,
    }
    values.update(overrides)
    return TimingConfig(**values)


def test_threshold_crossing_t_plus_one_lot_rounding_and_no_repeated_buys() -> None:
    frame = _frame([0.6, 0.7, 0.9, 0.95, 0.8], raw=[11.0] * 5)
    result = run_timing(
        frame,
        _zero_cost(initial_capital=10_050.0, lot_size=100),
    )

    assert [trade["side"] for trade in result["trades"]] == ["buy"]
    trade = result["trades"][0]
    assert trade["signal_time"] == "2024-01-03T15:00:00+08:00"
    assert trade["execution_time"] == "2024-01-04T09:30:00+08:00"
    assert trade["shares"] == 900
    assert trade["lots"] == 9
    assert len([signal for signal in result["signals"] if signal["side"] == "buy"]) == 1


def test_holding_return_starts_at_t_plus_one_open() -> None:
    result = run_timing(
        _frame(
            [0.6, 0.8, 0.8],
            adjusted=[10.0, 10.0, 12.0],
            raw=[10.0, 10.0, 12.0],
            adjusted_opens=[10.0, 10.0, 10.0],
            raw_opens=[10.0, 10.0, 10.0],
        ),
        _zero_cost(),
    )

    assert result["trades"][0]["raw_price"] == pytest.approx(10.0)
    assert result["equity_curve"][2]["net_value"] == pytest.approx(1.2)


def test_buy_day_close_signal_can_sell_only_at_next_session_open() -> None:
    result = run_timing(
        _frame([0.6, 0.8, -0.1, -0.1]),
        _zero_cost(minimum_holding_sessions=0),
    )

    buy, sell = result["trades"]
    assert buy["execution_date"] == "2024-01-04T00:00:00"
    assert sell["signal_date"] == "2024-01-04T00:00:00"
    assert sell["execution_date"] == "2024-01-05T00:00:00"
    assert sell["execution_time"] == "2024-01-05T09:30:00+08:00"


def test_minimum_trade_notional_blocks_small_buy() -> None:
    result = run_timing(
        _frame([0.6, 0.8, 0.9], raw=[5.0, 5.0, 5.0]),
        _zero_cost(initial_capital=900.0, lot_size=100),
    )

    assert result["trades"] == []
    assert result["blocked_orders"][0]["reason"] == "below_minimum_trade_notional"


@pytest.mark.parametrize(
    ("adjusted", "expected_reason"),
    [
        ([10.0, 10.0, 10.0, 9.0, 9.0], "fixed_stop"),
        ([10.0, 10.0, 10.0, 12.0, 10.5, 10.5], "trailing_stop"),
    ],
)
def test_fixed_and_trailing_stops_use_adjusted_prices(
    adjusted: list[float], expected_reason: str
) -> None:
    result = run_timing(
        _frame([0.6, 0.8] + [0.8] * (len(adjusted) - 2), adjusted=adjusted),
        _zero_cost(),
    )

    assert [trade["side"] for trade in result["trades"]] == ["buy", "sell"]
    assert result["trades"][1]["reason"] == expected_reason


def test_max_holding_and_cooldown_state_transitions() -> None:
    max_hold = run_timing(
        _frame([0.6, 0.8, 0.8, 0.8, 0.8, 0.8]),
        _zero_cost(max_holding_sessions=2),
    )
    assert max_hold["trades"][1]["reason"] == "max_holding"
    assert max_hold["trades"][1]["holding_sessions"] == 3

    cooldown = run_timing(
        _frame([0.6, 0.8, 0.8, -0.1, 0.6, 0.8, 0.6, 0.8, 0.8]),
        _zero_cost(cooldown_sessions=2),
    )
    assert [trade["side"] for trade in cooldown["trades"]] == [
        "buy",
        "sell",
        "buy",
    ]
    assert any(order["reason"] == "cooldown" for order in cooldown["blocked_orders"])


def test_etf_has_no_stamp_duty_and_stock_uses_historical_schedule() -> None:
    scores = [0.6, 0.8, 0.8, -0.1, -0.1]
    etf = run_timing(
        _frame(scores, start="2023-08-21"),
        _zero_cost(is_etf=True),
    )
    old_stock = run_timing(
        _frame(scores, start="2023-08-21"),
        _zero_cost(is_etf=False),
    )
    new_stock = run_timing(
        _frame(scores, start="2023-09-04"),
        _zero_cost(is_etf=False),
    )

    assert etf["trades"][1]["stamp_duty_rate"] == 0.0
    assert old_stock["trades"][1]["execution_time"].startswith("2023-08-25")
    assert old_stock["trades"][1]["stamp_duty_rate"] == pytest.approx(0.001)
    assert new_stock["trades"][1]["stamp_duty_rate"] == pytest.approx(0.0005)


def test_factor_contributions_are_preserved_in_trace_and_signals() -> None:
    frame = _frame([0.6, 0.8, 0.8])
    frame["contribution_momentum"] = [0.1, 0.2, 0.3]
    frame["contribution_value"] = [0.4, 0.5, 0.6]
    frame["factor_momentum"] = [0.01, 0.02, 0.03]
    frame["normalized_momentum"] = [0.5, 0.8, 1.0]

    result = run_timing(frame, _zero_cost())

    signal = result["signals"][0]
    assert signal["factor_contributions"] == {
        "contribution_momentum": 0.2,
        "contribution_value": 0.5,
    }
    assert signal["contribution_momentum"] == pytest.approx(0.2)
    assert signal["factor_details"]["factor_momentum"] == pytest.approx(0.02)
    assert signal["factor_details"]["normalized_momentum"] == pytest.approx(0.8)
    assert result["trades"][0]["factor_details"]["factor_momentum"] == pytest.approx(
        0.02
    )
    assert signal["reason_code"].startswith("买入_")
    assert result["score_trace"][1]["contribution_value"] == pytest.approx(0.5)


def test_future_rows_do_not_change_earlier_results() -> None:
    frame = _frame([0.6, 0.8, 0.8, 0.8, -0.1, -0.1, 0.9, -0.8])
    prefix = frame.iloc[:6].copy()

    short = run_timing(prefix, _zero_cost())
    full = run_timing(frame, _zero_cost())

    assert full["equity_curve"][:6] == short["equity_curve"]
    cutoff = "2024-01-09T15:00:00+08:00"
    assert [
        trade for trade in full["trades"] if trade["execution_time"] <= cutoff
    ] == short["trades"]


def test_long_run_never_generates_micro_or_partial_sell_orders() -> None:
    size = 300
    scores = [0.6, 0.8, 0.8] + [-0.1] * (size - 3)
    result = run_timing(_frame(scores), _zero_cost())

    assert [trade["side"] for trade in result["trades"]] == ["buy", "sell"]
    assert result["trades"][1]["shares"] == result["trades"][0]["shares"]
    assert len(result["blocked_orders"]) < 3


def test_stale_adjusted_bars_trigger_t_plus_one_exit() -> None:
    frame = _frame([0.6, 0.8, 0.8, 0.8, 0.8, 0.8])
    frame.loc[3:4, ["open", "high", "low", "close"]] = float("nan")

    result = run_timing(frame, _zero_cost(max_stale_sessions=1))

    assert [trade["side"] for trade in result["trades"]] == ["buy", "sell"]
    assert result["trades"][1]["reason"] == "stale_data"
    assert pd.Timestamp(result["trades"][1]["execution_date"]) > pd.Timestamp(
        result["trades"][1]["signal_date"]
    )


def test_timing_uses_shared_unadjusted_price_limit_check() -> None:
    frame = _frame([0.6, 0.8, 0.9], raw=[10.0, 10.0, 11.0])
    frame["symbol"] = "600001"
    frame["trade_reference_close"] = [10.0, 10.0, 10.0]

    result = run_timing(frame, _zero_cost())

    assert result["trades"] == []
    assert result["blocked_orders"][0]["reason"] == "sealed_limit_up"


def test_mean_reversion_buys_after_low_recovery_and_sells_after_high_reversal() -> None:
    positions = [
        0.30,
        0.18,
        0.22,
        0.26,
        0.45,
        0.82,
        0.78,
        0.74,
        0.60,
    ]
    result = run_timing(
        _frame(
            [0.0] * len(positions),
            price_positions=positions,
        ),
        _zero_cost(
            timing_style="mean_reversion",
            minimum_holding_sessions=2,
        ),
    )

    assert [trade["side"] for trade in result["trades"]] == ["buy", "sell"]
    buy, sell = result["trades"]
    assert buy["reason"] == "low_zone_recovery"
    assert sell["reason"] == "high_zone_reversal"
    assert pd.Timestamp(buy["execution_date"]) > pd.Timestamp(buy["signal_date"])
    assert pd.Timestamp(sell["execution_date"]) > pd.Timestamp(sell["signal_date"])
    assert buy["timing_price_position_60"] == pytest.approx(0.26)
    assert sell["timing_price_position_60"] == pytest.approx(0.74)


def test_mean_reversion_rejects_missing_causal_price_position() -> None:
    with pytest.raises(ValueError, match="timing_price_position_60"):
        run_timing(
            _frame([0.0, 0.0, 0.0]),
            _zero_cost(timing_style="mean_reversion"),
        )


def test_factor_dual_uses_entry_confirmation_and_sells_while_still_high() -> None:
    positions = [0.30, 0.18, 0.22, 0.26, 0.50, 0.66, 0.68, 0.60]
    entry_scores = [0.0, -0.4, 0.3, 0.7, 0.4, 0.1, 0.0, -0.1]
    exit_scores = [0.0, 0.0, 0.1, 0.1, 0.2, 0.75, 0.8, 0.8]
    result = run_timing(
        _frame(
            [0.0] * len(positions),
            price_positions=positions,
            entry_scores=entry_scores,
            exit_scores=exit_scores,
        ),
        _zero_cost(
            timing_style="factor_dual",
            entry_score_threshold=0.6,
            exit_score_threshold=0.7,
            minimum_holding_sessions=2,
        ),
    )

    assert [trade["side"] for trade in result["trades"]] == ["buy", "sell"]
    buy, sell = result["trades"]
    assert buy["reason"] == "entry_factor_confirmation"
    assert sell["reason"] == "exit_factor_risk"
    assert buy["entry_score"] == pytest.approx(0.7)
    assert sell["exit_score"] == pytest.approx(0.8)
    assert sell["timing_price_position_60"] == pytest.approx(0.68)
    assert pd.Timestamp(sell["execution_date"]) > pd.Timestamp(sell["signal_date"])


def test_factor_dual_low_setup_expires_before_late_recovery() -> None:
    positions = [0.18, 0.19, 0.22, 0.24, 0.30, 0.35]
    entry_scores = [-1.0, -0.8, 0.1, 0.3, 0.8, 1.0]
    result = run_timing(
        _frame(
            [0.0] * len(positions),
            price_positions=positions,
            entry_scores=entry_scores,
            exit_scores=[0.0] * len(positions),
        ),
        _zero_cost(
            timing_style="factor_dual",
            setup_expiry_sessions=2,
            entry_score_threshold=0.4,
        ),
    )

    assert result["trades"] == []
    assert result["signals"] == []
    assert result["score_trace"][-1]["low_zone_armed"] is False


def _regime_frame(
    positions: list[float],
    rsi_values: list[float],
    percent_b: list[float],
    regimes: list[str],
    entry_final: list[float],
    exit_final: list[float],
) -> pd.DataFrame:
    frame = _frame(
        [0.0] * len(positions),
        price_positions=positions,
        entry_scores=entry_final,
        exit_scores=exit_final,
    )
    frame["entry_score_final"] = entry_final
    frame["exit_score_final"] = exit_final
    frame["rsi_14"] = rsi_values
    frame["bollinger_percent_b_20"] = percent_b
    frame["market_regime"] = regimes
    frame["ma_200"] = 10.0
    frame["ma_slope_20"] = 0.01
    frame["distance_to_ma_200"] = 0.01
    frame["bollinger_mid_20"] = 10.0
    frame["bollinger_upper_20"] = 11.0
    frame["bollinger_lower_20"] = 9.0
    frame["bollinger_bandwidth_20"] = 0.2
    return frame


def test_regime_reversion_confirms_entry_and_exits_on_high_risk() -> None:
    frame = _regime_frame(
        [0.30, 0.18, 0.22, 0.28, 0.40, 0.70, 0.65],
        [40, 25, 32, 35, 50, 65, 60],
        [0.3, -0.2, 0.1, 0.2, 0.5, 0.9, 0.8],
        ["sideways"] * 7,
        [0.0, -0.2, 0.2, 0.6, 0.4, 0.1, 0.0],
        [0.0, -0.3, -0.2, -0.1, 0.2, 0.7, 0.6],
    )
    result = run_timing(
        frame,
        _zero_cost(
            timing_style="regime_reversion",
            entry_score_threshold=0.4,
            exit_score_threshold=0.5,
            minimum_holding_sessions=1,
        ),
    )
    assert [trade["side"] for trade in result["trades"]] == ["buy", "sell"]
    assert result["trades"][0]["reason"] == "regime_entry_confirmation"
    assert result["trades"][1]["reason"] == "regime_exit_risk"
    assert result["summary"]["entry_funnel"]["candidate_zone"] > 0
    assert result["summary"]["entry_funnel"]["orders_created"] == 1
    assert result["summary"]["entry_funnel"]["orders_filled"] == 1


def test_regime_reversion_rejects_downtrend_entry() -> None:
    frame = _regime_frame(
        [0.18, 0.20, 0.25, 0.30],
        [25, 32, 35, 40],
        [-0.2, 0.1, 0.2, 0.3],
        ["downtrend"] * 4,
        [-0.2, 0.3, 0.7, 0.8],
        [0.0] * 4,
    )
    result = run_timing(
        frame,
        _zero_cost(timing_style="regime_reversion"),
    )
    assert result["trades"] == []


def test_confirmation_count_trades_when_legacy_all_remains_sparse() -> None:
    frame = _regime_frame(
        [0.30, 0.18, 0.19, 0.30],
        [40, 25, 35, 40],
        [0.3, -0.2, -0.1, 0.1],
        ["sideways"] * 4,
        [-0.3, -0.2, 0.6, 0.7],
        [0.0] * 4,
    )
    modern = run_timing(
        frame,
        _zero_cost(
            timing_style="regime_reversion",
            regime_confirmation_required=2,
        ),
    )
    legacy = run_timing(
        frame,
        _zero_cost(
            timing_style="regime_reversion_legacy",
            regime_entry_mode="legacy_all",
        ),
    )

    assert [trade["side"] for trade in modern["trades"]] == ["buy"]
    assert legacy["trades"] == []
    assert modern["summary"]["entry_funnel"]["confirmation_passed"] > 0


def test_donchian_and_ma_cta_strategies_use_queued_open_orders() -> None:
    donchian = _frame(
        [0.0] * 8,
        adjusted=[9, 9, 10, 11, 11, 10, 8, 8],
        adjusted_opens=[9, 9, 10, 11, 11, 10, 8, 8],
    )
    donchian["atr_20"] = 1.0
    donchian["donchian_upper"] = [float("nan"), float("nan"), 10, 10, 11, 11, 11, 11]
    donchian["donchian_lower"] = [float("nan"), float("nan"), 8, 8, 9, 9, 9, 9]
    donchian_result = run_timing(
        donchian,
        _zero_cost(timing_style="donchian_atr"),
    )

    assert [trade["side"] for trade in donchian_result["trades"]] == [
        "buy",
        "sell",
    ]
    assert donchian_result["trades"][0]["reason"] == "donchian_breakout"
    assert donchian_result["trades"][1]["reason"] in {
        "donchian_exit",
        "atr_initial_stop",
        "atr_trailing_stop",
    }
    assert donchian_result["trades"][0]["execution_time"].endswith(
        "09:30:00+08:00"
    )

    moving_average = _frame([0.0] * 7)
    moving_average["atr_20"] = 0.5
    moving_average["ma_fast"] = [9, 9, 11, 12, 11, 8, 8]
    moving_average["ma_slow"] = [10, 10, 10, 10, 10, 10, 10]
    moving_average["ma_slow_slope"] = 0.01
    ma_result = run_timing(
        moving_average,
        _zero_cost(timing_style="ma_crossover_atr"),
    )
    assert [trade["side"] for trade in ma_result["trades"]] == [
        "buy",
        "sell",
    ]
    assert ma_result["trades"][0]["reason"] == "ma_crossover"
    assert ma_result["trades"][1]["reason"] == "ma_crossdown"


def test_atr_position_sizing_respects_risk_cap_and_round_lots() -> None:
    sized = calculate_position_size(
        cash=100_000,
        equity=100_000,
        execution_price=10,
        adjusted_price=10,
        atr=0.5,
        lot_size=100,
        mode="atr_risk",
        fixed_fraction=0.5,
        risk_per_trade=0.01,
        max_fraction=0.05,
        atr_stop_multiple=2.0,
    )

    assert sized.risk_cash == pytest.approx(1_000)
    assert sized.stop_distance == pytest.approx(1.0)
    assert sized.target_notional == pytest.approx(5_000)
    assert sized.shares == 500


def test_validation_baselines_share_t_plus_one_open_execution() -> None:
    frame = _frame(
        [0.0] * 6,
        adjusted=[10, 10, 11, 12, 9, 9],
        adjusted_opens=[10, 10, 10, 11, 12, 9],
    )
    frame["ma_200"] = [9, 9, 9, 10, 10, 10]

    buy_hold = run_timing(
        frame,
        _zero_cost(timing_style="buy_and_hold"),
    )
    ma_baseline = run_timing(
        frame,
        _zero_cost(timing_style="ma_200"),
    )

    assert [trade["side"] for trade in buy_hold["trades"]] == ["buy"]
    assert buy_hold["trades"][0]["execution_session"] == "T+1 open"
    assert [trade["side"] for trade in ma_baseline["trades"]] == [
        "buy",
        "sell",
    ]


def test_rsi_bollinger_strategy_trades_without_factor_scores() -> None:
    frame = _regime_frame(
        [0.4] * 7,
        [40, 25, 32, 40, 72, 68, 60],
        [0.3, -0.2, 0.1, 0.4, 1.1, 0.8, 0.5],
        ["sideways"] * 7,
        [0.0] * 7,
        [0.0] * 7,
    )
    result = run_timing(
        frame,
        _zero_cost(
            timing_style="rsi_bollinger",
            minimum_holding_sessions=1,
        ),
    )
    assert [trade["side"] for trade in result["trades"]] == ["buy", "sell"]
    assert result["trades"][0]["reason"] == "rsi_bollinger_entry"
    assert result["trades"][1]["reason"] in {
        "rsi_overbought_reversal",
        "bollinger_upper_reversal",
    }
