from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.factors.base import FactorUnavailableError, build_factor_observations
from app.factors.builtin import BUILTIN_FACTORS, MA200Factor
from app.timing.indicators import (
    average_true_range,
    bollinger_bands,
    distance_to_moving_average,
    donchian_channels,
    moving_average,
    moving_average_slope,
    wilder_rsi,
)


def _grouped_bars() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=8)
    bars = pd.concat(
        [
            pd.DataFrame(
                {"symbol": "AAA", "date": dates, "close": np.arange(1.0, 9.0)}
            ),
            pd.DataFrame(
                {"symbol": "BBB", "date": dates, "close": np.arange(11.0, 19.0)}
            ),
        ],
        ignore_index=True,
    )
    return bars.sample(frac=1.0, random_state=7)


@pytest.mark.parametrize("symbol,offset", [("AAA", 0.0), ("BBB", 10.0)])
def test_grouped_indicator_formulas_are_deterministic(
    symbol: str, offset: float
) -> None:
    bars = _grouped_bars()
    selected = bars["symbol"].eq(symbol)
    ordered_index = bars.loc[selected].sort_values("date").index
    close = pd.Series(np.arange(1.0, 9.0) + offset, index=ordered_index)
    expected_mid = close.rolling(3, min_periods=3).mean()
    expected_deviation = close.rolling(3, min_periods=3).std(ddof=0)
    expected_upper = expected_mid + 2.0 * expected_deviation
    expected_lower = expected_mid - 2.0 * expected_deviation
    expected = {
        "ma": expected_mid,
        "slope": expected_mid / expected_mid.shift(1) - 1.0,
        "distance": close / expected_mid - 1.0,
        "mid": expected_mid,
        "upper": expected_upper,
        "lower": expected_lower,
        "percent_b": (close - expected_lower) / (expected_upper - expected_lower),
        "bandwidth": (expected_upper - expected_lower) / expected_mid,
    }
    actual_bands = bollinger_bands(bars, 3)
    actual = {
        "ma": moving_average(bars, 3),
        "slope": moving_average_slope(bars, 3),
        "distance": distance_to_moving_average(bars, 3),
        **{column: actual_bands[column] for column in actual_bands},
    }

    for name, expected_values in expected.items():
        pd.testing.assert_series_equal(
            actual[name].loc[ordered_index],
            expected_values,
            check_names=False,
        )


def test_wilder_rsi_uses_sma_seed_then_recursive_smoothing() -> None:
    bars = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2026-01-05", periods=6),
            "close": [1.0, 2.0, 3.0, 2.0, 2.0, 4.0],
        }
    )

    values = wilder_rsi(bars, 3)

    assert values.iloc[:3].isna().all()
    assert values.iloc[3] == pytest.approx(2.0 / 3.0 * 100.0)
    assert values.iloc[4] == pytest.approx(2.0 / 3.0 * 100.0)
    assert values.iloc[5] == pytest.approx(86.66666666666667)


def test_flat_and_zero_width_inputs_have_explicit_nan_behavior() -> None:
    bars = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2026-01-05", periods=5),
            "close": 10.0,
        }
    )

    rsi = wilder_rsi(bars, 3)
    bands = bollinger_bands(bars, 3)

    assert rsi.iloc[:3].isna().all()
    assert rsi.iloc[3:].eq(50.0).all()
    assert bands["mid"].iloc[:2].isna().all()
    assert bands["mid"].iloc[2:].eq(10.0).all()
    assert bands["percent_b"].isna().all()
    assert bands["bandwidth"].iloc[2:].eq(0.0).all()


def test_indicators_are_unchanged_when_future_rows_are_truncated() -> None:
    bars = _grouped_bars()
    cutoff = pd.Timestamp("2026-01-12")
    truncated = bars[bars["date"] <= cutoff]
    calculations = (
        lambda frame: moving_average(frame, 3),
        lambda frame: moving_average_slope(frame, 3),
        lambda frame: distance_to_moving_average(frame, 3),
        lambda frame: wilder_rsi(frame, 3),
    )
    for calculate in calculations:
        pd.testing.assert_series_equal(
            calculate(bars).reindex(truncated.index),
            calculate(truncated),
            check_names=False,
        )
    pd.testing.assert_frame_equal(
        bollinger_bands(bars, 3).reindex(truncated.index),
        bollinger_bands(truncated, 3),
    )


def test_donchian_excludes_current_bar_and_atr_is_causal() -> None:
    bars = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2026-01-05", periods=7),
            "open": [1, 2, 3, 4, 5, 6, 7],
            "high": [2, 3, 4, 5, 6, 7, 8],
            "low": [0, 1, 2, 3, 4, 5, 6],
            "close": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    channels = donchian_channels(bars, entry_window=3, exit_window=2)

    assert channels["upper"].iloc[3] == pytest.approx(4.0)
    assert channels["lower"].iloc[2] == pytest.approx(0.0)
    atr = average_true_range(bars, 3)
    assert atr.iloc[:2].isna().all()
    assert atr.iloc[2:].eq(2.0).all()

    prefix = bars.iloc[:5].copy()
    pd.testing.assert_frame_equal(
        channels.iloc[:5],
        donchian_channels(prefix, entry_window=3, exit_window=2),
    )
    pd.testing.assert_series_equal(
        atr.iloc[:5],
        average_true_range(prefix, 3),
        check_names=False,
    )


def test_default_indicator_factors_are_registered_with_expected_metadata() -> None:
    expected = {
        "ma_200": (200, 1),
        "ma_slope_20": (20, 1),
        "distance_to_ma_200": (200, 1),
        "rsi_14": (14, 1),
        "bollinger_mid_20": (20, 1),
        "bollinger_upper_20": (20, 1),
        "bollinger_lower_20": (20, 1),
        "bollinger_percent_b_20": (20, 1),
        "bollinger_bandwidth_20": (20, -1),
    }
    registered = {factor.metadata.name: factor.metadata for factor in BUILTIN_FACTORS}

    for name, (lookback, direction) in expected.items():
        metadata = registered[name]
        assert metadata.lookback == lookback
        assert metadata.direction == direction
        assert metadata.display_name_zh
        assert metadata.description_zh
        assert metadata.applicable_assets == ("stock", "ETF")


def test_insufficient_history_returns_nan_and_observation_error_is_clear() -> None:
    bars = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2026-01-05", periods=30),
            "close": np.arange(30.0),
        }
    )
    factor = MA200Factor()

    assert factor.compute(bars).isna().all()
    with pytest.raises(
        FactorUnavailableError,
        match=r"ma_200.*requires at least 200 prior trading sessions",
    ):
        build_factor_observations(factor, bars)
