from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.factors.base import Factor, FactorMetadata
from app.factors.registry import FactorRegistry
from app.multifactor import (
    CompositeFactor,
    FactorComponentConfig,
    MultiFactorConfig,
    factor_correlation_report,
)
from app.timing import TimingConfig, run_timing
from app.config import Settings
from app.storage import Storage


class ColumnFactor(Factor):
    def __init__(self, name: str, column: str, direction: int = 1) -> None:
        self.column = column
        self.metadata = FactorMetadata(
            name=name,
            description=name,
            lookback=0,
            required_columns=(column,),
            direction=direction,
        )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(bars[self.column], errors="coerce")


def _frame() -> pd.DataFrame:
    dates = pd.to_datetime(
        ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
    )
    return pd.DataFrame(
        {
            "symbol": ["A", "B", "A", "B"],
            "date": dates,
            "close": [10.0, 20.0, 11.0, 19.0],
            "f1": [1.0, 3.0, 2.0, 4.0],
            "f2": [2.0, 4.0, np.nan, 6.0],
        }
    )


def test_cross_sectional_composite_respects_direction_and_weights() -> None:
    registry = FactorRegistry(
        [ColumnFactor("positive", "f1"), ColumnFactor("negative", "f2", -1)]
    )
    config = MultiFactorConfig(
        name="test",
        components=(
            FactorComponentConfig("positive", weight=1.0),
            FactorComponentConfig("negative", weight=1.0),
        ),
    )
    details = CompositeFactor(config, registry).compute_details(_frame())
    assert details.loc[0, "normalized_positive"] == pytest.approx(-1.0)
    assert details.loc[0, "normalized_negative"] == pytest.approx(-1.0)
    assert details.loc[0, "contribution_negative"] == pytest.approx(1.0)
    assert details.loc[0, "composite_score"] == pytest.approx(0.0)
    assert details.loc[1, "composite_score"] == pytest.approx(0.0)


def test_missing_values_renormalize_available_weights() -> None:
    registry = FactorRegistry(
        [ColumnFactor("one", "f1"), ColumnFactor("two", "f2")]
    )
    config = MultiFactorConfig(
        name="missing",
        components=(
            FactorComponentConfig("one", weight=1.0),
            FactorComponentConfig("two", weight=3.0),
        ),
    )
    details = CompositeFactor(config, registry).compute_details(_frame())
    assert details.loc[2, "valid_weight"] == pytest.approx(1.0)
    assert details.loc[2, "composite_score"] == pytest.approx(-1.0)


def test_drop_and_explicit_zero_missing_policies_are_distinct() -> None:
    registry = FactorRegistry(
        [ColumnFactor("one", "f1"), ColumnFactor("two", "f2")]
    )
    dropped = CompositeFactor(
        MultiFactorConfig(
            name="drop",
            components=(
                FactorComponentConfig("one", weight=1.0),
                FactorComponentConfig(
                    "two",
                    weight=3.0,
                    missing_policy="drop",
                ),
            ),
        ),
        registry,
    ).compute_details(_frame())
    zeroed = CompositeFactor(
        MultiFactorConfig(
            name="zero missing",
            components=(
                FactorComponentConfig("one", weight=1.0),
                FactorComponentConfig(
                    "two",
                    weight=3.0,
                    missing_policy="zero",
                ),
            ),
        ),
        registry,
    ).compute_details(_frame())

    assert pd.isna(dropped.loc[2, "composite_score"])
    assert zeroed.loc[2, "valid_weight"] == pytest.approx(4.0)
    assert zeroed.loc[2, "composite_score"] == pytest.approx(-0.25)


def test_negative_component_weight_reverses_its_contribution() -> None:
    registry = FactorRegistry([ColumnFactor("one", "f1")])
    details = CompositeFactor(
        MultiFactorConfig(
            name="negative weight",
            components=(FactorComponentConfig("one", weight=-2.0),),
        ),
        registry,
    ).compute_details(_frame())

    assert details.loc[0, "normalized_one"] == pytest.approx(-1.0)
    assert details.loc[0, "contribution_one"] == pytest.approx(2.0)
    assert details.loc[0, "composite_score"] == pytest.approx(1.0)


def test_zero_weight_does_not_participate_and_snapshot_is_stable() -> None:
    registry = FactorRegistry(
        [ColumnFactor("one", "f1"), ColumnFactor("two", "f2")]
    )
    config = MultiFactorConfig(
        name="zero",
        components=(
            FactorComponentConfig("one", weight=1.0),
            FactorComponentConfig("two", weight=0.0),
        ),
    )
    factor = CompositeFactor(config, registry)
    details = factor.compute_details(_frame())
    assert "factor_two" not in details.columns
    assert config.config_id == MultiFactorConfig(
        name="zero",
        components=(
            FactorComponentConfig("one", weight=1.0),
            FactorComponentConfig("two", weight=0.0),
        ),
    ).config_id


def test_rolling_standardization_uses_only_own_history() -> None:
    periods = 140
    bars = pd.DataFrame(
        {
            "symbol": "A",
            "date": pd.bdate_range("2023-01-02", periods=periods),
            "f1": np.arange(periods, dtype=float),
        }
    )
    registry = FactorRegistry([ColumnFactor("one", "f1")])
    config = MultiFactorConfig(
        name="rolling",
        mode="time_series",
        rolling_window=120,
        rolling_min_periods=60,
        components=(FactorComponentConfig("one", weight=1.0),),
    )
    factor = CompositeFactor(config, registry)
    original = factor.compute(bars)
    changed = bars.copy()
    changed.loc[changed.index[-1], "f1"] = 1_000_000
    recomputed = factor.compute(changed)
    pd.testing.assert_series_equal(
        original.iloc[:-1],
        recomputed.iloc[:-1],
        check_names=False,
    )


def test_removing_future_rows_does_not_change_historical_composite_scores() -> None:
    periods = 180
    bars = pd.DataFrame(
        {
            "symbol": "A",
            "date": pd.bdate_range("2023-01-02", periods=periods),
            "f1": np.sin(np.arange(periods) / 9.0),
            "f2": np.cos(np.arange(periods) / 13.0),
        }
    )
    registry = FactorRegistry(
        [ColumnFactor("one", "f1"), ColumnFactor("two", "f2", -1)]
    )
    config = MultiFactorConfig(
        name="causal",
        mode="time_series",
        rolling_window=60,
        rolling_min_periods=30,
        components=(
            FactorComponentConfig("one", weight=1.0),
            FactorComponentConfig("two", weight=0.5),
        ),
    )
    factor = CompositeFactor(config, registry)
    cutoff = 135
    full = factor.compute(bars)
    truncated = factor.compute(bars.iloc[:cutoff].copy())
    pd.testing.assert_series_equal(
        full.iloc[:cutoff],
        truncated,
        check_names=False,
    )


def test_different_time_series_factor_combinations_produce_different_signals() -> None:
    periods = 140
    dates = pd.bdate_range("2023-01-02", periods=periods)
    bars = pd.DataFrame(
        {
            "symbol": "A",
            "date": dates,
            "f1": np.sin(np.arange(periods) / 5.0),
            "f2": np.sin((np.arange(periods) - 12) / 5.0),
        }
    )
    registry = FactorRegistry(
        [ColumnFactor("one", "f1"), ColumnFactor("two", "f2")]
    )

    def run_for(name: str) -> dict:
        config = MultiFactorConfig(
            name=name,
            mode="time_series",
            rolling_window=20,
            rolling_min_periods=10,
            components=(FactorComponentConfig(name, weight=1.0),),
        )
        details = CompositeFactor(config, registry).compute_details(bars)
        signal = details.copy()
        for column in (
            "open",
            "high",
            "low",
            "close",
            "trade_open",
            "trade_high",
            "trade_low",
            "trade_close",
            "trade_reference_close",
        ):
            signal[column] = 10.0
        signal["trade_volume"] = 100_000.0
        signal["trend_score"] = signal["composite_score"]
        return run_timing(
            signal,
            TimingConfig(
                commission_rate=0,
                minimum_commission=0,
                slippage=0,
                position_sizing="full",
                minimum_holding_sessions=1,
                cooldown_sessions=0,
            ),
        )

    first = run_for("one")
    second = run_for("two")
    assert first["signals"]
    assert second["signals"]
    assert first["signals"][0]["signal_date"] != second["signals"][0]["signal_date"]


def test_correlation_report_flags_duplicates() -> None:
    details = pd.DataFrame(
        {
            "normalized_a": [1.0, 2.0, 3.0, 4.0],
            "normalized_b": [2.0, 4.0, 6.0, 8.0],
            "normalized_c": [1.0, -1.0, 1.0, -1.0],
            "contribution_a": [0.1, 0.2, 0.3, 0.4],
            "contribution_b": [0.2, 0.4, 0.6, 0.8],
            "contribution_c": [0.1, -0.1, 0.1, -0.1],
            "composite_score": [1.0, 2.0, 3.0, 4.0],
        }
    )
    report = factor_correlation_report(details)
    assert report["pearson"]["a"]["b"] == pytest.approx(1.0)
    assert any(
        {item["left"], item["right"]} == {"a", "b"}
        for item in report["high_correlation_pairs"]
    )


def test_multifactor_config_snapshot_persistence(tmp_path) -> None:
    storage = Storage(
        Settings(
            data_dir=tmp_path / "data",
            db_path=tmp_path / "quant.sqlite3",
        )
    )
    config = MultiFactorConfig(
        name="saved",
        components=(FactorComponentConfig("one", weight=1.0),),
    )
    saved = storage.save_multifactor_config(
        config.config_id,
        config.name,
        config.snapshot(),
    )
    assert saved["id"] == config.config_id
    loaded = storage.get_multifactor_config(config.config_id)
    assert loaded is not None
    assert loaded["config"] == config.snapshot()
    assert storage.list_multifactor_configs()[0]["name"] == "saved"

    restored_payload = loaded["config"]
    restored = MultiFactorConfig(
        name=restored_payload["name"],
        mode=restored_payload["mode"],
        rolling_window=restored_payload["rolling_window"],
        rolling_min_periods=restored_payload["rolling_min_periods"],
        zscore_clip=restored_payload["zscore_clip"],
        metadata=restored_payload["metadata"],
        components=tuple(
            FactorComponentConfig(**item)
            for item in restored_payload["components"]
        ),
    )
    registry = FactorRegistry([ColumnFactor("one", "f1")])
    pd.testing.assert_series_equal(
        CompositeFactor(config, registry).compute(_frame()),
        CompositeFactor(restored, registry).compute(_frame()),
    )
