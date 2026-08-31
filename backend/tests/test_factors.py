from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.factors.base import (
    Factor,
    FactorMetadata,
    FactorUnavailableError,
    assert_factor_is_causal,
)
from app.factors.benchmark import merge_benchmark_bars
from app.factors.builtin import (
    ATRRatio20Factor,
    AmountSurprise20Factor,
    BUILTIN_FACTORS,
    Amihud20Factor,
    Beta252Factor,
    DownsideVolatility20Factor,
    IdioVolatility60Factor,
    IntradayStrength20Factor,
    MABias20Factor,
    MaxReturn20Factor,
    Momentum20Factor,
    Momentum60Factor,
    Momentum25221Factor,
    OvernightReversal20Factor,
    PBFactor,
    PricePosition252Factor,
    PricePosition60Factor,
    RelativeStrength60Factor,
    ResidualMomentum60Factor,
    Reversal5Factor,
    Skewness60Factor,
    Volatility20Factor,
    VolumeChange20Factor,
    VolumePriceCorr20Factor,
)
from app.factors.evaluation import evaluate_factor
from app.factors.preprocessing import PreprocessConfig
from app.factors.registry import FactorRegistry, load_user_factors


class FutureCloseFactor(Factor):
    metadata = FactorMetadata(
        name="future_close",
        description="Intentionally invalid test factor.",
        lookback=0,
        required_columns=("close",),
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        ordered = bars.sort_values(["symbol", "date"])
        values = ordered.groupby("symbol", sort=False)["close"].shift(-1)
        return values.reindex(bars.index)


class NegativeSignalFactor(Factor):
    metadata = FactorMetadata(
        name="negative_signal",
        description="Lower raw values are preferred.",
        lookback=0,
        required_columns=("signal",),
        direction=-1,
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        return pd.to_numeric(bars["signal"], errors="coerce")


def test_momentum_alignment_has_no_future_lookahead(
    synthetic_bars: pd.DataFrame,
) -> None:
    bars = synthetic_bars[synthetic_bars["symbol"] == "600001"].copy()
    factor = Momentum20Factor()
    original = factor.compute(bars)
    cutoff = bars["date"].iloc[40]

    changed = bars.copy()
    changed.loc[changed["date"] > cutoff, "close"] *= 50.0
    recomputed = factor.compute(changed)

    before_cutoff = bars["date"] <= cutoff
    pd.testing.assert_series_equal(
        original.loc[before_cutoff],
        recomputed.loc[before_cutoff],
        check_names=False,
    )
    expected = bars["close"].iloc[20] / bars["close"].iloc[0] - 1.0
    assert original.iloc[20] == pytest.approx(expected)
    assert original.iloc[:20].isna().all()


def _formula_bars(periods: int = 300) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=periods)
    returns = np.where(np.arange(1, periods) % 3 == 0, -0.02, 0.01)
    close = 100.0 * np.cumprod(np.r_[1.0, 1.0 + returns])
    open_price = np.r_[close[0], close[:-1]] * (
        1.0 + 0.002 * np.sin(np.arange(periods))
    )
    volume = 1_000.0 + np.arange(periods) * 25.0
    benchmark_return = np.r_[
        np.nan, 0.003 + 0.004 * np.cos(np.arange(1, periods) / 5.0)
    ]
    benchmark_close = 200.0 * np.cumprod(
        np.r_[1.0, 1.0 + benchmark_return[1:]]
    )
    return pd.DataFrame(
        {
            "symbol": "600001",
            "date": dates,
            "open": open_price,
            "close": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "volume": volume,
            "amount": close * volume,
            "benchmark_close": benchmark_close,
            "benchmark_return": benchmark_return,
            "pb": 1.5 + np.arange(periods) / 100.0,
        }
    )


def _expected_rolling_residuals(
    stock_return: pd.Series,
    benchmark_return: pd.Series,
    *,
    compound: bool,
) -> pd.Series:
    expected = pd.Series(np.nan, index=stock_return.index, dtype=float)
    for end in range(59, len(stock_return)):
        stock_window = stock_return.iloc[end - 59 : end + 1].to_numpy(dtype=float)
        benchmark_window = benchmark_return.iloc[
            end - 59 : end + 1
        ].to_numpy(dtype=float)
        if not (
            np.isfinite(stock_window).all()
            and np.isfinite(benchmark_window).all()
        ):
            continue
        design = np.column_stack([np.ones(60), benchmark_window])
        coefficients = np.linalg.lstsq(
            design, stock_window, rcond=None
        )[0]
        residuals = stock_window - design @ coefficients
        expected.iloc[end] = (
            np.prod(1.0 + residuals) - 1.0
            if compound
            else residuals.std(ddof=0)
        )
    return expected


def test_all_builtin_factor_formulas_are_deterministic() -> None:
    bars = _formula_bars()
    returns = bars["close"].pct_change(fill_method=None)
    volume_change = bars["volume"].pct_change(fill_method=None)
    previous_close = bars["close"].shift(1)
    true_range = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - previous_close).abs(),
            (bars["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    expected = {
        "momentum_20": bars["close"] / bars["close"].shift(20) - 1.0,
        "momentum_60": bars["close"] / bars["close"].shift(60) - 1.0,
        "momentum_252_21": (
            bars["close"].shift(21) / bars["close"].shift(252) - 1.0
        ),
        "reversal_5": -(bars["close"] / bars["close"].shift(5) - 1.0),
        "volatility_20": returns.rolling(20, min_periods=20).std(ddof=0),
        "volume_change_20": bars["volume"] / bars["volume"].shift(20) - 1.0,
        "ma_bias_20": (
            bars["close"]
            / bars["close"].rolling(20, min_periods=20).mean()
            - 1.0
        ),
        "price_position_60": (
            bars["close"] - bars["low"].rolling(60, min_periods=60).min()
        )
        / (
            bars["high"].rolling(60, min_periods=60).max()
            - bars["low"].rolling(60, min_periods=60).min()
        ),
        "price_position_252": (
            bars["close"] - bars["low"].rolling(252, min_periods=252).min()
        )
        / (
            bars["high"].rolling(252, min_periods=252).max()
            - bars["low"].rolling(252, min_periods=252).min()
        ),
        "max_return_20": returns.rolling(20, min_periods=20).max(),
        "skewness_60": returns.rolling(60, min_periods=60).skew(),
        "atr_ratio_20": (
            true_range.rolling(20, min_periods=20).mean() / bars["close"]
        ),
        "overnight_reversal_20": -(
            bars["open"] / previous_close - 1.0
        ).rolling(20, min_periods=20).mean(),
        "intraday_strength_20": (
            bars["close"] / bars["open"] - 1.0
        ).rolling(20, min_periods=20).mean(),
        "amount_surprise_20": (
            bars["amount"]
            / bars["amount"].rolling(20, min_periods=20).mean()
            - 1.0
        ),
        "volume_price_corr_20": returns.rolling(
            20, min_periods=20
        ).corr(volume_change),
        "beta_252": (
            returns.rolling(252, min_periods=252).cov(
                bars["benchmark_return"]
            )
            / bars["benchmark_return"].rolling(
                252, min_periods=252
            ).var()
        ),
        "idio_volatility_60": _expected_rolling_residuals(
            returns, bars["benchmark_return"], compound=False
        ),
        "relative_strength_60": (
            bars["close"] / bars["close"].shift(60) - 1.0
        )
        - (
            bars["benchmark_close"]
            / bars["benchmark_close"].shift(60)
            - 1.0
        ),
        "residual_momentum_60": _expected_rolling_residuals(
            returns, bars["benchmark_return"], compound=True
        ),
        "downside_volatility_20": (
            returns.where(returns < 0)
            .rolling(20, min_periods=5)
            .std(ddof=0)
        ),
        "amihud_20": (
            returns.abs() / bars["amount"]
        ).rolling(20, min_periods=20).mean(),
        "pb": bars["pb"],
    }
    factors = {
        factor.metadata.name: factor
        for factor in (
            Momentum20Factor(),
            Momentum60Factor(),
            Momentum25221Factor(),
            Reversal5Factor(),
            Volatility20Factor(),
            VolumeChange20Factor(),
            MABias20Factor(),
            PricePosition60Factor(),
            PricePosition252Factor(),
            MaxReturn20Factor(),
            Skewness60Factor(),
            ATRRatio20Factor(),
            OvernightReversal20Factor(),
            IntradayStrength20Factor(),
            AmountSurprise20Factor(),
            VolumePriceCorr20Factor(),
            Beta252Factor(),
            IdioVolatility60Factor(),
            RelativeStrength60Factor(),
            ResidualMomentum60Factor(),
            DownsideVolatility20Factor(),
            Amihud20Factor(),
            PBFactor(),
        )
    }

    for name, expected_values in expected.items():
        pd.testing.assert_series_equal(
            factors[name].compute(bars),
            expected_values.astype(float),
            check_names=False,
        )

    first_downside_value = factors["downside_volatility_20"].compute(
        bars
    ).first_valid_index()
    assert first_downside_value == 15


@pytest.mark.parametrize("invalid_amount", [0.0, -1.0])
def test_amihud_rejects_nonpositive_amount_observations(
    invalid_amount: float,
) -> None:
    bars = _formula_bars()
    bars.loc[30, "amount"] = invalid_amount

    values = Amihud20Factor().compute(bars)

    assert values.iloc[30:50].isna().all()
    assert pd.notna(values.iloc[29])
    assert pd.notna(values.iloc[50])


def test_all_builtin_factors_pass_the_causality_guard() -> None:
    bars = _formula_bars()

    assert {factor.metadata.name for factor in BUILTIN_FACTORS} == {
        "momentum_20",
        "momentum_60",
        "momentum_252_21",
        "reversal_5",
        "volatility_20",
        "volume_change_20",
        "ma_bias_20",
        "price_position_60",
        "price_position_252",
        "max_return_20",
        "skewness_60",
        "atr_ratio_20",
        "overnight_reversal_20",
        "intraday_strength_20",
        "amount_surprise_20",
        "volume_price_corr_20",
        "ma_200",
        "ma_slope_20",
        "distance_to_ma_200",
        "rsi_14",
        "bollinger_mid_20",
        "bollinger_upper_20",
        "bollinger_lower_20",
        "bollinger_percent_b_20",
        "bollinger_bandwidth_20",
        "beta_252",
        "idio_volatility_60",
        "relative_strength_60",
        "residual_momentum_60",
        "downside_volatility_20",
        "amihud_20",
        "pb",
        "bp",
        "ep",
        "dividend_yield",
        "roe",
        "gross_margin",
        "operating_cashflow_to_assets",
        "accruals",
        "asset_growth",
        "market_cap_size",
    }
    for factor in BUILTIN_FACTORS:
        if factor.metadata.availability == "available":
            assert_factor_is_causal(factor, bars)


def test_evaluation_produces_positive_ic_and_json_safe_output(
    synthetic_bars: pd.DataFrame,
) -> None:
    dates = sorted(synthetic_bars["date"].unique())
    result = evaluate_factor(
        synthetic_bars,
        Momentum20Factor(),
        pd.Timestamp(dates[25]).date(),
        pd.Timestamp(dates[-8]).date(),
        forward_period=5,
        quantiles=5,
        preprocess=PreprocessConfig(
            industry_neutralize=True,
            market_cap_neutralize=True,
        ),
    )

    assert result["coverage"]["ratio"] > 0.95
    assert result["ic"]["mean"] > 0.8
    assert result["rank_ic"]["mean"] > 0.8
    assert result["quantile_returns"]
    assert result["quantile_net_values"]
    assert result["long_short"]
    assert result["turnover"]["series"]
    assert len(result["preprocessing"]["warnings"]) == 2
    json.dumps(result, allow_nan=False)


def test_negative_direction_preserves_raw_ic_and_drives_quantiles() -> None:
    dates = pd.bdate_range("2024-01-02", periods=6)
    frames = []
    for signal in range(1, 6):
        daily_return = (6 - signal) / 100.0
        close = 10.0 * np.power(1.0 + daily_return, np.arange(len(dates)))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": f"60000{signal}",
                    "date": dates,
                    "close": close,
                    "signal": float(signal),
                }
            )
        )
    bars = pd.concat(frames, ignore_index=True)

    result = evaluate_factor(
        bars,
        NegativeSignalFactor(),
        dates[0].date(),
        dates[-2].date(),
        forward_period=1,
        quantiles=5,
        preprocess=PreprocessConfig(winsorize=False, zscore=False),
    )

    assert result["direction"] == -1
    assert result["ic"]["mean"] < 0
    assert result["rank_ic"]["mean"] == pytest.approx(-1.0)
    assert result["raw_ic"]["mean"] == pytest.approx(result["ic"]["mean"])
    assert result["raw_rank_ic"]["mean"] == pytest.approx(-1.0)
    assert result["adjusted_ic"]["mean"] == pytest.approx(
        -result["raw_ic"]["mean"]
    )
    assert result["adjusted_rank_ic"]["mean"] == pytest.approx(1.0)
    assert result["summary"]["ic_mean"] == pytest.approx(
        result["summary"]["raw_ic_mean"]
    )
    assert result["summary"]["adjusted_ic_mean"] > 0
    assert result["raw_ic_series"]
    assert result["adjusted_ic_series"]
    assert all(
        row["q5"] > row["q1"] for row in result["quantile_returns"]
    )
    assert all(row["return"] > 0 for row in result["long_short"])


def test_factor_metadata_and_registry_payload_are_backward_compatible() -> None:
    legacy = FactorMetadata("legacy", "Legacy factor.", 0, ("close",))
    assert legacy.display_name == "legacy"
    assert legacy.display_name_zh == "legacy"
    assert legacy.description_zh == "Legacy factor."
    assert legacy.direction == 1
    assert legacy.direction_label == "positive"
    assert legacy.direction_kind == "positive"
    assert legacy.applicable_assets == ("stock", "ETF")
    with pytest.raises(ValueError, match=r"\+1 or -1"):
        FactorMetadata("invalid", "Invalid.", 0, ("close",), direction=0)
    with pytest.raises(ValueError, match="direction_kind"):
        FactorMetadata(
            "invalid_kind",
            "Invalid.",
            0,
            ("close",),
            direction_kind="unknown",
        )

    expected_metadata = {
        "momentum_20": ("20日动量", 1),
        "momentum_60": ("60日动量", 1),
        "momentum_252_21": ("12-1月动量", 1),
        "reversal_5": ("5日反转", 1),
        "volatility_20": ("20日波动率", -1),
        "volume_change_20": ("20日成交量变化率", 1),
        "ma_bias_20": ("20日均线偏离度", 1),
        "price_position_60": ("60日价格位置", 1),
        "price_position_252": ("252日价格位置", 1),
        "max_return_20": ("20日最大收益率", -1),
        "skewness_60": ("60日收益偏度", -1),
        "atr_ratio_20": ("20日真实波幅比率", -1),
        "overnight_reversal_20": ("20日隔夜反转", 1),
        "intraday_strength_20": ("20日日内强度", 1),
        "amount_surprise_20": ("20日成交额惊喜", 1),
        "volume_price_corr_20": ("20日量价相关性", 1),
        "ma_200": ("200日移动平均线", 1),
        "ma_slope_20": ("20日均线斜率", 1),
        "distance_to_ma_200": ("距200日均线", 1),
        "rsi_14": ("14日Wilder相对强弱指标", 1),
        "bollinger_mid_20": ("20日布林带中轨", 1),
        "bollinger_upper_20": ("20日布林带上轨", 1),
        "bollinger_lower_20": ("20日布林带下轨", 1),
        "bollinger_percent_b_20": ("20日布林带%B", 1),
        "bollinger_bandwidth_20": ("20日布林带宽度", -1),
        "beta_252": ("252日市场贝塔", -1),
        "idio_volatility_60": ("60日特质波动率", -1),
        "relative_strength_60": ("60日相对强度", 1),
        "residual_momentum_60": ("60日残差动量", 1),
        "downside_volatility_20": ("20日下行波动率", -1),
        "amihud_20": ("20日非流动性", -1),
        "pb": ("市净率", -1),
        "bp": ("账面市值比BP", 1),
        "ep": ("盈利收益率EP", 1),
        "dividend_yield": ("股息率", 1),
        "roe": ("ROE", 1),
        "gross_margin": ("毛利率", 1),
        "operating_cashflow_to_assets": ("经营现金流/资产", 1),
        "accruals": ("应计利润", -1),
        "asset_growth": ("资产增长率", -1),
        "market_cap_size": ("市值规模", -1),
    }
    payload = {
        item["name"]: item for item in FactorRegistry(BUILTIN_FACTORS).list()
    }
    assert set(payload) == set(expected_metadata)
    for name, (display_name_zh, direction) in expected_metadata.items():
        item = payload[name]
        assert item["display_name"]
        assert item["display_name_zh"] == display_name_zh
        assert item["description_zh"]
        assert item["direction"] == direction
        assert item["direction_label"] == (
            "positive" if direction == 1 else "negative"
        )
        assert item["direction_kind"] in {
            "positive",
            "negative",
            "exploratory",
        }
        assert item["applicable_assets"] == (
            ["stock"]
            if item["availability"] == "requires_point_in_time_fundamentals"
            else ["stock", "ETF"]
        )
        assert item["requirements"] == item["required_columns"]
        assert isinstance(item["lookback"], int)

    assert payload["skewness_60"]["direction_kind"] == "exploratory"
    assert payload["amount_surprise_20"]["direction_kind"] == "exploratory"
    assert payload["volume_price_corr_20"]["direction_kind"] == "exploratory"


def test_benchmark_merge_uses_exact_dates_and_preserves_order() -> None:
    panel = pd.DataFrame(
        {
            "symbol": ["600001", "600001", "600001"],
            "date": pd.to_datetime(["2024-01-04", "2024-01-02", "2024-01-03"]),
            "close": [10.2, 10.0, 10.1],
        },
        index=[9, 3, 7],
    )
    benchmark = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-04"]),
            "close": [100.0, 110.0],
        }
    )

    merged = merge_benchmark_bars(panel, benchmark)

    assert merged.index.tolist() == [9, 3, 7]
    assert merged.loc[9, "benchmark_close"] == pytest.approx(110.0)
    assert merged.loc[9, "benchmark_return"] == pytest.approx(0.1)
    assert merged.loc[3, "benchmark_close"] == pytest.approx(100.0)
    assert pd.isna(merged.loc[3, "benchmark_return"])
    assert pd.isna(merged.loc[7, "benchmark_close"])
    assert pd.isna(merged.loc[7, "benchmark_return"])


def test_benchmark_merge_and_relative_factor_are_causal() -> None:
    bars = _formula_bars(100).drop(
        columns=["benchmark_close", "benchmark_return"]
    )
    benchmark = pd.DataFrame(
        {
            "date": bars["date"],
            "close": 200.0
            * np.cumprod(1.0 + 0.001 * np.arange(1, len(bars) + 1)),
        }
    )
    cutoff = bars["date"].iloc[75]
    original_panel = merge_benchmark_bars(bars, benchmark)
    original = RelativeStrength60Factor().compute(original_panel)

    changed_benchmark = benchmark.copy()
    changed_benchmark.loc[
        changed_benchmark["date"] > cutoff, "close"
    ] *= 100.0
    changed_panel = merge_benchmark_bars(bars, changed_benchmark)
    recomputed = RelativeStrength60Factor().compute(changed_panel)

    before_cutoff = bars["date"] <= cutoff
    pd.testing.assert_series_equal(
        original.loc[before_cutoff],
        recomputed.loc[before_cutoff],
        check_names=False,
    )


def test_benchmark_factor_missing_inputs_have_clear_error() -> None:
    bars = _formula_bars().drop(
        columns=["benchmark_close", "benchmark_return"]
    )
    with pytest.raises(
        FactorUnavailableError,
        match="beta_252.*missing benchmark_close, benchmark_return",
    ):
        Beta252Factor().compute(bars)

    with pytest.raises(
        FactorUnavailableError,
        match="Benchmark bars are unavailable: missing close",
    ):
        merge_benchmark_bars(
            bars,
            pd.DataFrame({"date": bars["date"]}),
        )

    unusable = _formula_bars()
    unusable["benchmark_return"] = np.nan
    with pytest.raises(
        FactorUnavailableError,
        match="benchmark_return contains no usable benchmark values",
    ):
        Beta252Factor().compute(unusable)


def test_pb_is_explicitly_unavailable_without_point_in_time_field(
    synthetic_bars: pd.DataFrame,
) -> None:
    with pytest.raises(FactorUnavailableError, match="missing pb"):
        PBFactor().compute(synthetic_bars)

    with_pb = synthetic_bars.copy()
    with_pb["pb"] = np.nan
    with pytest.raises(FactorUnavailableError, match="not fabricated"):
        PBFactor().compute(with_pb)


def test_fundamental_placeholders_require_point_in_time_provenance(
    synthetic_bars: pd.DataFrame,
) -> None:
    bp_factor = next(
        factor for factor in BUILTIN_FACTORS if factor.metadata.name == "bp"
    )
    with pytest.raises(FactorUnavailableError, match="missing bp"):
        bp_factor.compute(synthetic_bars)

    bars = synthetic_bars.copy()
    bars["bp"] = 0.8
    bars["fundamentals_are_point_in_time"] = False
    with pytest.raises(FactorUnavailableError, match="does not certify"):
        bp_factor.compute(bars)

    bars["fundamentals_are_point_in_time"] = True
    values = bp_factor.compute(bars)
    assert values.eq(0.8).all()


def test_factor_causality_guard_rejects_future_shift(
    synthetic_bars: pd.DataFrame,
) -> None:
    dates = sorted(synthetic_bars["date"].unique())
    with pytest.raises(ValueError, match="causality check"):
        evaluate_factor(
            synthetic_bars,
            FutureCloseFactor(),
            pd.Timestamp(dates[5]).date(),
            pd.Timestamp(dates[-5]).date(),
        )


def test_user_factor_python_file_can_be_loaded(tmp_path) -> None:
    (tmp_path / "custom.py").write_text(
        "\n".join(
            [
                "import pandas as pd",
                "from app.factors.base import Factor, FactorMetadata",
                "class CustomFactor(Factor):",
                "    metadata = FactorMetadata('custom', 'test', 0, ('close',))",
                "    def compute(self, bars: pd.DataFrame) -> pd.Series:",
                "        return bars['close']",
                "FACTOR = CustomFactor()",
            ]
        ),
        encoding="utf-8",
    )
    registry = FactorRegistry()
    load_user_factors(registry, tmp_path)
    assert registry.get("custom").metadata.name == "custom"
    assert registry.warnings == []
