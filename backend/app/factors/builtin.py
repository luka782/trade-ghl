from __future__ import annotations

import numpy as np
import pandas as pd

from ..timing.indicators import (
    bollinger_bands,
    distance_to_moving_average,
    moving_average,
    moving_average_slope,
    wilder_rsi,
)
from .base import Factor, FactorMetadata, FactorUnavailableError


# 内置因子均遵守同一约定：先按 (symbol, date) 排序、在单证券内滚动计算，
# 再恢复调用方原始索引。严禁使用未来行；lookback 前的 NaN 是正常的预热期。
def _restore_order(
    bars: pd.DataFrame, ordered: pd.DataFrame, values: pd.Series
) -> pd.Series:
    result = pd.Series(np.nan, index=bars.index, dtype=float)
    result.loc[ordered.index] = pd.to_numeric(values, errors="coerce").to_numpy()
    return result.replace([np.inf, -np.inf], np.nan)


class _BenchmarkFactor(Factor):
    """需要基准行情的因子共有校验：基准列存在不代表其中有可用数值。"""
    def validate(self, bars: pd.DataFrame) -> None:
        super().validate(bars)
        for column in ("benchmark_close", "benchmark_return"):
            usable = pd.to_numeric(bars[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            if not usable.notna().any():
                raise FactorUnavailableError(
                    f"Factor '{self.metadata.name}' is unavailable: "
                    f"{column} contains no usable benchmark values"
                )


class Momentum20Factor(Factor):
    metadata = FactorMetadata(
        name="momentum_20",
        description="Twenty-session close-to-close momentum available at signal close T.",
        lookback=20,
        required_columns=("close",),
        display_name="20-Day Momentum",
        display_name_zh="20日动量",
        description_zh="当前收盘价相对20个交易日前收盘价的收益率。",
        direction=1,
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        ordered = bars.sort_values(["symbol", "date"])
        values = ordered.groupby("symbol", sort=False)["close"].transform(
            lambda close: close / close.shift(20) - 1.0
        )
        return _restore_order(bars, ordered, values)


class Reversal5Factor(Factor):
    metadata = FactorMetadata(
        name="reversal_5",
        description="Negative five-session close-to-close return known at signal close T.",
        lookback=5,
        required_columns=("close",),
        display_name="5-Day Reversal",
        display_name_zh="5日反转",
        description_zh="过去5个交易日收益率的相反数，值越大表示短期反转信号越强。",
        direction=1,
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        ordered = bars.sort_values(["symbol", "date"])
        values = ordered.groupby("symbol", sort=False)["close"].transform(
            lambda close: -(close / close.shift(5) - 1.0)
        )
        return _restore_order(bars, ordered, values)


class Volatility20Factor(Factor):
    metadata = FactorMetadata(
        name="volatility_20",
        description="Population standard deviation of the latest twenty daily returns.",
        lookback=20,
        required_columns=("close",),
        display_name="20-Day Volatility",
        display_name_zh="20日波动率",
        description_zh="最近20个交易日日收益率的总体标准差。",
        direction=-1,
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        ordered = bars.sort_values(["symbol", "date"])
        values = ordered.groupby("symbol", sort=False)["close"].transform(
            lambda close: close.pct_change(fill_method=None)
            .rolling(20, min_periods=20)
            .std(ddof=0)
        )
        return _restore_order(bars, ordered, values)


class VolumeChange20Factor(Factor):
    metadata = FactorMetadata(
        name="volume_change_20",
        description="Current volume relative to volume twenty sessions earlier.",
        lookback=20,
        required_columns=("volume",),
        display_name="20-Day Volume Change",
        display_name_zh="20日成交量变化率",
        description_zh="当前成交量相对20个交易日前成交量的变化率。",
        direction=1,
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        ordered = bars.sort_values(["symbol", "date"])
        values = ordered.groupby("symbol", sort=False)["volume"].transform(
            lambda volume: volume / volume.shift(20) - 1.0
        )
        return _restore_order(bars, ordered, values)


class MABias20Factor(Factor):
    metadata = FactorMetadata(
        name="ma_bias_20",
        description="Close relative to its trailing twenty-session moving average.",
        lookback=20,
        required_columns=("close",),
        display_name="20-Day Moving-Average Bias",
        display_name_zh="20日均线偏离度",
        description_zh="当前收盘价相对最近20个交易日收盘价均值的偏离比例。",
        direction=1,
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        ordered = bars.sort_values(["symbol", "date"])
        moving_average = ordered.groupby("symbol", sort=False)["close"].transform(
            lambda close: close.rolling(20, min_periods=20).mean()
        )
        values = ordered["close"] / moving_average - 1.0
        return _restore_order(bars, ordered, values)


class PricePosition60Factor(Factor):
    metadata = FactorMetadata(
        name="price_position_60",
        description="Close position within the trailing sixty-session high-low range.",
        lookback=60,
        required_columns=("high", "low", "close"),
        display_name="60-Day Price Position",
        display_name_zh="60日价格位置",
        description_zh="当前收盘价在最近60个交易日最高价与最低价区间中的相对位置。",
        direction=1,
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        ordered = bars.sort_values(["symbol", "date"])
        rolling_high = ordered.groupby("symbol", sort=False)["high"].transform(
            lambda high: high.rolling(60, min_periods=60).max()
        )
        rolling_low = ordered.groupby("symbol", sort=False)["low"].transform(
            lambda low: low.rolling(60, min_periods=60).min()
        )
        values = (ordered["close"] - rolling_low) / (rolling_high - rolling_low)
        return _restore_order(bars, ordered, values)


class DownsideVolatility20Factor(Factor):
    metadata = FactorMetadata(
        name="downside_volatility_20",
        description=(
            "Population standard deviation of negative returns in the latest "
            "twenty sessions, requiring at least five downside observations."
        ),
        lookback=20,
        required_columns=("close",),
        display_name="20-Day Downside Volatility",
        display_name_zh="20日下行波动率",
        description_zh=(
            "最近20个交易日内负收益率的总体标准差，至少需要5个下跌收益观测。"
        ),
        direction=-1,
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        ordered = bars.sort_values(["symbol", "date"])
        values = ordered.groupby("symbol", sort=False)["close"].transform(
            lambda close: close.pct_change(fill_method=None)
            .where(lambda returns: returns < 0)
            .rolling(20, min_periods=5)
            .std(ddof=0)
        )
        return _restore_order(bars, ordered, values)


class Amihud20Factor(Factor):
    metadata = FactorMetadata(
        name="amihud_20",
        description=(
            "Twenty-session mean absolute daily return divided by trading amount; "
            "nonpositive amount is invalid."
        ),
        lookback=20,
        required_columns=("close", "amount"),
        display_name="20-Day Amihud Illiquidity",
        display_name_zh="20日非流动性",
        description_zh=(
            "最近20个交易日绝对收益率与成交额之比的均值，非正成交额视为无效。"
        ),
        direction=-1,
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        ordered = bars.sort_values(["symbol", "date"])
        returns = ordered.groupby("symbol", sort=False)["close"].pct_change(
            fill_method=None
        )
        amount = pd.to_numeric(ordered["amount"], errors="coerce").where(
            lambda values: values > 0
        )
        daily_illiquidity = returns.abs() / amount
        values = daily_illiquidity.groupby(
            ordered["symbol"], sort=False
        ).transform(lambda ratio: ratio.rolling(20, min_periods=20).mean())
        return _restore_order(bars, ordered, values)


class Momentum60Factor(Factor):
    metadata = FactorMetadata(
        name="momentum_60",
        description="Sixty-session close-to-close momentum available at close T.",
        lookback=60,
        required_columns=("close",),
        display_name="60-Day Momentum",
        display_name_zh="60日动量",
        description_zh="当前收盘价相对60个交易日前收盘价的收益率。",
        direction=1,
        direction_kind="positive",
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        ordered = bars.sort_values(["symbol", "date"])
        values = ordered.groupby("symbol", sort=False)["close"].transform(
            lambda close: close / close.shift(60) - 1.0
        )
        return _restore_order(bars, ordered, values)


class Momentum25221Factor(Factor):
    metadata = FactorMetadata(
        name="momentum_252_21",
        description="Return from 252 sessions ago through 21 sessions ago.",
        lookback=252,
        required_columns=("close",),
        display_name="12-1 Month Momentum",
        display_name_zh="12-1月动量",
        description_zh="21个交易日前收盘价相对252个交易日前收盘价的收益率。",
        direction=1,
        direction_kind="positive",
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        ordered = bars.sort_values(["symbol", "date"])
        values = ordered.groupby("symbol", sort=False)["close"].transform(
            lambda close: close.shift(21) / close.shift(252) - 1.0
        )
        return _restore_order(bars, ordered, values)


class PricePosition252Factor(Factor):
    metadata = FactorMetadata(
        name="price_position_252",
        description="Close position within the trailing 252-session high-low range.",
        lookback=252,
        required_columns=("high", "low", "close"),
        display_name="252-Day Price Position",
        display_name_zh="252日价格位置",
        description_zh="当前收盘价在最近252个交易日最高价与最低价区间中的相对位置。",
        direction=1,
        direction_kind="positive",
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        ordered = bars.sort_values(["symbol", "date"])
        rolling_high = ordered.groupby("symbol", sort=False)["high"].transform(
            lambda high: high.rolling(252, min_periods=252).max()
        )
        rolling_low = ordered.groupby("symbol", sort=False)["low"].transform(
            lambda low: low.rolling(252, min_periods=252).min()
        )
        values = (ordered["close"] - rolling_low) / (rolling_high - rolling_low)
        return _restore_order(bars, ordered, values)


class MaxReturn20Factor(Factor):
    metadata = FactorMetadata(
        name="max_return_20",
        description="Maximum daily return observed in the latest twenty sessions.",
        lookback=20,
        required_columns=("close",),
        display_name="20-Day Maximum Return",
        display_name_zh="20日最大收益率",
        description_zh="最近20个交易日内日收益率的最大值。",
        direction=-1,
        direction_kind="negative",
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        ordered = bars.sort_values(["symbol", "date"])
        values = ordered.groupby("symbol", sort=False)["close"].transform(
            lambda close: close.pct_change(fill_method=None)
            .rolling(20, min_periods=20)
            .max()
        )
        return _restore_order(bars, ordered, values)


class Skewness60Factor(Factor):
    metadata = FactorMetadata(
        name="skewness_60",
        description="Sample skewness of daily returns over the latest sixty sessions.",
        lookback=60,
        required_columns=("close",),
        display_name="60-Day Return Skewness",
        display_name_zh="60日收益偏度",
        description_zh="最近60个交易日日收益率的样本偏度，作为探索性信号。",
        direction=-1,
        direction_kind="exploratory",
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        ordered = bars.sort_values(["symbol", "date"])
        values = ordered.groupby("symbol", sort=False)["close"].transform(
            lambda close: close.pct_change(fill_method=None)
            .rolling(60, min_periods=60)
            .skew()
        )
        return _restore_order(bars, ordered, values)


class ATRRatio20Factor(Factor):
    metadata = FactorMetadata(
        name="atr_ratio_20",
        description="Twenty-session average true range divided by current close.",
        lookback=20,
        required_columns=("high", "low", "close"),
        display_name="20-Day ATR Ratio",
        display_name_zh="20日真实波幅比率",
        description_zh="最近20个交易日平均真实波幅除以当前收盘价。",
        direction=-1,
        direction_kind="negative",
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        ordered = bars.sort_values(["symbol", "date"])
        previous_close = ordered.groupby("symbol", sort=False)["close"].shift(1)
        true_range = pd.concat(
            [
                ordered["high"] - ordered["low"],
                (ordered["high"] - previous_close).abs(),
                (ordered["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.groupby(ordered["symbol"], sort=False).transform(
            lambda values: values.rolling(20, min_periods=20).mean()
        )
        values = atr / ordered["close"]
        return _restore_order(bars, ordered, values)


class OvernightReversal20Factor(Factor):
    metadata = FactorMetadata(
        name="overnight_reversal_20",
        description="Negative mean overnight return over the latest twenty sessions.",
        lookback=20,
        required_columns=("open", "close"),
        display_name="20-Day Overnight Reversal",
        display_name_zh="20日隔夜反转",
        description_zh="最近20个交易日开盘价相对前收盘价收益率均值的相反数。",
        direction=1,
        direction_kind="positive",
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        ordered = bars.sort_values(["symbol", "date"])
        previous_close = ordered.groupby("symbol", sort=False)["close"].shift(1)
        overnight_return = ordered["open"] / previous_close - 1.0
        values = overnight_return.groupby(
            ordered["symbol"], sort=False
        ).transform(lambda returns: -returns.rolling(20, min_periods=20).mean())
        return _restore_order(bars, ordered, values)


class IntradayStrength20Factor(Factor):
    metadata = FactorMetadata(
        name="intraday_strength_20",
        description="Mean open-to-close return over the latest twenty sessions.",
        lookback=20,
        required_columns=("open", "close"),
        display_name="20-Day Intraday Strength",
        display_name_zh="20日日内强度",
        description_zh="最近20个交易日收盘价相对开盘价收益率的均值。",
        direction=1,
        direction_kind="positive",
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        ordered = bars.sort_values(["symbol", "date"])
        intraday_return = ordered["close"] / ordered["open"] - 1.0
        values = intraday_return.groupby(
            ordered["symbol"], sort=False
        ).transform(lambda returns: returns.rolling(20, min_periods=20).mean())
        return _restore_order(bars, ordered, values)


class AmountSurprise20Factor(Factor):
    metadata = FactorMetadata(
        name="amount_surprise_20",
        description="Current amount relative to its trailing twenty-session mean.",
        lookback=20,
        required_columns=("amount",),
        display_name="20-Day Amount Surprise",
        display_name_zh="20日成交额惊喜",
        description_zh="当前成交额相对最近20个交易日成交额均值的偏离比例，作为探索性信号。",
        direction=1,
        direction_kind="exploratory",
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        ordered = bars.sort_values(["symbol", "date"])
        rolling_amount = ordered.groupby("symbol", sort=False)["amount"].transform(
            lambda amount: amount.rolling(20, min_periods=20).mean()
        )
        values = ordered["amount"] / rolling_amount - 1.0
        return _restore_order(bars, ordered, values)


class VolumePriceCorr20Factor(Factor):
    metadata = FactorMetadata(
        name="volume_price_corr_20",
        description=(
            "Rolling correlation between daily return and volume growth over "
            "twenty sessions."
        ),
        lookback=20,
        required_columns=("close", "volume"),
        display_name="20-Day Volume-Price Correlation",
        display_name_zh="20日量价相关性",
        description_zh="最近20个交易日日收益率与成交量变化率的滚动相关系数，作为探索性信号。",
        direction=1,
        direction_kind="exploratory",
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        ordered = bars.sort_values(["symbol", "date"])
        values = pd.Series(np.nan, index=ordered.index, dtype=float)
        for _, group in ordered.groupby("symbol", sort=False):
            returns = group["close"].pct_change(fill_method=None)
            volume_change = group["volume"].pct_change(fill_method=None)
            values.loc[group.index] = returns.rolling(
                20, min_periods=20
            ).corr(volume_change)
        return _restore_order(bars, ordered, values)


class MA200Factor(Factor):
    metadata = FactorMetadata(
        name="ma_200",
        description="Trailing 200-session simple moving average available at close T.",
        lookback=200,
        required_columns=("close",),
        display_name="200-Day Moving Average",
        display_name_zh="200日移动平均线",
        description_zh="截至当前交易日的最近200个收盘价简单移动平均值。",
        direction=1,
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        return moving_average(bars, 200)


class MASlope20Factor(Factor):
    metadata = FactorMetadata(
        name="ma_slope_20",
        description="One-session percentage slope of the trailing 20-session average.",
        lookback=20,
        required_columns=("close",),
        display_name="20-Day Moving-Average Slope",
        display_name_zh="20日均线斜率",
        description_zh="20日移动平均线相对前一交易日的单日变化率。",
        direction=1,
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        return moving_average_slope(bars, 20)


class DistanceToMA200Factor(Factor):
    metadata = FactorMetadata(
        name="distance_to_ma_200",
        description="Close relative to its trailing 200-session moving average.",
        lookback=200,
        required_columns=("close",),
        display_name="Distance to 200-Day Moving Average",
        display_name_zh="距200日均线",
        description_zh="当前收盘价相对200日移动平均线的偏离比例。",
        direction=1,
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        return distance_to_moving_average(bars, 200)


class RSI14Factor(Factor):
    metadata = FactorMetadata(
        name="rsi_14",
        description="Fourteen-session Wilder relative strength index.",
        lookback=14,
        required_columns=("close",),
        display_name="14-Day Wilder RSI",
        display_name_zh="14日Wilder相对强弱指标",
        description_zh="采用Wilder平滑法计算的14日相对强弱指标，取值范围为0至100。",
        direction=1,
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        return wilder_rsi(bars, 14)


class BollingerMid20Factor(Factor):
    metadata = FactorMetadata(
        name="bollinger_mid_20",
        description="Middle line of 20-session Bollinger bands.",
        lookback=20,
        required_columns=("close",),
        display_name="20-Day Bollinger Middle Band",
        display_name_zh="20日布林带中轨",
        description_zh="最近20个收盘价的简单移动平均线。",
        direction=1,
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        return bollinger_bands(bars, 20)["mid"]


class BollingerUpper20Factor(Factor):
    metadata = FactorMetadata(
        name="bollinger_upper_20",
        description="Upper 20-session Bollinger band at two population deviations.",
        lookback=20,
        required_columns=("close",),
        display_name="20-Day Bollinger Upper Band",
        display_name_zh="20日布林带上轨",
        description_zh="20日中轨加上两倍总体标准差。",
        direction=1,
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        return bollinger_bands(bars, 20)["upper"]


class BollingerLower20Factor(Factor):
    metadata = FactorMetadata(
        name="bollinger_lower_20",
        description="Lower 20-session Bollinger band at two population deviations.",
        lookback=20,
        required_columns=("close",),
        display_name="20-Day Bollinger Lower Band",
        display_name_zh="20日布林带下轨",
        description_zh="20日中轨减去两倍总体标准差。",
        direction=1,
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        return bollinger_bands(bars, 20)["lower"]


class BollingerPercentB20Factor(Factor):
    metadata = FactorMetadata(
        name="bollinger_percent_b_20",
        description="Close position between the lower and upper 20-session bands.",
        lookback=20,
        required_columns=("close",),
        display_name="20-Day Bollinger Percent B",
        display_name_zh="20日布林带%B",
        description_zh="当前收盘价在20日布林带下轨与上轨之间的相对位置。",
        direction=1,
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        return bollinger_bands(bars, 20)["percent_b"]


class BollingerBandwidth20Factor(Factor):
    metadata = FactorMetadata(
        name="bollinger_bandwidth_20",
        description="Width of 20-session Bollinger bands divided by the middle line.",
        lookback=20,
        required_columns=("close",),
        display_name="20-Day Bollinger Bandwidth",
        display_name_zh="20日布林带宽度",
        description_zh="20日布林带上下轨之差除以中轨，较低表示波动较小。",
        direction=-1,
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        return bollinger_bands(bars, 20)["bandwidth"]


class Beta252Factor(_BenchmarkFactor):
    metadata = FactorMetadata(
        name="beta_252",
        description="252-session covariance with benchmark divided by benchmark variance.",
        lookback=252,
        required_columns=("close", "benchmark_close", "benchmark_return"),
        display_name="252-Day Market Beta",
        display_name_zh="252日市场贝塔",
        description_zh="最近252个交易日个股收益率与基准收益率协方差除以基准收益率方差。",
        direction=-1,
        direction_kind="negative",
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        ordered = bars.sort_values(["symbol", "date"])
        values = pd.Series(np.nan, index=ordered.index, dtype=float)
        for _, group in ordered.groupby("symbol", sort=False):
            stock_return = group["close"].pct_change(fill_method=None)
            benchmark_return = pd.to_numeric(
                group["benchmark_return"], errors="coerce"
            )
            covariance = stock_return.rolling(
                252, min_periods=252
            ).cov(benchmark_return)
            variance = benchmark_return.rolling(
                252, min_periods=252
            ).var()
            values.loc[group.index] = covariance / variance
        return _restore_order(bars, ordered, values)


def _rolling_ols_residual_measure(
    stock_return: pd.Series,
    benchmark_return: pd.Series,
    window: int,
    compound: bool,
) -> pd.Series:
    result = pd.Series(np.nan, index=stock_return.index, dtype=float)
    stock_values = pd.to_numeric(stock_return, errors="coerce").to_numpy(dtype=float)
    benchmark_values = pd.to_numeric(
        benchmark_return, errors="coerce"
    ).to_numpy(dtype=float)
    for end in range(window - 1, len(stock_values)):
        start = end - window + 1
        dependent = stock_values[start : end + 1]
        market = benchmark_values[start : end + 1]
        if not (np.isfinite(dependent).all() and np.isfinite(market).all()):
            continue
        design = np.column_stack([np.ones(window), market])
        coefficients, *_ = np.linalg.lstsq(design, dependent, rcond=None)
        residuals = dependent - design @ coefficients
        result.iloc[end] = (
            np.prod(1.0 + residuals) - 1.0
            if compound
            else residuals.std(ddof=0)
        )
    return result


class IdioVolatility60Factor(_BenchmarkFactor):
    metadata = FactorMetadata(
        name="idio_volatility_60",
        description=(
            "Population standard deviation of residuals from a sixty-session "
            "market-model OLS regression with intercept."
        ),
        lookback=60,
        required_columns=("close", "benchmark_close", "benchmark_return"),
        display_name="60-Day Idiosyncratic Volatility",
        display_name_zh="60日特质波动率",
        description_zh="个股收益率对含截距的基准收益率回归后，最近60期残差的总体标准差。",
        direction=-1,
        direction_kind="negative",
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        ordered = bars.sort_values(["symbol", "date"])
        values = pd.Series(np.nan, index=ordered.index, dtype=float)
        for _, group in ordered.groupby("symbol", sort=False):
            stock_return = group["close"].pct_change(fill_method=None)
            values.loc[group.index] = _rolling_ols_residual_measure(
                stock_return,
                group["benchmark_return"],
                window=60,
                compound=False,
            )
        return _restore_order(bars, ordered, values)


class RelativeStrength60Factor(_BenchmarkFactor):
    metadata = FactorMetadata(
        name="relative_strength_60",
        description="Security sixty-session return minus benchmark sixty-session return.",
        lookback=60,
        required_columns=("close", "benchmark_close", "benchmark_return"),
        display_name="60-Day Relative Strength",
        display_name_zh="60日相对强度",
        description_zh="个股60日收益率减去基准60日收益率。",
        direction=1,
        direction_kind="positive",
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        ordered = bars.sort_values(["symbol", "date"])
        stock_return = ordered.groupby("symbol", sort=False)["close"].transform(
            lambda close: close / close.shift(60) - 1.0
        )
        benchmark_return = ordered.groupby(
            "symbol", sort=False
        )["benchmark_close"].transform(
            lambda close: close / close.shift(60) - 1.0
        )
        return _restore_order(
            bars, ordered, stock_return - benchmark_return
        )


class ResidualMomentum60Factor(_BenchmarkFactor):
    metadata = FactorMetadata(
        name="residual_momentum_60",
        description=(
            "Compounded residual returns from a sixty-session market-model OLS "
            "regression with intercept."
        ),
        lookback=60,
        required_columns=("close", "benchmark_close", "benchmark_return"),
        display_name="60-Day Residual Momentum",
        display_name_zh="60日残差动量",
        description_zh="个股收益率对含截距的基准收益率回归后，最近60期残差收益的复合值。",
        direction=1,
        direction_kind="positive",
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        ordered = bars.sort_values(["symbol", "date"])
        values = pd.Series(np.nan, index=ordered.index, dtype=float)
        for _, group in ordered.groupby("symbol", sort=False):
            stock_return = group["close"].pct_change(fill_method=None)
            values.loc[group.index] = _rolling_ols_residual_measure(
                stock_return,
                group["benchmark_return"],
                window=60,
                compound=True,
            )
        return _restore_order(bars, ordered, values)


class PBFactor(Factor):
    metadata = FactorMetadata(
        name="pb",
        description=(
            "Point-in-time price-to-book ratio supplied by the data source; never "
            "backfilled from current fundamentals."
        ),
        lookback=0,
        required_columns=("pb",),
        availability="requires_point_in_time_pb",
        display_name="Price-to-Book",
        display_name_zh="市净率",
        description_zh="数据源提供的历史时点市净率，不使用当前基本面数据回填。",
        direction=-1,
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        values = pd.to_numeric(bars["pb"], errors="coerce")
        if not values.notna().any():
            raise FactorUnavailableError(
                "Factor 'pb' is unavailable because cached bars contain no "
                "point-in-time PB values. PB was not fabricated."
            )
        return values.replace([np.inf, -np.inf], np.nan)


class PointInTimeFundamentalFactor(Factor):
    def __init__(
        self,
        *,
        name: str,
        source_column: str,
        display_name: str,
        display_name_zh: str,
        description_zh: str,
        direction: int,
    ) -> None:
        self.source_column = source_column
        self.metadata = FactorMetadata(
            name=name,
            description=(
                f"Point-in-time {source_column} supplied by the data source; "
                "current fundamentals are never backfilled into history."
            ),
            lookback=0,
            required_columns=(
                source_column,
                "fundamentals_are_point_in_time",
            ),
            availability="requires_point_in_time_fundamentals",
            display_name=display_name,
            display_name_zh=display_name_zh,
            description_zh=description_zh,
            direction=direction,
            applicable_assets=("stock",),
        )

    def validate(self, bars: pd.DataFrame) -> None:
        super().validate(bars)
        provenance = bars["fundamentals_are_point_in_time"].fillna(False).astype(bool)
        if not provenance.any():
            raise FactorUnavailableError(
                f"Factor '{self.metadata.name}' is unavailable because the cache "
                "does not certify point-in-time fundamentals. Current financial "
                "data was not backfilled."
            )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        provenance = bars["fundamentals_are_point_in_time"].fillna(False).astype(bool)
        values = pd.to_numeric(bars[self.source_column], errors="coerce").where(
            provenance
        )
        if not values.notna().any():
            raise FactorUnavailableError(
                f"Factor '{self.metadata.name}' is unavailable because cached "
                f"point-in-time field '{self.source_column}' has no usable values."
            )
        return values.replace([np.inf, -np.inf], np.nan)


FUNDAMENTAL_PLACEHOLDER_FACTORS: tuple[Factor, ...] = (
    PointInTimeFundamentalFactor(
        name="bp",
        source_column="bp",
        display_name="Book-to-Market",
        display_name_zh="账面市值比BP",
        description_zh="历史时点账面价值与市值之比；没有时点数据时保持不可用。",
        direction=1,
    ),
    PointInTimeFundamentalFactor(
        name="ep",
        source_column="ep",
        display_name="Earnings Yield",
        display_name_zh="盈利收益率EP",
        description_zh="历史时点盈利与市值之比；没有时点数据时保持不可用。",
        direction=1,
    ),
    PointInTimeFundamentalFactor(
        name="dividend_yield",
        source_column="dividend_yield",
        display_name="Dividend Yield",
        display_name_zh="股息率",
        description_zh="历史时点股息率；没有时点数据时保持不可用。",
        direction=1,
    ),
    PointInTimeFundamentalFactor(
        name="roe",
        source_column="roe",
        display_name="Return on Equity",
        display_name_zh="ROE",
        description_zh="历史时点净资产收益率；没有时点数据时保持不可用。",
        direction=1,
    ),
    PointInTimeFundamentalFactor(
        name="gross_margin",
        source_column="gross_margin",
        display_name="Gross Margin",
        display_name_zh="毛利率",
        description_zh="历史时点毛利率；没有时点数据时保持不可用。",
        direction=1,
    ),
    PointInTimeFundamentalFactor(
        name="operating_cashflow_to_assets",
        source_column="operating_cashflow_to_assets",
        display_name="Operating Cash Flow to Assets",
        display_name_zh="经营现金流/资产",
        description_zh="历史时点经营现金流与总资产之比；没有时点数据时保持不可用。",
        direction=1,
    ),
    PointInTimeFundamentalFactor(
        name="accruals",
        source_column="accruals",
        display_name="Accruals",
        display_name_zh="应计利润",
        description_zh="历史时点应计利润指标，较低通常更优；没有时点数据时保持不可用。",
        direction=-1,
    ),
    PointInTimeFundamentalFactor(
        name="asset_growth",
        source_column="asset_growth",
        display_name="Asset Growth",
        display_name_zh="资产增长率",
        description_zh="历史时点资产增长率，较低通常更优；没有时点数据时保持不可用。",
        direction=-1,
    ),
    PointInTimeFundamentalFactor(
        name="market_cap_size",
        source_column="market_cap",
        display_name="Market Capitalization",
        display_name_zh="市值规模",
        description_zh="历史时点总市值，较小规模得分更高；没有时点数据时保持不可用。",
        direction=-1,
    ),
)


BUILTIN_FACTORS: tuple[Factor, ...] = (
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
    MA200Factor(),
    MASlope20Factor(),
    DistanceToMA200Factor(),
    RSI14Factor(),
    BollingerMid20Factor(),
    BollingerUpper20Factor(),
    BollingerLower20Factor(),
    BollingerPercentB20Factor(),
    BollingerBandwidth20Factor(),
    Beta252Factor(),
    IdioVolatility60Factor(),
    RelativeStrength60Factor(),
    ResidualMomentum60Factor(),
    DownsideVolatility20Factor(),
    Amihud20Factor(),
    PBFactor(),
    *FUNDAMENTAL_PLACEHOLDER_FACTORS,
)
