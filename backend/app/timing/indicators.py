from __future__ import annotations

import numpy as np
import pandas as pd


# 指标函数接受多证券面板，但始终按 symbol 分组、按 date 排序计算。
# 每个 T 日结果只依赖 T 日及更早价格，可直接作为 T 日收盘后的择时输入。
def _validate_inputs(
    bars: pd.DataFrame,
    *,
    window: int,
    price_column: str,
) -> None:
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    missing = [
        column
        for column in ("symbol", "date", price_column)
        if column not in bars.columns
    ]
    if missing:
        raise ValueError(
            "Indicator input is missing required columns: " + ", ".join(missing)
        )


def _ordered_prices(
    bars: pd.DataFrame, price_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """建立稳定的按证券/日期顺序，并保留原始行位置以便无损还原。"""
    ordered = bars.assign(_indicator_row=np.arange(len(bars))).sort_values(
        ["symbol", "date"], kind="stable"
    )
    prices = pd.to_numeric(ordered[price_column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    return ordered, prices


def _restore_order(
    bars: pd.DataFrame, ordered: pd.DataFrame, values: pd.Series
) -> pd.Series:
    """将按时间排序的计算结果放回调用方原始 DataFrame 的索引顺序。"""
    result = np.full(len(bars), np.nan, dtype=float)
    result[ordered["_indicator_row"].to_numpy(dtype=int)] = pd.to_numeric(
        values, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    return pd.Series(result, index=bars.index, dtype=float)


def moving_average(
    bars: pd.DataFrame,
    window: int,
    *,
    price_column: str = "close",
) -> pd.Series:
    """计算每只证券完整窗口的简单移动平均线（SMA）。"""
    _validate_inputs(bars, window=window, price_column=price_column)
    ordered, prices = _ordered_prices(bars, price_column)
    values = prices.groupby(ordered["symbol"], sort=False).transform(
        lambda price: price.rolling(window, min_periods=window).mean()
    )
    return _restore_order(bars, ordered, values)


def moving_average_slope(
    bars: pd.DataFrame,
    window: int,
    *,
    slope_periods: int = 1,
    price_column: str = "close",
) -> pd.Series:
    """计算均线在指定间隔内的日均百分比斜率，用于判别长期趋势方向。"""
    _validate_inputs(bars, window=window, price_column=price_column)
    if (
        isinstance(slope_periods, bool)
        or not isinstance(slope_periods, int)
        or slope_periods < 1
    ):
        raise ValueError("slope_periods must be a positive integer")
    ordered, prices = _ordered_prices(bars, price_column)

    def calculate(price: pd.Series) -> pd.Series:
        average = price.rolling(window, min_periods=window).mean()
        return (average / average.shift(slope_periods) - 1.0) / slope_periods

    values = prices.groupby(ordered["symbol"], sort=False).transform(calculate)
    return _restore_order(bars, ordered, values)


def distance_to_moving_average(
    bars: pd.DataFrame,
    window: int,
    *,
    price_column: str = "close",
) -> pd.Series:
    """Return close divided by its trailing moving average, minus one."""
    _validate_inputs(bars, window=window, price_column=price_column)
    ordered, prices = _ordered_prices(bars, price_column)
    average = prices.groupby(ordered["symbol"], sort=False).transform(
        lambda price: price.rolling(window, min_periods=window).mean()
    )
    values = prices / average - 1.0
    return _restore_order(bars, ordered, values)


def _wilder_rsi_for_group(prices: pd.Series, window: int) -> pd.Series:
    """对连续有效价格区间实现 Wilder 平滑 RSI。

    空值会断开价格序列，防止停牌/缺失行情被错误当作零收益连接起来。
    """
    values = prices.to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    finite = np.isfinite(values)
    start = 0
    while start < len(values):
        while start < len(values) and not finite[start]:
            start += 1
        end = start
        while end < len(values) and finite[end]:
            end += 1
        if end - start > window:
            changes = np.diff(values[start:end])
            average_gain = float(np.maximum(changes[:window], 0.0).mean())
            average_loss = float(np.maximum(-changes[:window], 0.0).mean())
            for offset in range(window, end - start):
                if offset > window:
                    change = changes[offset - 1]
                    average_gain = (
                        average_gain * (window - 1) + max(change, 0.0)
                    ) / window
                    average_loss = (
                        average_loss * (window - 1) + max(-change, 0.0)
                    ) / window
                position = start + offset
                if average_loss == 0.0:
                    result[position] = 50.0 if average_gain == 0.0 else 100.0
                else:
                    relative_strength = average_gain / average_loss
                    result[position] = 100.0 - 100.0 / (1.0 + relative_strength)
        start = end + 1
    return pd.Series(result, index=prices.index, dtype=float)


def wilder_rsi(
    bars: pd.DataFrame,
    window: int = 14,
    *,
    price_column: str = "close",
) -> pd.Series:
    """计算 Wilder RSI；首个值用完整窗口的涨跌幅均值初始化。"""
    _validate_inputs(bars, window=window, price_column=price_column)
    ordered, prices = _ordered_prices(bars, price_column)
    values = prices.groupby(ordered["symbol"], sort=False).transform(
        lambda price: _wilder_rsi_for_group(price, window)
    )
    return _restore_order(bars, ordered, values)


def bollinger_bands(
    bars: pd.DataFrame,
    window: int = 20,
    *,
    standard_deviations: float = 2.0,
    price_column: str = "close",
) -> pd.DataFrame:
    """计算因果布林带及派生指标。

    ``percent_b`` 表示价格在上下轨间的位置，``bandwidth`` 描述波动收缩/
    扩张；两者均使用包含 T 日收盘在内的滚动窗口，不读取未来价格。
    """
    _validate_inputs(bars, window=window, price_column=price_column)
    if not np.isfinite(standard_deviations) or standard_deviations <= 0:
        raise ValueError("standard_deviations must be a positive finite number")
    ordered, prices = _ordered_prices(bars, price_column)
    grouped = prices.groupby(ordered["symbol"], sort=False)
    mid = grouped.transform(
        lambda price: price.rolling(window, min_periods=window).mean()
    )
    deviation = grouped.transform(
        lambda price: price.rolling(window, min_periods=window).std(ddof=0)
    )
    upper = mid + standard_deviations * deviation
    lower = mid - standard_deviations * deviation
    width = upper - lower
    ordered_values = {
        "mid": mid,
        "upper": upper,
        "lower": lower,
        "percent_b": (prices - lower) / width,
        "bandwidth": width / mid,
    }
    return pd.DataFrame(
        {
            name: _restore_order(bars, ordered, values)
            for name, values in ordered_values.items()
        },
        index=bars.index,
    )


def average_true_range(
    bars: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """计算只使用当日及更早 OHLC 的简单移动平均真实波幅。"""
    _validate_inputs(bars, window=window, price_column="close")
    missing = [column for column in ("high", "low") if column not in bars]
    if missing:
        raise ValueError(
            "ATR input is missing required columns: " + ", ".join(missing)
        )
    ordered, close = _ordered_prices(bars, "close")
    high = pd.to_numeric(
        ordered["high"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    low = pd.to_numeric(
        ordered["low"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    previous_close = close.groupby(
        ordered["symbol"], sort=False
    ).shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    values = true_range.groupby(
        ordered["symbol"], sort=False
    ).transform(
        lambda item: item.rolling(window, min_periods=window).mean()
    )
    return _restore_order(bars, ordered, values)


def donchian_channels(
    bars: pd.DataFrame,
    entry_window: int = 55,
    exit_window: int = 20,
) -> pd.DataFrame:
    """计算不包含 T 日自身的 Donchian 上下轨，供 T 日收盘判断突破。

    ``shift(1)`` 保证上轨和下轨只来自 T-1 及更早行情；否则把 T 日最高/
    最低价放进比较窗口会改变突破定义，也容易掩盖前视错误。
    """
    _validate_inputs(bars, window=entry_window, price_column="close")
    if isinstance(exit_window, bool) or not isinstance(exit_window, int) or exit_window < 1:
        raise ValueError("exit_window must be a positive integer")
    missing = [column for column in ("high", "low") if column not in bars]
    if missing:
        raise ValueError(
            "Donchian input is missing required columns: " + ", ".join(missing)
        )
    ordered, _ = _ordered_prices(bars, "close")
    high = pd.to_numeric(
        ordered["high"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    low = pd.to_numeric(
        ordered["low"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    upper = high.groupby(
        ordered["symbol"], sort=False
    ).transform(
        lambda item: item.shift(1).rolling(
            entry_window, min_periods=entry_window
        ).max()
    )
    lower = low.groupby(
        ordered["symbol"], sort=False
    ).transform(
        lambda item: item.shift(1).rolling(
            exit_window, min_periods=exit_window
        ).min()
    )
    return pd.DataFrame(
        {
            "upper": _restore_order(bars, ordered, upper),
            "lower": _restore_order(bars, ordered, lower),
        },
        index=bars.index,
    )


ma = moving_average
ma_slope = moving_average_slope
distance_to_ma = distance_to_moving_average
rsi = wilder_rsi


__all__ = [
    "average_true_range",
    "bollinger_bands",
    "donchian_channels",
    "distance_to_ma",
    "distance_to_moving_average",
    "ma",
    "ma_slope",
    "moving_average",
    "moving_average_slope",
    "rsi",
    "wilder_rsi",
]
