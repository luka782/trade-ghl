from __future__ import annotations

from datetime import date
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd


Adjustment = Literal["qfq", "none"]
BAR_REQUIRED_COLUMNS = ("symbol", "date", "open", "high", "low", "close", "volume")
BAR_OPTIONAL_COLUMNS = (
    "amount",
    "prev_close",
    "is_st",
    "is_st_known",
    "industry",
    "market_cap",
    "pb",
    "bp",
    "ep",
    "dividend_yield",
    "roe",
    "gross_margin",
    "operating_cashflow_to_assets",
    "accruals",
    "asset_growth",
    "fundamentals_are_point_in_time",
)


class ProviderDataError(RuntimeError):
    """Raised when a market-data provider changes or returns an invalid contract."""


def normalize_symbol(value: object) -> str:
    symbol = str(value).strip().upper()
    for suffix in (".SH", ".SZ", ".BJ"):
        if symbol.endswith(suffix):
            symbol = symbol[: -len(suffix)]
            break
    for prefix in ("SH.", "SZ.", "BJ.", "SH", "SZ", "BJ"):
        if symbol.startswith(prefix):
            symbol = symbol[len(prefix) :]
            break
    if symbol.isdigit():
        symbol = symbol.zfill(6)
    if not symbol or not all(char.isalnum() or char in "._-" for char in symbol):
        raise ProviderDataError(f"Invalid symbol returned by provider: {value!r}")
    return symbol


def _coerce_bool(value: object) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "是", "st"}
    return bool(value)


def normalize_bars(frame: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """Validate and normalize provider bars.

    The normalized contract is one row per ``symbol``/``date`` with lowercase
    OHLCV columns. Optional point-in-time fields are preserved only when supplied;
    in particular, ``pb`` is never synthesized.
    """

    if not isinstance(frame, pd.DataFrame):
        raise ProviderDataError("Provider bars must be a pandas DataFrame")

    bars = frame.copy()
    if symbol is not None:
        bars["symbol"] = normalize_symbol(symbol)

    missing = [column for column in BAR_REQUIRED_COLUMNS if column not in bars.columns]
    if missing:
        raise ProviderDataError(
            "Provider bars missing normalized columns: " + ", ".join(missing)
        )

    if bars.empty:
        return bars.loc[:, list(dict.fromkeys((*BAR_REQUIRED_COLUMNS, *bars.columns)))]

    bars["symbol"] = bars["symbol"].map(normalize_symbol)
    bars["date"] = pd.to_datetime(bars["date"], errors="coerce").dt.normalize()
    numeric_columns = [
        column
        for column in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "prev_close",
            "market_cap",
            "pb",
            "bp",
            "ep",
            "dividend_yield",
            "roe",
            "gross_margin",
            "operating_cashflow_to_assets",
            "accruals",
            "asset_growth",
        )
        if column in bars.columns
    ]
    for column in numeric_columns:
        bars[column] = pd.to_numeric(bars[column], errors="coerce")

    invalid_dates = int(bars["date"].isna().sum())
    invalid_prices = int(
        bars[["open", "high", "low", "close"]].isna().any(axis=1).sum()
    )
    if invalid_dates or invalid_prices:
        raise ProviderDataError(
            f"Provider bars contain {invalid_dates} invalid dates and "
            f"{invalid_prices} rows with invalid OHLC prices"
        )
    if (bars[["open", "high", "low", "close"]] < 0).any().any():
        raise ProviderDataError("Provider bars contain negative OHLC prices")

    if "is_st" in bars.columns:
        bars["is_st"] = bars["is_st"].map(_coerce_bool).astype(bool)
        if "is_st_known" in bars.columns:
            bars["is_st_known"] = (
                bars["is_st_known"].map(_coerce_bool).astype(bool)
            )
        else:
            # Old caches and generic providers often synthesized is_st=False.
            # Absence of provenance must remain "unknown", not "known non-ST".
            bars["is_st_known"] = False
    else:
        bars["is_st"] = False
        bars["is_st_known"] = False
    if "fundamentals_are_point_in_time" in bars.columns:
        bars["fundamentals_are_point_in_time"] = (
            bars["fundamentals_are_point_in_time"].map(_coerce_bool).astype(bool)
        )

    bars = (
        bars.sort_values(["symbol", "date"])
        .drop_duplicates(["symbol", "date"], keep="last")
        .reset_index(drop=True)
    )
    shifted_close = bars.groupby("symbol", sort=False)["close"].shift(1)
    if "prev_close" not in bars.columns:
        bars["prev_close"] = shifted_close
    else:
        bars["prev_close"] = bars["prev_close"].fillna(shifted_close)

    ordered = [
        *BAR_REQUIRED_COLUMNS,
        *[column for column in BAR_OPTIONAL_COLUMNS if column in bars.columns],
    ]
    ordered.extend(column for column in bars.columns if column not in ordered)
    return bars.loc[:, ordered]


@runtime_checkable
class DataProvider(Protocol):
    name: str

    def list_stocks(self) -> pd.DataFrame:
        """Return columns ``symbol`` and ``name`` (additional columns allowed)."""

    def fetch_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        adjust: Adjustment,
    ) -> pd.DataFrame:
        """Return normalized daily bars for one stock."""

    def fetch_index(
        self, symbol: str, start_date: date, end_date: date
    ) -> pd.DataFrame | None:
        """Return normalized index bars, or ``None`` when unavailable."""

    def fetch_trade_calendar(
        self, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Return a ``trade_date`` column filtered to the requested range."""

    def fetch_adjustment_factors(
        self, symbol: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Return point-in-time ``date`` and ``qfq_factor`` columns."""
