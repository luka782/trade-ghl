from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.data.base import Adjustment, normalize_bars, normalize_symbol


def make_synthetic_bars(
    symbols: list[str],
    periods: int = 80,
    start: str = "2024-01-02",
) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=periods)
    frames: list[pd.DataFrame] = []
    for index, symbol in enumerate(symbols):
        drift = 0.0005 + index * 0.0007
        steps = np.arange(periods)
        close = (9.0 + index) * np.power(1.0 + drift, steps)
        close *= 1.0 + 0.002 * np.sin(steps / 4.0 + index)
        frame = pd.DataFrame(
            {
                "symbol": normalize_symbol(symbol),
                "date": dates,
                "open": close * 0.998,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000.0 + index * 10_000 + steps * 100,
                "amount": close * (1_000_000.0 + steps * 100),
                "is_st": False,
            }
        )
        frames.append(frame)
    return normalize_bars(pd.concat(frames, ignore_index=True))


class SyntheticProvider:
    name = "synthetic"

    def __init__(self, bars: pd.DataFrame) -> None:
        self.bars = normalize_bars(bars)
        index_dates = sorted(self.bars["date"].unique())
        index_close = 1000.0 * np.power(1.0008, np.arange(len(index_dates)))
        self.index_bars = normalize_bars(
            pd.DataFrame(
                {
                    "symbol": "000300",
                    "date": index_dates,
                    "open": index_close,
                    "high": index_close * 1.005,
                    "low": index_close * 0.995,
                    "close": index_close,
                    "volume": 10_000_000.0,
                }
            )
        )
        self.fetch_calls: list[tuple[str, date, date, Adjustment]] = []

    def list_stocks(self) -> pd.DataFrame:
        symbols = sorted(self.bars["symbol"].unique())
        return pd.DataFrame(
            {
                "symbol": symbols,
                "name": [f"Synthetic {symbol}" for symbol in symbols],
            }
        )

    def fetch_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        adjust: Adjustment,
    ) -> pd.DataFrame:
        normalized_symbol = normalize_symbol(symbol)
        self.fetch_calls.append((normalized_symbol, start_date, end_date, adjust))
        return self.bars[
            (self.bars["symbol"] == normalized_symbol)
            & (self.bars["date"] >= pd.Timestamp(start_date))
            & (self.bars["date"] <= pd.Timestamp(end_date))
        ].copy()

    def fetch_index(
        self, symbol: str, start_date: date, end_date: date
    ) -> pd.DataFrame | None:
        if normalize_symbol(symbol) not in {"000300", "000905"}:
            return None
        frame = self.index_bars.copy()
        frame["symbol"] = normalize_symbol(symbol)
        return frame[
            (frame["date"] >= pd.Timestamp(start_date))
            & (frame["date"] <= pd.Timestamp(end_date))
        ].copy()

    def fetch_trade_calendar(
        self, start_date: date, end_date: date
    ) -> pd.DataFrame:
        dates = pd.DatetimeIndex(sorted(self.bars["date"].unique()))
        dates = dates[
            (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))
        ]
        return pd.DataFrame({"trade_date": dates})

    def fetch_adjustment_factors(
        self, symbol: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        dates = pd.DatetimeIndex(sorted(self.bars["date"].unique()))
        dates = dates[
            (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))
        ]
        return pd.DataFrame(
            {
                "symbol": normalize_symbol(symbol),
                "date": dates,
                "qfq_factor": 1.0,
            }
        )


@pytest.fixture
def synthetic_bars() -> pd.DataFrame:
    return make_synthetic_bars(
        ["600001", "600002", "600003", "600004", "600005"]
    )


@pytest.fixture
def synthetic_provider(synthetic_bars: pd.DataFrame) -> SyntheticProvider:
    return SyntheticProvider(synthetic_bars)
