from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import app.data.akshare_provider as provider_module
from app.data.akshare_provider import AkShareProvider
from app.data.base import ProviderDataError


class FakeAkShare:
    def stock_info_a_code_name(self) -> pd.DataFrame:
        return pd.DataFrame({"code": ["1"], "name": ["平安银行"]})

    def stock_zh_a_hist(self, **_: object) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "日期": ["2024-01-02", "2024-01-03"],
                "开盘": [10.0, 10.1],
                "最高": [10.2, 10.3],
                "最低": [9.9, 10.0],
                "收盘": [10.1, 10.2],
                "成交量": [1000, 1100],
                "成交额": [10100, 11220],
                "市净率": [1.2, 1.21],
            }
        )

    def tool_trade_date_hist_sina(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"trade_date": ["2024-01-02", "2024-01-03", "2024-01-04"]}
        )

    def stock_zh_a_daily(self, **_: object) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03"],
                "qfq_factor": [1.0, 1.01],
            }
        )


def test_akshare_adapter_maps_chinese_columns_at_runtime(monkeypatch) -> None:
    monkeypatch.setattr(provider_module, "_akshare", lambda: FakeAkShare())
    provider = AkShareProvider()
    bars = provider.fetch_bars(
        "1", date(2024, 1, 2), date(2024, 1, 3), "qfq"
    )
    assert list(bars["symbol"].unique()) == ["000001"]
    assert {"date", "open", "high", "low", "close", "volume", "amount", "pb"}.issubset(
        bars.columns
    )
    assert bars["volume"].iloc[0] == 100_000
    assert not bars["is_st_known"].any()
    assert provider.list_stocks().to_dict(orient="records") == [
        {"symbol": "000001", "name": "平安银行"}
    ]
    calendar = provider.fetch_trade_calendar(
        date(2024, 1, 2), date(2024, 1, 3)
    )
    assert len(calendar) == 2
    factors = provider.fetch_adjustment_factors(
        "000001", date(2024, 1, 2), date(2024, 1, 3)
    )
    assert factors["qfq_factor"].tolist() == [1.0, 1.01]


def test_akshare_adapter_reports_schema_changes(monkeypatch) -> None:
    fake = FakeAkShare()
    fake.stock_zh_a_hist = lambda **_: pd.DataFrame({"日期": ["2024-01-02"]})
    monkeypatch.setattr(provider_module, "_akshare", lambda: fake)
    with pytest.raises(ProviderDataError, match="schema changed"):
        AkShareProvider().fetch_bars(
            "000001", date(2024, 1, 2), date(2024, 1, 3), "none"
        )


def test_akshare_adapter_falls_back_to_sina_history(monkeypatch) -> None:
    fake = FakeAkShare()
    fake.stock_zh_a_hist = lambda **_: (_ for _ in ()).throw(
        ConnectionError("primary unavailable")
    )
    fake.stock_zh_a_daily = lambda **_: pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "open": [10.0, 10.1],
            "high": [10.2, 10.3],
            "low": [9.9, 10.0],
            "close": [10.1, 10.2],
            "volume": [100_000, 110_000],
            "amount": [1_010_000, 1_122_000],
        }
    )
    monkeypatch.setattr(provider_module, "_akshare", lambda: fake)
    bars = AkShareProvider().fetch_bars(
        "000001", date(2024, 1, 2), date(2024, 1, 3), "qfq"
    )
    assert len(bars) == 2
    assert bars["volume"].iloc[0] == 100_000


def test_etf_history_falls_back_to_tencent_with_adjustment(monkeypatch) -> None:
    fake = FakeAkShare()
    fake.fund_etf_hist_em = lambda **_: (_ for _ in ()).throw(
        ConnectionError("primary ETF endpoint unavailable")
    )
    calls: list[dict[str, object]] = []

    def tencent_history(**kwargs: object) -> pd.DataFrame:
        calls.append(kwargs)
        return pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03"],
                "open": [1.0, 1.01],
                "high": [1.02, 1.03],
                "low": [0.99, 1.0],
                "close": [1.01, 1.02],
                "volume": [1_000_000, 1_100_000],
                "amount": [1_010_000, 1_122_000],
            }
        )

    fake.stock_zh_a_hist_tx = tencent_history
    monkeypatch.setattr(provider_module, "_akshare", lambda: fake)
    bars = AkShareProvider().fetch_bars(
        "515080", date(2024, 1, 2), date(2024, 1, 3), "qfq"
    )
    assert len(bars) == 2
    assert bars["symbol"].unique().tolist() == ["515080"]
    assert bars["is_st_known"].all()
    assert calls[0]["symbol"] == "sh515080"
    assert calls[0]["adjust"] == "qfq"
