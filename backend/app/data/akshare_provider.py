from __future__ import annotations

import importlib
from datetime import date
from typing import Any

import pandas as pd

from .base import Adjustment, ProviderDataError, normalize_bars, normalize_symbol


# AkShare 的字段名和接口偶有变化。本适配层将中文/英文列统一成内部数据契约，
# 使因子和回测引擎不依赖某个具体上游接口。
_BAR_ALIASES: dict[str, tuple[str, ...]] = {
    "date": ("日期", "date", "交易日期"),
    "open": ("开盘", "open", "开盘价"),
    "high": ("最高", "high", "最高价"),
    "low": ("最低", "low", "最低价"),
    "close": ("收盘", "close", "收盘价"),
    "volume": ("成交量", "volume", "成交量(手)"),
}
_OPTIONAL_ALIASES: dict[str, tuple[str, ...]] = {
    "amount": ("成交额", "amount"),
    "prev_close": ("昨收", "前收盘", "prev_close"),
    "is_st": ("是否ST", "is_st"),
    "industry": ("行业", "所属行业", "industry"),
    "market_cap": ("总市值", "流通市值", "market_cap"),
    "pb": ("市净率", "PB", "pb"),
}


def _akshare() -> Any:
    try:
        return importlib.import_module("akshare")
    except ImportError as exc:  # pragma: no cover - depends on optional runtime install
        raise RuntimeError(
            "AkShare is not installed. Install backend dependencies before downloading data."
        ) from exc


def _map_columns(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    """将上游不同字段别名映射为标准 OHLCV 列，并在模式漂移时快速失败。"""
    if not isinstance(frame, pd.DataFrame):
        raise ProviderDataError(f"{context} returned a non-DataFrame value")

    rename: dict[str, str] = {}
    missing: list[str] = []
    for normalized, aliases in _BAR_ALIASES.items():
        source = next((candidate for candidate in aliases if candidate in frame.columns), None)
        if source is None:
            missing.append(f"{normalized} ({'/'.join(aliases)})")
        else:
            rename[source] = normalized
    if missing:
        actual = ", ".join(map(str, frame.columns))
        raise ProviderDataError(
            f"{context} schema changed; missing {', '.join(missing)}. "
            f"Actual columns: [{actual}]"
        )

    for normalized, aliases in _OPTIONAL_ALIASES.items():
        source = next((candidate for candidate in aliases if candidate in frame.columns), None)
        if source is not None:
            rename[source] = normalized
    return frame.rename(columns=rename)


def _market_prefixed_symbol(symbol: str) -> str:
    """将六码证券代码转为新浪类接口要求的 sh/sz/bj 市场前缀代码。"""
    normalized = normalize_symbol(symbol)
    if normalized.startswith(("4", "8")):
        return f"bj{normalized}"
    if normalized.startswith(("5", "6", "9")):
        return f"sh{normalized}"
    return f"sz{normalized}"


def _is_etf_symbol(symbol: str) -> bool:
    normalized = normalize_symbol(symbol)
    return normalized.startswith(("15", "51", "56", "58"))


class AkShareProvider:
    """AkShare 数据适配器，带运行时字段校验、统一单位和主/备接口回退。"""

    name = "akshare"

    def list_stocks(self) -> pd.DataFrame:
        frame = _akshare().stock_info_a_code_name()
        if not isinstance(frame, pd.DataFrame):
            raise ProviderDataError("AkShare stock list returned a non-DataFrame value")

        code_column = next(
            (column for column in ("code", "代码", "股票代码") if column in frame.columns),
            None,
        )
        name_column = next(
            (column for column in ("name", "名称", "股票简称") if column in frame.columns),
            None,
        )
        if code_column is None or name_column is None:
            raise ProviderDataError(
                "AkShare stock-list schema changed; expected code/name columns, got "
                f"{list(frame.columns)!r}"
            )
        stocks = frame.rename(
            columns={code_column: "symbol", name_column: "name"}
        ).copy()
        stocks["symbol"] = stocks["symbol"].map(normalize_symbol)
        stocks["name"] = stocks["name"].astype(str)
        return stocks.sort_values("symbol").reset_index(drop=True)

    def fetch_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        adjust: Adjustment,
    ) -> pd.DataFrame:
        """下载股票或 ETF 日线；东方财富接口失败时回退到备用接口。

        内部成交量统一为“股/份”，而东方财富通常以“手”返回，因此要乘以 100。
        """
        normalized_symbol = normalize_symbol(symbol)
        ak = _akshare()
        if _is_etf_symbol(normalized_symbol):
            return self._fetch_etf_bars(
                ak,
                normalized_symbol,
                start_date,
                end_date,
                adjust,
            )
        try:
            frame = ak.stock_zh_a_hist(
                symbol=normalized_symbol,
                period="daily",
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
                adjust="qfq" if adjust == "qfq" else "",
            )
            mapped = _map_columns(
                frame, f"AkShare stock_zh_a_hist({normalized_symbol})"
            )
            # Eastmoney reports stock volume in lots; the normalized contract uses shares.
            mapped["volume"] = pd.to_numeric(
                mapped["volume"], errors="coerce"
            ) * 100.0
        except Exception as primary_error:
            try:
                frame = ak.stock_zh_a_daily(
                    symbol=_market_prefixed_symbol(normalized_symbol),
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=end_date.strftime("%Y%m%d"),
                    adjust="qfq" if adjust == "qfq" else "",
                )
                mapped = _map_columns(
                    frame, f"AkShare stock_zh_a_daily({normalized_symbol})"
                )
            except Exception as fallback_error:
                raise ProviderDataError(
                    f"Both AkShare stock history endpoints failed for "
                    f"{normalized_symbol}; primary: {primary_error}; "
                    f"fallback: {fallback_error}"
                ) from fallback_error
        if "is_st" in mapped.columns:
            mapped["is_st_known"] = mapped["is_st"].notna()
        mapped["symbol"] = normalized_symbol
        return normalize_bars(mapped)

    def _fetch_etf_bars(
        self,
        ak: Any,
        symbol: str,
        start_date: date,
        end_date: date,
        adjust: Adjustment,
    ) -> pd.DataFrame:
        try:
            frame = ak.fund_etf_hist_em(
                symbol=symbol,
                period="daily",
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
                adjust="qfq" if adjust == "qfq" else "",
            )
            mapped = _map_columns(frame, f"AkShare fund_etf_hist_em({symbol})")
            # Eastmoney reports ETF volume in lots; normalized storage uses shares.
            mapped["volume"] = pd.to_numeric(
                mapped["volume"], errors="coerce"
            ) * 100.0
        except Exception as primary_error:
            try:
                frame = ak.stock_zh_a_hist_tx(
                    symbol=_market_prefixed_symbol(symbol),
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=end_date.strftime("%Y%m%d"),
                    adjust="qfq" if adjust == "qfq" else "",
                )
                mapped = _map_columns(
                    frame,
                    f"AkShare stock_zh_a_hist_tx ETF fallback({symbol})",
                )
            except Exception as fallback_error:
                raise ProviderDataError(
                    f"Both AkShare ETF history endpoints failed for {symbol}; "
                    f"primary: {primary_error}; fallback: {fallback_error}"
                ) from fallback_error
        mapped["symbol"] = symbol
        mapped["is_st"] = False
        mapped["is_st_known"] = True
        return normalize_bars(mapped)

    def fetch_index(
        self, symbol: str, start_date: date, end_date: date
    ) -> pd.DataFrame | None:
        normalized_symbol = normalize_symbol(symbol)
        ak = _akshare()
        em_symbol = (
            f"csi{normalized_symbol}"
            if normalized_symbol == "000905"
            else f"sh{normalized_symbol}"
        )
        candidates = (
            (
                "index_zh_a_hist",
                lambda: ak.index_zh_a_hist(
                    symbol=normalized_symbol,
                    period="daily",
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=end_date.strftime("%Y%m%d"),
                ),
            ),
            (
                "stock_zh_index_daily_em",
                lambda: ak.stock_zh_index_daily_em(
                    symbol=em_symbol,
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=end_date.strftime("%Y%m%d"),
                ),
            ),
            (
                "stock_zh_index_daily",
                lambda: ak.stock_zh_index_daily(symbol=f"sh{normalized_symbol}"),
            ),
        )
        for endpoint, fetch in candidates:
            try:
                frame = fetch()
                if frame is None or frame.empty:
                    continue
                mapped = _map_columns(
                    frame, f"AkShare {endpoint}({normalized_symbol})"
                )
                mapped["symbol"] = normalized_symbol
                normalized = normalize_bars(mapped)
                return normalized[
                    (normalized["date"] >= pd.Timestamp(start_date))
                    & (normalized["date"] <= pd.Timestamp(end_date))
                ].reset_index(drop=True)
            except Exception:
                continue
        return None

    def fetch_trade_calendar(
        self, start_date: date, end_date: date
    ) -> pd.DataFrame:
        frame = _akshare().tool_trade_date_hist_sina()
        if not isinstance(frame, pd.DataFrame):
            raise ProviderDataError(
                "AkShare trade calendar returned a non-DataFrame value"
            )
        date_column = next(
            (
                column
                for column in ("trade_date", "日期", "date")
                if column in frame.columns
            ),
            None,
        )
        if date_column is None:
            raise ProviderDataError(
                "AkShare trade-calendar schema changed; expected trade_date, got "
                f"{list(frame.columns)!r}"
            )
        calendar = frame.rename(columns={date_column: "trade_date"})[
            ["trade_date"]
        ].copy()
        calendar["trade_date"] = pd.to_datetime(
            calendar["trade_date"], errors="coerce"
        ).dt.normalize()
        calendar = calendar.dropna().drop_duplicates().sort_values("trade_date")
        return calendar[
            (calendar["trade_date"] >= pd.Timestamp(start_date))
            & (calendar["trade_date"] <= pd.Timestamp(end_date))
        ].reset_index(drop=True)

    def fetch_adjustment_factors(
        self, symbol: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        normalized_symbol = normalize_symbol(symbol)
        frame = _akshare().stock_zh_a_daily(
            symbol=_market_prefixed_symbol(normalized_symbol),
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
            adjust="qfq-factor",
        )
        if not isinstance(frame, pd.DataFrame):
            raise ProviderDataError(
                "AkShare adjustment-factor endpoint returned a non-DataFrame value"
            )
        date_column = next(
            (column for column in ("date", "日期") if column in frame.columns),
            None,
        )
        factor_column = next(
            (
                column
                for column in ("qfq_factor", "factor", "前复权因子")
                if column in frame.columns
            ),
            None,
        )
        if date_column is None or factor_column is None:
            raise ProviderDataError(
                "AkShare adjustment-factor schema changed; expected date and "
                f"qfq_factor, got {list(frame.columns)!r}"
            )
        factors = frame.rename(
            columns={date_column: "date", factor_column: "qfq_factor"}
        )[["date", "qfq_factor"]].copy()
        factors["date"] = pd.to_datetime(factors["date"], errors="coerce").dt.normalize()
        factors["qfq_factor"] = pd.to_numeric(
            factors["qfq_factor"], errors="coerce"
        )
        factors = (
            factors.dropna()
            .drop_duplicates("date", keep="last")
            .sort_values("date")
        )
        factors = factors[
            (factors["date"] >= pd.Timestamp(start_date))
            & (factors["date"] <= pd.Timestamp(end_date))
        ].reset_index(drop=True)
        factors.insert(0, "symbol", normalized_symbol)
        return factors
