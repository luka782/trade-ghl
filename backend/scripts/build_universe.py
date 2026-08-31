from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from app.data.akshare_provider import AkShareProvider
from app.data.base import normalize_symbol
from app.json_utils import json_safe
from app.storage import Storage


SH_PREFIXES = ("600", "601", "603", "605")
SZ_PREFIXES = ("000", "001", "002", "003")
EXCLUDED_NAME_PARTS = ("ST", "SST", "退", "PT")
REFERENCE_SYMBOL = "000001"
TARGET_PER_MARKET = 50
SCREEN_BATCH_SIZE = 15
DOWNLOAD_BATCH_SIZE = 10
SCREEN_WORKERS = 5
MAX_RETRIES = 3


@dataclass(frozen=True)
class ResearchDates:
    latest_complete_date: date
    evaluation_start: date
    warmup_start: date
    oos_start: date
    oos_end: date
    insample_end: date


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            json_safe(value),
            ensure_ascii=False,
            indent=2,
            default=str,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _market(symbol: str) -> str | None:
    if symbol.startswith(SH_PREFIXES):
        return "上海主板"
    if symbol.startswith(SZ_PREFIXES):
        return "深圳主板"
    return None


def _excluded_name(name: object) -> bool:
    normalized = str(name).upper().replace(" ", "")
    return any(part in normalized for part in EXCLUDED_NAME_PARTS)


def determine_dates(provider: AkShareProvider) -> ResearchDates:
    today = date.today()
    calendar = provider.fetch_trade_calendar(today - timedelta(days=45), today)
    candidates = pd.to_datetime(calendar["trade_date"]).dt.date.tolist()
    if datetime.now().time() < datetime.strptime("16:00", "%H:%M").time():
        candidates = [item for item in candidates if item < today]
    reference = provider.fetch_bars(
        REFERENCE_SYMBOL,
        today - timedelta(days=45),
        today,
        "none",
    )
    available = set(pd.to_datetime(reference["date"]).dt.date)
    complete = [item for item in candidates if item in available]
    if not complete:
        raise RuntimeError("无法从交易日历和参考股票共同确定完整交易日")
    latest = max(complete)
    evaluation_start = latest.replace(year=latest.year - 3)

    long_calendar = provider.fetch_trade_calendar(
        evaluation_start - timedelta(days=750),
        latest,
    )
    trading_dates = pd.to_datetime(long_calendar["trade_date"]).dt.date.tolist()
    before_start = [item for item in trading_dates if item < evaluation_start]
    if len(before_start) < 420:
        raise RuntimeError("交易日历不足 420 个预热交易日（最低要求为 300）")
    warmup_start = before_start[-420]

    current_month = latest.replace(day=1)
    last_complete_month_end = current_month - timedelta(days=1)
    oos_start = (last_complete_month_end.replace(day=1) - timedelta(days=1)).replace(
        day=1
    )
    for _ in range(10):
        oos_start = (oos_start - timedelta(days=1)).replace(day=1)
    # The loop above plus the initial step yields twelve complete calendar months.
    insample_end = oos_start - timedelta(days=1)
    return ResearchDates(
        latest_complete_date=latest,
        evaluation_start=evaluation_start,
        warmup_start=warmup_start,
        oos_start=oos_start,
        oos_end=last_complete_month_end,
        insample_end=insample_end,
    )


def load_mainboard_candidates(
    provider: AkShareProvider,
    report_dir: Path,
) -> pd.DataFrame:
    cached = report_dir / "screening_candidates.csv"
    if cached.exists():
        return pd.read_csv(cached, dtype={"symbol": str})

    candidates = provider.list_stocks().copy()
    candidates["symbol"] = candidates["symbol"].map(normalize_symbol)
    candidates["market"] = candidates["symbol"].map(_market)
    candidates = candidates[
        candidates["market"].notna()
        & ~candidates["name"].map(_excluded_name)
    ].copy()
    candidates = candidates.sort_values(
        ["market", "symbol"],
        ascending=[True, True],
    )
    candidates = candidates.drop_duplicates("symbol", keep="first").reset_index(
        drop=True
    )
    _write_csv(cached, candidates)
    return candidates


def _fetch_screen_row(
    provider: AkShareProvider,
    row: dict[str, Any],
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    symbol = str(row["symbol"])
    error: str | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            bars = provider.fetch_bars(symbol, start_date, end_date, "none")
            bars = bars[bars["date"] <= pd.Timestamp(end_date)].sort_values("date")
            latest = (
                pd.Timestamp(bars["date"].max()).date().isoformat()
                if not bars.empty
                else None
            )
            recent = bars.tail(60)
            amount = pd.to_numeric(recent.get("amount"), errors="coerce")
            if amount is None or amount.notna().sum() < 55:
                amount = (
                    pd.to_numeric(recent["close"], errors="coerce")
                    * pd.to_numeric(recent["volume"], errors="coerce")
                )
            mean_amount = float(amount.mean()) if amount.notna().any() else math.nan
            eligible = (
                len(bars) >= 250
                and len(recent) >= 55
                and latest == end_date.isoformat()
                and math.isfinite(mean_amount)
                and mean_amount > 0
            )
            return {
                **row,
                "history_rows": len(bars),
                "recent_rows": len(recent),
                "latest_trade_date": latest,
                "mean_amount_60d": mean_amount,
                "eligible": eligible,
                "screen_status": "ok" if eligible else "insufficient_data",
                "screen_error": None,
            }
        except Exception as exc:
            error = str(exc)
            if attempt < MAX_RETRIES:
                time.sleep(0.75 * (2 ** (attempt - 1)))
    return {
        **row,
        "history_rows": 0,
        "recent_rows": 0,
        "latest_trade_date": None,
        "mean_amount_60d": None,
        "eligible": False,
        "screen_status": "error",
        "screen_error": error,
    }


def screen_liquidity(
    provider: AkShareProvider,
    candidates: pd.DataFrame,
    dates: ResearchDates,
    report_dir: Path,
) -> pd.DataFrame:
    progress_path = report_dir / "liquidity_screen.csv"
    existing = (
        pd.read_csv(progress_path, dtype={"symbol": str})
        if progress_path.exists()
        else pd.DataFrame()
    )
    completed = set(existing["symbol"].astype(str)) if not existing.empty else set()
    pending = candidates[~candidates["symbol"].isin(completed)].to_dict(
        orient="records"
    )
    screen_start = dates.latest_complete_date - timedelta(days=430)
    rows = existing.to_dict(orient="records")

    for offset in range(0, len(pending), SCREEN_BATCH_SIZE):
        batch = pending[offset : offset + SCREEN_BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=SCREEN_WORKERS) as executor:
            futures = {
                executor.submit(
                    _fetch_screen_row,
                    provider,
                    row,
                    screen_start,
                    dates.latest_complete_date,
                ): row["symbol"]
                for row in batch
            }
            for future in as_completed(futures):
                rows.append(future.result())
        frame = pd.DataFrame(rows).sort_values(["market", "symbol"])
        _write_csv(progress_path, frame)
        print(
            f"SCREEN_PROGRESS {min(offset + len(batch), len(pending))}/"
            f"{len(pending)}"
        )
        time.sleep(0.75)
    return pd.DataFrame(rows)


def _update_with_retry(
    storage: Storage,
    provider: AkShareProvider,
    symbol: str,
    start_date: date,
    end_date: date,
    adjust: str,
) -> dict[str, Any]:
    last_error: str | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = storage.update_symbol(
                provider,
                symbol,
                start_date,
                end_date,
                adjust,  # type: ignore[arg-type]
            )
            if result["status"] != "no_data":
                return result
            last_error = "no_data"
        except Exception as exc:
            last_error = str(exc)
        if attempt < MAX_RETRIES:
            time.sleep(1.0 * (2 ** (attempt - 1)))
    raise RuntimeError(last_error or "unknown download error")


def download_ranked_universe(
    provider: AkShareProvider,
    storage: Storage,
    screened: pd.DataFrame,
    dates: ResearchDates,
    report_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    eligible = screened[screened["eligible"].astype(str).str.lower() == "true"].copy()
    eligible["mean_amount_60d"] = pd.to_numeric(
        eligible["mean_amount_60d"], errors="coerce"
    )
    eligible = eligible.sort_values(
        ["market", "mean_amount_60d", "symbol"],
        ascending=[True, False, True],
    )
    progress_path = report_dir / "download_progress.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.exists()
        else {}
    )
    if progress.get("selection_scope") != "all_current_mainboard_v1":
        progress = {
            "selection_scope": "all_current_mainboard_v1",
            "successes": [],
            "failures": [],
        }
    success_by_market: dict[str, list[dict[str, Any]]] = {
        "上海主板": [],
        "深圳主板": [],
    }
    for item in progress["successes"]:
        success_by_market[item["market"]].append(item)
    attempted = {
        (item["market"], item["symbol"])
        for item in [*progress["successes"], *progress["failures"]]
    }

    for market in ("上海主板", "深圳主板"):
        market_rows = eligible[eligible["market"] == market].to_dict(orient="records")
        pending = [
            row for row in market_rows if (market, str(row["symbol"])) not in attempted
        ]
        for offset in range(0, len(pending), DOWNLOAD_BATCH_SIZE):
            if len(success_by_market[market]) >= TARGET_PER_MARKET:
                break
            batch = pending[offset : offset + DOWNLOAD_BATCH_SIZE]
            for row in batch:
                if len(success_by_market[market]) >= TARGET_PER_MARKET:
                    break
                symbol = str(row["symbol"])
                try:
                    qfq = _update_with_retry(
                        storage,
                        provider,
                        symbol,
                        dates.warmup_start,
                        dates.latest_complete_date,
                        "qfq",
                    )
                    raw = _update_with_retry(
                        storage,
                        provider,
                        symbol,
                        dates.warmup_start,
                        dates.latest_complete_date,
                        "none",
                    )
                    cached = storage.read_bars(
                        symbol,
                        "qfq",
                        dates.warmup_start,
                        dates.latest_complete_date,
                    )
                    item = {
                        "symbol": symbol,
                        "name": row["name"],
                        "market": market,
                        "latest_trade_date": (
                            pd.Timestamp(cached["date"].max()).date().isoformat()
                        ),
                        "mean_amount_60d": float(row["mean_amount_60d"]),
                        "data_start_date": (
                            pd.Timestamp(cached["date"].min()).date().isoformat()
                        ),
                        "data_end_date": (
                            pd.Timestamp(cached["date"].max()).date().isoformat()
                        ),
                        "data_rows": len(cached),
                        "download_status": "success",
                        "qfq_status": qfq["status"],
                        "raw_status": raw["status"],
                    }
                    progress["successes"].append(item)
                    success_by_market[market].append(item)
                except Exception as exc:
                    progress["failures"].append(
                        {
                            "symbol": symbol,
                            "name": row["name"],
                            "market": market,
                            "stage": "full_download",
                            "error": str(exc),
                        }
                    )
                _write_json(progress_path, progress)
            print(
                f"DOWNLOAD_PROGRESS {market} "
                f"{len(success_by_market[market])}/{TARGET_PER_MARKET}"
            )
            time.sleep(1.0)

    if any(
        len(success_by_market[market]) < TARGET_PER_MARKET
        for market in success_by_market
    ):
        raise RuntimeError(
            "成功股票不足："
            + ", ".join(
                f"{market}={len(rows)}" for market, rows in success_by_market.items()
            )
        )
    final_rows = [
        *success_by_market["上海主板"][:TARGET_PER_MARKET],
        *success_by_market["深圳主板"][:TARGET_PER_MARKET],
    ]
    final = pd.DataFrame(final_rows).sort_values(["market", "mean_amount_60d"], ascending=[True, False])
    _write_csv(report_dir / "stock_universe.csv", final)
    failures = [
        *progress["failures"],
        *screened.loc[
            screened["screen_status"] != "ok",
            ["symbol", "name", "market", "screen_status", "screen_error"],
        ].to_dict(orient="records"),
    ]
    _write_json(report_dir / "data_failures.json", failures)
    return final, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-root",
        type=Path,
        default=_backend_root() / "reports",
    )
    args = parser.parse_args()

    provider = AkShareProvider()
    storage = Storage()
    dates = determine_dates(provider)
    report_dir = args.report_root / dates.latest_complete_date.isoformat()
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_json(report_dir / "research_dates.json", asdict(dates))

    candidates = load_mainboard_candidates(provider, report_dir)
    screened = screen_liquidity(provider, candidates, dates, report_dir)
    final, failures = download_ranked_universe(
        provider,
        storage,
        screened,
        dates,
        report_dir,
    )
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "selection_method": (
            "Candidates are the complete current Shanghai and Shenzhen main-board "
            "A-share code/name list after ST/delisting name filters. Each market is "
            "ranked by mean completed-day amount over the latest 60 sessions."
        ),
        "shortlist_note": (
            "No incomplete 2026-08-24 quote is used. The current security list still "
            "introduces survivorship bias when applied to earlier sample dates."
        ),
        "dates": asdict(dates),
        "successful_symbols": len(final),
        "failed_records": len(failures),
    }
    _write_json(report_dir / "universe_metadata.json", metadata)
    print(f"UNIVERSE_COMPLETE {report_dir} {len(final)}")


if __name__ == "__main__":
    main()
