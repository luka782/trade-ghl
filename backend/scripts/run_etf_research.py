from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from app.backtest.engine import BacktestConfig, run_backtest
from app.config import QFQ_REVISION_WARNING
from app.data.akshare_provider import AkShareProvider
from app.factors.evaluation import evaluate_factor
from app.factors.preprocessing import PreprocessConfig
from app.factors.registry import factor_registry
from app.main import _attach_execution_fields
from app.storage import Storage
from scripts.run_factor_research import (
    RESEARCH_FACTORS,
    _load_panel,
    _metadata_record,
    _read_dates,
    _summary_row,
    _write_json,
)


ETF_UNIVERSE: tuple[tuple[str, str, str], ...] = (
    ("515080", "中证红利ETF招商", "红利"),
    ("512480", "半导体ETF国联安", "半导体"),
    ("512980", "传媒ETF广发", "传媒"),
    ("159755", "电池ETF广发", "新能源电池"),
    ("588200", "科创芯片ETF嘉实", "科创芯片"),
    ("510050", "上证50ETF华夏", "宽基"),
    ("510300", "沪深300ETF华泰柏瑞", "宽基"),
    ("510500", "中证500ETF南方", "宽基"),
    ("512100", "中证1000ETF南方", "宽基"),
    ("159915", "创业板ETF易方达", "成长宽基"),
    ("588000", "科创50ETF华夏", "成长宽基"),
    ("512880", "证券ETF国泰", "证券"),
    ("512800", "银行ETF华宝", "银行"),
    ("512690", "酒ETF鹏华", "消费"),
    ("159928", "消费ETF汇添富", "消费"),
    ("512170", "医疗ETF华宝", "医疗"),
    ("512660", "军工ETF国泰", "军工"),
    ("512400", "有色金属ETF南方", "有色金属"),
    ("515030", "新能源车ETF华夏", "新能源车"),
    ("516160", "新能源ETF南方", "新能源"),
)


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _update_with_retry(
    storage: Storage,
    provider: AkShareProvider,
    symbol: str,
    start_date: Any,
    end_date: Any,
    adjust: str,
) -> dict[str, Any]:
    last_error: str | None = None
    for attempt in range(1, 4):
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
        if attempt < 3:
            time.sleep(float(2 ** (attempt - 1)))
    raise RuntimeError(last_error or "unknown ETF download error")


def run(report_dir: Path) -> None:
    output_dir = report_dir / "etfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = _read_dates(report_dir)
    storage = Storage()
    provider = AkShareProvider()
    universe_rows: list[dict[str, Any]] = []
    data_failures: list[dict[str, Any]] = []

    for symbol, name, category in ETF_UNIVERSE:
        try:
            qfq = _update_with_retry(
                storage,
                provider,
                symbol,
                dates["warmup_start"],
                dates["latest_complete_date"],
                "qfq",
            )
            raw = _update_with_retry(
                storage,
                provider,
                symbol,
                dates["warmup_start"],
                dates["latest_complete_date"],
                "none",
            )
            bars = storage.read_bars(
                symbol,
                "qfq",
                dates["warmup_start"],
                dates["latest_complete_date"],
            )
            universe_rows.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "asset_type": "ETF",
                    "category": category,
                    "market": "上海ETF" if symbol.startswith("5") else "深圳ETF",
                    "data_start_date": pd.Timestamp(bars["date"].min())
                    .date()
                    .isoformat(),
                    "data_end_date": pd.Timestamp(bars["date"].max())
                    .date()
                    .isoformat(),
                    "data_rows": len(bars),
                    "download_status": "success",
                    "qfq_status": qfq["status"],
                    "raw_status": raw["status"],
                }
            )
            print(f"ETF_DOWNLOAD_OK {symbol} {name}")
        except Exception as exc:
            failure = {
                "symbol": symbol,
                "name": name,
                "stage": "download",
                "error": str(exc),
            }
            data_failures.append(failure)
            print(f"ETF_DOWNLOAD_FAILED {symbol}: {exc}")
        time.sleep(0.5)

    universe = pd.DataFrame(universe_rows)
    universe.to_csv(
        output_dir / "etf_universe.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_json(output_dir / "data_failures.json", data_failures)
    if len(universe) != len(ETF_UNIVERSE):
        raise RuntimeError(
            f"ETF data incomplete: {len(universe)}/{len(ETF_UNIVERSE)} succeeded"
        )

    symbols = universe["symbol"].astype(str).tolist()
    panel = _load_panel(
        storage,
        symbols,
        dates["warmup_start"],
        dates["latest_complete_date"],
    )
    execution_panel = _attach_execution_fields(
        storage,
        panel,
        symbols,
        "qfq",
    )
    benchmark = provider.fetch_index(
        "000300",
        dates["evaluation_start"],
        dates["latest_complete_date"],
    )
    splits = {
        "full": (dates["evaluation_start"], dates["latest_complete_date"]),
        "in_sample": (dates["evaluation_start"], dates["insample_end"]),
        "out_of_sample": (dates["oos_start"], dates["oos_end"]),
    }

    definitions: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    research_failures: list[dict[str, Any]] = []
    factor_dir = output_dir / "factors"
    backtest_dir = output_dir / "backtests"
    factor_dir.mkdir(parents=True, exist_ok=True)
    backtest_dir.mkdir(parents=True, exist_ok=True)

    for factor_name in RESEARCH_FACTORS:
        factor = factor_registry.get(factor_name)
        metadata = _metadata_record(factor)
        definitions.append({**metadata, "tested": True})
        details: dict[str, Any] = {
            "metadata": metadata,
            "universe_size": len(symbols),
            "asset_type": "ETF",
            "splits": {},
        }
        for split_name, (split_start, split_end) in splits.items():
            try:
                period_panel = panel[panel["date"] <= pd.Timestamp(split_end)].copy()
                period_execution = execution_panel[
                    execution_panel["date"] <= pd.Timestamp(split_end)
                ].copy()
                analysis = evaluate_factor(
                    period_panel,
                    factor,
                    split_start,
                    split_end,
                    forward_period=5,
                    quantiles=5,
                    preprocess=PreprocessConfig(),
                )
                backtest = run_backtest(
                    period_execution,
                    factor,
                    BacktestConfig(
                        start_date=split_start,
                        end_date=split_end,
                        top_n=5,
                        rebalance="M",
                        commission_rate=0.0003,
                        minimum_commission=5.0,
                        stamp_duty_rate=0.0,
                        historical_stamp_duty=False,
                        slippage_rate=0.0005,
                    ),
                    benchmark,
                )
                run_id = (
                    f"etf-research-{dates['latest_complete_date'].isoformat()}-"
                    f"{factor_name}-{split_name}-{uuid4().hex[:8]}"
                )
                warnings = [
                    *[str(item) for item in backtest.get("warnings", [])],
                    QFQ_REVISION_WARNING,
                    (
                        "ETF research uses the current selected fund list across "
                        "history and may contain fund-survival bias."
                    ),
                ]
                backtest.update(
                    {
                        "id": run_id,
                        "factor_name_zh": metadata["display_name_zh"],
                        "research_split": split_name,
                        "asset_type": "ETF",
                        "warnings": list(dict.fromkeys(warnings)),
                    }
                )
                request = {
                    "factor_name": factor_name,
                    "symbols": symbols,
                    "start_date": split_start.isoformat(),
                    "end_date": split_end.isoformat(),
                    "top_n": 5,
                    "rebalance": "M",
                    "benchmark": "CSI300",
                    "adjust": "qfq",
                    "research_split": split_name,
                    "asset_type": "ETF",
                }
                storage.save_backtest(
                    run_id,
                    factor_name,
                    request,
                    backtest["summary"],
                    backtest,
                )
                details["splits"][split_name] = {
                    "analysis": analysis,
                    "backtest": backtest,
                }
                _write_json(
                    backtest_dir / f"{factor_name}_{split_name}.json",
                    backtest,
                )
                summary_rows.append(
                    {
                        **_summary_row(
                            metadata,
                            split_name,
                            split_start,
                            split_end,
                            analysis,
                            backtest,
                            run_id,
                        ),
                        "asset_type": "ETF",
                        "portfolio_size": 5,
                        "stamp_duty": 0.0,
                    }
                )
                print(f"ETF_RESEARCH_OK {factor_name} {split_name}")
            except Exception as exc:
                failure = {
                    "factor_name": factor_name,
                    "factor_name_zh": metadata["display_name_zh"],
                    "split": split_name,
                    "error": str(exc),
                }
                research_failures.append(failure)
                details["splits"][split_name] = {
                    "status": "error",
                    "error": str(exc),
                }
                print(f"ETF_RESEARCH_FAILED {factor_name} {split_name}: {exc}")
        _write_json(factor_dir / f"{factor_name}.json", details)

    _write_json(output_dir / "factor_definitions.json", definitions)
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "factor_test_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_json(output_dir / "research_failures.json", research_failures)
    _write_json(
        output_dir / "overview.json",
        {
            "generated_at": datetime.now().isoformat(),
            "asset_type": "ETF",
            "universe_size": len(symbols),
            "successful_research_rows": len(summary_rows),
            "failed_research_rows": len(research_failures),
            "dates": {key: value.isoformat() for key, value in dates.items()},
            "portfolio": "direction-adjusted Top 5, monthly rebalance",
            "fees": "commission + minimum commission + slippage; ETF stamp duty 0",
            "warnings": [
                "Current ETF list is applied historically and may contain survival bias.",
                QFQ_REVISION_WARNING,
                "Only domestic equity ETFs were selected to retain the T+1 model.",
            ],
        },
    )
    print(
        f"ETF_PIPELINE_COMPLETE data={len(universe_rows)} "
        f"research={len(summary_rows)} failures="
        f"{len(data_failures) + len(research_failures)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-date", default="2026-08-21")
    parser.add_argument(
        "--report-root",
        type=Path,
        default=_backend_root() / "reports",
    )
    args = parser.parse_args()
    run(args.report_root / args.report_date)


if __name__ == "__main__":
    main()
