from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from app.backtest.engine import BacktestConfig, run_backtest
from app.config import QFQ_REVISION_WARNING, SURVIVORSHIP_WARNING
from app.data.akshare_provider import AkShareProvider
from app.factors.evaluation import evaluate_factor
from app.factors.preprocessing import PreprocessConfig
from app.factors.registry import factor_registry
from app.json_utils import json_safe
from app.main import _attach_execution_fields
from app.storage import Storage


RESEARCH_FACTORS = (
    "momentum_20",
    "reversal_5",
    "volatility_20",
    "volume_change_20",
    "ma_bias_20",
    "price_position_60",
    "downside_volatility_20",
    "amihud_20",
)


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _read_dates(report_dir: Path) -> dict[str, date]:
    payload = json.loads(
        (report_dir / "research_dates.json").read_text(encoding="utf-8")
    )
    return {key: date.fromisoformat(value) for key, value in payload.items()}


def _load_panel(
    storage: Storage,
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for symbol in symbols:
        bars = storage.read_bars(symbol, "qfq", start_date, end_date)
        if bars.empty:
            missing.append(symbol)
        else:
            frames.append(bars)
    if missing:
        raise RuntimeError(f"股票缺少前复权缓存: {', '.join(missing)}")
    return pd.concat(frames, ignore_index=True)


def _metadata_record(factor: Any) -> dict[str, Any]:
    metadata = factor.metadata
    direction = int(getattr(metadata, "direction", 1))
    display_name = (
        getattr(metadata, "display_name_zh", None)
        or getattr(metadata, "display_name", None)
        or metadata.name
    )
    return {
        "name": metadata.name,
        "display_name_zh": display_name,
        "description": metadata.description,
        "description_zh": getattr(metadata, "description_zh", metadata.description),
        "lookback": metadata.lookback,
        "required_columns": list(metadata.required_columns),
        "direction": direction,
        "direction_label": "正向" if direction > 0 else "负向",
        "availability": metadata.availability,
    }


def _number(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            return float(value)
    return None


def _quantile_monotonicity(analysis: dict[str, Any]) -> float | None:
    rows = analysis.get("quantile_summary", [])
    if len(rows) < 2:
        return None
    frame = pd.DataFrame(rows)
    values = pd.to_numeric(frame["mean_return"], errors="coerce")
    quantiles = pd.to_numeric(frame["quantile"], errors="coerce")
    valid = values.notna() & quantiles.notna()
    if valid.sum() < 2:
        return None
    value = quantiles[valid].rank(method="average").corr(
        values[valid].rank(method="average")
    )
    return float(value) if pd.notna(value) else None


def _summary_row(
    metadata: dict[str, Any],
    split_name: str,
    split_start: date,
    split_end: date,
    analysis: dict[str, Any],
    backtest: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    factor_summary = analysis.get("summary", {})
    backtest_summary = backtest.get("summary", {})
    direction = int(metadata["direction"])
    raw_ic = _number(factor_summary, "raw_ic_mean", "ic_mean")
    raw_rank_ic = _number(factor_summary, "raw_rank_ic_mean", "rank_ic_mean")
    adjusted_ic = _number(factor_summary, "adjusted_ic_mean")
    adjusted_rank_ic = _number(factor_summary, "adjusted_rank_ic_mean")
    if adjusted_ic is None and raw_ic is not None:
        adjusted_ic = raw_ic * direction
    if adjusted_rank_ic is None and raw_rank_ic is not None:
        adjusted_rank_ic = raw_rank_ic * direction
    quantile_curve = analysis.get("quantile_net_values", [])
    long_short_return = None
    if quantile_curve:
        value = quantile_curve[-1].get("long_short")
        if isinstance(value, (int, float)) and np.isfinite(value):
            long_short_return = float(value) - 1.0
    return {
        "factor_name": metadata["name"],
        "factor_name_zh": metadata["display_name_zh"],
        "direction": metadata["direction_label"],
        "split": split_name,
        "start_date": split_start.isoformat(),
        "end_date": split_end.isoformat(),
        "raw_ic_mean": raw_ic,
        "raw_rank_ic_mean": raw_rank_ic,
        "raw_ic_ir": _number(factor_summary, "raw_ic_ir", "ic_ir"),
        "raw_win_rate": _number(factor_summary, "raw_win_rate", "win_rate"),
        "adjusted_ic_mean": adjusted_ic,
        "adjusted_rank_ic_mean": adjusted_rank_ic,
        "adjusted_ic_ir": _number(factor_summary, "adjusted_ic_ir"),
        "adjusted_win_rate": _number(factor_summary, "adjusted_win_rate"),
        "coverage": _number(factor_summary, "coverage"),
        "turnover": _number(factor_summary, "turnover"),
        "quantile_monotonicity": _quantile_monotonicity(analysis),
        "long_short_return": long_short_return,
        "backtest_total_return": _number(backtest_summary, "total_return"),
        "backtest_annualized_return": _number(
            backtest_summary, "annualized_return"
        ),
        "benchmark_return": _number(backtest_summary, "benchmark_return"),
        "excess_return": _number(backtest_summary, "excess_return"),
        "sharpe": _number(backtest_summary, "sharpe"),
        "max_drawdown": _number(backtest_summary, "max_drawdown"),
        "trade_count": backtest_summary.get("trade_count"),
        "blocked_trade_count": backtest_summary.get("blocked_trade_count"),
        "total_cost": _number(backtest_summary, "total_cost"),
        "backtest_run_id": run_id,
        "status": "success",
    }


def run_research(report_dir: Path) -> None:
    storage = Storage()
    provider = AkShareProvider()
    dates = _read_dates(report_dir)
    universe = pd.read_csv(
        report_dir / "stock_universe.csv",
        dtype={"symbol": str},
    )
    symbols = universe["symbol"].astype(str).tolist()
    if len(symbols) != 100:
        raise RuntimeError(f"股票池必须为 100 只，实际为 {len(symbols)}")

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
        "full": (
            dates["evaluation_start"],
            dates["latest_complete_date"],
        ),
        "in_sample": (
            dates["evaluation_start"],
            dates["insample_end"],
        ),
        "out_of_sample": (
            dates["oos_start"],
            dates["oos_end"],
        ),
    }

    definitions: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    detail_dir = report_dir / "factors"
    backtest_dir = report_dir / "backtests"
    detail_dir.mkdir(parents=True, exist_ok=True)
    backtest_dir.mkdir(parents=True, exist_ok=True)

    for factor_name in RESEARCH_FACTORS:
        factor = factor_registry.get(factor_name)
        metadata = _metadata_record(factor)
        definitions.append({**metadata, "tested": True})
        factor_detail: dict[str, Any] = {
            "metadata": metadata,
            "universe_size": len(symbols),
            "splits": {},
        }
        for split_name, (split_start, split_end) in splits.items():
            try:
                period_panel = panel[
                    panel["date"] <= pd.Timestamp(split_end)
                ].copy()
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
                        top_n=10,
                        rebalance="M",
                        commission_rate=0.0003,
                        minimum_commission=5.0,
                        stamp_duty_rate=0.0005,
                        historical_stamp_duty=True,
                        slippage_rate=0.0005,
                    ),
                    benchmark,
                )
                warnings = [
                    *[str(item) for item in backtest.get("warnings", [])],
                    SURVIVORSHIP_WARNING,
                    QFQ_REVISION_WARNING,
                ]
                backtest["warnings"] = list(dict.fromkeys(warnings))
                run_id = (
                    f"research-{dates['latest_complete_date'].isoformat()}-"
                    f"{factor_name}-{split_name}-{uuid4().hex[:8]}"
                )
                request = {
                    "factor_name": factor_name,
                    "symbols": symbols,
                    "start_date": split_start.isoformat(),
                    "end_date": split_end.isoformat(),
                    "top_n": 10,
                    "rebalance": "M",
                    "benchmark": "CSI300",
                    "adjust": "qfq",
                    "research_split": split_name,
                }
                backtest.update(
                    {
                        "id": run_id,
                        "factor_name_zh": metadata["display_name_zh"],
                        "research_split": split_name,
                        "benchmark": "CSI300",
                    }
                )
                storage.save_backtest(
                    run_id,
                    factor_name,
                    request,
                    backtest["summary"],
                    backtest,
                )
                split_payload = {
                    "start_date": split_start,
                    "end_date": split_end,
                    "analysis": analysis,
                    "backtest": backtest,
                }
                factor_detail["splits"][split_name] = split_payload
                _write_json(
                    backtest_dir / f"{factor_name}_{split_name}.json",
                    backtest,
                )
                summary_rows.append(
                    _summary_row(
                        metadata,
                        split_name,
                        split_start,
                        split_end,
                        analysis,
                        backtest,
                        run_id,
                    )
                )
                print(f"RESEARCH_OK {factor_name} {split_name}")
            except Exception as exc:
                failure = {
                    "factor_name": factor_name,
                    "factor_name_zh": metadata["display_name_zh"],
                    "split": split_name,
                    "error": str(exc),
                }
                failures.append(failure)
                factor_detail["splits"][split_name] = {
                    "status": "error",
                    "error": str(exc),
                }
                summary_rows.append(
                    {
                        "factor_name": factor_name,
                        "factor_name_zh": metadata["display_name_zh"],
                        "direction": metadata["direction_label"],
                        "split": split_name,
                        "start_date": split_start.isoformat(),
                        "end_date": split_end.isoformat(),
                        "status": "error",
                        "error": str(exc),
                    }
                )
                print(f"RESEARCH_FAILED {factor_name} {split_name}: {exc}")
        _write_json(detail_dir / f"{factor_name}.json", factor_detail)

    pb = factor_registry.get("pb")
    definitions.append(
        {
            **_metadata_record(pb),
            "tested": False,
            "unavailable_reason": (
                "缓存行情不含可靠的 point-in-time PB，未使用当前值回填历史。"
            ),
        }
    )
    _write_json(report_dir / "factor_definitions.json", definitions)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        report_dir / "factor_test_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_json(report_dir / "research_failures.json", failures)
    _write_json(
        report_dir / "factor_test_overview.json",
        {
            "generated_at": datetime.now().isoformat(),
            "universe_size": len(symbols),
            "dates": {key: value.isoformat() for key, value in dates.items()},
            "factors_requested": list(RESEARCH_FACTORS),
            "successful_rows": int((summary["status"] == "success").sum()),
            "failed_rows": int((summary["status"] != "success").sum()),
            "warnings": [
                SURVIVORSHIP_WARNING,
                QFQ_REVISION_WARNING,
                "Historical ST intervals and IPO no-limit windows remain unavailable.",
            ],
        },
    )
    print(
        f"RESEARCH_COMPLETE {report_dir} "
        f"success={(summary['status'] == 'success').sum()} failures={len(failures)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-date",
        default=None,
        help="Report folder YYYY-MM-DD; defaults to newest folder.",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=_backend_root() / "reports",
    )
    args = parser.parse_args()
    if args.report_date:
        report_dir = args.report_root / args.report_date
    else:
        candidates = sorted(
            path for path in args.report_root.glob("20??-??-??") if path.is_dir()
        )
        if not candidates:
            raise RuntimeError("没有股票池报告目录，请先运行 build_universe")
        report_dir = candidates[-1]
    run_research(report_dir)


if __name__ == "__main__":
    main()
