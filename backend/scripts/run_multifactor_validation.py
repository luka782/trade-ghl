from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from app.backtest.engine import BacktestConfig, run_backtest
from app.backtest.metrics import calculate_metrics
from app.data.akshare_provider import AkShareProvider
from app.factors.benchmark import merge_benchmark_bars
from app.factors.base import Factor, FactorMetadata
from app.factors.evaluation import evaluate_factor
from app.factors.preprocessing import PreprocessConfig
from app.factors.registry import factor_registry
from app.json_utils import json_safe
from app.main import (
    _attach_execution_fields,
    _multifactor_diagnostics,
)
from app.multifactor import (
    CompositeFactor,
    FactorComponentConfig,
    MultiFactorConfig,
    factor_correlation_report,
    template_config,
)
from app.storage import Storage
from app.timing import TimingConfig, run_timing
from scripts.build_universe import ResearchDates, determine_dates
from scripts.run_factor_research import _quantile_monotonicity


TEMPLATE_LABELS = {
    "trend": "趋势组合",
    "low_risk": "低风险组合",
    "price_volume": "量价组合",
    "balanced": "均衡组合",
}
TIMING_SYMBOLS = (
    ("515080", "中证红利ETF招商", True),
    ("600519", "贵州茅台", False),
    ("603986", "兆易创新", False),
)
CORRELATION_AUDIT_FACTORS = (
    "momentum_20",
    "ma_bias_20",
    "price_position_60",
    "volatility_20",
    "downside_volatility_20",
    "atr_ratio_20",
)


class PrecomputedCompositeFactor(Factor):
    def __init__(self, name: str, score_column: str) -> None:
        self.score_column = score_column
        self.metadata = FactorMetadata(
            name=name,
            description="Precomputed causal composite score for validation reuse.",
            lookback=0,
            required_columns=(score_column,),
            display_name=name,
            display_name_zh=name,
            direction=1,
        )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        return pd.to_numeric(bars[self.score_column], errors="coerce")


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


def _source_report_dir(root: Path) -> Path:
    candidates = sorted(
        path.parent
        for path in root.glob("*/stock_universe.csv")
        if (path.parent / "etfs" / "etf_universe.csv").exists()
    )
    if not candidates:
        raise RuntimeError("未找到同时包含主板100和ETF20的历史研究报告")
    return candidates[-1]


def _symbols(path: Path, expected: int) -> list[str]:
    frame = pd.read_csv(path, dtype={"symbol": str})
    values = frame["symbol"].astype(str).str.zfill(6).tolist()
    if len(values) != expected:
        raise RuntimeError(f"{path} 应包含 {expected} 个标的，实际 {len(values)}")
    return values


def _update_with_retry(
    storage: Storage,
    provider: AkShareProvider,
    symbol: str,
    start_date: date,
    end_date: date,
    adjust: str,
) -> None:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            result = storage.update_symbol(
                provider,
                symbol,
                start_date,
                end_date,
                adjust,  # type: ignore[arg-type]
            )
            if result["status"] != "no_data":
                return
            raise RuntimeError("no_data")
        except Exception as exc:  # network/provider boundary
            last_error = exc
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"{symbol} {adjust} 更新失败: {last_error}")


def _refresh_data(
    storage: Storage,
    provider: AkShareProvider,
    symbols: list[str],
    dates: ResearchDates,
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for index, symbol in enumerate(symbols, start=1):
        for adjust in ("qfq", "none"):
            try:
                _update_with_retry(
                    storage,
                    provider,
                    symbol,
                    dates.warmup_start,
                    dates.latest_complete_date,
                    adjust,
                )
            except Exception as exc:
                failures.append(
                    {"symbol": symbol, "adjust": adjust, "error": str(exc)}
                )
        print(f"VALIDATION_DATA {index}/{len(symbols)} {symbol}", flush=True)
    return failures


def _load_panel(
    storage: Storage,
    symbols: list[str],
    dates: ResearchDates,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for symbol in symbols:
        bars = storage.read_bars(
            symbol,
            "qfq",
            dates.warmup_start,
            dates.latest_complete_date,
        )
        if bars.empty:
            missing.append(symbol)
        else:
            frames.append(bars)
    if missing:
        raise RuntimeError("缺少前复权缓存: " + ", ".join(missing))
    return pd.concat(frames, ignore_index=True)


def _config(template: str, mode: str) -> MultiFactorConfig:
    base = template_config(template, mode=mode)  # type: ignore[arg-type]
    return MultiFactorConfig(
        name=TEMPLATE_LABELS[template],
        components=base.components,
        mode=base.mode,
        rolling_window=252,
        rolling_min_periods=120,
        zscore_clip=3.0,
        metadata={
            "template": template,
            "weight_source": "research_default_not_optimized",
        },
    )


def _splits(dates: ResearchDates) -> dict[str, tuple[date, date]]:
    return {
        "full": (dates.evaluation_start, dates.latest_complete_date),
        "in_sample": (dates.evaluation_start, dates.insample_end),
        "out_of_sample": (dates.oos_start, dates.oos_end),
    }


def _reweight_details(
    base: pd.DataFrame,
    config: MultiFactorConfig,
) -> pd.DataFrame:
    result = base[["symbol", "date"]].copy()
    contribution_columns: list[str] = []
    weight_columns: list[str] = []
    for component in config.components:
        if not component.enabled or component.weight == 0:
            continue
        name = component.factor_name
        raw_column = f"factor_{name}"
        normalized_column = f"normalized_{name}"
        normalized = pd.to_numeric(base[normalized_column], errors="coerce")
        metadata = factor_registry.get(name).metadata
        direction = (
            component.direction
            if component.direction is not None
            else metadata.direction
        )
        contribution_column = f"contribution_{name}"
        weight_column = f"valid_weight_{name}"
        result[raw_column] = base[raw_column]
        result[normalized_column] = normalized
        result[contribution_column] = (
            normalized * direction * component.weight
        )
        result[weight_column] = (
            normalized.notna().astype(float) * abs(component.weight)
        )
        contribution_columns.append(contribution_column)
        weight_columns.append(weight_column)
    numerator = result[contribution_columns].sum(axis=1, min_count=1)
    denominator = result[weight_columns].sum(axis=1)
    result["composite_score"] = numerator / denominator.where(
        denominator.gt(0)
    )
    result["valid_weight"] = denominator
    return result


def _selection_validation(
    output_dir: Path,
    storage: Storage,
    provider: AkShareProvider,
    symbols: list[str],
    asset_type: str,
    dates: ResearchDates,
    benchmark: pd.DataFrame,
) -> list[dict[str, Any]]:
    panel = merge_benchmark_bars(_load_panel(storage, symbols, dates), benchmark)
    panel = _attach_execution_fields(storage, panel, symbols, "qfq")
    base_details = CompositeFactor(
        _config("balanced", "cross_sectional")
    ).compute_details(panel)
    rows: list[dict[str, Any]] = []
    top_n = 5 if asset_type == "ETF" else 10
    for template in TEMPLATE_LABELS:
        config = _config(template, "cross_sectional")
        details = _reweight_details(base_details, config)
        score_column = f"validation_score_{template}"
        scored_panel = panel.copy()
        scored_panel[score_column] = details["composite_score"].to_numpy()
        factor = PrecomputedCompositeFactor(config.name, score_column)
        for split_name, (split_start, split_end) in _splits(dates).items():
            split_mask = scored_panel["date"].le(pd.Timestamp(split_end))
            split_panel = scored_panel[
                split_mask
            ].reset_index(drop=True)
            split_details = details.loc[split_mask].reset_index(drop=True)
            analysis = evaluate_factor(
                split_panel,
                factor,
                split_start,
                split_end,
                forward_period=5,
                quantiles=5,
                preprocess=PreprocessConfig(winsorize=False, zscore=False),
            )
            period = split_panel["date"].between(
                pd.Timestamp(split_start), pd.Timestamp(split_end)
            )
            diagnostics = _multifactor_diagnostics(
                split_panel.loc[period].reset_index(drop=True),
                split_details.loc[period].reset_index(drop=True),
                5,
            )
            backtest = run_backtest(
                split_panel,
                factor,
                BacktestConfig(
                    start_date=split_start,
                    end_date=split_end,
                    top_n=top_n,
                    rebalance="M",
                    commission_rate=0.0003,
                    minimum_commission=5.0,
                    minimum_trade_notional=1_000.0,
                    rebalance_tolerance=0.001,
                    stamp_duty_rate=0.0 if asset_type == "ETF" else 0.0005,
                    historical_stamp_duty=asset_type != "ETF",
                    slippage_rate=0.0005,
                ),
                benchmark[benchmark["date"].le(pd.Timestamp(split_end))],
            )
            factor_summary = analysis["summary"]
            strategy_summary = backtest["summary"]
            row = {
                "asset_type": asset_type,
                "template": template,
                "template_name_zh": config.name,
                "split": split_name,
                "start_date": split_start,
                "end_date": split_end,
                "raw_ic": factor_summary.get("raw_ic_mean"),
                "adjusted_ic": factor_summary.get("adjusted_ic_mean"),
                "raw_rank_ic": factor_summary.get("raw_rank_ic_mean"),
                "adjusted_rank_ic": factor_summary.get(
                    "adjusted_rank_ic_mean"
                ),
                "icir": factor_summary.get("raw_ic_ir"),
                "coverage": factor_summary.get("coverage"),
                "quantile_monotonicity": _quantile_monotonicity(analysis),
                "factor_turnover": factor_summary.get("turnover"),
                **{
                    key: strategy_summary.get(key)
                    for key in (
                        "total_return",
                        "annualized_return",
                        "benchmark_return",
                        "excess_return",
                        "max_drawdown",
                        "sharpe",
                        "turnover",
                        "trade_count",
                        "total_cost",
                    )
                },
                "config_id": config.config_id,
            }
            rows.append(row)
            _write_json(
                output_dir
                / asset_type.lower()
                / template
                / f"{split_name}.json",
                {
                    "summary": row,
                    "config_snapshot": config.snapshot(),
                    "correlation_report": diagnostics,
                    "factor_evaluation": analysis,
                    "backtest": backtest,
                },
            )
            print(
                f"VALIDATION_SELECTION {asset_type} {template} {split_name}",
                flush=True,
            )
    return rows


def _correlation_audit(
    output_dir: Path,
    storage: Storage,
    symbols: list[str],
    dates: ResearchDates,
) -> None:
    panel = _load_panel(storage, symbols, dates)
    config = MultiFactorConfig(
        name="重点重复性检查",
        components=tuple(
            FactorComponentConfig(name, weight=1.0)
            for name in CORRELATION_AUDIT_FACTORS
        ),
        mode="cross_sectional",
        metadata={"purpose": "correlation_audit_not_weight_optimization"},
    )
    details = CompositeFactor(config).compute_details(panel)
    reports: dict[str, Any] = {}
    for split_name, (start_date, end_date) in {
        "full": (
            dates.evaluation_start,
            dates.latest_complete_date,
        ),
        "out_of_sample": (dates.oos_start, dates.oos_end),
    }.items():
        mask = details["date"].between(
            pd.Timestamp(start_date), pd.Timestamp(end_date)
        )
        reports[split_name] = factor_correlation_report(
            details.loc[mask].reset_index(drop=True)
        )
    _write_json(
        output_dir / "priority_factor_correlation.json",
        {
            "factors": CORRELATION_AUDIT_FACTORS,
            "config_snapshot": config.snapshot(),
            "reports": reports,
            "note": (
                "绝对相关系数超过0.8仅标记为高度重复，不自动删除或调整权重。"
            ),
        },
    )


def _timing_frame(
    panel: pd.DataFrame,
    config: MultiFactorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    factor = CompositeFactor(config)
    details = factor.compute_details(panel)
    detail_columns = [
        column for column in details if column not in {"symbol", "date"}
    ]
    signal = panel.merge(
        details[["symbol", "date", *detail_columns]],
        on=["symbol", "date"],
        how="left",
        validate="one_to_one",
    )
    trend_names = {
        "momentum_20",
        "momentum_60",
        "momentum_252_21",
        "ma_bias_20",
        "price_position_60",
        "price_position_252",
        "relative_strength_60",
        "residual_momentum_60",
    }
    trend_components = [
        item
        for item in config.components
        if item.enabled
        and item.weight != 0
        and item.factor_name in trend_names
        and f"contribution_{item.factor_name}" in signal
    ]
    numerator = pd.Series(0.0, index=signal.index)
    denominator = pd.Series(0.0, index=signal.index)
    for component in config.components:
        if not component.enabled or component.weight == 0:
            continue
        metadata = factor_registry.get(component.factor_name).metadata
        signal[f"weight_{component.factor_name}"] = component.weight
        signal[f"direction_{component.factor_name}"] = (
            component.direction
            if component.direction is not None
            else metadata.direction
        )
    for component in trend_components:
        numerator += pd.to_numeric(
            signal[f"contribution_{component.factor_name}"],
            errors="coerce",
        ).fillna(0)
        denominator += pd.to_numeric(
            signal[f"valid_weight_{component.factor_name}"],
            errors="coerce",
        ).fillna(0)
    signal["trend_score"] = (
        numerator / denominator.where(denominator.gt(0))
        if trend_components
        else signal["composite_score"]
    )
    return signal, details


def _timing_validation(
    output_dir: Path,
    storage: Storage,
    symbols: dict[str, tuple[str, bool]],
    dates: ResearchDates,
    benchmark: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    config = _config("balanced", "time_series")
    for symbol, (name, is_etf) in symbols.items():
        panel = merge_benchmark_bars(
            _load_panel(storage, [symbol], dates),
            benchmark,
        )
        panel = _attach_execution_fields(storage, panel, [symbol], "qfq")
        signal, details = _timing_frame(panel, config)
        for split_name, (split_start, split_end) in _splits(dates).items():
            end_mask = signal["date"].le(pd.Timestamp(split_end))
            split_signal = signal.loc[end_mask].reset_index(drop=True)
            split_details = details.loc[end_mask].reset_index(drop=True)
            split_panel = panel.loc[end_mask].reset_index(drop=True)
            period = split_signal["date"].between(
                pd.Timestamp(split_start), pd.Timestamp(split_end)
            )
            result = run_timing(
                split_signal.loc[period].reset_index(drop=True),
                TimingConfig(
                    buy_threshold=0.7,
                    sell_threshold=0.0,
                    fixed_stop=0.08,
                    trailing_stop=0.10,
                    max_holding_sessions=60,
                    minimum_holding_sessions=2,
                    cooldown_sessions=5,
                    initial_capital=1_000_000,
                    commission_rate=0.0003,
                    minimum_commission=5,
                    slippage=0.0005,
                    minimum_trade_notional=1_000,
                    lot_size=100,
                    is_etf=is_etf,
                    max_stale_sessions=20,
                ),
            )
            benchmark_split = benchmark[
                benchmark["date"].between(
                    pd.Timestamp(split_start), pd.Timestamp(split_end)
                )
            ]
            metrics = calculate_metrics(
                pd.DataFrame(result["equity_curve"]),
                benchmark_split,
            )
            result["summary"].update(
                {
                    "benchmark_return": metrics["benchmark_return"],
                    "benchmark_annualized_return": metrics[
                        "benchmark_annualized_return"
                    ],
                    "excess_return": metrics["excess_return"],
                }
            )
            period_details = split_details.loc[period].reset_index(drop=True)
            period_panel = split_panel.loc[period].reset_index(drop=True)
            diagnostics = _multifactor_diagnostics(
                period_panel,
                period_details,
                5,
            )
            summary = result["summary"]
            row = {
                "symbol": symbol,
                "name": name,
                "asset_type": "ETF" if is_etf else "stock",
                "split": split_name,
                "start_date": split_start,
                "end_date": split_end,
                **{
                    key: summary.get(key)
                    for key in (
                        "total_return",
                        "annualized_return",
                        "benchmark_return",
                        "excess_return",
                        "max_drawdown",
                        "sharpe",
                        "win_rate",
                        "profit_loss_ratio",
                        "trade_count",
                        "average_holding_sessions",
                        "turnover",
                        "total_cost",
                        "commission",
                        "stamp_duty",
                        "slippage_cost",
                    )
                },
                "config_id": config.config_id,
            }
            rows.append(row)
            _write_json(
                output_dir / "timing" / symbol / f"{split_name}.json",
                {
                    "summary": row,
                    "config_snapshot": config.snapshot(),
                    "correlation_report": diagnostics,
                    "result": result,
                },
            )
            print(
                f"VALIDATION_TIMING {symbol} {split_name}",
                flush=True,
            )
    return rows


def run(
    skip_download: bool = False,
    correlation_only: bool = False,
) -> Path:
    backend = _backend_root()
    reports_root = backend / "reports"
    source = _source_report_dir(reports_root)
    provider = AkShareProvider()
    storage = Storage()
    dates = determine_dates(provider)
    output = (
        reports_root
        / dates.latest_complete_date.isoformat()
        / "multifactor_validation"
    )
    stock_symbols = _symbols(source / "stock_universe.csv", 100)
    etf_symbols = _symbols(source / "etfs" / "etf_universe.csv", 20)
    all_symbols = list(dict.fromkeys([*stock_symbols, *etf_symbols]))
    if not skip_download:
        failures = _refresh_data(
            storage,
            provider,
            all_symbols,
            dates,
        )
        _write_json(output / "data_failures.json", failures)
        if failures:
            raise RuntimeError(
                f"真实数据更新有 {len(failures)} 项失败，详见 data_failures.json"
            )
    benchmark = provider.fetch_index(
        "000300",
        dates.warmup_start,
        dates.latest_complete_date,
    )
    if benchmark is None or benchmark.empty:
        raise RuntimeError("沪深300基准数据不可用")
    benchmark = benchmark.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"]).dt.normalize()
    _correlation_audit(output, storage, stock_symbols, dates)
    if correlation_only:
        return output

    selection_rows = [
        *_selection_validation(
            output,
            storage,
            provider,
            stock_symbols,
            "stock",
            dates,
            benchmark,
        ),
        *_selection_validation(
            output,
            storage,
            provider,
            etf_symbols,
            "ETF",
            dates,
            benchmark,
        ),
    ]
    timing_rows = _timing_validation(
        output,
        storage,
        {
            symbol: (name, is_etf)
            for symbol, name, is_etf in TIMING_SYMBOLS
        },
        dates,
        benchmark,
    )
    pd.DataFrame(selection_rows).to_csv(
        output / "selection_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(timing_rows).to_csv(
        output / "timing_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_json(
        output / "overview.json",
        {
            "generated_at": pd.Timestamp.now(),
            "source_universe_report": str(source),
            "dates": asdict(dates),
            "warmup_trading_sessions": 420,
            "selection_rows": len(selection_rows),
            "timing_rows": len(timing_rows),
            "templates": TEMPLATE_LABELS,
            "weight_note": (
                "模板权重仅为研究默认值，未在完整样本上优化，也不称为最优权重。"
            ),
            "limitations": [
                "股票池和ETF池按当前可得名单回溯，仍有幸存者偏差。",
                "历史ST区间、IPO及重新上市无涨跌停窗口仍不完整。",
                "前复权历史可能因后续公司行动重述；成交约束使用不复权行情。",
                "日线收盘成交不包含盘口深度，涨跌停收盘价按保守不可成交处理。",
            ],
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="仅使用现有缓存；缓存不足时验证会失败",
    )
    parser.add_argument(
        "--correlation-only",
        action="store_true",
        help="只生成重点因子重复性报告",
    )
    args = parser.parse_args()
    output = run(
        skip_download=args.skip_download,
        correlation_only=args.correlation_only,
    )
    print(f"VALIDATION_COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
