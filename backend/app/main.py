from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .backtest.engine import BacktestConfig, run_backtest
from .backtest.metrics import calculate_metrics
from .config import QFQ_REVISION_WARNING, SURVIVORSHIP_WARNING
from .data.akshare_provider import AkShareProvider
from .data.base import DataProvider, ProviderDataError, normalize_symbol
from .factors.base import Factor, FactorUnavailableError
from .factors.benchmark import merge_benchmark_bars
from .factors.evaluation import evaluate_factor
from .factors.preprocessing import PreprocessConfig
from .factors.registry import factor_registry
from .json_utils import json_safe
from .multifactor import (
    MULTIFACTOR_TEMPLATES,
    CompositeFactor,
    FactorComponentConfig,
    MultiFactorConfig,
    factor_correlation_report,
)
from .schemas import (
    BacktestRequest,
    DownloadRequest,
    FactorAnalyzeRequest,
    MultiFactorAnalyzeRequest,
    MultiFactorBacktestRequest,
    MultiFactorConfigRequest,
    TimingBacktestRequest,
    TimingWalkForwardRequest,
)
from .storage import Storage
from .timing import TimingConfig, run_timing
from .timing.indicators import (
    average_true_range,
    bollinger_bands,
    distance_to_moving_average,
    donchian_channels,
    moving_average,
    moving_average_slope,
    wilder_rsi,
)
from .validation.diagnostics import (
    cscv_pbo,
    deflated_sharpe_ratio,
    sharpe_ratio,
    walk_forward_efficiency,
)
from .validation.protocol import (
    ValidationProtocol,
    common_recent_evaluation_period,
)
from .validation.search import (
    generate_preregistered_candidates,
    parameter_perturbations,
    perturbation_stability,
    robust_multi_symbol_objective,
)
from .validation.walk_forward import generate_rolling_folds


# API 层只负责编排“加载数据 → 计算信号/指标 → 调用回测或验证 → 持久化结果”。
# 市场规则留在 backtest/timing，因子公式留在 factors，避免 HTTP 端点演变为
# 无法单元测试的业务巨型函数。
BENCHMARK_SYMBOLS = {"CSI300": "000300", "CSI500": "000905"}
FACTOR_ALIASES = {
    "momentum_20d": "momentum_20",
    "volatility_20d": "volatility_20",
    "volume_change_20d": "volume_change_20",
}
_WALK_FORWARD_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def _local_instrument_name_map() -> dict[str, str]:
    """从已生成的研究报告补全证券中文名称，缺失报告时返回空映射。"""
    report_root = Path(__file__).resolve().parents[1] / "reports"
    patterns = (
        "20??-??-??/screening_candidates.csv",
        "20??-??-??/stock_universe.csv",
        "20??-??-??/etfs/etf_universe.csv",
    )
    names: dict[str, str] = {}
    for pattern in patterns:
        for path in sorted(report_root.glob(pattern)):
            try:
                frame = pd.read_csv(path, dtype={"symbol": str})
            except Exception:
                continue
            if not {"symbol", "name"}.issubset(frame.columns):
                continue
            for symbol, name in frame.loc[:, ["symbol", "name"]].itertuples(
                index=False,
                name=None,
            ):
                if pd.notna(symbol) and pd.notna(name):
                    names[normalize_symbol(symbol)] = str(name)
    return names


def _resolve_factor(name: str) -> Factor:
    """处理历史别名后从注册表取得因子；未知因子转换为明确的 HTTP 404。"""
    canonical_name = FACTOR_ALIASES.get(name.strip().lower(), name.strip())
    try:
        return factor_registry.get(canonical_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _preprocess_config(value: Any) -> PreprocessConfig:
    """兼容旧版字符串预处理选项和新版结构化配置。"""
    if isinstance(value, str):
        modes = {
            "none": {"winsorize": False, "zscore": False},
            "winsorize": {"winsorize": True, "zscore": False},
            "zscore": {"winsorize": False, "zscore": True},
            "winsorize_zscore": {"winsorize": True, "zscore": True},
        }
        return PreprocessConfig(**modes[value])
    return PreprocessConfig(**value.model_dump())


def _to_multifactor_config(
    request: MultiFactorConfigRequest,
    *,
    mode: str | None = None,
) -> MultiFactorConfig:
    """将 API 请求模型转换为核心领域模型，避免 Web 层类型渗入计算模块。"""
    return MultiFactorConfig(
        name=request.name,
        mode=(mode or request.mode),  # type: ignore[arg-type]
        rolling_window=request.rolling_window,
        rolling_min_periods=request.rolling_min_periods,
        zscore_clip=request.zscore_clip,
        metadata=dict(request.metadata),
        components=tuple(
            FactorComponentConfig(**item.model_dump())
            for item in request.components
        ),
    )


def _prefix_timing_details(
    details: pd.DataFrame,
    config: MultiFactorConfig,
    prefix: str,
) -> pd.DataFrame:
    """给买入/卖出组合因子明细加前缀，允许同帧审计两套独立评分。"""
    renamed = details.loc[:, ["symbol", "date"]].copy()
    for component in config.components:
        if not component.enabled or component.weight == 0:
            continue
        name = component.factor_name
        for source_prefix in (
            "factor_",
            "normalized_",
            "contribution_",
            "valid_weight_",
        ):
            source = f"{source_prefix}{name}"
            if source in details:
                renamed[
                    f"{source_prefix}{prefix}__{name}"
                ] = details[source].to_numpy()
        metadata = factor_registry.get(name).metadata
        renamed[f"weight_{prefix}__{name}"] = component.weight
        renamed[f"direction_{prefix}__{name}"] = (
            component.direction
            if component.direction is not None
            else metadata.direction
        )
    renamed[f"{prefix}_score"] = details["composite_score"].to_numpy()
    renamed[f"{prefix}_valid_weight"] = details["valid_weight"].to_numpy()
    return renamed


def _attach_regime_columns(
    signal_frame: pd.DataFrame, options: Any
) -> pd.DataFrame:
    frame = signal_frame.copy()
    frame["ma_200"] = moving_average(frame, options.ma_period)
    frame["ma_slope_20"] = moving_average_slope(
        frame, options.ma_period, slope_periods=options.ma_slope_period
    )
    frame["distance_to_ma_200"] = distance_to_moving_average(
        frame, options.ma_period
    )
    frame["rsi_14"] = wilder_rsi(frame, options.rsi_period)
    bands = bollinger_bands(
        frame,
        options.bollinger_window,
        standard_deviations=options.bollinger_std,
    )
    for source, target in (
        ("mid", "bollinger_mid_20"),
        ("upper", "bollinger_upper_20"),
        ("lower", "bollinger_lower_20"),
        ("percent_b", "bollinger_percent_b_20"),
        ("bandwidth", "bollinger_bandwidth_20"),
    ):
        frame[target] = bands[source]
    close = pd.to_numeric(frame["close"], errors="coerce")
    ma_value = pd.to_numeric(frame["ma_200"], errors="coerce")
    ma_slope = pd.to_numeric(frame["ma_slope_20"], errors="coerce")
    frame["market_regime"] = np.select(
        [
            close.gt(ma_value) & ma_slope.gt(0),
            close.lt(ma_value) & ma_slope.lt(0),
        ],
        ["uptrend", "downtrend"],
        default="sideways",
    )
    rsi_value = pd.to_numeric(frame["rsi_14"], errors="coerce")
    percent_b = pd.to_numeric(
        frame["bollinger_percent_b_20"], errors="coerce"
    )
    entry_rsi = ((options.rsi_oversold + 10 - rsi_value) / 10).clip(-3, 3)
    entry_bb = ((0.2 - percent_b) / 0.2).clip(-3, 3)
    entry_regime = frame["market_regime"].map(
        {"uptrend": 1.0, "sideways": 0.25, "downtrend": -1.0}
    )
    exit_rsi = ((rsi_value - (options.rsi_overbought - 10)) / 10).clip(
        -3, 3
    )
    exit_bb = ((percent_b - 0.8) / 0.2).clip(-3, 3)
    exit_regime = frame["market_regime"].map(
        {"uptrend": -0.5, "sideways": 0.2, "downtrend": 1.0}
    )
    entry_den = (
        options.entry_factor_weight
        + options.entry_rsi_weight
        + options.entry_bollinger_weight
        + options.entry_regime_weight
    )
    exit_den = (
        options.exit_factor_weight
        + options.exit_rsi_weight
        + options.exit_bollinger_weight
        + options.exit_regime_weight
    )
    frame["entry_score_final"] = (
        options.entry_factor_weight * frame["entry_score"]
        + options.entry_rsi_weight * entry_rsi
        + options.entry_bollinger_weight * entry_bb
        + options.entry_regime_weight * entry_regime
    ) / entry_den
    frame["exit_score_final"] = (
        options.exit_factor_weight * frame["exit_score"]
        + options.exit_rsi_weight * exit_rsi
        + options.exit_bollinger_weight * exit_bb
        + options.exit_regime_weight * exit_regime
    ) / exit_den
    frame["composite_score"] = (
        frame["entry_score_final"] - frame["exit_score_final"]
    ) / 2
    frame["trend_score"] = entry_regime
    for name, values in (
        ("regime_entry_factor", frame["entry_score"]),
        ("regime_entry_rsi", entry_rsi),
        ("regime_entry_bollinger", entry_bb),
        ("regime_entry_market", entry_regime),
        ("regime_exit_factor", frame["exit_score"]),
        ("regime_exit_rsi", exit_rsi),
        ("regime_exit_bollinger", exit_bb),
        ("regime_exit_market", exit_regime),
    ):
        frame[f"contribution_{name}"] = values
    return frame


def _attach_timing_indicator_columns(
    signal_frame: pd.DataFrame, options: Any
) -> pd.DataFrame:
    """按策略和仓位需求集中准备因果指标，供API与Walk-Forward共用。"""
    frame = signal_frame.copy()
    frame["atr_20"] = average_true_range(frame, options.atr_period)
    if options.timing_style in {
        "regime_reversion",
        "regime_reversion_legacy",
        "rsi_bollinger",
    }:
        frame = _attach_regime_columns(frame, options)
    if options.timing_style == "donchian_atr":
        channels = donchian_channels(
            frame,
            options.donchian_entry_window,
            options.donchian_exit_window,
        )
        frame["donchian_upper"] = channels["upper"]
        frame["donchian_lower"] = channels["lower"]
        if options.donchian_trend_filter:
            frame["ma_200"] = moving_average(
                frame, options.ma_period
            )
            frame["ma_slope_20"] = moving_average_slope(
                frame,
                options.ma_period,
                slope_periods=options.ma_slope_period,
            )
    elif options.timing_style == "ma_crossover_atr":
        frame["ma_fast"] = moving_average(
            frame, options.ma_fast_period
        )
        frame["ma_slow"] = moving_average(
            frame, options.ma_slow_period
        )
        frame["ma_slow_slope"] = moving_average_slope(
            frame,
            options.ma_slow_period,
            slope_periods=options.ma_slope_period,
        )
    elif options.timing_style == "ma_200":
        frame["ma_200"] = moving_average(
            frame, options.ma_period
        )
    return frame


def _timing_config_from_options(options: Any, is_etf: bool) -> TimingConfig:
    names = TimingConfig.__dataclass_fields__.keys()
    payload = {
        name: (
            options.slippage_rate
            if name == "slippage"
            else is_etf
            if name == "is_etf"
            else getattr(options, name)
        )
        for name in names
        if name == "is_etf"
        or name == "slippage"
        or hasattr(options, name)
    }
    return TimingConfig(**payload)


def _is_etf(symbol: str) -> bool:
    return str(symbol).startswith(("15", "51", "56", "58"))


def _candidate_options(base: Any, parameters: dict[str, Any]) -> Any:
    updates = dict(parameters)
    preset = int(updates.pop("weight_preset", 0))
    presets = (
        {},
        {
            "entry_factor_weight": 0.55,
            "entry_rsi_weight": 0.20,
            "entry_bollinger_weight": 0.15,
            "entry_regime_weight": 0.10,
            "exit_factor_weight": 0.55,
            "exit_rsi_weight": 0.15,
            "exit_bollinger_weight": 0.15,
            "exit_regime_weight": 0.15,
        },
        {
            "entry_factor_weight": 0.25,
            "entry_rsi_weight": 0.30,
            "entry_bollinger_weight": 0.30,
            "entry_regime_weight": 0.15,
            "exit_factor_weight": 0.25,
            "exit_rsi_weight": 0.25,
            "exit_bollinger_weight": 0.25,
            "exit_regime_weight": 0.25,
        },
    )
    updates.update(presets[preset])
    updates["timing_style"] = "regime_reversion"
    return base.model_copy(update=updates)


def _build_walk_forward_base_frames(
    storage: Storage,
    provider: DataProvider,
    body: TimingWalkForwardRequest,
    evaluation_start: date,
    evaluation_end: date,
) -> dict[str, pd.DataFrame]:
    entry_config = _to_multifactor_config(
        body.entry_config, mode="time_series"
    )
    exit_config = _to_multifactor_config(
        body.exit_config, mode="time_series"
    )
    entry_factor = CompositeFactor(entry_config)
    exit_factor = CompositeFactor(exit_config)
    lookback = max(
        entry_factor.metadata.lookback,
        exit_factor.metadata.lookback,
        300,
    )
    frames: dict[str, pd.DataFrame] = {}
    for symbol in body.symbols:
        panel, missing = _load_panel(
            storage,
            [symbol],
            body.adjust,
            evaluation_start,
            evaluation_end,
            lookback,
        )
        if missing:
            raise ValueError(f"Missing cached bars for {symbol}")
        panel, _, warning = _attach_benchmark_columns(
            panel, provider, body.benchmark
        )
        if warning:
            raise ValueError(warning)
        panel = _attach_execution_fields(
            storage, panel, [symbol], body.adjust
        )
        entry = _prefix_timing_details(
            entry_factor.compute_details(panel), entry_config, "entry"
        )
        exit_values = _prefix_timing_details(
            exit_factor.compute_details(panel), exit_config, "exit"
        )
        frame = panel.merge(
            entry, on=["symbol", "date"], validate="one_to_one"
        ).merge(
            exit_values, on=["symbol", "date"], validate="one_to_one"
        )
        frame["composite_score"] = (
            frame["entry_score"] - frame["exit_score"]
        ) / 2
        frame["trend_score"] = frame["entry_score"]
        rolling_high = frame["high"].rolling(60, min_periods=60).max()
        rolling_low = frame["low"].rolling(60, min_periods=60).min()
        frame["timing_price_position_60"] = (
            frame["close"] - rolling_low
        ) / (rolling_high - rolling_low).where(rolling_high.ne(rolling_low))
        frames[symbol] = frame
    return frames


def _evaluate_regime_segment(
    frames: dict[str, pd.DataFrame],
    dates: tuple[pd.Timestamp, ...] | pd.DatetimeIndex,
    options: Any,
) -> tuple[dict[str, Any], pd.Series]:
    symbol_metrics: dict[str, dict[str, Any]] = {}
    returns: list[pd.Series] = []
    allowed = pd.DatetimeIndex(dates)
    for symbol, base in frames.items():
        signal = _attach_timing_indicator_columns(base, options)
        segment = signal[
            pd.to_datetime(signal["date"]).dt.normalize().isin(allowed)
        ].reset_index(drop=True)
        if segment.empty:
            continue
        result = run_timing(
            segment,
            _timing_config_from_options(options, _is_etf(symbol)),
        )
        summary = dict(result["summary"])
        evidence_warning = None
        sharpe_value = summary.get("sharpe")
        if sharpe_value is None or not np.isfinite(float(sharpe_value)):
            summary["sharpe"] = -2.0
            evidence_warning = "insufficient closed trades"
        for key, fallback in (
            ("annualized_return", 0.0),
            ("max_drawdown", 0.0),
            ("turnover", 0.0),
        ):
            value = summary.get(key)
            if value is None or not np.isfinite(float(value)):
                summary[key] = fallback
        if int(summary.get("round_trip_count") or 0) < 2:
            evidence_warning = "fewer than two closed trades"
        summary["evidence_warning"] = evidence_warning
        annual = summary.get("annualized_return")
        drawdown = summary.get("max_drawdown")
        summary["calmar"] = (
            float(annual) / abs(float(drawdown))
            if annual is not None
            and drawdown not in (None, 0)
            and np.isfinite(float(annual))
            else float("nan")
        )
        summary["market_exposure"] = float(
            np.mean(
                [
                    float(row.get("position_value", 0)) > 0
                    for row in result["equity_curve"]
                ]
            )
        )
        symbol_metrics[symbol] = summary
        curve = pd.DataFrame(result["equity_curve"])
        series = pd.Series(
            pd.to_numeric(curve["daily_return"], errors="coerce").to_numpy(),
            index=pd.to_datetime(curve["date"]),
            name=symbol,
        )
        returns.append(series)
    objective = robust_multi_symbol_objective(
        symbol_metrics, required_symbols=frames.keys()
    )
    if objective.get("available"):
        sparse = sum(
            int(item.get("round_trip_count") or 0) < 2
            for item in symbol_metrics.values()
        )
        objective["sparse_symbol_count"] = sparse
        objective["objective"] = float(objective["objective"]) - 0.25 * sparse
    objective["symbol_metrics"] = symbol_metrics
    combined = (
        pd.concat(returns, axis=1).mean(axis=1).fillna(0.0)
        if returns
        else pd.Series(dtype=float)
    )
    return objective, combined


def _comparison_view(metrics: dict[str, Any]) -> dict[str, Any]:
    symbols = list((metrics.get("symbol_metrics") or {}).values())
    if not symbols:
        return metrics
    def median(name: str) -> float | None:
        values = [
            float(item[name])
            for item in symbols
            if item.get(name) is not None
            and np.isfinite(float(item[name]))
        ]
        return float(np.median(values)) if values else None
    return {
        **metrics,
        "total_return": median("total_return"),
        "annualized_return": median("annualized_return"),
        "sharpe": median("sharpe"),
        "calmar": median("calmar"),
        "max_drawdown": median("max_drawdown"),
        "win_rate": median("win_rate"),
        "profit_factor": median("profit_loss_ratio"),
        "market_exposure": median("market_exposure"),
        "turnover": median("turnover"),
        "trade_count": sum(
            int(item.get("trade_count") or 0) for item in symbols
        ),
        "closed_trades": sum(
            int(item.get("round_trip_count") or 0) for item in symbols
        ),
        "total_cost": sum(
            float(item.get("total_cost") or 0) for item in symbols
        ),
        "evidence_sufficient": all(
            int(item.get("round_trip_count") or 0) >= 5
            and float(item.get("market_exposure") or 0.0) >= 0.10
            for item in symbols
        ),
    }


def _run_walk_forward_research(
    storage: Storage,
    provider: DataProvider,
    body: TimingWalkForwardRequest,
    job_id: str,
) -> None:
    try:
        storage.update_walk_forward_job(
            job_id, status="running", progress=0.01
        )
        symbol_dates: dict[str, list[pd.Timestamp]] = {}
        for symbol in body.symbols:
            bars = storage.read_bars(
                symbol, body.adjust, date(1900, 1, 1), date.today()
            )
            if bars.empty:
                raise ValueError(f"No cached data for {symbol}")
            symbol_dates[symbol] = list(pd.to_datetime(bars["date"]))
        period = common_recent_evaluation_period(
            symbol_dates,
            evaluation_months=36,
            locked_oos_months=12,
        )
        protocol = ValidationProtocol(
            symbols=tuple(body.symbols),
            evaluation_start=period.evaluation_start.date(),
            evaluation_end=period.evaluation_end.date(),
            locked_oos_start=period.locked_oos_start.date(),
            locked_oos_end=period.locked_oos_end.date(),
            train_sessions=body.protocol.train_months * 21,
            validation_sessions=body.protocol.validation_months * 21,
            test_sessions=body.protocol.test_months * 21,
            step_sessions=body.protocol.test_months * 21,
            purge_sessions=body.protocol.purge_sessions,
            embargo_sessions=body.protocol.embargo_sessions,
            candidate_count=96,
            metadata={
                "minimum_round_trips_per_symbol": (
                    body.protocol.minimum_round_trips_per_symbol
                ),
                "minimum_market_exposure": (
                    body.protocol.minimum_market_exposure
                ),
                "locked_oos_policy": "single_evaluation_after_selection",
                "execution_policy": "T_close_signal_T_plus_1_open",
            },
        )
        common_dates = pd.DatetimeIndex(symbol_dates[body.symbols[0]])
        for symbol in body.symbols[1:]:
            common_dates = common_dates.intersection(symbol_dates[symbol])
        common_dates = common_dates[
            (common_dates >= period.evaluation_start)
            & (common_dates <= period.evaluation_end)
        ].sort_values()
        folds = generate_rolling_folds(
            common_dates,
            train_sessions=protocol.train_sessions,
            validation_sessions=protocol.validation_sessions,
            test_sessions=protocol.test_sessions,
            step_sessions=protocol.step_sessions,
            purge_sessions=protocol.purge_sessions,
            embargo_sessions=protocol.embargo_sessions,
            locked_oos_start=protocol.locked_oos_start,
        )
        frames = _build_walk_forward_base_frames(
            storage,
            provider,
            body,
            period.evaluation_start.date(),
            period.evaluation_end.date(),
        )
        candidates = generate_preregistered_candidates(count=96)
        validation_matrix = np.full(
            (len(folds), len(candidates)), np.nan
        )
        trade_matrix = np.zeros(
            (len(folds), len(candidates)), dtype=int
        )
        exposure_matrix = np.zeros(
            (len(folds), len(candidates)), dtype=float
        )
        candidate_rows: list[dict[str, Any]] = []
        trial_returns: dict[int, list[pd.Series]] = {
            index: [] for index in range(len(candidates))
        }
        for candidate_index, candidate in enumerate(candidates):
            options = _candidate_options(
                body.options, dict(candidate.parameters)
            )
            fold_scores: list[float] = []
            for fold_index, fold in enumerate(folds):
                metrics, combined = _evaluate_regime_segment(
                    frames, fold.validation_dates, options
                )
                value = (
                    float(metrics["objective"])
                    if metrics.get("available")
                    else float("nan")
                )
                validation_matrix[fold_index, candidate_index] = value
                trade_matrix[fold_index, candidate_index] = sum(
                    int(item.get("round_trip_count") or 0)
                    for item in (
                        metrics.get("symbol_metrics") or {}
                    ).values()
                )
                exposure_values = [
                    float(item.get("market_exposure") or 0.0)
                    for item in (
                        metrics.get("symbol_metrics") or {}
                    ).values()
                ]
                exposure_matrix[
                    fold_index, candidate_index
                ] = (
                    float(np.median(exposure_values))
                    if exposure_values
                    else 0.0
                )
                fold_scores.append(value)
                trial_returns[candidate_index].append(combined)
            total_trades = int(trade_matrix[:, candidate_index].sum())
            mean_exposure = float(
                np.mean(exposure_matrix[:, candidate_index])
            )
            minimum_round_trips = (
                len(body.symbols)
                * body.protocol.minimum_round_trips_per_symbol
            )
            eligible = (
                np.isfinite(fold_scores).any()
                and total_trades >= minimum_round_trips
                and mean_exposure
                >= body.protocol.minimum_market_exposure
            )
            combined_validation = (
                pd.concat(
                    trial_returns[candidate_index], axis=0
                ).sort_index()
                if trial_returns[candidate_index]
                else pd.Series(dtype=float)
            )
            validation_sharpe = sharpe_ratio(
                combined_validation.dropna()
            )
            candidate_rows.append(
                {
                    **candidate.to_dict(),
                    "median_validation_objective": (
                        float(np.nanmedian(fold_scores))
                        if np.isfinite(fold_scores).any()
                        and total_trades > 0
                        else None
                    ),
                    "validation_sharpe": (
                        float(validation_sharpe)
                        if np.isfinite(validation_sharpe)
                        else None
                    ),
                    "status": "eligible" if eligible else "eliminated",
                    "validation_round_trip_count": total_trades,
                    "validation_mean_market_exposure": mean_exposure,
                    "eliminated_reason": (
                        None
                        if eligible
                        else (
                            "insufficient_round_trips"
                            if total_trades < minimum_round_trips
                            else "insufficient_market_exposure"
                            if mean_exposure
                            < body.protocol.minimum_market_exposure
                            else "no_finite_objective"
                        )
                    ),
                }
            )
            storage.update_walk_forward_job(
                job_id,
                progress=0.05 + 0.70 * (candidate_index + 1) / len(candidates),
                summary={
                    "candidate_count": len(candidates),
                    "completed_candidates": candidate_index + 1,
                    "fold_count": len(folds),
                    "protocol_hash": protocol.protocol_hash,
                },
            )
        eliminated = np.array(
            [row["status"] == "eliminated" for row in candidate_rows]
        )
        validation_matrix[:, eliminated] = np.nan
        winners: list[dict[str, Any]] = []
        for fold_index, fold in enumerate(folds):
            row = validation_matrix[fold_index]
            if not np.isfinite(row).any():
                continue
            winner_index = int(np.nanargmax(row))
            options = _candidate_options(
                body.options,
                dict(candidates[winner_index].parameters),
            )
            test_metrics, _ = _evaluate_regime_segment(
                frames, fold.test_dates, options
            )
            winners.append(
                {
                    "fold": fold.snapshot(),
                    "candidate": candidates[winner_index].to_dict(),
                    "validation_objective": float(row[winner_index]),
                    "test": _comparison_view(test_metrics),
                }
            )
        medians = np.asarray(
            [
                float(np.nanmedian(column))
                if np.isfinite(column).any()
                else float("nan")
                for column in validation_matrix.T
            ]
        )
        if not np.isfinite(medians).any():
            raise ValueError(
                "All preregistered candidates were eliminated by the "
                "preregistered trade/exposure requirements"
            )
        final_index = int(np.nanargmax(medians))
        final_candidate = candidates[final_index]
        final_options = _candidate_options(
            body.options, dict(final_candidate.parameters)
        )
        oos_dates = tuple(
            common_dates[
                common_dates >= pd.Timestamp(protocol.locked_oos_start)
            ]
        )
        final_metrics, final_returns = _evaluate_regime_segment(
            frames, oos_dates, final_options
        )
        validation_returns = pd.concat(
            trial_returns[final_index], axis=0
        ).sort_index()
        trial_sharpes = [
            row["validation_sharpe"]
            for row in candidate_rows
            if row["validation_sharpe"] is not None
        ]
        aligned_trials = pd.concat(
            [
                pd.concat(trial_returns[index], axis=0)
                .sort_index()
                .rename(str(index))
                for index in range(len(candidates))
            ],
            axis=1,
        )
        comparison_options = {
            "buy_and_hold": body.options.model_copy(
                update={
                    "timing_style": "buy_and_hold",
                    "position_sizing": "full",
                    "cooldown_sessions": 0,
                }
            ),
            "ma_200": body.options.model_copy(
                update={
                    "timing_style": "ma_200",
                    "position_sizing": "full",
                    "cooldown_sessions": 0,
                }
            ),
            "trend": body.options.model_copy(
                update={"timing_style": "trend"}
            ),
            "rsi_bollinger": final_options.model_copy(
                update={"timing_style": "rsi_bollinger"}
            ),
            "factor_dual": body.options.model_copy(
                update={"timing_style": "factor_dual"}
            ),
            "regime_reversion_legacy": final_options.model_copy(
                update={
                    "timing_style": "regime_reversion_legacy",
                    "regime_entry_mode": "legacy_all",
                }
            ),
            "regime_reversion": final_options,
            "donchian_atr": body.options.model_copy(
                update={"timing_style": "donchian_atr"}
            ),
            "ma_crossover_atr": body.options.model_copy(
                update={"timing_style": "ma_crossover_atr"}
            ),
        }
        comparison_metrics: dict[str, dict[str, Any]] = {}
        for name, model_options in comparison_options.items():
            metrics = final_metrics
            if name != "regime_reversion":
                metrics, _ = _evaluate_regime_segment(
                    frames, oos_dates, model_options
                )
            comparison_metrics[name] = _comparison_view(metrics)
        buy_hold_return = comparison_metrics["buy_and_hold"].get(
            "total_return"
        )
        if buy_hold_return is not None:
            for metrics in comparison_metrics.values():
                total_return = metrics.get("total_return")
                metrics["excess_return"] = (
                    float(total_return) - float(buy_hold_return)
                    if total_return is not None
                    else None
                )
        perturbations = parameter_perturbations(
            dict(final_candidate.parameters)
        )
        perturbation_scores: list[float] = []
        for parameters in perturbations:
            options = _candidate_options(
                body.options, dict(parameters)
            )
            scores: list[float] = []
            for fold in folds:
                metrics, _ = _evaluate_regime_segment(
                    frames, fold.validation_dates, options
                )
                if metrics.get("available") and np.isfinite(
                    float(metrics.get("objective", np.nan))
                ):
                    scores.append(float(metrics["objective"]))
            if scores:
                perturbation_scores.append(float(np.median(scores)))
        result = {
            "protocol": protocol.snapshot().to_dict(),
            "protocol_hash": protocol.protocol_hash,
            "candidates": candidate_rows,
            "folds": winners,
            "selected_candidate": final_candidate.to_dict(),
            "locked_oos": final_metrics,
            "diagnostics": {
                "candidate_count": len(candidates),
                "fold_count": len(folds),
                "walk_forward_efficiency": walk_forward_efficiency(
                    [item["validation_objective"] for item in winners],
                    [
                        item["test"].get("objective")
                        for item in winners
                        if item["test"].get("available")
                    ],
                ),
                "deflated_sharpe": deflated_sharpe_ratio(
                    final_returns,
                    trial_sharpes=trial_sharpes,
                    number_of_trials=len(candidates),
                ),
                "pbo": cscv_pbo(aligned_trials),
                "perturbation_count": len(perturbations),
                "perturbation_stability": perturbation_stability(
                    float(medians[final_index]),
                    perturbation_scores,
                ),
                "evidence_sufficient": (
                    len(body.symbols) >= 8
                    and bool(
                        comparison_metrics.get(
                            "regime_reversion", {}
                        ).get("evidence_sufficient")
                    )
                ),
            },
            "model_comparison": comparison_metrics,
            "preregistered_model_candidates": {
                "regime_reversion": len(candidates),
                "buy_and_hold": 1,
                "ma_200": 1,
                "trend": 1,
                "rsi_bollinger": 1,
                "factor_dual": 1,
                "regime_reversion_legacy": 1,
                "donchian_atr": 1,
                "ma_crossover_atr": 1,
            },
            "warnings": [
                "Only three common years are used; evidence is limited.",
                *(
                    [
                        "Fewer than eight representative symbols were "
                        "available; cross-symbol evidence is insufficient."
                    ]
                    if len(body.symbols) < 8
                    else []
                ),
                SURVIVORSHIP_WARNING,
                QFQ_REVISION_WARNING,
            ],
        }
        report_dir = (
            storage.data_dir.parent
            / "reports"
            / protocol.evaluation_end.isoformat()
            / "regime_walk_forward"
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{job_id}.json"
        report_path.write_text(
            json.dumps(
                json_safe(result),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        storage.update_walk_forward_job(
            job_id,
            status="completed",
            progress=1.0,
            summary={
                "candidate_count": len(candidates),
                "fold_count": len(folds),
                "protocol_hash": protocol.protocol_hash,
                "selected_candidate": final_candidate.to_dict(),
                "locked_oos": final_metrics,
            },
            result=json_safe(result),
            report_path=str(report_path),
        )
    except Exception as exc:
        storage.update_walk_forward_job(
            job_id, status="failed", error=str(exc), progress=1.0
        )


def _multifactor_effective_signature(
    config: MultiFactorConfig,
) -> tuple[Any, ...]:
    components: list[tuple[Any, ...]] = []
    for component in config.components:
        if not component.enabled or component.weight == 0:
            continue
        metadata = factor_registry.get(component.factor_name).metadata
        components.append(
            (
                component.factor_name,
                float(component.weight),
                (
                    component.direction
                    if component.direction is not None
                    else metadata.direction
                ),
                component.normalization,
                component.winsorize,
                component.missing_policy,
            )
        )
    return (
        config.mode,
        config.rolling_window,
        config.rolling_min_periods,
        float(config.zscore_clip),
        tuple(sorted(components)),
    )


def _attach_benchmark_columns(
    panel: pd.DataFrame,
    provider: DataProvider,
    benchmark: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None, str | None]:
    symbol = BENCHMARK_SYMBOLS[benchmark]
    start_date = pd.Timestamp(panel["date"].min()).date()
    end_date = pd.Timestamp(panel["date"].max()).date()
    try:
        bars = provider.fetch_index(symbol, start_date, end_date)
    except Exception as exc:
        return panel, None, f"{benchmark} benchmark unavailable: {exc}"
    if bars is None or bars.empty:
        return panel, None, f"{benchmark} benchmark unavailable"
    try:
        return merge_benchmark_bars(panel, bars), bars, None
    except (FactorUnavailableError, ValueError) as exc:
        return panel, None, f"{benchmark} benchmark unavailable: {exc}"


def _multifactor_diagnostics(
    panel: pd.DataFrame,
    details: pd.DataFrame,
    forward_period: int = 5,
) -> dict[str, Any]:
    report = factor_correlation_report(details)
    close = pd.to_numeric(panel["close"], errors="coerce")
    forward_return = close.groupby(panel["symbol"], sort=False).transform(
        lambda values: values.shift(-forward_period) / values - 1.0
    )
    diagnostic_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(panel["date"]),
            "forward": forward_return,
        }
    )

    def mean_daily_ic(values: pd.Series) -> tuple[float | None, int]:
        frame = diagnostic_frame.assign(
            factor=pd.to_numeric(values, errors="coerce")
        ).dropna()
        daily_values: list[float] = []
        for _, group in frame.groupby("date", sort=True):
            if (
                len(group) >= 2
                and group["factor"].nunique() > 1
                and group["forward"].nunique() > 1
            ):
                value = group["factor"].corr(group["forward"])
                if pd.notna(value):
                    daily_values.append(float(value))
        return (
            float(pd.Series(daily_values).mean()) if daily_values else None,
            len(daily_values),
        )

    composite_ic, composite_ic_dates = mean_daily_ic(details["composite_score"])
    contribution_columns = [
        column
        for column in details.columns
        if column.startswith("contribution_")
    ]
    total_numerator = details[contribution_columns].sum(axis=1, min_count=1)
    marginal: dict[str, Any] = {}
    for column in (
        item for item in details.columns if item.startswith("normalized_")
    ):
        name = column.removeprefix("normalized_")
        values = pd.to_numeric(details[column], errors="coerce")
        factor_ic, factor_ic_dates = mean_daily_ic(values)
        contribution_column = f"contribution_{name}"
        valid_weight_column = f"valid_weight_{name}"
        leave_out_ic: float | None = None
        leave_out_dates = 0
        if (
            contribution_column in details.columns
            and valid_weight_column in details.columns
        ):
            denominator = pd.to_numeric(
                details["valid_weight"], errors="coerce"
            ) - pd.to_numeric(details[valid_weight_column], errors="coerce")
            leave_out_score = (
                total_numerator
                - pd.to_numeric(details[contribution_column], errors="coerce")
            ) / denominator.where(denominator.gt(0))
            leave_out_ic, leave_out_dates = mean_daily_ic(leave_out_score)
        marginal[name] = {
            "ic": factor_ic,
            "ic_dates": factor_ic_dates,
            "leave_one_out_composite_ic": leave_out_ic,
            "leave_one_out_ic_dates": leave_out_dates,
            "marginal_ic": (
                composite_ic - leave_out_ic
                if composite_ic is not None and leave_out_ic is not None
                else None
            ),
        }
    report["marginal_ic"] = marginal
    report["composite_ic"] = composite_ic
    report["composite_ic_dates"] = composite_ic_dates
    return json_safe(report)


def _resolve_symbols(
    requested: list[str] | None, storage: Storage, adjust: str
) -> list[str]:
    symbols = requested or storage.list_symbols(adjust)  # type: ignore[arg-type]
    if not symbols:
        raise HTTPException(
            status_code=400,
            detail="No symbols were requested and no cached symbols are available.",
        )
    return symbols


def _load_panel(
    storage: Storage,
    symbols: list[str],
    adjust: str,
    start_date: Any,
    end_date: Any,
    lookback: int,
    forward_period: int = 0,
) -> tuple[pd.DataFrame, list[str]]:
    # Load at least 300 trading sessions before the requested period. The
    # 650-calendar-day floor also leaves enough observations for a 252-session
    # factor followed by rolling standardization.
    warmup_start = start_date - timedelta(days=max(lookback * 3, 650))
    extended_end = end_date + timedelta(days=max(forward_period * 3, 0))
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for symbol in symbols:
        bars = storage.read_bars(
            symbol,
            adjust,  # type: ignore[arg-type]
            warmup_start,
            extended_end,
        )
        if bars.empty:
            missing.append(symbol)
        else:
            frames.append(bars)
    if not frames:
        raise HTTPException(
            status_code=400,
            detail="None of the requested symbols has cached bars. Download data first.",
        )
    return pd.concat(frames, ignore_index=True), missing


def _attach_execution_fields(
    storage: Storage,
    panel: pd.DataFrame,
    symbols: list[str],
    adjust: str,
) -> pd.DataFrame:
    """Attach unadjusted fields used only for execution constraints.

    Adjusted prices remain the accounting series so dividends/splits do not create
    artificial returns. Exchange price limits and suspension checks must use raw
    prices and volume.
    """

    source_columns = (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "prev_close",
        "is_st",
        "is_st_known",
    )
    if adjust == "none":
        output = panel.copy()
        for column in source_columns:
            output[f"trade_{column}"] = output[column]
        output["trade_reference_close"] = output["prev_close"]
        return output

    raw_frames: list[pd.DataFrame] = []
    missing_raw: list[str] = []
    start_date = pd.Timestamp(panel["date"].min()).date()
    end_date = pd.Timestamp(panel["date"].max()).date()
    for symbol in symbols:
        raw = storage.read_bars(symbol, "none", start_date, end_date)
        if raw.empty:
            missing_raw.append(symbol)
        else:
            raw_frames.append(raw)
    if missing_raw:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unadjusted execution data is required for qfq backtests. "
                f"Download qfq data again to populate it for: {', '.join(missing_raw)}"
            ),
        )

    raw_panel = pd.concat(raw_frames, ignore_index=True)
    execution = raw_panel.loc[:, ["symbol", "date", *source_columns]].rename(
        columns={column: f"trade_{column}" for column in source_columns}
    )
    output = panel.merge(
        execution,
        on=["symbol", "date"],
        how="left",
        validate="one_to_one",
    )
    adjusted_return = output["close"] / output["prev_close"]
    output["trade_reference_close"] = output["trade_close"] / adjusted_return
    valid_adjusted_return = adjusted_return.gt(0) & adjusted_return.lt(float("inf"))
    output.loc[~valid_adjusted_return, "trade_reference_close"] = output[
        "trade_prev_close"
    ]
    return output


def create_app(
    provider: DataProvider | None = None, storage: Storage | None = None
) -> FastAPI:
    application = FastAPI(
        title="A-share Quant Research MVP",
        version="0.1.0",
        description=(
            "Factor research and conservative T+1-close A-share backtesting API."
        ),
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.state.provider = provider or AkShareProvider()
    application.state.storage = storage or Storage()
    application.state.instrument_names = _local_instrument_name_map()
    application.state.provider_names_loaded = False

    def resolve_instrument_names(symbols: list[str]) -> dict[str, str]:
        names: dict[str, str] = application.state.instrument_names
        missing = [symbol for symbol in symbols if symbol not in names]
        if missing and not application.state.provider_names_loaded:
            try:
                frame = application.state.provider.list_stocks()
                if {"symbol", "name"}.issubset(frame.columns):
                    for symbol, name in frame.loc[
                        :, ["symbol", "name"]
                    ].itertuples(index=False, name=None):
                        names[normalize_symbol(symbol)] = str(name)
            except Exception:
                pass
            application.state.provider_names_loaded = True
        return names

    @application.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "a-share-quant-mvp",
            "version": application.version,
            "provider": application.state.provider.name,
            "database": "ok",
        }

    @application.get("/api/data/status")
    def data_status() -> dict[str, Any]:
        status = application.state.storage.dataset_status()
        datasets = status.get("datasets", [])
        symbols = list(
            dict.fromkeys(
                str(item.get("symbol"))
                for item in datasets
                if item.get("symbol")
            )
        )
        names = resolve_instrument_names(symbols)
        for item in datasets:
            symbol = str(item.get("symbol", ""))
            item["name"] = names.get(symbol)
        return {
            "status": "ok",
            "provider": application.state.provider.name,
            **status,
        }

    @application.get("/api/research/universe")
    def research_universe() -> dict[str, Any]:
        report_root = Path(__file__).resolve().parents[1] / "reports"
        candidates = sorted(report_root.glob("20??-??-??/stock_universe.csv"))
        if not candidates:
            return {
                "status": "empty",
                "count": 0,
                "stocks": [],
                "warnings": ["尚未生成主板研究股票池。"],
            }
        path = candidates[-1]
        frame = pd.read_csv(path, dtype={"symbol": str})
        records = json_safe(frame.to_dict(orient="records"))
        market_counts = {
            str(key): int(value)
            for key, value in frame["market"].value_counts().to_dict().items()
        }
        return {
            "status": "ready",
            "report_date": path.parent.name,
            "count": len(records),
            "market_counts": market_counts,
            "latest_trade_date": (
                str(frame["latest_trade_date"].max())
                if "latest_trade_date" in frame.columns and not frame.empty
                else None
            ),
            "stocks": records,
            "warnings": [SURVIVORSHIP_WARNING],
        }

    @application.get("/api/research/etfs")
    def research_etfs() -> dict[str, Any]:
        report_root = Path(__file__).resolve().parents[1] / "reports"
        candidates = sorted(
            report_root.glob("20??-??-??/etfs/etf_universe.csv")
        )
        if not candidates:
            return {
                "status": "empty",
                "count": 0,
                "etfs": [],
                "warnings": ["尚未生成ETF研究测试集。"],
            }
        path = candidates[-1]
        frame = pd.read_csv(path, dtype={"symbol": str})
        records = json_safe(frame.to_dict(orient="records"))
        return {
            "status": "ready",
            "report_date": path.parents[1].name,
            "count": len(records),
            "market_counts": {
                str(key): int(value)
                for key, value in frame["market"].value_counts().to_dict().items()
            },
            "category_counts": {
                str(key): int(value)
                for key, value in frame["category"].value_counts().to_dict().items()
            },
            "latest_trade_date": (
                str(frame["data_end_date"].max()) if not frame.empty else None
            ),
            "etfs": records,
            "warnings": [
                "ETF列表按当前存续产品构建，历史研究存在基金存续偏差。"
            ],
        }

    @application.get("/api/data/stocks/{symbol}/bars")
    def stock_bars(
        symbol: str,
        adjust: str = Query(default="qfq", pattern="^(qfq|none)$"),
        start_date: date | None = Query(default=None),
        end_date: date | None = Query(default=None),
        limit: int = Query(default=250, ge=20, le=2000),
    ) -> dict[str, Any]:
        try:
            normalized_symbol = normalize_symbol(symbol)
        except ProviderDataError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if start_date and end_date and start_date > end_date:
            raise HTTPException(
                status_code=422, detail="start_date must be on or before end_date"
            )
        bars = application.state.storage.read_bars(
            normalized_symbol,
            adjust,  # type: ignore[arg-type]
            start_date,
            end_date,
        )
        if bars.empty:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No cached {adjust} bars are available for "
                    f"{normalized_symbol}."
                ),
            )
        bars = bars.sort_values("date").tail(limit).copy()
        bars["change_pct"] = bars["close"] / bars["prev_close"] - 1.0
        columns = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "prev_close",
            "change_pct",
        ]
        for column in columns:
            if column not in bars.columns:
                bars[column] = None

        name = resolve_instrument_names([normalized_symbol]).get(
            normalized_symbol
        )

        if normalized_symbol.startswith(("51", "56", "58")):
            market = "上海ETF"
        elif normalized_symbol.startswith("15"):
            market = "深圳ETF"
        elif normalized_symbol.startswith(("600", "601", "603", "605")):
            market = "上海主板"
        elif normalized_symbol.startswith(("000", "001", "002", "003")):
            market = "深圳主板"
        elif normalized_symbol.startswith(("300", "301")):
            market = "创业板"
        elif normalized_symbol.startswith(("688", "689")):
            market = "科创板"
        else:
            market = "其他"

        latest = bars.iloc[-1]
        return {
            "symbol": normalized_symbol,
            "name": name,
            "market": market,
            "adjust": adjust,
            "start_date": pd.Timestamp(bars["date"].min()).date().isoformat(),
            "end_date": pd.Timestamp(bars["date"].max()).date().isoformat(),
            "count": len(bars),
            "latest": json_safe(
                {
                    column: latest[column]
                    for column in columns
                }
            ),
            "bars": json_safe(bars.loc[:, columns].to_dict(orient="records")),
        }

    @application.get("/api/data/stocks")
    def stocks(limit: int = Query(default=100, ge=1, le=5000)) -> dict[str, Any]:
        try:
            frame = application.state.provider.list_stocks()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Unable to load the current stock list: {exc}",
            ) from exc
        if not {"symbol", "name"}.issubset(frame.columns):
            raise HTTPException(
                status_code=502,
                detail="Provider stock list does not satisfy symbol/name contract.",
            )
        records = json_safe(frame.head(limit).to_dict(orient="records"))
        return {
            "stocks": records,
            "count": len(records),
            "is_current_snapshot": True,
            "universe_warning": SURVIVORSHIP_WARNING,
            "warnings": [SURVIVORSHIP_WARNING],
        }

    @application.get("/api/data/calendar")
    def trade_calendar(
        start_date: date = Query(...),
        end_date: date = Query(...),
    ) -> dict[str, Any]:
        if start_date > end_date:
            raise HTTPException(
                status_code=422, detail="start_date must be on or before end_date"
            )
        try:
            frame = application.state.provider.fetch_trade_calendar(
                start_date, end_date
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Unable to load the A-share trade calendar: {exc}",
            ) from exc
        dates = (
            pd.to_datetime(frame["trade_date"], errors="coerce")
            .dropna()
            .dt.strftime("%Y-%m-%d")
            .tolist()
            if "trade_date" in frame.columns
            else []
        )
        return {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "count": len(dates),
            "trade_dates": dates,
        }

    @application.get("/api/data/adjustment-factors/{symbol}")
    def adjustment_factors(
        symbol: str,
        start_date: date = Query(...),
        end_date: date = Query(...),
    ) -> dict[str, Any]:
        if start_date > end_date:
            raise HTTPException(
                status_code=422, detail="start_date must be on or before end_date"
            )
        try:
            frame = application.state.provider.fetch_adjustment_factors(
                symbol, start_date, end_date
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Unable to load adjustment factors for {symbol}: {exc}",
            ) from exc
        return {
            "symbol": symbol,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "factor_type": "qfq",
            "factors": json_safe(frame.to_dict(orient="records")),
        }

    @application.post("/api/data/download")
    def download(body: DownloadRequest) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for symbol in body.symbols:
            try:
                research_result = application.state.storage.update_symbol(
                    application.state.provider,
                    symbol,
                    body.start_date,
                    body.end_date,
                    body.adjust,
                )
                result = research_result
                if body.adjust == "qfq":
                    execution_result = application.state.storage.update_symbol(
                        application.state.provider,
                        symbol,
                        body.start_date,
                        body.end_date,
                        "none",
                    )
                    statuses = {
                        research_result["status"],
                        execution_result["status"],
                    }
                    if "no_data" in statuses:
                        combined_status = "no_data"
                    elif "updated" in statuses:
                        combined_status = "updated"
                    else:
                        combined_status = "up_to_date"
                    result = {
                        **research_result,
                        "status": combined_status,
                        "execution_adjust": "none",
                        "execution_cache": execution_result,
                    }
            except Exception as exc:
                result = {
                    "symbol": symbol,
                    "status": "error",
                    "error": str(exc),
                }
            results.append(result)
        successes = sum(
            result["status"] in {"updated", "up_to_date"} for result in results
        )
        return {
            "status": (
                "completed"
                if successes == len(results)
                else "failed"
                if successes == 0
                else "partial"
            ),
            "adjust": body.adjust,
            "results": results,
        }

    @application.get("/api/multifactor/templates")
    def multifactor_templates() -> dict[str, Any]:
        return {
            "templates": MULTIFACTOR_TEMPLATES,
            "factors": factor_registry.list(),
            "note": (
                "Template weights are research defaults only. They were not "
                "optimized on full-sample returns and are not claimed to be optimal."
            ),
        }

    @application.get("/api/multifactor/configs")
    def list_multifactor_configs() -> dict[str, Any]:
        items = application.state.storage.list_multifactor_configs()
        return {"items": items, "count": len(items)}

    @application.get("/api/multifactor/configs/{config_id}")
    def get_multifactor_config(config_id: str) -> dict[str, Any]:
        item = application.state.storage.get_multifactor_config(config_id)
        if item is None:
            raise HTTPException(
                status_code=404, detail="Multifactor config not found"
            )
        return item

    @application.post("/api/multifactor/configs")
    def save_multifactor_config(
        body: MultiFactorConfigRequest,
    ) -> dict[str, Any]:
        config = _to_multifactor_config(body)
        return application.state.storage.save_multifactor_config(
            config.config_id,
            config.name,
            config.snapshot(),
        )

    @application.get("/api/factors")
    def factors() -> dict[str, Any]:
        return {
            "factors": factor_registry.list(),
            "warnings": factor_registry.warnings,
            "pb_note": (
                "PB is available only when cached bars include a point-in-time PB "
                "field; the service never fabricates or current-date-backfills it."
            ),
        }

    @application.post("/api/factors/analyze")
    def analyze_factor(body: FactorAnalyzeRequest) -> dict[str, Any]:
        factor = _resolve_factor(body.factor_name)
        symbols = _resolve_symbols(
            body.symbols, application.state.storage, body.adjust
        )
        panel, missing = _load_panel(
            application.state.storage,
            symbols,
            body.adjust,
            body.start_date,
            body.end_date,
            factor.metadata.lookback,
            body.forward_period,
        )
        panel, _, benchmark_warning = _attach_benchmark_columns(
            panel,
            application.state.provider,
            body.benchmark,
        )
        preprocess = _preprocess_config(body.preprocess)
        try:
            result = evaluate_factor(
                panel,
                factor,
                body.start_date,
                body.end_date,
                body.forward_period,
                body.quantiles,
                preprocess,
            )
        except FactorUnavailableError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        warnings = [
            *[str(item) for item in result.get("warnings", [])],
            SURVIVORSHIP_WARNING,
        ]
        if body.adjust == "qfq":
            warnings.append(QFQ_REVISION_WARNING)
        if benchmark_warning:
            warnings.append(benchmark_warning)
        result.update(
            {
                "factor_name_zh": factor.metadata.display_name_zh,
                "factor_description_zh": factor.metadata.description_zh,
                "factor_direction": factor.metadata.direction,
                "symbols": symbols,
                "missing_symbols": missing,
                "universe_warning": SURVIVORSHIP_WARNING,
                "warnings": list(dict.fromkeys(warnings)),
            }
        )
        return json_safe(result)

    @application.post("/api/multifactor/analyze")
    def analyze_multifactor(body: MultiFactorAnalyzeRequest) -> dict[str, Any]:
        config = _to_multifactor_config(body.config, mode="cross_sectional")
        factor = CompositeFactor(config)
        symbols = _resolve_symbols(
            body.symbols, application.state.storage, body.adjust
        )
        panel, missing = _load_panel(
            application.state.storage,
            symbols,
            body.adjust,
            body.start_date,
            body.end_date,
            factor.metadata.lookback,
            body.forward_period,
        )
        panel, _, benchmark_warning = _attach_benchmark_columns(
            panel,
            application.state.provider,
            body.benchmark,
        )
        try:
            result = evaluate_factor(
                panel,
                factor,
                body.start_date,
                body.end_date,
                body.forward_period,
                body.quantiles,
                PreprocessConfig(winsorize=False, zscore=False),
            )
            details = factor.compute_details(panel)
        except (FactorUnavailableError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        mask = (
            panel["date"].ge(pd.Timestamp(body.start_date))
            & panel["date"].le(pd.Timestamp(body.end_date))
        )
        diagnostics = _multifactor_diagnostics(
            panel.loc[mask].reset_index(drop=True),
            details.loc[mask].reset_index(drop=True),
            body.forward_period,
        )
        saved = application.state.storage.save_multifactor_config(
            config.config_id,
            config.name,
            config.snapshot(),
        )
        warnings = [SURVIVORSHIP_WARNING, QFQ_REVISION_WARNING]
        if benchmark_warning:
            warnings.append(benchmark_warning)
        result.update(
            {
                "factor_name": factor.metadata.name,
                "factor_name_zh": config.name,
                "config_id": config.config_id,
                "config_snapshot": config.snapshot(),
                "saved_config": saved,
                "correlation_report": diagnostics,
                "contribution_summary": diagnostics.get(
                    "mean_absolute_contribution", {}
                ),
                "symbols": symbols,
                "missing_symbols": missing,
                "warnings": list(
                    dict.fromkeys(
                        [
                            *[str(item) for item in result.get("warnings", [])],
                            *warnings,
                        ]
                    )
                ),
            }
        )
        return json_safe(result)

    @application.post("/api/multifactor/backtests")
    def backtest_multifactor(
        body: MultiFactorBacktestRequest,
    ) -> dict[str, Any]:
        config = _to_multifactor_config(body.config, mode="cross_sectional")
        factor = CompositeFactor(config)
        symbols = _resolve_symbols(
            body.symbols, application.state.storage, body.adjust
        )
        if len(symbols) <= body.top_n:
            raise HTTPException(
                status_code=422,
                detail="Multifactor selection requires universe size > top_n.",
            )
        panel, missing = _load_panel(
            application.state.storage,
            symbols,
            body.adjust,
            body.start_date,
            body.end_date,
            factor.metadata.lookback,
        )
        panel, benchmark_bars, benchmark_warning = _attach_benchmark_columns(
            panel,
            application.state.provider,
            body.benchmark,
        )
        panel = _attach_execution_fields(
            application.state.storage,
            panel,
            [symbol for symbol in symbols if symbol not in missing],
            body.adjust,
        )
        try:
            details = factor.compute_details(panel)
            result = run_backtest(
                panel,
                factor,
                BacktestConfig(
                    start_date=body.start_date,
                    end_date=body.end_date,
                    top_n=body.top_n,
                    rebalance=body.rebalance,
                    commission_rate=body.commission_rate,
                    minimum_commission=body.minimum_commission,
                    minimum_trade_notional=body.minimum_trade_notional,
                    rebalance_tolerance=body.rebalance_tolerance,
                    stamp_duty_rate=body.stamp_duty_rate,
                    historical_stamp_duty=body.historical_stamp_duty,
                    slippage_rate=body.slippage_rate,
                    max_stale_sessions=body.max_stale_sessions,
                ),
                benchmark_bars,
            )
        except (FactorUnavailableError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        period_mask = (
            panel["date"].ge(pd.Timestamp(body.start_date))
            & panel["date"].le(pd.Timestamp(body.end_date))
        )
        diagnostics = _multifactor_diagnostics(
            panel.loc[period_mask].reset_index(drop=True),
            details.loc[period_mask].reset_index(drop=True),
        )
        run_id = uuid4().hex
        warnings = [
            *[str(item) for item in result.get("warnings", [])],
            SURVIVORSHIP_WARNING,
            QFQ_REVISION_WARNING,
        ]
        if benchmark_warning:
            warnings.append(benchmark_warning)
        result.update(
            {
                "id": run_id,
                "factor_name": factor.metadata.name,
                "factor_name_zh": config.name,
                "config_id": config.config_id,
                "config_snapshot": config.snapshot(),
                "correlation_report": diagnostics,
                "contribution_summary": diagnostics.get(
                    "mean_absolute_contribution", {}
                ),
                "symbols": symbols,
                "missing_symbols": missing,
                "benchmark": body.benchmark,
                "warnings": list(dict.fromkeys(warnings)),
            }
        )
        request_payload = body.model_dump(mode="json")
        application.state.storage.save_multifactor_config(
            config.config_id,
            config.name,
            config.snapshot(),
        )
        application.state.storage.save_backtest(
            run_id,
            factor.metadata.name,
            request_payload,
            result["summary"],
            result,
        )
        return json_safe(result)

    @application.post("/api/timing/backtests")
    def timing_backtest(body: TimingBacktestRequest) -> dict[str, Any]:
        options = body.options
        config = _to_multifactor_config(body.config, mode="time_series")
        factor = CompositeFactor(config)
        entry_config = _to_multifactor_config(
            body.entry_config or body.config,
            mode="time_series",
        )
        exit_config = _to_multifactor_config(
            body.exit_config or body.config,
            mode="time_series",
        )
        entry_factor = CompositeFactor(entry_config)
        exit_factor = CompositeFactor(exit_config)
        if (
            options.timing_style
            in {
                "factor_dual",
                "regime_reversion",
                "regime_reversion_legacy",
            }
            and _multifactor_effective_signature(entry_config)
            == _multifactor_effective_signature(exit_config)
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    "智能双评分的买入配置与卖出配置完全相同，无法产生独立的 "
                    "entry_score 和 exit_score。请恢复智能默认配置或调整因子、"
                    "方向、权重及标准化设置。"
                ),
            )
        required_lookback = (
            max(
                entry_factor.metadata.lookback,
                exit_factor.metadata.lookback,
            )
            if options.timing_style
            in {
                "factor_dual",
                "regime_reversion",
                "regime_reversion_legacy",
            }
            else factor.metadata.lookback
        )
        required_lookback = max(
            required_lookback,
            options.atr_period,
        )
        if options.timing_style in {
            "regime_reversion",
            "regime_reversion_legacy",
        }:
            required_lookback = max(
                required_lookback,
                options.ma_period + options.ma_slope_period,
                options.rsi_period + 1,
                options.bollinger_window,
            )
        elif options.timing_style == "rsi_bollinger":
            required_lookback = max(
                required_lookback,
                options.rsi_period + 1,
                options.bollinger_window,
            )
        elif options.timing_style == "donchian_atr":
            required_lookback = max(
                required_lookback,
                options.donchian_entry_window + 1,
                options.donchian_exit_window + 1,
                (
                    options.ma_period + options.ma_slope_period
                    if options.donchian_trend_filter
                    else 0
                ),
            )
        elif options.timing_style == "ma_crossover_atr":
            required_lookback = max(
                required_lookback,
                options.ma_slow_period + options.ma_slope_period,
            )
        elif options.timing_style == "ma_200":
            required_lookback = max(
                required_lookback,
                options.ma_period,
            )
        panel, missing = _load_panel(
            application.state.storage,
            [body.symbol],
            body.adjust,
            body.start_date,
            body.end_date,
            required_lookback,
        )
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"No cached bars for {body.symbol}",
            )
        panel, benchmark_bars, benchmark_warning = _attach_benchmark_columns(
            panel,
            application.state.provider,
            body.benchmark,
        )
        panel = _attach_execution_fields(
            application.state.storage,
            panel,
            [body.symbol],
            body.adjust,
        )
        try:
            if options.timing_style in {
                "factor_dual",
                "regime_reversion",
                "regime_reversion_legacy",
            }:
                entry_details = _prefix_timing_details(
                    entry_factor.compute_details(panel),
                    entry_config,
                    "entry",
                )
                exit_details = _prefix_timing_details(
                    exit_factor.compute_details(panel),
                    exit_config,
                    "exit",
                )
                signal_frame = panel.merge(
                    entry_details,
                    on=["symbol", "date"],
                    how="left",
                    validate="one_to_one",
                ).merge(
                    exit_details,
                    on=["symbol", "date"],
                    how="left",
                    validate="one_to_one",
                )
                signal_frame["composite_score"] = (
                    signal_frame["entry_score"]
                    - signal_frame["exit_score"]
                ) / 2.0
                signal_frame["trend_score"] = signal_frame["entry_score"]
            else:
                details = factor.compute_details(panel)
                detail_columns = [
                    column
                    for column in details.columns
                    if column not in {"symbol", "date"}
                ]
                signal_frame = panel.merge(
                    details.loc[:, ["symbol", "date", *detail_columns]],
                    on=["symbol", "date"],
                    how="left",
                    validate="one_to_one",
                )
                for component in config.components:
                    if not component.enabled or component.weight == 0:
                        continue
                    component_factor = factor_registry.get(
                        component.factor_name
                    )
                    signal_frame[
                        f"weight_{component.factor_name}"
                    ] = component.weight
                    signal_frame[
                        f"direction_{component.factor_name}"
                    ] = (
                        component.direction
                        if component.direction is not None
                        else component_factor.metadata.direction
                    )
        except (FactorUnavailableError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        rolling_high = signal_frame.groupby(
            "symbol", sort=False
        )["high"].transform(
            lambda values: values.rolling(60, min_periods=60).max()
        )
        rolling_low = signal_frame.groupby(
            "symbol", sort=False
        )["low"].transform(
            lambda values: values.rolling(60, min_periods=60).min()
        )
        signal_frame["timing_price_position_60"] = (
            pd.to_numeric(signal_frame["close"], errors="coerce")
            - rolling_low
        ) / (rolling_high - rolling_low).where(rolling_high.ne(rolling_low))
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
            and f"contribution_{item.factor_name}" in signal_frame.columns
        ]
        if trend_components:
            numerator = sum(
                (
                    pd.to_numeric(
                        signal_frame[f"contribution_{item.factor_name}"],
                        errors="coerce",
                    ).fillna(0.0)
                    for item in trend_components
                ),
                start=pd.Series(0.0, index=signal_frame.index),
            )
            denominator = sum(
                (
                    pd.to_numeric(
                        signal_frame[f"valid_weight_{item.factor_name}"],
                        errors="coerce",
                    ).fillna(0.0)
                    for item in trend_components
                ),
                start=pd.Series(0.0, index=signal_frame.index),
            )
            signal_frame["trend_score"] = numerator / denominator.where(
                denominator.gt(0)
            )
        else:
            signal_frame["trend_score"] = signal_frame["composite_score"]
        if options.timing_style == "rsi_bollinger":
            signal_frame["entry_score"] = 0.0
            signal_frame["exit_score"] = 0.0
        signal_frame = _attach_timing_indicator_columns(
            signal_frame, options
        )
        signal_frame = signal_frame[
            signal_frame["date"].ge(pd.Timestamp(body.start_date))
            & signal_frame["date"].le(pd.Timestamp(body.end_date))
        ].reset_index(drop=True)
        try:
            result = run_timing(
                signal_frame,
                _timing_config_from_options(options, body.is_etf),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if any(
            column.startswith("normalized_")
            for column in signal_frame.columns
        ):
            correlation_report = factor_correlation_report(signal_frame)
            result["correlation_report"] = correlation_report
            result["contribution_summary"] = correlation_report.get(
                "mean_absolute_contribution", {}
            )

        if benchmark_bars is not None and not benchmark_bars.empty:
            metrics = calculate_metrics(
                pd.DataFrame(result["equity_curve"]),
                benchmark_bars,
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
            benchmark_by_date = {
                pd.Timestamp(item["date"]).normalize(): item["net_value"]
                for item in metrics["benchmark_curve"]
            }
            for item in result["equity_curve"]:
                item["strategy"] = item["net_value"]
                item["benchmark"] = benchmark_by_date.get(
                    pd.Timestamp(item["date"]).normalize()
                )
            result["benchmark_curve"] = metrics["benchmark_curve"]

        run_id = uuid4().hex
        instrument_name = resolve_instrument_names([body.symbol]).get(body.symbol)
        warnings = [
            *[str(item) for item in result.get("warnings", [])],
            QFQ_REVISION_WARNING,
        ]
        if benchmark_warning:
            warnings.append(benchmark_warning)
        result.update(
            {
                "id": run_id,
                "symbol": body.symbol,
                "name": instrument_name,
                "factor_name": (
                    options.timing_style
                    if options.timing_style
                    in {"donchian_atr", "ma_crossover_atr"}
                    else factor.metadata.name
                    if options.timing_style
                    not in {
                        "factor_dual",
                        "regime_reversion",
                        "regime_reversion_legacy",
                    }
                    else (
                        f"{options.timing_style}_{entry_config.config_id}_"
                        f"{exit_config.config_id}"
                    )
                ),
                "factor_name_zh": (
                    "Donchian突破 + ATR"
                    if options.timing_style == "donchian_atr"
                    else "双均线趋势 + ATR"
                    if options.timing_style == "ma_crossover_atr"
                    else config.name
                    if options.timing_style
                    not in {
                        "factor_dual",
                        "regime_reversion",
                        "regime_reversion_legacy",
                    }
                    else (
                        "综合趋势反转"
                        if options.timing_style
                        in {
                            "regime_reversion",
                            "regime_reversion_legacy",
                        }
                        else "智能双评分择时"
                    )
                ),
                "config_id": config.config_id,
                "config_snapshot": config.snapshot(),
                "timing_options_snapshot": options.model_dump(mode="json"),
                "entry_config_id": (
                    entry_config.config_id
                    if options.timing_style
                    in {
                        "factor_dual",
                        "regime_reversion",
                        "regime_reversion_legacy",
                    }
                    else None
                ),
                "entry_config_snapshot": (
                    entry_config.snapshot()
                    if options.timing_style
                    in {
                        "factor_dual",
                        "regime_reversion",
                        "regime_reversion_legacy",
                    }
                    else None
                ),
                "exit_config_id": (
                    exit_config.config_id
                    if options.timing_style
                    in {
                        "factor_dual",
                        "regime_reversion",
                        "regime_reversion_legacy",
                    }
                    else None
                ),
                "exit_config_snapshot": (
                    exit_config.snapshot()
                    if options.timing_style
                    in {
                        "factor_dual",
                        "regime_reversion",
                        "regime_reversion_legacy",
                    }
                    else None
                ),
                "benchmark": body.benchmark,
                "asset_type": "ETF" if body.is_etf else "stock",
                "warnings": list(dict.fromkeys(warnings)),
            }
        )
        request_payload = body.model_dump(mode="json")
        application.state.storage.save_multifactor_config(
            config.config_id,
            config.name,
            config.snapshot(),
        )
        if options.timing_style in {
            "factor_dual",
            "regime_reversion",
            "regime_reversion_legacy",
        }:
            application.state.storage.save_multifactor_config(
                entry_config.config_id,
                entry_config.name,
                entry_config.snapshot(),
            )
            application.state.storage.save_multifactor_config(
                exit_config.config_id,
                exit_config.name,
                exit_config.snapshot(),
            )
        application.state.storage.save_backtest(
            run_id,
            result["factor_name"],
            request_payload,
            result["summary"],
            result,
        )
        return json_safe(result)

    @application.post("/api/timing/walk-forward")
    def create_timing_walk_forward(
        body: TimingWalkForwardRequest,
    ) -> dict[str, Any]:
        request_payload = body.model_dump(mode="json")
        existing = (
            application.state.storage.find_completed_walk_forward_job(
                request_payload
            )
        )
        if existing is not None:
            return json_safe(
                {
                    **existing,
                    "reused_locked_oos": True,
                    "message": (
                        "An identical preregistered protocol already "
                        "evaluated locked OOS; the stored result was reused."
                    ),
                }
            )
        job_id = uuid4().hex
        application.state.storage.create_walk_forward_job(
            job_id, request_payload
        )
        _WALK_FORWARD_EXECUTOR.submit(
            _run_walk_forward_research,
            application.state.storage,
            application.state.provider,
            body,
            job_id,
        )
        return application.state.storage.get_walk_forward_job(job_id) or {
            "id": job_id,
            "status": "pending",
        }

    @application.get("/api/timing/walk-forward/{job_id}")
    def get_timing_walk_forward(job_id: str) -> dict[str, Any]:
        job = application.state.storage.get_walk_forward_job(job_id)
        if job is None:
            raise HTTPException(
                status_code=404, detail="Walk-forward job not found"
            )
        return json_safe(job)

    @application.post("/api/backtests")
    def create_backtest(body: BacktestRequest) -> dict[str, Any]:
        factor = _resolve_factor(body.factor_name)
        symbols = _resolve_symbols(
            body.symbols, application.state.storage, body.adjust
        )
        if len(symbols) <= body.top_n:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Factor-ranking backtests require a universe larger than "
                    f"top_n; received {len(symbols)} symbol(s) and top_n="
                    f"{body.top_n}. Load the ETF 20 or main-board 100 universe."
                ),
            )
        panel, missing = _load_panel(
            application.state.storage,
            symbols,
            body.adjust,
            body.start_date,
            body.end_date,
            factor.metadata.lookback,
        )
        panel, benchmark_bars, benchmark_warning = _attach_benchmark_columns(
            panel,
            application.state.provider,
            body.benchmark,
        )
        panel = _attach_execution_fields(
            application.state.storage,
            panel,
            [symbol for symbol in symbols if symbol not in missing],
            body.adjust,
        )

        config = BacktestConfig(
            start_date=body.start_date,
            end_date=body.end_date,
            top_n=body.top_n,
            rebalance=body.rebalance,
            commission_rate=body.commission_rate,
            minimum_commission=body.minimum_commission,
            minimum_trade_notional=body.minimum_trade_notional,
            rebalance_tolerance=body.rebalance_tolerance,
            stamp_duty_rate=body.stamp_duty_rate,
            historical_stamp_duty=body.historical_stamp_duty,
            slippage_rate=body.slippage_rate,
            max_stale_sessions=body.max_stale_sessions,
        )
        try:
            result = run_backtest(panel, factor, config, benchmark_bars)
        except FactorUnavailableError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        run_id = uuid4().hex
        warnings = [
            *[str(item) for item in result.get("warnings", [])],
            SURVIVORSHIP_WARNING,
        ]
        if body.adjust == "qfq":
            warnings.append(QFQ_REVISION_WARNING)
        if benchmark_warning:
            warnings.append(benchmark_warning)
        result.update(
            {
                "id": run_id,
                "factor_name_zh": factor.metadata.display_name_zh,
                "factor_description_zh": factor.metadata.description_zh,
                "factor_direction": factor.metadata.direction,
                "benchmark": body.benchmark,
                "benchmark_warning": benchmark_warning,
                "symbols": symbols,
                "missing_symbols": missing,
                "universe_warning": SURVIVORSHIP_WARNING,
                "warnings": list(dict.fromkeys(warnings)),
            }
        )
        request_payload = body.model_dump(mode="json")
        application.state.storage.save_backtest(
            run_id,
            factor.metadata.name,
            request_payload,
            result["summary"],
            result,
        )
        return json_safe(result)

    @application.get("/api/backtests")
    def list_backtests() -> dict[str, Any]:
        runs = application.state.storage.list_backtests()
        return {
            "runs": runs,
            "backtests": runs,
            "items": runs,
            "count": len(runs),
            "total": len(runs),
        }

    @application.get("/api/backtests/{run_id}")
    def get_backtest(run_id: str) -> dict[str, Any]:
        run = application.state.storage.get_backtest(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Backtest run not found")
        return run

    @application.delete("/api/backtests/{run_id}")
    def delete_backtest(run_id: str) -> dict[str, Any]:
        deleted = application.state.storage.delete_backtest(run_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Backtest run not found")
        return {"id": run_id, "deleted": True}

    return application


app = create_app()
