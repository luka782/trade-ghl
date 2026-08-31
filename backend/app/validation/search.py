from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd


# 参数空间固定、随机种子固定：研究者必须在看样本外结果之前确定候选集合，
# 以降低“不断尝试直到得到好结果”的数据窥探风险。
DEFAULT_PARAMETER_GRID: Mapping[str, tuple[Any, ...]] = MappingProxyType(
    {
        "ma_period": (120, 150, 200, 250),
        "rsi_period": (7, 14, 21),
        "rsi_oversold": (20.0, 25.0, 30.0, 35.0),
        "rsi_overbought": (65.0, 70.0, 75.0, 80.0),
        "bollinger_window": (20, 30, 40),
        "bollinger_std": (1.5, 2.0, 2.5),
        "setup_expiry_sessions": (10, 20, 30, 45),
        "weight_preset": (0, 1, 2),
    }
)


def _canonical(parameters: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(sorted(parameters.items())),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    parameters: Mapping[str, Any]
    preregistration_index: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "parameters", MappingProxyType(dict(self.parameters))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "preregistration_index": self.preregistration_index,
            "parameters": dict(self.parameters),
        }


def generate_preregistered_candidates(
    parameter_grid: Mapping[str, Sequence[Any]] | None = None,
    *,
    count: int = 96,
    seed: int = 20260825,
) -> tuple[Candidate, ...]:
    """由完整网格确定性抽样出预注册候选集。

    候选 ID 基于参数内容哈希，而不是数组下标；因此即使显示顺序变化也能追溯
    到完全相同的一组策略参数。
    """

    if count < 1:
        raise ValueError("count must be positive")
    grid = parameter_grid or DEFAULT_PARAMETER_GRID
    if not grid:
        raise ValueError("parameter_grid cannot be empty")
    keys = tuple(sorted(grid))
    choices: list[tuple[Any, ...]] = []
    for key in keys:
        values = tuple(grid[key])
        if not values:
            raise ValueError(f"Parameter {key} has no candidate values")
        choices.append(values)
    combinations = [
        dict(zip(keys, values))
        for values in itertools.product(*choices)
    ]
    if count > len(combinations):
        raise ValueError(
            f"Requested {count} candidates from only {len(combinations)} combinations"
        )
    order = np.random.default_rng(seed).permutation(len(combinations))[:count]
    result: list[Candidate] = []
    for registration_index, source_index in enumerate(order):
        parameters = combinations[int(source_index)]
        digest = hashlib.sha256(_canonical(parameters).encode("utf-8")).hexdigest()[:16]
        result.append(
            Candidate(
                candidate_id=f"candidate-{digest}",
                parameters=parameters,
                preregistration_index=registration_index,
            )
        )
    return tuple(result)


def _metric_rows(
    symbol_metrics: Mapping[str, Mapping[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(symbol_metrics, pd.DataFrame):
        frame = symbol_metrics.copy()
        if "symbol" not in frame:
            frame.insert(0, "symbol", frame.index.astype(str))
    else:
        frame = pd.DataFrame.from_dict(symbol_metrics, orient="index")
        frame.insert(0, "symbol", frame.index.astype(str))
    if frame.empty:
        raise ValueError("symbol_metrics cannot be empty")
    return frame.reset_index(drop=True)


def aggregate_symbol_metrics(
    symbol_metrics: Mapping[str, Mapping[str, Any]] | pd.DataFrame,
    *,
    required_symbols: Iterable[str] | None = None,
) -> dict[str, Any]:
    """汇总跨标的中位数、下四分位和最差表现，避免单一标的主导结论。"""

    frame = _metric_rows(symbol_metrics)
    if required_symbols is not None:
        required = {str(value) for value in required_symbols}
        present = set(frame["symbol"].astype(str))
        missing = sorted(required - present)
        if missing:
            raise ValueError(f"Missing required symbols: {', '.join(missing)}")
        frame = frame[frame["symbol"].astype(str).isin(required)]
    metric_names = (
        "sharpe",
        "annualized_return",
        "max_drawdown",
        "turnover",
    )
    result: dict[str, Any] = {"symbol_count": len(frame)}
    for name in metric_names:
        if name not in frame:
            continue
        values = pd.to_numeric(frame[name], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        finite = values.dropna()
        result[f"{name}_median"] = (
            float(finite.median()) if len(finite) else float("nan")
        )
        result[f"{name}_lower_quartile"] = (
            float(finite.quantile(0.25)) if len(finite) else float("nan")
        )
        result[f"{name}_worst"] = (
            float(finite.min()) if len(finite) else float("nan")
        )
        result[f"{name}_coverage"] = len(finite) / len(frame)
    return result


def robust_multi_symbol_objective(
    symbol_metrics: Mapping[str, Mapping[str, Any]] | pd.DataFrame,
    *,
    required_symbols: Iterable[str] | None = None,
    minimum_coverage: float = 1.0,
    drawdown_soft_limit: float = 0.25,
) -> dict[str, Any]:
    """以跨标的广泛稳定性评分，而不是追逐某一证券的最高历史收益。"""

    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1]")
    summary = aggregate_symbol_metrics(
        symbol_metrics, required_symbols=required_symbols
    )
    required = ("sharpe", "annualized_return", "max_drawdown")
    incomplete = [
        name
        for name in required
        if summary.get(f"{name}_coverage", 0.0) < minimum_coverage
        or not np.isfinite(summary.get(f"{name}_median", np.nan))
    ]
    if incomplete:
        return {
            **summary,
            "available": False,
            "reason": "insufficient metric coverage: " + ", ".join(incomplete),
        }
    median_sharpe = float(summary["sharpe_median"])
    lower_sharpe = float(summary["sharpe_lower_quartile"])
    median_return = float(summary["annualized_return_median"])
    # Drawdowns are conventionally negative; penalize excess magnitude.
    worst_drawdown = abs(float(summary["max_drawdown_worst"]))
    drawdown_penalty = max(0.0, worst_drawdown - drawdown_soft_limit)
    turnover = float(summary.get("turnover_median", 0.0))
    if not np.isfinite(turnover):
        turnover = 0.0
    score = (
        0.50 * median_sharpe
        + 0.30 * lower_sharpe
        + 0.20 * median_return
        - drawdown_penalty
        - 0.01 * max(0.0, turnover)
    )
    return {
        **summary,
        "available": True,
        "objective": float(score),
        "drawdown_penalty": drawdown_penalty,
        "turnover_penalty": 0.01 * max(0.0, turnover),
    }


def parameter_perturbations(
    parameters: Mapping[str, Any],
    *,
    relative_step: float = 0.10,
    absolute_steps: Mapping[str, float] | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[dict[str, Any], ...]:
    """为每个参数生成单变量上下扰动，用于检查最优点附近是否稳定。"""

    if relative_step <= 0:
        raise ValueError("relative_step must be positive")
    absolute_steps = absolute_steps or {}
    bounds = bounds or {}
    probes: list[dict[str, Any]] = []
    for name in sorted(parameters):
        value = parameters[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        step = float(
            absolute_steps.get(
                name,
                abs(float(value)) * relative_step if float(value) != 0 else relative_step,
            )
        )
        for direction in (-1.0, 1.0):
            perturbed = dict(parameters)
            candidate = float(value) + direction * step
            if name in bounds:
                candidate = min(max(candidate, bounds[name][0]), bounds[name][1])
            if isinstance(value, int):
                candidate = int(round(candidate))
            if candidate == value:
                continue
            perturbed[name] = candidate
            probes.append(perturbed)
    return tuple(probes)


def perturbation_stability(
    base_score: float,
    perturbed_scores: Iterable[Any],
) -> dict[str, Any]:
    values = np.asarray(list(perturbed_scores), dtype=float)
    values = values[np.isfinite(values)]
    if not np.isfinite(base_score) or not len(values):
        return {
            "available": False,
            "reason": "finite base and perturbation scores are required",
        }
    denominator = max(abs(float(base_score)), 1e-12)
    degradation = (float(base_score) - values) / denominator
    return {
        "available": True,
        "perturbations": len(values),
        "median_relative_degradation": float(np.median(degradation)),
        "worst_relative_degradation": float(np.max(degradation)),
        "positive_fraction": float(np.mean(values > 0)),
    }


__all__ = [
    "Candidate",
    "DEFAULT_PARAMETER_GRID",
    "aggregate_symbol_metrics",
    "generate_preregistered_candidates",
    "parameter_perturbations",
    "perturbation_stability",
    "robust_multi_symbol_objective",
]
