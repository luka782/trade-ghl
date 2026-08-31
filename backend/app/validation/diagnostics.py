from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


_NORMAL = NormalDist()
_EULER_GAMMA = 0.5772156649015329


# 以下诊断指标不用于制造“策略有效”的结论，而用于量化多次试参后，
# 最优历史指标可能只是偶然出现的程度。
def _finite(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def sharpe_ratio(
    returns: Iterable[Any],
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """按年化周期计算样本 Sharpe；波动为零或样本不足时不返回伪精确数值。"""
    values = _finite(returns)
    if len(values) < 2:
        return float("nan")
    excess = values - risk_free_rate / periods_per_year
    deviation = float(np.std(excess, ddof=1))
    if deviation <= 0:
        return float("nan")
    return float(np.mean(excess) / deviation * math.sqrt(periods_per_year))


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    observations: int,
    *,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability that an observed (non-annualized) Sharpe beats a benchmark."""

    if observations < 2:
        return float("nan")
    denominator_sq = (
        1.0
        - skewness * observed_sharpe
        + ((kurtosis - 1.0) / 4.0) * observed_sharpe**2
    )
    if not np.isfinite(denominator_sq) or denominator_sq <= 0:
        return float("nan")
    statistic = (
        (observed_sharpe - benchmark_sharpe)
        * math.sqrt(observations - 1)
        / math.sqrt(denominator_sq)
    )
    return float(_NORMAL.cdf(statistic))


def expected_maximum_sharpe(
    trial_sharpes: Iterable[Any],
    *,
    number_of_trials: int | None = None,
) -> float:
    """估计多次近似正态试验中仅由选择偏差产生的最佳 Sharpe 基准。"""

    trials = _finite(trial_sharpes)
    count = int(number_of_trials if number_of_trials is not None else len(trials))
    if count <= 1:
        return float(np.mean(trials)) if len(trials) else 0.0
    if len(trials) < 2:
        return 0.0
    mean = float(np.mean(trials))
    standard_deviation = float(np.std(trials, ddof=1))
    if standard_deviation <= 0:
        return mean
    first = _NORMAL.inv_cdf(1.0 - 1.0 / count)
    second = _NORMAL.inv_cdf(1.0 - 1.0 / (count * math.e))
    return mean + standard_deviation * (
        (1.0 - _EULER_GAMMA) * first + _EULER_GAMMA * second
    )


def deflated_sharpe_ratio(
    returns: Iterable[Any],
    *,
    trial_sharpes: Iterable[Any] | None = None,
    number_of_trials: int | None = None,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Compute the Deflated Sharpe Ratio with an auditable benchmark.

    Trial Sharpes are expected in annualized units. Internally all values are
    converted to per-period Sharpe for the finite-sample PSR formula.
    """

    values = _finite(returns)
    if len(values) < 3:
        return {
            "available": False,
            "reason": "at least 3 finite return observations are required",
        }
    deviation = float(np.std(values, ddof=1))
    if deviation <= 0:
        return {
            "available": False,
            "reason": "returns have zero variance",
        }
    observed = float(np.mean(values) / deviation)
    annualized = observed * math.sqrt(periods_per_year)
    supplied = _finite([] if trial_sharpes is None else trial_sharpes)
    trial_count = int(
        number_of_trials
        if number_of_trials is not None
        else max(1, len(supplied))
    )
    if trial_count < 1:
        raise ValueError("number_of_trials must be positive")
    benchmark_annualized = (
        expected_maximum_sharpe(supplied, number_of_trials=trial_count)
        if trial_count > 1 and len(supplied)
        else 0.0
    )
    centered = values - float(np.mean(values))
    second_moment = float(np.mean(centered**2))
    if second_moment <= 0:
        return {"available": False, "reason": "returns have zero variance"}
    skewness = float(np.mean(centered**3) / second_moment**1.5)
    kurtosis = float(np.mean(centered**4) / second_moment**2)
    ratio = probabilistic_sharpe_ratio(
        observed,
        benchmark_annualized / math.sqrt(periods_per_year),
        len(values),
        skewness=skewness,
        kurtosis=kurtosis,
    )
    return {
        "available": bool(np.isfinite(ratio)),
        "deflated_sharpe_ratio": ratio,
        "observed_sharpe": annualized,
        "expected_max_sharpe": benchmark_annualized,
        "number_of_trials": trial_count,
        "observations": len(values),
        "skewness": skewness,
        "kurtosis": kurtosis,
    }


def walk_forward_efficiency(
    in_sample: float | Iterable[Any],
    out_of_sample: float | Iterable[Any],
) -> dict[str, Any]:
    """计算样本外/样本内效率；多折数据使用中位数以降低异常折影响。"""

    in_values = _finite(
        [in_sample] if np.isscalar(in_sample) else in_sample  # type: ignore[arg-type]
    )
    out_values = _finite(
        [out_of_sample] if np.isscalar(out_of_sample) else out_of_sample  # type: ignore[arg-type]
    )
    if not len(in_values) or not len(out_values):
        return {"available": False, "reason": "finite IS and OOS metrics are required"}
    in_metric = float(np.median(in_values))
    out_metric = float(np.median(out_values))
    if abs(in_metric) <= 1e-12:
        return {"available": False, "reason": "IS metric is zero"}
    return {
        "available": True,
        "efficiency": out_metric / in_metric,
        "in_sample_metric": in_metric,
        "out_of_sample_metric": out_metric,
        "in_sample_folds": len(in_values),
        "out_of_sample_folds": len(out_values),
    }


def _strategy_score(values: np.ndarray) -> np.ndarray:
    means = np.nanmean(values, axis=0)
    deviations = np.nanstd(values, axis=0, ddof=1)
    return np.divide(
        means,
        deviations,
        out=np.full_like(means, np.nan, dtype=float),
        where=deviations > 0,
    )


def cscv_pbo(
    strategy_returns: pd.DataFrame | np.ndarray,
    *,
    partitions: int = 8,
) -> dict[str, Any]:
    """用组合对称交叉验证估计回测过拟合概率（PBO）。

    每次用一半时间块挑选最优候选，再到互补时间块检查其排名。若最优候选在
    留出块常落入后半名，PBO 会升高，提示参数选择缺乏稳健性。
    """

    if partitions < 4 or partitions % 2:
        raise ValueError("partitions must be an even integer of at least 4")
    frame = (
        strategy_returns.copy()
        if isinstance(strategy_returns, pd.DataFrame)
        else pd.DataFrame(np.asarray(strategy_returns, dtype=float))
    )
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    observations, strategies = frame.shape
    if strategies < 2:
        return {
            "available": False,
            "reason": "CSCV/PBO requires at least 2 strategies",
        }
    if observations < partitions * 2:
        return {
            "available": False,
            "reason": (
                f"CSCV/PBO requires at least {partitions * 2} observations "
                f"for {partitions} partitions"
            ),
        }
    blocks = [block for block in np.array_split(np.arange(observations), partitions)]
    combinations = list(itertools.combinations(range(partitions), partitions // 2))
    logits: list[float] = []
    degradations: list[float] = []
    for selected in combinations:
        selected_set = set(selected)
        train_rows = np.concatenate([blocks[index] for index in selected])
        test_rows = np.concatenate(
            [blocks[index] for index in range(partitions) if index not in selected_set]
        )
        train_scores = _strategy_score(frame.iloc[train_rows].to_numpy(dtype=float))
        test_scores = _strategy_score(frame.iloc[test_rows].to_numpy(dtype=float))
        finite_train = np.flatnonzero(np.isfinite(train_scores))
        finite_test = np.flatnonzero(np.isfinite(test_scores))
        if not len(finite_train) or len(finite_test) < 2:
            continue
        winner = int(
            max(finite_train, key=lambda index: (train_scores[index], -index))
        )
        if not np.isfinite(test_scores[winner]):
            continue
        # Average ranks use one=worst. The half-step prevents infinite logits.
        rank = float(
            pd.Series(test_scores).rank(method="average", na_option="keep").iloc[winner]
        )
        omega = min(max((rank - 0.5) / len(finite_test), 1e-12), 1.0 - 1e-12)
        logits.append(math.log(omega / (1.0 - omega)))
        degradations.append(float(train_scores[winner] - test_scores[winner]))
    if len(logits) < 2:
        return {
            "available": False,
            "reason": "insufficient finite symmetric CSCV splits",
            "valid_splits": len(logits),
        }
    return {
        "available": True,
        "pbo": float(np.mean(np.asarray(logits) <= 0.0)),
        "median_logit": float(np.median(logits)),
        "median_performance_degradation": float(np.median(degradations)),
        "valid_splits": len(logits),
        "partitions": partitions,
        "strategies": strategies,
        "observations": observations,
    }


__all__ = [
    "cscv_pbo",
    "deflated_sharpe_ratio",
    "expected_maximum_sharpe",
    "probabilistic_sharpe_ratio",
    "sharpe_ratio",
    "walk_forward_efficiency",
]
