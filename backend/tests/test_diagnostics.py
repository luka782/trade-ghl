from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.validation import (
    baseline_metric_helpers,
    causal_moving_average_returns,
    cscv_pbo,
    deflated_sharpe_ratio,
    parameter_perturbations,
    perturbation_stability,
    probabilistic_sharpe_ratio,
    robust_multi_symbol_objective,
    walk_forward_efficiency,
)


def test_probabilistic_and_deflated_sharpe_known_fixtures() -> None:
    assert probabilistic_sharpe_ratio(0.2, 0.2, 100) == pytest.approx(0.5)
    returns = np.tile([0.012, -0.004, 0.009, -0.002, 0.006], 60)

    diagnostic = deflated_sharpe_ratio(
        returns,
        trial_sharpes=[-0.2, 0.0, 0.2, 0.4, 0.6],
        number_of_trials=96,
    )

    assert diagnostic["available"] is True
    assert 0.0 <= diagnostic["deflated_sharpe_ratio"] <= 1.0
    assert diagnostic["observed_sharpe"] > diagnostic["expected_max_sharpe"]
    assert diagnostic["number_of_trials"] == 96


def test_deflated_sharpe_reports_unavailable_reason() -> None:
    result = deflated_sharpe_ratio([0.01, 0.01, 0.01], number_of_trials=96)
    assert result == {
        "available": False,
        "reason": "returns have zero variance",
    }


def test_cscv_pbo_fixture_is_deterministic_and_bounded() -> None:
    rng = np.random.default_rng(42)
    observations = 160
    returns = pd.DataFrame(
        {
            "stable": 0.001 + rng.normal(0.0, 0.01, observations),
            "noise_a": rng.normal(0.0, 0.012, observations),
            "noise_b": rng.normal(0.0, 0.012, observations),
            "reversal": np.r_[
                np.full(observations // 2, 0.004),
                np.full(observations // 2, -0.004),
            ]
            + rng.normal(0.0, 0.008, observations),
        }
    )

    first = cscv_pbo(returns, partitions=8)
    second = cscv_pbo(returns, partitions=8)

    assert first == second
    assert first["available"] is True
    assert 0.0 <= first["pbo"] <= 1.0
    assert first["valid_splits"] > 1
    assert first["strategies"] == 4


def test_cscv_pbo_has_explicit_unavailable_reason() -> None:
    result = cscv_pbo(pd.DataFrame({"only": np.arange(20)}))
    assert result["available"] is False
    assert "at least 2 strategies" in result["reason"]


def test_walk_forward_efficiency_uses_fold_medians() -> None:
    result = walk_forward_efficiency([2.0, 1.0, 3.0], [1.0, 0.5, 1.5])
    assert result["available"] is True
    assert result["in_sample_metric"] == pytest.approx(2.0)
    assert result["out_of_sample_metric"] == pytest.approx(1.0)
    assert result["efficiency"] == pytest.approx(0.5)


def test_robust_objective_requires_all_symbols_and_penalizes_drawdown() -> None:
    metrics = {
        symbol: {
            "sharpe": 1.0 + index * 0.1,
            "annualized_return": 0.12 + index * 0.01,
            "max_drawdown": -0.15 - index * 0.02,
            "turnover": 1.0,
        }
        for index, symbol in enumerate(("A", "B", "C", "D"))
    }
    result = robust_multi_symbol_objective(
        metrics,
        required_symbols=("A", "B", "C", "D"),
    )
    worse = {
        symbol: {**values, "max_drawdown": -0.60}
        for symbol, values in metrics.items()
    }

    assert result["available"] is True
    assert result["symbol_count"] == 4
    assert robust_multi_symbol_objective(worse)["objective"] < result["objective"]
    with pytest.raises(ValueError, match="Missing required symbols"):
        robust_multi_symbol_objective(metrics, required_symbols=("A", "E"))


def test_parameter_perturbation_fixture_and_stability() -> None:
    probes = parameter_perturbations(
        {"threshold": 0.5, "holding": 20, "style": "factor_dual"},
        relative_step=0.10,
        bounds={"threshold": (0.0, 1.0), "holding": (1, 100)},
    )

    assert probes == (
        {"threshold": 0.5, "holding": 18, "style": "factor_dual"},
        {"threshold": 0.5, "holding": 22, "style": "factor_dual"},
        {"threshold": 0.45, "holding": 20, "style": "factor_dual"},
        {"threshold": 0.55, "holding": 20, "style": "factor_dual"},
    )
    stability = perturbation_stability(1.0, [0.9, 1.1, 0.8, 1.0])
    assert stability["available"] is True
    assert stability["median_relative_degradation"] == pytest.approx(0.05)
    assert stability["worst_relative_degradation"] == pytest.approx(0.2)


def test_causal_baseline_does_not_change_when_future_prices_are_appended() -> None:
    prices = pd.Series(np.linspace(100.0, 140.0, 30))
    prefix = causal_moving_average_returns(prices.iloc[:20], window=5)
    full = causal_moving_average_returns(prices, window=5)
    metrics = baseline_metric_helpers(
        {"A": prices, "B": prices * 1.05},
        moving_average_window=5,
    )

    pd.testing.assert_series_equal(
        prefix.reset_index(drop=True),
        full.iloc[:20].reset_index(drop=True),
    )
    assert set(metrics) == {
        "cash",
        "buy_and_hold",
        "equal_weight_buy_and_hold",
        "equal_weight_moving_average",
    }
