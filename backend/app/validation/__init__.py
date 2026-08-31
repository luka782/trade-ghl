"""Read-only validation primitives, independent of API and storage layers."""

from .benchmarks import (
    baseline_metric_helpers,
    buy_and_hold_returns,
    causal_moving_average_returns,
    equal_weight_returns,
    relative_baseline_metrics,
    return_metrics,
)
from .diagnostics import (
    cscv_pbo,
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    walk_forward_efficiency,
)
from .protocol import (
    EvaluationPeriod,
    ProtocolSnapshot,
    ValidationProtocol,
    assert_locked_oos_excluded,
    canonical_json,
    common_recent_evaluation_period,
    development_mask,
)
from .search import (
    Candidate,
    DEFAULT_PARAMETER_GRID,
    aggregate_symbol_metrics,
    generate_preregistered_candidates,
    parameter_perturbations,
    perturbation_stability,
    robust_multi_symbol_objective,
)
from .walk_forward import (
    TrainOnlyStandardScaler,
    WalkForwardFold,
    fit_train_only_scaler,
    fold_frames,
    generate_rolling_folds,
    run_walk_forward,
)

__all__ = [
    "Candidate",
    "DEFAULT_PARAMETER_GRID",
    "EvaluationPeriod",
    "ProtocolSnapshot",
    "TrainOnlyStandardScaler",
    "ValidationProtocol",
    "WalkForwardFold",
    "aggregate_symbol_metrics",
    "assert_locked_oos_excluded",
    "baseline_metric_helpers",
    "buy_and_hold_returns",
    "canonical_json",
    "causal_moving_average_returns",
    "common_recent_evaluation_period",
    "cscv_pbo",
    "deflated_sharpe_ratio",
    "development_mask",
    "equal_weight_returns",
    "expected_maximum_sharpe",
    "fit_train_only_scaler",
    "fold_frames",
    "generate_preregistered_candidates",
    "generate_rolling_folds",
    "parameter_perturbations",
    "perturbation_stability",
    "probabilistic_sharpe_ratio",
    "relative_baseline_metrics",
    "return_metrics",
    "robust_multi_symbol_objective",
    "run_walk_forward",
    "sharpe_ratio",
    "walk_forward_efficiency",
]
