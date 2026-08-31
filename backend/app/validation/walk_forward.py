from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from .protocol import assert_locked_oos_excluded


# 本模块把时间序列验证拆成互不重叠的训练、验证、测试区间。它不负责选参数，
# 而负责保证选参过程不接触最终锁定的样本外数据。
def _dates(values: Iterable[Any]) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(list(values), errors="coerce"))
    if index.isna().any():
        raise ValueError("Trading dates contain invalid values")
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.normalize().unique().sort_values()


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """一个按时间严格递进的 Walk-Forward 折叠及其审计边界。"""
    fold_id: int
    train_dates: tuple[pd.Timestamp, ...]
    validation_dates: tuple[pd.Timestamp, ...]
    test_dates: tuple[pd.Timestamp, ...]
    purge_dates: tuple[pd.Timestamp, ...] = ()
    embargo_dates: tuple[pd.Timestamp, ...] = ()

    def __post_init__(self) -> None:
        groups = {
            "train": self.train_dates,
            "purge": self.purge_dates,
            "validation": self.validation_dates,
            "embargo": self.embargo_dates,
            "test": self.test_dates,
        }
        if not self.train_dates or not self.validation_dates or not self.test_dates:
            raise ValueError("Train, validation, and test intervals cannot be empty")
        sets = {name: set(values) for name, values in groups.items()}
        names = tuple(groups)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                if sets[left].intersection(sets[right]):
                    raise ValueError(f"Fold {self.fold_id} has overlapping {left}/{right}")
        ordered = [
            values
            for values in (
                self.train_dates,
                self.purge_dates,
                self.validation_dates,
                self.embargo_dates,
                self.test_dates,
            )
            if values
        ]
        if any(left[-1] >= right[0] for left, right in zip(ordered, ordered[1:])):
            raise ValueError(f"Fold {self.fold_id} intervals are not chronological")

    @property
    def train_start(self) -> pd.Timestamp:
        return self.train_dates[0]

    @property
    def train_end(self) -> pd.Timestamp:
        return self.train_dates[-1]

    @property
    def validation_start(self) -> pd.Timestamp:
        return self.validation_dates[0]

    @property
    def validation_end(self) -> pd.Timestamp:
        return self.validation_dates[-1]

    @property
    def test_start(self) -> pd.Timestamp:
        return self.test_dates[0]

    @property
    def test_end(self) -> pd.Timestamp:
        return self.test_dates[-1]

    def snapshot(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "train_start": self.train_start.date().isoformat(),
            "train_end": self.train_end.date().isoformat(),
            "validation_start": self.validation_start.date().isoformat(),
            "validation_end": self.validation_end.date().isoformat(),
            "test_start": self.test_start.date().isoformat(),
            "test_end": self.test_end.date().isoformat(),
            "train_sessions": len(self.train_dates),
            "validation_sessions": len(self.validation_dates),
            "test_sessions": len(self.test_dates),
            "purge_sessions": len(self.purge_dates),
            "embargo_sessions": len(self.embargo_dates),
        }


def generate_rolling_folds(
    trading_dates: Iterable[Any],
    *,
    train_sessions: int,
    validation_sessions: int,
    test_sessions: int,
    step_sessions: int | None = None,
    purge_sessions: int = 0,
    embargo_sessions: int = 0,
    locked_oos_start: Any | None = None,
    expanding_train: bool = False,
) -> tuple[WalkForwardFold, ...]:
    """生成位于最终样本外区间之前的滚动训练/验证/测试窗口。

    purge 在训练和验证间留出标签持有期，embargo 在验证和测试间再隔离一段
    时间，降低重叠收益标签造成的信息泄露。锁定 OOS 日期绝不会出现在任一折中。
    """

    for name, value in (
        ("train_sessions", train_sessions),
        ("validation_sessions", validation_sessions),
        ("test_sessions", test_sessions),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if purge_sessions < 0 or embargo_sessions < 0:
        raise ValueError("purge_sessions and embargo_sessions must be non-negative")
    step = test_sessions if step_sessions is None else step_sessions
    if step < 1:
        raise ValueError("step_sessions must be positive")

    dates = _dates(trading_dates)
    if locked_oos_start is not None:
        lock = pd.Timestamp(locked_oos_start).normalize()
        dates = dates[dates < lock]
    required = (
        train_sessions
        + purge_sessions
        + validation_sessions
        + embargo_sessions
        + test_sessions
    )
    folds: list[WalkForwardFold] = []
    anchor = 0
    while anchor + required <= len(dates):
        train_start = 0 if expanding_train else anchor
        train_end = anchor + train_sessions
        purge_end = train_end + purge_sessions
        validation_end = purge_end + validation_sessions
        embargo_end = validation_end + embargo_sessions
        test_end = embargo_end + test_sessions
        fold = WalkForwardFold(
            fold_id=len(folds),
            train_dates=tuple(dates[train_start:train_end]),
            purge_dates=tuple(dates[train_end:purge_end]),
            validation_dates=tuple(dates[purge_end:validation_end]),
            embargo_dates=tuple(dates[validation_end:embargo_end]),
            test_dates=tuple(dates[embargo_end:test_end]),
        )
        if locked_oos_start is not None:
            assert_locked_oos_excluded(
                (*fold.train_dates, *fold.validation_dates, *fold.test_dates),
                locked_oos_start,
                context=f"fold {fold.fold_id}",
            )
        folds.append(fold)
        anchor += step
    if not folds:
        raise ValueError(
            f"Insufficient pre-OOS history: need at least {required} sessions"
        )
    return tuple(folds)


@dataclass(frozen=True, slots=True)
class TrainOnlyStandardScaler:
    """仅用训练集拟合、并记录拟合日期的不可变标准化器。"""

    means: Mapping[str, float]
    scales: Mapping[str, float]
    fit_dates: tuple[pd.Timestamp, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "means", MappingProxyType(dict(self.means)))
        object.__setattr__(self, "scales", MappingProxyType(dict(self.scales)))
        if set(self.means) != set(self.scales):
            raise ValueError("means and scales must have identical features")
        if any(not np.isfinite(value) for value in self.means.values()):
            raise ValueError("Scaler means must be finite")
        if any(not np.isfinite(value) or value <= 0 for value in self.scales.values()):
            raise ValueError("Scaler scales must be finite and positive")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.means)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [name for name in self.feature_names if name not in frame]
        if missing:
            raise ValueError(f"Missing scaler features: {', '.join(missing)}")
        result = frame.copy()
        for name in self.feature_names:
            values = pd.to_numeric(result[name], errors="coerce")
            result[name] = (values - self.means[name]) / self.scales[name]
        return result


def fit_train_only_scaler(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    train_dates: Iterable[Any],
    *,
    date_column: str = "date",
) -> TrainOnlyStandardScaler:
    """仅使用当前折训练日期拟合均值/标准差，禁止验证或测试数据参与。"""

    if date_column not in frame:
        raise ValueError(f"Missing date column: {date_column}")
    if not feature_columns:
        raise ValueError("feature_columns cannot be empty")
    missing = [name for name in feature_columns if name not in frame]
    if missing:
        raise ValueError(f"Missing features: {', '.join(missing)}")
    allowed = _dates(train_dates)
    frame_dates = pd.to_datetime(frame[date_column], errors="coerce")
    if frame_dates.isna().any():
        raise ValueError("Frame contains invalid dates")
    if frame_dates.dt.tz is not None:
        frame_dates = frame_dates.dt.tz_localize(None)
    mask = frame_dates.dt.normalize().isin(allowed)
    train = frame.loc[mask, list(feature_columns)].apply(
        pd.to_numeric, errors="coerce"
    )
    if train.empty:
        raise ValueError("No frame rows match train_dates")
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in feature_columns:
        values = train[name].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            raise ValueError(f"Feature {name} has no finite training values")
        means[name] = float(values.mean())
        scale = float(values.std(ddof=0))
        scales[name] = scale if np.isfinite(scale) and scale > 0 else 1.0
    return TrainOnlyStandardScaler(
        means=means,
        scales=scales,
        fit_dates=tuple(allowed),
    )


def fold_frames(
    frame: pd.DataFrame,
    fold: WalkForwardFold,
    *,
    date_column: str = "date",
) -> Mapping[str, pd.DataFrame]:
    """Materialize disjoint train/validation/test frames for a fold."""

    if date_column not in frame:
        raise ValueError(f"Missing date column: {date_column}")
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    if dates.isna().any():
        raise ValueError("Frame contains invalid dates")
    normalized = dates.dt.tz_localize(None) if dates.dt.tz is not None else dates
    normalized = normalized.dt.normalize()
    result = {
        name: frame.loc[normalized.isin(values)].copy()
        for name, values in (
            ("train", fold.train_dates),
            ("validation", fold.validation_dates),
            ("test", fold.test_dates),
        )
    }
    return MappingProxyType(result)


def run_walk_forward(
    frame: pd.DataFrame,
    folds: Sequence[WalkForwardFold],
    candidates: Sequence[Any],
    evaluate: Callable[[Any, pd.DataFrame, pd.DataFrame, str], float],
    *,
    date_column: str = "date",
) -> tuple[dict[str, Any], ...]:
    """Generic candidate selection on validation, followed by untouched test.

    The callback receives `(candidate, train_frame, evaluation_frame, phase)`.
    Test is called exactly once per fold, only for the validation winner.
    """

    if not candidates:
        raise ValueError("At least one candidate is required")
    results: list[dict[str, Any]] = []
    for fold in folds:
        parts = fold_frames(frame, fold, date_column=date_column)
        validation_scores = [
            float(evaluate(item, parts["train"], parts["validation"], "validation"))
            for item in candidates
        ]
        finite = np.isfinite(validation_scores)
        if not finite.any():
            raise ValueError(f"Fold {fold.fold_id} has no finite candidate score")
        winner_index = max(
            (index for index, valid in enumerate(finite) if valid),
            key=lambda index: (validation_scores[index], -index),
        )
        test_score = float(
            evaluate(
                candidates[winner_index],
                parts["train"],
                parts["test"],
                "test",
            )
        )
        results.append(
            {
                "fold_id": fold.fold_id,
                "winner_index": winner_index,
                "winner": candidates[winner_index],
                "validation_score": validation_scores[winner_index],
                "test_score": test_score,
                "fold": fold.snapshot(),
            }
        )
    return tuple(results)


__all__ = [
    "TrainOnlyStandardScaler",
    "WalkForwardFold",
    "fit_train_only_scaler",
    "fold_frames",
    "generate_rolling_folds",
    "run_walk_forward",
]
