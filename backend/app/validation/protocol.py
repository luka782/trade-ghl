from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType
from typing import Any

import pandas as pd


# 验证协议在运行前固化，而不是从结果倒推参数。规范 JSON 和 SHA-256 让同一
# 协议可跨机器核验，且防止任务运行后悄悄修改样本区间或候选空间。
def _timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tz is not None:
        result = result.tz_localize(None)
    return result.normalize()


def _json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, (date,)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def canonical_json(value: Any) -> str:
    """将协议序列化为确定性 JSON，作为审计哈希的唯一输入。"""

    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True, slots=True)
class ProtocolSnapshot(Mapping[str, Any]):
    """不可变、规范序列化的研究协议记录及其 SHA-256 指纹。"""

    payload: Mapping[str, Any]
    canonical: str
    sha256: str

    @classmethod
    def create(cls, payload: Mapping[str, Any]) -> "ProtocolSnapshot":
        normalized = _json_value(payload)
        canonical = canonical_json(normalized)
        return cls(
            payload=_freeze(normalized),
            canonical=canonical,
            sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.canonical)

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def __iter__(self):
        return iter(self.payload)

    def __len__(self) -> int:
        return len(self.payload)

    @property
    def hash(self) -> str:
        return self.sha256


@dataclass(frozen=True, slots=True)
class EvaluationPeriod:
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp
    locked_oos_start: pd.Timestamp
    locked_oos_end: pd.Timestamp

    def __post_init__(self) -> None:
        for name in (
            "evaluation_start",
            "evaluation_end",
            "locked_oos_start",
            "locked_oos_end",
        ):
            object.__setattr__(self, name, _timestamp(getattr(self, name)))
        if not (
            self.evaluation_start
            <= self.locked_oos_start
            <= self.locked_oos_end
            <= self.evaluation_end
        ):
            raise ValueError("Evaluation and locked OOS dates are inconsistent")


@dataclass(frozen=True, slots=True)
class ValidationProtocol:
    """Preregistered, immutable walk-forward protocol.

    The evaluation window is exactly the most recent 36 complete calendar
    months shared by every symbol. Its final 12 complete months are a locked,
    final OOS interval and are never eligible for fitting or candidate choice.
    """

    symbols: tuple[str, ...]
    evaluation_start: date
    evaluation_end: date
    locked_oos_start: date
    locked_oos_end: date
    train_sessions: int = 126
    validation_sessions: int = 42
    test_sessions: int = 42
    step_sessions: int = 42
    purge_sessions: int = 5
    embargo_sessions: int = 5
    candidate_count: int = 96
    candidate_seed: int = 20260825
    objective: str = "robust_multi_symbol"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        symbols = tuple(dict.fromkeys(str(item) for item in self.symbols))
        if len(symbols) < 1:
            raise ValueError("At least one symbol is required")
        object.__setattr__(self, "symbols", symbols)
        period = EvaluationPeriod(
            self.evaluation_start,
            self.evaluation_end,
            self.locked_oos_start,
            self.locked_oos_end,
        )
        object.__setattr__(self, "evaluation_start", period.evaluation_start.date())
        object.__setattr__(self, "evaluation_end", period.evaluation_end.date())
        object.__setattr__(self, "locked_oos_start", period.locked_oos_start.date())
        object.__setattr__(self, "locked_oos_end", period.locked_oos_end.date())
        for name in (
            "train_sessions",
            "validation_sessions",
            "test_sessions",
            "step_sessions",
            "candidate_count",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("purge_sessions", "embargo_sessions"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        object.__setattr__(self, "metadata", _freeze(dict(self.metadata)))

    def snapshot(self) -> ProtocolSnapshot:
        payload = {
            "symbols": self.symbols,
            "evaluation_start": self.evaluation_start,
            "evaluation_end": self.evaluation_end,
            "locked_oos_start": self.locked_oos_start,
            "locked_oos_end": self.locked_oos_end,
            "train_sessions": self.train_sessions,
            "validation_sessions": self.validation_sessions,
            "test_sessions": self.test_sessions,
            "step_sessions": self.step_sessions,
            "purge_sessions": self.purge_sessions,
            "embargo_sessions": self.embargo_sessions,
            "candidate_count": self.candidate_count,
            "candidate_seed": self.candidate_seed,
            "objective": self.objective,
            "metadata": dict(self.metadata),
        }
        return ProtocolSnapshot.create(payload)

    @property
    def protocol_hash(self) -> str:
        return self.snapshot().sha256

    @property
    def snapshot_hash(self) -> str:
        return self.protocol_hash


def common_recent_evaluation_period(
    symbol_dates: Mapping[str, Iterable[Any]],
    *,
    as_of: Any | None = None,
    evaluation_months: int = 36,
    locked_oos_months: int = 12,
) -> EvaluationPeriod:
    """返回全部标的共有的最近完整月评价区间。

    若任一标的缺少所需历史，函数直接失败而非静默缩短区间；这样可避免不同
    标的在不同时间段上被不公平地比较。
    `as_of` defaults to today; the current, incomplete calendar month is always
    excluded. Boundaries are the first/last shared trading observations inside
    the requested complete calendar months.
    """

    if not symbol_dates:
        raise ValueError("symbol_dates cannot be empty")
    if evaluation_months < 1:
        raise ValueError("evaluation_months must be positive")
    if not 1 <= locked_oos_months < evaluation_months:
        raise ValueError("locked_oos_months must be within evaluation months")

    normalized: list[pd.DatetimeIndex] = []
    for symbol, values in symbol_dates.items():
        index = pd.DatetimeIndex(pd.to_datetime(list(values), errors="coerce"))
        if index.tz is not None:
            index = index.tz_localize(None)
        index = index.dropna().normalize().unique().sort_values()
        if index.empty:
            raise ValueError(f"No valid dates for symbol {symbol}")
        normalized.append(index)

    common = normalized[0]
    for index in normalized[1:]:
        common = common.intersection(index)
    if common.empty:
        raise ValueError("Symbols have no shared trading dates")

    cutoff = _timestamp(as_of if as_of is not None else pd.Timestamp.today())
    last_complete_month_end = cutoff.to_period("M").start_time - pd.Timedelta(days=1)
    eligible = common[common <= last_complete_month_end]
    if eligible.empty:
        raise ValueError("No shared observations in a complete calendar month")

    end_month = eligible[-1].to_period("M")
    calendar_start = (end_month - (evaluation_months - 1)).start_time
    calendar_end = end_month.end_time.normalize()
    in_period = common[(common >= calendar_start) & (common <= calendar_end)]
    if in_period.empty or any(index[0] > calendar_start for index in normalized):
        raise ValueError(
            f"All symbols need {evaluation_months} complete months of common history"
        )

    oos_calendar_start = (
        end_month - (locked_oos_months - 1)
    ).start_time
    oos = in_period[in_period >= oos_calendar_start]
    if oos.empty:
        raise ValueError("Locked OOS interval has no shared observations")
    return EvaluationPeriod(
        evaluation_start=in_period[0],
        evaluation_end=in_period[-1],
        locked_oos_start=oos[0],
        locked_oos_end=in_period[-1],
    )


def development_mask(
    dates: Iterable[Any],
    locked_oos_start: Any,
) -> pd.Series:
    """Boolean mask that strictly excludes every locked OOS observation."""

    values = pd.Series(pd.to_datetime(list(dates), errors="coerce"))
    if values.isna().any():
        raise ValueError("dates contain invalid values")
    if values.dt.tz is not None:
        values = values.dt.tz_localize(None)
    return values.dt.normalize().lt(_timestamp(locked_oos_start))


def assert_locked_oos_excluded(
    dates: Iterable[Any],
    locked_oos_start: Any,
    *,
    context: str = "development data",
) -> None:
    values = pd.DatetimeIndex(pd.to_datetime(list(dates), errors="coerce"))
    if values.isna().any():
        raise ValueError("dates contain invalid values")
    if values.tz is not None:
        values = values.tz_localize(None)
    offending = values.normalize()[values.normalize() >= _timestamp(locked_oos_start)]
    if len(offending):
        raise ValueError(
            f"{context} includes {len(offending)} locked OOS observation(s), "
            f"starting at {offending.min().date().isoformat()}"
        )


__all__ = [
    "EvaluationPeriod",
    "ProtocolSnapshot",
    "ValidationProtocol",
    "assert_locked_oos_excluded",
    "canonical_json",
    "common_recent_evaluation_period",
    "development_mask",
]
