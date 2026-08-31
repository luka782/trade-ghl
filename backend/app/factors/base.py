from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


class FactorUnavailableError(ValueError):
    """Raised when a factor's required point-in-time input is unavailable."""


@dataclass(frozen=True, slots=True)
class FactorMetadata:
    """因子的声明式元数据，也是 API 展示和运行前校验的依据。

    ``direction`` 只用于把“越小越好”的原始值转换为统一的排名方向，不会
    修改原始 IC，因此研究结果仍能看到因子本身与未来收益的真实关系。
    """
    name: str
    description: str
    lookback: int
    required_columns: tuple[str, ...]
    availability: str = "available"
    display_name: str | None = None
    display_name_zh: str | None = None
    description_zh: str | None = None
    direction: int = 1
    direction_kind: str | None = None
    applicable_assets: tuple[str, ...] = ("stock", "ETF")

    def __post_init__(self) -> None:
        if self.direction not in (-1, 1):
            raise ValueError("factor direction must be +1 or -1")
        direction_kind = self.direction_kind or (
            "positive" if self.direction == 1 else "negative"
        )
        if direction_kind not in ("positive", "negative", "exploratory"):
            raise ValueError(
                "factor direction_kind must be positive, negative, or exploratory"
            )
        if (
            direction_kind in ("positive", "negative")
            and (direction_kind == "positive") != (self.direction == 1)
        ):
            raise ValueError(
                "factor direction_kind must agree with the numeric direction"
            )
        if not self.applicable_assets:
            raise ValueError("factor applicable_assets must not be empty")
        invalid_assets = set(self.applicable_assets) - {"stock", "ETF"}
        if invalid_assets:
            raise ValueError(
                "factor applicable_assets may contain only stock and ETF"
            )
        object.__setattr__(self, "direction_kind", direction_kind)
        object.__setattr__(
            self, "applicable_assets", tuple(dict.fromkeys(self.applicable_assets))
        )
        object.__setattr__(self, "display_name", self.display_name or self.name)
        object.__setattr__(
            self,
            "display_name_zh",
            self.display_name_zh or self.display_name or self.name,
        )
        object.__setattr__(
            self,
            "description_zh",
            self.description_zh or self.description,
        )

    @property
    def direction_label(self) -> str:
        return "positive" if self.direction == 1 else "negative"


class Factor(ABC):
    metadata: FactorMetadata

    @abstractmethod
    def compute(self, bars: pd.DataFrame) -> pd.Series:
        """Return values indexed exactly like ``bars`` using data through row T only."""

    def validate(self, bars: pd.DataFrame) -> None:
        missing = [
            column
            for column in self.metadata.required_columns
            if column not in bars.columns
        ]
        if missing:
            raise FactorUnavailableError(
                f"Factor '{self.metadata.name}' is unavailable: missing "
                + ", ".join(missing)
            )


def assert_factor_is_causal(
    factor: Factor,
    bars: pd.DataFrame,
    full_values: pd.Series | None = None,
    sample_count: int = 5,
) -> None:
    """通过历史截断重算检测常见的未来数据泄露。

    对若干历史截止日删除后续行情，再比较该日的完整样本值与截断样本值。
    这能捕获 ``shift(-n)`` 等常见错误；它不能数学上证明任意插件绝对因果，
    但能阻止最危险的泄露静默进入因子研究和回测。
    """

    if bars.empty or sample_count < 1:
        return
    dates = pd.DatetimeIndex(pd.to_datetime(bars["date"]).drop_duplicates().sort_values())
    if len(dates) < 2:
        return
    values = full_values if full_values is not None else factor.compute(bars)
    if not values.index.equals(bars.index):
        values = values.reindex(bars.index)
    values = pd.to_numeric(values, errors="coerce")

    first_check = min(max(factor.metadata.lookback, 0), len(dates) - 2)
    check_indices = np.linspace(
        first_check,
        len(dates) - 2,
        num=min(sample_count, len(dates) - first_check - 1),
        dtype=int,
    )
    for index in sorted(set(check_indices.tolist())):
        cutoff = dates[index]
        truncated = bars[pd.to_datetime(bars["date"]) <= cutoff].copy()
        truncated_values = factor.compute(truncated)
        if not truncated_values.index.equals(truncated.index):
            truncated_values = truncated_values.reindex(truncated.index)
        current_indices = truncated.index[pd.to_datetime(truncated["date"]) == cutoff]
        full_current = pd.to_numeric(
            values.reindex(current_indices), errors="coerce"
        ).to_numpy(dtype=float)
        truncated_current = pd.to_numeric(
            truncated_values.reindex(current_indices), errors="coerce"
        ).to_numpy(dtype=float)
        if not np.allclose(
            full_current,
            truncated_current,
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(
                f"Factor '{factor.metadata.name}' failed the causality check at "
                f"{cutoff.date()}; its T value changes when future rows are removed."
            )


def build_factor_observations(factor: Factor, bars: pd.DataFrame) -> pd.DataFrame:
    """将原始行情转换为标准因子观测表，并在进入评价前完成质量/因果校验。"""
    factor.validate(bars)
    values = factor.compute(bars)
    if not values.index.equals(bars.index):
        values = values.reindex(bars.index)
    numeric_values = pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    if not numeric_values.notna().any():
        raise FactorUnavailableError(
            f"Factor '{factor.metadata.name}' produced no usable values. "
            f"It requires at least {factor.metadata.lookback} prior trading "
            "sessions per symbol and complete required fields."
        )
    assert_factor_is_causal(factor, bars, values)
    columns = ["symbol", "date"]
    columns.extend(
        column
        for column in ("industry", "market_cap")
        if column in bars.columns
    )
    observations = bars.loc[:, columns].copy()
    observations["factor"] = numeric_values
    return observations
