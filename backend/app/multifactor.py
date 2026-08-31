from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from .factors.base import (
    Factor,
    FactorMetadata,
    FactorUnavailableError,
    assert_factor_is_causal,
)
from .factors.registry import FactorRegistry, factor_registry
from .json_utils import json_safe


# 多因子组合支持两种不可互换的比较口径：
# - cross_sectional：同一交易日横向比较不同股票，适用于选股；
# - time_series：同一股票与自身历史比较，适用于单标的择时。
# 缺失值策略会直接影响组合分母，因而也属于策略定义的一部分。
Normalization = Literal["auto", "cross_sectional", "rolling", "none"]
MissingPolicy = Literal["renormalize", "drop", "zero"]
CompositeMode = Literal["cross_sectional", "time_series"]


@dataclass(frozen=True, slots=True)
class FactorComponentConfig:
    factor_name: str
    weight: float = 1.0
    enabled: bool = True
    direction: int | None = None
    normalization: Normalization = "auto"
    winsorize: bool = True
    missing_policy: MissingPolicy = "renormalize"

    def __post_init__(self) -> None:
        if self.direction not in (None, -1, 1):
            raise ValueError("component direction must be -1, +1, or None")
        if not np.isfinite(self.weight):
            raise ValueError("component weight must be finite")


@dataclass(frozen=True, slots=True)
class MultiFactorConfig:
    name: str
    components: tuple[FactorComponentConfig, ...]
    mode: CompositeMode = "cross_sectional"
    rolling_window: int = 252
    rolling_min_periods: int = 120
    zscore_clip: float = 3.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        enabled = [
            item for item in self.components if item.enabled and item.weight != 0
        ]
        if not enabled:
            raise ValueError("at least one enabled nonzero factor is required")
        if self.rolling_window < 2:
            raise ValueError("rolling_window must be at least 2")
        if not 2 <= self.rolling_min_periods <= self.rolling_window:
            raise ValueError(
                "rolling_min_periods must be between 2 and rolling_window"
            )
        if self.zscore_clip <= 0:
            raise ValueError("zscore_clip must be positive")

    def snapshot(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["components"] = [asdict(item) for item in self.components]
        return payload

    @property
    def config_id(self) -> str:
        # 配置内容排序后再哈希，使相同策略在不同任务/不同机器上获得相同标识。
        # 该 ID 用于结果审计；它不是安全凭证。
        encoded = json.dumps(
            self.snapshot(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


def _cross_sectional_normalize(
    values: pd.Series,
    dates: pd.Series,
    winsorize: bool,
) -> pd.Series:
    """按交易日做截尾 Z-Score，消除不同因子的量纲差异。

    只读取当日全部股票在收盘时已经可得的值；不会使用未来日期的数据。
    方差为零说明当日无横截面区分度，统一记为 0 而不是产生无穷大。
    """
    output = pd.Series(np.nan, index=values.index, dtype=float)
    for _, indices in dates.groupby(dates, sort=True).groups.items():
        current = pd.to_numeric(values.loc[indices], errors="coerce")
        valid = current.dropna()
        if valid.empty:
            continue
        if winsorize:
            current = current.clip(
                lower=valid.quantile(0.01),
                upper=valid.quantile(0.99),
            )
        standard_deviation = current.std(ddof=0)
        if pd.isna(standard_deviation):
            continue
        output.loc[indices] = (
            current.where(current.isna(), 0.0)
            if standard_deviation == 0
            else (current - current.mean()) / standard_deviation
        )
    return output


def _rolling_normalize(
    values: pd.Series,
    symbols: pd.Series,
    window: int,
    min_periods: int,
    clip: float,
) -> pd.Series:
    """对每只证券使用截至 T-1 的滚动均值和标准差进行标准化。

    ``shift(1)`` 是关键：T 日因子值可以参与当天信号，但不能参与它自己的
    历史基准估计，否则会形成细微的前视偏差。
    """
    frame = pd.DataFrame(
        {
            "value": pd.to_numeric(values, errors="coerce"),
            "symbol": symbols.astype(str),
        },
        index=values.index,
    )
    mean = frame.groupby("symbol", sort=False)["value"].transform(
        lambda series: series.shift(1)
        .rolling(window, min_periods=min_periods)
        .mean()
    )
    standard_deviation = frame.groupby("symbol", sort=False)["value"].transform(
        lambda series: series.shift(1)
        .rolling(window, min_periods=min_periods)
        .std(ddof=0)
    )
    normalized = (frame["value"] - mean) / standard_deviation.where(
        standard_deviation.ne(0)
    )
    return normalized.clip(-clip, clip)


class CompositeFactor(Factor):
    def __init__(
        self,
        config: MultiFactorConfig,
        registry: FactorRegistry | None = None,
    ) -> None:
        self.config = config
        self.registry = registry or factor_registry
        factors = [
            self.registry.get(item.factor_name)
            for item in config.components
            if item.enabled and item.weight != 0
        ]
        requirements = tuple(
            dict.fromkeys(
                column
                for factor in factors
                for column in factor.metadata.required_columns
            )
        )
        self.metadata = FactorMetadata(
            name=f"multifactor_{config.config_id}",
            description=f"Composite factor: {config.name}",
            lookback=max(
                max(factor.metadata.lookback for factor in factors),
                config.rolling_window
                if config.mode == "time_series"
                else 0,
            ),
            required_columns=requirements,
            display_name=config.name,
            display_name_zh=config.name,
            description_zh=f"多因子组合：{config.name}",
            direction=1,
        )

    def compute_details(self, bars: pd.DataFrame) -> pd.DataFrame:
        """计算原始因子、标准化值、逐因子贡献及最终综合分。

        返回明细而不仅是综合分，前端才能解释一次买卖由哪些因子驱动；同时也
        便于检查某因子缺失、权重被重归一化等情况。
        """
        self.validate(bars)
        if not {"symbol", "date"}.issubset(bars.columns):
            raise ValueError("composite factors require symbol and date columns")
        result = bars.loc[:, ["symbol", "date"]].copy()
        weighted_columns: list[str] = []
        valid_weight_columns: list[str] = []
        drop_masks: list[pd.Series] = []

        for component in self.config.components:
            if not component.enabled or component.weight == 0:
                continue
            factor = self.registry.get(component.factor_name)
            factor.validate(bars)
            raw = factor.compute(bars)
            if not raw.index.equals(bars.index):
                raw = raw.reindex(bars.index)
            assert_factor_is_causal(factor, bars, raw)
            # 因子实现必须先通过截断重算因果性检查，之后才允许参加组合，
            # 避免某个单因子把未来行情泄露给整体策略。
            raw = pd.to_numeric(raw, errors="coerce").replace(
                [np.inf, -np.inf],
                np.nan,
            )
            normalization = component.normalization
            if normalization == "auto":
                normalization = (
                    "cross_sectional"
                    if self.config.mode == "cross_sectional"
                    else "rolling"
                )
            if normalization == "cross_sectional":
                normalized = _cross_sectional_normalize(
                    raw,
                    bars["date"],
                    component.winsorize,
                )
            elif normalization == "rolling":
                normalized = _rolling_normalize(
                    raw,
                    bars["symbol"],
                    self.config.rolling_window,
                    self.config.rolling_min_periods,
                    self.config.zscore_clip,
                )
            else:
                normalized = raw

            direction = (
                component.direction
                if component.direction is not None
                else factor.metadata.direction
            )
            contribution = normalized * direction * component.weight
            if component.missing_policy == "zero":
                # 明确选择 zero 时，缺失项当作没有贡献，仍保留完整权重。
                contribution = contribution.fillna(0.0)
                valid_weight = pd.Series(
                    abs(component.weight),
                    index=bars.index,
                    dtype=float,
                )
            else:
                # 默认 renormalize：只由当期可用因子构成分母，避免缺失被误判
                # 为低分；这也让不同日期的有效权重可被审计。
                valid_weight = normalized.notna().astype(float) * abs(
                    component.weight
                )
            if component.missing_policy == "drop":
                drop_masks.append(normalized.notna())

            result[f"factor_{component.factor_name}"] = raw
            result[f"normalized_{component.factor_name}"] = normalized
            contribution_column = f"contribution_{component.factor_name}"
            weight_column = f"valid_weight_{component.factor_name}"
            result[contribution_column] = contribution
            result[weight_column] = valid_weight
            weighted_columns.append(contribution_column)
            valid_weight_columns.append(weight_column)

        numerator = result[weighted_columns].sum(axis=1, min_count=1)
        denominator = result[valid_weight_columns].sum(axis=1)
        # 用绝对权重归一化，允许多头/反向因子混合且不改变整体分数尺度。
        result["composite_score"] = numerator / denominator.where(denominator.gt(0))
        if drop_masks:
            complete = pd.concat(drop_masks, axis=1).all(axis=1)
            result.loc[~complete, "composite_score"] = np.nan
        result["valid_weight"] = denominator
        if not result["composite_score"].notna().any():
            raise FactorUnavailableError(
                f"Composite '{self.config.name}' produced no usable scores. "
                f"Provide at least {self.metadata.lookback} prior trading "
                "sessions and all required point-in-time fields."
            )
        return result

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        return self.compute_details(bars)["composite_score"]


def factor_correlation_report(details: pd.DataFrame) -> dict[str, Any]:
    normalized_columns = [
        column for column in details.columns if column.startswith("normalized_")
    ]
    contribution_columns = [
        column for column in details.columns if column.startswith("contribution_")
    ]
    normalized = details.loc[:, normalized_columns].rename(
        columns=lambda value: value.removeprefix("normalized_")
    )
    contributions = details.loc[:, contribution_columns].rename(
        columns=lambda value: value.removeprefix("contribution_")
    )
    pearson = normalized.corr(method="pearson")
    rank = normalized.rank(method="average").corr(method="pearson")
    composite = pd.to_numeric(details["composite_score"], errors="coerce")
    score_correlations = {
        column: normalized[column].corr(composite)
        for column in normalized.columns
    }
    contribution_means = {
        column: contributions[column].abs().mean()
        for column in contributions.columns
    }
    duplicate_pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(pearson.columns):
        for right in pearson.columns[left_index + 1 :]:
            value = pearson.loc[left, right]
            if pd.notna(value) and abs(float(value)) > 0.8:
                duplicate_pairs.append(
                    {
                        "left": left,
                        "right": right,
                        "correlation": float(value),
                    }
                )
    return json_safe(
        {
            "pearson": pearson.to_dict(orient="index"),
            "rank": rank.to_dict(orient="index"),
            "score_correlations": score_correlations,
            "mean_absolute_contribution": contribution_means,
            "high_correlation_pairs": duplicate_pairs,
        }
    )


MULTIFACTOR_TEMPLATES: dict[str, dict[str, float]] = {
    "trend": {
        "momentum_20": 1.0,
        "momentum_60": 1.0,
        "momentum_252_21": 1.0,
        "ma_bias_20": 1.0,
        "price_position_252": 1.0,
        "relative_strength_60": 1.0,
    },
    "low_risk": {
        "volatility_20": 1.0,
        "downside_volatility_20": 1.0,
        "beta_252": 1.0,
        "idio_volatility_60": 1.0,
        "atr_ratio_20": 1.0,
        "max_return_20": 1.0,
    },
    "price_volume": {
        "volume_change_20": 1.0,
        "amount_surprise_20": 1.0,
        "volume_price_corr_20": 1.0,
        "amihud_20": 1.0,
    },
    "balanced": {
        # Trend sleeve: 40% split equally across six factors.
        "momentum_20": 0.0666667,
        "momentum_60": 0.0666667,
        "momentum_252_21": 0.0666667,
        "ma_bias_20": 0.0666667,
        "price_position_252": 0.0666667,
        "relative_strength_60": 0.0666667,
        # Low-risk sleeve: 30% split equally across six factors.
        "volatility_20": 0.05,
        "downside_volatility_20": 0.05,
        "beta_252": 0.05,
        "idio_volatility_60": 0.05,
        "atr_ratio_20": 0.05,
        "max_return_20": 0.05,
        # Price-volume sleeve: 15% split across three exploratory signals.
        "volume_change_20": 0.05,
        "amount_surprise_20": 0.05,
        "volume_price_corr_20": 0.05,
        # Liquidity sleeve: 15%.
        "amihud_20": 0.15,
    },
}


def template_config(
    template: str,
    mode: CompositeMode = "cross_sectional",
) -> MultiFactorConfig:
    try:
        weights = MULTIFACTOR_TEMPLATES[template]
    except KeyError as exc:
        raise KeyError(f"unknown multifactor template: {template}") from exc
    return MultiFactorConfig(
        name=template,
        mode=mode,
        components=tuple(
            FactorComponentConfig(factor_name=name, weight=weight)
            for name, weight in weights.items()
        ),
        metadata={"template": template},
    )
