from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """横截面因子预处理开关；全部操作都按单个交易日独立执行。"""
    winsorize: bool = True
    zscore: bool = True
    industry_neutralize: bool = False
    market_cap_neutralize: bool = False
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99

    def __post_init__(self) -> None:
        if not 0 <= self.winsor_lower < self.winsor_upper <= 1:
            raise ValueError("winsor bounds must satisfy 0 <= lower < upper <= 1")


def preprocess_factor(
    observations: pd.DataFrame, config: PreprocessConfig
) -> tuple[pd.DataFrame, dict[str, object]]:
    """对每个交易日独立去极值、中性化及标准化，返回处理审计信息。

    不跨日期拟合参数，所以不会让未来截面的分布影响 T 日因子值。
    """

    result = observations.copy()
    result["factor"] = pd.to_numeric(result["factor"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    warnings: list[str] = []

    if config.winsorize:
        # 以当日 1%/99% 分位截断极端值，降低单只异常证券对排序的影响。
        for _, indices in result.groupby("date", sort=True).groups.items():
            values = result.loc[indices, "factor"]
            valid = values.dropna()
            if valid.empty:
                continue
            lower = valid.quantile(config.winsor_lower)
            upper = valid.quantile(config.winsor_upper)
            result.loc[indices, "factor"] = values.clip(lower=lower, upper=upper)

    neutral_columns: list[str] = []
    if config.industry_neutralize:
        if "industry" in result.columns and result["industry"].notna().any():
            neutral_columns.append("industry")
        else:
            warnings.append(
                "Industry neutralization was requested but no industry field exists; skipped."
            )
    if config.market_cap_neutralize:
        if "market_cap" in result.columns and result["market_cap"].notna().any():
            neutral_columns.append("market_cap")
        else:
            warnings.append(
                "Market-cap neutralization was requested but no market_cap field exists; skipped."
            )

    skipped_dates = 0
    if neutral_columns:
        # 截面回归后的残差代表剔除行业/规模暴露后的因子值。样本数不足以支持
        # 回归时宁可跳过并记录警告，也不输出不可靠的残差。
        for _, indices in result.groupby("date", sort=True).groups.items():
            group = result.loc[indices]
            valid = group["factor"].notna()
            design_parts: list[pd.DataFrame] = []

            if "industry" in neutral_columns:
                industry_valid = group["industry"].notna()
                valid &= industry_valid
                design_parts.append(
                    pd.get_dummies(
                        group["industry"].astype("string"),
                        prefix="industry",
                        dtype=float,
                    )
                )
            if "market_cap" in neutral_columns:
                cap = pd.to_numeric(group["market_cap"], errors="coerce")
                valid &= cap.gt(0)
                design_parts.append(
                    pd.DataFrame(
                        {"log_market_cap": np.log(cap.where(cap.gt(0)))},
                        index=group.index,
                    )
                )

            if not valid.any():
                skipped_dates += 1
                continue
            design = pd.concat(design_parts, axis=1).loc[valid]
            design.insert(0, "intercept", 1.0)
            target = group.loc[valid, "factor"].astype(float)
            if len(target) <= design.shape[1]:
                skipped_dates += 1
                continue
            coefficients, *_ = np.linalg.lstsq(
                design.to_numpy(dtype=float),
                target.to_numpy(dtype=float),
                rcond=None,
            )
            residuals = target.to_numpy(dtype=float) - design.to_numpy(
                dtype=float
            ).dot(coefficients)
            result.loc[target.index, "factor"] = residuals

    if skipped_dates:
        warnings.append(
            f"Neutralization was skipped on {skipped_dates} date(s) with insufficient data."
        )

    if config.zscore:
        # 标准化使不同因子的权重可比较；横截面完全相同则统一设为 0。
        for _, indices in result.groupby("date", sort=True).groups.items():
            values = result.loc[indices, "factor"]
            mean = values.mean()
            standard_deviation = values.std(ddof=0)
            if pd.isna(standard_deviation):
                continue
            if standard_deviation == 0:
                result.loc[indices, "factor"] = values.where(values.isna(), 0.0)
            else:
                result.loc[indices, "factor"] = (
                    values - mean
                ) / standard_deviation

    return result, {
        "config": asdict(config),
        "warnings": warnings,
        "neutralized_on": neutral_columns,
    }
