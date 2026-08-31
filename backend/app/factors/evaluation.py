from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ..json_utils import json_safe
from .base import Factor, build_factor_observations
from .preprocessing import PreprocessConfig, preprocess_factor


def _correlation(left: pd.Series, right: pd.Series, rank: bool = False) -> float:
    """计算 Pearson 或 Spearman（秩）相关；样本不足时明确返回 NaN。"""
    valid = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(valid) < 2:
        return float("nan")
    if rank:
        valid = valid.rank(method="average")
    if valid["left"].nunique() < 2 or valid["right"].nunique() < 2:
        return float("nan")
    return float(valid["left"].corr(valid["right"]))


def _series_statistics(values: pd.Series) -> dict[str, float | int]:
    finite = pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if finite.empty:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "ir": float("nan"),
            "win_rate": float("nan"),
        }
    standard_deviation = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
    mean = float(finite.mean())
    return {
        "count": len(finite),
        "mean": mean,
        "std": standard_deviation,
        "ir": mean / standard_deviation if standard_deviation > 0 else float("nan"),
        "win_rate": float((finite > 0).mean()),
    }


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    output = frame.copy()
    if "date" in output.columns:
        output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")
    return output.to_dict(orient="records")


def evaluate_factor(
    bars: pd.DataFrame,
    factor: Factor,
    start_date: date,
    end_date: date,
    forward_period: int = 5,
    quantiles: int = 5,
    preprocess: PreprocessConfig | None = None,
) -> dict[str, Any]:
    """评估因子横截面预测能力、分组收益和换手。

    因子值取自 T 日收盘；``forward_return`` 仅作为事后标签，不会反馈到
    因子计算。分组净值使用下一交易日收益，使因子分层结果和可交易时点一致。
    """
    if forward_period < 1:
        raise ValueError("forward_period must be at least 1")
    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")
    if bars.empty:
        raise ValueError("No bars are available for factor evaluation")

    ordered = bars.sort_values(["symbol", "date"]).copy()
    observations = build_factor_observations(factor, ordered)
    observations["forward_return"] = ordered.groupby(
        "symbol", sort=False
    )["close"].transform(
        lambda close: close.shift(-forward_period) / close - 1.0
    )
    # forward_return 是“研究标签”：用于衡量 T 日信号对未来的解释力；
    # 它绝不传回 Factor.compute，因此不构成未来函数。
    observations["next_return"] = ordered.groupby(
        "symbol", sort=False
    )["close"].transform(lambda close: close.shift(-1) / close - 1.0)
    observations, preprocess_report = preprocess_factor(
        observations, preprocess or PreprocessConfig()
    )
    period = observations[
        (observations["date"] >= pd.Timestamp(start_date))
        & (observations["date"] <= pd.Timestamp(end_date))
    ].copy()

    eligible = period["forward_return"].notna()
    valid = eligible & period["factor"].notna()
    valid_data = period.loc[valid].copy()
    coverage_total = int(eligible.sum())
    coverage_valid = int(valid.sum())

    ic_rows: list[dict[str, Any]] = []
    valid_data["rank_score"] = (
        valid_data["factor"] * factor.metadata.direction
    )
    # 原始 IC 与方向调整 IC 同时保留：前者反映原始定义，后者反映按策略
    # 排名后“高分是否对应更高未来收益”。
    for signal_date, group in valid_data.groupby("date", sort=True):
        raw_ic = _correlation(group["factor"], group["forward_return"])
        raw_rank_ic = _correlation(
            group["factor"], group["forward_return"], rank=True
        )
        adjusted_ic = _correlation(
            group["rank_score"], group["forward_return"]
        )
        adjusted_rank_ic = _correlation(
            group["rank_score"], group["forward_return"], rank=True
        )
        ic_rows.append(
            {
                "date": signal_date,
                "ic": raw_ic,
                "rank_ic": raw_rank_ic,
                "raw_ic": raw_ic,
                "raw_rank_ic": raw_rank_ic,
                "adjusted_ic": adjusted_ic,
                "adjusted_rank_ic": adjusted_rank_ic,
                "count": len(group),
            }
        )
    ic_frame = pd.DataFrame(
        ic_rows,
        columns=[
            "date",
            "ic",
            "rank_ic",
            "raw_ic",
            "raw_rank_ic",
            "adjusted_ic",
            "adjusted_rank_ic",
            "count",
        ],
    )

    if not valid_data.empty:
        # 每个日期独立分位数分组，避免不同日期证券池规模与因子量纲影响分层。
        percentile_rank = valid_data.groupby("date", sort=False)["rank_score"].rank(
            method="first", pct=True
        )
        valid_data["quantile"] = np.ceil(percentile_rank * quantiles).clip(
            1, quantiles
        ).astype(int)

    daily_quantile = (
        valid_data.groupby(["date", "quantile"], sort=True)["next_return"]
        .mean()
        .unstack("quantile")
        if not valid_data.empty
        else pd.DataFrame()
    )
    daily_quantile = daily_quantile.reindex(columns=range(1, quantiles + 1))
    daily_quantile.columns = [f"q{column}" for column in daily_quantile.columns]
    quantile_net = (1.0 + daily_quantile.fillna(0.0)).cumprod()

    quantile_summary = [
        {
            "quantile": int(column[1:]),
            "mean_return": daily_quantile[column].mean(),
            "win_rate": (daily_quantile[column].dropna() > 0).mean()
            if daily_quantile[column].notna().any()
            else float("nan"),
        }
        for column in daily_quantile.columns
    ]

    bottom_column = "q1"
    top_column = f"q{quantiles}"
    if daily_quantile.empty:
        long_short = pd.DataFrame(columns=["date", "return", "net_value"])
    else:
        spread = daily_quantile[top_column] - daily_quantile[bottom_column]
        long_short = pd.DataFrame(
            {
                "date": daily_quantile.index,
                "return": spread,
                "net_value": (1.0 + spread.fillna(0.0)).cumprod(),
            }
        )

    turnover_rows: list[dict[str, Any]] = []
    previous_members: dict[int, set[str]] = {}
    if not valid_data.empty:
        for signal_date, group in valid_data.groupby("date", sort=True):
            row: dict[str, Any] = {"date": signal_date}
            for quantile_number in range(1, quantiles + 1):
                members = set(
                    group.loc[
                        group["quantile"] == quantile_number, "symbol"
                    ].astype(str)
                )
                previous = previous_members.get(quantile_number)
                if previous is None or not previous or not members:
                    row[f"q{quantile_number}"] = float("nan")
                else:
                    overlap = len(previous & members)
                    row[f"q{quantile_number}"] = 1.0 - overlap / max(
                        len(previous), len(members)
                    )
                previous_members[quantile_number] = members
            turnover_rows.append(row)
    turnover = pd.DataFrame(turnover_rows)

    factor_values = valid_data["factor"].dropna()
    distribution = {
        "count": len(factor_values),
        "mean": factor_values.mean(),
        "std": factor_values.std(ddof=1) if len(factor_values) > 1 else 0.0,
        "min": factor_values.min() if not factor_values.empty else float("nan"),
        "p01": factor_values.quantile(0.01)
        if not factor_values.empty
        else float("nan"),
        "p25": factor_values.quantile(0.25)
        if not factor_values.empty
        else float("nan"),
        "median": factor_values.median()
        if not factor_values.empty
        else float("nan"),
        "p75": factor_values.quantile(0.75)
        if not factor_values.empty
        else float("nan"),
        "p99": factor_values.quantile(0.99)
        if not factor_values.empty
        else float("nan"),
        "max": factor_values.max() if not factor_values.empty else float("nan"),
    }
    if factor_values.empty:
        factor_distribution: list[dict[str, Any]] = []
    elif factor_values.min() == factor_values.max():
        factor_distribution = [
            {"bin": f"{factor_values.iloc[0]:.4f}", "count": len(factor_values)}
        ]
    else:
        counts, edges = np.histogram(factor_values.to_numpy(dtype=float), bins=12)
        factor_distribution = [
            {
                "bin": f"{edges[index]:.4f}~{edges[index + 1]:.4f}",
                "count": int(count),
            }
            for index, count in enumerate(counts)
        ]

    quantile_returns_output = daily_quantile.reset_index().rename(
        columns={"index": "date"}
    )
    quantile_net_output = quantile_net.reset_index().rename(
        columns={"index": "date"}
    )
    if not long_short.empty:
        quantile_net_output = quantile_net_output.merge(
            long_short[["date", "net_value"]]
            .reset_index(drop=True)
            .rename(columns={"net_value": "long_short"}),
            on="date",
            how="left",
        )
    top_turnover = (
        pd.to_numeric(turnover[top_column], errors="coerce").mean()
        if top_column in turnover.columns
        else float("nan")
    )
    ic_statistics = _series_statistics(
        ic_frame["ic"] if "ic" in ic_frame.columns else pd.Series(dtype=float)
    )
    rank_ic_statistics = _series_statistics(
        ic_frame["rank_ic"]
        if "rank_ic" in ic_frame.columns
        else pd.Series(dtype=float)
    )
    adjusted_ic_statistics = _series_statistics(
        ic_frame["adjusted_ic"]
        if "adjusted_ic" in ic_frame.columns
        else pd.Series(dtype=float)
    )
    adjusted_rank_ic_statistics = _series_statistics(
        ic_frame["adjusted_rank_ic"]
        if "adjusted_rank_ic" in ic_frame.columns
        else pd.Series(dtype=float)
    )
    summary = {
        "ic_mean": ic_statistics["mean"],
        "ic_std": ic_statistics["std"],
        "ic_ir": ic_statistics["ir"],
        "rank_ic_mean": rank_ic_statistics["mean"],
        "rank_ic_std": rank_ic_statistics["std"],
        "rank_ic_ir": rank_ic_statistics["ir"],
        "win_rate": ic_statistics["win_rate"],
        "raw_ic_mean": ic_statistics["mean"],
        "raw_ic_std": ic_statistics["std"],
        "raw_ic_ir": ic_statistics["ir"],
        "raw_rank_ic_mean": rank_ic_statistics["mean"],
        "raw_rank_ic_std": rank_ic_statistics["std"],
        "raw_rank_ic_ir": rank_ic_statistics["ir"],
        "raw_win_rate": ic_statistics["win_rate"],
        "adjusted_ic_mean": adjusted_ic_statistics["mean"],
        "adjusted_ic_std": adjusted_ic_statistics["std"],
        "adjusted_ic_ir": adjusted_ic_statistics["ir"],
        "adjusted_rank_ic_mean": adjusted_rank_ic_statistics["mean"],
        "adjusted_rank_ic_std": adjusted_rank_ic_statistics["std"],
        "adjusted_rank_ic_ir": adjusted_rank_ic_statistics["ir"],
        "adjusted_win_rate": adjusted_ic_statistics["win_rate"],
        "coverage": coverage_valid / coverage_total if coverage_total else 0.0,
        "turnover": top_turnover,
    }
    result = {
        "factor_name": factor.metadata.name,
        "direction": factor.metadata.direction,
        "direction_label": factor.metadata.direction_label,
        "direction_kind": factor.metadata.direction_kind,
        "applicable_assets": list(factor.metadata.applicable_assets),
        "forward_period": forward_period,
        "quantile_return_period": 1,
        "quantiles": quantiles,
        "preprocessing": preprocess_report,
        "warnings": preprocess_report["warnings"],
        "summary": summary,
        "metrics": summary,
        "coverage": {
            "valid": coverage_valid,
            "eligible": coverage_total,
            "ratio": coverage_valid / coverage_total if coverage_total else 0.0,
        },
        "ic": {
            **ic_statistics,
            "series": _records(
                ic_frame[["date", "ic", "count"]]
                if not ic_frame.empty
                else pd.DataFrame(columns=["date", "ic", "count"])
            ),
        },
        "rank_ic": {
            **rank_ic_statistics,
            "series": _records(
                ic_frame[["date", "rank_ic", "count"]]
                if not ic_frame.empty
                else pd.DataFrame(columns=["date", "rank_ic", "count"])
            ),
        },
        "raw_ic": {
            **ic_statistics,
            "series": _records(
                ic_frame[["date", "raw_ic", "count"]]
                if not ic_frame.empty
                else pd.DataFrame(columns=["date", "raw_ic", "count"])
            ),
        },
        "raw_rank_ic": {
            **rank_ic_statistics,
            "series": _records(
                ic_frame[["date", "raw_rank_ic", "count"]]
                if not ic_frame.empty
                else pd.DataFrame(columns=["date", "raw_rank_ic", "count"])
            ),
        },
        "adjusted_ic": {
            **adjusted_ic_statistics,
            "series": _records(
                ic_frame[["date", "adjusted_ic", "count"]]
                if not ic_frame.empty
                else pd.DataFrame(columns=["date", "adjusted_ic", "count"])
            ),
        },
        "adjusted_rank_ic": {
            **adjusted_rank_ic_statistics,
            "series": _records(
                ic_frame[["date", "adjusted_rank_ic", "count"]]
                if not ic_frame.empty
                else pd.DataFrame(columns=["date", "adjusted_rank_ic", "count"])
            ),
        },
        "ic_series": _records(ic_frame),
        "raw_ic_series": _records(
            ic_frame[["date", "raw_ic", "raw_rank_ic", "count"]]
        ),
        "adjusted_ic_series": _records(
            ic_frame[["date", "adjusted_ic", "adjusted_rank_ic", "count"]]
        ),
        "quantile_summary": quantile_summary,
        "quantile_returns": _records(quantile_returns_output),
        "quantile_net_values": _records(quantile_net_output),
        "long_short": _records(long_short),
        "turnover": {
            "mean_top_quantile": top_turnover,
            "series": _records(turnover),
        },
        "distribution": distribution,
        "factor_distribution": factor_distribution,
    }
    return json_safe(result)
