from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FactorUnavailableError


def merge_benchmark_bars(
    panel: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
) -> pd.DataFrame:
    """Attach exact-date benchmark close and causal return to a security panel.

    Benchmark returns are calculated on the benchmark's own ordered history
    before the exact-date merge. Missing benchmark dates remain missing; this
    helper never forward-fills or backward-fills benchmark observations.
    """

    if "date" not in panel.columns:
        raise ValueError("Security panel is missing required column: date")
    missing = [
        column for column in ("date", "close") if column not in benchmark_bars.columns
    ]
    if missing:
        raise FactorUnavailableError(
            "Benchmark bars are unavailable: missing " + ", ".join(missing)
        )

    benchmark = benchmark_bars.loc[:, ["date", "close"]].copy()
    benchmark["date"] = pd.to_datetime(
        benchmark["date"], errors="coerce"
    ).dt.normalize()
    if benchmark["date"].isna().any():
        raise ValueError("Benchmark bars contain invalid dates")
    if benchmark["date"].duplicated().any():
        duplicates = benchmark.loc[benchmark["date"].duplicated(), "date"]
        raise ValueError(
            "Benchmark bars must contain exactly one row per date; duplicate date "
            f"{duplicates.iloc[0].date()}"
        )

    benchmark = benchmark.sort_values("date")
    benchmark["benchmark_close"] = pd.to_numeric(
        benchmark["close"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    if not benchmark["benchmark_close"].notna().any():
        raise FactorUnavailableError(
            "Benchmark bars are unavailable: close contains no numeric values"
        )
    benchmark["benchmark_return"] = benchmark["benchmark_close"].pct_change(
        fill_method=None
    )
    normalized = benchmark[
        ["date", "benchmark_close", "benchmark_return"]
    ]

    result = panel.copy()
    original_index = result.index
    result["_factor_original_order"] = np.arange(len(result))
    result["_factor_merge_date"] = pd.to_datetime(
        result["date"], errors="coerce"
    ).dt.normalize()
    if result["_factor_merge_date"].isna().any():
        raise ValueError("Security panel contains invalid dates")
    result = result.drop(
        columns=["benchmark_close", "benchmark_return"], errors="ignore"
    ).merge(
        normalized,
        left_on="_factor_merge_date",
        right_on="date",
        how="left",
        sort=False,
        validate="many_to_one",
        suffixes=("", "_benchmark"),
    )
    result = result.sort_values("_factor_original_order")
    result = result.drop(
        columns=["_factor_original_order", "_factor_merge_date", "date_benchmark"]
    )
    result.index = original_index
    return result
