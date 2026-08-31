from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _finite(value: Any) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) else None


def _criterion(value: float | None, threshold: float) -> bool:
    return value is not None and value > threshold


def summarize(report_dir: Path) -> None:
    source = pd.read_csv(report_dir / "factor_test_summary.csv")
    oos = source[(source["split"] == "out_of_sample") & (source["status"] == "success")].copy()
    rows: list[dict[str, Any]] = []
    for record in oos.to_dict(orient="records"):
        rank_ic = _finite(record.get("adjusted_rank_ic_mean"))
        ic_ir = _finite(record.get("adjusted_ic_ir"))
        monotonicity = _finite(record.get("quantile_monotonicity"))
        excess = _finite(record.get("excess_return"))
        turnover = _finite(record.get("turnover"))
        checks = {
            "adjusted_rank_ic_positive": _criterion(rank_ic, 0.02),
            "adjusted_ic_ir_positive": _criterion(ic_ir, 0.10),
            "quantile_monotonic": _criterion(monotonicity, 0.50),
            "net_excess_positive": _criterion(excess, 0.0),
        }
        evidence_score = sum(checks.values())
        if evidence_score == 4:
            verdict = "样本外证据较一致"
        elif evidence_score >= 2:
            verdict = "样本外证据混合"
        else:
            verdict = "样本外未支持预期方向"
        cautions: list[str] = []
        if rank_ic is not None and rank_ic <= 0:
            cautions.append("方向调整后RankIC非正")
        if monotonicity is not None and monotonicity <= 0:
            cautions.append("分组收益不单调")
        if excess is not None and excess <= 0:
            cautions.append("扣费后未跑赢基准")
        if turnover is not None and turnover >= 0.30:
            cautions.append("因子分组换手偏高")
        if _finite(record.get("max_drawdown")) is not None and float(
            record["max_drawdown"]
        ) <= -0.40:
            cautions.append("策略最大回撤超过40%")
        rows.append(
            {
                **record,
                "evidence_score": evidence_score,
                "verdict": verdict,
                "cautions": cautions,
                "criteria": checks,
            }
        )
    rows.sort(
        key=lambda row: (
            row["evidence_score"],
            _finite(row.get("adjusted_rank_ic_mean")) or float("-inf"),
            _finite(row.get("excess_return")) or float("-inf"),
        ),
        reverse=True,
    )
    comparison = pd.DataFrame(
        [
            {
                "factor_name": row["factor_name"],
                "factor_name_zh": row["factor_name_zh"],
                "direction": row["direction"],
                "adjusted_rank_ic_mean": row["adjusted_rank_ic_mean"],
                "adjusted_ic_ir": row["adjusted_ic_ir"],
                "quantile_monotonicity": row["quantile_monotonicity"],
                "turnover": row["turnover"],
                "excess_return": row["excess_return"],
                "max_drawdown": row["max_drawdown"],
                "evidence_score": row["evidence_score"],
                "verdict": row["verdict"],
                "cautions": "；".join(row["cautions"]),
            }
            for row in rows
        ]
    )
    comparison.to_csv(
        report_dir / "oos_factor_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "generated_at": datetime.now().isoformat(),
        "method": {
            "period": "2025-08-01 至 2026-07-31（最近12个完整月）",
            "criteria": {
                "adjusted_rank_ic_mean": "> 0.02",
                "adjusted_ic_ir": "> 0.10",
                "quantile_monotonicity": "> 0.50",
                "net_excess_return": "> 0",
            },
            "warning": (
                "证据评分只是统一的样本外筛查规则，不是投资建议，也不替代多市场、"
                "多时期和历史时点股票池验证。"
            ),
        },
        "ranked_factors": rows,
    }
    (report_dir / "factor_conclusions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"SUMMARY_COMPLETE {report_dir} factors={len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-date",
        default="2026-08-21",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=_backend_root() / "reports",
    )
    args = parser.parse_args()
    summarize(args.report_root / args.report_date)


if __name__ == "__main__":
    main()
