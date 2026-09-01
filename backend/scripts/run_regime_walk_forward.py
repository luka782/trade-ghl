from __future__ import annotations

import argparse
import json
from uuid import uuid4

from app.data.akshare_provider import AkShareProvider
from app.main import _run_walk_forward_research
from app.schemas import TimingWalkForwardRequest
from app.storage import Storage


SYMBOLS = [
    "515080",
    "510300",
    "588200",
    "600519",
    "600036",
    "603986",
    "600487",
    "002460",
]


def _component(name: str, weight: float, direction: int | None = None) -> dict:
    return {
        "factor_name": name,
        "weight": weight,
        "enabled": True,
        "direction": direction,
        "normalization": "auto",
        "winsorize": True,
        "missing_policy": "renormalize",
    }


def default_request() -> TimingWalkForwardRequest:
    entry = {
        "name": "综合趋势反转买入评分",
        "mode": "time_series",
        "rolling_window": 126,
        "rolling_min_periods": 60,
        "zscore_clip": 3,
        "components": [
            _component("reversal_5", 0.30, 1),
            _component("overnight_reversal_20", 0.20, 1),
            _component("price_position_60", 0.25, -1),
            _component("intraday_strength_20", 0.15, 1),
            _component("amount_surprise_20", 0.10, 1),
        ],
    }
    exit_config = {
        "name": "综合趋势反转卖出评分",
        "mode": "time_series",
        "rolling_window": 126,
        "rolling_min_periods": 60,
        "zscore_clip": 3,
        "components": [
            _component("price_position_60", 0.25, 1),
            _component("ma_bias_20", 0.20, 1),
            _component("max_return_20", 0.15, 1),
            _component("intraday_strength_20", 0.15, -1),
            _component("volume_price_corr_20", 0.15, -1),
            _component("atr_ratio_20", 0.10, 1),
        ],
    }
    return TimingWalkForwardRequest.model_validate(
        {
            "symbols": SYMBOLS,
            "config": entry,
            "entry_config": entry,
            "exit_config": exit_config,
            "options": {
                "timing_style": "regime_reversion",
                "regime_entry_mode": "confirmation_count",
                "regime_confirmation_required": 2,
                "position_sizing": "atr_risk",
                "risk_per_trade": 0.01,
                "max_position_fraction": 0.5,
                "minimum_holding_sessions": 0,
            },
            "adjust": "qfq",
            "benchmark": "CSI300",
            "protocol": {
                "evaluation_years": 3,
                "locked_oos_months": 12,
                "train_months": 6,
                "validation_months": 2,
                "test_months": 2,
                "purge_sessions": 5,
                "embargo_sessions": 5,
                "minimum_round_trips_per_symbol": 1,
                "minimum_market_exposure": 0.02,
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", default=None)
    args = parser.parse_args()
    storage = Storage()
    job_id = args.job_id or uuid4().hex
    request = default_request()
    payload = request.model_dump(mode="json")
    existing = storage.find_completed_walk_forward_job(payload)
    if existing is not None:
        print(json.dumps(existing, ensure_ascii=False, default=str))
        return
    if storage.get_walk_forward_job(job_id) is None:
        storage.create_walk_forward_job(
            job_id, payload
        )
    _run_walk_forward_research(
        storage, AkShareProvider(), request, job_id
    )
    print(
        json.dumps(
            storage.get_walk_forward_job(job_id),
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
