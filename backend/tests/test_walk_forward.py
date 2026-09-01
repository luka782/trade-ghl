from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from app.config import Settings
from app.storage import Storage
from app.validation import (
    ValidationProtocol,
    assert_locked_oos_excluded,
    common_recent_evaluation_period,
    fit_train_only_scaler,
    generate_preregistered_candidates,
    generate_rolling_folds,
    run_walk_forward,
)


def test_common_period_is_three_years_with_last_twelve_months_locked() -> None:
    dates = pd.bdate_range("2023-07-03", "2026-08-24")
    symbol_dates = {
        symbol: dates.delete(index)
        for index, symbol in enumerate(("A", "B", "C", "D"))
    }

    period = common_recent_evaluation_period(
        symbol_dates,
        as_of="2026-08-25",
    )

    assert period.evaluation_start == pd.Timestamp("2023-08-01")
    assert period.evaluation_end == pd.Timestamp("2026-07-31")
    assert period.locked_oos_start == pd.Timestamp("2025-08-01")
    assert period.locked_oos_end == pd.Timestamp("2026-07-31")


def test_protocol_snapshot_and_hash_are_immutable_and_deterministic() -> None:
    protocol = ValidationProtocol(
        symbols=("A", "B", "C", "D"),
        evaluation_start=pd.Timestamp("2023-08-01").date(),
        evaluation_end=pd.Timestamp("2026-07-31").date(),
        locked_oos_start=pd.Timestamp("2025-08-01").date(),
        locked_oos_end=pd.Timestamp("2026-07-31").date(),
        metadata={"purpose": "final"},
    )
    same = ValidationProtocol(
        symbols=("A", "B", "C", "D"),
        evaluation_start=pd.Timestamp("2023-08-01").date(),
        evaluation_end=pd.Timestamp("2026-07-31").date(),
        locked_oos_start=pd.Timestamp("2025-08-01").date(),
        locked_oos_end=pd.Timestamp("2026-07-31").date(),
        metadata={"purpose": "final"},
    )

    assert protocol.protocol_hash == same.protocol_hash
    assert protocol.snapshot().sha256 == protocol.protocol_hash
    with pytest.raises(TypeError):
        protocol.snapshot().payload["candidate_count"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        protocol.metadata["purpose"] = "changed"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        protocol.candidate_count = 1  # type: ignore[misc]


def test_fold_boundaries_have_purge_embargo_no_overlap_and_exclude_oos() -> None:
    dates = pd.bdate_range("2024-01-02", periods=180)
    locked_oos_start = dates[160]
    folds = generate_rolling_folds(
        dates,
        train_sessions=60,
        validation_sessions=20,
        test_sessions=20,
        step_sessions=20,
        purge_sessions=5,
        embargo_sessions=4,
        locked_oos_start=locked_oos_start,
    )

    for fold in folds:
        groups = [
            set(fold.train_dates),
            set(fold.purge_dates),
            set(fold.validation_dates),
            set(fold.embargo_dates),
            set(fold.test_dates),
        ]
        assert all(
            left.isdisjoint(right)
            for index, left in enumerate(groups)
            for right in groups[index + 1 :]
        )
        assert fold.train_end < fold.validation_start < fold.test_start
        assert fold.test_end < locked_oos_start
        assert len(fold.purge_dates) == 5
        assert len(fold.embargo_dates) == 4


def test_locked_oos_guard_rejects_boundary_and_later_dates() -> None:
    with pytest.raises(ValueError, match="locked OOS"):
        assert_locked_oos_excluded(
            ["2025-07-31", "2025-08-01"],
            "2025-08-01",
        )
    assert_locked_oos_excluded(["2025-07-31"], "2025-08-01")


def test_scaler_is_fit_on_train_only() -> None:
    dates = pd.bdate_range("2024-01-02", periods=8)
    frame = pd.DataFrame(
        {
            "date": dates,
            "feature": [1.0, 2.0, 3.0, 4.0, 10_000.0, 20_000.0, 30_000.0, 40_000.0],
        }
    )
    scaler = fit_train_only_scaler(frame, ["feature"], dates[:4])

    assert scaler.means["feature"] == pytest.approx(2.5)
    assert scaler.scales["feature"] == pytest.approx(np.std([1, 2, 3, 4], ddof=0))
    transformed = scaler.transform(frame)
    assert transformed.loc[:3, "feature"].mean() == pytest.approx(0.0)
    assert set(scaler.fit_dates) == set(dates[:4])


def test_preregistered_candidate_generation_is_deterministic_and_exact() -> None:
    first = generate_preregistered_candidates(seed=17)
    second = generate_preregistered_candidates(seed=17)
    different_order = generate_preregistered_candidates(seed=18)

    assert len(first) == 96
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert len({item.candidate_id for item in first}) == 96
    assert [item.candidate_id for item in first] != [
        item.candidate_id for item in different_order
    ]
    with pytest.raises(TypeError):
        first[0].parameters["fixed_stop"] = 0.99  # type: ignore[index]


def test_walk_forward_tests_only_validation_winner() -> None:
    dates = pd.bdate_range("2024-01-02", periods=34)
    frame = pd.DataFrame({"date": dates, "value": np.arange(len(dates))})
    folds = generate_rolling_folds(
        dates,
        train_sessions=10,
        validation_sessions=5,
        test_sessions=5,
        purge_sessions=2,
        embargo_sessions=2,
        step_sessions=10,
    )
    calls: list[tuple[int, str]] = []

    def evaluate(
        candidate: int,
        train: pd.DataFrame,
        evaluation: pd.DataFrame,
        phase: str,
    ) -> float:
        calls.append((candidate, phase))
        assert set(train.index).isdisjoint(evaluation.index)
        return float(candidate)

    results = run_walk_forward(frame, folds, [1, 3, 2], evaluate)

    assert all(item["winner"] == 3 for item in results)
    assert sum(phase == "test" for _, phase in calls) == len(folds)
    assert all(candidate == 3 for candidate, phase in calls if phase == "test")


def test_identical_completed_protocol_reuses_locked_oos(tmp_path) -> None:
    storage = Storage(
        Settings(
            data_dir=tmp_path / "data",
            db_path=tmp_path / "quant.sqlite3",
        )
    )
    request = {"symbols": ["A", "B"], "protocol": {"locked": 12}}
    storage.create_walk_forward_job("one", request)
    storage.update_walk_forward_job(
        "one", status="completed", progress=1.0, result={"ok": True}
    )

    reused = storage.find_completed_walk_forward_job(request)

    assert reused is not None
    assert reused["id"] == "one"
    assert reused["result"] == {"ok": True}
    assert storage.find_completed_walk_forward_job(
        {"symbols": ["A", "C"], "protocol": {"locked": 12}}
    ) is None
