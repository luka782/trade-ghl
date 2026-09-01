from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Settings
from .data.base import Adjustment, DataProvider, normalize_bars, normalize_symbol
from .json_utils import json_safe


class Storage:
    """Parquet 行情缓存与 SQLite 元数据/研究任务持久化层。

    Parquet 负责大量按证券读取的时序行情；SQLite 只保存目录、配置快照和
    JSON 化结果。这种职责拆分兼顾本地研究性能与部署时的备份便利性。
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.data_dir = self.settings.data_dir
        self.db_path = self.settings.db_path
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        # WAL 允许读取历史结果时后台任务仍写入；RLock 保护本进程内的
        # Parquet 读写临界区。SQLite 仍不适合高并发多用户写入。
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    symbol TEXT NOT NULL,
                    adjustment TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    parquet_path TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (symbol, adjustment)
                );

                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    factor_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    result_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_backtest_created
                    ON backtest_runs(created_at DESC);

                CREATE TABLE IF NOT EXISTS multifactor_configs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    config_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_multifactor_config_created
                    ON multifactor_configs(created_at DESC);

                CREATE TABLE IF NOT EXISTS timing_walk_forward_jobs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL,
                    request_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    error TEXT,
                    report_path TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_timing_wf_created
                    ON timing_walk_forward_jobs(created_at DESC);
                """
            )

    def _cache_path(self, symbol: str, adjust: Adjustment) -> Path:
        normalized_symbol = normalize_symbol(symbol)
        return self.data_dir / "bars" / adjust / f"{normalized_symbol}.parquet"

    def _read_all(self, symbol: str, adjust: Adjustment) -> pd.DataFrame:
        path = self._cache_path(symbol, adjust)
        if not path.exists():
            return pd.DataFrame()
        bars = pd.read_parquet(path)
        return normalize_bars(bars)

    def read_bars(
        self,
        symbol: str,
        adjust: Adjustment,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        with self._lock:
            bars = self._read_all(symbol, adjust)
        if bars.empty:
            return bars
        if start_date is not None:
            bars = bars[bars["date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            bars = bars[bars["date"] <= pd.Timestamp(end_date)]
        return bars.reset_index(drop=True)

    def _write_bars(
        self, symbol: str, adjust: Adjustment, bars: pd.DataFrame
    ) -> None:
        """通过临时文件原子替换 Parquet，避免下载中断留下半个数据文件。"""
        if bars.empty:
            return
        normalized_symbol = normalize_symbol(symbol)
        normalized = normalize_bars(bars, normalized_symbol)
        path = self._cache_path(normalized_symbol, adjust)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.parquet")
        normalized.to_parquet(temporary, index=False)
        os.replace(temporary, path)

        now = datetime.now(timezone.utc).isoformat()
        start_date = pd.Timestamp(normalized["date"].min()).date().isoformat()
        end_date = pd.Timestamp(normalized["date"].max()).date().isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO datasets (
                    symbol, adjustment, start_date, end_date, row_count,
                    parquet_path, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, adjustment) DO UPDATE SET
                    start_date = excluded.start_date,
                    end_date = excluded.end_date,
                    row_count = excluded.row_count,
                    parquet_path = excluded.parquet_path,
                    updated_at = excluded.updated_at
                """,
                (
                    normalized_symbol,
                    adjust,
                    start_date,
                    end_date,
                    len(normalized),
                    str(path),
                    now,
                ),
            )

    def update_symbol(
        self,
        provider: DataProvider,
        symbol: str,
        start_date: date,
        end_date: date,
        adjust: Adjustment,
    ) -> dict[str, Any]:
        """按缺口拉取并合并行情；扩展前复权数据时刷新完整历史区间。

        前复权序列会受后来除权除息重缩放。直接在尾部拼接新旧 qfq 数据会让
        同一证券的历史价格处于不同复权尺度，因此必须整体刷新。
        """
        if start_date > end_date:
            raise ValueError("start_date must be on or before end_date")

        normalized_symbol = normalize_symbol(symbol)
        with self._lock:
            cached = self._read_all(normalized_symbol, adjust)
            ranges: list[tuple[date, date]] = []
            if cached.empty:
                ranges.append((start_date, end_date))
            else:
                cached_start = pd.Timestamp(cached["date"].min()).date()
                cached_end = pd.Timestamp(cached["date"].max()).date()
                if adjust == "qfq" and end_date > cached_end:
                    # A later corporate action can rescale the entire forward-adjusted
                    # history, so extending qfq data must refresh the full cached range.
                    ranges.append((min(start_date, cached_start), end_date))
                else:
                    if start_date < cached_start:
                        ranges.append(
                            (
                                start_date,
                                min(end_date, cached_start - timedelta(days=1)),
                            )
                        )
                    if end_date > cached_end:
                        ranges.append(
                            (max(start_date, cached_end + timedelta(days=1)), end_date)
                        )
                ranges = [item for item in ranges if item[0] <= item[1]]

            refresh_qfq = (
                not cached.empty
                and adjust == "qfq"
                and end_date > pd.Timestamp(cached["date"].max()).date()
            )
            chunks: list[pd.DataFrame] = (
                [] if refresh_qfq else [cached] if not cached.empty else []
            )
            fetched_rows = 0
            for range_start, range_end in ranges:
                fetched = provider.fetch_bars(
                    normalized_symbol, range_start, range_end, adjust
                )
                if fetched is None:
                    continue
                normalized = normalize_bars(fetched, normalized_symbol)
                fetched_rows += len(normalized)
                if not normalized.empty:
                    chunks.append(normalized)

            if refresh_qfq and not chunks:
                chunks.append(cached)
            if chunks:
                merged = normalize_bars(pd.concat(chunks, ignore_index=True), normalized_symbol)
                self._write_bars(normalized_symbol, adjust, merged)
            else:
                merged = pd.DataFrame()

        requested = (
            merged[
                (merged["date"] >= pd.Timestamp(start_date))
                & (merged["date"] <= pd.Timestamp(end_date))
            ]
            if not merged.empty
            else merged
        )
        if requested.empty:
            status = "no_data"
        elif fetched_rows:
            status = "updated"
        else:
            status = "up_to_date"
        return {
            "symbol": normalized_symbol,
            "status": status,
            "requested_start": start_date.isoformat(),
            "requested_end": end_date.isoformat(),
            "fetched_rows": fetched_rows,
            "cached_rows": len(requested),
            "cache_start": (
                pd.Timestamp(merged["date"].min()).date().isoformat()
                if not merged.empty
                else None
            ),
            "cache_end": (
                pd.Timestamp(merged["date"].max()).date().isoformat()
                if not merged.empty
                else None
            ),
        }

    def list_symbols(self, adjust: Adjustment | None = None) -> list[str]:
        sql = "SELECT DISTINCT symbol FROM datasets"
        parameters: tuple[Any, ...] = ()
        if adjust is not None:
            sql += " WHERE adjustment = ?"
            parameters = (adjust,)
        sql += " ORDER BY symbol"
        with self._connect() as connection:
            return [
                str(row["symbol"])
                for row in connection.execute(sql, parameters).fetchall()
            ]

    def dataset_status(self) -> dict[str, Any]:
        with self._connect() as connection:
            summary = connection.execute(
                """
                SELECT COUNT(*) AS dataset_count,
                       COUNT(DISTINCT symbol) AS symbol_count,
                       COALESCE(SUM(row_count), 0) AS row_count,
                       MIN(start_date) AS min_date,
                       MAX(end_date) AS max_date,
                       MAX(updated_at) AS last_updated
                FROM datasets
                """
            ).fetchone()
            rows = connection.execute(
                """
                SELECT symbol, adjustment, start_date, end_date, row_count, updated_at
                FROM datasets
                ORDER BY symbol, adjustment
                """
            ).fetchall()
        datasets = [{**dict(row), "status": "ready"} for row in rows]
        cache_bytes = sum(
            path.stat().st_size
            for path in (self.data_dir / "bars").rglob("*.parquet")
        ) if (self.data_dir / "bars").exists() else 0
        return {
            "dataset_count": int(summary["dataset_count"]),
            "symbol_count": int(summary["symbol_count"]),
            "total_symbols": int(summary["symbol_count"]),
            "row_count": int(summary["row_count"]),
            "total_rows": int(summary["row_count"]),
            "min_date": summary["min_date"],
            "max_date": summary["max_date"],
            "latest_trade_date": summary["max_date"],
            "last_updated": summary["last_updated"],
            "updated_at": summary["last_updated"],
            "cache_bytes": cache_bytes,
            "datasets": datasets,
        }

    def save_backtest(
        self,
        run_id: str,
        factor_name: str,
        request: dict[str, Any],
        summary: dict[str, Any],
        result: dict[str, Any],
        status: str = "completed",
    ) -> None:
        payload = lambda value: json.dumps(  # noqa: E731
            value, ensure_ascii=False, allow_nan=False, default=str
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO backtest_runs (
                    id, created_at, factor_name, status,
                    request_json, summary_json, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    datetime.now(timezone.utc).isoformat(),
                    factor_name,
                    status,
                    payload(request),
                    payload(summary),
                    payload(result),
                ),
            )

    def list_backtests(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, created_at, factor_name, status, request_json, summary_json
                FROM backtest_runs
                ORDER BY created_at DESC
                """
            ).fetchall()
        return [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "factor_name": row["factor_name"],
                "status": row["status"],
                "request": json.loads(row["request_json"]),
                "params": json.loads(row["request_json"]),
                "summary": json.loads(row["summary_json"]),
            }
            for row in rows
        ]

    def get_backtest(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, created_at, factor_name, status,
                       request_json, summary_json, result_json
                FROM backtest_runs
                WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "factor_name": row["factor_name"],
            "status": row["status"],
            "request": json.loads(row["request_json"]),
            "params": json.loads(row["request_json"]),
            "summary": json.loads(row["summary_json"]),
            "result": json.loads(row["result_json"]),
        }

    def delete_backtest(self, run_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM backtest_runs WHERE id = ?",
                (run_id,),
            )
        return cursor.rowcount > 0

    def save_multifactor_config(
        self,
        config_id: str,
        name: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        created_at = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(
            config,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO multifactor_configs (id, name, created_at, config_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    config_json = excluded.config_json
                """,
                (config_id, name, created_at, payload),
            )
        return {
            "id": config_id,
            "name": name,
            "created_at": created_at,
            "config": config,
        }

    def list_multifactor_configs(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, name, created_at, config_json
                FROM multifactor_configs
                ORDER BY created_at DESC
                """
            ).fetchall()
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "created_at": row["created_at"],
                "config": json.loads(row["config_json"]),
            }
            for row in rows
        ]

    def get_multifactor_config(self, config_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, name, created_at, config_json
                FROM multifactor_configs
                WHERE id = ?
                """,
                (config_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "name": row["name"],
            "created_at": row["created_at"],
            "config": json.loads(row["config_json"]),
        }

    def find_completed_walk_forward_job(
        self, request: dict[str, Any]
    ) -> dict[str, Any] | None:
        """返回完全相同预注册请求的既有结果，避免重复查看锁定OOS。"""
        canonical = json.dumps(
            request,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, request_json
                FROM timing_walk_forward_jobs
                WHERE status = 'completed'
                ORDER BY created_at DESC
                """
            ).fetchall()
        for row in rows:
            stored = json.dumps(
                json.loads(row["request_json"]),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            if stored == canonical:
                return self.get_walk_forward_job(str(row["id"]))
        return None

    def create_walk_forward_job(
        self, job_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO timing_walk_forward_jobs (
                    id, created_at, updated_at, status, progress,
                    request_json, summary_json, result_json
                ) VALUES (?, ?, ?, 'pending', 0, ?, '{}', '{}')
                """,
                (
                    job_id,
                    now,
                    now,
                    json.dumps(request, ensure_ascii=False, default=str),
                ),
            )
        return self.get_walk_forward_job(job_id) or {}

    def update_walk_forward_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: float | None = None,
        summary: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        report_path: str | None = None,
    ) -> None:
        current = self.get_walk_forward_job(job_id)
        if current is None:
            raise KeyError(job_id)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE timing_walk_forward_jobs
                SET updated_at = ?, status = ?, progress = ?,
                    summary_json = ?, result_json = ?, error = ?,
                    report_path = ?
                WHERE id = ?
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    status or current["status"],
                    (
                        float(progress)
                        if progress is not None
                        else current["progress"]
                    ),
                    json.dumps(
                        json_safe(
                            summary
                            if summary is not None
                            else current["summary"]
                        ),
                        ensure_ascii=False,
                        allow_nan=False,
                        default=str,
                    ),
                    json.dumps(
                        json_safe(
                            result
                            if result is not None
                            else current["result"]
                        ),
                        ensure_ascii=False,
                        allow_nan=False,
                        default=str,
                    ),
                    error,
                    report_path or current.get("report_path"),
                    job_id,
                ),
            )

    def get_walk_forward_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM timing_walk_forward_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "task_id": row["id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "status": row["status"],
            "progress": float(row["progress"]),
            "request": json.loads(row["request_json"]),
            "summary": json.loads(row["summary_json"]),
            "result": json.loads(row["result_json"]),
            "error": row["error"],
            "report_path": row["report_path"],
        }
