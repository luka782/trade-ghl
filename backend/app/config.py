from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


SURVIVORSHIP_WARNING = (
    "The universe is based on explicitly requested or currently cached symbols. "
    "Historical index membership is not inferred, so results may contain survivorship bias."
)
QFQ_REVISION_WARNING = (
    "Forward-adjusted history can be restated after later corporate actions. "
    "Keep data snapshots for reproducible research; raw prices are used separately "
    "for execution constraints."
)


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path
    db_path: Path

    @classmethod
    def from_env(cls) -> "Settings":
        backend_dir = Path(__file__).resolve().parents[1]
        data_dir = Path(
            os.environ.get("QUANT_DATA_DIR", backend_dir / ".quant_data")
        ).expanduser()
        db_path = Path(
            os.environ.get("QUANT_DB_PATH", data_dir / "quant.sqlite3")
        ).expanduser()
        return cls(data_dir=data_dir.resolve(), db_path=db_path.resolve())
