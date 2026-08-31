from __future__ import annotations

from collections.abc import Iterable
import importlib.util
import os
from pathlib import Path
import sys

from .base import Factor
from .builtin import BUILTIN_FACTORS


class FactorRegistry:
    def __init__(self, factors: Iterable[Factor] = ()) -> None:
        self._factors: dict[str, Factor] = {}
        self.warnings: list[str] = []
        for factor in factors:
            self.register(factor)

    def register(self, factor: Factor) -> None:
        name = factor.metadata.name
        if name in self._factors:
            raise ValueError(f"Factor already registered: {name}")
        self._factors[name] = factor

    def get(self, name: str) -> Factor:
        try:
            return self._factors[name]
        except KeyError as exc:
            raise KeyError(f"Unknown factor: {name}") from exc

    def list(self) -> list[dict[str, object]]:
        return [
            {
                "name": metadata.name,
                "display_name": metadata.display_name,
                "display_name_zh": metadata.display_name_zh,
                "description": metadata.description,
                "description_zh": metadata.description_zh,
                "direction": metadata.direction,
                "direction_label": metadata.direction_label,
                "direction_kind": metadata.direction_kind,
                "applicable_assets": list(metadata.applicable_assets),
                "lookback": metadata.lookback,
                "requirements": list(metadata.required_columns),
                "required_columns": list(metadata.required_columns),
                "availability": metadata.availability,
            }
            for metadata in (factor.metadata for factor in self._factors.values())
        ]


factor_registry = FactorRegistry(BUILTIN_FACTORS)


def load_user_factors(
    registry: FactorRegistry,
    directory: Path | None = None,
) -> None:
    """Load local factor plug-ins that export ``FACTOR`` or ``FACTORS``.

    A broken user file is isolated and reported through ``registry.warnings`` so
    that one research experiment cannot prevent the API from starting.
    """

    factor_dir = directory or Path(
        os.environ.get(
            "QUANT_USER_FACTOR_DIR",
            Path(__file__).resolve().parents[2] / "user_factors",
        )
    )
    if not factor_dir.exists():
        return
    for path in sorted(factor_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module_name = f"quant_user_factor_{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise RuntimeError("could not create a Python module specification")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            exported: list[object] = []
            if hasattr(module, "FACTOR"):
                exported.append(module.FACTOR)
            if hasattr(module, "FACTORS"):
                exported.extend(list(module.FACTORS))
            if not exported:
                raise ValueError("file must export FACTOR or FACTORS")
            if not all(isinstance(item, Factor) for item in exported):
                raise TypeError("exported values must be Factor instances")
            for item in exported:
                registry.register(item)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            registry.warnings.append(f"{path.name}: {exc}")


load_user_factors(factor_registry)
