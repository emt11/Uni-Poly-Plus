"""Small, model-agnostic result writer for new experiments.

The contract keeps experiment identity at the result root and publishes JSON
files with a temporary file followed by an atomic rename. Existing historical
result producers are not migrated by this module.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def create_manifest(root: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Publish an experiment manifest once and refuse accidental overwrite."""

    path = Path(root) / "manifest.json"
    if path.exists():
        raise FileExistsError(f"experiment manifest already exists: {path}")
    _atomic_json(path, dict(manifest))
    return path


def write_unit(
    root: str | Path,
    *,
    arm: str,
    task: str,
    fold: int,
    run: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Path:
    """Write the two JSON records for one declared unit."""

    unit = Path(root) / "units" / arm / task / f"fold{int(fold)}"
    _atomic_json(unit / "run.json", dict(run))
    _atomic_json(unit / "metrics.json", dict(metrics))
    return unit


def write_runtime(root: str | Path, runtime: Mapping[str, Any]) -> Path:
    path = Path(root) / "runtime.json"
    _atomic_json(path, dict(runtime))
    return path


def finalize_aggregate(
    root: str | Path,
    *,
    expected_units: Iterable[str],
    unit_status: Mapping[str, str],
    aggregate: Mapping[str, Any],
) -> Path:
    """Publish aggregate.json only when every declared unit passed."""

    expected = tuple(expected_units)
    missing = [unit for unit in expected if unit_status.get(unit) != "PASS"]
    if missing:
        raise RuntimeError(f"cannot publish aggregate; incomplete units: {missing}")
    path = Path(root) / "aggregate.json"
    _atomic_json(path, dict(aggregate))
    return path


__all__ = [
    "create_manifest",
    "finalize_aggregate",
    "write_runtime",
    "write_unit",
]
