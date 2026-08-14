"""Atomic checkpoint lifecycle helpers.

These helpers deliberately contain no experiment or artifact identity logic.
The resumable file is a runtime state snapshot; the final file is a minimal
downstream state dict and the completion marker is only a status signal.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch


TRAIN_STATE_SCHEMA = "mts-train-state-v3"


def atomic_torch_save(payload: Any, path: str | os.PathLike[str]) -> None:
    """Write a torch payload and atomically publish it at *path*."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_resume_state(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and structurally validate a runtime resume state."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != TRAIN_STATE_SCHEMA:
        raise RuntimeError("resume state has an unsupported runtime schema")
    required = {
        "train_module", "optimizer", "scheduler", "epoch", "next_step_idx",
        "global_step", "optimizer_steps_completed", "rng_state_by_rank",
        "loader_generator_state_by_rank", "sampler_state_by_rank",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"resume state is missing runtime keys: {missing}")
    for key in ("epoch", "next_step_idx", "global_step", "optimizer_steps_completed"):
        if not isinstance(payload[key], int):
            raise RuntimeError(f"resume state field {key} must be an integer")
    return payload


def save_completion_marker(final_path: str | os.PathLike[str]) -> None:
    """Atomically publish the deliberately minimal completion marker."""

    final = Path(final_path)
    marker = Path(str(final) + ".complete.json")
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_name(f"{marker.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps({"status": "complete"}) + "\n", encoding="utf-8")
        os.replace(temporary, marker)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_completion_marker(final_path: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Return a valid completion marker, otherwise ``None``."""

    marker = Path(str(final_path) + ".complete.json")
    if not marker.is_file():
        return None
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) and value.get("status") == "complete" else None


def save_final_state(
    state_dict: Mapping[str, torch.Tensor],
    final_path: str | os.PathLike[str],
    strict_load: Any | None = None,
) -> None:
    """Write the minimal downstream checkpoint and publish its marker.

    ``strict_load`` is an optional callback supplied by the real model
    constructor.  It is intentionally a callback rather than a trainer or
    identity contract: the marker is published only after the caller's
    ``strict=True`` state-dict load has succeeded.
    """

    final = Path(final_path)
    marker = Path(str(final) + ".complete.json")
    if marker.exists():
        marker.unlink()
    payload = {"state_dict": {key: value.detach().cpu().clone() for key, value in state_dict.items()}}
    atomic_torch_save(payload, final)
    loaded = torch.load(final, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict) or not isinstance(loaded.get("state_dict"), dict):
        raise RuntimeError("final checkpoint did not round-trip as a state_dict")
    if strict_load is not None:
        strict_load(loaded["state_dict"])
    save_completion_marker(final)
