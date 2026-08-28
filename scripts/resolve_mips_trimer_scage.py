#!/usr/bin/env python3
"""Resolve and validate an MTS-GLT-v2 pretraining configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "schema", "experiment_id", "dataset_name", "cache_root", "line_sidecar_root",
    "line_label_counts", "result_root", "output_path", "output_kind",
    "masked_atom", "masked_line", "infonce", "use_star_rbf", "use_mcl",
    "use_md200", "coordinate_denoising", "topology_attention_variant",
    "atom_mask_ratio", "line_mask_ratio", "infonce_temperature",
    "atom_loss_weight", "line_loss_weight", "infonce_loss_weight",
    "projection_dim", "batch_size", "loader_workers", "prefetch_factor",
    "gradient_accumulation_steps", "global_batch_size", "max_optimizer_steps",
    "stop_after_steps", "checkpoint_interval_steps", "probe_steps", "amp_dtype",
    "seed", "lr", "warmup_steps", "end_lr", "weight_decay", "cache_layers",
    "glt_layers", "glt_attention_variant",
}


def _project_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())


def resolve(source: Path) -> tuple[Path, dict]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("MTS-GLT-v2 configuration must be a JSON object")
    if payload.get("schema") != "mts-glt-v2":
        raise ValueError("only schema=mts-glt-v2 is retained")
    unknown = sorted(set(payload) - REQUIRED)
    missing = sorted(REQUIRED - set(payload))
    if unknown:
        raise ValueError("unknown MTS-GLT-v2 fields: " + ", ".join(unknown))
    if missing:
        raise ValueError("missing MTS-GLT-v2 fields: " + ", ".join(missing))
    if not all(payload[name] is True for name in ("masked_atom", "masked_line", "infonce")):
        raise ValueError("MTS-GLT-v2 enables masked atom, masked line and InfoNCE")
    if not all(payload[name] is False for name in ("use_star_rbf", "use_mcl", "use_md200", "coordinate_denoising")):
        raise ValueError("MTS-GLT-v2 pretraining keeps Star-RBF, MCL, MD200 and coordinate denoising off")
    fixed = {
        "topology_attention_variant": "o8",
        "atom_mask_ratio": 0.30,
        "line_mask_ratio": 0.40,
        "glt_layers": 6,
        "glt_attention_variant": "mips",
        "checkpoint_interval_steps": 2000,
        "output_kind": "trajectory",
    }
    for name, expected in fixed.items():
        observed = payload.get(name)
        if isinstance(expected, float):
            if abs(float(observed) - expected) > 1e-12:
                raise ValueError(f"{name} must be {expected}")
        elif observed != expected:
            raise ValueError(f"{name} must be {expected!r}")
    if int(payload["stop_after_steps"]) < 1 or int(payload["stop_after_steps"]) > int(payload["max_optimizer_steps"]):
        raise ValueError("stop_after_steps must be within max_optimizer_steps")
    probes = [int(value) for value in payload["probe_steps"]]
    if not probes or any(value < 1 or value > int(payload["stop_after_steps"]) for value in probes):
        raise ValueError("probe_steps must fall within the current run")
    if int(payload["gradient_accumulation_steps"]) < 1 or int(payload["global_batch_size"]) < 1:
        raise ValueError("batch and accumulation values must be positive")
    if int(payload["batch_size"]) < 1 or int(payload["loader_workers"]) < 0 or int(payload["prefetch_factor"]) < 1:
        raise ValueError("batch_size/loader_workers/prefetch_factor are invalid")
    resolved = dict(payload)
    for name in ("cache_root", "result_root", "output_path", "line_sidecar_root", "line_label_counts"):
        resolved[name] = _project_path(str(payload[name]))
    result_root = Path(resolved["result_root"])
    result_root.mkdir(parents=True, exist_ok=True)
    output = result_root / "resolved_input.json"
    resolved["source_config"] = str(source.resolve())
    resolved["project_root"] = str(PROJECT_ROOT)
    output.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output, resolved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--print-path", action="store_true")
    ns = parser.parse_args(argv)
    try:
        output, _ = resolve(ns.config.resolve())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"No active MTS-GLT-v2 configuration: {exc}")
        return 2
    print(output if ns.print_path else f"resolved_input={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
