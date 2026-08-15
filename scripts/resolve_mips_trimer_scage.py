#!/usr/bin/env python3
"""Resolve the explicit B0 experiment JSON into an immutable run input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "schema", "experiment_id", "dataset_name", "cache_root", "sidecar_root",
    "result_root", "output_path", "output_kind", "noise_sigma",
    "coordinate_loss_weight", "masked_atom", "coordinate_denoising",
    "use_star_rbf", "use_mcl", "use_md200", "topology_attention_variant",
    "star_rbf_upper", "graph_mask_ratio", "batch_size", "loader_workers",
    "prefetch_factor", "gradient_accumulation_steps", "global_batch_size",
    "max_optimizer_steps", "checkpoint_interval_steps", "probe_steps",
    "amp_dtype", "seed", "lr", "warmup_steps", "end_lr", "weight_decay",
    "cache_layers",
}
STAR_RBF_UPPER = 3.75


def _project_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())


def resolve(source: Path) -> tuple[Path, dict]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("B0 experiment must be a JSON object")
    unknown = sorted(set(payload) - REQUIRED)
    missing = sorted(REQUIRED - set(payload))
    if unknown:
        raise ValueError("unknown B0 fields: " + ", ".join(unknown))
    if missing:
        raise ValueError("missing B0 fields: " + ", ".join(missing))
    if payload["schema"] != "mts-b0-v2":
        raise ValueError("only schema mts-b0-v2 is active")
    if payload["use_star_rbf"] is not True or payload["use_mcl"] is not False:
        raise ValueError("B0 requires use_star_rbf=true and use_mcl=false")
    if payload["use_md200"] is not True:
        raise ValueError("B0 downstream contract requires use_md200=true")
    if payload["masked_atom"] is not True or payload["coordinate_denoising"] is not True:
        raise ValueError("B0 requires masked atom and coordinate denoising")
    if payload["topology_attention_variant"] != "o8":
        raise ValueError("B0 topology_attention_variant must be o8")
    if abs(float(payload["star_rbf_upper"]) - STAR_RBF_UPPER) > 1e-9:
        raise ValueError(
            f"B0 requires star_rbf_upper={STAR_RBF_UPPER}, "
            f"got {payload['star_rbf_upper']}"
        )
    if float(payload["noise_sigma"]) <= 0 or float(payload["coordinate_loss_weight"]) <= 0:
        raise ValueError("B0 noise_sigma and coordinate_loss_weight must be positive")
    if int(payload["checkpoint_interval_steps"]) != 2000:
        raise ValueError("B0 checkpoint_interval_steps is fixed at 2000")
    if [int(value) for value in payload["probe_steps"]] != [5000, 10000, 20000]:
        raise ValueError("B0-v2 probe_steps must be exactly [5000, 10000, 20000]")
    if int(payload["gradient_accumulation_steps"]) < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if int(payload["global_batch_size"]) < 1:
        raise ValueError("global_batch_size must be positive")
    if int(payload["batch_size"]) < 1 or int(payload["loader_workers"]) < 0:
        raise ValueError("B0 batch_size/loader_workers are invalid")
    if int(payload["prefetch_factor"]) < 1:
        raise ValueError("B0 prefetch_factor must be positive")
    if int(payload["max_optimizer_steps"]) < 1:
        raise ValueError("B0 max_optimizer_steps must be positive")
    if payload["output_kind"] != "trajectory":
        raise ValueError("B0-v2 output_kind must be trajectory")
    resolved = dict(payload)
    for name in ("cache_root", "sidecar_root", "result_root", "output_path"):
        resolved[name] = _project_path(str(payload[name]))
    result_root = Path(resolved["result_root"])
    result_root.mkdir(parents=True, exist_ok=True)
    output = result_root / "resolved_input.json"
    resolved["source_config"] = str(source.resolve())
    resolved["project_root"] = str(PROJECT_ROOT)
    output.write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output, resolved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--print-path", action="store_true")
    ns = parser.parse_args(argv)
    try:
        output, _ = resolve(ns.config.resolve())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"No active MTS configuration schema: {exc}")
        return 2
    if ns.print_path:
        print(output)
    else:
        print(f"resolved_input={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
