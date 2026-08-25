#!/usr/bin/env python3
"""Resolve an explicit active MTS experiment JSON into a run input."""

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
GLT_REQUIRED = {
    "schema", "experiment_id", "dataset_name", "cache_root", "line_sidecar_root",
    "result_root", "output_path", "output_kind", "masked_atom", "masked_line",
    "infonce", "use_star_rbf", "use_mcl", "use_md200", "coordinate_denoising",
    "topology_attention_variant", "atom_mask_ratio", "line_mask_ratio",
    "infonce_temperature", "atom_loss_weight", "line_loss_weight",
    "infonce_loss_weight", "projection_dim", "batch_size", "loader_workers",
    "prefetch_factor", "gradient_accumulation_steps", "global_batch_size",
    "max_optimizer_steps", "checkpoint_interval_steps", "probe_steps",
    "amp_dtype", "seed", "lr", "warmup_steps", "end_lr", "weight_decay",
    "cache_layers",
}
GLT_V2_REQUIRED = GLT_REQUIRED | {
    "line_label_counts", "glt_layers", "glt_attention_variant",
    "stop_after_steps",
}
STAR_RBF_UPPER = 3.75


def _project_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())


def resolve(source: Path) -> tuple[Path, dict]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("B0 experiment must be a JSON object")
    schema = payload.get("schema")
    required = (
        REQUIRED if schema == "mts-b0-v2"
        else GLT_REQUIRED if schema == "mts-glt-v1"
        else GLT_V2_REQUIRED if schema in {"mts-glt-v2", "mts-glt-graphgate-v1"}
        else set()
    )
    if not required:
        raise ValueError(f"unsupported MTS schema: {schema}")
    optional = (
        {"glt_geometry_mode", "initial_state_path"}
        if schema == "mts-glt-graphgate-v1" else set()
    )
    unknown = sorted(set(payload) - required - optional)
    missing = sorted(required - set(payload))
    if unknown:
        raise ValueError("unknown B0 fields: " + ", ".join(unknown))
    if missing:
        raise ValueError("missing B0 fields: " + ", ".join(missing))
    if schema == "mts-b0-v2":
        if payload["use_star_rbf"] is not True or payload["use_mcl"] is not False:
            raise ValueError("B0 requires use_star_rbf=true and use_mcl=false")
        if payload["use_md200"] is not True:
            raise ValueError("B0 downstream contract requires use_md200=true")
        if payload["masked_atom"] is not True or payload["coordinate_denoising"] is not True:
            raise ValueError("B0 requires masked atom and coordinate denoising")
        if abs(float(payload["star_rbf_upper"]) - STAR_RBF_UPPER) > 1e-9:
            raise ValueError(f"B0 requires star_rbf_upper={STAR_RBF_UPPER}")
        if float(payload["noise_sigma"]) <= 0 or float(payload["coordinate_loss_weight"]) <= 0:
            raise ValueError("B0 noise_sigma and coordinate_loss_weight must be positive")
    else:
        if not all(payload[name] is True for name in ("masked_atom", "masked_line", "infonce")):
            raise ValueError("GLT requires all three pretraining objectives")
        if not all(payload[name] is False for name in (
            "use_star_rbf", "use_mcl", "use_md200", "coordinate_denoising"
        )):
            raise ValueError("GLT requires Star-RBF/MCL/MD200/coordinate denoising off")
        if abs(float(payload["atom_mask_ratio"]) - 0.30) > 1e-12:
            raise ValueError("GLT atom mask ratio must be 0.30")
        if abs(float(payload["line_mask_ratio"]) - 0.40) > 1e-12:
            raise ValueError("GLT line mask ratio must be 0.40")
        if schema in {"mts-glt-v2", "mts-glt-graphgate-v1"}:
            if int(payload["glt_layers"]) not in {6, 12}:
                raise ValueError("GLT-v2 layers must be 6 or 12")
            if payload["glt_attention_variant"] not in {"mips", "paper"}:
                raise ValueError("GLT-v2 attention variant is invalid")
            if float(payload["infonce_loss_weight"]) not in {0.0, 0.25, 1.0}:
                raise ValueError("GLT-v2 InfoNCE weight must be 0, 0.25, or 1")
            if schema == "mts-glt-graphgate-v1" and (
                int(payload["glt_layers"]) != 6
                or payload["glt_attention_variant"] != "mips"
                or float(payload["infonce_loss_weight"]) != 1.0
            ):
                raise ValueError("GraphGate-v1 fixes 6 layers, MIPS attention and InfoNCE weight 1")
            if schema == "mts-glt-graphgate-v1" and payload.get("glt_geometry_mode", "full") not in {"full", "off"}:
                raise ValueError("GraphGate-v1 glt_geometry_mode must be full or off")
            stop_after = int(payload["stop_after_steps"])
            if not 1 <= stop_after <= int(payload["max_optimizer_steps"]):
                raise ValueError("GLT-v2 stop_after_steps is invalid")
    if payload["topology_attention_variant"] != "o8":
        raise ValueError("topology_attention_variant must be o8")
    if int(payload["checkpoint_interval_steps"]) != 2000:
        raise ValueError("checkpoint_interval_steps is fixed at 2000")
    probes = [int(value) for value in payload["probe_steps"]]
    if schema in {"mts-glt-v2", "mts-glt-graphgate-v1"}:
        if not probes or any(value < 1 or value > int(payload["stop_after_steps"]) for value in probes):
            raise ValueError("GLT-v2 probe_steps must fall within the current run")
    elif probes != [5000, 10000, 20000]:
        raise ValueError("probe_steps must be exactly [5000, 10000, 20000]")
    if int(payload["gradient_accumulation_steps"]) < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if int(payload["global_batch_size"]) < 1:
        raise ValueError("global_batch_size must be positive")
    if int(payload["batch_size"]) < 1 or int(payload["loader_workers"]) < 0:
        raise ValueError("batch_size/loader_workers are invalid")
    if int(payload["prefetch_factor"]) < 1:
        raise ValueError("B0 prefetch_factor must be positive")
    if int(payload["max_optimizer_steps"]) < 1:
        raise ValueError("B0 max_optimizer_steps must be positive")
    if payload["output_kind"] != "trajectory":
        raise ValueError("output_kind must be trajectory")
    resolved = dict(payload)
    path_names = ["cache_root", "result_root", "output_path"]
    path_names.append("sidecar_root" if schema == "mts-b0-v2" else "line_sidecar_root")
    if schema in {"mts-glt-v2", "mts-glt-graphgate-v1"}:
        path_names.append("line_label_counts")
    if schema == "mts-glt-graphgate-v1" and payload.get("initial_state_path"):
        path_names.append("initial_state_path")
    for name in path_names:
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
