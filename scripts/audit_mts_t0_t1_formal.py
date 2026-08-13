#!/usr/bin/env python3
"""Audit the isolated identity and reuse boundary for the formal T-Eval run.

The audit is deliberately read-only with respect to production artifacts.  It
records the historical shard identities that were considered for reuse, the
resolved T0/T1 contracts, the immutable source checkpoint/cache hashes, and
the strict T1 initialization evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("PYTHON", "/opt/conda/envs/MTS/bin/python")
TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
FOLDS = tuple(range(5))
SOURCE_CHECKPOINT_SHA256 = (
    "56c8a0ba148280fef22ecb953705a14259fb5e87258c0cae67799d7484375e77"
)
HISTORICAL_ROOTS = (
    ROOT / "results/mts_sota_v3/G0_canonical_angle20_v1",
    ROOT / "results/mts_geometry_injection_ablation_v1/A3_star_mcl_real",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def resolve_config(path: Path) -> dict[str, Any]:
    output = subprocess.check_output(
        [PYTHON, "scripts/resolve_mips_trimer_scage.py", str(path)],
        cwd=ROOT,
        text=True,
    )
    return json.loads(output)


def _read_first_shard(root: Path) -> tuple[dict[str, Any] | None, Path | None]:
    paths = sorted((root / "shards" / "42").glob("*/fold_*.csv"))
    if not paths:
        return None, None
    frame = pd.read_csv(paths[0])
    if len(frame) != 1:
        raise RuntimeError(f"historical shard is not one row: {paths[0]}")
    return frame.iloc[0].to_dict(), paths[0]


def _artifact_counts(root: Path) -> dict[str, Any]:
    expected = {
        (task, fold)
        for task in TASKS
        for fold in FOLDS
    }
    shards = sorted((root / "shards" / "42").glob("*/fold_*.csv"))
    predictions = sorted((root / "predictions" / "42").glob("*/fold_*.npz"))
    shard_keys = set()
    for path in shards:
        try:
            task = path.parent.name
            fold = int(path.stem.removeprefix("fold_"))
        except ValueError:
            continue
        shard_keys.add((task, fold))
    prediction_keys = set()
    for path in predictions:
        try:
            task = path.parent.name
            fold = int(path.stem.removeprefix("fold_"))
        except ValueError:
            continue
        prediction_keys.add((task, fold))
    return {
        "shard_count": len(shards),
        "prediction_count": len(predictions),
        "expected_shard_count": len(expected),
        "expected_prediction_count": len(expected),
        "shard_key_complete": shard_keys == expected,
        "prediction_key_complete": prediction_keys == expected,
        "shard_keys": sorted([list(key) for key in shard_keys]),
        "prediction_keys": sorted([list(key) for key in prediction_keys]),
    }


def _historical_identity(root: Path) -> dict[str, Any]:
    row, path = _read_first_shard(root)
    counts = _artifact_counts(root)
    fields = (
        "experiment_id",
        "config_hash",
        "resolved_config_hash",
        "feature_config_hash",
        "graph_model_config_hash",
        "geometry_model_config_hash",
        "source_geometry_model_config_hash",
        "training_config_hash",
        "finetune_config_hash",
        "finetune_profile_hash",
        "checkpoint_sha256",
        "cache_store_sha256",
        "evaluation_protocol",
        "amp_dtype",
        "batch_size",
        "eval_batch_size",
        "seed",
    )
    identity = {field: (row or {}).get(field) for field in fields}
    return {
        "root": str(root),
        "exists": root.is_dir(),
        "counts": counts,
        "first_shard": str(path) if path else None,
        "identity": identity,
        "reusable_as_matched_control": False,
        "reuse_reason": (
            "historical artifacts use a different experiment/config/training "
            "identity than this cycle; they remain read-only reference data"
        ),
    }


def _find_cache_files() -> dict[str, Any]:
    roots = ROOT / "data/processed/mips_trimer_scage"
    matches: list[dict[str, str]] = []
    if roots.is_dir():
        for path in sorted(roots.rglob(".done")) + sorted(roots.rglob(".frozen")):
            try:
                matches.append({"path": str(path), "sha256": sha256(path)})
            except OSError:
                continue
    stores = []
    if roots.is_dir():
        for path in sorted(roots.rglob("validation/store.json")):
            stores.append({"path": str(path), "sha256": sha256(path)})
    return {"marker_files": matches, "store_files": stores}


def build_reuse_audit(
    *, source_checkpoint: Path, t0_config: Path, t1_config: Path
) -> dict[str, Any]:
    source_sha = sha256(source_checkpoint)
    source_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    source_meta = dict(source_payload.get("meta") or {})
    best_result = ROOT / "results/best_result.csv"
    configs = {"t0": resolve_config(t0_config), "t1": resolve_config(t1_config)}
    return {
        "schema": "mts-t0-t1-reuse-audit-v1",
        "cycle_id": "mts_t0_t1_fixed_pretraining_formal_v2",
        "phase": "T-Eval",
        "historical_roots": [_historical_identity(path) for path in HISTORICAL_ROOTS],
        "new_protocol": {
            "t0_config": str(t0_config),
            "t1_config": str(t1_config),
            "t0_config_hash": configs["t0"]["config_hash"],
            "t1_config_hash": configs["t1"]["config_hash"],
            "t0_graph_model_config_hash": configs["t0"]["graph_model_config_hash"],
            "t1_graph_model_config_hash": configs["t1"]["graph_model_config_hash"],
            "evaluation_protocol": "historical_shared5",
            "seed": 42,
            "tasks": list(TASKS),
            "folds": list(FOLDS),
            "train_batch_size": 32,
            "eval_batch_size": 64,
            "amp_dtype": "fp32",
            "dataloader_workers": 2,
            "schedule": "lpt_v1",
        },
        "source_checkpoint": {
            "path": str(source_checkpoint),
            "sha256": source_sha,
            "expected_sha256": SOURCE_CHECKPOINT_SHA256,
            "sha256_matches_contract": source_sha == SOURCE_CHECKPOINT_SHA256,
            "model_identity": source_meta.get("model_identity", "T0"),
            "graph_model_config_hash": source_meta.get("graph_model_config_hash"),
            "cache_bundle_hash": source_meta.get("cache_bundle_hash"),
            "topology_cache_artifact_hash": source_meta.get("topology_cache_artifact_hash"),
            "trimer_cache_artifact_hash": source_meta.get("trimer_cache_artifact_hash"),
            "source_cohort_hash": source_meta.get("source_cohort_hash"),
            "optimizer_steps": source_meta.get("optimizer_steps"),
        },
        "production_artifacts_before": {
            "best_result_csv": {
                "path": str(best_result),
                "exists": best_result.is_file(),
                "sha256": sha256(best_result) if best_result.is_file() else None,
            },
            "source_checkpoint": {
                "path": str(source_checkpoint),
                "sha256": source_sha,
            },
            "cache": _find_cache_files(),
        },
        "reusable_as_matched_control": False,
        "historical_artifacts_modified": False,
        "best_result_updated": False,
    }


def _diff_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if type(left) is not type(right):
        return [prefix or "<root>"]
    if isinstance(left, dict):
        paths = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_diff_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list):
        if len(left) != len(right):
            return [prefix]
        paths = []
        for index, (a, b) in enumerate(zip(left, right)):
            paths.extend(_diff_paths(a, b, f"{prefix}[{index}]"))
        return paths
    return [] if left == right else [prefix]


def _init_evidence(
    *, init_checkpoint: Path, source_checkpoint: Path, target_config: Path,
    target_resolved: dict[str, Any]
) -> dict[str, Any]:
    payload = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    state = dict(payload.get("state_dict") or {})
    local_outputs = {}
    for index in (4, 5):
        key = f"encoders.graph.encoder.layers.{index}.attention.local_output.weight"
        value = state.get(key)
        local_outputs[str(index)] = {
            "present": value is not None,
            "shape": list(value.shape) if value is not None else None,
            "zero": bool(value is not None and torch.count_nonzero(value).item() == 0),
            "finite": bool(value is not None and torch.isfinite(value).all().item()),
        }
    manifest_path = init_checkpoint.with_suffix(init_checkpoint.suffix + ".initialization.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    source_sha = sha256(source_checkpoint)
    return {
        "path": str(init_checkpoint),
        "sha256": sha256(init_checkpoint),
        "manifest": manifest,
        "meta": {
            key: meta.get(key)
            for key in (
                "model_identity", "source_model_identity", "topology_attention_variant",
                "graph_model_config_hash", "source_graph_model_config_hash",
                "target_config_hash", "initialization", "init_artifact",
                "optimizer_state_inherited", "scheduler_state_inherited",
                "sampler_state_inherited", "source_optimizer_steps", "optimizer_steps",
                "pretraining_status", "parent_checkpoint_sha256",
            )
        },
        "strict_parent_sha256": meta.get("parent_checkpoint_sha256") == source_sha,
        "strict_target_config_hash": meta.get("target_config_hash") == target_resolved["config_hash"],
        "strict_target_graph_hash": meta.get("graph_model_config_hash") == target_resolved["graph_model_config_hash"],
        "strict_source_graph_hash": meta.get("source_graph_model_config_hash") is not None,
        "local_outputs": local_outputs,
        "all_local_outputs_zero": all(item["zero"] for item in local_outputs.values()),
        "strict_load_marker": bool(manifest and manifest.get("strict_graph_load")),
        "source_checkpoint_sha256": source_sha,
        "target_config": str(target_config),
    }


def build_identity_audit(
    *, t0_config: Path, t1_config: Path, source_checkpoint: Path,
    init_checkpoint: Path | None,
) -> dict[str, Any]:
    t0_raw = json.loads(t0_config.read_text())
    t1_raw = json.loads(t1_config.read_text())
    t0 = resolve_config(t0_config)
    t1 = resolve_config(t1_config)
    raw_diff = _diff_paths(t0_raw, t1_raw)
    resolved_diff = _diff_paths(t0, t1)
    allowed_raw = {"experiment_id", "topology_attention_variant"}
    allowed_resolved = {
        "config_path", "config_hash", "experiment_id", "graph_model_config_hash",
        "topology_attention_variant",
    }
    init = None
    if init_checkpoint is not None and init_checkpoint.is_file():
        init = _init_evidence(
            init_checkpoint=init_checkpoint,
            source_checkpoint=source_checkpoint,
            target_config=t1_config,
            target_resolved=t1,
        )
    return {
        "schema": "mts-t0-t1-identity-audit-v1",
        "cycle_id": "mts_t0_t1_fixed_pretraining_formal_v2",
        "phase": "T-Eval",
        "raw_configs": {
            "t0": {"path": str(t0_config), "sha256": sha256(t0_config), "resolved": t0},
            "t1": {"path": str(t1_config), "sha256": sha256(t1_config), "resolved": t1},
        },
        "allowed_raw_differences": sorted(allowed_raw),
        "observed_raw_differences": sorted(raw_diff),
        "allowed_resolved_differences": sorted(allowed_resolved),
        "observed_resolved_differences": sorted(resolved_diff),
        "raw_configs_match_except_declared": set(raw_diff) == allowed_raw,
        "resolved_contracts_match_except_declared": set(resolved_diff) <= allowed_resolved,
        "shared_resolved_identity": {
            key: t0[key]
            for key in (
                "feature_config_hash", "geometry_model_config_hash",
                "source_geometry_model_config_hash", "topology_representation",
                "graph_geometry_mode", "evaluation_protocol", "finetune_profile",
                "modalities", "fusion_mode", "finetune_mode",
            )
        },
        "t0_variant": t0["topology_attention_variant"],
        "t1_variant": t1["topology_attention_variant"],
        "source_checkpoint": {
            "path": str(source_checkpoint),
            "sha256": sha256(source_checkpoint),
            "expected_sha256": SOURCE_CHECKPOINT_SHA256,
            "sha256_matches_contract": sha256(source_checkpoint) == SOURCE_CHECKPOINT_SHA256,
        },
        "t1_initialization": init,
        "passed": bool(
            set(raw_diff) == allowed_raw
            and set(resolved_diff) <= allowed_resolved
            and t0["topology_attention_variant"] == "o8"
            and t1["topology_attention_variant"] == "msta_last2"
            and sha256(source_checkpoint) == SOURCE_CHECKPOINT_SHA256
            and (init is None or (
                init["strict_parent_sha256"]
                and init["strict_target_config_hash"]
                and init["strict_target_graph_hash"]
                and init["all_local_outputs_zero"]
                and init["strict_load_marker"]
                and init["meta"].get("source_optimizer_steps") == 20000
                and init["meta"].get("optimizer_steps") == 0
            ))
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("reuse", "identity", "all"), default="all")
    parser.add_argument("--source-checkpoint", type=Path, default=ROOT / "pretrained_models/mts/mts_joint_pretraining_pi1m_v2_seed42_canonical_angle20_v1.pth")
    parser.add_argument("--t0-config", type=Path, default=ROOT / "configs/mts/experiments/T0_o8_matched_t1_formal_v1.json")
    parser.add_argument("--t1-config", type=Path, default=ROOT / "configs/mts/experiments/T1_msta_formal_v1.json")
    parser.add_argument("--init-checkpoint", type=Path, default=ROOT / "pretrained_models/mts_multiscale_topology/t1_formal_v1/mts_t1_formal_function_preserving_init.pth")
    parser.add_argument("--reuse-output", type=Path, default=ROOT / "results/mts_multiscale_topology/t0_t1_formal_v1/reuse_audit.json")
    parser.add_argument("--identity-output", type=Path, default=ROOT / "results/mts_multiscale_topology/t0_t1_formal_v1/identity_audit.json")
    args = parser.parse_args()
    source = args.source_checkpoint.resolve()
    t0 = args.t0_config.resolve()
    t1 = args.t1_config.resolve()
    if args.mode in {"reuse", "all"}:
        write_json(args.reuse_output.resolve(), build_reuse_audit(source_checkpoint=source, t0_config=t0, t1_config=t1))
    if args.mode in {"identity", "all"}:
        init = args.init_checkpoint.resolve()
        payload = build_identity_audit(
            t0_config=t0,
            t1_config=t1,
            source_checkpoint=source,
            init_checkpoint=init if init.is_file() else None,
        )
        write_json(args.identity_output.resolve(), payload)
        if not payload["passed"]:
            raise SystemExit("formal T0/T1 identity audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
