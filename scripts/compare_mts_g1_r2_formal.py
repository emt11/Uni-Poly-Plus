#!/usr/bin/env python3
"""Strict formal comparison of historical G1 and Star-RBF v2 R2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_mts_t0_t1_formal import (
    FOLDS,
    TASKS,
    _read_arm,
    _write_json,
    compare,
    resolve,
    sha256,
)

G1_INIT = "mts_g_family_step0_v2_seed42"
R2_INIT = "mts_star_rbf_v2_r2_step0_seed42"


def _checkpoint(path: Path, resolved: dict, *, r2: bool) -> dict:
    path = path.resolve()
    complete = Path(str(path) + ".complete.json")
    if not path.is_file() or not complete.is_file():
        raise RuntimeError(f"checkpoint/completion marker missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    expected = {
        "g_family_arm": "g1",
        "model_identity": "T1",
        "initialization": "fresh_paired",
        "optimizer_steps": 20000,
        "smoke_only": False,
        "paired_init_id": R2_INIT if r2 else G1_INIT,
        "shared_step0_id": R2_INIT if r2 else G1_INIT,
        "pretraining_objective": "masked_atom_only",
        "g_family_bundle_hash": resolved["g_family_bundle_hash"],
        "experiment_id": resolved["experiment_id"],
    }
    if r2:
        expected.update({
            "star_rbf_definition": "trimer_periodic_relation_rbf_v2",
            "star_rbf_upper": float(resolved["star_rbf_v2_upper"]),
            "star_rbf_v2_artifact_hash": resolved["star_rbf_v2_bundle"]["PI1M_v2"]["artifact_hash"],
            "star_rbf_v2_model_semantic_hash": resolved["star_rbf_v2_model_semantic_hash"],
            "backbone_definition": "legacy_g1_frozen",
        })
    mismatch = {}
    for key, value in expected.items():
        observed = meta.get(key)
        equal = float(observed) == value if key == "star_rbf_upper" and observed is not None else observed == value
        if not equal:
            mismatch[key] = (observed, value)
    if mismatch:
        raise RuntimeError(f"checkpoint identity mismatch: {mismatch}")
    state = payload.get("state_dict") or {}
    if not state or any(
        not torch.isfinite(value).all().item()
        for value in state.values()
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex())
    ):
        raise RuntimeError(f"checkpoint has missing/non-finite state: {path}")
    completion = json.loads(complete.read_text(encoding="utf-8"))
    checkpoint_sha = sha256(path)
    if completion.get("optimizer_steps") != 20000 or completion.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError(f"completion marker mismatch: {path}")
    return {
        "path": str(path), "sha256": checkpoint_sha,
        "training_wall_seconds": meta.get("training_wall_seconds"),
        "identity": expected,
    }


def _validate_rows(info: dict, resolved: dict, *, r2: bool) -> None:
    for key, record in info["records"].items():
        row = record["row"]
        expected = {
            "g_family_arm": "g1",
            "g_family_bundle_hash": resolved["g_family_bundle_hash"],
            "checkpoint_g_family_bundle_hash": resolved["g_family_bundle_hash"],
            "shared_step0_id": R2_INIT if r2 else G1_INIT,
            "relation_geometry_artifact_hash": resolved["relation_geometry_bundle"]["downstream_union"]["artifact_hash"],
        }
        if r2:
            expected.update({
                "star_rbf_definition": "trimer_periodic_relation_rbf_v2",
                "star_rbf_v2_artifact_hash": resolved["star_rbf_v2_bundle"]["downstream_union"]["artifact_hash"],
                "checkpoint_star_rbf_v2_artifact_hash": resolved["star_rbf_v2_bundle"]["PI1M_v2"]["artifact_hash"],
                "star_rbf_v2_source_artifact_hash": resolved["star_rbf_v2_bundle"]["PI1M_v2"]["artifact_hash"],
                "star_rbf_v2_model_semantic_hash": resolved["star_rbf_v2_model_semantic_hash"],
                "backbone_definition": "legacy_g1_frozen",
            })
        mismatch = {field: (row.get(field), value) for field, value in expected.items() if str(row.get(field)) != str(value)}
        if mismatch:
            raise RuntimeError(f"shard identity mismatch {key}: {mismatch}")


def _promotion(stats: dict, comparison: dict) -> dict:
    without_xc = next(row for row in comparison["leave_one_task_out"] if row["excluded_task"] == "xc")
    gates = {
        "task_delta_mean_positive": float(stats["mean"]) > 0.0,
        "task_delta_median_positive": float(stats["median"]) > 0.0,
        "positive_tasks_at_least_5_of_8": int(stats["positive_count"]) >= 5,
        "positive_without_xc": float(without_xc["delta_r2_mean"]) > 0.0,
    }
    return {"gates": gates, "passed": all(gates.values()), "codex_decision_required": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1-root", type=Path, required=True)
    parser.add_argument("--r2-root", type=Path, required=True)
    parser.add_argument("--g1-checkpoint", type=Path, required=True)
    parser.add_argument("--r2-checkpoint", type=Path, required=True)
    parser.add_argument("--g1-config", type=Path, default=ROOT / "configs/mts/experiments/G1_t1_msta_angle_matched_formal_v1.json")
    parser.add_argument("--r2-config", type=Path, default=ROOT / "configs/mts/experiments/R2_g1_periodic_relation_rbf_v2_legacy_backbone_formal_v1.json")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    g1_config, r2_config = resolve(args.g1_config), resolve(args.r2_config)
    g1 = _read_arm(
        name="G1", root=args.g1_root.resolve(), config_path=args.g1_config.resolve(),
        checkpoint=args.g1_checkpoint, expected_model_identity="T1", allow_init=False,
        workers=args.workers, fresh_paired=True, paired_init_id=G1_INIT, expected_t0_sha256=None,
    )
    r2 = _read_arm(
        name="R2", root=args.r2_root.resolve(), config_path=args.r2_config.resolve(),
        checkpoint=args.r2_checkpoint, expected_model_identity="T1", allow_init=False,
        workers=args.workers, fresh_paired=True, paired_init_id=R2_INIT, expected_t0_sha256=None,
    )
    g1_checkpoint = _checkpoint(args.g1_checkpoint, g1_config, r2=False)
    r2_checkpoint = _checkpoint(args.r2_checkpoint, r2_config, r2=True)
    _validate_rows(g1, g1_config, r2=False)
    _validate_rows(r2, r2_config, r2=True)
    if g1_config["feature_config_hash"] != r2_config["feature_config_hash"]:
        raise RuntimeError("G1/R2 frozen feature/cache identity differs")
    if g1_config["relation_geometry_bundle"] != r2_config["relation_geometry_bundle"]:
        raise RuntimeError("G1/R2 G1 path-cosine sidecar differs")
    result = compare(g1, r2)
    stats = result["task_delta_statistics"]
    promotion = _promotion(stats, result)
    payload = {
        "schema": "mts-g1-r2-formal-comparison-v1",
        "experiment_id": r2_config["experiment_id"],
        "status": "formal_comparison_verified",
        "arms": {"G1": {"config": g1_config, "checkpoint": g1_checkpoint, "counts": g1["counts"]},
                 "R2": {"config": r2_config, "checkpoint": r2_checkpoint, "counts": r2["counts"]}},
        "comparison": result, "promotion": promotion,
        "limitations": ["historical_shared5 is a shared validation/test fold, not an independent blind test.",
                        "R2 evidence applies to legacy_g1_frozen backbone semantics."],
    }
    _write_json(args.output_json.resolve(), payload)
    rows = []
    for row in result["task_rows"]:
        rows.append({"task": row["task"], "g1_r2_mean": row["t0"]["r2_mean"],
                     "g1_r2_std": row["t0"]["r2_std"], "r2_r2_mean": row["t1"]["r2_mean"],
                     "r2_r2_std": row["t1"]["r2_std"], "r2_minus_g1": row["delta_t"],
                     "improved_folds": sum(value > 0 for value in row["paired_delta_values"]["r2"])})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False, float_format="%.17g")
    lines = ["# G1 vs Star-RBF v2 R2 formal comparison", "", "| task | G1 R2 | R2 R2 | delta | improved folds |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['task']} | {row['g1_r2_mean']:.6f} | {row['r2_r2_mean']:.6f} | {row['r2_minus_g1']:.6f} | {row['improved_folds']}/5 |")
    lines += ["", f"Task delta mean/median: {stats['mean']:.6f} / {stats['median']:.6f}",
              f"Positive tasks: {stats['positive_count']}/8", f"Promotion gates passed: {promotion['passed']}",
              "", "R2 remains a legacy-backbone experiment; production promotion requires Codex review."]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "promotion": promotion, "output": str(args.output_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
