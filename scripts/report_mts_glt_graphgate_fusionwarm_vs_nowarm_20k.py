#!/usr/bin/env python3
"""Report the paired Warm5 minus Warm0 GraphGate schedule ablation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/mts/glt_graphgate_v1/fusionwarm_vs_nowarm_20k_v1.json"
OUTPUT = ROOT / "results/mts_glt_graphgate_v1/fusionwarm_vs_nowarm_20k"
WARM5 = ROOT / "results/mts_glt_graphgate_v1/fusion_realization_v1/folds"
TASKS = ("xc", "ei", "egc")
FOLDS = (0, 1, 2)


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _flatten_tensors(value, prefix=""):
    if torch.is_tensor(value):
        return {prefix: value}
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(_flatten_tensors(item, f"{prefix}.{key}" if prefix else str(key)))
        return result
    return {}


def _verify_checkpoint(config):
    published = torch.load(ROOT / config["checkpoint"], map_location="cpu", weights_only=False)
    probe = torch.load(
        ROOT / "results/mts_glt_graphgate_v1/pretrain/mts_glt_graphgate_probe_020k.pth",
        map_location="cpu", weights_only=False,
    )
    left, right = _flatten_tensors(published), _flatten_tensors(probe)
    if set(left) != set(right) or any(not torch.equal(left[key], right[key]) for key in left):
        raise SystemExit("published 20k checkpoint differs from probe_020k tensor state")


def _load(root, task, fold, expected_warm_epochs):
    shard = root / "shards/42" / task / f"fold_{fold}.csv"
    prediction = root / "predictions/42" / task / f"fold_{fold}.npz"
    if not shard.is_file() or not prediction.is_file():
        raise SystemExit(f"missing unit: {root} {task}/fold{fold}")
    row = pd.read_csv(shard).iloc[0]
    with np.load(prediction, allow_pickle=False) as payload:
        metadata = json.loads(str(np.asarray(payload["metadata"]).item()))
        y_true = np.asarray(payload["y_true"], dtype=np.float64)
        y_pred = np.asarray(payload["y_pred"], dtype=np.float64)
        indices = np.asarray(payload["sample_indices"], dtype=np.int64)
    expected_checkpoint = (
        ROOT / "pretrained_models/mts_glt_graphgate_v1/mts_glt_graphgate_v1.pth"
    ).resolve()
    checks = {
        "task": str(row["task"]) == task,
        "seed": int(row["seed"]) == 42,
        "precision": str(row["amp_dtype"]) == "fp32",
        "batches": int(row["train_batch_size"]) == 32 and int(row["eval_batch_size"]) == 64,
        "mode": str(row["mts_glt_mode"]) == "o8_glt_graph",
        "geometry": str(row["mts_glt_geometry_mode"]) == "full",
        "fusion": str(row["mts_glt_fusion_strategy"]) == "fusion_warm",
        "warm_epochs": int(row["mts_glt_fusion_warm_epochs"]) == expected_warm_epochs,
        "alpha": np.isclose(float(row["mts_glt_initial_alpha"]), 0.05),
        "loss": str(row["regression_loss"]) == "huber" and np.isclose(float(row["huber_beta"]), 0.5),
        "protocol": str(row["evaluation_protocol"]) == "historical_shared5",
        "checkpoint": Path(str(row["checkpoint_path"])).resolve() == expected_checkpoint,
        "fold": metadata["task"] == task and int(metadata["fold"]) == fold,
    }
    if expected_warm_epochs == 0:
        checks["joint_from_start"] = str(
            row["mts_glt_fusion_stage2_trainability"]
        ) == "joint"
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"protocol mismatch {task}/fold{fold}: {failed}")
    if not np.isfinite(y_pred).all() or y_true.shape != y_pred.shape or y_true.shape != indices.shape:
        raise SystemExit(f"invalid predictions {task}/fold{fold}")
    return {
        "r2": float(r2_score(y_true, y_pred)), "y_true": y_true,
        "indices": indices, "fold_seed": int(metadata["fold_seed"]),
    }


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    _verify_checkpoint(config)
    rows = []
    for task in TASKS:
        for fold in FOLDS:
            warm0 = _load(OUTPUT / "Warm0", task, fold, 0)
            warm5 = _load(WARM5, task, fold, 5)
            if warm0["fold_seed"] != warm5["fold_seed"]:
                raise SystemExit(f"fold seed mismatch {task}/fold{fold}")
            if not np.array_equal(warm0["indices"], warm5["indices"]):
                raise SystemExit(f"sample index mismatch {task}/fold{fold}")
            if not np.array_equal(warm0["y_true"], warm5["y_true"]):
                raise SystemExit(f"target mismatch {task}/fold{fold}")
            rows.append({
                "task": task, "fold": fold,
                "Warm0_R2": warm0["r2"], "Warm5_R2": warm5["r2"],
                "DeltaWarm": warm5["r2"] - warm0["r2"],
            })
    folds = pd.DataFrame(rows).sort_values(["task", "fold"])
    task_rows = []
    for task in TASKS:
        selected = folds[folds.task == task]
        task_rows.append({
            "task": task,
            "Warm0": float(selected.Warm0_R2.mean()),
            "Warm5": float(selected.Warm5_R2.mean()),
            "DeltaWarm": float(selected.DeltaWarm.mean()),
        })
    tasks = pd.DataFrame(task_rows)
    task_delta = tasks.DeltaWarm.to_numpy(dtype=np.float64)
    fold_delta = folds.DeltaWarm.to_numpy(dtype=np.float64)
    macro_delta = float(task_delta.mean())
    median_task = float(np.median(task_delta))
    positive_tasks = int((task_delta > 0).sum())
    decision = (
        "GO_candidate"
        if macro_delta > 0 and median_task > 0 and positive_tasks >= 2
        else "STOP_candidate"
    )
    summary = {
        "schema": "mts-glt-graphgate-fusionwarm-vs-nowarm-20k-summary-v1",
        "checkpoint": str((ROOT / config["checkpoint"]).resolve()),
        "new_warm0_runs": 9,
        "reused_warm5_runs": 9,
        "failed_runs": 0,
        "task_results": task_rows,
        "macro3_warm0": float(tasks.Warm0.mean()),
        "macro3_warm5": float(tasks.Warm5.mean()),
        "macro3_delta_warm": macro_delta,
        "median_task_delta_warm": median_task,
        "positive_tasks": positive_tasks,
        "positive_folds": int((fold_delta > 0).sum()),
        "fold_delta_median": float(np.median(fold_delta)),
        "fold_delta_p25": float(np.quantile(fold_delta, 0.25)),
        "fold_delta_p75": float(np.quantile(fold_delta, 0.75)),
        "warm_ablation_decision": decision,
        "shared_validation_test": True,
    }
    _atomic_text(OUTPUT / "per_fold_results.csv", folds.to_csv(index=False))
    _atomic_text(
        OUTPUT / "fusionwarm_vs_nowarm_20k_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
