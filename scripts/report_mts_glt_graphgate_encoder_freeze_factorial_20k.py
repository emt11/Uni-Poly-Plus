#!/usr/bin/env python3
"""Report the paired 20k FusionWarm encoder-trainability factorial."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/mts/glt_graphgate_v1/encoder_freeze_factorial_20k_v1.json"
OUTPUT = ROOT / "results/mts_glt_graphgate_v1/encoder_freeze_factorial_20k"
F3_ROOT = ROOT / "results/mts_glt_graphgate_v1/fusion_realization_v1/folds"
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
        merged = {}
        for key, item in value.items():
            merged.update(_flatten_tensors(item, f"{prefix}.{key}" if prefix else str(key)))
        return merged
    return {}


def _verify_checkpoint_family(config):
    published = torch.load(ROOT / config["checkpoint"], map_location="cpu", weights_only=False)
    probe = torch.load(
        ROOT / "results/mts_glt_graphgate_v1/pretrain/mts_glt_graphgate_probe_020k.pth",
        map_location="cpu", weights_only=False,
    )
    left, right = _flatten_tensors(published), _flatten_tensors(probe)
    if set(left) != set(right) or any(not torch.equal(left[key], right[key]) for key in left):
        raise SystemExit("published 20k checkpoint differs from probe_020k tensor state")
    if int(published.get("step", -1)) != 20000 or int(probe.get("step", -1)) != 20000:
        raise SystemExit("checkpoint is not a 20k probe")


def _load_unit(root, condition, task, fold):
    prediction = root / "predictions/42" / task / f"fold_{fold}.npz"
    shard = root / "shards/42" / task / f"fold_{fold}.csv"
    if not prediction.is_file() or not shard.is_file():
        raise SystemExit(f"missing {condition} unit: {task}/fold{fold}")
    row = pd.read_csv(shard).iloc[0]
    with np.load(prediction, allow_pickle=False) as payload:
        metadata = json.loads(str(np.asarray(payload["metadata"]).item()))
        y_true = np.asarray(payload["y_true"], dtype=np.float64)
        y_pred = np.asarray(payload["y_pred"], dtype=np.float64)
        indices = np.asarray(payload["sample_indices"], dtype=np.int64)
    expected_policy = {"F0": "both_frozen", "F1": "o8_only", "F2": "glt_query_only", "F3": "joint"}[condition]
    checks = {
        "task": str(row["task"]) == task,
        "seed": int(row["seed"]) == 42,
        "precision": str(row["amp_dtype"]) == "fp32",
        "train_batch": int(row["train_batch_size"]) == 32,
        "eval_batch": int(row["eval_batch_size"]) == 64,
        "mode": str(row["mts_glt_mode"]) == "o8_glt_graph",
        "geometry": str(row["mts_glt_geometry_mode"]) == "full",
        "fusion": str(row["mts_glt_fusion_strategy"]) == "fusion_warm",
        "warm_epochs": int(row["mts_glt_fusion_warm_epochs"]) == 5,
        "alpha": np.isclose(float(row["mts_glt_initial_alpha"]), 0.05),
        "loss": str(row["regression_loss"]) == "huber" and np.isclose(float(row["huber_beta"]), 0.5),
        "protocol": str(row["evaluation_protocol"]) == "historical_shared5",
        "checkpoint": Path(str(row["checkpoint_path"])).resolve() == (ROOT / "pretrained_models/mts_glt_graphgate_v1/mts_glt_graphgate_v1.pth").resolve(),
        "fold_metadata": metadata["task"] == task and int(metadata["fold"]) == fold and int(metadata["seed"]) == 42,
        "stage2_policy": (
            expected_policy == "joint"
            if "mts_glt_fusion_stage2_trainability" not in row.index
            else str(row["mts_glt_fusion_stage2_trainability"]) == expected_policy
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"{condition} protocol mismatch for {task}/fold{fold}: {failed}")
    if y_true.shape != y_pred.shape or y_true.shape != indices.shape or not np.isfinite(y_pred).all():
        raise SystemExit(f"invalid prediction payload: {condition} {task}/fold{fold}")
    return {
        "r2": float(r2_score(y_true, y_pred)), "y_true": y_true,
        "indices": indices, "fold_seed": int(metadata["fold_seed"]),
    }


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    _verify_checkpoint_family(config)
    rows = []
    for task in TASKS:
        for fold in FOLDS:
            units = {
                condition: _load_unit(
                    F3_ROOT if condition == "F3" else OUTPUT / condition,
                    condition, task, fold,
                )
                for condition in ("F0", "F1", "F2", "F3")
            }
            reference = units["F3"]
            for condition, unit in units.items():
                if unit["fold_seed"] != reference["fold_seed"]:
                    raise SystemExit(f"fold seed mismatch: {condition} {task}/fold{fold}")
                if not np.array_equal(unit["indices"], reference["indices"]):
                    raise SystemExit(f"sample index mismatch: {condition} {task}/fold{fold}")
                if not np.array_equal(unit["y_true"], reference["y_true"]):
                    raise SystemExit(f"target mismatch: {condition} {task}/fold{fold}")
            f0, f1, f2, f3 = (units[key]["r2"] for key in ("F0", "F1", "F2", "F3"))
            rows.append({
                "task": task, "fold": fold, "fold_seed": reference["fold_seed"],
                "F0": f0, "F1": f1, "F2": f2, "F3": f3,
                "C_O8": f1 - f0, "C_GLT": f2 - f0,
                "C_joint": f3 - f1 - f2 + f0, "C_total": f3 - f0,
                "F3_minus_F1": f3 - f1, "F3_minus_F2": f3 - f2,
            })
    folds = pd.DataFrame(rows).sort_values(["task", "fold"])
    task_rows = []
    metric_columns = ["F0", "F1", "F2", "F3", "C_O8", "C_GLT", "C_joint", "C_total", "F3_minus_F1", "F3_minus_F2"]
    for task in TASKS:
        selected = folds[folds.task == task]
        task_rows.append({"task": task, **{key: float(selected[key].mean()) for key in metric_columns}})
    tasks = pd.DataFrame(task_rows)
    contrasts = ["C_O8", "C_GLT", "C_joint", "C_total", "F3_minus_F1", "F3_minus_F2"]
    summary = {
        "schema": "mts-glt-graphgate-encoder-freeze-factorial-20k-summary-v1",
        "checkpoint": str((ROOT / config["checkpoint"]).resolve()),
        "f3_reused": True,
        "f3_source": str(F3_ROOT.resolve()),
        "new_runs": {"F0": 9, "F1": 9, "F2": 9, "F3": 0},
        "failed_runs": 0,
        "task_results": tasks.to_dict(orient="records"),
        "macro3": {key: float(tasks[key].mean()) for key in metric_columns},
        "direction_consistency": {
            key: {
                "positive_tasks": int((tasks[key] > 0).sum()),
                "positive_folds": int((folds[key] > 0).sum()),
            }
            for key in contrasts
        },
        "shared_validation_test": True,
    }
    summary_dir = OUTPUT / "summary"
    _atomic_text(summary_dir / "per_fold_results.csv", folds.to_csv(index=False))
    _atomic_text(summary_dir / "task_results.csv", tasks.to_csv(index=False))
    _atomic_text(
        summary_dir / "fusionwarm_encoder_factorial_20k_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
