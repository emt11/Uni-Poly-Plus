#!/usr/bin/env python3
"""Report the full matched 5k Full/Off O8/FusionWarm eight-task five-fold result."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/mts/glt_graphgate_v1/matched_fusionwarm_causal_8task_fold01234_v1.json"
FOLD012 = ROOT / (
    "results/mts_glt_graphgate_v1/trimer_validation_v1/matched_5k/"
    "fusionwarm_causal_v1/eight_task_fold012"
)
OUTPUT = ROOT / (
    "results/mts_glt_graphgate_v1/trimer_validation_v1/matched_5k/"
    "fusionwarm_causal_v1/eight_task_fold01234"
)
TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
FOLDS = (0, 1, 2, 3, 4)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _load_fold012():
    path = FOLD012 / "summary/per_fold_results.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 24:
        raise RuntimeError(f"expected 24 fold-0/1/2 rows: {path}")
    return [
        {
            "task": row["task"],
            "fold": int(row["fold"]),
            "fold_seed": int(row["fold_seed"]),
            **{
                key: float(row[key])
                for key in (
                    "full_o8_r2",
                    "full_fw_r2",
                    "delta_full",
                    "off_o8_r2",
                    "off_fw_r2",
                    "delta_off",
                    "interaction",
                    "full_fw_minus_off_fw",
                )
            },
        }
        for row in rows
    ]


def _prediction(arm, mode, task, fold, expected_checkpoint):
    leaf = "fusionwarm" if mode == "o8_glt_graph" else "o8"
    root = OUTPUT / arm / leaf
    prediction_path = root / f"predictions/42/{task}/fold_{fold}.npz"
    with np.load(prediction_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        indices = np.asarray(data["sample_indices"], dtype=np.int64)
        target = np.asarray(data["y_true"], dtype=np.float64)
        prediction = np.asarray(data["y_pred"], dtype=np.float64)
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise RuntimeError(f"non-finite prediction: {prediction_path}")
    denominator = np.square(target - target.mean()).sum()
    r2 = 1.0 - np.square(target - prediction).sum() / denominator
    shard = root / f"shards/42/{task}/fold_{fold}.csv"
    with shard.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"expected one result row: {shard}")
    checkpoint = Path(rows[0]["checkpoint_path"]).resolve()
    if checkpoint != expected_checkpoint.resolve():
        raise RuntimeError(f"checkpoint mismatch: {shard}: {checkpoint}")
    expected_strategy = "fusion_warm" if mode == "o8_glt_graph" else "legacy_zero"
    if rows[0]["mts_glt_fusion_strategy"] != expected_strategy:
        raise RuntimeError(f"fusion strategy mismatch: {shard}")
    if rows[0]["mts_glt_geometry_mode"] != arm:
        raise RuntimeError(f"geometry arm mismatch: {shard}")
    return {
        "r2": float(r2),
        "fold_seed": int(metadata["fold_seed"]),
        "indices": indices,
        "target": target,
    }


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if tuple(config["tasks"]) != TASKS or tuple(config["summary_folds"]) != FOLDS:
        raise RuntimeError("canonical task/fold roster mismatch")
    checkpoints = {
        arm: (ROOT / config[f"{arm}_checkpoint"]).resolve()
        for arm in ("full", "off")
    }
    forbidden = (ROOT / "results/mts_glt_graphgate_v1/pretrain/mts_glt_graphgate_probe_005k.pth").resolve()
    if forbidden in set(checkpoints.values()):
        raise RuntimeError("main-trajectory checkpoint is forbidden")

    per_fold = _load_fold012()
    for task in TASKS:
        for fold in (3, 4):
            units = {
                f"{arm}_{mode}": _prediction(arm, mode, task, fold, checkpoints[arm])
                for arm in ("full", "off")
                for mode in ("o8_only", "o8_glt_graph")
            }
            reference = units["full_o8_only"]
            for name, unit in units.items():
                if unit["fold_seed"] != reference["fold_seed"]:
                    raise RuntimeError(f"fold seed mismatch: {task}/fold{fold}/{name}")
                if not np.array_equal(unit["indices"], reference["indices"]):
                    raise RuntimeError(f"sample-index mismatch: {task}/fold{fold}/{name}")
                if not np.array_equal(unit["target"], reference["target"]):
                    raise RuntimeError(f"target mismatch: {task}/fold{fold}/{name}")
            full_o8 = units["full_o8_only"]["r2"]
            full_fw = units["full_o8_glt_graph"]["r2"]
            off_o8 = units["off_o8_only"]["r2"]
            off_fw = units["off_o8_glt_graph"]["r2"]
            delta_full = full_fw - full_o8
            delta_off = off_fw - off_o8
            per_fold.append(
                {
                    "task": task.upper(),
                    "fold": fold,
                    "fold_seed": reference["fold_seed"],
                    "full_o8_r2": full_o8,
                    "full_fw_r2": full_fw,
                    "delta_full": delta_full,
                    "off_o8_r2": off_o8,
                    "off_fw_r2": off_fw,
                    "delta_off": delta_off,
                    "interaction": delta_full - delta_off,
                    "full_fw_minus_off_fw": full_fw - off_fw,
                }
            )
    per_fold.sort(key=lambda row: (TASKS.index(row["task"].lower()), row["fold"]))
    expected_pairs = {(task.upper(), fold) for task in TASKS for fold in FOLDS}
    if {(row["task"], row["fold"]) for row in per_fold} != expected_pairs:
        raise RuntimeError("five-fold task/fold coverage mismatch")

    metrics = (
        "full_o8_r2",
        "full_fw_r2",
        "delta_full",
        "off_o8_r2",
        "off_fw_r2",
        "delta_off",
        "interaction",
        "full_fw_minus_off_fw",
    )
    per_task = {}
    for task in (task.upper() for task in TASKS):
        rows = [row for row in per_fold if row["task"] == task]
        per_task[task] = {
            key: float(np.mean([row[key] for row in rows])) for key in metrics
        }

    task_interactions = np.asarray(
        [per_task[task.upper()]["interaction"] for task in TASKS], dtype=np.float64
    )
    task_delta_full = np.asarray(
        [per_task[task.upper()]["delta_full"] for task in TASKS], dtype=np.float64
    )
    task_fused_contrast = np.asarray(
        [per_task[task.upper()]["full_fw_minus_off_fw"] for task in TASKS],
        dtype=np.float64,
    )
    fold_interactions = np.asarray(
        [row["interaction"] for row in per_fold], dtype=np.float64
    )
    fold_delta_full = np.asarray(
        [row["delta_full"] for row in per_fold], dtype=np.float64
    )
    macro = float(task_interactions.mean())
    median = float(np.median(task_interactions))
    positive_tasks = int((task_interactions > 0).sum())
    passed = macro > 0 and median > 0 and positive_tasks >= 5
    payload = {
        "protocol": config["protocol"],
        "seed": config["seed"],
        "tasks": [task.upper() for task in TASKS],
        "folds": list(FOLDS),
        "checkpoints": {key: str(value) for key, value in checkpoints.items()},
        "run_accounting": {
            "new_fusionwarm_runs": 32,
            "new_o8_runs": 32,
            "reused_fusionwarm_runs": 48,
            "reused_o8_runs": 48,
            "failed_runs": 0,
        },
        "per_fold": per_fold,
        "per_task": per_task,
        "macro8_interaction": macro,
        "median_task_interaction": median,
        "positive_interaction_tasks": positive_tasks,
        "macro8_delta_full": float(task_delta_full.mean()),
        "median_task_delta_full": float(np.median(task_delta_full)),
        "positive_delta_full_tasks": int((task_delta_full > 0).sum()),
        "positive_interaction_folds": int((fold_interactions > 0).sum()),
        "positive_delta_full_folds": int((fold_delta_full > 0).sum()),
        "fold_interaction_mean": float(fold_interactions.mean()),
        "fold_interaction_median": float(np.median(fold_interactions)),
        "fold_interaction_p25": float(np.percentile(fold_interactions, 25)),
        "fold_interaction_p75": float(np.percentile(fold_interactions, 75)),
        "fold_interaction_min": float(fold_interactions.min()),
        "fold_interaction_max": float(fold_interactions.max()),
        "full_fw_minus_off_fw_macro8": float(task_fused_contrast.mean()),
        "fused_contrast_note": "This is NOT the interaction metric.",
        "screening_comparison": {
            "fold012_macro8_interaction": 0.011007640892150589,
            "fold012_positive_interaction_tasks": 7,
            "fold012_macro8_delta_full": 0.006371248582006972,
        },
        "final_matched_decision": "PASS" if passed else "FAIL",
    }
    summary = OUTPUT / "summary"
    _atomic_write(
        summary / "matched_fusionwarm_causal_8task_fold01234_summary.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    csv_path = summary / "per_fold_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_fold[0]))
        writer.writeheader()
        writer.writerows(per_fold)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
