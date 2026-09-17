#!/usr/bin/env python3
"""Re-verify and compare matched Arm-B and DND 8-task fold shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scripts.create_mips_split_manifests import build_manifest
from scripts.finetune_glt_o8_control import ALLOWED_TASKS_8, DEFAULT_TASKS_7
from src.training.glt_dual_runtime import write_json


def _raw_targets(path):
    frame = pd.read_csv(path)
    columns = [name for name in frame.columns if str(name).lower() not in {"smiles", "sample_key"}]
    if len(columns) != 1:
        raise ValueError(f"raw task CSV must have one label column: {path}")
    values = frame[columns[0]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"raw task labels are nonfinite: {path}")
    return frame, values


def _manifest(task, raw_path, split_root):
    path = Path(split_root) / f"{task}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing fixed split manifest: {path}")
    actual = json.loads(path.read_text(encoding="utf-8"))
    expected = build_manifest(task, raw_path, "outer5_inner20")
    for field in ("protocol", "sample_count", "sample_order_hash"):
        if actual.get(field) != expected[field]:
            raise ValueError(f"split identity mismatch for {task}: {field}")
    if actual.get("validation_is_test") is not False or len(actual.get("folds", [])) != 5:
        raise ValueError(f"split is not separated five-fold for {task}")
    for old, new in zip(actual["folds"], expected["folds"]):
        for field in ("fold", "train_indices", "validation_indices", "test_indices"):
            if old.get(field) != new[field]:
                raise ValueError(f"split identity mismatch for {task}: {field}")
    return actual


def _shard(root, task, fold, manifest, raw_values):
    folder = Path(root) / f"{task}_fold{fold}" / task / f"fold{fold}"
    metrics_path, predictions_path = folder / "metrics.json", folder / "predictions.csv"
    if not metrics_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(f"missing formal shard: {task} fold {fold} under {root}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (metrics.get("task") != task or int(metrics.get("fold", -1)) != int(fold)
            or metrics.get("protocol") != "outer5_inner20"
            or metrics.get("outer_test") != "RUN_ONCE"):
        raise ValueError(f"invalid shard protocol/identity: {task} fold {fold}")
    prediction = pd.read_csv(predictions_path)
    required = {"row_index", "target", "prediction"}
    if set(prediction.columns) != required or prediction.empty:
        raise ValueError(f"invalid prediction columns/empty shard: {task} fold {fold}")
    test = np.asarray(manifest["folds"][int(fold)]["test_indices"], dtype=np.int64)
    observed_rows = prediction["row_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(observed_rows, test) or len(np.unique(observed_rows)) != len(test):
        raise ValueError(f"outer-test indices mismatch: {task} fold {fold}")
    expected_targets = raw_values[test]
    observed_targets = prediction["target"].to_numpy(dtype=np.float64)
    if not np.allclose(observed_targets, expected_targets, rtol=1e-7, atol=1e-8):
        raise ValueError(f"raw-label mismatch: {task} fold {fold}")
    observed_predictions = prediction["prediction"].to_numpy(dtype=np.float64)
    if not np.isfinite(observed_predictions).all():
        raise ValueError(f"nonfinite prediction: {task} fold {fold}")
    recomputed = {
        "task": task,
        "fold": int(fold),
        "test_r2": float(r2_score(expected_targets, observed_predictions)),
        "test_mae": float(mean_absolute_error(expected_targets, observed_predictions)),
        "test_rmse": float(np.sqrt(mean_squared_error(expected_targets, observed_predictions))),
    }
    for name in ("test_r2", "test_mae", "test_rmse"):
        if not np.isclose(float(metrics.get(name, np.nan)), recomputed[name], rtol=1e-8, atol=1e-10):
            raise ValueError(f"stored metric mismatch: {task} fold {fold} {name}")
    train = set(manifest["folds"][int(fold)]["train_indices"])
    validation = set(manifest["folds"][int(fold)]["validation_indices"])
    if train & validation or train & set(test) or validation & set(test):
        raise ValueError(f"split overlap: {task} fold {fold}")
    return recomputed, prediction


def _aggregate(root, task_roots, split_root, raw_root):
    folds, tasks = [], {}
    for task in ALLOWED_TASKS_8:
        raw_path = Path(raw_root) / f"smi_{task}.csv"
        _, raw_values = _raw_targets(raw_path)
        manifest = _manifest(task, raw_path, split_root)
        root_for_task = task_roots[task]
        rows, predictions = [], []
        for fold in range(5):
            row, prediction = _shard(root_for_task, task, fold, manifest, raw_values)
            rows.append(row)
            predictions.append(prediction)
            folds.append({**row, "arm_root": str(Path(root_for_task).resolve())})
        merged = pd.concat(predictions, ignore_index=True)
        if len(merged) != len(raw_values) or sorted(merged["row_index"].tolist()) != list(range(len(raw_values))):
            raise ValueError(f"OOF coverage is not exactly once: {task}")
        tasks[task] = {
            "folds": rows,
            "test_r2": {"mean": float(np.mean([r["test_r2"] for r in rows])),
                         "std": float(np.std([r["test_r2"] for r in rows], ddof=1))},
            "test_mae": {"mean": float(np.mean([r["test_mae"] for r in rows])),
                          "std": float(np.std([r["test_mae"] for r in rows], ddof=1))},
            "test_rmse": {"mean": float(np.mean([r["test_rmse"] for r in rows])),
                           "std": float(np.std([r["test_rmse"] for r in rows], ddof=1))},
            "pooled_oof": {
                "r2": float(r2_score(merged["target"], merged["prediction"])),
                "mae": float(mean_absolute_error(merged["target"], merged["prediction"])),
                "rmse": float(np.sqrt(mean_squared_error(merged["target"], merged["prediction"]))),
            },
            "sample_count": int(len(raw_values)), "oof_count": int(len(merged)),
        }
    return {"tasks": tasks, "folds": folds,
            "macro8_r2": float(np.mean([tasks[t]["test_r2"]["mean"] for t in ALLOWED_TASKS_8]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-b-7-root", required=True)
    parser.add_argument("--arm-b-egc-root", required=True)
    parser.add_argument("--student-root", required=True)
    parser.add_argument("--split-root", default="data/splits/mips_outer5_inner20")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"aggregation output is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    arm_b_roots = {task: (args.arm_b_7_root if task in DEFAULT_TASKS_7 else args.arm_b_egc_root)
                   for task in ALLOWED_TASKS_8}
    arm_b = _aggregate(None, arm_b_roots, args.split_root, args.raw_root)
    student = _aggregate(None, {task: args.student_root for task in ALLOWED_TASKS_8},
                         args.split_root, args.raw_root)
    paired = []
    for task in ALLOWED_TASKS_8:
        b_rows = {row["fold"]: row for row in arm_b["tasks"][task]["folds"]}
        d_rows = {row["fold"]: row for row in student["tasks"][task]["folds"]}
        for fold in range(5):
            paired.append({
                "task": task, "fold": fold,
                "student_minus_arm_b_r2": d_rows[fold]["test_r2"] - b_rows[fold]["test_r2"],
            })
    macro7 = {
        "arm_b": float(np.mean([arm_b["tasks"][t]["test_r2"]["mean"] for t in DEFAULT_TASKS_7])),
        "student": float(np.mean([student["tasks"][t]["test_r2"]["mean"] for t in DEFAULT_TASKS_7])),
    }
    macro7["student_minus_arm_b"] = macro7["student"] - macro7["arm_b"]
    payload = {
        "schema": "eq3d-dnd-o8-student-aggregation-v1",
        "tasks": list(ALLOWED_TASKS_8), "fold_count": 40,
        "arm_b_7_root": str(Path(args.arm_b_7_root).resolve()),
        "arm_b_egc_root": str(Path(args.arm_b_egc_root).resolve()),
        "student_root": str(Path(args.student_root).resolve()),
        "arm_b": arm_b, "student": student,
        "macro7": macro7,
        "macro8": {
            "arm_b": arm_b["macro8_r2"], "student": student["macro8_r2"],
            "student_minus_arm_b": student["macro8_r2"] - arm_b["macro8_r2"],
        },
        "paired_fold_deltas": paired, "oof_verification": "PASS",
    }
    write_json(output / "summary.json", payload)
    pd.DataFrame(paired).to_csv(output / "paired_fold_deltas.csv", index=False)
    pd.DataFrame(arm_b["folds"] + student["folds"]).to_csv(output / "all_fold_metrics.csv", index=False)
    print(json.dumps({"status": "PASS", "fold_count": 40,
                      "output": str(output / "summary.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
