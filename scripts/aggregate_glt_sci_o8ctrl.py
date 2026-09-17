#!/usr/bin/env python3
"""Aggregate the fixed 7-task O8 attribution arms without Macro8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scripts.finetune_glt_o8_control import TASKS
from src.training.glt_dual_runtime import write_json


def _load_arm(root):
    root = Path(root)
    folds = []
    per_task = {}
    for task in TASKS:
        rows = []
        for fold in range(5):
            folder = root / f"{task}_fold{fold}" / task / f"fold{fold}"
            metrics_path = folder / "metrics.json"
            predictions_path = folder / "predictions.csv"
            if not metrics_path.is_file() or not predictions_path.is_file():
                raise FileNotFoundError(f"missing formal O8 shard: {task} fold {fold}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics.get("protocol") != "outer5_inner20" or metrics.get("outer_test") != "RUN_ONCE":
                raise ValueError(f"invalid O8 shard protocol: {task} fold {fold}")
            prediction = pd.read_csv(predictions_path)
            if prediction.empty or not np.isfinite(prediction[["target", "prediction"]].to_numpy()).all():
                raise ValueError(f"nonfinite O8 predictions: {task} fold {fold}")
            y_true = prediction["target"].to_numpy(dtype=float)
            y_pred = prediction["prediction"].to_numpy(dtype=float)
            result = {
                "task": task, "fold": fold,
                "test_r2": float(r2_score(y_true, y_pred)),
                "test_mae": float(mean_absolute_error(y_true, y_pred)),
                "test_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            }
            rows.append(result)
            folds.append(result)
        per_task[task] = rows
    task_summary = {}
    for task, rows in per_task.items():
        task_summary[task] = {
            metric: {"mean": float(np.mean([row[metric] for row in rows])),
                     "std": float(np.std([row[metric] for row in rows], ddof=1))}
            for metric in ("test_r2", "test_mae", "test_rmse")
        }
        all_prediction = []
        for fold in range(5):
            path = root / f"{task}_fold{fold}" / task / f"fold{fold}" / "predictions.csv"
            all_prediction.append(pd.read_csv(path))
        merged = pd.concat(all_prediction, ignore_index=True)
        if merged["row_index"].nunique() != len(merged):
            raise ValueError(f"duplicate OOF row index: {task}")
        task_summary[task]["pooled_oof"] = {
            "r2": float(r2_score(merged["target"], merged["prediction"])),
            "mae": float(mean_absolute_error(merged["target"], merged["prediction"])),
            "rmse": float(np.sqrt(mean_squared_error(merged["target"], merged["prediction"]))),
        }
    return {"folds": folds, "tasks": task_summary,
            "macro7_r2": float(np.mean([item["test_r2"]["mean"] for item in task_summary.values()]))}


def _reference_task_data(reference):
    """Normalize the two existing fixed-reference report schemas.

    ``aggregation_review_7task/summary.json`` contains fold-level values and
    is the preferred source for paired deltas.  The older
    ``three_way_comparison_7task.json`` contains only per-task means; it remains
    usable for the headline means, but cannot provide paired fold deltas.
    """
    if reference.get("tasks"):
        tasks = reference["tasks"]
        means = {task: float(tasks[task]["test_r2"]["mean"])
                 for task in TASKS}
        folds = {
            task: {int(item["fold"]): float(item["test_r2"])
                   for item in tasks[task].get("folds", [])}
            for task in TASKS
        }
        macro = reference.get("macro7_r2")
        if macro is None:
            macro = float(np.mean(list(means.values())))
        return means, folds, float(macro)
    if reference.get("per_task"):
        tasks = reference["per_task"]
        means = {task: float(tasks[task]["test_r2_mean"])
                 for task in TASKS}
        return means, {task: {} for task in TASKS}, float(
            reference.get("macro7_r2", reference.get("macro_r2")))
    raise ValueError("unsupported fixed 7-task reference summary schema")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-summary", required=True)
    parser.add_argument("--arm-a-root", required=True)
    parser.add_argument("--arm-b-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reference = json.loads(Path(args.reference_summary).read_text(encoding="utf-8"))
    if reference.get("tasks"):
        if reference.get("status") != "PASS" or int(reference.get("task_count", 0)) != 7:
            raise ValueError("reference summary is not the fixed 7-task PASS report")
    elif set(reference.get("per_task", {})) != set(TASKS):
        raise ValueError("reference summary does not cover the fixed 7 tasks")
    arm_a, arm_b = _load_arm(args.arm_a_root), _load_arm(args.arm_b_root)
    reference_means, reference_folds, reference_macro = _reference_task_data(reference)
    comparison = {}
    paired = []
    for task in TASKS:
        r_mean = reference_means[task]
        a_mean = float(arm_a["tasks"][task]["test_r2"]["mean"])
        b_mean = float(arm_b["tasks"][task]["test_r2"]["mean"])
        comparison[task] = {
            "reference_r2_mean": r_mean, "arm_a_r2_mean": a_mean,
            "arm_b_r2_mean": b_mean, "reference_minus_arm_a": r_mean - a_mean,
            "reference_minus_arm_b": r_mean - b_mean, "arm_a_minus_arm_b": a_mean - b_mean,
        }
        for fold in range(5):
            a_row = next(row for row in arm_a["folds"]
                         if row["task"] == task and row["fold"] == fold)
            b_row = next(row for row in arm_b["folds"]
                         if row["task"] == task and row["fold"] == fold)
            row = {"task": task, "fold": fold,
                   "arm_a_minus_arm_b": a_row["test_r2"] - b_row["test_r2"]}
            if fold in reference_folds[task]:
                r_fold = reference_folds[task][fold]
                row.update({"reference_minus_arm_a": r_fold - a_row["test_r2"],
                            "reference_minus_arm_b": r_fold - b_row["test_r2"]})
            else:
                row.update({"reference_minus_arm_a": None,
                            "reference_minus_arm_b": None})
            paired.append(row)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema": "glt-sci-o8ctrl-comparison-v1", "tasks": list(TASKS),
        "reference": reference, "arm_a": arm_a, "arm_b": arm_b,
        "comparison": comparison, "paired_fold_delta": paired,
        "macro7": {"reference": reference_macro,
                    "arm_a": arm_a["macro7_r2"], "arm_b": arm_b["macro7_r2"]},
        "macro8": None,
    }
    write_json(output / "summary.json", payload)
    pd.DataFrame(paired).to_csv(output / "paired_fold_delta.csv", index=False)


if __name__ == "__main__":
    main()
