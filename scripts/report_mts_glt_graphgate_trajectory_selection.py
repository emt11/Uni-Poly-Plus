#!/usr/bin/env python3
"""Report 5k FusionWarm against paired 5k O8 and 20k FusionWarm."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/mts_glt_graphgate_v1/trajectory_selection_v1"
OLD_5K = ROOT / "results/mts_glt_graphgate_v1/downstream/screen_5k/o8_only"
FW_20K = ROOT / "results/mts_glt_graphgate_v1/fusion_realization_v1/fusion_realization_fold_results.csv"
TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
SCREEN = ("xc", "ei", "egc")


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _prediction(root, task, fold):
    path = root / f"predictions/42/{task}/fold_{fold}.npz"
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        y_true = np.asarray(data["y_true"], dtype=np.float64)
        y_pred = np.asarray(data["y_pred"], dtype=np.float64)
        indices = np.asarray(data["sample_indices"], dtype=np.int64)
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise RuntimeError(f"non-finite prediction: {path}")
    denominator = np.sum(np.square(y_true - y_true.mean()))
    r2 = 1.0 - np.sum(np.square(y_true - y_pred)) / denominator
    return {"r2": float(r2), "metadata": metadata, "indices": indices, "target": y_true}


def _load_20k():
    with FW_20K.open(newline="", encoding="utf-8") as handle:
        return {(row["task"], int(row["fold"])): row for row in csv.DictReader(handle)}


def _unit_roots(scope, task, fold):
    if (task in SCREEN and fold < 3):
        o8 = OLD_5K
    else:
        o8 = OUTPUT / "o8_only_5k"
    fused = OUTPUT / "fusion_warm_5k"
    return o8, fused


def _summarize(tasks, folds):
    historical = _load_20k()
    rows = []
    for task in tasks:
        for fold in folds:
            o8_root, fused_root = _unit_roots("", task, fold)
            o8 = _prediction(o8_root, task, fold)
            fused = _prediction(fused_root, task, fold)
            if o8["metadata"]["fold_seed"] != fused["metadata"]["fold_seed"]:
                raise RuntimeError(f"fold seed mismatch: {task}/fold{fold}")
            if not np.array_equal(o8["indices"], fused["indices"]) or not np.array_equal(o8["target"], fused["target"]):
                raise RuntimeError(f"prediction pairing mismatch: {task}/fold{fold}")
            ref = historical[(task, fold)]
            delta_20k = float(ref["delta_vs_o8"])
            rows.append({
                "task": task, "fold": int(fold),
                "fold_seed": int(o8["metadata"]["fold_seed"]),
                "o8_5k_r2": o8["r2"], "fusion_warm_5k_r2": fused["r2"],
                "delta_5k": fused["r2"] - o8["r2"],
                "delta_20k": delta_20k,
                "delta_5k_minus_20k": fused["r2"] - o8["r2"] - delta_20k,
            })
    task_rows = []
    for task in tasks:
        selected = [row for row in rows if row["task"] == task]
        for key in ("o8_5k_r2", "fusion_warm_5k_r2", "delta_5k", "delta_20k", "delta_5k_minus_20k"):
            values = np.asarray([row[key] for row in selected], dtype=np.float64)
            if not np.isfinite(values).all():
                raise RuntimeError(f"non-finite summary: {task}/{key}")
        task_rows.append({
            "task": task,
            "o8_5k_mean": float(np.mean([x["o8_5k_r2"] for x in selected])),
            "fusion_warm_5k_mean": float(np.mean([x["fusion_warm_5k_r2"] for x in selected])),
            "fusion_warm_5k_sample_std": float(np.std([x["fusion_warm_5k_r2"] for x in selected], ddof=1)),
            "delta_5k": float(np.mean([x["delta_5k"] for x in selected])),
            "delta_20k": float(np.mean([x["delta_20k"] for x in selected])),
            "delta_5k_minus_20k": float(np.mean([x["delta_5k_minus_20k"] for x in selected])),
            "positive_folds_5k": int(sum(x["delta_5k"] > 0 for x in selected)),
        })
    delta5 = np.asarray([row["delta_5k"] for row in task_rows])
    delta20 = np.asarray([row["delta_20k"] for row in task_rows])
    summary = {
        "schema": "mts-glt-graphgate-trajectory-selection-v1",
        "tasks": task_rows,
        "macro": {
            "o8_5k": float(np.mean([row["o8_5k_mean"] for row in task_rows])),
            "fusion_warm_5k": float(np.mean([row["fusion_warm_5k_mean"] for row in task_rows])),
            "delta_5k": float(delta5.mean()), "delta_20k": float(delta20.mean()),
        },
        "median_delta_5k_minus_20k": float(np.median(delta5 - delta20)),
        "tasks_delta_5k_gt_20k": int(np.sum(delta5 > delta20)),
        "folds": rows,
    }
    return summary


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("screen5k", "formal5k"), required=True)
    args = parser.parse_args(argv)
    tasks = SCREEN if args.scope == "screen5k" else TASKS
    folds = range(3) if args.scope == "screen5k" else range(5)
    summary = _summarize(tasks, folds)
    if args.scope == "screen5k":
        run_formal = bool(
            summary["macro"]["delta_5k"] > summary["macro"]["delta_20k"]
            and summary["median_delta_5k_minus_20k"] > 0
            and summary["tasks_delta_5k_gt_20k"] >= 2
        )
        summary["run_formal5k"] = run_formal
        summary["decision"] = "run_formal5k" if run_formal else "stop_after_screen5k"
        name = "screen5k_decision.json"
    else:
        delta5 = np.asarray([row["delta_5k"] for row in summary["tasks"]])
        delta20 = np.asarray([row["delta_20k"] for row in summary["tasks"]])
        start_late = bool(
            summary["macro"]["fusion_warm_5k"] > 0.8376006087
            and summary["macro"]["delta_5k"] > 0.0033001972
            and np.median(delta5 - delta20) > 0
            and np.sum(delta5 > delta20) >= 5
        )
        summary["run_late_infonce"] = start_late
        summary["decision"] = "five_k_neural_complementarity_confirmed" if start_late else "five_k_not_better_than_twenty_k"
        name = "formal5k_decision.json"
    _write_csv(OUTPUT / f"{args.scope}_fold_results.csv", summary["folds"])
    _atomic_json(OUTPUT / name, summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
