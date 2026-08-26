#!/usr/bin/env python3
"""Report S4, C5-Mixed, and MS45 matched downstream contrasts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/mts_glt_v2/mscontact_v1"
OUTPUT = BASE / "c5_mixed_v1"
CONFIG = ROOT / "configs/mts/mscontact_v1/c5_downstream.json"


def load_r2(arm, task, fold, seed):
    path = BASE / "downstream" / arm / "shards" / str(seed) / task / f"fold_{fold}.csv"
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise RuntimeError(f"expected one shard row: {path}")
    value = float(frame.iloc[0]["avg_test_r2"])
    if not np.isfinite(value):
        raise RuntimeError(f"non-finite R2: {path}")
    return value, str(path.resolve())


def contrast_summary(tasks, folds, column):
    task_values = tasks[column].to_numpy(dtype=np.float64)
    fold_values = folds[column].to_numpy(dtype=np.float64)
    return {
        "macro3_delta": float(task_values.mean()),
        "median_task_delta": float(np.median(task_values)),
        "positive_tasks": int((task_values > 0).sum()),
        "positive_folds": int((fold_values > 0).sum()),
        "fold_delta": {
            "mean": float(fold_values.mean()),
            "median": float(np.median(fold_values)),
            "p25": float(np.percentile(fold_values, 25)),
            "p75": float(np.percentile(fold_values, 75)),
            "min": float(fold_values.min()),
            "max": float(fold_values.max()),
        },
    }


def markdown_table(frame):
    columns = list(frame.columns)
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def main():
    config = json.loads(CONFIG.read_text())
    rows = []
    for task in config["tasks"]:
        for fold in config["folds"]:
            values = {}
            paths = {}
            for arm in ("s4", "c5_mixed", "ms45"):
                values[arm], paths[arm] = load_r2(arm, task, fold, config["seed"])
            rows.append({
                "task": task,
                "fold": int(fold),
                "s4_r2": values["s4"],
                "c5_r2": values["c5_mixed"],
                "ms45_r2": values["ms45"],
                "c5_minus_s4": values["c5_mixed"] - values["s4"],
                "ms45_minus_c5": values["ms45"] - values["c5_mixed"],
                "ms45_minus_s4": values["ms45"] - values["s4"],
                "s4_shard": paths["s4"],
                "c5_shard": paths["c5_mixed"],
                "ms45_shard": paths["ms45"],
            })
    folds = pd.DataFrame(rows)
    tasks = folds.groupby("task", sort=False).agg(
        s4=("s4_r2", "mean"),
        c5=("c5_r2", "mean"),
        ms45=("ms45_r2", "mean"),
        c5_minus_s4=("c5_minus_s4", "mean"),
        ms45_minus_c5=("ms45_minus_c5", "mean"),
        ms45_minus_s4=("ms45_minus_s4", "mean"),
    ).reset_index()
    contrasts = {
        "C5-S4": contrast_summary(tasks, folds, "c5_minus_s4"),
        "MS45-C5": contrast_summary(tasks, folds, "ms45_minus_c5"),
        "MS45-S4": contrast_summary(tasks, folds, "ms45_minus_s4"),
    }
    summary = {
        "schema": "mts-glt-v2-mscontact-c5-mixed-screening-v1",
        "scientific_question": "explicit shell decomposition versus matched unified <5 A contacts",
        "macro3": {
            "s4": float(tasks["s4"].mean()),
            "c5": float(tasks["c5"].mean()),
            "ms45": float(tasks["ms45"].mean()),
        },
        "contrasts": contrasts,
        "explicit_multiscale_interpretation": (
            "near_tie_task_dependent; no practical superiority established"
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    folds.to_csv(OUTPUT / "per_fold_results.csv", index=False)
    tasks.to_csv(OUTPUT / "task_results.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    metrics_path = BASE / "c5_mixed" / "training_metrics.jsonl"
    metrics = pd.DataFrame([
        json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()
    ])
    if len(metrics) != 5000 or int(metrics.iloc[-1]["step"]) != 5000:
        raise RuntimeError("C5-Mixed pretraining metrics are incomplete")
    metrics.insert(0, "arm", "c5_mixed")
    metrics["total_loss"] = (
        metrics["masked_atom_loss"] + metrics["masked_line_loss"]
        + metrics["infonce_loss"] * metrics["infonce_weight"]
    )
    metrics.to_csv(OUTPUT / "pretrain_metrics.csv", index=False)
    scheduler_path = ROOT / "logs/mts_glt_v2/mscontact_v1/downstream/c5_mixed/scheduler_report.json"
    scheduler = json.loads(scheduler_path.read_text())
    if len(scheduler.get("completed", [])) != 9 or scheduler.get("pending") != 0:
        raise RuntimeError("C5-Mixed downstream scheduler is incomplete")
    manifest = {
        "schema": "mts-glt-v2-mscontact-c5-mixed-manifest-v1",
        "status": "completed_screening",
        "pretraining_gate": str((OUTPUT / "pretrain_gate.json").resolve()),
        "pretrain_config": str((ROOT / "configs/mts/mscontact_v1/c5_mixed_5k.json").resolve()),
        "downstream_config": str(CONFIG.resolve()),
        "checkpoint": str((ROOT / "pretrained_models/mts_glt_v2/mscontact_v1/c5_mixed_005k.pth").resolve()),
        "matched_controls": {
            "s4": str((ROOT / "pretrained_models/mts_glt_v2/mscontact_v1/s4_005k.pth").resolve()),
            "ms45": str((ROOT / "pretrained_models/mts_glt_v2/mscontact_v1/ms45_005k.pth").resolve()),
        },
        "completed_pretrain_steps": 5000,
        "completed_downstream_runs": 9,
        "scheduler_report": str(scheduler_path.resolve()),
        "follow_up_started": False,
    }
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    display = tasks.rename(columns={
        "task": "Task", "s4": "S4", "c5": "C5", "ms45": "MS45",
        "c5_minus_s4": "C5-S4", "ms45_minus_c5": "MS45-C5",
        "ms45_minus_s4": "MS45-S4",
    }).copy()
    for column in display.columns[1:]:
        display[column] = display[column].map(lambda value: f"{value:.6f}")
    lines = [
        "# MTS-GLT-v2-MSContact-v1 / C5-Mixed", "",
        "Matched question: does explicit S4/S45 decomposition outperform one unified <5 A contact aggregation?", "",
        markdown_table(display), "",
    ]
    for name in ("C5-S4", "MS45-C5", "MS45-S4"):
        value = contrasts[name]
        lines.extend([
            f"## {name}", "",
            f"- macro3 delta: {value['macro3_delta']:.6f}",
            f"- median task delta: {value['median_task_delta']:.6f}",
            f"- positive tasks/folds: {value['positive_tasks']}/3; {value['positive_folds']}/9", "",
        ])
    lines.extend([
        "Explicit shell decomposition is a near tie with unified C5: the macro3 difference is +0.000078 and task directions are mixed. This screening does not establish a practical advantage for MS45 over C5.", "",
        "`C5-S4` is the outer-shell information increment; `MS45-C5` is the explicit organization increment; `MS45-S4` is the total effect.", "",
        "No 20k, 8x5, alternative cutoff, hierarchical fusion, or follow-up experiment was started.",
    ])
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
