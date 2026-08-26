#!/usr/bin/env python3
"""Formal 8-task x 5-fold S4 versus C5-Mixed report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/mts_glt_v2/mscontact_v1/formal_8x5_s4_c5_v1"
LOG_ROOT = ROOT / "logs/mts_glt_v2/mscontact_v1/formal_8x5_s4_c5_v1"
CONFIG = ROOT / "configs/mts/mscontact_v1/formal_8x5_s4_c5.json"


def load_r2(arm, task, fold, seed):
    path = OUTPUT / arm / "shards" / str(seed) / task / f"fold_{fold}.csv"
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise RuntimeError(f"expected one formal shard row: {path}")
    value = float(frame.iloc[0]["avg_test_r2"])
    if not np.isfinite(value):
        raise RuntimeError(f"non-finite formal R2: {path}")
    return value, str(path.resolve())


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
            s4, s4_path = load_r2("s4", task, fold, config["seed"])
            c5, c5_path = load_r2("c5_mixed", task, fold, config["seed"])
            rows.append({
                "task": task, "fold": int(fold), "s4_r2": s4, "c5_r2": c5,
                "c5_minus_s4": c5 - s4,
                "s4_shard": s4_path, "c5_shard": c5_path,
            })
    folds = pd.DataFrame(rows)
    tasks = folds.groupby("task", sort=False).agg(
        s4=("s4_r2", "mean"), c5=("c5_r2", "mean"),
        delta=("c5_minus_s4", "mean"),
        positive_folds=("c5_minus_s4", lambda values: int((values > 0).sum())),
    ).reset_index()
    values = folds["c5_minus_s4"].to_numpy(dtype=np.float64)
    macro_s4 = float(tasks["s4"].mean())
    macro_c5 = float(tasks["c5"].mean())
    macro_delta = float(tasks["delta"].mean())
    positive_tasks = int((tasks["delta"] > 0).sum())
    positive_folds = int((values > 0).sum())
    decision = (
        "FINAL_CANDIDATE"
        if macro_delta > 0 and positive_tasks >= 5 and positive_folds >= 21
        else "STOP_C5"
    )
    prior = {
        task.upper(): {
            "formal_delta": float(tasks.loc[tasks["task"] == task, "delta"].iloc[0]),
            "screening_direction_retained": bool(
                float(tasks.loc[tasks["task"] == task, "delta"].iloc[0])
                * {"xc": 1.0, "ei": 1.0, "eea": -1.0}[task] > 0
            ),
        }
        for task in ("xc", "ei", "eea")
    }
    summary = {
        "schema": "mts-glt-v2-mscontact-formal-s4-c5-v1",
        "protocol": config["evaluation_protocol"],
        "macro8_s4": macro_s4,
        "macro8_c5": macro_c5,
        "macro8_delta": macro_delta,
        "median_task_delta": float(tasks["delta"].median()),
        "positive_tasks": positive_tasks,
        "positive_folds": positive_folds,
        "fold_delta": {
            "mean": float(values.mean()), "median": float(np.median(values)),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
            "min": float(values.min()), "max": float(values.max()),
        },
        "screening_direction_check": prior,
        "decision": decision,
    }
    folds.to_csv(OUTPUT / "per_fold_results.csv", index=False)
    tasks.to_csv(OUTPUT / "task_results.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    scheduler = {}
    for arm in ("s4", "c5_mixed"):
        path = LOG_ROOT / arm / "scheduler_report.json"
        payload = json.loads(path.read_text())
        if int(payload.get("units", -1)) != 40 or int(payload.get("pending", -1)) != 0:
            raise RuntimeError(f"formal scheduler incomplete: {path}")
        scheduler[arm] = {
            "report": str(path.resolve()),
            "completed": len(payload.get("completed", [])),
            "skipped_reused": len(payload.get("skipped", [])),
        }
    manifest = {
        "schema": "mts-glt-v2-mscontact-formal-s4-c5-manifest-v1",
        "status": "completed_formal_8x5",
        "config": str(CONFIG.resolve()),
        "checkpoints": {
            arm: str((ROOT / config["checkpoint"][arm]).resolve())
            for arm in ("s4", "c5_mixed")
        },
        "tasks": config["tasks"], "folds": config["folds"],
        "seed": config["seed"], "protocol": config["evaluation_protocol"],
        "scheduler": scheduler,
        "reuse_manifest": str((OUTPUT / "reuse_manifest.json").resolve()),
        "follow_up_started": False,
    }
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    display = tasks.rename(columns={
        "task": "Task", "s4": "S4", "c5": "C5", "delta": "C5-S4",
        "positive_folds": "positive folds",
    }).copy()
    for column in ("S4", "C5", "C5-S4"):
        display[column] = display[column].map(lambda value: f"{value:.6f}")
    lines = [
        "# MTS-GLT-v2-MSContact-v1: Formal S4 vs C5-Mixed", "",
        markdown_table(display), "",
        f"- macro8 S4: {macro_s4:.6f}",
        f"- macro8 C5: {macro_c5:.6f}",
        f"- macro8 delta: {macro_delta:+.6f}",
        f"- median task delta: {summary['median_task_delta']:+.6f}",
        f"- positive tasks/folds: {positive_tasks}/8; {positive_folds}/40",
        f"- decision: `{decision}`", "",
        "`C5-S4` is the matched 4-5 A outer-shell information increment.", "",
        "This is a historical_shared5 formal matched comparison, not an independent blind test.", "",
        "No MS45, pretraining, 20k, hierarchical fusion, cutoff search, or follow-up experiment was started.",
    ]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
