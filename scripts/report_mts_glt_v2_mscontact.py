#!/usr/bin/env python3
"""Summarize the primary MS45-S4 downstream contrast."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/mts/mscontact_v1/downstream.json"
OUTPUT = ROOT / "results/mts_glt_v2/mscontact_v1"
LOG_ROOT = ROOT / "logs/mts_glt_v2/mscontact_v1"
PRETRAINED_ROOT = ROOT / "pretrained_models/mts_glt_v2/mscontact_v1"


def load_r2(arm, task, fold, seed):
    path = OUTPUT / "downstream" / arm / "shards" / str(seed) / task / f"fold_{fold}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise RuntimeError(f"expected one shard row: {path}")
    return float(frame.iloc[0]["avg_test_r2"]), str(path)


def markdown_table(frame):
    """Render the small report table without pandas' optional tabulate dependency."""
    columns = list(frame.columns)
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def load_pretrain_metrics(arm):
    path = OUTPUT / arm / "training_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 5000 or int(rows[-1]["step"]) != 5000:
        raise RuntimeError(f"incomplete 5k pretraining metrics: {path}")
    frame = pd.DataFrame(rows)
    frame.insert(0, "arm", arm)
    frame["total_loss"] = (
        frame["masked_atom_loss"]
        + frame["masked_line_loss"]
        + frame["infonce_loss"] * frame["infonce_weight"]
    )
    return frame


def pretrain_summary(frame):
    final = frame.iloc[-1]
    tail = frame.loc[frame["step"].between(4501, 5000)]
    metrics = ("masked_atom_loss", "masked_line_loss", "infonce_loss", "total_loss")
    return {
        "metrics_rows": int(len(frame)),
        "all_finite": bool(np.isfinite(frame[list(metrics)].to_numpy()).all()),
        "final": {name: float(final[name]) for name in metrics},
        "steps_4501_5000_mean": {name: float(tail[name].mean()) for name in metrics},
    }


def main():
    config = json.loads(CONFIG.read_text())
    pretrain_frames = {arm: load_pretrain_metrics(arm) for arm in ("s4", "ms45")}
    pd.concat(pretrain_frames.values(), ignore_index=True).to_csv(
        OUTPUT / "pretrain_metrics.csv", index=False
    )
    rows = []
    for task in config["tasks"]:
        for fold in config["folds"]:
            s4, s4_path = load_r2("s4", task, fold, config["seed"])
            ms45, ms45_path = load_r2("ms45", task, fold, config["seed"])
            rows.append({
                "task": task, "fold": int(fold), "s4_r2": s4,
                "ms45_r2": ms45, "ms45_minus_s4": ms45 - s4,
                "s4_shard": s4_path, "ms45_shard": ms45_path,
            })
    folds = pd.DataFrame(rows)
    tasks = folds.groupby("task", sort=False).agg(
        s4=("s4_r2", "mean"), ms45=("ms45_r2", "mean"),
        delta=("ms45_minus_s4", "mean"),
        positive_folds=("ms45_minus_s4", lambda values: int((values > 0).sum())),
    ).reset_index()
    values = folds["ms45_minus_s4"].to_numpy()
    summary = {
        "schema": "mts-glt-v2-mscontact-screening-v1",
        "primary_contrast": "MS45-S4",
        "macro3_delta": float(tasks["delta"].mean()),
        "median_task_delta": float(tasks["delta"].median()),
        "positive_tasks": int((tasks["delta"] > 0).sum()),
        "positive_folds": int((values > 0).sum()),
        "macro3": {
            "s4": float(tasks["s4"].mean()),
            "ms45": float(tasks["ms45"].mean()),
        },
        "pretraining": {
            arm: pretrain_summary(frame) for arm, frame in pretrain_frames.items()
        },
        "fold_delta": {
            "mean": float(values.mean()), "median": float(np.median(values)),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
            "min": float(values.min()), "max": float(values.max()),
        },
    }
    summary["continue_condition"] = bool(
        summary["macro3_delta"] > 0
        and summary["median_task_delta"] > 0
        and summary["positive_tasks"] >= 2
        and summary["positive_folds"] >= 5
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    folds.to_csv(OUTPUT / "per_fold_results.csv", index=False)
    tasks.to_csv(OUTPUT / "task_results.csv", index=False)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    scheduler_reports = {}
    for arm in ("s4", "ms45"):
        scheduler_path = LOG_ROOT / "downstream" / arm / "scheduler_report.json"
        scheduler = json.loads(scheduler_path.read_text())
        if len(scheduler.get("completed", [])) != 9 or scheduler.get("pending") != 0:
            raise RuntimeError(f"incomplete downstream scheduler: {scheduler_path}")
        scheduler_reports[arm] = str(scheduler_path)
    manifest = {
        "schema": "mts-glt-v2-mscontact-run-manifest-v1",
        "status": "completed_screening",
        "scientific_contrast": "MS45-S4",
        "pretrain_configs": {
            arm: str(ROOT / f"configs/mts/mscontact_v1/{arm}_5k.json")
            for arm in ("s4", "ms45")
        },
        "downstream_config": str(CONFIG),
        "sidecars": {
            "pretrain": str(ROOT / "data/processed/mips_trimer_scage/spatial_contact_v1/PI1M_v2"),
            "downstream": str(ROOT / "data/processed/mips_trimer_scage/spatial_contact_v1/downstream_union"),
        },
        "checkpoints": {
            arm: str(PRETRAINED_ROOT / f"{arm}_005k.pth") for arm in ("s4", "ms45")
        },
        "pretrain_metrics_rows": {arm: 5000 for arm in ("s4", "ms45")},
        "downstream": {
            "tasks": config["tasks"],
            "folds": config["folds"],
            "seed": config["seed"],
            "protocol": config["evaluation_protocol"],
            "completed_runs": {arm: 9 for arm in ("s4", "ms45")},
            "scheduler_reports": scheduler_reports,
        },
        "artifacts": {
            "sidecar_qc": str(OUTPUT / "sidecar_qc.json"),
            "matched_init": str(OUTPUT / "step0_matched_init.json"),
            "pretrain_acceptance": str(OUTPUT / "pretrain_5k_acceptance.json"),
            "per_fold_results": str(OUTPUT / "per_fold_results.csv"),
            "task_results": str(OUTPUT / "task_results.csv"),
            "summary": str(OUTPUT / "summary.json"),
            "report": str(OUTPUT / "REPORT.md"),
        },
        "follow_up_started": False,
    }
    (OUTPUT / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    report_tasks = tasks.copy()
    for column in ("s4", "ms45", "delta"):
        report_tasks[column] = report_tasks[column].map(lambda value: f"{value:.6f}")
    lines = [
        "# MTS-GLT-v2-MSContact-v1", "",
        "Primary matched contrast: `MS45 - S4`.", "",
        markdown_table(report_tasks), "",
        f"- S4 macro3: {summary['macro3']['s4']:.6f}",
        f"- MS45 macro3: {summary['macro3']['ms45']:.6f}",
        f"- macro3 delta: {summary['macro3_delta']:.6f}",
        f"- median task delta: {summary['median_task_delta']:.6f}",
        f"- positive tasks/folds: {summary['positive_tasks']}/3; {summary['positive_folds']}/9",
        f"- continue condition: {summary['continue_condition']}", "",
        "The pretraining arms each completed 5,000 optimizer steps with finite MA/ML/NCE metrics, and all 18 downstream runs completed.", "",
        "This screening supports an outer-shell contribution candidate. It does not establish that explicit shell decomposition beats a mixed <5 A graph, and no follow-up experiment was started.",
    ]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
