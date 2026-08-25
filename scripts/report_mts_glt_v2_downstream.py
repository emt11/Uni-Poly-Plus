#!/usr/bin/env python3
"""Summarize paired MTS-GLT-v2 O8-only and atom-fused shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _read_fold(root, mode, task, fold):
    path = root / mode / "shards" / "42" / task / f"fold_{fold}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise RuntimeError(f"expected one shard row: {path}")
    value = float(frame.iloc[0]["avg_test_r2"])
    if not np.isfinite(value):
        raise RuntimeError(f"non-finite R2: {path}")
    return value


def summarize(run_name, tasks, folds, fused_mode="o8_glt_atom", base_mode="o8_only", result_base=None):
    root = Path(result_base) / run_name if result_base else ROOT / "results/mts_glt_v2/downstream" / run_name
    task_rows = {}
    for task in tasks:
        o8 = np.asarray([_read_fold(root, base_mode, task, fold) for fold in folds])
        fused = np.asarray([_read_fold(root, fused_mode, task, fold) for fold in folds])
        delta = fused - o8
        task_rows[task] = {
            "o8_mean_r2": float(o8.mean()),
            "o8_sample_std_r2": float(o8.std(ddof=1)) if len(o8) > 1 else 0.0,
            "fused_mean_r2": float(fused.mean()),
            "fused_sample_std_r2": float(fused.std(ddof=1)) if len(fused) > 1 else 0.0,
            "mean_delta": float(delta.mean()),
            "sample_std_delta": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
            "median_fold_delta": float(np.median(delta)),
            "positive_folds": int((delta > 0).sum()),
            "fold_o8_r2": o8.tolist(),
            "fold_fused_r2": fused.tolist(),
        }
    task_deltas = np.asarray([row["mean_delta"] for row in task_rows.values()])
    payload = {
        "schema": "mts-glt-v2-paired-downstream-summary-v1",
        "run_name": run_name,
        "tasks": task_rows,
        "macro_o8": float(np.mean([row["o8_mean_r2"] for row in task_rows.values()])),
        "macro_fused": float(np.mean([row["fused_mean_r2"] for row in task_rows.values()])),
        "macro_delta": float(task_deltas.mean()),
        "median_task_delta": float(np.median(task_deltas)),
        "positive_tasks": int((task_deltas > 0).sum()),
    }
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--result-base")
    parser.add_argument("--tasks", nargs="+", default=["xc", "ei", "egb"])
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--fused-mode", choices=("o8_glt_atom", "o8_glt_atom_desc", "o8_glt_graph"),
        default="o8_glt_atom",
    )
    parser.add_argument(
        "--base-mode", choices=("o8_only", "o8_glt_atom"), default="o8_only"
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    payload = summarize(
        args.run_name, args.tasks, args.folds, args.fused_mode, args.base_mode,
        args.result_base,
    )
    output = Path(args.output) if args.output else (
        (Path(args.result_base) if args.result_base else ROOT / "results/mts_glt_v2/downstream")
        / args.run_name / "paired_summary.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({
        key: payload[key] for key in (
            "macro_o8", "macro_fused", "macro_delta",
            "median_task_delta", "positive_tasks",
        )
    }, sort_keys=True))


if __name__ == "__main__":
    main()
