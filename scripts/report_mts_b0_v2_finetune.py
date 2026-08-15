#!/usr/bin/env python3
"""Aggregate the B0-v2 formal 8x5 finetune shards into the report table.

Each shard holds exactly one fold; per_fold_metrics contains the fold test R2.
The report prints per-task 5-fold mean +/- sample std (3 decimals), the 8-task
macro mean, the 7-task macro mean (excluding xc), the xc contribution and the
ei value, matching the AGENTS.md reporting protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TASKS = ["eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc"]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    root = Path(args.results_dir)
    shards = root / "shards" / "42"
    task_rows = {}
    for task in args.tasks:
        fold_r2 = []
        for fold in range(5):
            shard = shards / task / f"fold_{fold}.csv"
            if not shard.is_file():
                raise SystemExit(f"missing shard: {shard}")
            frame = pd.read_csv(shard)
            if len(frame) != 1:
                raise SystemExit(f"shard must contain exactly one fold row: {shard}")
            row = frame.iloc[0]
            metrics = json.loads(str(row["per_fold_metrics"]))
            if len(metrics) != 1 or int(metrics[0].get("fold", -1)) != fold:
                raise SystemExit(f"shard fold mismatch: {shard}")
            fold_r2.append(float(metrics[0]["test_r2"]))
        task_rows[task] = np.asarray(fold_r2)

    means = {task: float(np.mean(values)) for task, values in task_rows.items()}
    stds = {task: float(np.std(values, ddof=1)) for task, values in task_rows.items()}
    macro_8 = float(np.mean([means[task] for task in args.tasks]))
    seven = [task for task in args.tasks if task != "xc"]
    macro_7 = float(np.mean([means[task] for task in seven]))
    report = {
        "schema": "mts-b0-v2-formal-finetune-report-v1",
        "per_task_mean_std": {
            task: {
                "mean_r2": round(means[task], 6),
                "sample_std_r2": round(stds[task], 6),
                "folds": [float(value) for value in task_rows[task]],
            }
            for task in args.tasks
        },
        "macro_8_task_mean_r2": round(macro_8, 6),
        "macro_7_task_mean_r2_excluding_xc": round(macro_7, 6),
        "xc_contribution": round(means.get("xc", 0.0) / max(macro_8, 1e-12), 6),
        "ei_mean_r2": round(means.get("ei", float("nan")), 6),
    }
    print("task    mean+-std")
    for task in args.tasks:
        print(f"{task:6s} {means[task]:.3f} +/- {stds[task]:.3f}")
    print(f"macro8 {macro_8:.3f}  macro7(exc xc) {macro_7:.3f}  "
          f"xc {means.get('xc', float('nan')):.3f}  ei {means.get('ei', float('nan')):.3f}")
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    import sys as _sys
    raise SystemExit(main(_sys.argv[1:]))
