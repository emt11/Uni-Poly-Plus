#!/usr/bin/env python3
"""Report the complete 8-task x 5-fold Warm5 minus Warm0 ablation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from report_mts_glt_graphgate_fusionwarm_vs_nowarm_20k import (
    ROOT,
    OUTPUT,
    WARM5,
    _atomic_text,
    _load,
    _verify_checkpoint,
)


CONFIG = ROOT / "configs/mts/glt_graphgate_v1/fusionwarm_vs_nowarm_20k_8task_fold01234_v1.json"
REPORT_ROOT = OUTPUT / "eight_task_fold01234"
TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
FOLDS = (0, 1, 2, 3, 4)


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    _verify_checkpoint(config)
    rows = []
    for task in TASKS:
        for fold in FOLDS:
            warm0 = _load(OUTPUT / "Warm0", task, fold, 0)
            warm5 = _load(WARM5, task, fold, 5)
            if warm0["fold_seed"] != warm5["fold_seed"]:
                raise SystemExit(f"fold seed mismatch {task}/fold{fold}")
            if not np.array_equal(warm0["indices"], warm5["indices"]):
                raise SystemExit(f"sample index mismatch {task}/fold{fold}")
            if not np.array_equal(warm0["y_true"], warm5["y_true"]):
                raise SystemExit(f"target mismatch {task}/fold{fold}")
            rows.append({
                "task": task,
                "fold": fold,
                "Warm0_R2": warm0["r2"],
                "Warm5_R2": warm5["r2"],
                "DeltaWarm": warm5["r2"] - warm0["r2"],
            })

    folds = pd.DataFrame(rows).sort_values(["task", "fold"])
    task_rows = []
    for task in TASKS:
        selected = folds[folds.task == task]
        task_rows.append({
            "task": task,
            "Warm0_mean": float(selected.Warm0_R2.mean()),
            "Warm5_mean": float(selected.Warm5_R2.mean()),
            "DeltaWarm_mean": float(selected.DeltaWarm.mean()),
            "positive_folds": int((selected.DeltaWarm > 0).sum()),
        })
    tasks = pd.DataFrame(task_rows)
    task_delta = tasks.DeltaWarm_mean.to_numpy(dtype=np.float64)
    fold_delta = folds.DeltaWarm.to_numpy(dtype=np.float64)
    macro_warm0 = float(tasks.Warm0_mean.mean())
    macro_warm5 = float(tasks.Warm5_mean.mean())
    macro_delta = float(task_delta.mean())
    median_task = float(np.median(task_delta))
    positive_tasks = int((task_delta > 0).sum())
    criteria = {
        "macro8_gt_zero": bool(macro_delta > 0),
        "median_task_gt_zero": bool(median_task > 0),
        "positive_tasks_ge_5": bool(positive_tasks >= 5),
    }
    decision = "PASS" if all(criteria.values()) else "FAIL"
    summary = {
        "schema": "mts-glt-graphgate-fusionwarm-vs-nowarm-20k-8task-fold01234-summary-v1",
        "tasks": list(TASKS),
        "folds": list(FOLDS),
        "checkpoint": str((ROOT / config["checkpoint"]).resolve()),
        "new_warm0_runs": 31,
        "new_warm5_runs": 0,
        "reused_warm0_runs": 9,
        "reused_warm5_runs": 40,
        "failed_runs": 0,
        "task_results": task_rows,
        "macro8_warm0": macro_warm0,
        "macro8_warm5": macro_warm5,
        "macro8_delta_warm": macro_delta,
        "macro8_difference_check": float(macro_warm5 - macro_warm0),
        "median_task_delta_warm": median_task,
        "positive_delta_warm_tasks": positive_tasks,
        "positive_delta_warm_folds": int((fold_delta > 0).sum()),
        "fold_delta_warm_mean": float(fold_delta.mean()),
        "fold_delta_warm_median": float(np.median(fold_delta)),
        "fold_delta_warm_p25": float(np.quantile(fold_delta, 0.25)),
        "fold_delta_warm_p75": float(np.quantile(fold_delta, 0.75)),
        "fold_delta_warm_min": float(fold_delta.min()),
        "fold_delta_warm_max": float(fold_delta.max()),
        "criteria": criteria,
        "final_warm_decision": decision,
        "screening_macro3_delta_warm": 0.013960853910172962,
        "screening_positive_tasks": 3,
        "screening_positive_folds": 8,
        "shared_validation_test": True,
    }
    if not np.isclose(macro_delta, macro_warm5 - macro_warm0, atol=1e-12, rtol=0.0):
        raise SystemExit("macro8 delta does not equal Warm5 minus Warm0")

    _atomic_text(REPORT_ROOT / "per_fold_results.csv", folds.to_csv(index=False))
    _atomic_text(REPORT_ROOT / "task_results.csv", tasks.to_csv(index=False))
    _atomic_text(
        REPORT_ROOT / "fusionwarm_vs_nowarm_20k_8task_fold01234_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
