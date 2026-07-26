#!/usr/bin/env python3
"""Compare one shared downstream configuration against a shared baseline.

Candidate promotion is based on inner-validation metrics across the complete task
suite. Test metrics are reported only for final evaluation and target tracking.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--best-target",
        default="results/best_result.csv",
        help="CSV containing task and best_r2; all requested tasks must exceed it to stop.",
    )
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument(
        "--selection-metric",
        choices=("avg_best_val_r2",),
        default="avg_best_val_r2",
        help="Inner-validation metric used for candidate promotion.",
    )
    parser.add_argument(
        "--min-task-wins",
        type=int,
        default=5,
        help="Minimum number of tasks whose selection metric must not decrease.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow diagnostics on a task subset; partial runs cannot be promoted.",
    )
    parser.add_argument("--min-mean-r2-gain", type=float, default=0.0)
    parser.add_argument("--first-stage-mean-r2", type=float, default=0.829)
    parser.add_argument("--max-task-r2-drop", type=float, default=0.01)
    parser.add_argument("--max-mean-std-increase", type=float, default=0.01)
    parser.add_argument("--fail-on-reject", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def load_results(path):
    frame = pd.read_csv(path)
    required = {
        "task",
        "avg_best_val_r2",
        "std_best_val_r2",
        "avg_test_r2",
        "std_test_r2",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["task"].duplicated().any():
        duplicates = frame.loc[frame["task"].duplicated(), "task"].tolist()
        raise ValueError(f"{path} contains duplicate tasks: {duplicates}")
    return frame.set_index("task")


def main():
    args = parse_args()
    baseline = load_results(args.baseline)
    candidate = load_results(args.candidate)
    best_target = pd.read_csv(args.best_target).set_index("task")
    if "best_r2" not in best_target.columns:
        raise ValueError(f"{args.best_target} is missing column: best_r2")
    expected_tasks = set(best_target.index)
    tasks = args.tasks or sorted(expected_tasks)
    missing = [task for task in tasks if task not in baseline.index or task not in candidate.index]
    if missing:
        raise ValueError(f"Requested tasks missing from baseline or candidate: {missing}")
    if not tasks:
        raise ValueError("No matching tasks to compare")
    missing_targets = [task for task in tasks if task not in best_target.index]
    if missing_targets:
        raise ValueError(f"Requested tasks missing from best target: {missing_targets}")
    full_task_set = len(tasks) == len(expected_tasks) and set(tasks) == expected_tasks
    if not full_task_set and not args.allow_partial:
        raise ValueError(
            "Candidate selection requires the complete task suite; "
            "use --allow-partial only for non-promotable diagnostics"
        )

    rows = []
    for task in tasks:
        base_row = baseline.loc[task]
        candidate_row = candidate.loc[task]
        rows.append({
            "task": task,
            "baseline_selection": float(base_row[args.selection_metric]),
            "candidate_selection": float(candidate_row[args.selection_metric]),
            "selection_gain": float(
                candidate_row[args.selection_metric] - base_row[args.selection_metric]
            ),
            "baseline_selection_std": float(base_row["std_best_val_r2"]),
            "candidate_selection_std": float(candidate_row["std_best_val_r2"]),
            "selection_std_change": float(
                candidate_row["std_best_val_r2"] - base_row["std_best_val_r2"]
            ),
            "baseline_r2": float(base_row["avg_test_r2"]),
            "candidate_r2": float(candidate_row["avg_test_r2"]),
            "r2_gain": float(candidate_row["avg_test_r2"] - base_row["avg_test_r2"]),
            "baseline_std": float(base_row["std_test_r2"]),
            "candidate_std": float(candidate_row["std_test_r2"]),
            "std_change": float(candidate_row["std_test_r2"] - base_row["std_test_r2"]),
            "target_best_r2": float(best_target.loc[task, "best_r2"]),
            "gap_to_target_best": float(
                candidate_row["avg_test_r2"] - best_target.loc[task, "best_r2"]
            ),
        })

    mean_selection_gain = sum(row["selection_gain"] for row in rows) / len(rows)
    mean_selection_std_change = (
        sum(row["selection_std_change"] for row in rows) / len(rows)
    )
    selection_task_wins = sum(row["selection_gain"] >= 0.0 for row in rows)
    worst_selection_gain = min(row["selection_gain"] for row in rows)
    mean_gain = sum(row["r2_gain"] for row in rows) / len(rows)
    baseline_mean_r2 = sum(row["baseline_r2"] for row in rows) / len(rows)
    candidate_mean_r2 = sum(row["candidate_r2"] for row in rows) / len(rows)
    promote = (
        full_task_set
        and mean_selection_gain >= args.min_mean_r2_gain
        and mean_selection_std_change <= args.max_mean_std_increase
        and worst_selection_gain >= -args.max_task_r2_drop
        and selection_task_wins >= args.min_task_wins
    )
    first_stage_met = (
        full_task_set
        and candidate_mean_r2 >= args.first_stage_mean_r2
    )
    stop_condition_met = (
        full_task_set
        and all(row["gap_to_target_best"] >= 0.0 for row in rows)
    )
    report = {
        "baseline": str(Path(args.baseline)),
        "candidate": str(Path(args.candidate)),
        "tasks": tasks,
        "baseline_mean_r2": baseline_mean_r2,
        "candidate_mean_r2": candidate_mean_r2,
        "mean_test_r2_gain": mean_gain,
        "selection_metric": args.selection_metric,
        "mean_selection_gain": mean_selection_gain,
        "worst_selection_gain": worst_selection_gain,
        "selection_task_wins": selection_task_wins,
        "mean_selection_std_change": mean_selection_std_change,
        "full_task_set": full_task_set,
        "thresholds": {
            "min_mean_r2_gain": args.min_mean_r2_gain,
            "min_task_wins": args.min_task_wins,
            "first_stage_mean_r2": args.first_stage_mean_r2,
            "max_task_r2_drop": args.max_task_r2_drop,
            "max_mean_std_increase": args.max_mean_std_increase,
        },
        "verdict": "promote" if promote else "reject",
        "promotion_basis": "complete-task inner-validation only",
        "first_stage_condition": "all_8_tasks_mean_r2_at_least_threshold",
        "first_stage_met": first_stage_met,
        "stop_condition": "all_8_tasks_reach_or_exceed_best_result.best_r2",
        "stop_condition_met": stop_condition_met,
        "per_task": rows,
    }
    output = Path(args.output) if args.output else Path(args.candidate).with_suffix(".comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.fail_on_reject and not promote:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
