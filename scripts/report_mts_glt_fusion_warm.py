#!/usr/bin/env python3
"""Summarize matched FusionWarm folds and apply the predeclared GO rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/mts_glt_v1/fusion_warm"
FORMAL = ROOT / "results/mts_glt_v1/final_report.json"
ALL_TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
SCREENING_TASKS = ("ei", "nc", "eps")


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path, payload):
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_units(tasks):
    units = []
    for task in tasks:
        for fold in range(5):
            path = OUTPUT / "fusion_audit_units" / task / f"fold_{fold}.json"
            if not path.is_file():
                raise SystemExit(f"missing FusionWarm audit unit: {path}")
            unit = json.loads(path.read_text(encoding="utf-8"))
            if unit.get("fusion_strategy") != "fusion_warm":
                raise SystemExit(f"not a FusionWarm unit: {path}")
            units.append(unit)
    return units


def _summarize(tasks, units, formal):
    fold_rows, trajectory_rows, task_rows = [], [], []
    for unit in units:
        task, fold = str(unit["task"]), int(unit["fold"])
        old = formal["tasks"][task]
        old_fused = float(old["o8_glt"]["fold_r2"][fold])
        o8 = float(old["o8_only"]["fold_r2"][fold])
        fusion_warm = float(unit["rerun_fused_r2"])
        fold_rows.append({
            "task": task,
            "fold": fold,
            "o8_only_r2": o8,
            "old_o8_glt_r2": old_fused,
            "fusion_warm_r2": fusion_warm,
            "fusion_warm_minus_o8": fusion_warm - o8,
            "fusion_warm_minus_old_fused": fusion_warm - old_fused,
            "best_epoch": int(unit["best_epoch"]),
            "final_tanh_gate": float(unit["final_tanh_gate"]),
            "projection_relative_change": float(unit["projection_relative_change"]),
        })
        for row in unit["gate_trajectory"]:
            trajectory_rows.append({"task": task, "fold": fold, **row})

    frame = pd.DataFrame(fold_rows).sort_values(["task", "fold"])
    for task in tasks:
        selected = frame[frame.task == task]
        task_rows.append({
            "task": task,
            "o8_only_mean": float(selected.o8_only_r2.mean()),
            "old_o8_glt_mean": float(selected.old_o8_glt_r2.mean()),
            "fusion_warm_mean": float(selected.fusion_warm_r2.mean()),
            "fusion_warm_sample_std": float(selected.fusion_warm_r2.std(ddof=1)),
            "delta_vs_o8": float(selected.fusion_warm_minus_o8.mean()),
            "delta_vs_old_fused": float(
                selected.fusion_warm_minus_old_fused.mean()
            ),
            "positive_folds_vs_o8": int(
                (selected.fusion_warm_minus_o8 > 0).sum()
            ),
            "positive_folds_vs_old_fused": int(
                (selected.fusion_warm_minus_old_fused > 0).sum()
            ),
        })
    task_frame = pd.DataFrame(task_rows)
    macro = {
        "o8_only": float(task_frame.o8_only_mean.mean()),
        "old_o8_glt": float(task_frame.old_o8_glt_mean.mean()),
        "fusion_warm": float(task_frame.fusion_warm_mean.mean()),
    }
    deltas_o8 = task_frame.delta_vs_o8.to_numpy()
    deltas_old = task_frame.delta_vs_old_fused.to_numpy()
    minimum_improvement = bool(
        deltas_old.mean() > 0
        and np.median(deltas_old) > 0
        and int((deltas_old > 0).sum()) >= 2
    )
    expand = bool(
        deltas_o8.mean() > 0
        and np.median(deltas_o8) > 0
        and int((deltas_o8 > 0).sum()) >= 2
    )
    initial_rho_medians = np.asarray([
        unit["initial_validation"]["rho"]["median"] for unit in units
    ], dtype=np.float64)
    best_rho_medians = np.asarray([
        unit["validation"]["rho"]["median"] for unit in units
    ], dtype=np.float64)
    final_tanh_gates = np.asarray([
        unit["final_tanh_gate"] for unit in units
    ], dtype=np.float64)
    projection_changes = np.asarray([
        unit["projection_relative_change"] for unit in units
    ], dtype=np.float64)
    summary = {
        "schema": "mts-glt-v1-fusion-warm-report-v1",
        "evaluation_protocol": "historical_shared5",
        "independent_blind_test": False,
        "tasks": task_rows,
        "macro": macro,
        "delta_vs_o8": {
            "macro": float(deltas_o8.mean()),
            "median_task": float(np.median(deltas_o8)),
            "positive_tasks": int((deltas_o8 > 0).sum()),
        },
        "delta_vs_old_fused": {
            "macro": float(deltas_old.mean()),
            "median_task": float(np.median(deltas_old)),
            "positive_tasks": int((deltas_old > 0).sum()),
        },
        "minimum_improvement": minimum_improvement,
        "expand_8x5": expand,
        "decision": "go_expand_8x5" if expand else "stop_after_screening",
        "mechanism": {
            "folds": len(units),
            "initial_rho_median_across_folds": float(
                np.median(initial_rho_medians)
            ),
            "best_state_rho_median_across_folds": float(
                np.median(best_rho_medians)
            ),
            "mean_final_tanh_gate": float(final_tanh_gates.mean()),
            "mean_abs_final_tanh_gate": float(
                np.abs(final_tanh_gates).mean()
            ),
            "mean_projection_relative_change": float(
                projection_changes.mean()
            ),
        },
    }
    return frame, pd.DataFrame(trajectory_rows), task_frame, summary


def _markdown(scope, task_frame, summary):
    lines = [
        "# MTS-GLT-v1 FusionWarm",
        "",
        "> historical_shared5共享验证/测试fold，非独立盲测；预训练checkpoint未改变。",
        "",
        f"- scope: `{scope}`",
        f"- decision: `{summary['decision']}`",
        "",
        "| Task | O8-only | old O8+GLT | FusionWarm | FW-O8 | FW-old | positive folds vs O8 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in task_frame.to_dict("records"):
        lines.append(
            f"| {row['task']} | {row['o8_only_mean']:.6f} | "
            f"{row['old_o8_glt_mean']:.6f} | {row['fusion_warm_mean']:.6f} | "
            f"{row['delta_vs_o8']:+.6f} | {row['delta_vs_old_fused']:+.6f} | "
            f"{row['positive_folds_vs_o8']}/5 |"
        )
    lines.extend([
        "",
        f"- FusionWarm macro: `{summary['macro']['fusion_warm']:.6f}`",
        f"- macro delta vs O8: `{summary['delta_vs_o8']['macro']:+.6f}`",
        f"- median task delta vs O8: `{summary['delta_vs_o8']['median_task']:+.6f}`",
        f"- positive tasks vs O8: `{summary['delta_vs_o8']['positive_tasks']}/{len(task_frame)}`",
        f"- macro delta vs old fused: `{summary['delta_vs_old_fused']['macro']:+.6f}`",
        f"- median initial rho: `{summary['mechanism']['initial_rho_median_across_folds']:.6f}`",
        f"- median best-state rho: `{summary['mechanism']['best_state_rho_median_across_folds']:.6f}`",
        f"- mean abs final tanh(gate): `{summary['mechanism']['mean_abs_final_tanh_gate']:.6f}`",
        f"- mean projection relative change: `{summary['mechanism']['mean_projection_relative_change']:.6f}`",
        "",
    ])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("screening", "full"), default="screening")
    args = parser.parse_args(argv)
    tasks = ALL_TASKS if args.scope == "full" else SCREENING_TASKS
    formal = json.loads(FORMAL.read_text(encoding="utf-8"))
    units = _read_units(tasks)
    folds, trajectory, task_frame, summary = _summarize(tasks, units, formal)
    folds.to_csv(OUTPUT / "fusion_warm_fold_results.csv", index=False)
    trajectory.to_csv(OUTPUT / "fusion_warm_trajectory.csv", index=False)
    report_name = (
        "fusion_warm_screening_report.json"
        if args.scope == "screening" else "fusion_warm_full_report.json"
    )
    _atomic_json(OUTPUT / report_name, summary)
    _atomic_text(
        OUTPUT / "fusion_warm_report.md",
        _markdown(args.scope, task_frame, summary),
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
