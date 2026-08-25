#!/usr/bin/env python3
"""Summarize GraphGate FusionWarm against its paired formal references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/mts_glt_graphgate_v1/fusion_realization_v1"
FORMAL = ROOT / "results/mts_glt_graphgate_v1/downstream/formal_20k"
ALL_TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
SCREENING_TASKS = ("xc", "ei", "eps")


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
                raise SystemExit(f"missing FusionWarm unit: {path}")
            unit = json.loads(path.read_text(encoding="utf-8"))
            if unit.get("fusion_strategy") != "fusion_warm":
                raise SystemExit(f"unit is not FusionWarm: {path}")
            if unit.get("fusion_kind") != "channel":
                raise SystemExit(f"unit is not GraphGate channel fusion: {path}")
            units.append(unit)
    return units


def _historical(tasks):
    paired = json.loads((FORMAL / "paired_summary.json").read_text(encoding="utf-8"))
    audit = {}
    for task in tasks:
        for fold in range(5):
            path = (
                FORMAL / "o8_glt_graph/fusion_audit_units"
                / task / f"fold_{fold}.json"
            )
            item = json.loads(path.read_text(encoding="utf-8"))
            audit[(task, fold)] = item
    return paired, audit


def _summarize(tasks, units, paired, historical_audit):
    rows, trajectory = [], []
    for unit in units:
        task, fold = str(unit["task"]), int(unit["fold"])
        reference = paired["tasks"][task]
        expected_seed = int(historical_audit[(task, fold)]["fold_seed"])
        if int(unit["fold_seed"]) != expected_seed:
            raise SystemExit(
                f"fold seed mismatch for {task}/fold{fold}: "
                f"{unit['fold_seed']} != {expected_seed}"
            )
        warm = float(unit["rerun_fused_r2"])
        o8 = float(reference["fold_o8_r2"][fold])
        legacy = float(reference["fold_fused_r2"][fold])
        rows.append({
            "task": task, "fold": fold, "fold_seed": expected_seed,
            "o8_only_r2": o8, "legacy_fused_r2": legacy,
            "fusion_warm_r2": warm,
            "delta_vs_o8": warm - o8,
            "delta_vs_legacy": warm - legacy,
            "best_epoch": int(unit["best_epoch"]),
            "projection_relative_change": float(unit["projection_relative_change"]),
            "best_rho_median": float(unit["validation"]["rho"]["median"]),
            "best_alpha_mean_abs": float(unit["final_alpha"]["mean_abs"]),
        })
        for item in unit["gate_trajectory"]:
            trajectory.append({"task": task, "fold": fold, **item})

    folds = pd.DataFrame(rows).sort_values(["task", "fold"])
    task_rows = []
    for task in tasks:
        selected = folds[folds.task == task]
        task_rows.append({
            "task": task,
            "o8_only_mean": float(selected.o8_only_r2.mean()),
            "legacy_fused_mean": float(selected.legacy_fused_r2.mean()),
            "fusion_warm_mean": float(selected.fusion_warm_r2.mean()),
            "fusion_warm_sample_std": float(selected.fusion_warm_r2.std(ddof=1)),
            "delta_vs_o8": float(selected.delta_vs_o8.mean()),
            "delta_vs_legacy": float(selected.delta_vs_legacy.mean()),
            "positive_folds_vs_o8": int((selected.delta_vs_o8 > 0).sum()),
            "positive_folds_vs_legacy": int((selected.delta_vs_legacy > 0).sum()),
        })
    tasks_frame = pd.DataFrame(task_rows)
    delta_o8 = tasks_frame.delta_vs_o8.to_numpy(dtype=np.float64)
    delta_legacy = tasks_frame.delta_vs_legacy.to_numpy(dtype=np.float64)
    vs_o8 = {
        "macro": float(delta_o8.mean()),
        "median_task": float(np.median(delta_o8)),
        "positive_tasks": int((delta_o8 > 0).sum()),
    }
    vs_legacy = {
        "macro": float(delta_legacy.mean()),
        "median_task": float(np.median(delta_legacy)),
        "positive_tasks": int((delta_legacy > 0).sum()),
    }
    threshold = 5 if len(tasks) == 8 else 2
    pass_o8 = bool(
        vs_o8["macro"] > 0 and vs_o8["median_task"] > 0
        and vs_o8["positive_tasks"] >= threshold
    )
    pass_legacy = bool(
        vs_legacy["macro"] > 0 and vs_legacy["median_task"] > 0
        and vs_legacy["positive_tasks"] >= threshold
    )
    expand = bool(len(tasks) == 3 and pass_o8 and pass_legacy)
    if len(tasks) == 8:
        decision = (
            "fusion_realization_success" if pass_o8 and pass_legacy
            else (
                "cold_start_but_additive_fusion_insufficient"
                if pass_legacy else "fusion_warm_hypothesis_rejected"
            )
        )
    else:
        decision = "go_expand_8x5" if expand else "stop_after_screening"
    summary = {
        "schema": "mts-glt-graphgate-fusion-realization-v1",
        "evaluation_protocol": "historical_shared5",
        "independent_blind_test": False,
        "tasks": task_rows,
        "macro": {
            "o8_only": float(tasks_frame.o8_only_mean.mean()),
            "legacy_fused": float(tasks_frame.legacy_fused_mean.mean()),
            "fusion_warm": float(tasks_frame.fusion_warm_mean.mean()),
        },
        "delta_vs_o8": vs_o8,
        "delta_vs_legacy": vs_legacy,
        "pass_vs_o8": pass_o8,
        "pass_vs_legacy": pass_legacy,
        "expand_8x5": expand,
        "decision": decision,
        "mechanism": {
            "median_best_rho": float(folds.best_rho_median.median()),
            "mean_best_alpha_abs": float(folds.best_alpha_mean_abs.mean()),
            "mean_projection_relative_change": float(
                folds.projection_relative_change.mean()
            ),
        },
    }
    return folds, pd.DataFrame(trajectory), tasks_frame, summary


def _markdown(scope, task_frame, summary):
    lines = [
        "# MTS-GLT-GraphGate Fusion Realization v1", "",
        "> historical_shared5共享验证/测试fold，非独立盲测；未重新预训练。", "",
        f"- scope: `{scope}`",
        f"- decision: `{summary['decision']}`", "",
        "| Task | O8-only | legacy fused | FusionWarm | FW-O8 | FW-legacy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in task_frame.to_dict("records"):
        lines.append(
            f"| {row['task']} | {row['o8_only_mean']:.6f} | "
            f"{row['legacy_fused_mean']:.6f} | {row['fusion_warm_mean']:.6f} | "
            f"{row['delta_vs_o8']:+.6f} | {row['delta_vs_legacy']:+.6f} |"
        )
    lines.extend([
        "",
        f"- FusionWarm macro: `{summary['macro']['fusion_warm']:.6f}`",
        f"- delta vs O8: `{summary['delta_vs_o8']['macro']:+.6f}`",
        f"- delta vs legacy fused: `{summary['delta_vs_legacy']['macro']:+.6f}`",
        f"- median best rho: `{summary['mechanism']['median_best_rho']:.6f}`",
        f"- mean best |alpha|: `{summary['mechanism']['mean_best_alpha_abs']:.6f}`",
        f"- mean projection change: `{summary['mechanism']['mean_projection_relative_change']:.6f}`",
        "",
    ])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("screening", "full"), default="screening")
    args = parser.parse_args(argv)
    tasks = ALL_TASKS if args.scope == "full" else SCREENING_TASKS
    units = _read_units(tasks)
    paired, historical_audit = _historical(tasks)
    folds, trajectory, task_frame, summary = _summarize(
        tasks, units, paired, historical_audit
    )
    folds.to_csv(OUTPUT / "fusion_realization_fold_results.csv", index=False)
    trajectory.to_csv(OUTPUT / "fusion_realization_trajectory.csv", index=False)
    name = (
        "fusion_realization_full.json"
        if args.scope == "full" else "fusion_realization_screening.json"
    )
    _atomic_json(OUTPUT / name, summary)
    _atomic_text(
        OUTPUT / "fusion_realization_report.md",
        _markdown(args.scope, task_frame, summary),
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
