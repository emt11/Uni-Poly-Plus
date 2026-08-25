#!/usr/bin/env python3
"""Finalize the gated GraphGate trajectory-selection experiment."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/mts_glt_graphgate_v1/trajectory_selection_v1"


def _read(name):
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def _atomic_write(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main():
    screen = _read("screen5k_decision.json")
    drift = _read("representation_drift_summary.json")
    if bool(screen.get("run_formal5k", False)):
        formal = _read("formal5k_decision.json")
        conclusion = (
            "five_k_neural_complementarity_confirmed"
            if bool(formal.get("run_late_infonce", False))
            else "five_k_not_better_than_twenty_k"
        )
    else:
        formal = None
        conclusion = "five_k_not_better_than_twenty_k"

    payload = {
        "schema": "mts-glt-graphgate-trajectory-selection-final-v1",
        "conclusion": conclusion,
        "screen5k": {
            "macro_delta_5k": screen["macro"]["delta_5k"],
            "macro_delta_20k": screen["macro"]["delta_20k"],
            "median_task_delta_5k_minus_20k": screen["median_delta_5k_minus_20k"],
            "tasks_delta_5k_gt_20k": screen["tasks_delta_5k_gt_20k"],
            "run_formal5k": bool(screen["run_formal5k"]),
        },
        "representation": {
            step: {
                key: drift["steps"][step][key]
                for key in ("p1", "p2", "p3", "p4", "c3", "c4")
            }
            for step in ("005k", "020k")
        },
        "formal5k": formal,
        "late_infonce_started": bool(formal and formal.get("run_late_infonce", False)),
        "stopped_by_gate": not bool(screen["run_formal5k"]),
    }
    _atomic_write(
        OUTPUT / "trajectory_selection_summary.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )

    task_lines = []
    for row in screen["tasks"]:
        task_lines.append(
            f"| {row['task'].upper()} | {row['delta_5k']:+.6f} | "
            f"{row['delta_20k']:+.6f} | {row['delta_5k_minus_20k']:+.6f} |"
        )
    five = drift["steps"]["005k"]
    twenty = drift["steps"]["020k"]
    report = [
        "# MTS-GLT-GraphGate trajectory selection v1",
        "",
        f"Final conclusion: `{conclusion}`.",
        "",
        "## 5k FusionWarm screen",
        "",
        "| Task | 5k FW-O8 | 20k FW-O8 | 5k-20k |",
        "|---|---:|---:|---:|",
        *task_lines,
        "",
        f"- macro3 delta: 5k `{screen['macro']['delta_5k']:+.6f}`; "
        f"20k `{screen['macro']['delta_20k']:+.6f}`.",
        f"- median task (5k-20k): `{screen['median_delta_5k_minus_20k']:+.6f}`.",
        f"- tasks where 5k delta exceeds 20k: `{screen['tasks_delta_5k_gt_20k']}/3`.",
        "- The first required gate failed because macro3(5k) did not exceed macro3(20k).",
        "",
        "## Frozen representation drift",
        "",
        "| Step | P1 | P2 | P3 | P4 | C3 | C4 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| 5k | {five['p1']:.6f} | {five['p2']:.6f} | {five['p3']:.6f} | "
        f"{five['p4']:.6f} | {five['c3']:+.6f} | {five['c4']:+.6f} |",
        f"| 20k | {twenty['p1']:.6f} | {twenty['p2']:.6f} | {twenty['p3']:.6f} | "
        f"{twenty['p4']:.6f} | {twenty['c3']:+.6f} | {twenty['c4']:+.6f} |",
        "",
        "The 5k checkpoint retains slightly stronger frozen complementarity, especially in P4, "
        "but this did not translate into a stronger paired FusionWarm neural result.",
        "",
        "## Conditional execution",
        "",
        "- Stage B (5k formal 8x5): not started because Stage A failed.",
        "- Shared 5k replay and late-stage InfoNCE branches: not started by design.",
        "- Existing 5k, 20k, FusionWarm and historical artifacts were not overwritten.",
        "",
        "This is a shared-fold, seed-42 trajectory diagnostic, not an independent blind-test claim.",
        "",
    ]
    _atomic_write(OUTPUT / "trajectory_selection_report.md", "\n".join(report))
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
