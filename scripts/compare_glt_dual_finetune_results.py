#!/usr/bin/env python3
"""Phase 13 downstream comparison of verified aggregations, per-task and macro.

Reads aggregator summaries that already passed verified re-aggregation
(protocol, fold identity, recomputed metrics, OOF coverage).  The reporting
format follows the project convention for downstream results: per-task test R2,
best fold R2, the mean-minus-best gap, and the macro average -- no MAE/RMSE/epoch.

The aggregator deliberately emits no macro when the task count is not 8, so a
non-8-task macro is derived here from the verified per-task means and labelled
with its task count.  Read-only; writes one JSON and one Markdown table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise SystemExit(f"aggregation did not pass verification: {path}")
    tasks, folds = int(payload.get("task_count", 0)), int(payload.get("fold_count", 0))
    if tasks < 1 or folds != 5 * tasks:
        raise SystemExit(f"aggregation is not complete five-fold-per-task: {path}")
    return payload


def _view(body):
    folds = [float(row["test_r2"]) for row in body["folds"]]
    mean = float(body["test_r2"]["mean"])
    return {
        "test_r2_mean": mean,
        "test_r2_std": float(body["test_r2"]["std"]),
        "test_r2_best": max(folds),
        "gap_mean_minus_best": mean - max(folds),
        "pooled_oof_r2": float(body["pooled_oof"]["r2"]),
        "folds_completed": int(body["fold_count"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="append", required=True,
                        help="name=path of a reference aggregation, repeatable")
    parser.add_argument("--candidate", required=True, help="name=path of the candidate")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()

    def parse(value):
        name, _, path = value.partition("=")
        return name, _load(path)

    references = [parse(value) for value in args.reference]
    candidate_name, candidate = parse(args.candidate)
    candidate_tasks = {name: _view(body) for name, body in candidate["tasks"].items()}
    task_count = len(candidate_tasks)
    candidate_macro = statistics.fmean(v["test_r2_mean"] for v in candidate_tasks.values())
    candidate_macro_pooled = statistics.fmean(v["pooled_oof_r2"] for v in candidate_tasks.values())

    comparison = {
        "candidate": candidate_name,
        "shared_task_count": task_count,
        "macro_field_note": ("the aggregator emits macro8_r2 only for an 8-task aggregation; with "
                             f"{task_count} tasks this macro is derived from the verified per-task "
                             "means and must not be compared with 8-task macro8 numbers"),
        "references": {},
    }
    for name, payload in references:
        all_tasks = {task: _view(body) for task, body in payload["tasks"].items()}
        excluded = sorted(set(all_tasks) - set(candidate_tasks))
        shared = {task: body for task, body in all_tasks.items() if task in candidate_tasks}
        if not shared:
            raise SystemExit(f"no shared tasks between {name} and {candidate_name}")
        per_task = {
            task: {
                "reference_test_r2_mean": shared[task]["test_r2_mean"],
                "candidate_test_r2_mean": candidate_tasks[task]["test_r2_mean"],
                "delta_test_r2_mean": candidate_tasks[task]["test_r2_mean"]
                                       - shared[task]["test_r2_mean"],
                "reference_test_r2_best": shared[task]["test_r2_best"],
                "candidate_test_r2_best": candidate_tasks[task]["test_r2_best"],
                "delta_test_r2_best": candidate_tasks[task]["test_r2_best"]
                                      - shared[task]["test_r2_best"],
            }
            for task in sorted(shared)
        }
        reference_macro = statistics.fmean(v["test_r2_mean"] for v in shared.values())
        reference_macro_pooled = statistics.fmean(v["pooled_oof_r2"] for v in shared.values())
        comparison["references"][name] = {
            "tasks_excluded_for_comparison": excluded,
            "reference_macro_r2": reference_macro,
            "reference_macro_pooled_oof_r2": reference_macro_pooled,
            "delta_macro_r2": candidate_macro - reference_macro,
            "delta_macro_pooled_oof_r2": candidate_macro_pooled - reference_macro_pooled,
            "reference_published_macro8_r2": payload.get("macro8_r2"),
            "tasks": per_task,
            "tasks_improved": sorted(t for t, body in per_task.items()
                                     if body["delta_test_r2_mean"] > 0),
            "tasks_worsened": sorted(t for t, body in per_task.items()
                                     if body["delta_test_r2_mean"] < 0),
        }

    report = {
        "per_task": candidate_tasks,
        "macro_label": f"macro{task_count}",
        "macro_r2": candidate_macro,
        "macro_pooled_oof_r2": candidate_macro_pooled,
        "candidate_aggregation_root": candidate.get("shard_root"),
        "protocol": candidate.get("protocol"),
        "verification": candidate.get("verification"),
        "interpretation": candidate.get("interpretation"),
        "comparison": comparison,
    }
    Path(args.output_json).write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    lines = [f"| task | {candidate_name} test R2 | best fold R2 | gap (mean-best) |",
             "|---|---|---|---|"]
    for name, body in sorted(candidate_tasks.items()):
        lines.append("| {} | {:.6f} | {:.6f} | {:+.6f} |".format(
            name, body["test_r2_mean"], body["test_r2_best"], body["gap_mean_minus_best"]))
    lines.append(f"| **macro{task_count}** | **{candidate_macro:.6f}** | | |")
    lines.append("")
    lines.append(f"pooled out-of-fold macro{task_count}: {candidate_macro_pooled:.6f}")
    lines.append("")
    lines.append("| task | " + " | ".join(f"{name} mean R2" for name, _ in references)
                 + f" | {candidate_name} mean R2 | delta vs {references[0][0]} |")
    lines.append("|---" * (len(references) + 3) + "|")
    first_name, _ = references[0]
    for task in sorted(candidate_tasks):
        cells = [f"{comparison['references'][name]['tasks'][task]['reference_test_r2_mean']:.6f}"
                 for name, _ in references]
        delta = comparison["references"][first_name]["tasks"][task]["delta_test_r2_mean"]
        lines.append(f"| {task} | " + " | ".join(cells)
                     + f" | {candidate_tasks[task]['test_r2_mean']:.6f} | {delta:+.6f} |")
    macro_cells = " | ".join(
        f"{comparison['references'][name]['reference_macro_r2']:.6f}" for name, _ in references)
    lines.append(f"| **macro{task_count}** | {macro_cells} | **{candidate_macro:.6f}** | "
                 f"{comparison['references'][first_name]['delta_macro_r2']:+.6f} |")
    Path(args.output_markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
