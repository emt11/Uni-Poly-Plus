#!/usr/bin/env python3
"""Summarise the matched MTS-GLT-v1 downstream evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np


TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
MODES = ("o8_only", "o8_glt")


def _read_fold(root: Path, mode: str, task: str, fold: int) -> float:
    path = root / mode / "shards" / "42" / task / f"fold_{fold}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"expected one row in {path}, got {len(rows)}")
    value = float(rows[0]["avg_test_r2"])
    if not np.isfinite(value):
        raise RuntimeError(f"non-finite R2 in {path}")
    prediction = root / mode / "predictions" / "42" / task / f"fold_{fold}.npz"
    with np.load(prediction, allow_pickle=False) as arrays:
        for name in arrays.files:
            value_array = arrays[name]
            if np.issubdtype(value_array.dtype, np.number) and not np.isfinite(value_array).all():
                raise RuntimeError(f"non-finite {name} in {prediction}")
    return value


def _pool_summary(metrics_path: Path) -> dict:
    pools = []
    optimizer_batches = set()
    with metrics_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            pools.append(float(row["mean_infonce_pool_size"]))
            optimizer_batches.add(int(row["optimizer_batch"]))
    if len(pools) != 20_000 or optimizer_batches != {1008}:
        raise RuntimeError("formal pretraining metrics are incomplete or inconsistent")
    values = np.asarray(pools, dtype=np.float64)
    return {
        "steps": len(pools),
        "optimizer_batch": 1008,
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def build_report(project_root: Path) -> dict:
    downstream = project_root / "results/mts_glt_v1/downstream"
    results = {}
    for task in TASKS:
        task_result = {}
        for mode in MODES:
            folds = [_read_fold(downstream, mode, task, fold) for fold in range(5)]
            task_result[mode] = {
                "fold_r2": folds,
                "mean_r2": statistics.mean(folds),
                "sample_std_r2": statistics.stdev(folds),
            }
        task_result["fused_minus_o8_only"] = (
            task_result["o8_glt"]["mean_r2"]
            - task_result["o8_only"]["mean_r2"]
        )
        task_result["improved_folds"] = sum(
            fused > base for fused, base in zip(
                task_result["o8_glt"]["fold_r2"],
                task_result["o8_only"]["fold_r2"],
            )
        )
        results[task] = task_result
    macro = {
        mode: statistics.mean(results[task][mode]["mean_r2"] for task in TASKS)
        for mode in MODES
    }
    pi_qc = json.loads(
        (project_root / "results/mts_glt_v1/sidecar_PI1M_v2_qc.json").read_text()
    )
    downstream_qc = json.loads(
        (project_root / "results/mts_glt_v1/sidecar_downstream_union_qc.json").read_text()
    )
    return {
        "schema": "mts-glt-v1-formal-report-v1",
        "evaluation_protocol": "historical_shared5",
        "independent_blind_test": False,
        "seed": 42,
        "checkpoint": "pretrained_models/mts_glt_v1/mts_glt_v1_seed42_final.pth",
        "tasks": results,
        "macro8": macro,
        "macro8_fused_minus_o8_only": macro["o8_glt"] - macro["o8_only"],
        "positive_tasks": sum(results[task]["fused_minus_o8_only"] > 0 for task in TASKS),
        "sidecar": {
            "pi1m_v2": pi_qc,
            "downstream_union": downstream_qc,
        },
        "infonce_pool": _pool_summary(
            project_root / "results/mts_glt_v1/pretrain/training_metrics.jsonl"
        ),
        "artifact_counts": {
            mode: {"shards": 40, "predictions": 40} for mode in MODES
        },
    }


def _markdown(report: dict) -> str:
    lines = [
        "# MTS-GLT-v1 正式匹配实验报告",
        "",
        "> 口径：seed 42、historical_shared5；validation/test fold 共享，非独立盲测。",
        "",
        "| Task | O8-only R² | O8+GLT R² | Delta | Improved folds |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        row = report["tasks"][task]
        lines.append(
            f"| {task} | {row['o8_only']['mean_r2']:.6f} ± "
            f"{row['o8_only']['sample_std_r2']:.6f} | "
            f"{row['o8_glt']['mean_r2']:.6f} ± "
            f"{row['o8_glt']['sample_std_r2']:.6f} | "
            f"{row['fused_minus_o8_only']:+.6f} | {row['improved_folds']}/5 |"
        )
    lines += [
        "",
        f"- O8-only macro8: `{report['macro8']['o8_only']:.6f}`",
        f"- O8+GLT macro8: `{report['macro8']['o8_glt']:.6f}`",
        f"- Macro delta: `{report['macro8_fused_minus_o8_only']:+.6f}`",
        f"- Positive tasks: `{report['positive_tasks']}/8`",
        f"- PI1M_v2 GLT-valid coverage: "
        f"`{report['sidecar']['pi1m_v2']['geometry_valid_fraction']:.4%}`",
        f"- Downstream-union GLT-valid coverage: "
        f"`{report['sidecar']['downstream_union']['geometry_valid_fraction']:.4%}`",
        f"- Mean InfoNCE pool: `{report['infonce_pool']['mean']:.2f}` "
        f"(optimizer batch `{report['infonce_pool']['optimizer_batch']}`)",
        "- 两种模式均为 40/40 shard 与 40/40 finite prediction。",
        "- 本报告不自动修改生产默认。",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", default="results/mts_glt_v1")
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    output_root = project_root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    report = build_report(project_root)
    (output_root / "final_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "final_report.md").write_text(_markdown(report), encoding="utf-8")
    print(output_root / "final_report.md")


if __name__ == "__main__":
    main()
