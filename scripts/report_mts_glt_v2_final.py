#!/usr/bin/env python3
"""Create the final evidence report for the staged MTS-GLT-v2 protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/mts_glt_v2"
TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sidecar_coverage(cohort: str) -> float:
    path = (
        ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_v1"
        / cohort / "graph_geometry_valid.npy"
    )
    values = np.load(path, mmap_mode="r")
    return float(np.count_nonzero(values) / values.size)


def _training_resources(run_name: str) -> dict:
    path = RESULT_ROOT / run_name / "training_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    steady = rows[20:] if len(rows) > 20 else rows
    pools = np.asarray([row["mean_infonce_pool_size"] for row in rows], dtype=float)
    return {
        "steps": len(rows),
        "optimizer_batch": int(rows[-1]["optimizer_batch"]),
        "samples_per_second_mean_after_20": float(np.mean([
            row["samples_per_second"] for row in steady
        ])),
        "optimizer_steps_per_second_mean_after_20": float(np.mean([
            1.0 / row["step_seconds"] for row in steady
        ])),
        "peak_memory_bytes": int(max(row["peak_memory_bytes"] for row in rows)),
        "wall_hours_from_step_times": float(sum(
            row["step_seconds"] for row in rows
        ) / 3600.0),
        "infonce_pool": {
            "mean": float(pools.mean()),
            "min": float(pools.min()),
            "max": float(pools.max()),
            "p05": float(np.quantile(pools, 0.05)),
            "p95": float(np.quantile(pools, 0.95)),
        },
    }


def build() -> dict:
    architecture = _read(RESULT_ROOT / "architecture_screen/selection_5k.json")
    infonce = _read(RESULT_ROOT / "infonce_screen/selection.json")
    compact = _read(RESULT_ROOT / "compact19_probe/selected_5k/report.json")
    decision = _read(RESULT_ROOT / "screening_decision.json")
    payload = {
        "schema": "mts-glt-v2-final-report-v1",
        "seed": 42,
        "evaluation_protocol": "historical_shared5",
        "independent_blind_test": False,
        "architecture_selection": architecture,
        "infonce_selection": infonce,
        "compact19_probe": compact,
        "screening_decision": decision,
        "sidecar_geometry_valid_fraction": {
            "PI1M_v2": _sidecar_coverage("PI1M_v2"),
            "downstream_union": _sidecar_coverage("downstream_union"),
        },
    }
    if not decision["go_20k"]:
        payload.update({
            "status": "stopped_by_screening_gate",
            "formal_evaluation": None,
            "promoted": False,
        })
        return payload

    trajectory = _read(RESULT_ROOT / "formal/trajectory_selection.json")
    selected = trajectory["selected"]
    paired_path = (
        RESULT_ROOT / "downstream" / selected["run_name"] / "paired_summary.json"
    )
    paired = _read(paired_path)
    fused_mode = (
        "o8_glt_atom_desc" if decision["compact19_enabled"] else "o8_glt_atom"
    )
    task_rows = {}
    for task in TASKS:
        row = paired["tasks"][task]
        o8 = np.asarray(row["fold_o8_r2"], dtype=float)
        fused = np.asarray(row["fold_fused_r2"], dtype=float)
        task_rows[task] = {
            "o8_mean_r2": float(o8.mean()),
            "o8_sample_std_r2": float(o8.std(ddof=1)),
            "fused_mean_r2": float(fused.mean()),
            "fused_sample_std_r2": float(fused.std(ddof=1)),
            "fused_minus_o8": float((fused - o8).mean()),
            "improved_folds": int(np.count_nonzero(fused > o8)),
            "fold_o8_r2": o8.tolist(),
            "fold_fused_r2": fused.tolist(),
        }
    promoted = bool(
        paired["macro_delta"] > 0
        and paired["median_task_delta"] > 0
        and paired["positive_tasks"] >= 5
    )
    result_path = RESULT_ROOT / "downstream" / selected["run_name"]
    artifact_counts = {}
    for mode in ("o8_only", fused_mode):
        artifact_counts[mode] = {
            "shards": len(list((result_path / mode / "shards").rglob("fold_*.csv"))),
            "predictions": len(list(
                (result_path / mode / "predictions").rglob("fold_*.npz")
            )),
        }
    payload.update({
        "status": "formal_evaluation_complete",
        "trajectory_selection": trajectory,
        "selected_checkpoint": selected["checkpoint"],
        "fused_mode": fused_mode,
        "formal_evaluation": {
            "macro_o8": float(paired["macro_o8"]),
            "macro_fused": float(paired["macro_fused"]),
            "macro_delta": float(paired["macro_delta"]),
            "median_task_delta": float(paired["median_task_delta"]),
            "positive_tasks": int(paired["positive_tasks"]),
            "tasks": task_rows,
            "artifact_counts": artifact_counts,
        },
        "training_resources": _training_resources(decision["formal_run_name"]),
        "promoted": promoted,
        "production_default_modified": False,
    })
    return payload


def render(payload: dict) -> str:
    lines = [
        "# MTS-GLT-v2 分阶段实验最终报告",
        "",
        "> 口径：seed 42、historical_shared5；不是独立盲测。",
        "",
        "## 阶段选择",
        "",
        f"- 5k 架构：`{payload['architecture_selection']['ranked'][0]['name']}`",
        f"- InfoNCE 权重：`{payload['screening_decision']['selected_weight']}`",
        f"- Compact19 冻结准入：`{payload['screening_decision']['compact19_admitted']}`",
        f"- Compact19 正式启用：`{payload['screening_decision']['compact19_enabled']}`",
        f"- 20k 门控：`{payload['screening_decision']['go_20k']}`",
        "",
    ]
    if payload["formal_evaluation"] is None:
        lines.extend([
            "## 结论",
            "",
            "筛选门未通过，协议按计划停止，没有启动正式 20k 或 8×5。",
            "未修改生产默认。",
        ])
        return "\n".join(lines) + "\n"
    formal = payload["formal_evaluation"]
    lines.extend([
        "## 匹配 8×5 结果",
        "",
        "| Task | O8-only R² | O8+GLT R² | Delta | Improved folds |",
        "|---|---:|---:|---:|---:|",
    ])
    for task in TASKS:
        row = formal["tasks"][task]
        lines.append(
            f"| {task} | {row['o8_mean_r2']:.6f} ± "
            f"{row['o8_sample_std_r2']:.6f} | {row['fused_mean_r2']:.6f} ± "
            f"{row['fused_sample_std_r2']:.6f} | "
            f"{row['fused_minus_o8']:+.6f} | {row['improved_folds']}/5 |"
        )
    lines.extend([
        "",
        f"- O8-only macro8：`{formal['macro_o8']:.6f}`",
        f"- O8+GLT macro8：`{formal['macro_fused']:.6f}`",
        f"- Macro delta：`{formal['macro_delta']:+.6f}`",
        f"- Median task delta：`{formal['median_task_delta']:+.6f}`",
        f"- Positive tasks：`{formal['positive_tasks']}/8`",
        f"- 晋级门通过：`{payload['promoted']}`",
        "- 执行者未修改生产默认。",
        "",
        "## 资源与覆盖率",
        "",
        f"- PI1M_v2 GLT-valid：`{payload['sidecar_geometry_valid_fraction']['PI1M_v2']:.4%}`",
        f"- Downstream GLT-valid：`{payload['sidecar_geometry_valid_fraction']['downstream_union']:.4%}`",
        f"- 正式训练平均吞吐（step 20后）：`{payload['training_resources']['samples_per_second_mean_after_20']:.2f} samples/s`",
        f"- 峰值显存：`{payload['training_resources']['peak_memory_bytes'] / 2**30:.2f} GiB/rank`",
        f"- 平均 InfoNCE pool：`{payload['training_resources']['infonce_pool']['mean']:.2f}`；optimizer batch `1008`。",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    payload = build()
    json_path = RESULT_ROOT / "final_report.json"
    md_path = RESULT_ROOT / "final_report.md"
    for path, content in (
        (json_path, json.dumps(payload, indent=2, sort_keys=True) + "\n"),
        (md_path, render(payload)),
    ):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    print(json.dumps({
        "status": payload["status"],
        "promoted": payload["promoted"],
        "report": str(md_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
