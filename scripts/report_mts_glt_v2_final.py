#!/usr/bin/env python3
"""Render the retained MTS-GLT-v2-Base-5k evidence report."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/mts/glt_v2_base_5k_v1.json"
MANIFEST = ROOT / "results/mts_glt_v2/base_5k_v1/baseline_manifest.json"
PAIRED = ROOT / "results/mts_glt_v2/downstream/formal/a6_h_w1_20k_probe_005k/paired_summary.json"
RESULT_ROOT = ROOT / "results/mts_glt_v2"
TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def build() -> dict:
    config = _read(CONFIG)
    manifest = _read(MANIFEST)
    paired = _read(PAIRED)
    task_rows = {}
    for task in TASKS:
        row = paired["tasks"][task]
        o8 = [float(value) for value in row["fold_o8_r2"]]
        fused = [float(value) for value in row["fold_fused_r2"]]
        task_rows[task] = {
            "o8_mean_r2": float(sum(o8) / len(o8)),
            "o8_sample_std_r2": _sample_std(o8),
            "fused_mean_r2": float(sum(fused) / len(fused)),
            "fused_sample_std_r2": _sample_std(fused),
            "fused_minus_o8": float(sum(f - o for f, o in zip(fused, o8)) / len(o8)),
            "improved_folds": int(sum(f > o for f, o in zip(fused, o8))),
            "fold_o8_r2": o8,
            "fold_fused_r2": fused,
        }
    return {
        "schema": "mts-glt-v2-baseline-report-v1",
        "status": "formal_evaluation_complete",
        "baseline": manifest["baseline_name"],
        "baseline_version": manifest["baseline_version"],
        "seed": int(manifest["seed"]),
        "evaluation_protocol": manifest["evaluation_protocol"],
        "independent_blind_test": bool(manifest["independent_blind_test"]),
        "selected_checkpoint": manifest["pretrained_checkpoint"],
        "pretraining": {
            "dataset": config["pretraining"]["dataset"],
            "checkpoint_step": int(config["checkpoint_step"]),
            "trajectory_end_step": int(config["pretraining_trajectory_end_step"]),
            "objectives": [name for name, value in config["pretraining"]["objectives"].items() if value["enabled"]],
            "global_batch_size": int(config["pretraining"]["global_batch_size"]),
            "precision": config["pretraining"]["precision"],
            "learning_rate": float(config["pretraining"]["learning_rate"]),
            "warmup_optimizer_steps": int(config["pretraining"]["warmup_optimizer_steps"]),
        },
        "architecture": config["model"],
        "downstream": config["downstream"],
        "formal_evaluation": {
            "run_name": paired["run_name"],
            "macro_fused": float(paired["macro_fused"]),
            "macro_o8": float(paired["macro_o8"]),
            "descriptive_fused_increment": float(paired["macro_delta"]),
            "median_task_increment": float(paired["median_task_delta"]),
            "positive_fused_tasks": int(paired["positive_tasks"]),
            "tasks": task_rows,
            "independent_blind_test": bool(manifest["independent_blind_test"]),
        },
        "artifacts": {
            "baseline_manifest": str(MANIFEST.relative_to(ROOT)),
            "baseline_config": str(CONFIG.relative_to(ROOT)),
            "paired_downstream_summary": str(PAIRED.relative_to(ROOT)),
        },
    }


def render(payload: dict) -> str:
    formal = payload["formal_evaluation"]
    lines = [
        "# MTS-GLT-v2-Base-5k 基线报告",
        "",
        "> seed 42、historical_shared5；validation 与 test 共用 fold，不是独立盲测。",
        "",
        "## 配置摘要",
        "",
        f"- checkpoint：`{payload['selected_checkpoint']}`",
        f"- 预训练数据：`{payload['pretraining']['dataset']}`；选定 step：`{payload['pretraining']['checkpoint_step']}`；轨迹终点：`{payload['pretraining']['trajectory_end_step']}`",
        f"- 预训练目标：`{', '.join(payload['pretraining']['objectives'])}`",
        f"- 下游协议：`{payload['evaluation_protocol']}`；schedule：`{payload['downstream']['schedule']}`",
        "",
        "## 成对下游结果",
        "",
        "| Task | O8-only R² | O8+GLT R² | Delta | Improved folds |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        row = formal["tasks"][task]
        lines.append(
            f"| {task} | {row['o8_mean_r2']:.6f} ± {row['o8_sample_std_r2']:.6f} | "
            f"{row['fused_mean_r2']:.6f} ± {row['fused_sample_std_r2']:.6f} | "
            f"{row['fused_minus_o8']:+.6f} | {row['improved_folds']}/5 |"
        )
    lines.extend([
        "",
        f"- O8-only macro8：`{formal['macro_o8']:.6f}`",
        f"- O8+GLT macro8：`{formal['macro_fused']:.6f}`",
        f"- 描述性 delta：`{formal['descriptive_fused_increment']:+.6f}`",
        f"- 正向任务：`{formal['positive_fused_tasks']}/8`",
        "",
        "## 口径",
        "",
        "该表是同一 checkpoint、split 和 seed 的 matched 描述性比较；不把差值解释为因果 interaction，也不构成独立盲测。",
        "",
    ])
    return "\n".join(lines)


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
    print(json.dumps({"status": payload["status"], "report": str(md_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
