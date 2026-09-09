#!/usr/bin/env python3
"""Aggregate the completed N+1/N+2 historical_shared5 campaign."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
FOLDS = tuple(range(5))


def _fold(version, task, fold):
    root = ROOT / "results/mts_glt_v2_distill" / version / "downstream"
    shard = root / "shards/42" / task / f"fold_{fold}.csv"
    prediction = root / "predictions/42" / task / f"fold_{fold}.npz"
    checkpoint = root / "shards/42" / task / f"fold_{fold}_best.pt"
    if not (shard.is_file() and prediction.is_file() and checkpoint.is_file()):
        raise FileNotFoundError(f"incomplete downstream unit: {version}/{task}/{fold}")
    frame = pd.read_csv(shard)
    if len(frame) != 1:
        raise RuntimeError(f"invalid shard: {shard}")
    metrics = json.loads(frame.iloc[0]["per_fold_metrics"])
    if len(metrics) != 1 or int(metrics[0]["fold"]) != fold:
        raise RuntimeError(f"invalid fold metadata: {shard}")
    row = metrics[0]
    values = {name: float(row[f"test_{name}"]) for name in ("r2", "mae", "rmse")}
    if not np.isfinite(list(values.values())).all():
        raise RuntimeError(f"non-finite metrics: {shard}")
    with np.load(prediction, allow_pickle=False) as payload:
        metadata = json.loads(str(np.asarray(payload["metadata"]).item()))
        y_true, y_pred = np.asarray(payload["y_true"]), np.asarray(payload["y_pred"])
        if (
            metadata.get("task") != task or int(metadata.get("fold", -1)) != fold
            or int(metadata.get("seed", -1)) != 42 or y_true.shape != y_pred.shape
            or not np.isfinite(y_true).all() or not np.isfinite(y_pred).all()
        ):
            raise RuntimeError(f"invalid prediction artifact: {prediction}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        saved.get("schema") != "mts-glt-distill-finetune-v1"
        or saved.get("task") != task or int(saved.get("fold", -1)) != fold
        or int(saved.get("seed", -1)) != 42
    ):
        raise RuntimeError(f"invalid finetuned checkpoint: {checkpoint}")
    return {
        **values,
        "best_epoch": int(row["best_epoch"]),
        "epochs_ran": int(len(row["training_epoch_timing"])),
        "training_steps": int(row["training_steps"]),
        "fold_wall_seconds": float(row["fold_wall_seconds"]),
    }


def _pretraining(version, stage, expected_steps):
    root = ROOT / "results/mts_glt_v2_distill" / version / stage
    rows = [json.loads(line) for line in (root / "training_metrics.jsonl").read_text().splitlines()]
    if len(rows) != expected_steps or int(rows[-1]["step"]) != expected_steps:
        raise RuntimeError(f"incomplete pretraining metrics: {version}/{stage}")
    if not all(
        math.isfinite(float(value))
        for row in rows for value in row.values() if isinstance(value, (int, float))
    ):
        raise RuntimeError(f"non-finite pretraining metrics: {version}/{stage}")
    return {
        "optimizer_steps": expected_steps,
        "first_loss": float(rows[0]["loss"]),
        "final_loss": float(rows[-1]["loss"]),
        "wall_hours_from_step_timing": float(sum(row["step_seconds"] for row in rows) / 3600.0),
        "three_gpu_hours_from_step_timing": float(3 * sum(row["step_seconds"] for row in rows) / 3600.0),
        "rank0_peak_memory_bytes": int(max(row["peak_memory_bytes"] for row in rows)),
    }


def _deployment(version):
    path = ROOT / "results/mts_glt_v2_distill" / version / "student/student_deploy_020k.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    keys = tuple(payload.get("state_dict", ()))
    forbidden = ("teacher", "atom_head", "line_projection", "glt")
    if (
        payload.get("schema") != "mts-glt-distill-student-deploy-v1"
        or payload.get("version") != version or int(payload.get("step", -1)) != 20000
        or any(any(token in key for token in forbidden) for key in keys)
    ):
        raise RuntimeError(f"invalid O8+MD200 deployment bundle: {path}")
    return {"path": str(path.relative_to(ROOT)), "tensor_count": len(keys), "contains_only_o8_md200": True}


def main():
    rows = []
    for version in ("n_plus_2", "n_plus_1"):
        for task in TASKS:
            for fold in FOLDS:
                rows.append({"version": version, "task": task, "fold": fold, **_fold(version, task, fold)})
    frame = pd.DataFrame(rows)
    tasks = {}
    for task in TASKS:
        tasks[task] = {}
        for version in ("n_plus_2", "n_plus_1"):
            part = frame[(frame.version == version) & (frame.task == task)]
            tasks[task][version] = {
                f"{metric}_{stat}": float(getattr(part[metric], stat)(ddof=1))
                if stat == "std" else float(getattr(part[metric], stat)())
                for metric in ("r2", "mae", "rmse") for stat in ("mean", "std")
            }
        paired = frame[(frame.version == "n_plus_2") & (frame.task == task)].sort_values("fold")
        control = frame[(frame.version == "n_plus_1") & (frame.task == task)].sort_values("fold")
        delta = paired.r2.to_numpy() - control.r2.to_numpy()
        tasks[task]["n_plus_2_minus_n_plus_1"] = {
            "r2_by_fold": delta.tolist(), "mean_r2": float(delta.mean()),
            "positive_folds": int((delta > 0).sum()),
        }
    macro = {
        version: float(np.mean([tasks[task][version]["r2_mean"] for task in TASKS]))
        for version in ("n_plus_2", "n_plus_1")
    }
    all_delta = []
    positive_tasks = 0
    for task in TASKS:
        values = tasks[task]["n_plus_2_minus_n_plus_1"]["r2_by_fold"]
        all_delta.extend(values)
        positive_tasks += int(tasks[task]["n_plus_2_minus_n_plus_1"]["mean_r2"] > 0)
    pretraining = {
        version: {
            "teacher": _pretraining(version, "teacher", 5000),
            "student": _pretraining(version, "student", 20000),
        }
        for version in ("n_plus_2", "n_plus_1")
    }
    downstream = {
        version: {
            "completed_units": int((frame.version == version).sum()),
            "training_steps": int(frame.loc[frame.version == version, "training_steps"].sum()),
            "one_gpu_hours": float(frame.loc[frame.version == version, "fold_wall_seconds"].sum() / 3600.0),
            "minimum_epochs_ran": int(frame.loc[frame.version == version, "epochs_ran"].min()),
            "maximum_epochs_ran": int(frame.loc[frame.version == version, "epochs_ran"].max()),
            "maximum_best_epoch": int(frame.loc[frame.version == version, "best_epoch"].max()),
        }
        for version in ("n_plus_2", "n_plus_1")
    }
    sidecars = {
        version: json.loads((ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_distill_v1" / version / "metadata.json").read_text())
        for version in ("n_plus_2", "n_plus_1")
    }
    deployments = {version: _deployment(version) for version in ("n_plus_2", "n_plus_1")}
    scheduler = {
        version: json.loads((ROOT / "logs/mts_glt_v2_distill" / version / "downstream/scheduler_report.json").read_text())
        for version in ("n_plus_2", "n_plus_1")
    }
    if any(report.get("pending") != 0 or len(report.get("completed", ())) != 40 for report in scheduler.values()):
        raise RuntimeError("downstream scheduler reports are incomplete")
    historical_path = ROOT / "results/mts_glt_v2/downstream/formal/a6_h_w1_20k_probe_005k/paired_summary.json"
    historical = json.loads(historical_path.read_text()) if historical_path.is_file() else None
    payload = {
        "schema": "mts-glt-distill-comparison-v1",
        "evaluation_protocol": "historical_shared5",
        "independent_blind_test": False,
        "completed_units": int(len(frame)), "tasks": tasks, "macro8_r2": macro,
        "macro8_r2_n_plus_2_minus_n_plus_1": macro["n_plus_2"] - macro["n_plus_1"],
        "positive_tasks_n_plus_2_minus_n_plus_1": positive_tasks,
        "positive_folds_n_plus_2_minus_n_plus_1": int((np.asarray(all_delta) > 0).sum()),
        "pretraining": pretraining,
        "downstream": downstream,
        "sidecars": sidecars,
        "deployments": deployments,
        "scheduler_reports": scheduler,
        "historical_glt_v2": historical,
        "attribution_limits": [
            "N+1/N+2 simultaneously change boundary-state sharing and length treatment.",
            "The new route also changes O8, pretraining, MD fusion, and downstream input relative to GLT-v2.",
            "No no-distillation control was trained, so distillation has no isolated causal estimate.",
            "historical_shared5 is a shared development-fold evaluation, not an independent blind test.",
        ],
    }
    output = ROOT / "results/mts_glt_v2_distill/comparison"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "all_fold_metrics.csv", index=False)
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# N+1 / N+2 正式结果", "",
        "协议：`historical_shared5`；validation 与 test 共用 outer fold，不是独立盲测。", "",
        "| Task | N+2 R² mean±std | N+1 R² mean±std | N+2−N+1 |",
        "|---|---:|---:|---:|",
    ]
    for task in TASKS:
        two, one = tasks[task]["n_plus_2"], tasks[task]["n_plus_1"]
        delta = tasks[task]["n_plus_2_minus_n_plus_1"]["mean_r2"]
        lines.append(
            f"| {task} | {two['r2_mean']:.6f} ± {two['r2_std']:.6f} | "
            f"{one['r2_mean']:.6f} ± {one['r2_std']:.6f} | {delta:+.6f} |"
        )
    lines.extend([
        "", f"- N+2 macro8 R²：`{macro['n_plus_2']:.10f}`",
        f"- N+1 macro8 R²：`{macro['n_plus_1']:.10f}`",
        f"- 配对 macro 差值 N+2−N+1：`{macro['n_plus_2'] - macro['n_plus_1']:+.10f}`",
        f"- 正增量任务：`{positive_tasks}/8`；正增量 fold：`{int((np.asarray(all_delta) > 0).sum())}/40`",
        "", "两个版本均完成教师 5k、学生 20k 和 40/40 微调。部署包只含 O8 与原子条件 MD200；下游不实例化 GLT，不读取坐标或 line sidecar。",
        "",
        f"预训练 step-time 折算：N+2 教师/学生 `{pretraining['n_plus_2']['teacher']['wall_hours_from_step_timing']:.3f}/{pretraining['n_plus_2']['student']['wall_hours_from_step_timing']:.3f}` 小时；N+1 教师/学生 `{pretraining['n_plus_1']['teacher']['wall_hours_from_step_timing']:.3f}/{pretraining['n_plus_1']['student']['wall_hours_from_step_timing']:.3f}` 小时。",
        f"下游累计单卡时间：N+2 `{downstream['n_plus_2']['one_gpu_hours']:.3f}` GPU-hours，N+1 `{downstream['n_plus_1']['one_gpu_hours']:.3f}` GPU-hours。",
        "", "N+1/N+2 同时改变边界状态共享与跨 RU 长度处理；与旧 GLT-v2 的差异还混合了 O8、预训练、MD 融合与下游输入变化，因此均只作整体方案的描述性比较。",
    ])
    (output / "final_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"completed_units": len(frame), "macro8_r2": macro}, sort_keys=True))


if __name__ == "__main__":
    main()
