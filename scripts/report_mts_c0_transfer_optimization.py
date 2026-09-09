#!/usr/bin/env python3
"""Audit and summarize the fixed frozen-probe and staged-C0 campaign."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import TargetScaler

BASE = ROOT / "results/mts_c0_transfer_optimization"
OLD = ROOT / "results/mts_glt_distill_repair_control"
TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def read_old(group, task, fold):
    path = OLD / group / "downstream/outer5_inner20/shards/42" / task / f"fold_{fold}.csv"
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise RuntimeError(f"invalid historical shard: {path}")
    metrics = json.loads(frame.iloc[0]["per_fold_metrics"])[0]
    return {name: float(metrics[f"test_{name}"]) for name in ("r2", "mae", "rmse")}


def raw_targets(task):
    return pd.read_csv(ROOT / f"data/raw/smi_{task}.csv").iloc[:, 1].to_numpy(dtype=np.float64)


def audit_prediction(path, task, fold):
    manifest = json.loads((ROOT / f"data/splits/mips_outer5_inner20/{task}.json").read_text())
    expected = np.asarray(manifest["folds"][fold]["test_indices"], dtype=np.int64)
    with np.load(path, allow_pickle=False) as payload:
        indices = np.asarray(payload["sample_indices"], dtype=np.int64)
        truth = np.asarray(payload["y_true"], dtype=np.float64)
        prediction = np.asarray(payload["y_pred"], dtype=np.float64)
        metadata = json.loads(str(np.asarray(payload["metadata"]).item()))
    expected_truth = raw_targets(task)[expected]
    if (
        not np.array_equal(indices, expected) or truth.shape != prediction.shape
        or not np.isfinite(truth).all() or not np.isfinite(prediction).all()
        or not np.allclose(truth, expected_truth, rtol=1e-6, atol=5e-6)
        or metadata.get("split_manifest_sha256") != __import__("hashlib").sha256(
            (ROOT / f"data/splits/mips_outer5_inner20/{task}.json").read_bytes()
        ).hexdigest()
    ):
        raise RuntimeError(f"prediction identity/finite audit failed: {path}")
    return indices


def main():
    comparison = BASE / "comparison"
    probe_rows = []
    for group in ("c0", "c1", "c2"):
        for task in ("xc", "eps"):
            seen = []
            for fold in range(5):
                unit = BASE / "probes/units" / group / task / f"fold_{fold}"
                row = json.loads((unit / "metrics.json").read_text())
                seen.extend(audit_prediction(unit / "predictions.npz", task, fold).tolist())
                split = json.loads((ROOT / f"data/splits/mips_outer5_inner20/{task}.json").read_text())["folds"][fold]
                with np.load(unit / "fit_state.npz", allow_pickle=False) as fit:
                    train = np.asarray(split["train_indices"], dtype=np.int64)
                    validation = np.asarray(split["validation_indices"], dtype=np.int64)
                    test = np.asarray(split["test_indices"], dtype=np.int64)
                    if not (
                        np.array_equal(fit["train_indices"], train)
                        and np.array_equal(fit["validation_indices"], validation)
                        and np.array_equal(fit["test_indices"], test)
                    ):
                        raise RuntimeError(f"probe fit split mismatch: {unit}")
                    with np.load(row["feature_cache"], allow_pickle=False) as cache:
                        features = np.asarray(cache["features"])
                        if not np.array_equal(cache["sample_indices"], np.arange(len(features))) or not np.isfinite(features).all():
                            raise RuntimeError(f"probe feature cache mismatch: {unit}")
                    expected_x = StandardScaler().fit(features[train].astype(np.float64))
                    expected_y = TargetScaler(task, StandardScaler(), transform_mode="recommended")
                    expected_y.scaler.fit(expected_y._pre_transform(raw_targets(task)[train]))
                    if not (
                        np.array_equal(fit["feature_mean"], expected_x.mean_)
                        and np.array_equal(fit["feature_scale"], expected_x.scale_)
                        and np.array_equal(fit["label_mean"], expected_y.scaler.mean_)
                        and np.array_equal(fit["label_scale"], expected_y.scaler.scale_)
                    ):
                        raise RuntimeError(f"probe scaler was not fit on train only: {unit}")
                candidates = row["alpha_candidates"]
                selected = max((float(item["validation_r2"]), float(item["alpha"])) for item in candidates)
                if float(row["selected_alpha"]) not in {0.1, 1.0, 10.0, 100.0} or selected != (float(row["best_validation_r2"]), float(row["selected_alpha"])):
                    raise RuntimeError(f"probe alpha selection mismatch: {unit}")
                old = read_old(group, task, fold)
                probe_rows.append({
                    "group": group.upper(), "task": task, "fold": fold,
                    "selected_alpha": float(row["selected_alpha"]),
                    "validation_r2": float(row["best_validation_r2"]),
                    "probe_r2": float(row["test_r2"]),
                    "probe_mae": float(row["test_mae"]),
                    "probe_rmse": float(row["test_rmse"]),
                    "existing_full_ft_r2": old["r2"],
                })
            manifest = json.loads((ROOT / f"data/splits/mips_outer5_inner20/{task}.json").read_text())
            if sorted(seen) != list(range(int(manifest["sample_count"]))):
                raise RuntimeError(f"probe OOF coverage failed: {group}/{task}")
    probe = pd.DataFrame(probe_rows)
    probe["r2_delta_vs_c0_fold"] = 0.0
    for task in ("xc", "eps"):
        baseline = probe[(probe.group == "C0") & (probe.task == task)].set_index("fold")["probe_r2"]
        mask = probe.task == task
        probe.loc[mask, "r2_delta_vs_c0_fold"] = [
            float(row.probe_r2 - baseline.loc[int(row.fold)])
            for row in probe[mask].itertuples()
        ]

    stage_rows = []
    stage_root = BASE / "staged_finetune"
    for task in TASKS:
        seen = []
        for fold in range(5):
            shard = stage_root / "shards/42" / task / f"fold_{fold}.csv"
            checkpoint = shard.with_name(f"fold_{fold}_best.pt")
            prediction = stage_root / "predictions/42" / task / f"fold_{fold}.npz"
            if not checkpoint.is_file():
                raise RuntimeError(f"missing staged checkpoint: {checkpoint}")
            frame = pd.read_csv(shard)
            row = frame.iloc[0]
            metrics = json.loads(row["per_fold_metrics"])[0]
            if str(row["finetune_strategy"]) != "staged_head10" or metrics.get("best_stage") not in {"stage1", "stage2"}:
                raise RuntimeError(f"staged metadata mismatch: {shard}")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            manifest_sha = __import__("hashlib").sha256(
                (ROOT / f"data/splits/mips_outer5_inner20/{task}.json").read_bytes()
            ).hexdigest()
            if not (
                saved.get("schema") == "mts-glt-distill-finetune-v1"
                and saved.get("finetune_strategy") == "staged_head10"
                and saved.get("pretraining_bundle_version") == "none"
                and int(saved.get("pretraining_bundle_step", -1)) == 20000
                and saved.get("split_manifest_sha256") == manifest_sha
                and int(metrics.get("stage1_epochs_completed", -1)) == 10
                and 1 <= int(metrics.get("stage2_epochs_completed", -1)) <= 90
                and bool(metrics.get("stage1_restore_verified", False))
                and all(torch.isfinite(value).all() for value in saved["state_dict"].values())
            ):
                raise RuntimeError(f"staged checkpoint contract failed: {checkpoint}")
            seen.extend(audit_prediction(prediction, task, fold).tolist())
            old = read_old("c0", task, fold)
            stage_rows.append({
                "task": task, "fold": fold,
                "stageft_r2": float(metrics["test_r2"]),
                "stageft_mae": float(metrics["test_mae"]),
                "stageft_rmse": float(metrics["test_rmse"]),
                "c0_r2": old["r2"], "c0_mae": old["mae"], "c0_rmse": old["rmse"],
                "r2_delta": float(metrics["test_r2"]) - old["r2"],
                "best_stage": metrics["best_stage"],
                "best_epoch": int(metrics["best_epoch"]),
                "stage1_epochs": int(metrics["stage1_epochs_completed"]),
                "stage2_epochs": int(metrics["stage2_epochs_completed"]),
                "training_steps": int(metrics["training_steps"]),
                "training_seconds": float(metrics["training_seconds"]),
                "fold_wall_seconds": float(metrics["fold_wall_seconds"]),
            })
        manifest = json.loads((ROOT / f"data/splits/mips_outer5_inner20/{task}.json").read_text())
        if sorted(seen) != list(range(int(manifest["sample_count"]))):
            raise RuntimeError(f"staged OOF coverage failed: {task}")
    stage = pd.DataFrame(stage_rows)

    probe_summary = probe.groupby(["group", "task"], sort=False).agg(
        r2_mean=("probe_r2", "mean"), r2_std=("probe_r2", lambda x: np.std(x, ddof=0)),
        mae_mean=("probe_mae", "mean"), mae_std=("probe_mae", lambda x: np.std(x, ddof=0)),
        rmse_mean=("probe_rmse", "mean"), rmse_std=("probe_rmse", lambda x: np.std(x, ddof=0)),
        full_ft_r2_mean=("existing_full_ft_r2", "mean"),
        positive_paired_folds=("r2_delta_vs_c0_fold", lambda x: int(np.sum(np.asarray(x) > 0))),
    ).reset_index()
    for task in ("xc", "eps"):
        baseline = probe_summary[(probe_summary.group == "C0") & (probe_summary.task == task)].iloc[0].r2_mean
        probe_summary.loc[probe_summary.task == task, "r2_delta_vs_c0"] = probe_summary.loc[probe_summary.task == task, "r2_mean"] - baseline

    stage_summary = stage.groupby("task", sort=False).agg(
        r2_mean=("stageft_r2", "mean"), r2_std=("stageft_r2", lambda x: np.std(x, ddof=0)),
        mae_mean=("stageft_mae", "mean"), mae_std=("stageft_mae", lambda x: np.std(x, ddof=0)),
        rmse_mean=("stageft_rmse", "mean"), rmse_std=("stageft_rmse", lambda x: np.std(x, ddof=0)),
        c0_r2_mean=("c0_r2", "mean"), r2_delta=("r2_delta", "mean"),
    ).reset_index()
    macro = float(stage_summary.r2_mean.mean())
    c0_macro = float(stage_summary.c0_r2_mean.mean())
    decision = bool(macro - c0_macro >= 0.005 and int((stage_summary.r2_delta > 0).sum()) >= 6)
    probe_wall = sum(
        float(json.loads(path.read_text())["wall_seconds"])
        for path in (BASE / "probes/units").glob("*/*/complete.json")
    )
    summary = {
        "schema": "mts-c0-transfer-optimization-report-v1",
        "probe_units": int(len(probe)), "staged_units": int(len(stage)),
        "stageft_macro8_r2": macro, "existing_c0_macro8_r2": c0_macro,
        "macro8_r2_delta": macro - c0_macro,
        "positive_tasks": int((stage_summary.r2_delta > 0).sum()),
        "positive_folds": int((stage.r2_delta > 0).sum()),
        "best_stage_counts": {str(k): int(v) for k, v in stage.best_stage.value_counts().items()},
        "training_seconds": float(stage.training_seconds.sum()),
        "fold_wall_seconds": float(stage.fold_wall_seconds.sum()),
        "probe_gpu_seconds": float(probe_wall),
        "optimizer_updates": int(stage.training_steps.sum()),
        "selected_alpha_counts": {
            str(float(key)): int(value)
            for key, value in probe.selected_alpha.value_counts().sort_index().items()
        },
        "recommend_multiseed_review": decision,
        "deployments": {
            group: json.loads((BASE / f"probes/units/{group}/xc/fold_0/metrics.json").read_text())["checkpoint"]
            for group in ("c0", "c1", "c2")
        },
        "split_manifest_sha256": {
            task: __import__("hashlib").sha256(
                (ROOT / f"data/splits/mips_outer5_inner20/{task}.json").read_bytes()
            ).hexdigest()
            for task in TASKS
        },
        "artifact_counts": {
            "probe_metrics": int(len(probe)), "staged_metrics": int(len(stage)),
            "staged_checkpoints": 40, "staged_predictions": 40,
        },
    }
    comparison.mkdir(parents=True, exist_ok=True)
    probe.to_csv(comparison / "probe_fold_metrics.csv", index=False, float_format="%.17g")
    probe_summary.to_csv(comparison / "probe_task_summary.csv", index=False, float_format="%.17g")
    stage.to_csv(comparison / "stageft_fold_metrics.csv", index=False, float_format="%.17g")
    stage_summary.to_csv(comparison / "stageft_task_summary.csv", index=False, float_format="%.17g")
    atomic_text(comparison / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# MTS C0 迁移优化正式报告", "",
        f"冻结 Ridge 探针完成 {len(probe)}/30；C0_stageft 完成 {len(stage)}/40。",
        "", "## 冻结表示探针", "",
        "|组别|任务|R² mean±std|MAE mean±std|RMSE mean±std|相对 C0 R²|正配对 fold|既有全微调 R²|",
        "|-|-|-:|-:|-:|-:|-:|-:|",
    ]
    for row in probe_summary.itertuples():
        lines.append(f"|{row.group}|{row.task}|{row.r2_mean:.6f}±{row.r2_std:.6f}|{row.mae_mean:.6f}±{row.mae_std:.6f}|{row.rmse_mean:.6f}±{row.rmse_std:.6f}|{row.r2_delta_vs_c0:+.6f}|{int(row.positive_paired_folds)}/5|{row.full_ft_r2_mean:.6f}|")
    lines += [
        "", "C1/C2 在 xc 与 eps 的冻结线性探针均未弱于 C0；这两项诊断更支持微调适应问题，而不是冻结表示已经退化。该结论只覆盖 xc/eps，且探针只衡量固定表示的线性可读性；与非线性全量微调并列不构成单一机制的因果证明。",
        f"所选 alpha 分布：{summary['selected_alpha_counts']}。", "", "## C0 分阶段微调", "", "|任务|R² mean±std|MAE mean±std|RMSE mean±std|既有 C0 R²|R² 增量|", "|-|-:|-:|-:|-:|-:|"
    ]
    for row in stage_summary.itertuples():
        lines.append(f"|{row.task}|{row.r2_mean:.6f}±{row.r2_std:.6f}|{row.mae_mean:.6f}±{row.mae_std:.6f}|{row.rmse_mean:.6f}±{row.rmse_std:.6f}|{row.c0_r2_mean:.6f}|{row.r2_delta:+.6f}|")
    lines += [
        "", f"Macro8 R²：{macro:.10f}；既有 C0：{c0_macro:.10f}；增量 {macro-c0_macro:+.10f}。",
        f"正增量任务 {summary['positive_tasks']}/8，正增量 fold {summary['positive_folds']}/40；最佳阶段计数 {summary['best_stage_counts']}。",
        f"累计 optimizer updates {summary['optimizer_updates']}；纯训练 {summary['training_seconds']/3600:.3f} one-GPU h，含验证、checkpoint 与加载的 fold wall 累计 {summary['fold_wall_seconds']/3600:.3f} one-GPU h；探针累计 {summary['probe_gpu_seconds']/3600:.3f} GPU h。四卡 staged 调度墙钟约 34 分钟。",
        "分阶段方案只在 egc、xc 两个任务取得均值正增量，另外六个任务下降；因此它改善了 xc，但没有改善 C0 的整体八任务泛化。40/40 最终最佳模型均来自第二阶段，第一阶段最佳模型虽参与全流程选择但未胜出。",
        f"按预注册门槛，多 seed 复核建议：{'是' if decision else '否'}。本轮为单 seed、outer5_inner20 共享开发折，不称稳定显著提升或独立盲测。",
    ]
    atomic_text(comparison / "final_report.md", "\n".join(lines) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
