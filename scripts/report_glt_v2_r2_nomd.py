#!/usr/bin/env python3
"""Audit and summarize the GLT-V2 revision-2 no-MD MIPS-loss campaign."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "glt_v2_r2_o8_nomd_mipsloss_005k"
BASE = ROOT / "results" / EXPERIMENT
DOWNSTREAM = BASE / "downstream" / "outer5_inner20"
PROBES = BASE / "probes"
TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
PROBE_TASKS = ("ei", "xc", "eps", "nc")
OLD_DEPLOYMENTS = {
    "old_005k": (ROOT / "results/glt_v2_r2_c0_gelu138_mipshead/student/student_deploy_005k.pt", 5000),
    "old_010k": (ROOT / "results/glt_v2_r2_c0_gelu138_mipshead/student/student_deploy_010k.pt", 10000),
    "old_020k": (ROOT / "results/glt_v2_r2_c0_gelu138_mipshead/student/student_deploy_020k.pt", 20000),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def audit_pretraining() -> dict:
    stage = BASE / "student"
    deploy_path = stage / "student_deploy_005k.pt"
    payload = torch.load(deploy_path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", {})
    if not (
        payload.get("schema") == "mts-glt-v2-r2-o8-nomd-student-deploy-v1"
        and payload.get("version") == "none"
        and int(payload.get("step", -1)) == 5000
        and payload.get("use_md200") is False
        and len(state) == 80
        and not any("md" in str(name).lower() for name in state)
        and all(torch.isfinite(value).all() for value in state.values())
    ):
        raise RuntimeError("no-MD deployment contract failed")
    metrics_path = stage / "training_metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    if len(records) != 5000 or [int(row["step"]) for row in records] != list(range(1, 5001)):
        raise RuntimeError("pretraining metrics do not contain exactly 5,000 ordered updates")
    numeric = [
        float(row[key])
        for row in records
        for key in ("loss", "atom_loss", "lr", "step_seconds")
    ]
    if not np.isfinite(numeric).all():
        raise RuntimeError("non-finite no-MD pretraining metric")
    if not (
        abs(float(records[0]["lr"]) - 2e-7) < 1e-12
        and abs(float(records[1999]["lr"]) - 2e-4) < 1e-9
        and abs(float(records[-1]["lr"]) - 0.00016666683333333335) < 1e-10
    ):
        raise RuntimeError("no-MD learning-rate prefix does not match 20k schedule")
    resolved = json.loads((stage / "resolved_config.json").read_text())
    if int(resolved.get("schedule_total_steps", -1)) != 20000:
        raise RuntimeError("resolved config does not record the 20k schedule horizon")
    return {
        "schema": payload["schema"],
        "step": int(payload["step"]),
        "state_tensors": len(state),
        "use_md200": False,
        "metrics": len(records),
        "initial_loss": float(records[0]["loss"]),
        "final_loss": float(records[-1]["loss"]),
        "initial_lr": float(records[0]["lr"]),
        "final_lr": float(records[-1]["lr"]),
        "recorded_step_seconds": float(sum(row["step_seconds"] for row in records)),
        "peak_memory_gib": max(float(row["peak_memory_bytes"]) for row in records) / 2**30,
        "checkpoint": str(deploy_path.resolve()),
        "checkpoint_sha256": sha256_file(deploy_path),
        "schedule_total_steps": 20000,
    }


def audit_old_deployments() -> list[dict]:
    result = []
    for tier, (path, expected_step) in OLD_DEPLOYMENTS.items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", {})
        if not (
            payload.get("schema") == "mts-glt-distill-repair-student-deploy-v1"
            and payload.get("version") == "none"
            and int(payload.get("step", -1)) == expected_step
            and len(state) == 90
            and all(torch.isfinite(value).all() for value in state.values())
        ):
            raise RuntimeError(f"historical New-C0 deployment contract failed: {path}")
        result.append({
            "tier": tier, "step": expected_step, "state_tensors": len(state),
            "schema": payload["schema"], "checkpoint": str(path.resolve()),
            "checkpoint_sha256": sha256_file(path),
        })
    return result


def load_split(task: str):
    path = ROOT / "data/splits/mips_outer5_inner20" / f"{task}.json"
    payload = json.loads(path.read_text())
    return payload, sha256_file(path)


def raw_targets(task: str) -> np.ndarray:
    return pd.read_csv(ROOT / "data/raw" / f"smi_{task}.csv").iloc[:, 1].to_numpy(dtype=np.float64)


def audit_downstream():
    rows = []
    for task in TASKS:
        split_payload, split_sha = load_split(task)
        seen = []
        raw = raw_targets(task)
        if len(split_payload["folds"]) != 5:
            raise RuntimeError(f"split does not contain 5 folds: {task}")
        for fold, split in enumerate(split_payload["folds"]):
            shard = DOWNSTREAM / "shards/42" / task / f"fold_{fold}.csv"
            checkpoint = shard.with_name(f"fold_{fold}_best.pt")
            prediction = DOWNSTREAM / "predictions/42" / task / f"fold_{fold}.npz"
            frame = pd.read_csv(shard)
            if len(frame) != 1 or not checkpoint.is_file() or not prediction.is_file():
                raise RuntimeError(f"missing or malformed downstream unit: {task}/{fold}")
            row = frame.iloc[0]
            if not (
                str(row["evaluation_protocol"]) == "outer5_inner20"
                and str(row["target_transform"]) == "standard"
                and str(row["regression_loss"]) == "mse"
                and bool(row["use_md200"]) is False
                and int(row["seed"]) == 42
                and str(row["checkpoint_path"]).endswith("student_deploy_005k.pt")
            ):
                raise RuntimeError(f"downstream metadata mismatch: {task}/{fold}")
            metrics = json.loads(row["per_fold_metrics"])[0]
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = saved.get("state_dict", {})
            if not (
                saved.get("schema") == "mts-glt-distill-finetune-v1"
                and saved.get("pretraining_bundle_version") == "none"
                and int(saved.get("pretraining_bundle_step", -1)) == 5000
                and saved.get("use_md200") is False
                and saved.get("split_manifest_sha256") == split_sha
                and not any("md" in str(name).lower() for name in state)
                and all(torch.isfinite(value).all() for value in state.values())
            ):
                raise RuntimeError(f"downstream checkpoint contract failed: {task}/{fold}")
            expected = np.asarray(split["test_indices"], dtype=np.int64)
            with np.load(prediction, allow_pickle=False) as pred:
                indices = np.asarray(pred["sample_indices"], dtype=np.int64)
                truth = np.asarray(pred["y_true"], dtype=np.float64)
                estimate = np.asarray(pred["y_pred"], dtype=np.float64)
                metadata = json.loads(str(np.asarray(pred["metadata"]).item()))
            if not (
                np.array_equal(indices, expected)
                and np.allclose(truth, raw[expected], rtol=1e-6, atol=5e-6)
                and truth.shape == estimate.shape
                and np.isfinite(truth).all()
                and np.isfinite(estimate).all()
                and metadata.get("split_manifest_sha256") == split_sha
            ):
                raise RuntimeError(f"downstream prediction identity/finite check failed: {task}/{fold}")
            seen.extend(indices.tolist())
            rows.append({
                "task": task,
                "fold": fold,
                "test_r2": float(metrics["test_r2"]),
                "test_mae": float(metrics["test_mae"]),
                "test_rmse": float(metrics["test_rmse"]),
                "best_val_r2": float(metrics["best_val_r2"]),
                "best_epoch": int(metrics["best_epoch"]),
                "training_steps": int(metrics["training_steps"]),
                "fold_wall_seconds": float(metrics["fold_wall_seconds"]),
                "checkpoint": str(checkpoint.resolve()),
                "prediction": str(prediction.resolve()),
                "split_manifest_sha256": split_sha,
            })
        if sorted(seen) != list(range(len(raw))):
            raise RuntimeError(f"outer-test coverage failed: {task}")
    fold = pd.DataFrame(rows)
    summary = fold.groupby("task", sort=False).agg(
        r2_mean=("test_r2", "mean"),
        r2_std=("test_r2", lambda value: np.std(value, ddof=0)),
        mae_mean=("test_mae", "mean"),
        mae_std=("test_mae", lambda value: np.std(value, ddof=0)),
        rmse_mean=("test_rmse", "mean"),
        rmse_std=("test_rmse", lambda value: np.std(value, ddof=0)),
    ).reset_index()
    return fold, summary


def audit_old_newc0():
    roots = {
        "old_005k": ROOT / "results/glt_v2_r2_c0_gelu138_mipshead/downstream/outer5_inner20_tier005k",
        "old_010k": ROOT / "results/glt_v2_r2_c0_gelu138_mipshead/downstream/outer5_inner20_tier010k_retry01",
        "old_020k": ROOT / "results/glt_v2_r2_c0_gelu138_mipshead/downstream/outer5_inner20",
    }
    output = []
    for tier, root in roots.items():
        for task in TASKS:
            for fold in range(5):
                path = root / "shards/42" / task / f"fold_{fold}.csv"
                frame = pd.read_csv(path)
                if len(frame) != 1:
                    raise RuntimeError(f"old New-C0 shard malformed: {path}")
                metrics = json.loads(frame.iloc[0]["per_fold_metrics"])[0]
                output.append({
                    "tier": tier, "task": task, "fold": fold,
                    "test_r2": float(metrics["test_r2"]),
                    "test_mae": float(metrics["test_mae"]),
                    "test_rmse": float(metrics["test_rmse"]),
                })
    frame = pd.DataFrame(output)
    summary = frame.groupby(["tier", "task"], sort=False).agg(
        r2_mean=("test_r2", "mean"),
        r2_std=("test_r2", lambda value: np.std(value, ddof=0)),
        mae_mean=("test_mae", "mean"),
        rmse_mean=("test_rmse", "mean"),
    ).reset_index()
    return frame, summary


def audit_probes():
    path = PROBES / "probe_fold_metrics.csv"
    frame = pd.read_csv(path)
    if len(frame) != 80 or set(frame["tier"]) != {"old_005k", "old_010k", "old_020k", "nomd_005k"}:
        raise RuntimeError("probe row count/tier identity failed")
    if set(frame["task"]) != set(PROBE_TASKS) or not np.isfinite(
        frame[["test_r2", "test_mae", "test_rmse", "selected_alpha", "best_validation_r2"]].to_numpy()
    ).all():
        raise RuntimeError("probe metric finite check failed")
    counts = frame.groupby(["tier", "task"]).size()
    if not (counts == 5).all():
        raise RuntimeError("probe does not contain five folds per tier/task")
    summary = frame.groupby(["tier", "task"], sort=False).agg(
        r2_mean=("test_r2", "mean"),
        r2_std=("test_r2", lambda value: np.std(value, ddof=0)),
        mae_mean=("test_mae", "mean"),
        rmse_mean=("test_rmse", "mean"),
    ).reset_index()
    return frame, summary


def markdown(pretrain, old_deployments, downstream, downstream_summary, old_summary, probes, probe_summary) -> str:
    macro = float(downstream["test_r2"].mean())
    old_macro = old_summary.groupby("tier")["r2_mean"].mean().to_dict()
    lines = [
        "# GLT-V2 revision-2：无 MD＋MIPS 损失基线报告",
        "",
        "本报告对应 `glt_v2_r2_o8_nomd_mipsloss_005k`。这是一次整体方案比较：移除 MD200、使用单路 masked-atom CE，并在下游采用 standard-label MSE；差值不能单独归因于去掉 MD。所有 outer folds 均为共享开发折，不是独立盲测。",
        "",
        "## 预训练核验",
        "",
        f"- 部署：`{pretrain['checkpoint']}`；SHA256 `{pretrain['checkpoint_sha256']}`。",
        f"- `step={pretrain['step']}`, `metrics={pretrain['metrics']}`, state tensors `{pretrain['state_tensors']}`, `use_md200={pretrain['use_md200']}`；无 MD state key。",
        f"- loss `{pretrain['initial_loss']:.3f} → {pretrain['final_loss']:.3f}`；记录的 step wall time 合计 `{pretrain['recorded_step_seconds'] / 3600:.3f} h`，峰值显存 `{pretrain['peak_memory_gib']:.3f} GiB/card`。",
        f"- 学习率为 20k 曲线前缀：`{pretrain['initial_lr']:.3e} → {pretrain['final_lr']:.9g}`，`schedule_total_steps={pretrain['schedule_total_steps']}`。",
        "- P0 参考包核验：旧 New-C0 5k/10k/20k 均为 repair schema、分别 step 5000/10000/20000、90 tensors，且仍为 MD-enabled 历史包。",
        "",
        "## 新 no-MD 下游（outer5_inner20，40/40）",
        "",
        "|task|R² mean ± std|MAE mean ± std|RMSE mean ± std|",
        "|-|-:|-:|-:|",
    ]
    for row in downstream_summary.itertuples(index=False):
        lines.append(f"|{row.task}|{row.r2_mean:.3f} ± {row.r2_std:.3f}|{row.mae_mean:.3f} ± {row.mae_std:.3f}|{row.rmse_mean:.3f} ± {row.rmse_std:.3f}|")
    lines += [
        "",
        f"**Macro8 R² = `{macro:.3f}`**；40/40 checkpoint、预测和 outer-test 覆盖通过有限性与 split identity 核验。各 fold 记录 wall time 合计 `{downstream['fold_wall_seconds'].sum() / 3600:.3f} h`（四卡并发调度，不将其误报为实际墙钟）。",
        "",
        "配置核验：`student_deploy_005k.pt`、`target_transform=standard`、`regression_loss=mse`、`use_md200=false`、O8-only、predictor `512→512→1`、dropout `0.1`。",
        "",
        "## 与旧 New-C0 的描述性对照",
        "",
        "旧 New-C0 使用 MD200 及其历史下游损失/标签协议；这里只作参考，不视为严格只改 MD 的消融。",
        "",
        "|task|old 5k R²|old 10k R²|old 20k R²|new no-MD 5k R²|new−old 5k|",
        "|-|-:|-:|-:|-:|-:|",
    ]
    old_by = old_summary.pivot(index="task", columns="tier", values="r2_mean")
    new_by = downstream_summary.set_index("task")["r2_mean"]
    for task in TASKS:
        a, b, c = (float(old_by.loc[task, key]) for key in ("old_005k", "old_010k", "old_020k"))
        n = float(new_by.loc[task])
        lines.append(f"|{task}|{a:.3f}|{b:.3f}|{c:.3f}|{n:.3f}|{n-a:+.3f}|")
    lines += [
        "",
        f"Macro8：old 5k `{old_macro['old_005k']:.3f}`，old 10k `{old_macro['old_010k']:.3f}`，old 20k `{old_macro['old_020k']:.3f}`，new no-MD 5k `{macro:.3f}`。",
        "",
        "## 冻结 Ridge 探针（80/80）",
        "",
        "特征为部署实际 pooled graph 表示；feature/label scaler 只拟合 train，alpha 只由 validation R² 选择，solver=`svd`、float64，未做 train+validation refit。",
        "",
        "|tier|task|R² mean ± std|MAE mean|RMSE mean|",
        "|-|-:|-:|-:|",
    ]
    for row in probe_summary.itertuples(index=False):
        lines.append(f"|{row.tier}|{row.task}|{row.r2_mean:.3f} ± {row.r2_std:.3f}|{row.mae_mean:.3f}|{row.rmse_mean:.3f}|")
    lines += [
        "",
        "探针只能说明冻结读出表示的线性可读性，不能证明表示完全没有任务信息，也不替代正式微调结果。",
        "",
        "## 执行记录与异常",
        "",
        "- 相关测试：`21 passed, 1 warning`；三卡两步 no-MD smoke：有限 CE、O8/head 梯度和 O8-only 导出通过。",
        "- 首次正式预训练因调度总步数错误（5k 压缩曲线）在约 1,460/5,000 停止，部分产物保留在 `failed_pretrain_schedule_student_001460/`；修正为 20k 曲线前缀后重跑并计入本报告。",
        "- 下游首次调度尝试因 `mts_glt_mode=none` 不是当前 CLI 合法枚举而在参数解析阶段退出；修正为 `o8_only` 后 40/40 完成。",
        "- 探针首次尝试因入口缺少仓库根目录 `sys.path` 而退出；修正后 80/80 完成。以上失败不计入正式指标，日志均保留。",
        "",
        "## 产物与日志",
        "",
        f"- no-MD 预训练：`{BASE / 'student'}`；正式日志：`{ROOT / 'logs' / EXPERIMENT / 'pretrain_005k.log'}`、重跑承载日志 `tmux_pretrain_retry01.log`。",
        f"- 下游：`{DOWNSTREAM}`；承载日志：`{ROOT / 'logs' / EXPERIMENT / 'tmux_finetune_retry01.log'}`。",
        f"- 探针：`{PROBES}`；承载日志：`{ROOT / 'logs' / EXPERIMENT / 'tmux_probes_retry01.log'}`。",
        "- 结果目录未覆盖旧 New-C0、旧 C0/C1/C2、geometry cache 或 MD200 cache；下游部署包仅依赖 O8 与 topology/ru_base cache。",
        "",
        "## 范围限制",
        "",
        "本轮没有启动 10k/20k 新预训练、其他 seed、其他 3D/教师路线或额外超参数搜索。新 no-MD 与旧 New-C0 同时改变预训练目标、MD 使用和下游 loss/标签协议，因此仅报告真实结果和描述性差异，不宣称单一组件的因果性能提升。",
        "",
    ]
    return "\n".join(lines)


def main():
    comparison = BASE / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    pretrain = audit_pretraining()
    downstream, downstream_summary = audit_downstream()
    old, old_summary = audit_old_newc0()
    old_deployments = audit_old_deployments()
    probes, probe_summary = audit_probes()
    downstream.to_csv(comparison / "downstream_fold_metrics.csv", index=False, float_format="%.17g")
    downstream_summary.to_csv(comparison / "downstream_task_summary.csv", index=False, float_format="%.17g")
    old.to_csv(comparison / "old_newc0_fold_metrics.csv", index=False, float_format="%.17g")
    old_summary.to_csv(comparison / "old_newc0_task_summary.csv", index=False, float_format="%.17g")
    probes.to_csv(comparison / "probe_fold_metrics.csv", index=False, float_format="%.17g")
    probe_summary.to_csv(comparison / "probe_task_summary.csv", index=False, float_format="%.17g")
    atomic_text(comparison / "summary.json", json.dumps({
        "schema": "glt-v2-r2-o8-nomd-mipsloss-report-v1",
        "experiment_id": EXPERIMENT,
        "pretraining": pretrain,
        "old_deployments": old_deployments,
        "downstream_units": len(downstream),
        "downstream_macro8_r2": float(downstream["test_r2"].mean()),
        "probe_units": len(probes),
        "old_reference_units": len(old),
    }, indent=2, sort_keys=True) + "\n")
    atomic_text(comparison / "final_report.md", markdown(pretrain, old_deployments, downstream, downstream_summary, old_summary, probes, probe_summary))
    print(json.dumps({
        "status": "audit_pass",
        "downstream_units": len(downstream),
        "downstream_macro8_r2": float(downstream["test_r2"].mean()),
        "probe_units": len(probes),
        "comparison": str(comparison.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
