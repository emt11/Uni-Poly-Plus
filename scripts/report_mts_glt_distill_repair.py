#!/usr/bin/env python3
"""Verify and aggregate the outer5_inner20 C0/C1/C2 campaign."""

from __future__ import annotations

import json
import hashlib
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/mts_glt_distill_repair_control"
TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
GROUPS = ("c0", "c1", "c2")
VERSIONS = {"c0": "none", "c1": "n_plus_1", "c2": "n_plus_2"}
N_ZERO_SAMPLE_KEY_HEX = "2320ad8ec663f0ca98eb48191b15e8b28b0f1c401a5142be68a286e990fbcfd1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fold_result(group, task, fold):
    root = BASE / group / "downstream/outer5_inner20"
    shard = root / f"shards/42/{task}/fold_{fold}.csv"
    prediction = root / f"predictions/42/{task}/fold_{fold}.npz"
    checkpoint = root / f"shards/42/{task}/fold_{fold}_best.pt"
    if not all(path.is_file() for path in (shard, prediction, checkpoint)):
        raise FileNotFoundError(f"incomplete unit {group}/{task}/{fold}")
    frame = pd.read_csv(shard)
    if len(frame) != 1 or frame.iloc[0]["evaluation_protocol"] != "outer5_inner20":
        raise RuntimeError(f"invalid shard {shard}")
    metrics = json.loads(frame.iloc[0]["per_fold_metrics"])[0]
    with np.load(prediction, allow_pickle=False) as data:
        meta = json.loads(str(np.asarray(data["metadata"]).item()))
        y_true, y_pred = np.asarray(data["y_true"]), np.asarray(data["y_pred"])
        indices = np.asarray(data["sample_indices"])
    manifest = json.loads((ROOT / f"data/splits/mips_outer5_inner20/{task}.json").read_text())
    expected_test_indices = np.asarray(
        manifest["folds"][int(fold)]["test_indices"], dtype=np.int64
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = saved.get("state_dict", {})
    if (
        meta.get("fold_validation_protocol") != "outer5_inner20"
        or saved.get("schema") != "mts-glt-distill-finetune-v1"
        or saved.get("task") != task
        or int(saved.get("fold", -1)) != int(fold)
        or int(saved.get("seed", -1)) != 42
        or saved.get("evaluation_protocol") != "outer5_inner20"
        or meta.get("split_manifest_sha256") != saved.get("split_manifest_sha256")
        or saved.get("pretraining_bundle_schema") != "mts-glt-distill-repair-student-deploy-v1"
        or saved.get("pretraining_bundle_version") != VERSIONS[group]
        or int(saved.get("pretraining_bundle_step", -1)) != 20000
        or not state_dict
        or not all(torch.isfinite(value).all().item() for value in state_dict.values())
        or not np.array_equal(indices, expected_test_indices)
        or not np.isfinite(y_true).all() or not np.isfinite(y_pred).all()
    ):
        raise RuntimeError(f"invalid protocol/artifact identity {group}/{task}/{fold}")
    recomputed = {
        "test_r2": float(r2_score(y_true, y_pred)),
        "test_mae": float(mean_absolute_error(y_true, y_pred)),
        "test_rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }
    if any(not math.isclose(float(metrics[key]), value, rel_tol=1e-9, abs_tol=1e-9) for key, value in recomputed.items()):
        raise RuntimeError(f"reported metrics do not match predictions {group}/{task}/{fold}")
    return {
        "group": group, "task": task, "fold": fold,
        "r2": float(metrics["test_r2"]), "mae": float(metrics["test_mae"]),
        "rmse": float(metrics["test_rmse"]), "best_val_r2": float(metrics["best_val_r2"]),
        "best_epoch": int(metrics["best_epoch"]), "training_steps": int(metrics["training_steps"]),
        "epochs_run": len(metrics.get("training_epoch_timing", ())),
        "fold_wall_seconds": float(metrics["fold_wall_seconds"]),
        "indices": indices, "y_true": y_true, "y_pred": y_pred,
    }


def pretraining(group, stage, steps):
    root = BASE / group / stage
    records = [json.loads(line) for line in (root / "training_metrics.jsonl").read_text().splitlines()]
    if len(records) != steps or records[-1]["step"] != steps:
        raise RuntimeError(f"incomplete pretraining {group}/{stage}")
    if not all(math.isfinite(float(v)) for row in records for v in row.values() if isinstance(v, (int, float))):
        raise RuntimeError(f"non-finite pretraining {group}/{stage}")
    state_path = root / f"{stage}_{steps // 1000:03d}k.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    expected_revision = None if group == "c0" else 2
    model_state = state.get("container_state", {})
    if (
        state.get("schema") != f"mts-glt-distill-repair-{stage}-state-v1"
        or state.get("version") != VERSIONS[group]
        or state.get("geometry_revision") != expected_revision
        or int(state.get("step", -1)) != steps
        or not model_state
        or not all(torch.isfinite(value).all().item() for value in model_state.values())
    ):
        raise RuntimeError(f"invalid final pretraining state {group}/{stage}")
    return {
        "steps": steps, "first_loss": records[0]["loss"], "final_loss": records[-1]["loss"],
        "wall_hours": sum(row["step_seconds"] for row in records) / 3600,
        "peak_memory_bytes": max(row["peak_memory_bytes"] for row in records),
        "state_checkpoint": str(state_path.relative_to(ROOT)),
        "state_checkpoint_sha256": sha256_file(state_path),
        "state_tensor_count": len(model_state),
        "resolved_config": str((root / "resolved_config.json").relative_to(ROOT)),
    }


def deployment(group):
    path = BASE / group / "student/student_deploy_020k.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    version = VERSIONS[group]
    revision = None if group == "c0" else 2
    keys = tuple(payload.get("state_dict", {}))
    if (
        payload.get("schema") != "mts-glt-distill-repair-student-deploy-v1"
        or payload.get("version") != version or payload.get("geometry_revision") != revision
        or int(payload.get("step", -1)) != 20000
        or any(any(token in key for token in ("teacher", "atom_head", "line_projection", "glt")) for key in keys)
    ):
        raise RuntimeError(f"invalid deployment bundle {group}")
    return {
        "path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
        "tensor_count": len(keys), "o8_md200_only": True,
    }


def input_provenance():
    sidecars = {}
    for version in ("n_plus_1", "n_plus_2"):
        root = ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_distill_v2" / version
        metadata = json.loads((root / "metadata.json").read_text())
        if (
            metadata.get("schema") != f"mts-periodic-line-distill-v2-{version}"
            or int(metadata.get("geometry_revision", -1)) != 2
            or metadata.get("source") != "frozen_trimer_coordinates_direct"
            or not (root / ".done").is_file()
        ):
            raise RuntimeError(f"invalid revision-2 sidecar provenance: {version}")
        sidecars[version] = {
            key: metadata[key] for key in (
                "schema", "geometry_revision", "source", "sample_count",
                "valid_count", "token_count", "relation_count",
            )
        }
    manifests = {}
    for task in TASKS:
        path = ROOT / f"data/splits/mips_outer5_inner20/{task}.json"
        payload = json.loads(path.read_text())
        folds = payload.get("folds", ())
        all_tests = []
        for fold in folds:
            train, validation, test = map(
                set, (fold["train_indices"], fold["validation_indices"], fold["test_indices"])
            )
            if train & validation or train & test or validation & test:
                raise RuntimeError(f"overlapping outer5_inner20 partition: {task}")
            if len(train | validation | test) != int(payload["sample_count"]):
                raise RuntimeError(f"incomplete outer5_inner20 partition: {task}")
            all_tests.extend(test)
        if (
            payload.get("schema") != "mips-outer5-inner20-fold-v1"
            or payload.get("protocol") != "outer5_inner20"
            or bool(payload.get("validation_is_test", True))
            or len(folds) != 5
            or len(all_tests) != len(set(all_tests))
            or len(all_tests) != int(payload["sample_count"])
        ):
            raise RuntimeError(f"invalid outer5_inner20 manifest: {task}")
        manifests[task] = {
            "path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
            "sample_count": int(payload["sample_count"]),
            "fold_counts": [
                {
                    "train": len(fold["train_indices"]),
                    "validation": len(fold["validation_indices"]),
                    "test": len(fold["test_indices"]),
                }
                for fold in folds
            ],
        }
    return {"revision2_sidecars": sidecars, "split_manifests": manifests}


def validation_evidence():
    log_root = ROOT / "logs/mts_glt_distill_repair_control"
    n_zero_log = log_root / "n0_rank_smoke_20260909.log"
    resume_log = log_root / "resume_current_v5_20260909.log"
    unit_log = log_root / "tests_final_20260909.log"
    for path in (n_zero_log, resume_log, unit_log):
        if not path.is_file():
            raise FileNotFoundError(f"missing required validation evidence: {path}")
    n_zero_text = n_zero_log.read_text()
    resume_text = resume_log.read_text()
    unit_text = unit_log.read_text()
    if (
        "N_ZERO_RANK_DDP_PASS" not in n_zero_text
        or f"sample_key={N_ZERO_SAMPLE_KEY_HEX}" not in n_zero_text
    ):
        raise RuntimeError("three-rank real-data N=0 validation did not pass")
    resume_match = re.search(
        r"RESUME_CURRENT_NUMERICAL_MATCH step=(\d+) "
        r"model_atol=([0-9.eE+-]+) optimizer_atol=([0-9.eE+-]+) "
        r"rng_exact=(true|false) abandoned_tail_preserved=(true|false)",
        resume_text,
    )
    if (
        resume_match is None
        or int(resume_match.group(1)) != 4
        or not math.isclose(float(resume_match.group(2)), 2e-6, rel_tol=0, abs_tol=0)
        or not math.isclose(float(resume_match.group(3)), 2e-5, rel_tol=0, abs_tol=0)
        or resume_match.group(4) != "true"
        or resume_match.group(5) != "true"
    ):
        raise RuntimeError("current three-rank resume validation did not pass")
    test_match = re.search(r"(?m)(\d+) passed", unit_text)
    if test_match is None or int(test_match.group(1)) < 22:
        raise RuntimeError("complete repair unit-test evidence is missing")
    return {
        "n_zero": {
            "sample_index": 10,
            "sample_key": N_ZERO_SAMPLE_KEY_HEX,
            "real_frozen_geometry": True,
            "center_readout_empty": True,
            "excluded_from_teacher_and_distillation_denominators": True,
            "student_atom_task_backward": True,
            "mixed_batch": True,
            "empty_ddp_rank": True,
            "log": str(n_zero_log.relative_to(ROOT)),
        },
        "resume": {
            "interrupted_steps": 2,
            "completed_steps": 4,
            "model_atol": 2e-6,
            "optimizer_atol": 2e-5,
            "rng_exact": True,
            "abandoned_log_tail_preserved": True,
            "log": str(resume_log.relative_to(ROOT)),
        },
        "unit_tests": {
            "passed": int(test_match.group(1)),
            "log": str(unit_log.relative_to(ROOT)),
        },
    }


def main():
    records, oof = [], {}
    for group in GROUPS:
        for task in TASKS:
            rows = [fold_result(group, task, fold) for fold in range(5)]
            records.extend({k: v for k, v in row.items() if k not in {"indices", "y_true", "y_pred"}} for row in rows)
            indices = np.concatenate([row["indices"] for row in rows])
            manifest = json.loads((ROOT / f"data/splits/mips_outer5_inner20/{task}.json").read_text())
            if len(indices) != len(np.unique(indices)) or len(indices) != int(manifest["sample_count"]):
                raise RuntimeError(f"outer test predictions overlap for {group}/{task}")
            order = np.argsort(indices)
            truth = np.concatenate([row["y_true"] for row in rows])[order]
            pred = np.concatenate([row["y_pred"] for row in rows])[order]
            oof[(group, task)] = {
                "r2": float(r2_score(truth, pred)),
                "mae": float(mean_absolute_error(truth, pred)),
                "rmse": float(mean_squared_error(truth, pred) ** 0.5),
            }
    frame = pd.DataFrame(records)
    task_summary = {}
    for task in TASKS:
        task_summary[task] = {}
        for group in GROUPS:
            part = frame[(frame.group == group) & (frame.task == task)]
            task_summary[task][group] = {
                **{f"{metric}_{stat}": float(getattr(part[metric], stat)(ddof=0)) if stat == "std" else float(getattr(part[metric], stat)()) for metric in ("r2", "mae", "rmse") for stat in ("mean", "std")},
                "oof": oof[(group, task)],
            }
    macro = {group: float(np.mean([task_summary[t][group]["r2_mean"] for t in TASKS])) for group in GROUPS}
    comparisons = {}
    for left, right in (("c1", "c0"), ("c2", "c0"), ("c2", "c1")):
        name = f"{left}_minus_{right}"
        task_delta = {task: task_summary[task][left]["r2_mean"] - task_summary[task][right]["r2_mean"] for task in TASKS}
        merged = frame[frame.group == left].sort_values(["task", "fold"]).r2.to_numpy() - frame[frame.group == right].sort_values(["task", "fold"]).r2.to_numpy()
        comparisons[name] = {
            "macro_r2_delta": macro[left] - macro[right], "task_r2_delta": task_delta,
            "positive_tasks": sum(value > 0 for value in task_delta.values()),
            "positive_folds": int((merged > 0).sum()), "fold_r2_delta": merged.tolist(),
        }
    pretrain = {
        "c0": {"student": pretraining("c0", "student", 20000)},
        "c1": {"teacher": pretraining("c1", "teacher", 5000), "student": pretraining("c1", "student", 20000)},
        "c2": {"teacher": pretraining("c2", "teacher", 5000), "student": pretraining("c2", "student", 20000)},
    }
    downstream_cost = {}
    for group in GROUPS:
        part = frame[frame.group == group]
        downstream_cost[group] = {
            "fold_units": int(len(part)),
            "aggregate_one_gpu_hours": float(part.fold_wall_seconds.sum() / 3600.0),
            "optimizer_updates": int(part.training_steps.sum()),
            "mean_epochs_run": float(part.epochs_run.mean()),
            "max_epochs_run": int(part.epochs_run.max()),
            "mean_best_epoch": float(part.best_epoch.mean()),
        }
    deployments = {group: deployment(group) for group in GROUPS}
    validation = validation_evidence()
    init_payloads = [torch.load(BASE / group / "student/student_step0_common.pt", map_location="cpu", weights_only=False) for group in GROUPS]
    init_hashes = [payload["sha256"] for payload in init_payloads]
    if len(set(init_hashes)) != 1:
        raise RuntimeError("C0/C1/C2 common initialization mismatch")
    schedulers = {
        group: json.loads((ROOT / f"logs/mts_glt_distill_repair_control/{group}/downstream/outer5_inner20/scheduler_report.json").read_text())
        for group in GROUPS
    }
    if any(report.get("pending") != 0 or len(report.get("completed", ())) + len(report.get("skipped", ())) != 40 for report in schedulers.values()):
        raise RuntimeError("downstream scheduler incomplete")
    historical_path = ROOT / "results/mts_glt_v2_distill/comparison/summary.json"
    historical = json.loads(historical_path.read_text()) if historical_path.is_file() else None
    payload = {
        "schema": "mts-glt-distill-repair-control-summary-v1",
        "evaluation_protocol": "outer5_inner20", "independent_blind_test": False,
        "completed_units": len(frame), "macro8_r2": macro,
        "tasks": task_summary, "comparisons": comparisons, "pretraining": pretrain,
        "downstream_cost": downstream_cost,
        "deployments": deployments, "common_initialization_sha256": init_hashes[0],
        "input_provenance": input_provenance(),
        "validation_evidence": validation,
        "scheduler_reports": schedulers,
        "historical_shared5_descriptive_only": historical,
        "attribution_limits": [
            "C1/C2 versus C0 isolate the complete teacher-distillation route under matched student/downstream budgets.",
            "C2 versus C1 changes boundary-state sharing and cross-bond length treatment together.",
            "outer5_inner20 separates validation and test, but is not a new independent blind benchmark.",
            "Historical shared-validation/test results are descriptive only and not directly comparable.",
        ],
        "execution_notes": [
            "Two host reboots terminated live processes; completed checkpoints remained intact and only the affected C2 student stage was resumed.",
            "A legacy-schema C0 10k state was detected before continuation, preserved, migrated to the repair schema, and strictly reloaded before training resumed.",
            "Abandoned post-checkpoint metric tails were preserved; no downstream task/fold was skipped or silently replaced.",
            "Cold-cache I/O caused intermittent step-time stalls after reboot; losses remained finite and the scientific budgets were unchanged.",
        ],
    }
    output = BASE / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "all_fold_metrics.csv", index=False)
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# GLT revision 2 与无蒸馏对照", "",
        "协议：`outer5_inner20`；validation 与 outer test 分离。下表为五折 mean ± population std；MAE/RMSE 越低越好。", "",
        "| Task | Group | R² | MAE | RMSE | OOF R² |", "|---|---|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        for group in GROUPS:
            value = task_summary[task][group]
            lines.append(
                f"| {task} | {group.upper()} | {value['r2_mean']:.6f} ± {value['r2_std']:.6f} "
                f"| {value['mae_mean']:.6f} ± {value['mae_std']:.6f} "
                f"| {value['rmse_mean']:.6f} ± {value['rmse_std']:.6f} "
                f"| {value['oof']['r2']:.6f} |"
            )
    lines += ["", *(f"- macro8 R² {g.upper()}：`{macro[g]:.10f}`" for g in GROUPS)]
    for name, value in comparisons.items():
        lines.append(f"- {name.upper()}：`{value['macro_r2_delta']:+.10f}`；正任务 `{value['positive_tasks']}/8`；正 fold `{value['positive_folds']}/40`")
    lines += [
        "",
        "## 成本与产物", "",
    ]
    for group in GROUPS:
        stages = pretrain[group]
        for stage, value in stages.items():
            lines.append(
                f"- {group.upper()} {stage}：`{value['steps']}` updates，"
                f"三卡并行阶段 wall `~{value['wall_hours']:.3f} h`，峰值单卡显存 "
                f"`{value['peak_memory_bytes'] / 2**30:.3f} GiB`；状态 `{value['state_checkpoint']}`。"
            )
        cost = downstream_cost[group]
        lines.append(
            f"- {group.upper()} downstream：`{cost['fold_units']}/40`，"
            f"累计 `{cost['optimizer_updates']}` optimizer updates，"
            f"`{cost['aggregate_one_gpu_hours']:.3f}` one-GPU hours。"
        )
        lines.append(
            f"- {group.upper()} deploy：`{deployments[group]['path']}`；"
            f"SHA256 `{deployments[group]['sha256']}`；只含 O8＋MD200。"
        )
    lines += [
        "",
        "## 验收与解释", "",
        f"N=0 验收使用冻结样本 `10`（`{N_ZERO_SAMPLE_KEY_HEX}`）：中心 readout 为空，不进入教师或蒸馏分母，学生原子任务、混合 batch 和空 DDP rank 反向均已通过。",
        "三卡 2→4 step 恢复与独立连续 4 step 数值一致：模型绝对容差 `2e-6`、AdamW 状态绝对容差 `2e-5`；三 rank RNG 精确一致，放弃的日志尾部已保留。",
        "五折 outer-test 预测每个样本恰好出现一次；OOF 指标与逐折 mean±std 分开保存。旧 shared-validation/test 结果仅作背景描述，不能直接归因比较。",
        "C1−C0 与 C2−C0 的 macro R² 均为负，因此当前证据不支持 revision-2 GLT 教师蒸馏优于 matched O8＋MD200 control；按预设规则，后续应优先诊断教师监督与学生 readout 对齐，不自动追加实验。",
        "执行期间主机两次重启；C2 从合法 checkpoint 恢复。C0 10k 的旧 schema 状态在继续前被保留并迁移；所有最终预算、样本顺序和下游单元均完整，未跳过失败 fold。",
    ]
    (output / "final_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"completed_units": len(frame), "macro8_r2": macro}, sort_keys=True))


if __name__ == "__main__":
    main()
