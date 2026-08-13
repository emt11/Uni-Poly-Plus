#!/usr/bin/env python3
"""Strict paired comparison for the fresh-paired MTS T-Pretrain-0 cycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("PYTHON", "/opt/conda/envs/MTS/bin/python")
TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
FOLDS = tuple(range(5))
SOURCE_CHECKPOINT_SHA256 = (
    "56c8a0ba148280fef22ecb953705a14259fb5e87258c0cae67799d7484375e77"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def resolve(path: Path) -> dict[str, Any]:
    text = subprocess.check_output(
        [PYTHON, "scripts/resolve_mips_trimer_scage.py", str(path)],
        cwd=ROOT,
        text=True,
    )
    return json.loads(text)


def expected_finetune_hash(resolved: dict[str, Any]) -> str:
    payload = {
        "config_hash": resolved["config_hash"],
        "feature_hash": resolved["feature_config_hash"],
        "graph_hash": resolved["graph_model_config_hash"],
        "geometry_hash": resolved["geometry_model_config_hash"],
        "source_geometry_hash": resolved["source_geometry_model_config_hash"],
        "geometry_mode": resolved["graph_geometry_mode"],
        "modalities": resolved["modalities"],
        "fusion_mode": resolved["fusion_mode"],
        "finetune_mode": resolved["finetune_mode"],
        "evaluation_protocol": resolved["evaluation_protocol"],
        "modality_control": resolved["modality_control"],
        # ``resolve_mips_trimer_scage.py`` emits JSON ``null`` for an
        # un-controlled modality, while the launcher serializes the same
        # value as an empty shell variable in its finetune hash payload.
        # Normalize here so the verifier reconstructs the exact hash written
        # into the shard rather than treating equivalent identities as
        # different.
        "controlled_modality": resolved.get("controlled_modality") or "",
        "stage": "mts_property_finetune_legacy_v1",
        "dataset": "downstream_union",
        "epochs": 100,
        "batch_size": 32,
        "eval_batch_size": 64,
        "amp_dtype": "fp32",
        "patience": 10,
        "graph_wrapper_lr": 0.00001,
        "smiles_lora_lr": 0.000005,
        "adapter_lr": 0.0001,
        "head_lr": 0.0001,
        "weight_decay": 0.02,
        "warmup_epochs": 5,
        "scheduler": "cosine",
        "finetune_profile": "legacy_mts_huber_v1",
        "finetune_profile_hash": digest({
            "profile": "legacy_mts_huber_v1",
            "graph_wrapper_trainable_from_epoch": 0,
            "loss": "huber",
            "huber_beta": 0.5,
            "epochs": 100,
            "patience": 10,
            "batch_size": 32,
            "graph_wrapper_lr": 1e-5,
            "smiles_lora_lr": 5e-6,
            "adapter_lr": 1e-4,
            "head_lr": 1e-4,
            "weight_decay": 0.02,
            "warmup_epochs": 5,
            "scheduler": "cosine",
            "swa_start_epoch": -1,
            "gradient_clip": 1.0,
        }),
        "target_transform": "recommended",
        "loss": "huber",
        "huber_beta": 0.5,
        "gradient_clip": 1.0,
        "head_dropout": 0.25,
        "swa_start_epoch": -1,
    }
    return digest(payload)


def expected_profile_hash() -> str:
    return digest({
        "profile": "legacy_mts_huber_v1",
        "graph_wrapper_trainable_from_epoch": 0,
        "loss": "huber",
        "huber_beta": 0.5,
        "epochs": 100,
        "patience": 10,
        "batch_size": 32,
        "graph_wrapper_lr": 1e-5,
        "smiles_lora_lr": 5e-6,
        "adapter_lr": 1e-4,
        "head_lr": 1e-4,
        "weight_decay": 0.02,
        "warmup_epochs": 5,
        "scheduler": "cosine",
        "swa_start_epoch": -1,
        "gradient_clip": 1.0,
    })


def expected_training_hash(finetune_hash: str, seed: int = 42, workers: int = 2) -> str:
    return digest({
        "finetune_config_hash": finetune_hash,
        "seed": seed,
        "loader_workers": workers,
    })


def _float(row: dict[str, Any], key: str, *, path: Path) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"missing/non-numeric {key}: {path}") from exc
    if not np.isfinite(value):
        raise RuntimeError(f"non-finite {key}: {path}")
    return value


def _bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() not in {"false", "0", "no", ""}


def _read_arm(
    *, name: str, root: Path, config_path: Path, checkpoint: Path,
    expected_model_identity: str, allow_init: bool, workers: int,
    fresh_paired: bool, paired_init_id: str,
    expected_t0_sha256: str | None,
) -> dict[str, Any]:
    resolved = resolve(config_path)
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise RuntimeError(f"missing {name} checkpoint: {checkpoint}")
    checkpoint_sha = sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    if name == "T0" and expected_t0_sha256 and checkpoint_sha != expected_t0_sha256:
        raise RuntimeError("T0 checkpoint SHA256 does not match the fixed source contract")
    if fresh_paired:
        if meta.get("model_identity") != expected_model_identity:
            raise RuntimeError(f"{name} fresh-paired checkpoint model identity mismatch")
        if meta.get("initialization") != "fresh_paired":
            raise RuntimeError(f"{name} is not a fresh-paired checkpoint")
        if meta.get("optimizer_steps") != 20000:
            raise RuntimeError(f"{name} fresh-paired checkpoint must contain 20000 optimizer steps")
        if meta.get("parent_checkpoint") not in (None, ""):
            raise RuntimeError(f"{name} fresh-paired checkpoint unexpectedly has a parent checkpoint")
        if meta.get("paired_init_id") != paired_init_id:
            raise RuntimeError(f"{name} paired-init identity mismatch")
    elif expected_model_identity == "T1":
        if meta.get("model_identity") != "T1" or meta.get("initialization") != "function_preserving":
            raise RuntimeError("T1 formal checkpoint is not a function-preserving init")
        if not allow_init:
            raise RuntimeError("T1 comparison requires explicit init opt-in")
        if meta.get("optimizer_steps") != 0 or meta.get("source_optimizer_steps") != 20000:
            raise RuntimeError("T1 formal init optimizer-step semantics mismatch")
    expected = {(task, fold) for task in TASKS for fold in FOLDS}
    shard_paths = sorted((root / "shards" / "42").glob("*/fold_*.csv"))
    prediction_paths = sorted((root / "predictions" / "42").glob("*/fold_*.npz"))
    if len(shard_paths) != 40 or len(prediction_paths) != 40:
        raise RuntimeError(
            f"{name} requires exactly 40 shards + 40 predictions; "
            f"found {len(shard_paths)} + {len(prediction_paths)}"
        )
    profile_hash = expected_profile_hash()
    finetune_hash = expected_finetune_hash(resolved)
    training_hash = expected_training_hash(finetune_hash, workers=workers)
    cache_store = meta.get("target_contract", {}).get("store_json_sha256") or meta.get("source_contract", {}).get("cache_store_sha256") or meta.get("cache_bundle_hash")
    topology_artifact = meta.get("topology_cache_artifact_hash")
    trimer_artifact = meta.get("trimer_cache_artifact_hash")
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for shard in shard_paths:
        task = shard.parent.name
        try:
            fold = int(shard.stem.removeprefix("fold_"))
        except ValueError as exc:
            raise RuntimeError(f"invalid fold shard path: {shard}") from exc
        key = (task, fold)
        if key not in expected or key in records:
            raise RuntimeError(f"unexpected/duplicate shard: {shard}")
        frame = pd.read_csv(shard)
        if len(frame) != 1:
            raise RuntimeError(f"shard must contain exactly one row: {shard}")
        row = frame.iloc[0].to_dict()
        if str(row.get("task")) != task or int(row.get("seed", -1)) != 42:
            raise RuntimeError(f"task/seed mismatch: {shard}")
        if str(row.get("experiment_id")) != resolved["experiment_id"]:
            raise RuntimeError(f"experiment_id mismatch: {shard}")
        checks = {
            "config_hash": resolved["config_hash"],
            "resolved_config_hash": resolved["config_hash"],
            "feature_config_hash": resolved["feature_config_hash"],
            "graph_model_config_hash": resolved["graph_model_config_hash"],
            "geometry_model_config_hash": resolved["geometry_model_config_hash"],
            "source_geometry_model_config_hash": resolved["source_geometry_model_config_hash"],
            "training_config_hash": training_hash,
            "finetune_config_hash": finetune_hash,
            "finetune_profile_hash": profile_hash,
            "checkpoint_sha256": checkpoint_sha,
            "evaluation_protocol": "historical_shared5",
            "amp_dtype": "fp32",
            "batch_size": 32,
            "eval_batch_size": 64,
        }
        for field, value in checks.items():
            observed = row.get(field)
            if field in {"batch_size", "eval_batch_size"}:
                try:
                    equal = int(observed) == int(value)
                except (TypeError, ValueError):
                    equal = False
            else:
                equal = str(observed) == str(value)
            if not equal:
                raise RuntimeError(f"{name} {field} mismatch in {shard}: {observed!r} != {value!r}")
        if str(row.get("fold_validation_protocol")) != "shared_validation_test_fold" or _bool(row.get("independent_blind_test", True)):
            raise RuntimeError(f"historical_shared5 blind-test identity mismatch: {shard}")
        if cache_store is not None and str(row.get("cache_store_sha256")) != str(cache_store):
            raise RuntimeError(f"cache store identity mismatch: {shard}")
        if topology_artifact is not None and str(row.get("topology_cache_artifact_hash")) != str(topology_artifact):
            raise RuntimeError(f"topology artifact identity mismatch: {shard}")
        if trimer_artifact is not None and str(row.get("trimer_cache_artifact_hash")) != str(trimer_artifact):
            raise RuntimeError(f"Trimer artifact identity mismatch: {shard}")
        metrics = json.loads(str(row.get("per_fold_metrics", "[]")))
        if len(metrics) != 1 or int(metrics[0].get("fold", -1)) != fold:
            raise RuntimeError(f"invalid per-fold metrics: {shard}")
        metric = metrics[0]
        values = {
            "r2": _float(metric, "test_r2", path=shard),
            "mae": _float(metric, "test_mae", path=shard),
            "rmse": _float(metric, "test_rmse", path=shard),
            "wall_seconds": _float(row, "total_fold_wall_seconds", path=shard),
        }
        prediction = root / "predictions" / "42" / task / f"fold_{fold}.npz"
        if not prediction.is_file():
            raise RuntimeError(f"missing canonical prediction path: {prediction}")
        prediction_sha = sha256(prediction)
        if str(row.get("prediction_sha256")) != prediction_sha:
            raise RuntimeError(f"prediction hash mismatch: {prediction}")
        with np.load(prediction, allow_pickle=False) as arrays:
            y_true = np.asarray(arrays["y_true"], dtype=np.float64).reshape(-1)
            y_pred = np.asarray(arrays["y_pred"], dtype=np.float64).reshape(-1)
            indices = np.asarray(arrays["sample_indices"], dtype=np.int64).reshape(-1)
            metadata = json.loads(str(np.asarray(arrays["metadata"]).item()))
        if y_true.shape != y_pred.shape or y_true.shape != indices.shape or not y_true.size:
            raise RuntimeError(f"prediction shape mismatch: {prediction}")
        if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
            raise RuntimeError(f"non-finite prediction: {prediction}")
        metadata_checks = {
            "task": task,
            "fold": fold,
            "seed": 42,
            "finetune_config_hash": finetune_hash,
            "finetune_profile_hash": profile_hash,
            "checkpoint_sha256": checkpoint_sha,
        }
        for field, value in metadata_checks.items():
            if metadata.get(field) != value:
                raise RuntimeError(f"prediction metadata mismatch ({field}): {prediction}")
        records[key] = {
            "task": task,
            "fold": fold,
            "row": row,
            "metric": values,
            "prediction": {
                "path": str(prediction),
                "sha256": prediction_sha,
                "n": int(y_true.size),
                "indices": indices,
                "y_true": y_true,
                "y_pred": y_pred,
            },
        }
    if set(records) != expected:
        raise RuntimeError(f"{name} shard key set is incomplete")
    checkpoint_meta = {
        "model_identity": meta.get("model_identity", "T0" if name == "T0" else None),
        "initialization": meta.get("initialization"),
        "optimizer_steps": meta.get("optimizer_steps"),
        "source_optimizer_steps": meta.get("source_optimizer_steps"),
        "paired_init_id": meta.get("paired_init_id"),
        "experiment_id": meta.get("experiment_id"),
        "config_schema": meta.get("config_schema"),
        "config_hash": meta.get("config_hash"),
        "pretraining_dataset": meta.get("pretraining_dataset"),
        "pretrain_profile_id": (meta.get("pretrain_profile") or {}).get("profile_id"),
        "pretrain_code_identity_sha256": (
            meta.get("pretrain_code_identity") or {}
        ).get("sha256"),
        "target_config_hash": (meta.get("initialization_state") or {}).get(
            "target_config_hash"
        ),
        "target_graph_model_config_hash": (
            meta.get("initialization_state") or {}
        ).get("target_graph_model_config_hash"),
        "cache_bundle_hash": meta.get("cache_bundle_hash"),
        "topology_cache_artifact_hash": topology_artifact,
        "trimer_cache_artifact_hash": trimer_artifact,
        "cache_store_sha256": cache_store,
    }
    return {
        "name": name,
        "root": str(root.resolve()),
        "config": resolved,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "meta": checkpoint_meta,
            "fresh_paired_binding": (
                {
                    "path": str(checkpoint),
                    "sha256": checkpoint_sha,
                    "optimizer_steps": int(meta.get("optimizer_steps")),
                    "model_identity": meta.get("model_identity"),
                    "initialization": meta.get("initialization"),
                    "paired_init_id": meta.get("paired_init_id"),
                    "pretrain_config_identity": {
                        "checkpoint_experiment_id": meta.get("experiment_id"),
                        "checkpoint_config_hash": meta.get("config_hash"),
                        "target_config_hash": checkpoint_meta["target_config_hash"],
                        "graph_model_config_hash": meta.get("graph_model_config_hash"),
                        "feature_config_hash": meta.get("feature_config_hash"),
                        "pretrain_profile_id": checkpoint_meta["pretrain_profile_id"],
                        "pretrain_code_identity_sha256": checkpoint_meta[
                            "pretrain_code_identity_sha256"
                        ],
                    },
                }
                if fresh_paired else None
            ),
        },
        "counts": {"verified_shards": len(records), "verified_predictions": len(records)},
        "records": records,
        "expected_finetune_config_hash": finetune_hash,
        "expected_training_config_hash": training_hash,
        "expected_finetune_profile_hash": profile_hash,
    }


def _metric_summary(records: dict[tuple[str, int], dict[str, Any]], task: str) -> dict[str, Any]:
    values = [records[(task, fold)]["metric"] for fold in FOLDS]
    result = {}
    for metric in ("r2", "mae", "rmse"):
        data = np.asarray([item[metric] for item in values], dtype=np.float64)
        result[f"{metric}_mean"] = float(np.mean(data))
        result[f"{metric}_std"] = float(np.std(data, ddof=1))
        result[f"{metric}_values"] = [float(item) for item in data]
    result["wall_seconds_total"] = float(sum(item["wall_seconds"] for item in values))
    return result


def _report_config(resolved: dict[str, Any], *, fresh_paired: bool) -> dict[str, Any]:
    """Expose resolver identity without mislabeling a historical checkpoint hash."""

    report = json.loads(json.dumps(resolved))
    if fresh_paired:
        # The formal experiment resolver carries the old fixed-production
        # checkpoint contract for compatibility.  It is not an input to this
        # fresh-paired comparison, whose actual binding is recorded under
        # ``checkpoint.fresh_paired_binding``.
        report["shared_checkpoint_sha256"] = "not_applicable_to_fresh_paired"
        report["shared_checkpoint"] = None
    return report


def _sample_statistics(values: list[float]) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "sample_std": float(np.std(data, ddof=1)),
        "positive_count": int(np.count_nonzero(data > 0.0)),
        "negative_count": int(np.count_nonzero(data < 0.0)),
        "zero_count": int(np.count_nonzero(data == 0.0)),
        "count": int(data.size),
    }


def compare(t0: dict[str, Any], t1: dict[str, Any]) -> dict[str, Any]:
    t0_records, t1_records = t0["records"], t1["records"]
    fold_rows = []
    for task in TASKS:
        for fold in FOLDS:
            a = t0_records[(task, fold)]["metric"]
            b = t1_records[(task, fold)]["metric"]
            fold_rows.append({
                "task": task,
                "fold": fold,
                "t0_r2": a["r2"],
                "t1_r2": b["r2"],
                "delta_r2": b["r2"] - a["r2"],
                "t0_mae": a["mae"],
                "t1_mae": b["mae"],
                "delta_mae": b["mae"] - a["mae"],
                "t0_rmse": a["rmse"],
                "t1_rmse": b["rmse"],
                "delta_rmse": b["rmse"] - a["rmse"],
                "t0_wall_seconds": a["wall_seconds"],
                "t1_wall_seconds": b["wall_seconds"],
            })
    task_rows = []
    for task in TASKS:
        t0_summary = _metric_summary(t0_records, task)
        t1_summary = _metric_summary(t1_records, task)
        delta = {
            metric: [
                t1_summary[f"{metric}_values"][index] - t0_summary[f"{metric}_values"][index]
                for index in range(5)
            ]
            for metric in ("r2", "mae", "rmse")
        }
        task_rows.append({
            "task": task,
            "t0": {key: value for key, value in t0_summary.items() if not key.endswith("_values")},
            "t1": {key: value for key, value in t1_summary.items() if not key.endswith("_values")},
            "delta_t": float(np.mean(delta["r2"])),
            "paired_delta": {
                f"{metric}_mean": float(np.mean(values))
                for metric, values in delta.items()
            } | {
                f"{metric}_std": float(np.std(values, ddof=1))
                for metric, values in delta.items()
            },
            "paired_delta_values": delta,
        })
    fold_frame = pd.DataFrame(fold_rows)
    macro_fold = fold_frame.groupby("fold", sort=True)[["t0_r2", "t1_r2", "delta_r2"]].mean()
    macro = {
        "t0_r2_mean": float(macro_fold["t0_r2"].mean()),
        "t0_r2_std": float(macro_fold["t0_r2"].std(ddof=1)),
        "t1_r2_mean": float(macro_fold["t1_r2"].mean()),
        "t1_r2_std": float(macro_fold["t1_r2"].std(ddof=1)),
        "delta_r2_mean": float(macro_fold["delta_r2"].mean()),
        "delta_r2_std": float(macro_fold["delta_r2"].std(ddof=1)),
        "per_fold": [
            {"fold": int(index), "t0_r2": float(row.t0_r2), "t1_r2": float(row.t1_r2), "delta_r2": float(row.delta_r2)}
            for index, row in macro_fold.iterrows()
        ],
    }
    task_delta_statistics = _sample_statistics(
        [float(row["delta_t"]) for row in task_rows]
    )
    leave_one_task_out = []
    delta_macro_without = {}
    for excluded_task in TASKS:
        remaining = fold_frame[fold_frame["task"] != excluded_task]
        per_fold_frame = remaining.groupby("fold", sort=True)[
            ["t0_r2", "t1_r2", "delta_r2"]
        ].mean()
        per_fold = [
            {
                "fold": int(index),
                "t0_r2": float(row.t0_r2),
                "t1_r2": float(row.t1_r2),
                "delta_r2": float(row.delta_r2),
            }
            for index, row in per_fold_frame.iterrows()
        ]
        delta_key = f"delta_macro_without_{excluded_task}"
        stats = {
            "excluded_task": excluded_task,
            "remaining_task_count": len(TASKS) - 1,
            "t0_r2_mean": float(per_fold_frame["t0_r2"].mean()),
            "t0_r2_std": float(per_fold_frame["t0_r2"].std(ddof=1)),
            "t1_r2_mean": float(per_fold_frame["t1_r2"].mean()),
            "t1_r2_std": float(per_fold_frame["t1_r2"].std(ddof=1)),
            "delta_r2_mean": float(per_fold_frame["delta_r2"].mean()),
            "delta_r2_median": float(per_fold_frame["delta_r2"].median()),
            "delta_r2_sample_std": float(per_fold_frame["delta_r2"].std(ddof=1)),
            "per_fold": per_fold,
        }
        leave_one_task_out.append(stats)
        delta_macro_without[delta_key] = stats["delta_r2_mean"]
    return {
        "fold_rows": fold_rows,
        "task_rows": task_rows,
        "macro": macro,
        "task_delta_statistics": task_delta_statistics,
        "leave_one_task_out": leave_one_task_out,
        **delta_macro_without,
    }


def _write_csv(path: Path, result: dict[str, Any]) -> None:
    rows = []
    for row in result["fold_rows"]:
        rows.append({"level": "fold", **row})
    for row in result["task_rows"]:
        flat = {"level": "task", "task": row["task"], "fold": ""}
        for arm in ("t0", "t1"):
            for metric in ("r2", "mae", "rmse"):
                flat[f"{arm}_{metric}_mean"] = row[arm][f"{metric}_mean"]
                flat[f"{arm}_{metric}_std"] = row[arm][f"{metric}_std"]
        flat.update({f"paired_{key}": value for key, value in row["paired_delta"].items()})
        flat["delta_t"] = row["delta_t"]
        rows.append(flat)
    rows.append({
        "level": "task_delta_summary",
        "task": "all_tasks",
        "fold": "",
        **{f"delta_{key}": value for key, value in result["task_delta_statistics"].items()},
    })
    for row in result["leave_one_task_out"]:
        rows.append({
            "level": "leave_one_task_out",
            "task": row["excluded_task"],
            "fold": "",
            **{key: value for key, value in row.items() if key not in {"excluded_task", "per_fold"}},
        })
    rows.append({"level": "macro", "task": "macro", "fold": "", **result["macro"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    os.replace(temporary, path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0-root", type=Path, required=True)
    parser.add_argument("--t1-root", type=Path, required=True)
    parser.add_argument("--t0-config", type=Path, default=ROOT / "configs/mts/experiments/T0_o8_matched_t1_formal_v1.json")
    parser.add_argument("--t1-config", type=Path, default=ROOT / "configs/mts/experiments/T1_msta_formal_v1.json")
    parser.add_argument("--t0-checkpoint", type=Path, default=ROOT / "pretrained_models/mts/mts_joint_pretraining_pi1m_v2_seed42_canonical_angle20_v1.pth")
    parser.add_argument("--t1-checkpoint", type=Path, default=ROOT / "pretrained_models/mts_multiscale_topology/t1_formal_v1/mts_t1_formal_function_preserving_init.pth")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--execution-summary", type=Path)
    parser.add_argument("--tmux-session", default="Uni-Poly")
    parser.add_argument("--t0-window", default="mts-t0-formal-v2")
    parser.add_argument("--t1-window", default="mts-t1-formal-v2")
    parser.add_argument("--t0-log-dir", default="logs/mts_multiscale_topology/t0_t1_formal_v1/T0_o8")
    parser.add_argument("--t1-log-dir", default="logs/mts_multiscale_topology/t0_t1_formal_v1/T1_msta")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--fresh-paired",
        action="store_true",
        help="validate two newly trained 20k fresh-paired checkpoints instead of historical T0/function-preserving T1",
    )
    parser.add_argument(
        "--paired-init-id",
        default="mts_t_pretrain0_matched_v1",
        help="paired initialization identity required by --fresh-paired",
    )
    parser.add_argument(
        "--t0-expected-sha256",
        default=None,
        help="optional fixed T0 SHA256; omitted for --fresh-paired",
    )
    args = parser.parse_args()
    if args.fresh_paired and args.t0_expected_sha256:
        raise SystemExit("--fresh-paired cannot be combined with --t0-expected-sha256")
    expected_t0_sha256 = None if args.fresh_paired else (args.t0_expected_sha256 or SOURCE_CHECKPOINT_SHA256)
    t0 = _read_arm(
        name="T0", root=args.t0_root.resolve(), config_path=args.t0_config.resolve(),
        checkpoint=args.t0_checkpoint, expected_model_identity="T0", allow_init=False,
        workers=args.workers, fresh_paired=args.fresh_paired,
        paired_init_id=args.paired_init_id, expected_t0_sha256=expected_t0_sha256,
    )
    t1 = _read_arm(
        name="T1", root=args.t1_root.resolve(), config_path=args.t1_config.resolve(),
        checkpoint=args.t1_checkpoint, expected_model_identity="T1", allow_init=True,
        workers=args.workers, fresh_paired=args.fresh_paired,
        paired_init_id=args.paired_init_id, expected_t0_sha256=None,
    )
    if t0["config"]["feature_config_hash"] != t1["config"]["feature_config_hash"] or t0["config"]["geometry_model_config_hash"] != t1["config"]["geometry_model_config_hash"] or t0["config"]["source_geometry_model_config_hash"] != t1["config"]["source_geometry_model_config_hash"]:
        raise RuntimeError("T0/T1 cache or geometry identity is not paired")
    if t0["checkpoint"]["meta"]["cache_store_sha256"] != t1["checkpoint"]["meta"]["cache_store_sha256"]:
        raise RuntimeError("T0/T1 cache store identity differs")
    result = compare(t0, t1)
    t0_report = dict(t0)
    t1_report = dict(t1)
    t0_report["config"] = _report_config(t0["config"], fresh_paired=args.fresh_paired)
    t1_report["config"] = _report_config(t1["config"], fresh_paired=args.fresh_paired)
    phase = "T-Pretrain-0" if args.fresh_paired else "T-Eval"
    payload = {
        "schema": (
            "mts-t-pretrain0-formal-comparison-v2"
            if args.fresh_paired else "mts-t0-t1-formal-comparison-v1"
        ),
        "cycle_id": "mts_t_pretrain0_matched_v1" if args.fresh_paired else "mts_t0_t1_fixed_pretraining_formal_v2",
        "phase": phase,
        "protocol": {
            "tasks": list(TASKS),
            "folds": list(FOLDS),
            "seed": 42,
            "evaluation_protocol": "historical_shared5",
            "independent_blind_test": False,
            "train_batch_size": 32,
            "eval_batch_size": 64,
            "amp_dtype": "fp32",
            "dataloader_workers": args.workers,
            "schedule": "lpt_v1",
        },
        "arms": {
            "T0": {
                key: value for key, value in t0_report.items() if key != "records"
            },
            "T1": {
                key: value for key, value in t1_report.items() if key != "records"
            },
        },
        "verified_counts": {
            "T0": t0["counts"],
            "T1": t1["counts"],
        },
        "comparison": result,
        "limitations": (
            [
                "Both arms use fresh-paired 20k pretraining from the shared step-0 initialization; this is not a continuation or warm-start comparison.",
                "historical_shared5 reuses the project's shared validation/test fold and is not an independent blind test.",
                "No automatic promotion threshold is applied and T0 remains the production default.",
            ]
            if args.fresh_paired
            else [
                "T1 uses a function-preserving initialization from the fixed T0 20k checkpoint; this is not T1 20k pretraining.",
                "The comparison is fixed-T0-pretraining downstream fine-tuning evidence only; it is not an end-to-end T1 pretraining claim.",
                "historical_shared5 reuses the project's shared validation/test fold and is not an independent blind test.",
                "No automatic promotion threshold is applied and T0 remains the production default.",
            ]
        ),
    }
    _write_json(args.output_json.resolve(), payload)
    _write_csv(args.output_csv.resolve(), result)
    if args.execution_summary:
        summary = {
            "schema": (
                "mts-t-pretrain0-formal-execution-summary-v2"
                if args.fresh_paired else "mts-t0-t1-formal-execution-summary-v1"
            ),
            "cycle_id": (
                "mts_t_pretrain0_matched_v1"
                if args.fresh_paired
                else "mts_t0_t1_fixed_pretraining_formal_v2"
            ),
            "phase": phase,
            "status": "formal_comparison_verified",
            "tmux": {
                "session": args.tmux_session,
                "t0_window": args.t0_window,
                "t1_window": args.t1_window,
            },
            "logs": {"T0": args.t0_log_dir, "T1": args.t1_log_dir},
            "verified_counts": {"T0": t0["counts"], "T1": t1["counts"]},
            "formal_comparison_json": str(args.output_json.resolve()),
            "formal_comparison_csv": str(args.output_csv.resolve()),
            "macro": result["macro"],
            "limitations": payload["limitations"],
        }
        _write_json(args.execution_summary.resolve(), summary)
    print(json.dumps({"T0": t0["counts"], "T1": t1["counts"], "macro": result["macro"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
