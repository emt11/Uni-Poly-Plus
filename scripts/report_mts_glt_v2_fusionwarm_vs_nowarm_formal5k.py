#!/usr/bin/env python3
"""Strictly pair formal GLT-v2 Warm0 with the schedule-matched Warm5 arm."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/mts/glt_v2_fusionwarm_vs_nowarm_formal5k_v1.json"


def _atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _one_row(root: Path, task: str, fold: int):
    path = root / "shards" / "42" / task / f"fold_{fold}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise RuntimeError(f"expected one row in {path}")
    row = frame.iloc[0]
    value = float(row["avg_test_r2"])
    if not np.isfinite(value):
        raise RuntimeError(f"non-finite test R2 in {path}")
    return path, row, value


def _prediction(root: Path, task: str, fold: int):
    path = root / "predictions" / "42" / task / f"fold_{fold}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(np.asarray(payload["metadata"]).item()))
        result = {
            "y_true": np.asarray(payload["y_true"]).copy(),
            "y_pred": np.asarray(payload["y_pred"]).copy(),
            "sample_indices": np.asarray(payload["sample_indices"]).copy(),
            "metadata": metadata,
        }
    if not np.isfinite(result["y_true"]).all() or not np.isfinite(result["y_pred"]).all():
        raise RuntimeError(f"non-finite prediction payload: {path}")
    return path, result


def _validate_row(row, config, *, warm5):
    checkpoint = str((ROOT / config["checkpoint"]).resolve())
    expected = {
        "checkpoint_path": checkpoint,
        "mts_glt_mode": "o8_glt_atom",
        "amp_dtype": "fp32",
        "train_batch_size": 32,
        "eval_batch_size": 64,
        "seed": 42,
        "evaluation_protocol": "historical_shared5",
        "regression_loss": "huber",
        "huber_beta": 0.5,
        "head_dropout": 0.25,
        "refit_full_train": False,
    }
    for key, value in expected.items():
        observed = row.get(key)
        if isinstance(value, float):
            good = np.isclose(float(observed), value, atol=1e-12, rtol=0.0)
        elif isinstance(value, bool):
            good = bool(observed) is value
        else:
            good = observed == value
        if not good:
            raise RuntimeError(f"provenance mismatch {key}: {observed!r} != {value!r}")
    strategy = str(row.get("mts_glt_fusion_strategy"))
    if strategy != ("fusion_warm" if warm5 else "legacy_zero"):
        raise RuntimeError(f"unexpected fusion strategy: {strategy}")
    if warm5:
        if int(row.get("mts_glt_fusion_warm_epochs")) != 5:
            raise RuntimeError("Warm5 shard does not record five frozen epochs")
        if not np.isclose(float(row.get("mts_glt_initial_alpha")), 0.05):
            raise RuntimeError("Warm5 shard does not record alpha=0.05")


def _validate_pair(config, warm0_root, warm5_root, task, fold):
    _, warm0_row, warm0 = _one_row(warm0_root, task, fold)
    _, warm5_row, warm5 = _one_row(warm5_root, task, fold)
    _validate_row(warm0_row, config, warm5=False)
    _validate_row(warm5_row, config, warm5=True)
    _, pred0 = _prediction(warm0_root, task, fold)
    _, pred5 = _prediction(warm5_root, task, fold)
    for label, prediction in (("Warm0", pred0), ("Warm5", pred5)):
        metadata = prediction["metadata"]
        if (
            metadata.get("task") != task
            or int(metadata.get("fold", -1)) != fold
            or int(metadata.get("seed", -1)) != 42
        ):
            raise RuntimeError(f"{label} prediction metadata mismatch: {task}/fold{fold}")
    if not np.array_equal(pred0["sample_indices"], pred5["sample_indices"]):
        raise RuntimeError(f"sample indices differ: {task}/fold{fold}")
    if not np.array_equal(pred0["y_true"], pred5["y_true"]):
        raise RuntimeError(f"targets differ: {task}/fold{fold}")
    if int(pred0["metadata"]["fold_seed"]) != int(pred5["metadata"]["fold_seed"]):
        raise RuntimeError(f"fold seed differs: {task}/fold{fold}")
    return {
        "task": task, "fold": fold,
        "fold_seed": int(pred0["metadata"]["fold_seed"]),
        "Warm0_R2": warm0, "Warm5_R2": warm5,
        "DeltaWarm": warm5 - warm0,
    }


def main():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = ROOT / config["output_root"]
    warm0_root = ROOT / config["formal_warm0_root"]
    warm5_root = output / "Warm5"
    o8_root = ROOT / config["formal_o8_root"]

    rows = [
        _validate_pair(config, warm0_root, warm5_root, task, fold)
        for task in config["tasks"] for fold in config["folds"]
    ]
    folds = pd.DataFrame(rows)
    task_rows = []
    for task in config["tasks"]:
        subset = folds[folds.task == task]
        task_rows.append({
            "task": task,
            "Warm0": float(subset.Warm0_R2.mean()),
            "Warm5": float(subset.Warm5_R2.mean()),
            "DeltaWarm": float(subset.DeltaWarm.mean()),
            "Warm0_sample_std": float(subset.Warm0_R2.std(ddof=1)),
            "Warm5_sample_std": float(subset.Warm5_R2.std(ddof=1)),
            "positive_folds": int((subset.DeltaWarm > 0).sum()),
        })
    tasks = pd.DataFrame(task_rows)
    o8_values = []
    for task in config["tasks"]:
        values = [_one_row(o8_root, task, fold)[2] for fold in config["folds"]]
        o8_values.append(float(np.mean(values)))
    formal_o8 = float(np.mean(o8_values))
    macro0 = float(tasks.Warm0.mean())
    macro5 = float(tasks.Warm5.mean())
    macro_delta = float(tasks.DeltaWarm.mean())
    if not np.isclose(macro5 - macro0, macro_delta, atol=1e-12, rtol=0.0):
        raise RuntimeError("macro identity Warm5-Warm0 != mean task DeltaWarm")
    delta = folds.DeltaWarm.to_numpy(dtype=np.float64)
    positive_tasks = int((tasks.DeltaWarm > 0).sum())
    median_task = float(tasks.DeltaWarm.median())
    passed = macro_delta > 0 and median_task > 0 and positive_tasks >= 5

    summary = {
        "schema": "mts-glt-v2-warm5-vs-warm0-summary-v1",
        "checkpoint": str((ROOT / config["checkpoint"]).resolve()),
        "checkpoint_step": 5000,
        "tasks": config["tasks"], "folds": config["folds"], "seed": 42,
        "reused_warm0_runs": 40, "new_warm0_runs": 0,
        "new_warm5_runs": 40, "failed_runs": 0,
        "macro8_warm0": macro0, "macro8_warm5": macro5,
        "macro8_delta_warm": macro_delta,
        "median_task_delta_warm": median_task,
        "positive_delta_warm_tasks": positive_tasks,
        "positive_delta_warm_folds": int((delta > 0).sum()),
        "fold_mean": float(delta.mean()),
        "fold_median": float(np.median(delta)),
        "fold_p25": float(np.quantile(delta, 0.25)),
        "fold_p75": float(np.quantile(delta, 0.75)),
        "fold_min": float(delta.min()), "fold_max": float(delta.max()),
        "formal_o8_only_macro8": formal_o8,
        "warm0_fused_minus_o8": macro0 - formal_o8,
        "warm5_fused_minus_o8": macro5 - formal_o8,
        "final_glt_v2_warm_decision": "PASS" if passed else "FAIL",
        "evaluation_protocol": "historical_shared5",
        "independent_blind_test": False,
    }
    module_contract = {
        "schema": "mts-glt-v2-warm-module-contract-v1",
        "pretrained_frozen_stage1": [
            "encoders.graph.encoder.o8 excluding md_residual and disabled star_distance_bias",
            "encoders.graph.encoder.glt including learned geometry bases, line layers, final normalization, incidence projection, incidence shift embedding and atom output normalization",
        ],
        "downstream_trainable_stage1": [
            "encoders.graph.encoder.atom_fusion_norm",
            "encoders.graph.encoder.atom_fusion_projection",
            "encoders.graph.encoder.atom_channel_gate",
            "encoders.graph.encoder.o8.md_residual",
            "encoders.graph.norm", "encoders.graph.projection", "mlp",
        ],
        "stage2_trainable": [
            "all formal Warm0 trainable modules: O8 representation body, complete GLT representation stack, atom fusion, MD200 residual, graph output adapter and regression head"
        ],
        "always_disabled": [
            "encoders.graph.encoder.o8.star_distance_bias",
            "encoders.graph.encoder.compact19_residual"
        ],
        "optimizer_scheduler_contract": "Warm5 preserves the formal Warm0 optimizer groups, five-epoch LR warmup, scheduler trajectory, total epoch budget and early-stopping state; only representation-stack gradients and training mode differ during epochs 1-5."
    }
    manifest = {
        "schema": "mts-glt-v2-warm5-vs-warm0-run-manifest-v1",
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "warm0_source": str(warm0_root.relative_to(ROOT)),
        "warm5_source": str(warm5_root.relative_to(ROOT)),
        "formal_o8_source": str(o8_root.relative_to(ROOT)),
        "warm0_provenance_verified_units": 40,
        "paired_prediction_units_verified": 40,
        "checkpoint_selected_by_formal_20k_trajectory": True,
        "checkpoint_step": 5000,
        "scientific_variable": "representation stack frozen and eval during downstream epochs 1-5",
    }

    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / "per_fold_results.csv", index=False)
    tasks.to_csv(output / "task_results.csv", index=False)
    _atomic_text(
        output / "glt_v2_warm5_vs_warm0_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(
        output / "run_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(
        output / "warm_module_contract.json",
        json.dumps(module_contract, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(
        output / "config_snapshot.json",
        json.dumps(config, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
