#!/usr/bin/env python3
"""Validate and summarize the paired GraphGate readout screen."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/mts_glt_graphgate_v1/readout_causal_screen_v1"
LOGS = ROOT / "logs/mts_glt_graphgate_v1/readout_causal_screen_v1"
GQ = ROOT / "results/mts_glt_graphgate_v1/fusion_realization_v1/folds"
CHECKPOINT = str((ROOT / "pretrained_models/mts_glt_graphgate_v1/mts_glt_graphgate_v1.pth").resolve())
CONFIG = ROOT / "configs/mts/glt_graphgate_v1/readout_causal_screen_v1.json"
TASKS = ("xc", "ei", "eea")
ARMS = {
    "GQ": (GQ, "o8_glt_graph"),
    "GM": (OUTPUT / "GM", "o8_glt_graph_mean"),
    "AT": (OUTPUT / "AT", "o8_glt_atom_central"),
}


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path, payload):
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _unit(root, mode, task, fold):
    shard = root / "shards/42" / task / f"fold_{fold}.csv"
    prediction = root / "predictions/42" / task / f"fold_{fold}.npz"
    if not shard.is_file() or not prediction.is_file():
        raise SystemExit(f"missing unit: {shard} / {prediction}")
    frame = pd.read_csv(shard)
    if len(frame) != 1:
        raise SystemExit(f"expected one shard row: {shard}")
    row = frame.iloc[0]
    expected = {
        "task": task,
        "checkpoint_path": CHECKPOINT,
        "mts_glt_mode": mode,
        "mts_glt_geometry_mode": "full",
        "mts_glt_fusion_strategy": "fusion_warm",
        "evaluation_protocol": "historical_shared5",
        "amp_dtype": "fp32",
        "regression_loss": "huber",
    }
    for key, value in expected.items():
        if str(row.get(key)) != str(value):
            raise SystemExit(
                f"contract mismatch {shard}: {key}={row.get(key)!r}, expected {value!r}"
            )
    numeric = {
        "seed": 42, "train_batch_size": 32, "eval_batch_size": 64,
        "mts_glt_fusion_warm_epochs": 5,
    }
    for key, value in numeric.items():
        if int(row.get(key)) != value:
            raise SystemExit(f"contract mismatch {shard}: {key}")
    if not np.isclose(float(row["mts_glt_initial_alpha"]), 0.05):
        raise SystemExit(f"alpha mismatch: {shard}")
    if not np.isclose(float(row["huber_beta"]), 0.5):
        raise SystemExit(f"Huber mismatch: {shard}")
    metrics = json.loads(str(row["per_fold_metrics"]))
    if len(metrics) != 1 or int(metrics[0]["fold"]) != fold:
        raise SystemExit(f"fold metadata mismatch: {shard}")
    with np.load(prediction, allow_pickle=False) as payload:
        metadata = json.loads(str(np.asarray(payload["metadata"]).item()))
        arrays = {
            "y_true": np.asarray(payload["y_true"]),
            "y_pred": np.asarray(payload["y_pred"]),
            "sample_indices": np.asarray(payload["sample_indices"]),
        }
    if metadata["task"] != task or int(metadata["fold"]) != fold:
        raise SystemExit(f"prediction metadata mismatch: {prediction}")
    return {
        "r2": float(row["avg_test_r2"]),
        "fold_seed": int(metadata["fold_seed"]),
        **arrays,
    }


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _contrast(task_frame, fold_frame, name):
    task_values = task_frame[name].to_numpy(dtype=np.float64)
    fold_values = fold_frame[name].to_numpy(dtype=np.float64)
    return {
        "macro": float(task_values.mean()),
        "median_task": float(np.median(task_values)),
        "positive_tasks": int((task_values > 0).sum()),
        "positive_folds": int((fold_values > 0).sum()),
        "fold_distribution": _distribution(fold_values),
    }


def main():
    rows = []
    for task in TASKS:
        for fold in range(3):
            units = {
                arm: _unit(root, mode, task, fold)
                for arm, (root, mode) in ARMS.items()
            }
            reference = units["GQ"]
            for arm in ("GM", "AT"):
                candidate = units[arm]
                if candidate["fold_seed"] != reference["fold_seed"]:
                    raise SystemExit(f"fold seed mismatch: {task}/fold{fold}/{arm}")
                if not np.array_equal(candidate["sample_indices"], reference["sample_indices"]):
                    raise SystemExit(f"sample index mismatch: {task}/fold{fold}/{arm}")
                if not np.array_equal(candidate["y_true"], reference["y_true"]):
                    raise SystemExit(f"target mismatch: {task}/fold{fold}/{arm}")
            gq, gm, at = (units[name]["r2"] for name in ("GQ", "GM", "AT"))
            rows.append({
                "task": task, "fold": fold,
                "fold_seed": reference["fold_seed"],
                "GQ_R2": gq, "GM_R2": gm, "AT_R2": at,
                "DeltaQuery": gm - gq,
                "DeltaAtom": at - gm,
                "DeltaTotal": at - gq,
            })
    folds = pd.DataFrame(rows).sort_values(["task", "fold"])
    tasks = folds.groupby("task", sort=True)[
        ["GQ_R2", "GM_R2", "AT_R2", "DeltaQuery", "DeltaAtom", "DeltaTotal"]
    ].mean().reset_index()
    query = _contrast(tasks, folds, "DeltaQuery")
    atom = _contrast(tasks, folds, "DeltaAtom")
    total = _contrast(tasks, folds, "DeltaTotal")
    atom_positive = bool(
        atom["macro"] > 0 and atom["median_task"] > 0
        and atom["positive_tasks"] >= 2
    )
    total_go = bool(
        total["macro"] > 0 and total["median_task"] > 0
        and total["positive_tasks"] >= 2
    )
    summary = {
        "schema": "mts-glt-graphgate-readout-causal-screen-v1",
        "checkpoint": CHECKPOINT,
        "evaluation_protocol": "historical_shared5",
        "independent_blind_test": False,
        "reused_gq_runs": 9,
        "new_gm_runs": 9,
        "new_at_runs": 9,
        "failed_runs": 0,
        "macro3": {
            "GQ": float(tasks.GQ_R2.mean()),
            "GM": float(tasks.GM_R2.mean()),
            "AT": float(tasks.AT_R2.mean()),
        },
        "DeltaQuery": query,
        "DeltaAtom": atom,
        "DeltaTotal": total,
        "macro3_delta_query": query["macro"],
        "median_delta_query": query["median_task"],
        "positive_delta_query_tasks": query["positive_tasks"],
        "positive_delta_query_folds": query["positive_folds"],
        "macro3_delta_atom": atom["macro"],
        "median_delta_atom": atom["median_task"],
        "positive_delta_atom_tasks": atom["positive_tasks"],
        "positive_delta_atom_folds": atom["positive_folds"],
        "macro3_delta_total": total["macro"],
        "median_delta_total": total["median_task"],
        "positive_delta_total_tasks": total["positive_tasks"],
        "positive_delta_total_folds": total["positive_folds"],
        "atom_total_decision": "GO_candidate" if total_go else "STOP_candidate",
        "atom_alignment_readout_decision": (
            "POSITIVE" if atom_positive else "NOT_ESTABLISHED"
        ),
        "tasks": tasks.to_dict("records"),
    }
    manifest = {
        "schema": summary["schema"],
        "checkpoint": CHECKPOINT,
        "config": str(CONFIG.relative_to(ROOT)),
        "reference_GQ": str(GQ.relative_to(ROOT)),
        "new_results": {
            arm: str(root.relative_to(ROOT))
            for arm, (root, _) in ARMS.items() if arm != "GQ"
        },
        "logs": str(LOGS.relative_to(ROOT)),
        "units": 27,
        "reused_units": 9,
        "new_units": 18,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    folds.to_csv(OUTPUT / "per_fold_results.csv", index=False)
    tasks.to_csv(OUTPUT / "task_results.csv", index=False)
    _atomic_json(OUTPUT / "readout_screen_summary.json", summary)
    _atomic_json(OUTPUT / "run_manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
