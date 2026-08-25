#!/usr/bin/env python3
"""Report the matched 5k Full/Off FusionWarm causal interaction."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/mts/glt_graphgate_v1/matched_fusionwarm_causal_v1.json"
OUTPUT = ROOT / "results/mts_glt_graphgate_v1/trimer_validation_v1/matched_5k/fusionwarm_causal_v1"
HISTORICAL = ROOT / "results/mts_glt_graphgate_v1/trimer_validation_v1/matched_5k/neural"
TASKS = ("xc", "ei", "egc")
FOLDS = (0, 1, 2)
HISTORICAL_CHECKPOINTS = {
    arm: ROOT / f"pretrained_models/mts_glt_graphgate_v1/trimer_validation_v1/{arm}_5k.pth"
    for arm in ("full", "off")
}


def _atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _root(arm, mode, task):
    if mode == "o8_only" and task in {"xc", "ei"}:
        return HISTORICAL / f"trimer_validation_v1_matched_5k_{arm}" / "o8_only"
    leaf = "fusionwarm" if mode == "o8_glt_graph" else "o8"
    return OUTPUT / arm / leaf


def _prediction(arm, mode, task, fold, expected_checkpoint):
    root = _root(arm, mode, task)
    path = root / f"predictions/42/{task}/fold_{fold}.npz"
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        indices = np.asarray(data["sample_indices"], dtype=np.int64)
        target = np.asarray(data["y_true"], dtype=np.float64)
        prediction = np.asarray(data["y_pred"], dtype=np.float64)
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise RuntimeError(f"non-finite prediction: {path}")
    denominator = np.square(target - target.mean()).sum()
    r2 = 1.0 - np.square(target - prediction).sum() / denominator
    shard = root / f"shards/42/{task}/fold_{fold}.csv"
    with shard.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"expected one result row: {shard}")
    checkpoint = Path(rows[0]["checkpoint_path"]).resolve()
    allowed_checkpoints = {expected_checkpoint.resolve()}
    if mode == "o8_only" and task in {"xc", "ei"}:
        allowed_checkpoints.add(HISTORICAL_CHECKPOINTS[arm].resolve())
    if checkpoint not in allowed_checkpoints:
        raise RuntimeError(f"checkpoint mismatch: {shard}: {checkpoint}")
    expected_strategy = "fusion_warm" if mode == "o8_glt_graph" else "legacy_zero"
    if rows[0]["mts_glt_fusion_strategy"] != expected_strategy:
        raise RuntimeError(f"fusion strategy mismatch: {shard}")
    if rows[0]["mts_glt_geometry_mode"] != arm:
        raise RuntimeError(f"geometry arm mismatch: {shard}")
    return {
        "r2": float(r2), "fold_seed": int(metadata["fold_seed"]),
        "indices": indices, "target": target,
    }


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    checkpoints = {
        arm: (ROOT / config[f"{arm}_checkpoint"]).resolve()
        for arm in ("full", "off")
    }
    for arm, checkpoint in checkpoints.items():
        current = torch.load(checkpoint, map_location="cpu", weights_only=True)
        historical = torch.load(
            HISTORICAL_CHECKPOINTS[arm], map_location="cpu", weights_only=True
        )
        current_namespaces = current["namespaces"]
        historical_namespaces = historical["namespaces"]
        if set(current_namespaces) != set(historical_namespaces):
            raise RuntimeError(f"historical matched namespace mismatch: {arm}")
        for namespace in current_namespaces:
            if set(current_namespaces[namespace]) != set(historical_namespaces[namespace]):
                raise RuntimeError(f"historical matched state-key mismatch: {arm}/{namespace}")
            if any(
                not torch.equal(value, historical_namespaces[namespace][key])
                for key, value in current_namespaces[namespace].items()
            ):
                raise RuntimeError(f"historical matched tensor mismatch: {arm}/{namespace}")
    per_fold = []
    for task in TASKS:
        for fold in FOLDS:
            units = {
                f"{arm}_{mode}": _prediction(
                    arm, mode, task, fold, checkpoints[arm]
                )
                for arm in ("full", "off")
                for mode in ("o8_only", "o8_glt_graph")
            }
            reference = units["full_o8_only"]
            for name, unit in units.items():
                if unit["fold_seed"] != reference["fold_seed"]:
                    raise RuntimeError(f"fold seed mismatch: {task}/fold{fold}/{name}")
                if not np.array_equal(unit["indices"], reference["indices"]):
                    raise RuntimeError(f"sample-index mismatch: {task}/fold{fold}/{name}")
                if not np.array_equal(unit["target"], reference["target"]):
                    raise RuntimeError(f"target mismatch: {task}/fold{fold}/{name}")
            full_o8 = units["full_o8_only"]["r2"]
            full_fw = units["full_o8_glt_graph"]["r2"]
            off_o8 = units["off_o8_only"]["r2"]
            off_fw = units["off_o8_glt_graph"]["r2"]
            delta_full, delta_off = full_fw - full_o8, off_fw - off_o8
            per_fold.append({
                "task": task.upper(), "fold": fold,
                "fold_seed": reference["fold_seed"],
                "full_o8_r2": full_o8, "full_fw_r2": full_fw,
                "off_o8_r2": off_o8, "off_fw_r2": off_fw,
                "delta_full": delta_full, "delta_off": delta_off,
                "interaction": delta_full - delta_off,
                "full_fw_minus_off_fw": full_fw - off_fw,
            })

    per_task = {}
    for task in (value.upper() for value in TASKS):
        rows = [row for row in per_fold if row["task"] == task]
        per_task[task] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in (
                "full_o8_r2", "full_fw_r2", "delta_full", "off_o8_r2",
                "off_fw_r2", "delta_off", "interaction", "full_fw_minus_off_fw",
            )
        }
    task_interactions = np.asarray(
        [per_task[task]["interaction"] for task in per_task], dtype=np.float64
    )
    fold_interactions = np.asarray(
        [row["interaction"] for row in per_fold], dtype=np.float64
    )
    fused_contrast = np.asarray(
        [per_task[task]["full_fw_minus_off_fw"] for task in per_task],
        dtype=np.float64,
    )
    macro = float(task_interactions.mean())
    median = float(np.median(task_interactions))
    positive_tasks = int((task_interactions > 0).sum())
    decision = bool(macro > 0 and median > 0 and positive_tasks >= 2)
    payload = {
        "protocol": "matched_5k_full_off_fusionwarm_causal_v1",
        "seed": 42,
        "tasks": [task.upper() for task in TASKS],
        "folds": list(FOLDS),
        "checkpoints": {key: str(value) for key, value in checkpoints.items()},
        "per_fold": per_fold,
        "per_task": per_task,
        "macro3_interaction": macro,
        "median_task_interaction": median,
        "positive_interaction_tasks": positive_tasks,
        "full_fw_minus_off_fw_macro3": float(fused_contrast.mean()),
        "positive_interaction_folds": int((fold_interactions > 0).sum()),
        "fold_interaction_median": float(np.median(fold_interactions)),
        "fold_interaction_p25": float(np.percentile(fold_interactions, 25)),
        "fold_interaction_p75": float(np.percentile(fold_interactions, 75)),
        "screening_decision": "GO_candidate" if decision else "STOP_candidate",
    }
    _atomic_write(
        OUTPUT / "summary/matched_fusionwarm_causal_summary.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    with (OUTPUT / "summary/per_fold_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_fold[0]))
        writer.writeheader()
        writer.writerows(per_fold)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
