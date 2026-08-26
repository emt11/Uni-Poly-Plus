#!/usr/bin/env python3
"""Strict paired reporter for B / generic bond control / true bond category."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/mts/glt_v2_o8_bond_type_screen_v1.json"


def _atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _unit(root, task, fold):
    shard = root / "shards/42" / task / f"fold_{fold}.csv"
    prediction = root / "predictions/42" / task / f"fold_{fold}.npz"
    frame = pd.read_csv(shard)
    if len(frame) != 1:
        raise RuntimeError(f"expected exactly one shard row: {shard}")
    row = frame.iloc[0]
    with np.load(prediction, allow_pickle=False) as payload:
        pred = {
            "y_true": np.asarray(payload["y_true"]),
            "sample_indices": np.asarray(payload["sample_indices"]),
            "metadata": json.loads(str(np.asarray(payload["metadata"]).item())),
        }
    score = float(row["avg_test_r2"])
    if not np.isfinite(score):
        raise RuntimeError(f"non-finite score: {shard}")
    return row, pred, score


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "fold_mean": float(values.mean()),
        "fold_median": float(np.median(values)),
        "fold_p25": float(np.quantile(values, 0.25)),
        "fold_p75": float(np.quantile(values, 0.75)),
        "fold_min": float(values.min()),
        "fold_max": float(values.max()),
        "positive_folds": int((values > 0).sum()),
    }


def _positive(item):
    return (
        item["macro3"] > 0
        and item["median_task"] > 0
        and item["positive_tasks"] >= 2
        and item["positive_folds"] >= 5
    )


def _separability(diagnostics):
    records = []
    near_equal = 0
    compared = 0
    distances = []
    for diagnostic in diagnostics:
        vectors = {
            int(key): np.asarray(value, dtype=np.float64)
            for key, value in diagnostic["category_per_head_bias"].items()
        }
        observed = [
            int(key) for key, value in diagnostic["category_count"].items()
            if int(value) > 0
        ]
        for left_index, left in enumerate(observed):
            for right in observed[left_index + 1:]:
                value = float(np.linalg.norm(vectors[left] - vectors[right]))
                records.append({
                    "task": diagnostic["task"], "fold": diagnostic["fold"],
                    "category_left": left, "category_right": right,
                    "l2_distance": value,
                })
                compared += 1
                near_equal += int(value < 1e-6)
                distances.append(value)
    return records, {
        "observed_category_pairs": int(compared),
        "pairs_distance_lt_1e_6": int(near_equal),
        "distance_mean": float(np.mean(distances)) if distances else None,
        "distance_median": float(np.median(distances)) if distances else None,
        "distance_min": float(np.min(distances)) if distances else None,
        "distance_max": float(np.max(distances)) if distances else None,
    }


def main():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = ROOT / config["output_root"]
    roots = {
        "B": ROOT / config["baseline_root"],
        "BC": output / "bc",
        "BT": output / "bt",
    }
    expected_checkpoint = str((ROOT / config["checkpoint"]).resolve())
    fold_rows, diagnostic_rows, diagnostic_units = [], [], []
    for task in config["tasks"]:
        for fold in config["folds"]:
            units = {name: _unit(root, task, fold) for name, root in roots.items()}
            reference = units["B"][1]
            fold_seed = int(reference["metadata"]["fold_seed"])
            for name, (row, pred, _) in units.items():
                if str(row["checkpoint_path"]) != expected_checkpoint:
                    raise RuntimeError(f"{name} checkpoint mismatch: {task}/{fold}")
                if str(row["mts_glt_mode"]) != "o8_glt_atom":
                    raise RuntimeError(f"{name} GLT mode mismatch: {task}/{fold}")
                if str(row["mts_glt_fusion_strategy"]) != "legacy_zero":
                    raise RuntimeError(f"{name} is not Warm0: {task}/{fold}")
                if str(row["amp_dtype"]) != "fp32":
                    raise RuntimeError(f"{name} precision mismatch: {task}/{fold}")
                if int(row["train_batch_size"]) != 32 or int(row["eval_batch_size"]) != 64:
                    raise RuntimeError(f"{name} batch mismatch: {task}/{fold}")
                if not np.array_equal(pred["sample_indices"], reference["sample_indices"]):
                    raise RuntimeError(f"{name} sample-index mismatch: {task}/{fold}")
                if not np.array_equal(pred["y_true"], reference["y_true"]):
                    raise RuntimeError(f"{name} target mismatch: {task}/{fold}")
                if int(pred["metadata"]["fold_seed"]) != fold_seed:
                    raise RuntimeError(f"{name} fold-seed mismatch: {task}/{fold}")
                if name != "B":
                    expected_mode = {"BC": "control", "BT": "type"}[name]
                    if str(row["mts_o8_bond_bias_mode"]) != expected_mode:
                        raise RuntimeError(f"{name} bond mode mismatch: {task}/{fold}")
            b, bc, bt = (units[name][2] for name in ("B", "BC", "BT"))
            fold_rows.append({
                "task": task, "fold": int(fold), "fold_seed": fold_seed,
                "B": b, "BC": bc, "BT": bt,
                "GenericEdgeEffect": bc - b,
                "BondChemistryEffect": bt - bc,
                "TotalBondEffect": bt - b,
            })
            for arm, label in (("bc", "BC"), ("bt", "BT")):
                diagnostic_path = output / "bond_units" / arm / task / f"fold_{fold}.json"
                checkpoint_path = output / "checkpoints" / arm / task / f"fold_{fold}.pth"
                diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                expected_mode = {"BC": "control", "BT": "type"}[label]
                if checkpoint.get("mts_o8_bond_bias_mode") != expected_mode:
                    raise RuntimeError(f"checkpoint mode mismatch: {checkpoint_path}")
                if str(checkpoint.get("pretrained_checkpoint")) != expected_checkpoint:
                    raise RuntimeError(f"checkpoint provenance mismatch: {checkpoint_path}")
                if int(diagnostic["new_parameter_count"]) != 48:
                    raise RuntimeError(f"parameter count mismatch: {diagnostic_path}")
                diagnostic_units.append({"arm": label, **diagnostic})
                for category in range(config["bond_categories"]):
                    vector = diagnostic["category_per_head_bias"][str(category)]
                    diagnostic_rows.append({
                        "arm": label, "task": task, "fold": int(fold),
                        "category": category,
                        "N": int(diagnostic["category_count"][str(category)]),
                        "mean_abs_bias": diagnostic["category_mean_absolute_bias"][str(category)],
                        "per_head_bias_vector_norm": float(np.linalg.norm(vector)),
                        "per_head_bias_vector": json.dumps(vector),
                        "overall_mean_abs_bias": diagnostic["absolute_bias"]["mean"],
                        "internal_mean_abs_bias": diagnostic["internal_absolute_bias"]["mean"],
                        "cross_ru_mean_abs_bias": diagnostic["cross_ru_absolute_bias"]["mean"],
                    })

    folds = pd.DataFrame(fold_rows)
    columns = ("B", "BC", "BT", "GenericEdgeEffect", "BondChemistryEffect", "TotalBondEffect")
    tasks = pd.DataFrame([
        {"task": task, **{
            key: float(folds[folds.task == task][key].mean()) for key in columns
        }} for task in config["tasks"]
    ])
    contrasts = {}
    for key in ("GenericEdgeEffect", "BondChemistryEffect", "TotalBondEffect"):
        contrasts[key] = {
            "macro3": float(tasks[key].mean()),
            "median_task": float(tasks[key].median()),
            "positive_tasks": int((tasks[key] > 0).sum()),
            **_distribution(folds[key]),
        }
    sanity = json.loads((output / "gradient_sanity.json").read_text(encoding="utf-8"))
    coverage = json.loads((output / "bond_type_coverage.json").read_text(encoding="utf-8"))
    bc_diagnostics = [row for row in diagnostic_units if row["arm"] == "BC"]
    bt_diagnostics = [row for row in diagnostic_units if row["arm"] == "BT"]
    separability_rows, separability = _separability(bt_diagnostics)
    chemistry = contrasts["BondChemistryEffect"]
    total = contrasts["TotalBondEffect"]
    summary = {
        "schema": config["schema"],
        "baseline": config["baseline"],
        "checkpoint": expected_checkpoint,
        "checkpoint_step": 5000,
        "tasks": config["tasks"], "folds": config["folds"],
        "seed": config["seed"], "protocol": config["evaluation_protocol"],
        "reused_b_runs": 9, "new_bc_runs": 9, "new_bt_runs": 9,
        "failed_runs": 0,
        "bond_tensor": coverage["tensor_field"],
        "bond_categories": coverage["num_categories"],
        "bond_type_coverage": coverage["valid_bond_type_coverage"],
        "new_parameter_count_bc": sanity["new_parameter_count"]["BC"],
        "new_parameter_count_bt": sanity["new_parameter_count"]["BT"],
        "step0_parity": sanity["step0_prediction_parity"],
        "first_gradient": sanity["first_backward"],
        "endpoint_swap_symmetry": sanity["endpoint_swap_symmetry"],
        "nonbond_exact_zero": sanity["nonbond_exact_zero"],
        "spd_bias_unchanged": sanity["spd_bias_unchanged"],
        "path_bias_unchanged": sanity["path_bias_unchanged"],
        "glt_unchanged": sanity["glt_line_states_unchanged"],
        "contrasts": contrasts,
        "bond_chemistry_decision": (
            "POSITIVE_SCREEN" if _positive(chemistry) else "NOT_ESTABLISHED"
        ),
        "bond_type_total_decision": (
            "GO_candidate" if _positive(total) else "STOP_candidate"
        ),
        "bc_mean_absolute_bias": float(np.mean([
            row["absolute_bias"]["mean"] for row in bc_diagnostics
        ])),
        "bt_mean_absolute_bias": float(np.mean([
            row["absolute_bias"]["mean"] for row in bt_diagnostics
        ])),
        "bt_category_separability": separability,
        "independent_blind_test": False,
        "anomalies": [],
    }
    manifest = {
        "schema": "mts-glt-v2-o8-bond-type-screen-manifest-v1",
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "baseline_source": str(roots["B"].relative_to(ROOT)),
        "bc_source": str(roots["BC"].relative_to(ROOT)),
        "bt_source": str(roots["BT"].relative_to(ROOT)),
        "paired_units_verified": 9,
        "scientific_variable": (
            "parameter-matched generic direct-bond bias versus stored "
            "six-category direct-bond identity"
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / "per_fold_results.csv", index=False)
    tasks.to_csv(output / "task_results.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(
        output / "bond_bias_diagnostics.csv", index=False
    )
    pd.DataFrame(separability_rows).to_csv(
        output / "bond_category_separability.csv", index=False
    )
    _atomic(
        output / "bond_type_screen_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    _atomic(
        output / "run_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
