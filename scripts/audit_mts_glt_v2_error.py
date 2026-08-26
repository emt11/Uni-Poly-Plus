#!/usr/bin/env python3
"""Zero-training XC/EI attribution audit for the formal GLT-v2 Base-5k."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import UniDataset  # noqa: E402

TASKS = ("xc", "ei")
FORMAL = ROOT / "results/mts_glt_v2/downstream/formal/a6_h_w1_20k_probe_005k"
SIDECAR = ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_v1/downstream_union"
SPLITS = ROOT / "data/splits/mips_shared5"
QUANTILE_EDGES = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def finite_correlation(x, y, kind: str) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3 or np.ptp(x[valid]) == 0 or np.ptp(y[valid]) == 0:
        return None
    value = pearsonr(x[valid], y[valid]).statistic if kind == "pearson" else spearmanr(x[valid], y[valid]).statistic
    return float(value) if np.isfinite(value) else None


def make_dataset(task: str):
    return UniDataset(
        root=str(ROOT / "data"), dataset=f"smi_{task}",
        smiles_model_name=str(ROOT / "pretrained_models/encoders/PubChem10M_SMILES_BPE_450k"),
        graph_encoder_type="mips_trimer_scage", graph_input="star_linking",
        modalities=("graph",), cache_layers="ru_base,topology,trimer,md200",
        periodic_line_glt_sidecar=str(SIDECAR),
        experiment_id="glt-v2-scientific-reliability-error-audit-v1",
        feature_config_hash="manual", mips_core="paper_corrected",
        mips_max_hops=2, mips_use_descriptors=True,
    )


def structural_rows(task: str) -> dict[int, dict]:
    dataset = make_dataset(task)
    result = {}
    for index in range(len(dataset)):
        item = dataset[index]
        token_valid = item.glt_token_valid.bool().numpy()
        relation_valid = (
            item.glt_relation_valid.bool() & ~item.glt_relation_is_fallback.bool()
        ).numpy()
        endpoints = np.concatenate((
            item.glt_token_atom_a.numpy()[token_valid],
            item.glt_token_atom_b.numpy()[token_valid],
        )) if token_valid.any() else np.empty(0, dtype=np.int64)
        atom_count = int(item.canonical_atom_count)
        result[index] = {
            "atom_count": atom_count,
            "line_count": int(token_valid.sum()),
            "angle_relation_count": int(relation_valid.sum()),
            "cross_ru_line_count": int(
                np.count_nonzero(token_valid & (item.glt_token_shift.abs().numpy() == 1))
            ),
            "cross_ru_fraction": float(
                np.count_nonzero(token_valid & (item.glt_token_shift.abs().numpy() == 1))
                / max(1, int(token_valid.sum()))
            ),
            "geometry_valid_fraction": float(
                len(np.unique(endpoints)) / max(1, atom_count)
            ),
            "graph_geometry_valid": bool(item.glt_geometry_valid),
        }
    return result


def load_predictions(task: str, mode: str, fold: int):
    path = FORMAL / mode / "predictions/42" / task / f"fold_{fold}.npz"
    if not path.is_file():
        return None, path
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(np.asarray(payload["metadata"]).item()))
        arrays = {
            "target": np.asarray(payload["y_true"], dtype=np.float64),
            "prediction": np.asarray(payload["y_pred"], dtype=np.float64),
            "sample_id": np.asarray(payload["sample_indices"], dtype=np.int64),
        }
    if metadata.get("task") != task or int(metadata.get("fold", -1)) != fold or int(metadata.get("seed", -1)) != 42:
        raise RuntimeError(f"prediction metadata mismatch: {path}")
    return arrays, path


def distribution(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
        "min": float(values.min()), "p05": float(np.quantile(values, .05)),
        "p25": float(np.quantile(values, .25)), "p50": float(np.quantile(values, .5)),
        "p75": float(np.quantile(values, .75)), "p95": float(np.quantile(values, .95)),
        "max": float(values.max()),
    }


def run_task(task: str):
    raw = pd.read_csv(ROOT / f"data/raw/smi_{task}.csv")
    label_column = next(column for column in raw.columns if column != "smiles")
    labels = raw[label_column].to_numpy(dtype=np.float64)
    structures = structural_rows(task)
    split = json.loads((SPLITS / f"{task}.json").read_text())
    sample_rows, fold_rows, shift_rows = [], [], []
    sources = []
    for fold in range(5):
        o8, o8_path = load_predictions(task, "o8_only", fold)
        fused, fused_path = load_predictions(task, "o8_glt_atom", fold)
        if o8 is None or fused is None:
            fold_rows.append({"task": task, "fold": fold, "available": False})
            continue
        if not (
            np.array_equal(o8["sample_id"], fused["sample_id"])
            and np.array_equal(o8["target"], fused["target"])
        ):
            raise RuntimeError(f"O8/fused prediction identity mismatch: {task} fold {fold}")
        ids, target = o8["sample_id"], o8["target"]
        # Predictions were round-tripped through the target scaler in float32.
        if not np.allclose(target, labels[ids], rtol=0.0, atol=5e-5):
            raise RuntimeError(f"raw label/prediction mismatch: {task} fold {fold}")
        sources.extend((str(o8_path), str(fused_path)))
        fold_row = {"task": task, "fold": fold, "available": True}
        for name, values in (("o8", o8["prediction"]), ("fused", fused["prediction"])):
            slope, intercept = np.polyfit(target, values, 1)
            fold_row.update({
                f"{name}_intercept": float(intercept), f"{name}_slope": float(slope),
                f"{name}_pearson": finite_correlation(target, values, "pearson"),
                f"{name}_target_mean": float(target.mean()),
                f"{name}_target_std": float(target.std()),
                f"{name}_prediction_mean": float(values.mean()),
                f"{name}_prediction_std": float(values.std()),
                f"{name}_prediction_target_std_ratio": float(values.std() / max(target.std(), 1e-12)),
            })
        fold_rows.append(fold_row)
        manifest_fold = next(value for value in split["folds"] if int(value["fold"]) == fold)
        train = labels[np.asarray(manifest_fold["train_indices"], dtype=np.int64)]
        test = labels[np.asarray(manifest_fold["test_indices"], dtype=np.int64)]
        train_summary, test_summary = distribution(train), distribution(test)
        shift_rows.append({
            "task": task, "fold": fold,
            **{f"train_{key}": value for key, value in train_summary.items()},
            **{f"test_{key}": value for key, value in test_summary.items()},
            "mean_shift": float(test.mean() - train.mean()),
            "std_ratio": float(test.std() / max(train.std(), 1e-12)),
            "test_below_train_min_fraction": float(np.mean(test < train.min())),
            "test_above_train_max_fraction": float(np.mean(test > train.max())),
        })
        nearest = np.min(np.abs(target[:, None] - train[None, :]), axis=1)
        for position, sample_id in enumerate(ids):
            err_o8 = abs(target[position] - o8["prediction"][position])
            err_fused = abs(target[position] - fused["prediction"][position])
            sample_rows.append({
                "task": task, "fold": fold, "sample_id": int(sample_id),
                "target": float(target[position]), "pred_o8": float(o8["prediction"][position]),
                "pred_fused": float(fused["prediction"][position]),
                "err_o8": float(err_o8), "err_fused": float(err_fused),
                "geometry_correction": float(err_o8 - err_fused),
                "nearest_train_label_distance": float(nearest[position]),
                **structures[int(sample_id)],
            })
    if len({row["sample_id"] for row in sample_rows}) != len(labels):
        raise RuntimeError(f"five folds do not cover every {task} sample exactly once")
    return labels, sample_rows, fold_rows, shift_rows, sources


def summarize_task(task: str, sample_rows: list[dict], fold_rows: list[dict]):
    rows = [row for row in sample_rows if row["task"] == task]
    folds = [row for row in fold_rows if row["task"] == task and row.get("available")]
    target = np.asarray([row["target"] for row in rows])
    correction = np.asarray([row["geometry_correction"] for row in rows])
    order = np.argsort(target)
    quantile_rows = []
    for left, right in zip(QUANTILE_EDGES[:-1], QUANTILE_EDGES[1:]):
        start, stop = int(round(left * len(rows))), int(round(right * len(rows)))
        selected = [rows[index] for index in order[start:stop]]
        entry = {"task": task, "quantile": f"{left:.2f}-{right:.2f}", "n": len(selected)}
        for name in ("o8", "fused"):
            residual = np.asarray([row[f"pred_{name}"] - row["target"] for row in selected])
            absolute = np.abs(residual)
            entry.update({
                f"{name}_mean_residual": float(residual.mean()),
                f"{name}_mae": float(absolute.mean()),
                f"{name}_rmse": float(np.sqrt(np.mean(residual ** 2))),
            })
        entry["fused_minus_o8_mae"] = entry["fused_mae"] - entry["o8_mae"]
        quantile_rows.append(entry)
    correlations = []
    for field in (
        "target", "atom_count", "line_count", "angle_relation_count",
        "cross_ru_fraction", "geometry_valid_fraction",
    ):
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        correlations.append({
            "task": task, "field": field,
            "pearson": finite_correlation(correction, values, "pearson"),
            "spearman": finite_correlation(correction, values, "spearman"),
        })
    nearest = np.asarray([row["nearest_train_label_distance"] for row in rows])
    for error_field in ("err_o8", "err_fused"):
        values = np.asarray([row[error_field] for row in rows])
        correlations.append({
            "task": task, "field": f"nearest_train_label_distance_vs_{error_field}",
            "pearson": finite_correlation(nearest, values, "pearson"),
            "spearman": finite_correlation(nearest, values, "spearman"),
        })
    k = max(1, int(round(.10 * len(rows))))
    ranked = np.argsort(correction)
    group_summary = {}
    for name, indices in (("top_helped", ranked[-k:]), ("top_hurt", ranked[:k])):
        group_summary[name] = {
            field: {"mean": float(np.mean([rows[i][field] for i in indices])), "median": float(np.median([rows[i][field] for i in indices]))}
            for field in ("target", "atom_count", "line_count", "angle_relation_count", "cross_ru_fraction", "geometry_valid_fraction")
        }
    low, high = quantile_rows[0], quantile_rows[-1]
    shrink_consistent = all(row["fused_prediction_target_std_ratio"] < 1 for row in folds)
    extreme_direction = low["fused_mean_residual"] > 0 and high["fused_mean_residual"] < 0
    extreme_label = "YES" if shrink_consistent and extreme_direction else ("MIXED" if shrink_consistent or extreme_direction else "NO")
    shifts = [row for row in folds]
    # Label support is considered mixed unless all folds exhibit out-of-range test labels.
    shift_evidence = []
    # populated by caller from fold-label table; kept descriptive here
    broad_fraction = float(np.mean(correction > 0))
    complementarity = "BROAD" if broad_fraction > .5 else ("LOCALIZED" if np.any(correction > 0) else "WEAK")
    return {
        "task": task,
        "folds_available": len(folds),
        "prediction_target_std_ratio_o8": float(np.mean([row["o8_prediction_target_std_ratio"] for row in folds])),
        "prediction_target_std_ratio_fused": float(np.mean([row["fused_prediction_target_std_ratio"] for row in folds])),
        "slope_o8": float(np.mean([row["o8_slope"] for row in folds])),
        "slope_fused": float(np.mean([row["fused_slope"] for row in folds])),
        "mean_geometry_correction": float(correction.mean()),
        "fraction_fused_helps": broad_fraction,
        "extreme_quantile_bias": {"low": low["fused_mean_residual"], "high": high["fused_mean_residual"]},
        "EXTREME_SHRINKAGE": extreme_label,
        "GLT_SAMPLE_COMPLEMENTARITY": complementarity,
        "top_bottom_characteristics": group_summary,
        "quantile_rows": quantile_rows,
        "correlations": correlations,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    all_samples, all_folds, all_shifts, sources = [], [], [], []
    summaries, all_quantiles, all_correlations = {}, [], []
    for task in TASKS:
        _labels, samples, folds, shifts, task_sources = run_task(task)
        all_samples.extend(samples); all_folds.extend(folds); all_shifts.extend(shifts); sources.extend(task_sources)
        summary = summarize_task(task, samples, folds)
        all_quantiles.extend(summary.pop("quantile_rows")); all_correlations.extend(summary.pop("correlations"))
        task_shifts = [row for row in shifts if row["task"] == task]
        out_of_range = [row["test_below_train_min_fraction"] + row["test_above_train_max_fraction"] for row in task_shifts]
        summary["fold_label_shift"] = {"mean_out_of_train_range_fraction": float(np.mean(out_of_range)), "per_fold": out_of_range}
        summary[f"{task.upper()}_FOLD_SHIFT"] = "YES" if all(value > 0 for value in out_of_range) else ("MIXED" if any(value > 0 for value in out_of_range) else "NO")
        summaries[task] = summary
    atomic_csv(args.output_dir / "sample_geometry_corrections.csv", all_samples)
    atomic_csv(args.output_dir / "per_fold_error_statistics.csv", all_folds)
    atomic_csv(args.output_dir / "quantile_error_statistics.csv", all_quantiles)
    atomic_csv(args.output_dir / "fold_label_shift.csv", all_shifts)
    atomic_csv(args.output_dir / "structural_correlation.csv", all_correlations)
    atomic_json(args.output_dir / "error_audit_summary.json", {
        "schema": "mts-glt-v2-error-attribution-v1",
        "tasks": summaries,
        "prediction_sources": sorted(set(sources)),
        "checkpoint": str((ROOT / "results/mts_glt_v2/formal/a6_h_w1_20k/mts_glt_v2_probe_005k.pth").resolve()),
        "seed": 42, "protocol": "historical_shared5",
        "note": "Descriptive zero-training audit; labels are not hypothesis-test claims.",
    })


if __name__ == "__main__":
    main()
