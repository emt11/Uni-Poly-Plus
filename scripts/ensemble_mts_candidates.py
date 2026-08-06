#!/usr/bin/env python3
"""Leakage-safe prediction ensemble for final nested MTS candidates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from summarize_mips_trimer_scage import TASKS, _read_prediction, _read_shard


def _atomic_frame(path: Path, frame: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False, float_format="%.3f")
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output-root", default="results/mts_sota_v3/final_ensemble")
    args = parser.parse_args(argv)
    roots = [Path(value) for value in args.candidate]
    if len(roots) > 3:
        raise SystemExit("the final ensemble accepts at most three candidates")

    fold_rows = []
    correlations = []
    for task in TASKS:
        for fold in range(5):
            candidate_predictions = []
            candidate_errors = []
            validation_rmse = []
            reference = None
            reference_indices = None
            for root in roots:
                seed_predictions = []
                seed_val_rmse = []
                for seed in args.seeds:
                    row = _read_shard(
                        root, seed, task, fold,
                        evaluation_protocol="nested5",
                    )
                    truth, prediction, indices = _read_prediction(
                        root, seed, task, fold, row
                    )
                    metrics = json.loads(str(row["per_fold_metrics"]))[0]
                    val_rmse = float(metrics.get("best_val_rmse", np.nan))
                    if not np.isfinite(val_rmse) or val_rmse <= 0:
                        raise RuntimeError(
                            f"missing inner-validation RMSE: {root}/{task}/{fold}"
                        )
                    if reference is None:
                        reference, reference_indices = truth, indices
                    elif not np.array_equal(indices, reference_indices) or not np.allclose(
                        truth, reference, atol=0.0, rtol=0.0
                    ):
                        raise RuntimeError("candidate outer-test cohorts differ")
                    seed_predictions.append(prediction)
                    seed_val_rmse.append(val_rmse)
                averaged = np.mean(np.stack(seed_predictions), axis=0)
                candidate_predictions.append(averaged)
                candidate_errors.append(averaged - reference)
                validation_rmse.append(float(np.mean(seed_val_rmse)))

            if len(candidate_errors) > 1:
                matrix = np.corrcoef(np.stack(candidate_errors))
                for left in range(len(roots)):
                    for right in range(left + 1, len(roots)):
                        value = float(matrix[left, right])
                        correlations.append(value)
                        if np.isfinite(value) and value >= 0.95:
                            raise RuntimeError(
                                f"candidate error correlation {value:.4f} >= 0.95; "
                                "remove one redundant candidate"
                            )
            weights = 1.0 / np.asarray(validation_rmse, dtype=np.float64)
            weights = np.clip(weights, 0.1, 0.8)
            weights /= weights.sum()
            prediction = np.sum(
                np.stack(candidate_predictions) * weights[:, None], axis=0
            )
            fold_rows.append({
                "task": task,
                "fold": fold,
                "r2": float(r2_score(reference, prediction)),
                "mae": float(mean_absolute_error(reference, prediction)),
                "rmse": float(np.sqrt(mean_squared_error(reference, prediction))),
                "weights": json.dumps({
                    root.name: float(weight)
                    for root, weight in zip(roots, weights)
                }, sort_keys=True),
            })

    folds = pd.DataFrame(fold_rows)
    summary = []
    for task in TASKS:
        subset = folds[folds.task == task]
        row = {"task": task}
        for metric in ("r2", "mae", "rmse"):
            values = subset[metric].to_numpy(float)
            mean, std = float(values.mean()), float(values.std(ddof=1))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_report"] = f"{mean:.3f} ± {std:.3f}"
        summary.append(row)
    macro_by_fold = folds.groupby("fold")["r2"].mean().to_numpy(float)
    summary.append({
        "task": "macro",
        "r2_mean": float(macro_by_fold.mean()),
        "r2_std": float(macro_by_fold.std(ddof=1)),
        "r2_report": (
            f"{macro_by_fold.mean():.3f} ± {macro_by_fold.std(ddof=1):.3f}"
        ),
    })
    output = Path(args.output_root)
    _atomic_frame(output / "fold_metrics.csv", folds)
    _atomic_frame(output / "nested_summary.csv", pd.DataFrame(summary))
    print(json.dumps({
        "candidates": [str(path) for path in roots],
        "macro_r2": summary[-1]["r2_mean"],
        "max_error_correlation": max(correlations) if correlations else None,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
