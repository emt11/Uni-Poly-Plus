#!/usr/bin/env python3
"""Verify and aggregate GLT dual five-fold task shards from frozen predictions.

No aggregate is written until every fold has been re-verified:

* the fold protocol must be ``outer5_inner20``.  A *missing* protocol is
  accepted only with ``--allow-legacy-missing-protocol`` **and** a matching
  formal-shard ``run.json`` **and** a prediction identity match against the
  fixed split manifest; the compatibility is recorded in the output.  A
  protocol that is present but different is always rejected.
* row indices must be integers, unique, and exactly the manifest test set;
  the five folds must cover the task rows exactly once and every
  train/validation/test split must be disjoint.
* R2/MAE/RMSE are recomputed from the saved predictions and must match the
  stored metrics (rtol=1e-6, atol=1e-8); the stored targets must match the raw
  labels (rtol=1e-6, atol=1e-5).

Two R2 conventions are reported separately and are not interchangeable: the
mean of the per-fold R2 values, and the pooled out-of-fold R2 computed once
over all rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.glt_dual_runtime import TASKS, write_json

METRIC_RTOL, METRIC_ATOL = 1e-6, 1e-8
LABEL_RTOL, LABEL_ATOL = 1e-6, 1e-5
FOLD_PROTOCOL = "outer5_inner20"
FORMAL_SHARD_PROTOCOLS = {"outer5_inner20", "outer5_inner20_formal_shard"}


def _digest(values) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def load_task_reference(task: str, split_root: Path, raw_root: Path):
    """Fixed test identities, sample count and raw labels for one task."""

    manifest_path = split_root / f"{task}.json"
    csv_path = raw_root / f"smi_{task}.csv"
    if not manifest_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError(f"missing split manifest or raw csv: {manifest_path}, {csv_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(csv_path)
    smiles = frame.iloc[:, 0].astype(str).str.strip().tolist()
    if manifest.get("protocol") != FOLD_PROTOCOL or manifest.get("task") != task:
        raise ValueError(f"split manifest protocol/task mismatch: {manifest_path}")
    if manifest.get("validation_is_test") is not False or len(manifest.get("folds", [])) != 5:
        raise ValueError(f"split manifest is not a separated five-fold protocol: {manifest_path}")
    if int(manifest["sample_count"]) != len(smiles):
        raise ValueError(f"split manifest sample count differs from raw csv: {manifest_path}")
    if manifest.get("sample_order_hash") != _digest(smiles) and manifest.get(
        "sample_order_sha256"
    ) != _digest(smiles):
        raise ValueError(f"raw csv row order differs from the fixed manifest: {csv_path}")
    labels = frame.iloc[:, 1].to_numpy(dtype=np.float64)
    if not np.isfinite(labels).all():
        raise ValueError(f"nonfinite raw label: {csv_path}")
    return manifest, labels, str(csv_path)


def verify_fold_identity(task, fold, folder, manifest, labels, allow_legacy, shard_root):
    """Return (metrics, accepted_legacy) after every fold-level check passes."""

    metrics_path = folder / "metrics.json"
    predictions_path = folder / "predictions.csv"
    if not metrics_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(f"missing formal shard output: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    accepted_legacy = False
    protocol = metrics.get("protocol")
    if protocol is None:
        if not allow_legacy:
            raise ValueError(
                f"formal shard metrics lack protocol (use --allow-legacy-missing-protocol): {metrics_path}"
            )
        _verify_legacy_formal_shard(task, fold, shard_root, metrics)
        accepted_legacy = True
    elif protocol != FOLD_PROTOCOL:
        raise ValueError(f"formal shard protocol is not {FOLD_PROTOCOL}: {metrics_path}")
    if metrics.get("task") != task or int(metrics.get("fold", -1)) != fold:
        raise ValueError(f"formal shard metadata mismatch: {metrics_path}")

    prediction = pd.read_csv(predictions_path)
    if not {"row_index", "target", "prediction"}.issubset(prediction.columns):
        raise ValueError(f"formal shard prediction table is malformed: {predictions_path}")
    index = prediction["row_index"].to_numpy()
    if index.dtype.kind not in "iu" or prediction["row_index"].duplicated().any():
        raise ValueError(f"formal shard row indices are not unique integers: {predictions_path}")
    target = prediction["target"].to_numpy(dtype=np.float64)
    predicted = prediction["prediction"].to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or not np.isfinite(predicted).all():
        raise ValueError(f"formal shard contains nonfinite prediction: {predictions_path}")
    expected = sorted(int(value) for value in manifest["folds"][fold]["test_indices"])
    if sorted(int(value) for value in index) != expected:
        raise ValueError(f"fold {fold} predictions are not exactly the manifest test set: {task}")
    if not np.allclose(target, labels[index.astype(int)], rtol=LABEL_RTOL, atol=LABEL_ATOL):
        raise ValueError(f"fold {fold} prediction targets differ from raw labels: {task}")

    recomputed = {
        "test_r2": float(r2_score(target, predicted)),
        "test_mae": float(mean_absolute_error(target, predicted)),
        "test_rmse": float(np.sqrt(mean_squared_error(target, predicted))),
    }
    for name, value in recomputed.items():
        stored = metrics.get(name)
        if stored is None:
            raise ValueError(f"formal shard metrics lack {name}: {metrics_path}")
        if not np.isclose(float(stored), value, rtol=METRIC_RTOL, atol=METRIC_ATOL):
            raise ValueError(
                f"stored {name} does not reproduce from predictions: {metrics_path} "
                f"(stored={stored}, recomputed={value})"
            )
    metrics = dict(metrics, **{f"recomputed_{k}": v for k, v in recomputed.items()})
    return metrics, prediction, accepted_legacy


def _verify_legacy_formal_shard(task, fold, shard_root, metrics):
    """Legacy metrics carry no protocol: require the formal-shard run record."""

    run_path = Path(shard_root) / "run.json"
    if not run_path.is_file():
        raise ValueError(f"legacy metrics require a run.json to accept: {run_path}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("smoke") or not run.get("formal_shard"):
        raise ValueError(f"legacy metrics are not from a formal shard: {run_path}")
    if run.get("protocol") not in FORMAL_SHARD_PROTOCOLS:
        raise ValueError(f"legacy metrics run protocol mismatch: {run_path}")
    if list(run.get("selected_tasks", [])) != [task] or [
        int(value) for value in run.get("selected_folds", [])
    ] != [int(fold)]:
        raise ValueError(f"legacy metrics task/fold do not match the shard: {run_path}")
    command = run.get("command")
    if isinstance(command, list):
        command = " ".join(str(part) for part in command)
    if command:
        for required in ("--task", str(task), "--fold", str(fold), "--formal-shard"):
            if required not in command:
                raise ValueError(f"legacy metrics command is inconsistent: {run_path}")
    if metrics.get("smoke") is True:
        raise ValueError(f"legacy metrics are from a smoke run: {shard_root}")


def aggregate(root, output, *, tasks=TASKS, split_root=None, raw_root=None,
              allow_legacy_missing_protocol=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    split_root = Path(split_root).resolve() if split_root else None
    raw_root = Path(raw_root).resolve() if raw_root else None
    if split_root is None or raw_root is None:
        raise ValueError("--split-root and --raw-root are required for verified aggregation")
    rows, summary, legacy_accepted = [], {}, []
    for task in tasks:
        manifest, labels, csv_path = load_task_reference(task, split_root, raw_root)
        fold_metrics, frames = [], []
        for fold in range(5):
            folder = root / f"{task}_fold{fold}" / task / f"fold{fold}"
            metrics, prediction, accepted_legacy = verify_fold_identity(
                task, fold, folder, manifest, labels,
                allow_legacy_missing_protocol, root / f"{task}_fold{fold}",
            )
            if accepted_legacy:
                legacy_accepted.append(f"{task}/fold{fold}")
            train, validation, test = (
                set(manifest["folds"][fold][f"{split}_indices"])
                for split in ("train", "validation", "test")
            )
            if train & validation or train & test or validation & test:
                raise ValueError(f"fold {fold} splits are not disjoint: {task}")
            fold_metrics.append(metrics)
            frames.append(prediction)
        oof = pd.concat(frames).sort_values("row_index")
        if oof["row_index"].tolist() != list(range(len(labels))):
            raise ValueError(f"five folds do not cover the task rows exactly once: {task}")
        target = oof["target"].to_numpy(dtype=np.float64)
        predicted = oof["prediction"].to_numpy(dtype=np.float64)
        folds_mean_r2 = float(np.mean([row["test_r2"] for row in fold_metrics]))
        summary[task] = {
            "fold_count": 5,
            "sample_count": int(manifest["sample_count"]),
            "source_csv": csv_path,
            "folds": [
                {
                    "fold": int(row["fold"]),
                    "test_r2": float(row["test_r2"]),
                    "test_mae": float(row["test_mae"]),
                    "test_rmse": float(row["test_rmse"]),
                    "best_validation_r2": float(row.get("best_validation_r2", float("nan"))),
                    "best_epoch": int(row.get("best_epoch", -1)),
                    "recomputed_test_r2": row["recomputed_test_r2"],
                }
                for row in fold_metrics
            ],
            "test_r2": {"mean": folds_mean_r2,
                        "std": float(np.std([row["test_r2"] for row in fold_metrics], ddof=1)),
                        "convention": "mean of the five per-fold R2 values"},
            "test_mae": {"mean": float(np.mean([row["test_mae"] for row in fold_metrics])),
                         "std": float(np.std([row["test_mae"] for row in fold_metrics], ddof=1))},
            "test_rmse": {"mean": float(np.mean([row["test_rmse"] for row in fold_metrics])),
                          "std": float(np.std([row["test_rmse"] for row in fold_metrics], ddof=1))},
            "pooled_oof": {
                "r2": float(r2_score(target, predicted)),
                "mae": float(mean_absolute_error(target, predicted)),
                "rmse": float(np.sqrt(mean_squared_error(target, predicted))),
                "convention": "single R2 over all rows, one prediction per row",
            },
        }
        task_dir = output / task
        task_dir.mkdir(parents=True, exist_ok=True)
        oof.to_csv(task_dir / "oof.csv", index=False)
        rows.extend(fold_metrics)
    if set(summary) != set(tasks):
        raise ValueError("formal aggregation task coverage mismatch")
    result = {
        "status": "PASS",
        "protocol": FOLD_PROTOCOL,
        "r2_conventions": {
            "folds_mean_r2": "arithmetic mean of the five per-fold R2 values",
            "pooled_oof_r2": "single R2 over all out-of-fold rows",
            "note": "the two values are different estimators and must not be mixed",
        },
        "task_count": len(tasks),
        "fold_count": len(tasks) * 5,
        "tasks": summary,
        "macro8_r2": float(np.mean([summary[task]["test_r2"]["mean"] for task in tasks]))
        if len(tasks) == 8 else None,
        "macro8_pooled_oof_r2": float(np.mean([summary[task]["pooled_oof"]["r2"] for task in tasks]))
        if len(tasks) == 8 else None,
        "legacy_missing_protocol_accepted": legacy_accepted,
        "legacy_compatibility_used": bool(legacy_accepted),
        "verification": {
            "manifest_identity_checked": True,
            "raw_labels_checked": True,
            "predictions_recomputed": True,
            "metric_rtol": METRIC_RTOL, "metric_atol": METRIC_ATOL,
            "label_rtol": LABEL_RTOL, "label_atol": LABEL_ATOL,
            "oof_coverage": "exactly once per task row",
        },
        "interpretation": "fixed five-fold development evaluation; not independent blind evidence",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "shard_root": str(root),
    }
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "all_fold_metrics.csv", index=False)
    write_json(output / "summary.json", result)
    return result


def default_review_dir(timestamp=None):
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"comparison_review_{stamp}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output')
    parser.add_argument('--split-root', required=True)
    parser.add_argument('--raw-root', required=True)
    parser.add_argument('--task', action='append', dest='tasks')
    parser.add_argument('--allow-legacy-missing-protocol', action='store_true')
    args = parser.parse_args()
    tasks = list(args.tasks) if args.tasks else list(TASKS)
    output = Path(args.output) if args.output else Path(args.root) / default_review_dir()
    result = aggregate(args.root, output, tasks=tasks, split_root=args.split_root,
                       raw_root=args.raw_root,
                       allow_legacy_missing_protocol=args.allow_legacy_missing_protocol)
    print(json.dumps({"output": str(output), "macro8_r2": result["macro8_r2"],
                      "macro8_pooled_oof_r2": result["macro8_pooled_oof_r2"],
                      "legacy_compatibility_used": result["legacy_compatibility_used"]},
                     indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
