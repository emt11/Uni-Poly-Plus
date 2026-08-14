#!/usr/bin/env python3
"""Strict MTS fold/seed aggregation with prediction-level ensembling."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")


def _read_shard(root: Path, seed: int, task: str, fold: int,
                evaluation_protocol="historical_shared5") -> dict:
    path = root / "shards" / str(seed) / task / f"fold_{fold}.csv"
    if not path.is_file() or not path.stat().st_size:
        raise RuntimeError(f"missing Stage-3 shard: {path}")
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise RuntimeError(f"shard must contain exactly one row: {path}")
    row = frame.iloc[0].to_dict()
    if str(row.get("task")) != task or int(row.get("seed", -1)) != seed:
        raise RuntimeError(f"shard identity mismatch: {path}")
    expected_fold_protocol = (
        "nested_outer5_inner_hash10"
        if evaluation_protocol == "nested5"
        else "shared_validation_test_fold"
    )
    if str(row.get("fold_validation_protocol")) != expected_fold_protocol:
        raise RuntimeError(f"unexpected validation protocol: {path}")
    blind_value = row.get("independent_blind_test", True)
    claims_blind = (
        blind_value if isinstance(blind_value, (bool, np.bool_))
        else str(blind_value).strip().lower() not in {"false", "0"}
    )
    if claims_blind != (evaluation_protocol == "nested5"):
        raise RuntimeError(f"MTS result incorrectly claims an independent test: {path}")
    metrics = json.loads(str(row.get("per_fold_metrics", "[]")))
    if len(metrics) != 1 or int(metrics[0].get("fold", -1)) != fold:
        raise RuntimeError(f"invalid per_fold_metrics: {path}")
    for key in ("test_r2", "test_mae", "test_rmse", "best_val_r2"):
        if key in metrics[0] and not np.isfinite(float(metrics[0][key])):
            raise RuntimeError(f"non-finite metric {key}: {path}")
    return row


def _read_prediction(root: Path, seed: int, task: str, fold: int, row: dict):
    path = root / "predictions" / str(seed) / task / f"fold_{fold}.npz"
    if not path.is_file():
        raise RuntimeError(f"missing prediction shard: {path}")
    with np.load(path, allow_pickle=False) as payload:
        y_true = np.asarray(payload["y_true"], dtype=np.float64).reshape(-1)
        y_pred = np.asarray(payload["y_pred"], dtype=np.float64).reshape(-1)
        indices = np.asarray(payload["sample_indices"], dtype=np.int64).reshape(-1)
        metadata = json.loads(str(np.asarray(payload["metadata"]).item()))
    if y_true.shape != y_pred.shape or y_true.shape != indices.shape:
        raise RuntimeError(f"prediction array shape mismatch: {path}")
    expected = {"task": task, "fold": fold, "seed": seed}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"prediction metadata mismatch ({key}): {path}")
    expected_protocol = (
        "nested_outer5_inner_hash10"
        if row.get("fold_validation_protocol") == "nested_outer5_inner_hash10"
        else "shared_validation_test_fold"
    )
    if metadata.get("fold_validation_protocol", expected_protocol) != expected_protocol:
        raise RuntimeError(f"prediction protocol mismatch: {path}")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise RuntimeError(f"non-finite prediction data: {path}")
    return y_true, y_pred, indices


def _best_results(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    return {
        str(row.task): float(row.best_r2)
        for row in frame.itertuples(index=False)
    }


def _baseline_task_means(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    if {"task", "fold_test_r2"}.issubset(frame.columns):
        return frame.groupby("task")["fold_test_r2"].mean().astype(float).to_dict()
    if {"task", "r2_mean"}.issubset(frame.columns):
        frame = frame[frame["task"] != "macro"]
        return dict(zip(frame["task"].astype(str), frame["r2_mean"].astype(float)))
    raise RuntimeError(f"unsupported baseline summary format: {path}")


def _atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/mts_finetune_v2")
    parser.add_argument("--output-csv", default="results/mts_finetune_v2/mts_finetune_summary.csv")
    parser.add_argument("--output-md", default="results/mts_finetune_v2/mts_finetune_summary.md")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--best-results", default="results/best_result.csv")
    parser.add_argument("--gate-baseline-csv", default="")
    parser.add_argument("--gate-output", default="")
    parser.add_argument("--enforce-promotion", action="store_true")
    parser.add_argument(
        "--evaluation-protocol",
        choices=("historical_shared5", "nested5"),
        default="historical_shared5",
    )
    args = parser.parse_args()

    if sorted(args.folds) != list(range(5)) or len(args.folds) != 5:
        raise SystemExit("formal MTS summary requires exactly folds 0,1,2,3,4")
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("duplicate seeds are not allowed")

    root = Path(args.results_root)
    fold_rows = []
    for task in args.tasks:
        for fold in args.folds:
            truths = []
            predictions = []
            reference_indices = None
            for seed in args.seeds:
                row = _read_shard(
                    root, seed, task, fold, args.evaluation_protocol
                )
                y_true, y_pred, indices = _read_prediction(root, seed, task, fold, row)
                if reference_indices is None:
                    reference_indices = indices
                    reference_truth = y_true
                elif not np.array_equal(indices, reference_indices) or not np.allclose(
                    y_true, reference_truth, atol=0.0, rtol=0.0
                ):
                    raise RuntimeError(f"seed prediction cohorts differ: {task}/fold_{fold}")
                truths.append(y_true)
                predictions.append(y_pred)
            y_true = truths[0]
            y_pred = np.mean(np.stack(predictions, axis=0), axis=0)
            fold_rows.append({
                "task": task,
                "fold": int(fold),
                "r2": float(r2_score(y_true, y_pred)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            })

    folds = pd.DataFrame(fold_rows)
    expected = len(args.tasks) * 5
    if len(folds) != expected or folds.duplicated(["task", "fold"]).any():
        raise RuntimeError(f"expected {expected} unique task/fold results")

    best = _best_results(Path(args.best_results))
    output_rows = []
    for task in args.tasks:
        values = folds[folds.task == task].sort_values("fold")
        if len(values) != 5:
            raise RuntimeError(f"task {task} does not have five folds")
        summary = {"task": task}
        for metric in ("r2", "mae", "rmse"):
            metric_values = values[metric].to_numpy(dtype=np.float64)
            mean = float(np.mean(metric_values))
            std = float(np.std(metric_values, ddof=1))
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
            summary[f"{metric}_report"] = f"{mean:.3f} ± {std:.3f}"
        summary["best_r2"] = float(best[task])
        summary["gap_to_best"] = float(best[task] - summary["r2_mean"])
        output_rows.append(summary)

    macro_by_fold = folds.groupby("fold", sort=True)["r2"].mean()
    macro_values = macro_by_fold.to_numpy(dtype=np.float64)
    macro_mean = float(np.mean(macro_values))
    macro_std = float(np.std(macro_values, ddof=1))
    output_rows.append({
        "task": "macro",
        "r2_mean": macro_mean,
        "r2_std": macro_std,
        "r2_report": f"{macro_mean:.3f} ± {macro_std:.3f}",
        "mae_mean": np.nan,
        "mae_std": np.nan,
        "mae_report": "",
        "rmse_mean": np.nan,
        "rmse_std": np.nan,
        "rmse_report": "",
        "best_r2": float(np.mean([best[task] for task in args.tasks])),
        "gap_to_best": float(np.mean([best[task] for task in args.tasks]) - macro_mean),
    })

    summary_frame = pd.DataFrame(output_rows, columns=[
        "task", "r2_mean", "r2_std", "r2_report", "mae_mean", "mae_std",
        "mae_report", "rmse_mean", "rmse_std", "rmse_report", "best_r2",
        "gap_to_best",
    ])
    numeric = summary_frame.select_dtypes(include=[np.number]).columns
    summary_frame[numeric] = summary_frame[numeric].round(3)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_tmp = output_csv.with_name(output_csv.name + f".tmp.{os.getpid()}")
    summary_frame.to_csv(csv_tmp, index=False, float_format="%.3f")
    os.replace(csv_tmp, output_csv)

    lines = [
        "# MIPS-Trimer-SCAGE（MTS）下游汇总",
        "",
        f"- Seeds: {', '.join(map(str, args.seeds))}",
        "- Fold protocol: " + (
            "nested_outer5_inner_hash10"
            if args.evaluation_protocol == "nested5"
            else "shared_validation_test_fold"
        ),
        "- Independent blind test: " + (
            "true" if args.evaluation_protocol == "nested5" else "false"
        ),
        f"- Macro R² = {macro_mean:.3f} ± {macro_std:.3f}",
        "",
        "| 任务 | R² | MAE | RMSE | Best R² | Gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in output_rows[:-1]:
        lines.append(
            f"| {row['task']} | {row['r2_report']} | {row['mae_report']} | "
            f"{row['rmse_report']} | {row['best_r2']:.3f} | {row['gap_to_best']:.3f} |"
        )
    _atomic_text(Path(args.output_md), "\n".join(lines) + "\n")

    gate = None
    if args.gate_baseline_csv:
        baseline = _baseline_task_means(Path(args.gate_baseline_csv))
        current = {row["task"]: float(row["r2_mean"]) for row in output_rows[:-1]}
        changes = {task: current[task] - baseline[task] for task in args.tasks}
        baseline_macro = float(np.mean([baseline[task] for task in args.tasks]))
        gate = {
            "baseline_macro_r2": baseline_macro,
            "current_macro_r2": macro_mean,
            "macro_delta": macro_mean - baseline_macro,
            "nondecreasing_tasks": sum(value >= 0.0 for value in changes.values()),
            "worst_task_delta": min(changes.values()),
        }
        gate["passed"] = bool(
            gate["macro_delta"] >= 0.003
            and gate["nondecreasing_tasks"] >= 5
            and gate["worst_task_delta"] >= -0.015
        )
        gate_path = Path(args.gate_output or root / "seed42_promotion.json")
        _atomic_text(gate_path, json.dumps(gate, indent=2, sort_keys=True) + "\n")

    print(json.dumps({
        "seeds": args.seeds,
        "fold_results": len(fold_rows),
        "macro_r2_mean": macro_mean,
        "macro_r2_std": macro_std,
        "promotion": gate,
    }, sort_keys=True))
    if args.enforce_promotion and (gate is None or not gate["passed"]):
        raise SystemExit(4)


if __name__ == "__main__":
    main()
