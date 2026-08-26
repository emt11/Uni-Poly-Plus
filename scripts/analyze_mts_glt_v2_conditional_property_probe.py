#!/usr/bin/env python3
"""Frozen O8-conditional GLT-v2 property probes.

The historical shared5 protocol aliases validation to test.  To keep the
outer held-out rows untouched, this analysis deterministically carves its
validation partition from the official outer-train graphs.  No encoder or
downstream model is trained by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_mts_glt_v2_conditional_complementarity as conditional  # noqa: E402


TASKS = conditional.TASKS
FOLDS = conditional.FOLDS
PROPERTY_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
PRE_FEATURES = (
    ROOT / "results/mts_glt_v2/conditional_complementarity_audit_v1/pre_features"
)
PRE_GROUP_STATS = (
    ROOT
    / "results/mts_glt_v2/conditional_complementarity_audit_v1"
    / "residual_group_statistics.csv"
)
SPLITS = ROOT / "data/splits/mips_shared5"
OUTPUT = ROOT / "results/mts_glt_v2/conditional_property_probe_v1"
LOG_ROOT = ROOT / "logs/mts_glt_v2/conditional_property_probe_v1"


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_targets(split_payload: dict) -> np.ndarray:
    source = ROOT / str(split_payload["source_csv"])
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if len(header) != 2:
            raise RuntimeError(f"expected one target column in {source}: {header}")
        targets = np.asarray([float(row[1]) for row in reader], dtype=np.float64)
    if targets.shape != (int(split_payload["sample_count"]),):
        raise RuntimeError(f"target row count mismatch for {source}")
    if not np.isfinite(targets).all():
        raise RuntimeError(f"non-finite target in {source}")
    return targets


class ScalarRidge:
    def __init__(self, x_mean, x_scale, y_mean, coefficient, alpha, validation_r2):
        self.x_mean = x_mean
        self.x_scale = x_scale
        self.y_mean = float(y_mean)
        self.coefficient = coefficient
        self.alpha = float(alpha)
        self.validation_r2 = float(validation_r2)

    def predict(self, values):
        values = np.asarray(values, dtype=np.float64)
        return ((values - self.x_mean) / self.x_scale) @ self.coefficient + self.y_mean


def _ridge_coefficients(train_x: np.ndarray, centered_y: np.ndarray, alphas):
    rows, columns = train_x.shape
    if rows <= columns:
        gram = np.asarray(train_x @ train_x.T, dtype=np.float64)
        eigenvalues, vectors = np.linalg.eigh(gram)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        projected = vectors.T @ centered_y
        return (
            train_x.T @ vectors @ (projected / (eigenvalues + float(alpha)))
            for alpha in alphas
        )
    gram = np.asarray(train_x.T @ train_x, dtype=np.float64)
    eigenvalues, vectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected = vectors.T @ train_x.T @ centered_y
    return (
        vectors @ (projected / (eigenvalues + float(alpha)))
        for alpha in alphas
    )


def fit_scalar_ridge(train_x, train_y, validation_x, validation_y) -> ScalarRidge:
    train_x = np.asarray(train_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64).reshape(-1)
    validation_x = np.asarray(validation_x, dtype=np.float64)
    validation_y = np.asarray(validation_y, dtype=np.float64).reshape(-1)
    x_mean = train_x.mean(axis=0)
    x_scale = train_x.std(axis=0)
    x_scale[x_scale < 1e-12] = 1.0
    scaled_train = (train_x - x_mean) / x_scale
    scaled_validation = (validation_x - x_mean) / x_scale
    y_mean = float(train_y.mean())
    centered_y = train_y - y_mean
    best = None
    for alpha, coefficient in zip(
        PROPERTY_ALPHAS,
        _ridge_coefficients(scaled_train, centered_y, PROPERTY_ALPHAS),
    ):
        prediction = scaled_validation @ coefficient + y_mean
        score = float(r2_score(validation_y, prediction))
        if np.isfinite(score) and (best is None or score > best[0]):
            best = (score, float(alpha), np.asarray(coefficient, dtype=np.float64))
    if best is None:
        raise RuntimeError("property Ridge alpha selection produced no finite score")
    return ScalarRidge(x_mean, x_scale, y_mean, best[2], best[1], best[0])


def pool_atom_residuals(
    atom_residual: np.ndarray,
    atom_graph_ids: np.ndarray,
    atom_valid: np.ndarray,
    graph_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((int(graph_count), atom_residual.shape[1]), dtype=np.float64)
    counts = np.zeros(int(graph_count), dtype=np.int64)
    valid_rows = np.flatnonzero(np.asarray(atom_valid, dtype=bool))
    np.add.at(sums, atom_graph_ids[valid_rows], atom_residual[valid_rows])
    np.add.at(counts, atom_graph_ids[valid_rows], 1)
    pooled = np.zeros_like(sums)
    nonempty = counts > 0
    pooled[nonempty] = sums[nonempty] / counts[nonempty, None]
    return pooled, counts


def probe_feature_set(train_x, validation_x, test_x, targets, train, validation, test):
    fitted = fit_scalar_ridge(
        train_x, targets[train], validation_x, targets[validation]
    )
    prediction = fitted.predict(test_x)
    return {
        "r2": float(r2_score(targets[test], prediction)),
        "alpha": fitted.alpha,
        "validation_r2": fitted.validation_r2,
    }


def distribution(values: list[float]) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def decision(task_deltas: np.ndarray) -> str:
    return (
        "POSITIVE"
        if float(task_deltas.mean()) > 0
        and float(np.median(task_deltas)) > 0
        and int((task_deltas > 0).sum()) >= 5
        else "NOT_ESTABLISHED"
    )


def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_fold = []
    cross_enrichment_samples: dict[str, list[float]] = {task: [] for task in TASKS}

    for task_index, task in enumerate(TASKS):
        print(f"loading task={task}", flush=True)
        feature_path = PRE_FEATURES / f"{task}.npz"
        if not feature_path.is_file():
            device = conditional.torch.device(args.device)
            model, _ = conditional.load_model(device)
            arrays = conditional.extract_task(
                task, model, device, args.batch_size, args.workers
            )
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(feature_path, **arrays)
        with np.load(feature_path, allow_pickle=False) as cached:
            arrays = {name: np.asarray(cached[name]) for name in cached.files}
        split_payload = json.loads((SPLITS / f"{task}.json").read_text())
        if not bool(split_payload.get("validation_is_test")):
            raise RuntimeError("this audit expects the historical shared5 protocol")
        targets = load_targets(split_payload)
        graph_count = len(targets)
        if not np.array_equal(arrays["graph_id"], np.arange(graph_count)):
            raise RuntimeError(f"cached graph ordering mismatch for {task}")
        graph_valid = arrays["graph_valid"].astype(bool)
        atom_graph = arrays["atom_graph_id"].astype(np.int64)
        atom_valid = arrays["atom_valid"].astype(bool)

        for fold_payload in split_payload["folds"]:
            fold = int(fold_payload["fold"])
            outer_train = np.asarray(fold_payload["train_indices"], dtype=np.int64)
            test = np.asarray(fold_payload["test_indices"], dtype=np.int64)
            outer_train = outer_train[graph_valid[outer_train]]
            test = test[graph_valid[test]]
            train, validation = conditional.inner_train_validation(
                outer_train, task_index, fold, args.seed
            )
            if any(np.intersect1d(first, second).size for first, second in (
                (train, validation), (train, test), (validation, test)
            )):
                raise RuntimeError(f"graph split overlap for {task}/fold{fold}")

            graph_map = conditional.fit_ridge(
                arrays["graph_o8"][train], arrays["graph_glt"][train],
                arrays["graph_o8"][validation], arrays["graph_glt"][validation],
            )
            graph_prediction_all = graph_map.predict(arrays["graph_o8"])
            graph_residual_all = arrays["graph_glt"] - graph_prediction_all
            graph_stats, _ = conditional.residual_metrics(
                arrays["graph_glt"][test], graph_prediction_all[test],
                arrays["graph_glt"][train].mean(axis=0, keepdims=True),
            )
            graph_alignment_r2 = float(r2_score(
                arrays["graph_glt"][test], graph_prediction_all[test],
                multioutput="uniform_average",
            ))

            atom_train = np.flatnonzero(atom_valid & np.isin(atom_graph, train))
            atom_validation = np.flatnonzero(atom_valid & np.isin(atom_graph, validation))
            atom_test = np.flatnonzero(atom_valid & np.isin(atom_graph, test))
            atom_map = conditional.fit_ridge(
                arrays["atom_o8"][atom_train], arrays["atom_glt"][atom_train],
                arrays["atom_o8"][atom_validation], arrays["atom_glt"][atom_validation],
            )
            atom_prediction_all = atom_map.predict(arrays["atom_o8"])
            atom_residual_all = arrays["atom_glt"] - atom_prediction_all
            atom_pool, atom_counts = pool_atom_residuals(
                atom_residual_all, atom_graph, atom_valid, graph_count
            )
            if not bool((atom_counts[np.concatenate((train, validation, test))] > 0).all()):
                raise RuntimeError(f"empty valid atom pool for {task}/fold{fold}")
            atom_stats, atom_test_residual = conditional.residual_metrics(
                arrays["atom_glt"][atom_test], atom_prediction_all[atom_test],
                arrays["atom_glt"][atom_train].mean(axis=0, keepdims=True),
            )
            atom_alignment_r2 = float(r2_score(
                arrays["atom_glt"][atom_test], atom_prediction_all[atom_test],
                multioutput="uniform_average",
            ))

            features = {
                "p0": arrays["graph_o8"],
                "p1": np.concatenate((arrays["graph_o8"], arrays["graph_glt"]), axis=1),
                "p2": np.concatenate((arrays["graph_o8"], graph_residual_all), axis=1),
                "p3": np.concatenate((arrays["graph_o8"], atom_pool), axis=1),
                "rgraph_only": graph_residual_all,
                "ratom_only": atom_pool,
            }
            probes = {
                name: probe_feature_set(
                    value[train], value[validation], value[test],
                    targets, train, validation, test,
                )
                for name, value in features.items()
            }
            test_norm = np.linalg.norm(atom_test_residual, axis=1)
            test_cross = arrays["atom_cross_ru"][atom_test].astype(bool)
            cross_mean = float(test_norm[test_cross].mean()) if test_cross.any() else None
            noncross_mean = float(test_norm[~test_cross].mean()) if (~test_cross).any() else None
            enrichment = (
                cross_mean / max(noncross_mean, 1e-12)
                if cross_mean is not None and noncross_mean is not None else None
            )
            if enrichment is not None:
                cross_enrichment_samples[task].append(enrichment)
            row = {
                "task": task, "fold": fold,
                "n_train": len(train), "n_validation": len(validation), "n_test": len(test),
                "p0": probes["p0"]["r2"], "p1": probes["p1"]["r2"],
                "p2": probes["p2"]["r2"], "p3": probes["p3"]["r2"],
                "rgraph_only": probes["rgraph_only"]["r2"],
                "ratom_only": probes["ratom_only"]["r2"],
                "delta_raw": probes["p1"]["r2"] - probes["p0"]["r2"],
                "delta_graph_residual": probes["p2"]["r2"] - probes["p0"]["r2"],
                "delta_atom_residual": probes["p3"]["r2"] - probes["p0"]["r2"],
                "graph_residual_fraction": graph_stats["residual_fraction"],
                "atom_residual_fraction": atom_stats["residual_fraction"],
                "graph_alignment_r2": graph_alignment_r2,
                "atom_alignment_r2": atom_alignment_r2,
                "cross_ru_residual_enrichment": enrichment,
                "graph_map_alpha": graph_map.alpha, "atom_map_alpha": atom_map.alpha,
            }
            for name in probes:
                row[f"{name}_alpha"] = probes[name]["alpha"]
            per_fold.append(row)
            print(
                f"completed {task}/fold{fold}: "
                f"dRaw={row['delta_raw']:+.6f} "
                f"dGraph={row['delta_graph_residual']:+.6f} "
                f"dAtom={row['delta_atom_residual']:+.6f}",
                flush=True,
            )

    task_rows = []
    for task in TASKS:
        rows = [row for row in per_fold if row["task"] == task]
        task_rows.append({
            "task": task,
            **{
                key: float(np.mean([row[key] for row in rows]))
                for key in (
                    "p0", "p1", "p2", "p3", "rgraph_only", "ratom_only",
                    "delta_raw", "delta_graph_residual", "delta_atom_residual",
                    "graph_residual_fraction", "atom_residual_fraction",
                    "graph_alignment_r2", "atom_alignment_r2",
                )
            },
            "positive_delta_raw_folds": int(sum(row["delta_raw"] > 0 for row in rows)),
            "positive_graph_residual_folds": int(sum(row["delta_graph_residual"] > 0 for row in rows)),
            "positive_atom_residual_folds": int(sum(row["delta_atom_residual"] > 0 for row in rows)),
            "cross_ru_residual_enrichment": float(np.mean(cross_enrichment_samples[task])),
        })

    def aggregate(delta_key: str):
        task_delta = np.asarray([row[delta_key] for row in task_rows], dtype=np.float64)
        fold_delta = [row[delta_key] for row in per_fold]
        return {
            "macro": float(task_delta.mean()),
            "median_task": float(np.median(task_delta)),
            "positive_tasks": int((task_delta > 0).sum()),
            "positive_folds": int(sum(value > 0 for value in fold_delta)),
            "fold_distribution": distribution(fold_delta),
        }

    raw = aggregate("delta_raw")
    graph = aggregate("delta_graph_residual")
    atom = aggregate("delta_atom_residual")
    enrichment = np.asarray(
        [row["cross_ru_residual_enrichment"] for row in task_rows], dtype=np.float64
    )
    atom_gain = np.asarray([row["delta_atom_residual"] for row in task_rows])
    atom_fraction = np.asarray(
        [row["atom_residual_fraction"] for row in task_rows], dtype=np.float64
    )
    enrichment_correlation = spearmanr(enrichment, atom_gain)
    fraction_correlation = spearmanr(atom_fraction, atom_gain)
    with PRE_GROUP_STATS.open(newline="", encoding="utf-8") as handle:
        prior_group_rows = list(csv.DictReader(handle))
    localization = {
        "source_group_statistics": str(PRE_GROUP_STATS.resolve()),
        "prior_pre_residual_groups": prior_group_rows,
        "cross_ru_enrichment_by_task": {
            row["task"]: row["cross_ru_residual_enrichment"] for row in task_rows
        },
        "spearman_cross_ru_enrichment_vs_atom_probe_gain": {
            "rho": float(enrichment_correlation.statistic),
            "pvalue_descriptive": float(enrichment_correlation.pvalue),
        },
        "spearman_atom_residual_fraction_vs_atom_probe_gain": {
            "rho": float(fraction_correlation.statistic),
            "pvalue_descriptive": float(fraction_correlation.pvalue),
        },
        "interpretation": "descriptive eight-task association only; no significance mining or causal claim",
    }
    summary = {
        "schema": "mts-glt-v2-o8-conditional-property-probe-v1",
        "baseline": "MTS-GLT-v2-Base-5k",
        "checkpoint": str(conditional.CHECKPOINT.resolve()),
        "checkpoint_step": 5000,
        "tasks": list(TASKS), "folds": list(FOLDS),
        "seed": int(args.seed), "protocol": "historical_shared5",
        "independent_blind_test": False,
        "macro_p0": float(np.mean([row["p0"] for row in task_rows])),
        "macro_p1": float(np.mean([row["p1"] for row in task_rows])),
        "macro_p2": float(np.mean([row["p2"] for row in task_rows])),
        "macro_p3": float(np.mean([row["p3"] for row in task_rows])),
        "macro_delta_raw": raw["macro"],
        "median_delta_raw": raw["median_task"],
        "positive_delta_raw_tasks": raw["positive_tasks"],
        "positive_delta_raw_folds": raw["positive_folds"],
        "macro_delta_graph_residual": graph["macro"],
        "median_delta_graph_residual": graph["median_task"],
        "positive_graph_residual_tasks": graph["positive_tasks"],
        "positive_graph_residual_folds": graph["positive_folds"],
        "macro_delta_atom_residual": atom["macro"],
        "median_delta_atom_residual": atom["median_task"],
        "positive_atom_residual_tasks": atom["positive_tasks"],
        "positive_atom_residual_folds": atom["positive_folds"],
        "graph_residual_property_signal": decision(
            np.asarray([row["delta_graph_residual"] for row in task_rows])
        ),
        "atom_residual_property_signal": decision(atom_gain),
        "delta_distributions": {"raw": raw, "graph_residual": graph, "atom_residual": atom},
        "localization": localization,
    }
    contract = {
        "schema": "mts-glt-v2-conditional-property-probe-contract-v1",
        "representation": "fusion-pre O8 and GLT canonical atom states; graph mean pooling",
        "residual_term": "O8-unpredictable linear residual; not information-theoretic private information",
        "official_outer_test": True,
        "formal_validation_is_test": True,
        "selection_adaptation": "deterministic 20% validation carved from official outer-train; official test untouched",
        "mapping_alphas": list(conditional.ALPHAS),
        "property_alphas": list(PROPERTY_ALPHAS),
        "atom_split": "derived from graph partitions; no atom-random split",
        "target": "raw task property from source CSV",
    }
    atomic_json(output / "conditional_property_probe_summary.json", summary)
    write_csv(output / "per_fold_results.csv", per_fold, list(per_fold[0].keys()))
    write_csv(output / "task_results.csv", task_rows, list(task_rows[0].keys()))
    atomic_json(output / "probe_contract.json", contract)
    atomic_json(output / "run_manifest.json", {
        "schema": "mts-glt-v2-conditional-property-probe-run-v1",
        "status": "complete", "baseline": summary["baseline"],
        "checkpoint": summary["checkpoint"], "tasks": list(TASKS),
        "folds": list(FOLDS), "seed": int(args.seed),
        "pre_feature_cache": str(PRE_FEATURES.resolve()),
        "encoder_training": False, "downstream_finetuning": False,
        "checkpoint_retention": args.checkpoint_retention_status,
    })
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-retention-status", default="skipped_to_avoid_engine_overlap"
    )
    args = parser.parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
