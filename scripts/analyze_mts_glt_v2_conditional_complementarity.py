#!/usr/bin/env python3
"""Inference-only conditional complementarity audit for GLT-v2 Base-5k.

The formal downstream runner retained fold-best states only in memory.  This
script therefore performs every valid PRE decomposition and records POST and
counterfactual metrics as unavailable when the corresponding fold checkpoint
does not exist.  It never retrains or reconstructs a fold checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch_scatter import scatter


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.mts_glt_postmortem import linear_cka  # noqa: E402
from src.dataset import UniDataset  # noqa: E402
from src.modules import MTSGraphLineModelV2  # noqa: E402
from src.utils import get_data_loader  # noqa: E402


TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
FOLDS = tuple(range(5))
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
CHECKPOINT = ROOT / "results/mts_glt_v2/formal/a6_h_w1_20k/mts_glt_v2_probe_005k.pth"
FORMAL_ROOT = ROOT / "results/mts_glt_v2/downstream/formal/a6_h_w1_20k_probe_005k/o8_glt_atom"
OUTPUT = ROOT / "results/mts_glt_v2/conditional_complementarity_audit_v1"
SIDECAR = ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_v1/downstream_union"
SPLITS = ROOT / "data/splits/mips_shared5"


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_model(device: torch.device):
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if payload.get("schema") != "mts-glt-v2-probe-v1" or int(payload.get("step", -1)) != 5000:
        raise RuntimeError("formal selected checkpoint identity is invalid")
    model = MTSGraphLineModelV2(
        glt_layers=int(payload["glt_layers"]),
        glt_attention_variant=str(payload["glt_attention_variant"]),
        use_compact19=False,
    )
    state = {
        key[len("model."):]: value
        for key, value in payload["state_dict"].items()
        if key.startswith("model.")
    }
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("strict PRE checkpoint load unexpectedly returned incompatibilities")
    return model.to(device).eval(), payload


def make_dataset(task: str):
    return UniDataset(
        root=str(ROOT / "data"),
        dataset=f"smi_{task}",
        smiles_model_name=str(ROOT / "pretrained_models/encoders/PubChem10M_SMILES_BPE_450k"),
        graph_encoder_type="mips_trimer_scage",
        graph_input="star_linking",
        modalities=("graph",),
        cache_layers="ru_base,topology,trimer,md200",
        periodic_line_glt_sidecar=str(SIDECAR),
        experiment_id="glt-v2-conditional-complementarity-audit-v1",
        feature_config_hash="manual",
        mips_core="paper_corrected",
        mips_max_hops=2,
        mips_use_descriptors=True,
    )


def concatenate(parts, dtype):
    return torch.cat(parts, dim=0).numpy().astype(dtype, copy=False)


def extract_task(task: str, model, device: torch.device, batch_size: int, workers: int):
    dataset = make_dataset(task)
    indices = np.arange(len(dataset), dtype=np.int64)
    loader = get_data_loader(
        dataset,
        indices=indices,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(workers) > 0,
        prefetch_factor=2,
    )
    graph_o8, graph_glt, graph_valid, graph_ids = [], [], [], []
    atom_o8, atom_glt, atom_graph, atom_z = [], [], [], []
    atom_degree, atom_cross, atom_valid = [], [], []
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            count = len(batch.smiles)
            local_ids = torch.as_tensor(indices[cursor:cursor + count], dtype=torch.long)
            cursor += count
            batch = batch.to(device, non_blocking=True)
            z_o8, h_o8 = model._o8_canonical_states(batch)
            glt = model.glt(batch)
            h_glt = glt["atom_geometry_states"]
            if not all(torch.isfinite(value).all() for value in (z_o8, h_o8, h_glt, glt["graph_geometry"])):
                raise RuntimeError(f"non-finite PRE representation for {task}")
            graph_o8.append(z_o8.float().cpu())
            graph_glt.append(glt["graph_geometry"].float().cpu())
            graph_valid.append(batch.glt_geometry_valid.bool().cpu())
            graph_ids.append(local_ids)
            atom_o8.append(h_o8.float().cpu())
            atom_glt.append(h_glt.float().cpu())
            atom_graph.append(local_ids[batch.canonical_graph_index.long().cpu()])
            atom_z.append(batch.canonical_atomic_numbers.long().cpu())
            atom_valid.append(glt["atom_geometry_valid"].bool().cpu())

            canonical_count = int(h_o8.size(0))
            degree = torch.zeros(canonical_count, dtype=torch.long, device=device)
            cross = torch.zeros(canonical_count, dtype=torch.bool, device=device)
            token_valid = batch.glt_token_valid.bool()
            endpoints = torch.cat((
                batch.glt_token_atom_a.long()[token_valid],
                batch.glt_token_atom_b.long()[token_valid],
            ))
            if endpoints.numel():
                degree.index_add_(0, endpoints, torch.ones_like(endpoints))
            cross_tokens = token_valid & (batch.glt_token_shift.long().abs() == 1)
            cross_endpoints = torch.cat((
                batch.glt_token_atom_a.long()[cross_tokens],
                batch.glt_token_atom_b.long()[cross_tokens],
            ))
            if cross_endpoints.numel():
                cross[cross_endpoints] = True
            atom_degree.append(degree.cpu())
            atom_cross.append(cross.cpu())
    if cursor != len(dataset):
        raise RuntimeError(f"dataset extraction mismatch for {task}: {cursor} != {len(dataset)}")
    arrays = {
        "graph_id": concatenate(graph_ids, np.int64),
        "graph_o8": concatenate(graph_o8, np.float32),
        "graph_glt": concatenate(graph_glt, np.float32),
        "graph_valid": concatenate(graph_valid, bool),
        "atom_o8": concatenate(atom_o8, np.float32),
        "atom_glt": concatenate(atom_glt, np.float32),
        "atom_graph_id": concatenate(atom_graph, np.int64),
        "atom_z": concatenate(atom_z, np.int64),
        "atom_degree": concatenate(atom_degree, np.int64),
        "atom_cross_ru": concatenate(atom_cross, bool),
        "atom_valid": concatenate(atom_valid, bool),
    }
    return arrays


def inner_train_validation(indices: np.ndarray, task_index: int, fold: int, seed: int):
    indices = np.asarray(indices, dtype=np.int64)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(task_index), int(fold)]))
    shuffled = rng.permutation(indices)
    validation_count = max(1, int(round(0.2 * len(shuffled))))
    return np.sort(shuffled[validation_count:]), np.sort(shuffled[:validation_count])


class RidgeMap:
    def __init__(self, x_mean, x_scale, y_mean, y_scale, coefficient, alpha, validation_r2):
        self.x_mean = x_mean
        self.x_scale = x_scale
        self.y_mean = y_mean
        self.y_scale = y_scale
        self.coefficient = coefficient
        self.alpha = float(alpha)
        self.validation_r2 = float(validation_r2)

    def predict(self, values):
        scaled = (
            (np.asarray(values, dtype=np.float32) - self.x_mean) / self.x_scale
        ) @ self.coefficient
        return scaled * self.y_scale + self.y_mean


def fit_ridge(train_x, train_y, validation_x, validation_y):
    train_x = np.asarray(train_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.float32)
    x_mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    y_mean = train_y.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_scale = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    y_scale = train_y.std(axis=0, dtype=np.float64).astype(np.float32)
    x_scale[x_scale < 1e-8] = 1.0
    y_scale[y_scale < 1e-8] = 1.0
    train_x_scaled = (train_x - x_mean) / x_scale
    train_y_scaled = (train_y - y_mean) / y_scale
    validation_x_scaled = (np.asarray(validation_x, dtype=np.float32) - x_mean) / x_scale
    xtx = np.asarray(train_x_scaled.T @ train_x_scaled, dtype=np.float64)
    xty = np.asarray(train_x_scaled.T @ train_y_scaled, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(xtx)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected = eigenvectors.T @ xty
    best = None
    for alpha in ALPHAS:
        coefficient = eigenvectors @ (
            projected / (eigenvalues[:, None] + float(alpha))
        )
        prediction = (
            validation_x_scaled @ coefficient
        ) * y_scale + y_mean
        score = float(r2_score(validation_y, prediction, multioutput="uniform_average"))
        if best is None or score > best[0]:
            best = (score, float(alpha), coefficient)
    return RidgeMap(
        x_mean, x_scale, y_mean, y_scale,
        np.asarray(best[2], dtype=np.float32), best[1], best[0],
    )


def effective_rank(values):
    values = np.asarray(values, dtype=np.float64)
    values = values - values.mean(axis=0, keepdims=True)
    eigenvalues = np.maximum(np.linalg.eigvalsh(values.T @ values), 0.0)
    total = float(eigenvalues.sum())
    if total <= 0:
        return 0.0
    probabilities = eigenvalues[eigenvalues > 0] / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def residual_metrics(target, predicted, train_target_mean):
    residual = target - predicted
    denominator = np.square(target - train_target_mean).sum()
    fraction = float(np.square(residual).sum() / max(float(denominator), 1e-12))
    return {
        "residual_fraction": fraction,
        "raw_effective_rank": effective_rank(target),
        "predictable_effective_rank": effective_rank(predicted),
        "residual_effective_rank": effective_rank(residual),
    }, residual


def balanced_atom_indices(graph_ids, limit=32, seed=42):
    selected = []
    for graph_id in np.unique(graph_ids):
        candidates = np.flatnonzero(graph_ids == graph_id)
        if len(candidates) <= int(limit):
            selected.extend(candidates.tolist())
        else:
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(graph_id)]))
            selected.extend(np.sort(rng.choice(candidates, int(limit), replace=False)).tolist())
    return np.asarray(selected, dtype=np.int64)


def post_checkpoint_inventory():
    # The production fold loop explicitly did not persist best_model_state.
    candidates = list(FORMAL_ROOT.rglob("*.pth")) + list(FORMAL_ROOT.rglob("*.pt"))
    candidates += list((ROOT / "saved_models").rglob("*.pth")) if (ROOT / "saved_models").exists() else []
    return sorted({str(path.resolve()) for path in candidates})


def mean_or_none(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, checkpoint_payload = load_model(device)
    post_candidates = post_checkpoint_inventory()
    if post_candidates:
        raise RuntimeError(
            "ambiguous POST checkpoint candidates were found; explicit task/fold identity is required: "
            + ", ".join(post_candidates[:5])
        )

    per_fold = []
    rank_rows = []
    group_samples = []
    available_post = []
    missing_post = [f"{task}/fold{fold}" for task in TASKS for fold in FOLDS]
    feature_dir = output / "pre_features"
    feature_dir.mkdir(exist_ok=True)

    for task_index, task in enumerate(TASKS):
        feature_path = feature_dir / f"{task}.npz"
        if feature_path.is_file():
            print(f"reusing PRE task={task} features", flush=True)
            cached = np.load(feature_path, allow_pickle=False)
            arrays = {name: cached[name] for name in cached.files}
        else:
            print(f"extracting PRE task={task}", flush=True)
            arrays = extract_task(task, model, device, args.batch_size, args.workers)
            np.savez_compressed(feature_path, **arrays)
        split = json.loads((SPLITS / f"{task}.json").read_text())
        if int(split["sample_count"]) != len(arrays["graph_id"]):
            raise RuntimeError(f"split sample count mismatch for {task}")
        graph_valid = arrays["graph_valid"]
        for fold_payload in split["folds"]:
            fold = int(fold_payload["fold"])
            outer_train = np.asarray(fold_payload["train_indices"], dtype=np.int64)
            test_graphs = np.asarray(fold_payload["test_indices"], dtype=np.int64)
            outer_train = outer_train[graph_valid[outer_train]]
            test_graphs = test_graphs[graph_valid[test_graphs]]
            train_graphs, validation_graphs = inner_train_validation(
                outer_train, task_index, fold, args.seed
            )
            graph_map = fit_ridge(
                arrays["graph_o8"][train_graphs], arrays["graph_glt"][train_graphs],
                arrays["graph_o8"][validation_graphs], arrays["graph_glt"][validation_graphs],
            )
            graph_prediction = graph_map.predict(arrays["graph_o8"][test_graphs])
            graph_r2 = float(r2_score(
                arrays["graph_glt"][test_graphs], graph_prediction,
                multioutput="uniform_average",
            ))
            graph_stats, _ = residual_metrics(
                arrays["graph_glt"][test_graphs], graph_prediction,
                arrays["graph_glt"][train_graphs].mean(axis=0, keepdims=True),
            )
            graph_cka = float(linear_cka(
                arrays["graph_o8"][test_graphs], arrays["graph_glt"][test_graphs]
            ))

            valid_atom = arrays["atom_valid"]
            atom_graph = arrays["atom_graph_id"]
            atom_train = np.flatnonzero(valid_atom & np.isin(atom_graph, train_graphs))
            atom_validation = np.flatnonzero(valid_atom & np.isin(atom_graph, validation_graphs))
            atom_test = np.flatnonzero(valid_atom & np.isin(atom_graph, test_graphs))
            atom_map = fit_ridge(
                arrays["atom_o8"][atom_train], arrays["atom_glt"][atom_train],
                arrays["atom_o8"][atom_validation], arrays["atom_glt"][atom_validation],
            )
            atom_prediction = atom_map.predict(arrays["atom_o8"][atom_test])
            atom_r2 = float(r2_score(
                arrays["atom_glt"][atom_test], atom_prediction,
                multioutput="uniform_average",
            ))
            atom_stats, atom_residual = residual_metrics(
                arrays["atom_glt"][atom_test], atom_prediction,
                arrays["atom_glt"][atom_train].mean(axis=0, keepdims=True),
            )
            atom_cka_all = float(linear_cka(
                arrays["atom_o8"][atom_test], arrays["atom_glt"][atom_test]
            ))
            balanced = balanced_atom_indices(atom_graph[atom_test], limit=32, seed=args.seed)
            atom_cka_balanced = float(linear_cka(
                arrays["atom_o8"][atom_test][balanced], arrays["atom_glt"][atom_test][balanced]
            ))
            row = {
                "task": task, "fold": fold,
                "post_checkpoint_status": "missing_post_checkpoint",
                "pre_graph_alignment_r2": graph_r2,
                "pre_graph_cka": graph_cka,
                "pre_graph_residual_fraction": graph_stats["residual_fraction"],
                "pre_atom_alignment_r2": atom_r2,
                "pre_atom_cka_all": atom_cka_all,
                "pre_atom_cka_graph_balanced": atom_cka_balanced,
                "pre_atom_residual_fraction": atom_stats["residual_fraction"],
                "post_graph_alignment_r2": None,
                "post_graph_cka": None,
                "post_atom_alignment_r2": None,
                "post_atom_cka_all": None,
                "post_atom_residual_fraction": None,
                "r2_full": None, "r2_no3d": None, "r2_shared3d": None,
                "total_3d_use": None, "private_3d_sensitivity": None,
                "shared_3d_sensitivity": None, "private_response_ratio": None,
                "graph_alpha": graph_map.alpha, "atom_alpha": atom_map.alpha,
                "train_graphs": len(train_graphs), "validation_graphs": len(validation_graphs),
                "test_graphs": len(test_graphs), "test_atoms": len(atom_test),
            }
            per_fold.append(row)
            rank_rows.append({
                "task": task, "fold": fold, "stage": "PRE", "level": "graph", **graph_stats
            })
            rank_rows.append({
                "task": task, "fold": fold, "stage": "PRE", "level": "atom", **atom_stats
            })
            norms = np.linalg.norm(atom_residual, axis=1)
            for local, norm in zip(atom_test, norms):
                group_samples.append({
                    "task": task,
                    "atomic_number": int(arrays["atom_z"][local]),
                    "degree": int(arrays["atom_degree"][local]),
                    "cross_ru_incident": bool(arrays["atom_cross_ru"][local]),
                    "residual_norm": float(norm),
                })

    task_rows = []
    for task in TASKS:
        rows = [row for row in per_fold if row["task"] == task]
        task_rows.append({
            "task": task,
            "pre_graph_alignment_r2": mean_or_none(rows, "pre_graph_alignment_r2"),
            "pre_graph_cka": mean_or_none(rows, "pre_graph_cka"),
            "pre_graph_residual_fraction": mean_or_none(rows, "pre_graph_residual_fraction"),
            "pre_atom_alignment_r2": mean_or_none(rows, "pre_atom_alignment_r2"),
            "pre_atom_cka": mean_or_none(rows, "pre_atom_cka_graph_balanced"),
            "pre_atom_residual_fraction": mean_or_none(rows, "pre_atom_residual_fraction"),
            "post_graph_alignment_r2": None, "post_atom_alignment_r2": None,
            "post_atom_residual_fraction": None,
            "r2_full": None, "r2_no3d": None, "r2_shared3d": None,
            "total_3d_use": None, "private_3d_sensitivity": None,
            "shared_3d_sensitivity": None,
        })

    group_rows = []
    for group_type, key in (
        ("element", "atomic_number"), ("degree", "degree"),
        ("cross_ru_incidence", "cross_ru_incident"),
    ):
        values = sorted({row[key] for row in group_samples})
        for value in values:
            norms = np.asarray([row["residual_norm"] for row in group_samples if row[key] == value])
            if group_type == "element" and len(norms) < 100:
                continue
            group_rows.append({
                "stage": "PRE", "group_type": group_type, "group_value": value,
                "count": len(norms), "mean_residual_norm": float(norms.mean()),
                "median_residual_norm": float(np.median(norms)),
            })

    alignment_payload = {
        "schema": "mts-glt-v2-pre-post-alignment-v1",
        "post_status": "unavailable_missing_formal_fold_checkpoints",
        "per_fold": [{key: row[key] for key in (
            "task", "fold", "pre_graph_alignment_r2", "pre_graph_cka",
            "pre_atom_alignment_r2", "pre_atom_cka_all", "pre_atom_cka_graph_balanced",
            "post_graph_alignment_r2", "post_graph_cka", "post_atom_alignment_r2",
            "post_atom_cka_all",
        )} for row in per_fold],
    }
    counterfactual_payload = {
        "schema": "mts-glt-v2-fusion-counterfactual-v1",
        "status": "unavailable_missing_formal_fold_checkpoints",
        "reason": "Formal Warm0 fold-best states were restored only in memory and were not saved; retraining is forbidden by this audit.",
        "available_post_checkpoints": available_post,
        "missing_post_checkpoints": missing_post,
        "metrics": [],
    }
    summary = {
        "schema": "mts-glt-v2-conditional-complementarity-audit-v1",
        "baseline_name": "MTS-GLT-v2-Base-5k",
        "pretrained_checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_step": int(checkpoint_payload["step"]),
        "tasks": list(TASKS), "folds": list(FOLDS), "seed": int(args.seed),
        "protocol": "historical_shared5", "independent_blind_test": False,
        "available_post_checkpoints": available_post,
        "missing_post_checkpoints": missing_post,
        "macro_pre_graph_alignment_r2": mean_or_none(task_rows, "pre_graph_alignment_r2"),
        "macro_post_graph_alignment_r2": None,
        "macro_pre_atom_alignment_r2": mean_or_none(task_rows, "pre_atom_alignment_r2"),
        "macro_post_atom_alignment_r2": None,
        "macro_pre_graph_residual_fraction": mean_or_none(task_rows, "pre_graph_residual_fraction"),
        "macro_post_graph_residual_fraction": None,
        "macro_pre_atom_residual_fraction": mean_or_none(task_rows, "pre_atom_residual_fraction"),
        "macro_post_atom_residual_fraction": None,
        "macro_r2_full": None, "macro_r2_no3d": None, "macro_r2_shared3d": None,
        "macro_total_3d_use": None, "macro_private_3d_sensitivity": None,
        "macro_shared_3d_sensitivity": None,
        "median_private_3d_sensitivity": None,
        "positive_private_3d_tasks": None, "positive_private_3d_folds": None,
        "macro_private_response_ratio": None,
        "private_3d_utilization": "NOT_ESTABLISHED",
        "audit_scope": "PRE decomposition complete; POST and within-model counterfactual unavailable without saved fold checkpoints",
    }
    per_fold_fields = list(per_fold[0].keys())
    task_fields = list(task_rows[0].keys())
    write_csv(output / "per_fold_metrics.csv", per_fold, per_fold_fields)
    write_csv(output / "task_metrics.csv", task_rows, task_fields)
    write_csv(output / "residual_group_statistics.csv", group_rows, list(group_rows[0].keys()))
    atomic_json(output / "conditional_complementarity_summary.json", summary)
    atomic_json(output / "pre_post_alignment_metrics.json", alignment_payload)
    atomic_json(output / "residual_rank_metrics.json", {"schema": "mts-glt-v2-residual-rank-v1", "rows": rank_rows})
    atomic_json(output / "fusion_counterfactual_metrics.json", counterfactual_payload)
    atomic_json(output / "run_manifest.json", {
        "schema": "mts-glt-v2-conditional-complementarity-run-v1",
        "baseline": "MTS-GLT-v2-Base-5k",
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_step": 5000,
        "trajectory_end_step": 20000,
        "tasks": list(TASKS), "folds": list(FOLDS), "seed": int(args.seed),
        "protocol": "historical_shared5",
        "ridge_alphas": list(ALPHAS),
        "inner_validation": "deterministic 20% of outer-train graphs; test is never used for alpha selection",
        "atom_split": "derived from graph split; no atom-random split",
        "post_checkpoint_inventory": post_candidates,
        "status": "completed_with_missing_post_checkpoints",
    })
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
