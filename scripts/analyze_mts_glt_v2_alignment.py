#!/usr/bin/env python3
"""Audit graph- and canonical-atom alignment in the selected MTS-GLT-v2 model.

This is an inference-only diagnostic.  It deliberately uses the model's
existing O8 pooling, GLT atom incidence, and InfoNCE projectors without
changing training or forward semantics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.mts_glt_postmortem import linear_cka  # noqa: E402
from src.dataset import UniDataset  # noqa: E402
from src.modules import MTSGraphLineModelV2  # noqa: E402
from src.training.pretrain.config import (  # noqa: E402
    dataset_kwargs_from_args,
    parse_arguments,
)
from src.utils import get_data_loader  # noqa: E402


DEFAULT_REPORT = ROOT / "results/mts_glt_v2/final_report.json"
DEFAULT_CONFIG = ROOT / "configs/mts/glt_v2_formal_a6_h_w1_20k.json"
DEFAULT_OUTPUT = ROOT / "results/mts_glt_v2/alignment_audit_v1"
ALPHAS = np.asarray([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0], dtype=np.float64)


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _projection(output_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(512),
        nn.Linear(512, 512),
        nn.GELU(),
        nn.Linear(512, int(output_dim)),
    )


def _load_components(report_path: Path, device: torch.device):
    report = _json(report_path)
    checkpoint = Path(report["selected_checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "mts-glt-v2-probe-v1":
        raise RuntimeError(f"unexpected GLT-v2 probe schema: {payload.get('schema')}")
    state = payload["state_dict"]
    model = MTSGraphLineModelV2(
        glt_layers=int(payload["glt_layers"]),
        glt_attention_variant=str(payload["glt_attention_variant"]),
        use_compact19=False,
    )
    model_state = {
        key[len("model."):]: value
        for key, value in state.items()
        if key.startswith("model.")
    }
    model.load_state_dict(model_state, strict=True)
    projectors = {}
    for name in ("o8_projection", "glt_projection"):
        module = _projection(256)
        prefix = name + "."
        module.load_state_dict({
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }, strict=True)
        projectors[name] = module.to(device).eval()
    model = model.to(device).eval()
    provenance = {
        "formal_report": str(report_path.resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_schema": payload["schema"],
        "checkpoint_step": int(payload["step"]),
        "formal_trajectory_max_steps": int(
            report.get("training_resources", {}).get("steps", 0)
        ),
        "glt_layers": int(payload["glt_layers"]),
        "glt_attention_variant": str(payload["glt_attention_variant"]),
        "infonce_weight": float(payload["infonce_weight"]),
    }
    return model, projectors["o8_projection"], projectors["glt_projection"], provenance


def _select_graphs(valid_path: Path, maximum: int, seed: int):
    valid = np.load(valid_path, mmap_mode="r")
    valid_ids = np.flatnonzero(valid)
    count = min(int(maximum), int(valid_ids.size))
    rng = np.random.default_rng(int(seed))
    selected = np.sort(rng.choice(valid_ids, size=count, replace=False)).astype(np.int64)
    permutation = np.random.default_rng(int(seed)).permutation(selected)
    train_end = int(math.floor(0.70 * count))
    val_end = train_end + int(math.floor(0.15 * count))
    split = {
        "train": np.sort(permutation[:train_end]),
        "validation": np.sort(permutation[train_end:val_end]),
        "test": np.sort(permutation[val_end:]),
    }
    return selected, split, int(valid_ids.size), int(valid.size)


def _concatenate(parts, *, dtype=None):
    value = torch.cat(parts, dim=0).numpy()
    return value.astype(dtype, copy=False) if dtype is not None else value


def _extract(
    dataset,
    selected: np.ndarray,
    model,
    o8_projection,
    glt_projection,
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
):
    loader = get_data_loader(
        dataset,
        indices=selected,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(workers) > 0,
        prefetch_factor=2,
    )
    graph_o8, graph_glt, projection_o8, projection_glt = [], [], [], []
    graph_ids, graph_hashes = [], []
    atom_o8, atom_glt, atom_graph, atom_local, atom_z, atom_valid = [], [], [], [], [], []
    incident_counts = []
    internal_incidence = 0
    cross_incidence = 0
    valid_line_tokens = 0
    total_line_tokens = 0
    cursor = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            local_graph_count = len(batch.smiles)
            local_ids = selected[cursor:cursor + local_graph_count]
            cursor += local_graph_count
            batch = batch.to(device, non_blocking=True)
            if not bool(batch.glt_geometry_valid.bool().all()):
                raise RuntimeError("preselected geometry-valid batch contains invalid graph")
            z_o8, canonical_states = model._o8_canonical_states(batch)
            glt = model.glt(batch)
            if not torch.isfinite(glt["line_states"]).all():
                raise RuntimeError("non-finite final normalized line state")
            if not torch.isfinite(canonical_states).all() or not torch.isfinite(
                glt["atom_geometry_states"]
            ).all():
                raise RuntimeError("non-finite canonical atom representation")
            po8 = o8_projection(z_o8)
            pglt = glt_projection(glt["graph_geometry"])
            graph_o8.append(z_o8.float().cpu())
            graph_glt.append(glt["graph_geometry"].float().cpu())
            projection_o8.append(po8.float().cpu())
            projection_glt.append(pglt.float().cpu())
            graph_ids.append(torch.as_tensor(local_ids, dtype=torch.long))
            graph_hashes.append(batch.mts_sample_hash64.detach().cpu().long())

            local_id_tensor = torch.as_tensor(local_ids, device=device, dtype=torch.long)
            atom_o8.append(canonical_states.float().cpu())
            atom_glt.append(glt["atom_geometry_states"].float().cpu())
            atom_graph.append(local_id_tensor[batch.canonical_graph_index.long()].cpu())
            atom_local.append(batch.canonical_local_index.detach().cpu().long())
            atom_z.append(batch.canonical_atomic_numbers.detach().cpu().long())
            atom_valid.append(glt["atom_geometry_valid"].detach().cpu().bool())

            token_valid = batch.glt_token_valid.bool()
            token_shift = batch.glt_token_shift.long()
            counts = torch.zeros(
                canonical_states.size(0), device=device, dtype=torch.long
            )
            endpoints = torch.cat([
                batch.glt_token_atom_a.long()[token_valid],
                batch.glt_token_atom_b.long()[token_valid],
            ])
            if endpoints.numel():
                counts.index_add_(
                    0, endpoints, torch.ones_like(endpoints, dtype=torch.long)
                )
            incident_counts.append(counts.cpu())
            valid_line_tokens += int(token_valid.sum())
            total_line_tokens += int(token_valid.numel())
            internal_incidence += 2 * int((token_valid & (token_shift == 0)).sum())
            cross_incidence += 2 * int((token_valid & (token_shift.abs() == 1)).sum())
            if batch_index and batch_index % 25 == 0:
                print(f"extracted graphs={cursor}/{len(selected)}", flush=True)
    if cursor != len(selected):
        raise RuntimeError(f"extraction row mismatch: {cursor} != {len(selected)}")
    arrays = {
        "graph_id": _concatenate(graph_ids, dtype=np.int64),
        "graph_hash64": _concatenate(graph_hashes, dtype=np.int64),
        "z_o8": _concatenate(graph_o8, dtype=np.float32),
        "z_glt": _concatenate(graph_glt, dtype=np.float32),
        "p_o8": _concatenate(projection_o8, dtype=np.float32),
        "p_glt": _concatenate(projection_glt, dtype=np.float32),
        "atom_o8": _concatenate(atom_o8, dtype=np.float32),
        "atom_glt": _concatenate(atom_glt, dtype=np.float32),
        "atom_graph_id": _concatenate(atom_graph, dtype=np.int64),
        "atom_local_id": _concatenate(atom_local, dtype=np.int64),
        "atom_z": _concatenate(atom_z, dtype=np.int64),
        "atom_valid": _concatenate(atom_valid, dtype=bool),
        "incident_count": _concatenate(incident_counts, dtype=np.int64),
    }
    for name in ("z_o8", "z_glt", "p_o8", "p_glt", "atom_o8", "atom_glt"):
        if not np.isfinite(arrays[name]).all():
            raise RuntimeError(f"non-finite extracted representation: {name}")
    incidence = {
        "total_valid_graphs": int(len(selected)),
        "total_canonical_atoms": int(len(arrays["atom_graph_id"])),
        "total_line_tokens": int(total_line_tokens),
        "valid_line_tokens": int(valid_line_tokens),
        "internal_incidence_count": int(internal_incidence),
        "cross_ru_incidence_count": int(cross_incidence),
        "signed_shift_counts": None,
        "signed_shift_note": (
            "The token stores a canonical signed shift, but v2 applies abs(shift) "
            "to both endpoint incidences; endpoint-specific signed incidence is not "
            "unambiguously represented and is therefore not reported."
        ),
    }
    return arrays, incidence


def _indices_for_graphs(atom_graph_ids, graph_ids):
    return np.flatnonzero(np.isin(atom_graph_ids, graph_ids))


def _multioutput_r2(target, prediction):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    numerator = np.sum((target - prediction) ** 2, axis=0)
    denominator = np.sum((target - target.mean(axis=0)) ** 2, axis=0)
    valid = denominator > 1e-12
    scores = 1.0 - numerator[valid] / denominator[valid]
    return float(np.mean(scores))


class _RidgeMap:
    def __init__(self, x_mean, x_scale, y_mean, y_scale, coefficient, alpha):
        self.x_mean = x_mean
        self.x_scale = x_scale
        self.y_mean = y_mean
        self.y_scale = y_scale
        self.coefficient = coefficient
        self.alpha = float(alpha)

    def predict(self, features, *, raw=True):
        scaled = (
            (np.asarray(features, dtype=np.float32) - self.x_mean) / self.x_scale
        ) @ self.coefficient
        return scaled * self.y_scale + self.y_mean if raw else scaled


def _fit_ridge(train_x, train_y, val_x, val_y, test_x, test_y):
    train_x = np.asarray(train_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.float32)
    x_mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    y_mean = train_y.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_scale = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    y_scale = train_y.std(axis=0, dtype=np.float64).astype(np.float32)
    x_scale[x_scale < 1e-8] = 1.0
    y_scale[y_scale < 1e-8] = 1.0
    tx = (train_x - x_mean) / x_scale
    ty = (train_y - y_mean) / y_scale
    vx = (np.asarray(val_x, dtype=np.float32) - x_mean) / x_scale
    vy = (np.asarray(val_y, dtype=np.float32) - y_mean) / y_scale
    xtx = np.asarray(tx.T @ tx, dtype=np.float64)
    xty = np.asarray(tx.T @ ty, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(xtx)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected = eigenvectors.T @ xty
    best = None
    for alpha in ALPHAS:
        coefficient = eigenvectors @ (
            projected / (eigenvalues[:, None] + float(alpha))
        )
        val_prediction = vx @ coefficient
        score = _multioutput_r2(vy, val_prediction)
        if best is None or score > best[0]:
            best = (score, float(alpha), coefficient)
    fitted = _RidgeMap(
        x_mean, x_scale, y_mean, y_scale,
        np.asarray(best[2], dtype=np.float32), best[1],
    )
    test_prediction = fitted.predict(test_x, raw=True)
    return {
        "model": fitted,
        "alpha": fitted.alpha,
        "validation_mean_r2": float(best[0]),
        "test_mean_r2": _multioutput_r2(test_y, test_prediction),
    }


def _effective_rank(features):
    values = np.asarray(features, dtype=np.float64)
    values = values - values.mean(axis=0, keepdims=True)
    covariance = values.T @ values / max(1, len(values) - 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum())
    if total <= 0:
        return {"effective_rank": 0.0, "effective_rank_ratio": 0.0}
    probabilities = eigenvalues[eigenvalues > 0] / total
    rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    return {
        "effective_rank": rank,
        "effective_rank_ratio": rank / float(values.shape[1]),
        "hidden_dim": int(values.shape[1]),
    }


def _balanced_atom_indices(graph_ids, local_ids, *, limit=32, seed=42):
    selected = []
    for graph_id in np.unique(graph_ids):
        indices = np.flatnonzero(graph_ids == graph_id)
        if len(indices) <= int(limit):
            selected.extend(indices.tolist())
            continue
        order = np.argsort(local_ids[indices], kind="stable")
        ordered = indices[order]
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(graph_id)]))
        chosen = np.sort(rng.choice(len(ordered), size=int(limit), replace=False))
        selected.extend(ordered[chosen].tolist())
    return np.asarray(selected, dtype=np.int64)


def _cosine(first, second):
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    return np.sum(first * second, axis=1) / np.maximum(denominator, 1e-12)


def _negative_indices(graph_ids, local_ids, elements, *, seed=42):
    size = len(graph_ids)
    result = np.full(size, -1, dtype=np.int64)
    by_graph_element = {}
    by_element = {}
    for index, (graph, element) in enumerate(zip(graph_ids, elements)):
        by_graph_element.setdefault((int(graph), int(element)), []).append(index)
        by_element.setdefault(int(element), []).append(index)
    within = 0
    cross = 0
    for index in range(size):
        local = by_graph_element[(int(graph_ids[index]), int(elements[index]))]
        if len(local) > 1:
            position = local.index(index)
            result[index] = local[(position + 1) % len(local)]
            within += 1
            continue
        candidates = by_element[int(elements[index])]
        if len(candidates) > 1:
            rng = np.random.default_rng(np.random.SeedSequence([
                int(seed), int(graph_ids[index]), int(local_ids[index]), int(elements[index])
            ]))
            start = int(rng.integers(0, len(candidates)))
            for offset in range(len(candidates)):
                candidate = candidates[(start + offset) % len(candidates)]
                if int(graph_ids[candidate]) != int(graph_ids[index]):
                    result[index] = candidate
                    cross += 1
                    break
    return result, {"within_polymer": within, "cross_polymer": cross}


def _element_symbol(atomic_number: int) -> str:
    try:
        from rdkit import Chem
        return str(Chem.GetPeriodicTable().GetElementSymbol(int(atomic_number)))
    except Exception:
        return f"Z{int(atomic_number)}"


def _specificity(
    fitted, test_o8, test_glt, graph_ids, local_ids, elements, *, seed=42
):
    predicted = fitted.predict(test_o8, raw=True)
    negatives, sources = _negative_indices(
        graph_ids, local_ids, elements, seed=int(seed)
    )
    usable = negatives >= 0
    matched = _cosine(predicted[usable], test_glt[usable])
    negative = _cosine(predicted[usable], test_glt[negatives[usable]])
    margin = matched - negative
    element_rows = []
    usable_elements = elements[usable]
    for element in np.unique(usable_elements):
        mask = usable_elements == element
        if int(mask.sum()) < 100:
            continue
        element_rows.append({
            "atomic_number": int(element),
            "element": _element_symbol(int(element)),
            "count": int(mask.sum()),
            "matched_similarity": float(matched[mask].mean()),
            "negative_similarity": float(negative[mask].mean()),
            "matched_margin": float(margin[mask].mean()),
            "median_margin": float(np.median(margin[mask])),
            "positive_margin_fraction": float(np.mean(margin[mask] > 0)),
        })
    summary = {
        "count": int(usable.sum()),
        "anchors_without_same_element_negative": int((~usable).sum()),
        "negative_sources": sources,
        "matched_similarity": float(matched.mean()),
        "negative_similarity": float(negative.mean()),
        "matched_margin_mean": float(margin.mean()),
        "matched_margin_median": float(np.median(margin)),
        "positive_margin_atom_fraction": float(np.mean(margin > 0)),
    }
    return summary, element_rows


def _incidence_statistics(arrays, base):
    counts = arrays["incident_count"]
    atom_valid = arrays["atom_valid"]
    base.update({
        "atoms_with_incident_line": int(np.count_nonzero(counts > 0)),
        "atoms_without_incident_line": int(np.count_nonzero(counts == 0)),
        "canonical_atom_coverage_ratio": float(np.mean(atom_valid)),
        "incident_lines_per_atom": {
            "mean": float(np.mean(counts)),
            "median": float(np.median(counts)),
            "p25": float(np.quantile(counts, 0.25)),
            "p75": float(np.quantile(counts, 0.75)),
            "max": int(np.max(counts)),
        },
        "atom_valid_mask_matches_positive_incidence": bool(
            np.array_equal(atom_valid, counts > 0)
        ),
    })
    return base


def run(args) -> dict:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA alignment audit requested but CUDA is unavailable")
    model, o8_projection, glt_projection, provenance = _load_components(
        Path(args.formal_report).resolve(), device
    )
    formal_config = Path(args.formal_config).resolve()
    config_payload = _json(formal_config)
    resolved = ROOT / "results/mts_glt_v2/formal/a6_h_w1_20k/resolved_input.json"
    provenance.update({
        "formal_config": str(formal_config),
        "resolved_input": str(resolved.resolve()),
        "dataset": str(config_payload["dataset_name"]),
        "selection_note": (
            "The formal 20k trajectory selected its 5k probe for downstream "
            "evaluation; this audit follows final_report.json rather than the filename."
        ),
    })

    valid_path = Path(config_payload["line_sidecar_root"])
    if not valid_path.is_absolute():
        valid_path = ROOT / valid_path
    valid_path = valid_path / "graph_geometry_valid.npy"
    selected, splits, total_valid, dataset_rows = _select_graphs(
        valid_path, int(args.max_graphs), int(args.seed)
    )
    _atomic_json(output / "sampled_graph_ids.json", {
        "seed": int(args.seed),
        "method": "uniform_without_replacement_from_geometry_valid_PI1M_v2_rows",
        "heldout_split_available": False,
        "dataset_rows": dataset_rows,
        "geometry_valid_rows": total_valid,
        "sampled_graph_ids": selected.tolist(),
    })
    _atomic_json(output / "graph_split_ids.json", {
        "seed": int(args.seed),
        "split_ratio": [0.70, 0.15, 0.15],
        "train_graph_ids": splits["train"].tolist(),
        "validation_graph_ids": splits["validation"].tolist(),
        "test_graph_ids": splits["test"].tolist(),
    })

    dataset_args = parse_arguments([
        "--experiment_config", str(formal_config),
    ])
    dataset = UniDataset(**dataset_kwargs_from_args(dataset_args))
    arrays, incidence_base = _extract(
        dataset, selected, model, o8_projection, glt_projection,
        device=device, batch_size=int(args.batch_size), workers=int(args.workers),
    )
    if not np.array_equal(arrays["graph_id"], selected):
        raise RuntimeError("loader graph order differs from selected graph IDs")
    _atomic_json(output / "sampled_graph_ids.json", {
        "seed": int(args.seed),
        "method": "uniform_without_replacement_from_geometry_valid_PI1M_v2_rows",
        "heldout_split_available": False,
        "dataset_rows": dataset_rows,
        "geometry_valid_rows": total_valid,
        "sampled_graph_ids": selected.tolist(),
        "sample_hash64": arrays["graph_hash64"].tolist(),
    })

    graph_position = {int(value): index for index, value in enumerate(arrays["graph_id"])}
    graph_indices = {
        name: np.asarray([graph_position[int(value)] for value in ids], dtype=np.int64)
        for name, ids in splits.items()
    }
    graph_test = graph_indices["test"]
    normalized_po8 = arrays["p_o8"] / np.maximum(
        np.linalg.norm(arrays["p_o8"], axis=1, keepdims=True), 1e-12
    )
    normalized_pglt = arrays["p_glt"] / np.maximum(
        np.linalg.norm(arrays["p_glt"], axis=1, keepdims=True), 1e-12
    )
    graph_forward = _fit_ridge(
        arrays["z_o8"][graph_indices["train"]], arrays["z_glt"][graph_indices["train"]],
        arrays["z_o8"][graph_indices["validation"]], arrays["z_glt"][graph_indices["validation"]],
        arrays["z_o8"][graph_test], arrays["z_glt"][graph_test],
    )
    graph_reverse = _fit_ridge(
        arrays["z_glt"][graph_indices["train"]], arrays["z_o8"][graph_indices["train"]],
        arrays["z_glt"][graph_indices["validation"]], arrays["z_o8"][graph_indices["validation"]],
        arrays["z_glt"][graph_test], arrays["z_o8"][graph_test],
    )
    graph_metrics = {
        "num_test_graphs": int(len(graph_test)),
        "graph_raw_cka": float(linear_cka(
            arrays["z_o8"][graph_test], arrays["z_glt"][graph_test]
        )),
        "graph_projected_cka": float(linear_cka(
            normalized_po8[graph_test], normalized_pglt[graph_test]
        )),
        "o8_to_glt": {key: value for key, value in graph_forward.items() if key != "model"},
        "glt_to_o8": {key: value for key, value in graph_reverse.items() if key != "model"},
        "o8_effective_rank": _effective_rank(arrays["z_o8"][graph_test]),
        "glt_effective_rank": _effective_rank(arrays["z_glt"][graph_test]),
    }

    valid_atoms = arrays["atom_valid"]
    atom_indices = {
        name: _indices_for_graphs(arrays["atom_graph_id"], ids)
        for name, ids in splits.items()
    }
    atom_indices = {
        name: indices[valid_atoms[indices]] for name, indices in atom_indices.items()
    }
    atom_train, atom_val, atom_test = (
        atom_indices["train"], atom_indices["validation"], atom_indices["test"]
    )
    atom_forward = _fit_ridge(
        arrays["atom_o8"][atom_train], arrays["atom_glt"][atom_train],
        arrays["atom_o8"][atom_val], arrays["atom_glt"][atom_val],
        arrays["atom_o8"][atom_test], arrays["atom_glt"][atom_test],
    )
    atom_reverse = _fit_ridge(
        arrays["atom_glt"][atom_train], arrays["atom_o8"][atom_train],
        arrays["atom_glt"][atom_val], arrays["atom_o8"][atom_val],
        arrays["atom_glt"][atom_test], arrays["atom_o8"][atom_test],
    )
    balanced_local = _balanced_atom_indices(
        arrays["atom_graph_id"][atom_test], arrays["atom_local_id"][atom_test],
        limit=32, seed=int(args.seed),
    )
    balanced_test = atom_test[balanced_local]
    specificity, element_rows = _specificity(
        atom_forward["model"],
        arrays["atom_o8"][atom_test], arrays["atom_glt"][atom_test],
        arrays["atom_graph_id"][atom_test], arrays["atom_local_id"][atom_test],
        arrays["atom_z"][atom_test], seed=int(args.seed),
    )
    atom_metrics = {
        "diagnostic_atom_mask": "downstream atom_geometry_valid only",
        "num_train_atoms": int(len(atom_train)),
        "num_validation_atoms": int(len(atom_val)),
        "num_test_atoms": int(len(atom_test)),
        "num_balanced_test_atoms": int(len(balanced_test)),
        "atom_cka_all": float(linear_cka(
            arrays["atom_o8"][atom_test], arrays["atom_glt"][atom_test]
        )),
        "atom_cka_graph_balanced": float(linear_cka(
            arrays["atom_o8"][balanced_test], arrays["atom_glt"][balanced_test]
        )),
        "o8_to_glt": {key: value for key, value in atom_forward.items() if key != "model"},
        "glt_to_o8": {key: value for key, value in atom_reverse.items() if key != "model"},
        "matched_specificity": specificity,
        "o8_effective_rank": _effective_rank(arrays["atom_o8"][balanced_test]),
        "glt_effective_rank": _effective_rank(arrays["atom_glt"][balanced_test]),
    }
    with (output / "per_element_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "atomic_number", "element", "count", "matched_similarity",
            "negative_similarity", "matched_margin", "median_margin",
            "positive_margin_fraction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(element_rows)

    incidence = _incidence_statistics(arrays, incidence_base)
    graph_mean = 0.5 * (
        graph_forward["test_mean_r2"] + graph_reverse["test_mean_r2"]
    )
    atom_mean = 0.5 * (
        atom_forward["test_mean_r2"] + atom_reverse["test_mean_r2"]
    )
    gap = graph_mean - atom_mean
    if graph_mean >= 0.80 and gap >= 0.15:
        pattern = "strong_global_weaker_local"
    elif graph_mean >= 0.80 and atom_mean >= 0.80:
        pattern = "strong_global_and_local"
    elif graph_mean < 0.80 and atom_mean < 0.80:
        pattern = "weak_or_moderate_both"
    else:
        pattern = "mixed"
    summary = {
        "schema": "mts-glt-v2-alignment-audit-v1",
        "provenance": provenance,
        "checkpoint": provenance["checkpoint"],
        "dataset": config_payload["dataset_name"],
        "seed": int(args.seed),
        "num_graphs": int(len(selected)),
        "num_atoms": int(len(arrays["atom_graph_id"])),
        "num_alignment_valid_atoms": int(valid_atoms.sum()),
        "train_graphs": int(len(splits["train"])),
        "validation_graphs": int(len(splits["validation"])),
        "test_graphs": int(len(splits["test"])),
        "graph_raw_cka": graph_metrics["graph_raw_cka"],
        "graph_projected_cka": graph_metrics["graph_projected_cka"],
        "graph_r2_o8_to_glt": graph_forward["test_mean_r2"],
        "graph_r2_glt_to_o8": graph_reverse["test_mean_r2"],
        "atom_cka_all": atom_metrics["atom_cka_all"],
        "atom_cka_graph_balanced": atom_metrics["atom_cka_graph_balanced"],
        "atom_r2_o8_to_glt": atom_forward["test_mean_r2"],
        "atom_r2_glt_to_o8": atom_reverse["test_mean_r2"],
        "graph_mean_r2": float(graph_mean),
        "atom_mean_r2": float(atom_mean),
        "alignment_r2_gap": float(gap),
        "matched_similarity": specificity["matched_similarity"],
        "negative_similarity": specificity["negative_similarity"],
        "matched_margin_mean": specificity["matched_margin_mean"],
        "matched_margin_median": specificity["matched_margin_median"],
        "positive_margin_atom_fraction": specificity["positive_margin_atom_fraction"],
        "graph_o8_effective_rank": graph_metrics["o8_effective_rank"]["effective_rank"],
        "graph_glt_effective_rank": graph_metrics["glt_effective_rank"]["effective_rank"],
        "atom_o8_effective_rank": atom_metrics["o8_effective_rank"]["effective_rank"],
        "atom_glt_effective_rank": atom_metrics["glt_effective_rank"]["effective_rank"],
        "canonical_atom_coverage_ratio": incidence["canonical_atom_coverage_ratio"],
        "alignment_pattern": pattern,
        "anomalies": [
            "Formal downstream selection used the 5k probe from a completed 20k trajectory; "
            "the audit follows that selected checkpoint exactly."
        ] if provenance["checkpoint_step"] != 20000 else [],
    }
    _atomic_json(output / "graph_level_metrics.json", graph_metrics)
    _atomic_json(output / "atom_level_metrics.json", atom_metrics)
    _atomic_json(output / "incidence_statistics.json", incidence)
    _atomic_json(output / "alignment_audit_summary.json", summary)
    print(json.dumps({
        "status": "complete",
        "alignment_pattern": pattern,
        "summary": str(output / "alignment_audit_summary.json"),
    }, sort_keys=True), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-report", default=str(DEFAULT_REPORT))
    parser.add_argument("--formal-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--max-graphs", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
