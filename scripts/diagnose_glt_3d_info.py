#!/usr/bin/env python3
"""Frozen-feature route-selection diagnostic for GLT-3D-INFO-20260917-01.

This diagnostic never trains an encoder and never consumes outer-test labels or
predictions.  It extracts the already-frozen Fixed Concat representation once,
derives two CPU descriptors from the frozen Trimer coordinates, and evaluates
Ridge probes on the train/validation portions of the existing
``outer5_inner20`` manifests only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem  # noqa: F401  # keep the frozen chemistry environment explicit
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# Make both ``python -m scripts...`` and direct script execution resolve the
# repository's namespace packages identically.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.create_mips_split_manifests import build_manifest
from src.dataset.glt_dual import build_dual_sample, dual_glt_collate
from src.modules.glt_dual import build_dual_glt_model
from src.modules.glt_dual_pretrain import load_deployment
from src.training.glt_dual_runtime import open_source, write_json


PLAN_ID = "GLT-3D-INFO-20260917-01"
TASKS = ("eat", "eea", "egb", "ei", "eps", "nc", "xc")
PROBES = ("P0", "P1", "P2", "P3", "P4")
ALPHAS = (0.1, 1.0, 10.0, 100.0)
RBF_CENTERS = np.linspace(2.0, 6.0, 16, dtype=np.float64)
RBF_SIGMA = float(RBF_CENTERS[1] - RBF_CENTERS[0])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp.npz", delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _as_numpy(value, dtype=None):
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _require_trimer_fields(trimer):
    names = (
        "trimer_pos", "trimer_base_ru_atom_id", "trimer_ru_offset",
        "trimer_edge_index", "trimer_atomic_number",
    )
    missing = [name for name in names if not hasattr(trimer, name)]
    if missing:
        raise ValueError("trimer identity contract missing: " + ",".join(missing))


def _dihedral(points: np.ndarray, quad: tuple[int, int, int, int]) -> float | None:
    p0, p1, p2, p3 = (points[index] for index in quad)
    if not np.isfinite(np.asarray([p0, p1, p2, p3])).all():
        return None
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = float(np.linalg.norm(b1))
    if not np.isfinite(b1_norm) or b1_norm <= 1e-12:
        return None
    b1 = b1 / b1_norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    v_norm, w_norm = float(np.linalg.norm(v)), float(np.linalg.norm(w))
    if (not np.isfinite(v_norm) or not np.isfinite(w_norm)
            or v_norm <= 1e-12 or w_norm <= 1e-12):
        return None
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1, v), w))
    value = float(np.arctan2(y, x))
    return value if np.isfinite(value) else None


def proper_torsion_descriptor(static: dict, trimer) -> tuple[np.ndarray, int, int]:
    """Return 10-D Fourier/count descriptor and (valid, candidate) counts.

    A candidate is one non-self, exactly-two-hop line path whose middle token
    is a center physical bond.  The endpoint atoms are recovered from adjacent
    physical-bond endpoint sets; malformed identity is a contract error, not a
    geometry fallback.  Reversed quadruplets are canonicalized lexicographically
    before the signed dihedral is evaluated.
    """

    descriptor = np.zeros(10, dtype=np.float64)
    if not bool(static["geometry_valid"]):
        return descriptor, 0, 0
    _require_trimer_fields(trimer)
    positions = _as_numpy(trimer.trimer_pos, np.float64)
    token_a = _as_numpy(static["token_pos_index_a"], np.int64).reshape(-1)
    token_b = _as_numpy(static["token_pos_index_b"], np.int64).reshape(-1)
    path = _as_numpy(static["line_path"], np.int64)
    path_mask = _as_numpy(static["line_path_mask"], bool)
    self_flags = _as_numpy(static["line_is_self"], bool).reshape(-1)
    center = _as_numpy(static["token_center_mask"], bool).reshape(-1)
    if token_a.shape != token_b.shape or path.ndim != 2 or path.shape[1] != 3:
        raise ValueError("torsion static arrays have inconsistent shapes")
    if path_mask.shape != (path.shape[0], 2) or self_flags.size != path.shape[0]:
        raise ValueError("torsion line-path arrays have inconsistent shapes")
    if np.any(token_a < 0) or np.any(token_b < 0) or np.any(token_a >= len(positions)) or np.any(token_b >= len(positions)):
        raise ValueError("torsion token coordinate index is out of range")

    endpoint_sets = [set((int(token_a[index]), int(token_b[index])))
                     for index in range(token_a.size)]
    angles = []
    candidates = 0
    for row in range(path.shape[0]):
        if bool(self_flags[row]) or int(path_mask[row].sum()) != 2:
            continue
        tokens = path[row, :3]
        if np.any(tokens < 0) or np.any(tokens >= token_a.size):
            raise ValueError("torsion line path token index is out of range")
        middle = int(tokens[1])
        if not bool(center[middle]):
            continue
        candidates += 1
        first, second, third = (endpoint_sets[int(token)] for token in tokens)
        shared_left = first & second
        shared_right = second & third
        if len(shared_left) != 1 or len(shared_right) != 1:
            continue
        b = next(iter(shared_left))
        c = next(iter(shared_right))
        left = first - {b}
        right = third - {c}
        if len(left) != 1 or len(right) != 1:
            continue
        a, d = next(iter(left)), next(iter(right))
        quadruplet = (int(a), int(b), int(c), int(d))
        if len(set(quadruplet)) != 4:
            continue
        canonical = min(quadruplet, quadruplet[::-1])
        value = _dihedral(positions, canonical)
        if value is not None:
            angles.append(value)

    if angles:
        values = np.asarray(angles, dtype=np.float64)
        for k in range(1, 5):
            descriptor[2 * (k - 1)] = float(np.mean(np.sin(k * values)))
            descriptor[2 * (k - 1) + 1] = float(np.mean(np.cos(k * values)))
    valid = len(angles)
    descriptor[8] = float(valid)
    descriptor[9] = float(valid / candidates) if candidates else 0.0
    return descriptor, valid, candidates


def _build_neighbors(edge_index, count: int) -> list[set[int]]:
    edge = _as_numpy(edge_index, np.int64)
    if edge.ndim != 2 or edge.shape[0] != 2:
        raise ValueError("trimer edge index must have shape [2,E]")
    neighbors = [set() for _ in range(count)]
    for a, b in edge.T.tolist():
        a, b = int(a), int(b)
        if a < 0 or b < 0 or a >= count or b >= count:
            raise ValueError("trimer edge index is out of range")
        if a == b:
            continue
        neighbors[a].add(b)
        neighbors[b].add(a)
    return neighbors


def _radial_summary(distances: list[float]) -> tuple[np.ndarray, int]:
    if not distances:
        return np.zeros(16, dtype=np.float64), 0
    values = np.asarray(distances, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("nonbonded distance is nonfinite")
    basis = np.exp(-0.5 * ((values[:, None] - RBF_CENTERS[None, :]) / RBF_SIGMA) ** 2)
    summary = basis.mean(axis=0)
    if not np.isfinite(summary).all():
        raise ValueError("nonbonded RBF summary is nonfinite")
    return summary, int(values.size)


def nonbonded_descriptor(static: dict, trimer) -> tuple[np.ndarray, dict]:
    """Return 34-D center-center/center-neighbor radial descriptor."""

    descriptor = np.zeros(34, dtype=np.float64)
    stats = {
        "center_center_pairs": 0, "center_neighbor_pairs": 0,
        "center_center_candidate_pairs": 0, "center_neighbor_candidate_pairs": 0,
    }
    if not bool(static["geometry_valid"]):
        return descriptor, stats
    _require_trimer_fields(trimer)
    positions = _as_numpy(trimer.trimer_pos, np.float64)
    base = _as_numpy(trimer.trimer_base_ru_atom_id, np.int64).reshape(-1)
    offsets = _as_numpy(trimer.trimer_ru_offset, np.int64).reshape(-1)
    atomic = _as_numpy(trimer.trimer_atomic_number, np.int64).reshape(-1)
    if positions.ndim != 2 or positions.shape[1] != 3 or base.size != positions.shape[0] or offsets.size != positions.shape[0] or atomic.size != positions.shape[0]:
        raise ValueError("nonbonded Trimer identity dimensions disagree")
    heavy_field = getattr(trimer, "trimer_heavy_indices", None)
    heavy = (_as_numpy(heavy_field, np.int64).reshape(-1)
             if heavy_field is not None else np.flatnonzero(atomic > 1).astype(np.int64))
    if heavy.size and (heavy.min() < 0 or heavy.max() >= positions.shape[0]):
        raise ValueError("trimer heavy index is out of range")
    if len(set(heavy.tolist())) != heavy.size:
        raise ValueError("trimer heavy index is not unique")
    neighbors = _build_neighbors(trimer.trimer_edge_index, positions.shape[0])
    center = [int(index) for index in heavy if int(offsets[index]) == 0]
    neighbor = [int(index) for index in heavy if int(offsets[index]) in (-1, 1)]
    center_pairs = []
    neighbor_pairs = []
    for left_index, left in enumerate(center):
        for right in center[left_index + 1:]:
            stats["center_center_candidate_pairs"] += 1
            if right in neighbors[left] or neighbors[left] & neighbors[right]:
                continue
            distance = float(np.linalg.norm(positions[left] - positions[right]))
            if np.isfinite(distance) and 2.0 <= distance <= 6.0:
                center_pairs.append(distance)
    for left in center:
        for right in neighbor:
            stats["center_neighbor_candidate_pairs"] += 1
            if right in neighbors[left] or neighbors[left] & neighbors[right]:
                continue
            distance = float(np.linalg.norm(positions[left] - positions[right]))
            if np.isfinite(distance) and 2.0 <= distance <= 6.0:
                neighbor_pairs.append(distance)
    center_summary, center_count = _radial_summary(center_pairs)
    neighbor_summary, neighbor_count = _radial_summary(neighbor_pairs)
    descriptor[:16] = center_summary
    descriptor[16:32] = neighbor_summary
    descriptor[32] = float(center_count)
    descriptor[33] = float(neighbor_count)
    stats["center_center_pairs"] = center_count
    stats["center_neighbor_pairs"] = neighbor_count
    return descriptor, stats


def _probe_matrix(name: str, h2: np.ndarray, h3: np.ndarray,
                  torsion: np.ndarray, nonbonded: np.ndarray) -> np.ndarray:
    if name == "P0":
        return h2
    if name == "P1":
        return np.concatenate((h2, h3), axis=1)
    if name == "P2":
        return np.concatenate((h2, torsion), axis=1)
    if name == "P3":
        return np.concatenate((h2, nonbonded), axis=1)
    if name == "P4":
        return np.concatenate((h2, torsion, nonbonded), axis=1)
    raise ValueError(f"unknown probe {name}")


def _fit_ridge(X_train, y_train, X_eval, y_eval, groups):
    unique_groups = np.unique(groups)
    if unique_groups.size < 5:
        raise ValueError("outer-train has fewer than five canonical structures")
    inner = GroupKFold(n_splits=5)
    alpha_scores = {}
    for alpha in ALPHAS:
        scores = []
        for inner_train, inner_eval in inner.split(X_train, y_train, groups):
            x_scaler = StandardScaler().fit(X_train[inner_train])
            y_scaler = StandardScaler().fit(y_train[inner_train].reshape(-1, 1))
            model = Ridge(alpha=float(alpha))
            model.fit(
                x_scaler.transform(X_train[inner_train]),
                y_scaler.transform(y_train[inner_train].reshape(-1, 1)).reshape(-1),
            )
            prediction = y_scaler.inverse_transform(
                model.predict(x_scaler.transform(X_train[inner_eval])).reshape(-1, 1)
            ).reshape(-1)
            score = float(r2_score(y_train[inner_eval], prediction))
            if np.isfinite(score):
                scores.append(score)
        if not scores:
            raise FloatingPointError(f"no finite inner Ridge score for alpha={alpha}")
        alpha_scores[float(alpha)] = float(np.mean(scores))
    selected_alpha = max(ALPHAS, key=lambda alpha: (alpha_scores[float(alpha)], -float(alpha)))
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    model = Ridge(alpha=float(selected_alpha))
    model.fit(x_scaler.transform(X_train), y_scaler.transform(y_train.reshape(-1, 1)).reshape(-1))
    prediction = y_scaler.inverse_transform(
        model.predict(x_scaler.transform(X_eval)).reshape(-1, 1)
    ).reshape(-1)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("nonfinite outer-validation Ridge prediction")
    return selected_alpha, prediction, alpha_scores


def _feature_sidecar(source, model, device, batch_size: int):
    key_to_unique = {}
    unique_indices = []
    row_feature_index = np.empty(len(source), dtype=np.int32)
    for row, (key, smiles) in enumerate(source.samples):
        key = bytes(key)
        if key not in key_to_unique:
            key_to_unique[key] = len(unique_indices)
            unique_indices.append(row)
        row_feature_index[row] = key_to_unique[key]
    keys = list(key_to_unique)
    h2_rows, h3_rows, torsion_rows, nonbonded_rows = [], [], [], []
    geometry_valid_rows, graph3d_valid_rows = [], []
    torsion_valid_rows, torsion_candidate_rows = [], []
    pair_rows = []
    model.eval()
    for start in range(0, len(unique_indices), batch_size):
        chunk_indices = unique_indices[start:start + batch_size]
        data_rows, torsion_chunk, nonbonded_chunk = [], [], []
        geometry_chunk = []
        torsion_valid_chunk, torsion_candidate_chunk, pair_chunk = [], [], []
        for index in chunk_indices:
            topology, trimer, smiles = source[index]
            static = source.static_for(index)
            torsion, torsion_valid, torsion_candidates = proper_torsion_descriptor(static, trimer)
            nonbonded, pair_stats = nonbonded_descriptor(static, trimer)
            data_rows.append(build_dual_sample(topology, trimer, smiles, static=static))
            torsion_chunk.append(torsion)
            nonbonded_chunk.append(nonbonded)
            geometry_chunk.append(bool(static["geometry_valid"]))
            torsion_valid_chunk.append(torsion_valid)
            torsion_candidate_chunk.append(torsion_candidates)
            pair_chunk.append(pair_stats)
        batch = dual_glt_collate(data_rows).to(device)
        with torch.no_grad():
            encoded = model.encode(batch)
            h2 = model.norm2(encoded["graph_2d"])
            h3 = model.norm3(encoded["graph_3d"])
            valid = encoded["geometry_valid"].bool()
            h3 = torch.where(valid.unsqueeze(-1), h3, torch.zeros_like(h3))
        h2_np, h3_np = h2.detach().float().cpu().numpy(), h3.detach().float().cpu().numpy()
        if not np.isfinite(h2_np).all() or not np.isfinite(h3_np).all():
            raise FloatingPointError("nonfinite frozen Fixed Concat embedding")
        # A legal static row can contain no center-internal bond token.  The
        # frozen Galformer implementation then has no 3D readout and marks
        # graph_3d invalid even though the underlying Trimer coordinates are
        # valid.  Preserve both flags; h3 follows the model flag, while the
        # descriptor diagnostics retain the static geometry flag.
        valid_list = valid.detach().cpu().tolist()
        if any(model_valid and not static_valid
               for model_valid, static_valid in zip(valid_list, geometry_chunk)):
            raise ValueError("model graph_3d validity exceeds static geometry validity")
        h2_rows.append(h2_np)
        h3_rows.append(h3_np)
        torsion_rows.extend(torsion_chunk)
        nonbonded_rows.extend(nonbonded_chunk)
        geometry_valid_rows.extend(geometry_chunk)
        graph3d_valid_rows.extend(valid_list)
        torsion_valid_rows.extend(torsion_valid_chunk)
        torsion_candidate_rows.extend(torsion_candidate_chunk)
        pair_rows.extend(pair_chunk)
        print(json.dumps({"phase": "embedding", "done": min(start + batch_size, len(unique_indices)),
                          "total": len(unique_indices)}, ensure_ascii=False), flush=True)
    h2 = np.concatenate(h2_rows, axis=0) if h2_rows else np.empty((0, 512), dtype=np.float32)
    h3 = np.concatenate(h3_rows, axis=0) if h3_rows else np.empty((0, 512), dtype=np.float32)
    torsion = np.asarray(torsion_rows, dtype=np.float64)
    nonbonded = np.asarray(nonbonded_rows, dtype=np.float64)
    if len(keys) != len(unique_indices) or h2.shape[0] != len(keys):
        raise ValueError("feature sidecar unique-key count mismatch")
    return {
        "keys": np.asarray([key.hex() for key in keys], dtype="<U64"),
        "row_feature_index": row_feature_index,
        "h2": h2.astype(np.float32),
        "h3": h3.astype(np.float32),
        "torsion": torsion.astype(np.float32),
        "nonbonded": nonbonded.astype(np.float32),
        "geometry_valid": np.asarray(geometry_valid_rows, dtype=bool),
        "graph3d_valid": np.asarray(graph3d_valid_rows, dtype=bool),
        "torsion_valid_count": np.asarray(torsion_valid_rows, dtype=np.int32),
        "torsion_candidate_count": np.asarray(torsion_candidate_rows, dtype=np.int32),
        "pair_counts": np.asarray(
            [[row["center_center_pairs"], row["center_neighbor_pairs"]] for row in pair_rows],
            dtype=np.int32,
        ),
        "pair_candidate_counts": np.asarray(
            [[row["center_center_candidate_pairs"], row["center_neighbor_candidate_pairs"]]
             for row in pair_rows], dtype=np.int32,
        ),
    }


def _split_identity(split_root: Path, raw_root: Path):
    identities = {}
    manifests = {}
    for task in TASKS:
        path = split_root / f"{task}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing fixed split manifest: {path}")
        csv_path = raw_root / f"smi_{task}.csv"
        # Read-only validation: this diagnostic must never materialize or
        # rewrite a missing/changed split manifest.
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = build_manifest(task, csv_path, "outer5_inner20")
        for field in ("protocol", "sample_count", "sample_order_hash"):
            if manifest.get(field) != expected[field]:
                raise ValueError(f"fixed split manifest mismatch for {task}: {field}")
        if manifest.get("validation_is_test") is not False or len(manifest.get("folds", [])) != 5:
            raise ValueError(f"fixed split manifest is not five-fold separated: {task}")
        for actual_fold, expected_fold in zip(manifest["folds"], expected["folds"]):
            for field in ("fold", "train_indices", "validation_indices", "test_indices"):
                if actual_fold.get(field) != expected_fold[field]:
                    raise ValueError(f"fixed split manifest mismatch for {task}: {field}")
        manifests[task] = manifest
        identities[task] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "protocol": manifest["protocol"],
            "sample_count": int(manifest["sample_count"]),
            "sample_order_hash": manifest["sample_order_hash"],
            "folds": [{
                "fold": int(row["fold"]),
                "train_count": len(row["train_indices"]),
                "validation_count": len(row["validation_indices"]),
                "test_count": len(row["test_indices"]),
            } for row in manifest["folds"]],
        }
    return identities, manifests


def _probe_results(features, frame, manifests):
    task_positions = {
        task: np.flatnonzero(frame["task"].to_numpy() == task).astype(np.int64)
        for task in TASKS
    }
    labels = frame["label"].to_numpy(dtype=np.float64)
    keys = frame["sample_key"].astype(str).to_numpy()
    unique_matrices = {
        name: _probe_matrix(name, features["h2"], features["h3"],
                            features["torsion"], features["nonbonded"])
        for name in PROBES
    }
    matrices = {
        name: matrix[features["row_feature_index"]]
        for name, matrix in unique_matrices.items()
    }
    for matrix in matrices.values():
        if not np.isfinite(matrix).all():
            raise FloatingPointError("nonfinite probe feature matrix")
    records = []
    for task in TASKS:
        positions = task_positions[task]
        if len(positions) != int(manifests[task]["sample_count"]):
            raise ValueError(f"task row count differs from split manifest: {task}")
        for fold in manifests[task]["folds"]:
            fold_id = int(fold["fold"])
            train_local = np.asarray(fold["train_indices"], dtype=np.int64)
            validation_local = np.asarray(fold["validation_indices"], dtype=np.int64)
            train = positions[train_local]
            validation = positions[validation_local]
            train_groups = keys[train]
            train_geometry = features["geometry_valid"][features["row_feature_index"][train]]
            validation_geometry = features["geometry_valid"][features["row_feature_index"][validation]]
            train_graph3d = features["graph3d_valid"][features["row_feature_index"][train]]
            validation_graph3d = features["graph3d_valid"][features["row_feature_index"][validation]]
            train_torsion_valid = features["torsion_valid_count"][features["row_feature_index"][train]]
            train_torsion_candidates = features["torsion_candidate_count"][features["row_feature_index"][train]]
            validation_torsion_valid = features["torsion_valid_count"][features["row_feature_index"][validation]]
            validation_torsion_candidates = features["torsion_candidate_count"][features["row_feature_index"][validation]]
            train_pairs = features["pair_counts"][features["row_feature_index"][train]]
            validation_pairs = features["pair_counts"][features["row_feature_index"][validation]]
            for name in PROBES:
                alpha, prediction, alpha_scores = _fit_ridge(
                    matrices[name][train], labels[train], matrices[name][validation],
                    labels[validation], train_groups,
                )
                r2 = float(r2_score(labels[validation], prediction))
                mae = float(mean_absolute_error(labels[validation], prediction))
                rmse = float(np.sqrt(mean_squared_error(labels[validation], prediction)))
                row = {
                    "probe": name, "task": task, "fold": fold_id,
                    "validation_r2": r2, "validation_mae": mae,
                    "validation_rmse": rmse, "selected_alpha": float(alpha),
                    "inner_alpha_scores": {str(key): float(value) for key, value in alpha_scores.items()},
                    "feature_dim": int(matrices[name].shape[1]),
                    "train_sample_count": int(train.size),
                    "validation_sample_count": int(validation.size),
                    "train_unique_structure_count": int(np.unique(train_groups).size),
                    "validation_unique_structure_count": int(np.unique(keys[validation]).size),
                    "train_geometry_valid_fraction": float(np.mean(train_geometry)),
                    "validation_geometry_valid_fraction": float(np.mean(validation_geometry)),
                    "train_graph3d_valid_fraction": float(np.mean(train_graph3d)),
                    "validation_graph3d_valid_fraction": float(np.mean(validation_graph3d)),
                    "train_torsion_valid_fraction": float(
                        train_torsion_valid.sum() / train_torsion_candidates.sum()
                        if train_torsion_candidates.sum() else 0.0),
                    "validation_torsion_valid_fraction": float(
                        validation_torsion_valid.sum() / validation_torsion_candidates.sum()
                        if validation_torsion_candidates.sum() else 0.0),
                    "train_torsion_candidate_count": int(train_torsion_candidates.sum()),
                    "validation_torsion_candidate_count": int(validation_torsion_candidates.sum()),
                    "train_torsion_valid_count": int(train_torsion_valid.sum()),
                    "validation_torsion_valid_count": int(validation_torsion_valid.sum()),
                    "train_nonbonded_pair_counts": [int(value) for value in train_pairs.sum(axis=0)],
                    "validation_nonbonded_pair_counts": [int(value) for value in validation_pairs.sum(axis=0)],
                }
                if not all(np.isfinite(value) for value in (r2, mae, rmse)):
                    raise FloatingPointError(f"nonfinite probe metric: {task} fold {fold_id} {name}")
                records.append(row)
            print(json.dumps({"phase": "probe", "task": task, "fold": fold_id}, ensure_ascii=False), flush=True)
    return records


def _summarize(records):
    by_probe = {name: [row for row in records if row["probe"] == name] for name in PROBES}
    task_means = {}
    for name in PROBES:
        task_means[name] = {}
        for task in TASKS:
            values = [row["validation_r2"] for row in by_probe[name] if row["task"] == task]
            if len(values) != 5:
                raise ValueError(f"probe {name} task {task} does not have five folds")
            task_means[name][task] = float(np.mean(values))
    macro = {name: float(np.mean(list(task_means[name].values()))) for name in PROBES}
    deltas = {}
    for name in PROBES[1:]:
        task_delta = {task: task_means[name][task] - task_means["P0"][task] for task in TASKS}
        fold_delta = []
        for task in TASKS:
            p0 = {row["fold"]: row["validation_r2"] for row in by_probe["P0"] if row["task"] == task}
            px = {row["fold"]: row["validation_r2"] for row in by_probe[name] if row["task"] == task}
            fold_delta.extend(px[index] - p0[index] for index in range(5))
        deltas[name] = {
            "macro7_validation_r2_delta": float(macro[name] - macro["P0"]),
            "task_mean_delta": {task: float(value) for task, value in task_delta.items()},
            "positive_task_count": int(sum(value > 0.0 for value in task_delta.values())),
            "positive_fold_count": int(sum(value > 0.0 for value in fold_delta)),
            "fold_count": len(fold_delta),
        }
    effective_torsion = deltas["P2"]["macro7_validation_r2_delta"] >= 0.005 and deltas["P2"]["positive_task_count"] >= 4
    effective_nonbonded = deltas["P3"]["macro7_validation_r2_delta"] >= 0.005 and deltas["P3"]["positive_task_count"] >= 4
    current3d = deltas["P1"]["macro7_validation_r2_delta"] >= 0.005 and deltas["P1"]["positive_task_count"] >= 4
    if current3d:
        route = "STOP_GEOMETRY_REDESIGN_PRIORITIZE_FUSION_OR_DISTILLATION"
    elif effective_torsion:
        route = "GLT-Torsion-v2"
    elif effective_nonbonded and macro["P3"] >= macro["P2"] + 0.002:
        route = "Spatial/nonbonded GLT"
    elif effective_torsion and effective_nonbonded and abs(macro["P3"] - macro["P2"]) < 0.002:
        route = "GLT-Torsion-v2"
    else:
        route = "STOP_BOND_TOKEN_ROUTE_SWITCH_TO_EQUIVARIANT_3D_TEACHER"
    return {
        "per_probe_task_mean_validation_r2": task_means,
        "per_probe_macro7_validation_r2": macro,
        "deltas_vs_P0": deltas,
        "route_gate": {
            "current3d_gate_pass": bool(current3d),
            "torsion_gate_pass": bool(effective_torsion),
            "nonbonded_gate_pass": bool(effective_nonbonded and macro["P3"] >= macro["P2"] + 0.002),
            "torsion_nonbonded_close": bool(effective_torsion and effective_nonbonded and abs(macro["P3"] - macro["P2"]) < 0.002),
            "selected_route": route,
            "thresholds": {
                "macro7_delta": 0.005, "positive_task_count": 4,
                "nonbonded_over_torsion_delta": 0.002,
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--cohort-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--dual-static-root", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        if any(output.iterdir()):
            raise FileExistsError(f"diagnostic output already exists and is non-empty: {output}")
    else:
        output.mkdir(parents=True)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA device is unavailable")
        split_identities, manifests = _split_identity(Path(args.split_root), Path(args.raw_root))
        cohort_manifest_path = Path(args.cohort_root) / "manifest.json"
        static_manifest_path = Path(args.dual_static_root) / "manifest.json"
        if not cohort_manifest_path.is_file() or not static_manifest_path.is_file():
            raise FileNotFoundError("cohort/static manifest is missing")
        cohort_manifest = json.loads(cohort_manifest_path.read_text(encoding="utf-8"))
        static_manifest = json.loads(static_manifest_path.read_text(encoding="utf-8"))
        if int(cohort_manifest.get("sample_count", 0)) != 6265:
            raise ValueError("unexpected downstream cohort sample count")
        if int(cohort_manifest.get("unique_structure_count", 0)) != 3655:
            raise ValueError("unexpected downstream unique structure count")
        if static_manifest.get("format") != "glt-dual-static-v1":
            raise ValueError("downstream static artifact is not dual_static_v1")
        package = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = build_dual_glt_model("concat")
        load_deployment(model, package, expected_step=5000)
        model.to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        source, frame = open_source(
            args.cohort_root, args.cache_root, dual_static_root=args.dual_static_root,
        )
        try:
            features = _feature_sidecar(source, model, device, args.batch_size)
        finally:
            source.close()
        if features["h2"].shape != (3655, 512) or features["h3"].shape != (3655, 512):
            raise ValueError("feature sidecar has unexpected Fixed Concat dimensions")
        _atomic_npz(output / "feature_sidecar.npz", **features)
        sidecar_metadata = {
            "schema": "glt-3d-info-feature-sidecar-v1",
            "unique_structure_count": int(features["h2"].shape[0]),
            "property_row_count": int(features["row_feature_index"].size),
            "key_order": "first occurrence in frozen downstream cohort",
            "checkpoint": str(Path(args.checkpoint)),
            "checkpoint_sha256": _sha256_file(Path(args.checkpoint)),
            "h2": "norm2(graph_2d), 512-D",
            "h3": "norm3(graph_3d), 512-D; zeroed when the frozen encoder reports graph3d_valid=false",
            "geometry_valid": "static Trimer geometry flag",
            "graph3d_valid": "effective Fixed Concat Galformer graph_3d readout flag",
            "encoder_frozen": True,
            "outer_test_accessed": False,
        }
        _atomic_json(output / "feature_sidecar.json", sidecar_metadata)
        records = _probe_results(features, frame, manifests)
        summary = {
            "schema": "glt-3d-info-diagnostic-v1",
            "plan_id": PLAN_ID,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "baseline_commit": _git_value("rev-parse", "HEAD"),
            "working_tree_note": "Plan.md was pre-existing user modification and was not touched",
            "reference": {
                "checkpoint": str(Path(args.checkpoint)),
                "checkpoint_sha256": _sha256_file(Path(args.checkpoint)),
                "fusion_mode": "concat",
                "step": 5000,
                "tasks": list(TASKS),
                "cohort_manifest": str(cohort_manifest_path),
                "cohort_manifest_sha256": _sha256_file(cohort_manifest_path),
                "cohort_manifest_hash": cohort_manifest.get("manifest_hash"),
                "static_manifest": str(static_manifest_path),
                "static_manifest_sha256": _sha256_file(static_manifest_path),
                "static_manifest_hash": static_manifest.get("manifest_hash"),
                "sample_count": int(cohort_manifest["sample_count"]),
                "unique_structure_count": int(cohort_manifest["unique_structure_count"]),
            },
            "exact_split_identities": split_identities,
            "outer_test_accessed": False,
            "outer_test_policy": "No outer-test labels, coordinates, predictions, or metrics were used; only train/validation indices were passed to probes.",
            "descriptor_definition": {
                "torsion": {
                    "dimension": 10, "fourier_orders": [1, 2, 3, 4],
                    "features": "mean sin(k phi), mean cos(k phi), valid torsion count, valid torsion fraction",
                    "orientation": "lexicographic min(quadruplet, reversed quadruplet)",
                    "no_legal_torsion": "all zeros with count=0 and fraction=0",
                },
                "nonbonded": {
                    "dimension": 34, "distance_range_angstrom": [2.0, 6.0],
                    "rbf_bins": 16, "rbf_centers_angstrom": RBF_CENTERS.tolist(),
                    "rbf_sigma_angstrom": RBF_SIGMA,
                    "features": "mean center-center RBF, mean center-neighbor RBF, two pair counts",
                    "excluded": "direct bonded (1-2) and shared-neighbor (1-3) pairs",
                },
            },
            "feature_sidecar": {
                "path": str(output / "feature_sidecar.npz"),
                "metadata_path": str(output / "feature_sidecar.json"),
                "schema": "glt-3d-info-feature-sidecar-v1",
            },
            "probe": {
                "type": "Ridge",
                "alpha_candidates": list(ALPHAS),
                "alpha_selection": "5-fold GroupKFold inside outer-train by canonical sample key",
                "feature_scaler": "StandardScaler fit on inner/outer train only",
                "target_scaler": "StandardScaler fit on inner/outer train only",
                "feature_sets": {
                    "P0": "h2",
                    "P1": "h2 + h3",
                    "P2": "h2 + torsion_descriptor",
                    "P3": "h2 + nonbonded_descriptor",
                    "P4": "h2 + torsion_descriptor + nonbonded_descriptor",
                },
            },
            "fold_count": len(records) // len(PROBES),
            "records": records,
            "diagnostic_statistics": {
                "geometry_valid_count": int(features["geometry_valid"].sum()),
                "geometry_invalid_count": int((~features["geometry_valid"]).sum()),
                "graph3d_valid_count": int(features["graph3d_valid"].sum()),
                "graph3d_invalid_count": int((~features["graph3d_valid"]).sum()),
                "torsion_candidate_count": int(features["torsion_candidate_count"].sum()),
                "torsion_valid_count": int(features["torsion_valid_count"].sum()),
                "torsion_valid_fraction": float(
                    features["torsion_valid_count"].sum() / features["torsion_candidate_count"].sum()
                    if features["torsion_candidate_count"].sum() else 0.0),
                "nonbonded_pair_counts": {
                    "center_center": int(features["pair_counts"][:, 0].sum()),
                    "center_neighbor": int(features["pair_counts"][:, 1].sum()),
                },
                "nonbonded_pair_candidate_counts": {
                    "center_center": int(features["pair_candidate_counts"][:, 0].sum()),
                    "center_neighbor": int(features["pair_candidate_counts"][:, 1].sum()),
                },
            },
        }
        summary.update(_summarize(records))
        if summary["fold_count"] != 35 or len(records) != 35 * len(PROBES):
            raise ValueError("diagnostic did not produce exactly 35 folds for every probe")
        _atomic_json(output / "summary.json", summary)
        print(json.dumps({
            "status": "PASS", "summary": str(output / "summary.json"),
            "fold_count": summary["fold_count"], "outer_test_accessed": False,
            "route": summary["route_gate"]["selected_route"],
        }, ensure_ascii=False), flush=True)
    except Exception:
        # Keep partial diagnostic files for post-mortem, but never emit a
        # misleading summary claiming completion.
        raise


if __name__ == "__main__":
    main()
