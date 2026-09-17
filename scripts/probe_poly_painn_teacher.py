#!/usr/bin/env python3
"""Frozen residual-information probe for the PolyPaiNN teacher.

Only outer-train and outer-validation rows are read for fitting/evaluation.  In
particular, no outer-test labels, coordinates, predictions or metrics are
loaded by this script.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from scripts.create_mips_split_manifests import build_manifest
from src.dataset.glt_dual import build_dual_sample, dual_glt_collate
from src.dataset.poly_painn_teacher import build_teacher_sample, teacher_collate
from src.dataset.glt_o8_control import O8OnlySource, o8_collate
from src.modules.glt_o8_control import O8ControlModel, load_o8_deployment
from src.modules.poly_painn_teacher import PolyPaiNNTeacher, load_teacher_deployment
from src.training.glt_dual_runtime import open_source, require_tmux, write_json


TASKS = ("eat", "eea", "egb", "ei", "eps", "nc", "xc")
PROBES = ("T0", "T1")
ALPHAS = (0.1, 1.0, 10.0, 100.0)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fit_ridge(x_train, y_train, x_eval, y_eval, groups):
    if np.unique(groups).size < 5:
        raise ValueError("outer-train has fewer than five canonical structures")
    scores = {}
    for alpha in ALPHAS:
        values = []
        for inner_train, inner_eval in GroupKFold(n_splits=5).split(x_train, y_train, groups):
            xs = StandardScaler().fit(x_train[inner_train])
            ys = StandardScaler().fit(y_train[inner_train].reshape(-1, 1))
            model = Ridge(alpha=float(alpha)).fit(
                xs.transform(x_train[inner_train]),
                ys.transform(y_train[inner_train].reshape(-1, 1)).reshape(-1),
            )
            pred = ys.inverse_transform(model.predict(xs.transform(x_train[inner_eval])).reshape(-1, 1)).reshape(-1)
            value = float(r2_score(y_train[inner_eval], pred))
            if np.isfinite(value):
                values.append(value)
        if not values:
            raise FloatingPointError(f"no finite inner score for alpha={alpha}")
        scores[float(alpha)] = float(np.mean(values))
    alpha = max(ALPHAS, key=lambda value: (scores[float(value)], -float(value)))
    xs = StandardScaler().fit(x_train)
    ys = StandardScaler().fit(y_train.reshape(-1, 1))
    model = Ridge(alpha=float(alpha)).fit(
        xs.transform(x_train), ys.transform(y_train.reshape(-1, 1)).reshape(-1)
    )
    prediction = ys.inverse_transform(model.predict(xs.transform(x_eval)).reshape(-1, 1)).reshape(-1)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("nonfinite validation prediction")
    return float(alpha), prediction, scores


def _split_identities(split_root, raw_root):
    identities, manifests = {}, {}
    for task in TASKS:
        path = Path(split_root) / f"{task}.json"
        csv_path = Path(raw_root) / f"smi_{task}.csv"
        if not path.is_file() or not csv_path.is_file():
            raise FileNotFoundError(f"missing split fixture for {task}")
        actual = json.loads(path.read_text(encoding="utf-8"))
        expected = build_manifest(task, csv_path, "outer5_inner20")
        for field in ("protocol", "sample_count", "sample_order_hash"):
            if actual.get(field) != expected[field]:
                raise ValueError(f"fixed split mismatch for {task}: {field}")
        if actual.get("validation_is_test") is not False or len(actual.get("folds", [])) != 5:
            raise ValueError(f"fixed split is not five-fold separated for {task}")
        for old, new in zip(actual["folds"], expected["folds"]):
            for field in ("fold", "train_indices", "validation_indices", "test_indices"):
                if old.get(field) != new[field]:
                    raise ValueError(f"fixed split mismatch for {task}: {field}")
        manifests[task] = actual
        identities[task] = {"path": str(path), "sha256": _sha256(path),
                            "protocol": actual["protocol"], "sample_count": int(actual["sample_count"]),
                            "sample_order_hash": actual["sample_order_hash"],
                            "folds": [{"fold": int(row["fold"]),
                                       "train_count": len(row["train_indices"]),
                                       "validation_count": len(row["validation_indices"]),
                                       "test_count": len(row["test_indices"])}
                                      for row in actual["folds"]]}
    return identities, manifests


def _extract_features(source, teacher, o8, device, batch_size):
    first_by_key, rows = {}, np.empty(len(source), dtype=np.int32)
    for index, (key, _) in enumerate(source.samples):
        key = bytes(key)
        if key not in first_by_key:
            first_by_key[key] = len(first_by_key)
        rows[index] = first_by_key[key]
    unique_indices = [None] * len(first_by_key)
    for index, unique in enumerate(rows):
        if unique_indices[int(unique)] is None:
            unique_indices[int(unique)] = index
    h2_chunks, teacher_chunks, validity = [], [], []
    keys = []
    teacher.eval(); o8.eval()
    for start in range(0, len(unique_indices), int(batch_size)):
        indices = unique_indices[start:start + int(batch_size)]
        teacher_records, o8_records = [], []
        for index in indices:
            topology, trimer, smiles = source[index]
            key = source.samples[index][0].hex()
            noisy, target = build_teacher_sample(
                topology, trimer, seed=42, key=key, position=0, sigma=0.03,
            )
            noisy.pos = target["clean_pos"]
            teacher_records.append((noisy, target))
            o8_records.append(build_dual_sample(
                topology, trimer, smiles, static=source.static_for(index),
            ))
            keys.append(key)
            validity.append(bool(trimer.trimer_geometry_valid))
        teacher_data, _ = teacher_collate(teacher_records)
        o8_data = dual_glt_collate(o8_records)
        with torch.no_grad():
            teacher_out = teacher(teacher_data.to(device))
            o8_encoded = o8.encode(o8_data.to(device))
            h2 = o8.norm2(o8_encoded["graph_2d"])
        h2_chunks.append(h2.float().cpu().numpy())
        teacher_chunks.append(teacher_out["graph_scalar"].float().cpu().numpy())
        print(json.dumps({"phase": "embedding", "done": min(start + len(indices), len(unique_indices)),
                          "total": len(unique_indices)}, ensure_ascii=False), flush=True)
    h2 = np.concatenate(h2_chunks, axis=0)
    teacher_graph = np.concatenate(teacher_chunks, axis=0)
    if h2.shape != (len(unique_indices), 512) or teacher_graph.ndim != 2:
        raise ValueError("unexpected feature dimensions")
    if not np.isfinite(h2).all() or not np.isfinite(teacher_graph).all():
        raise FloatingPointError("nonfinite teacher/O8 features")
    return {
        "keys": np.asarray(keys, dtype="<U64"), "row_feature_index": rows,
        "h2": h2.astype(np.float32), "teacher_graph_scalar": teacher_graph.astype(np.float32),
        "geometry_valid": np.asarray(validity, dtype=bool),
    }


def _probe(features, frame, manifests):
    labels = frame["label"].to_numpy(dtype=np.float64)
    keys = frame["sample_key"].astype(str).to_numpy()
    matrices = {
        "T0": features["h2"],
        "T1": np.concatenate((features["h2"], features["teacher_graph_scalar"]), axis=1),
    }
    positions = {task: np.flatnonzero(frame["task"].to_numpy() == task) for task in TASKS}
    records = []
    for task in TASKS:
        task_pos = positions[task]
        if len(task_pos) != int(manifests[task]["sample_count"]):
            raise ValueError(f"task row count differs from split: {task}")
        for fold in manifests[task]["folds"]:
            fold_id = int(fold["fold"])
            train = task_pos[np.asarray(fold["train_indices"], dtype=np.int64)]
            validation = task_pos[np.asarray(fold["validation_indices"], dtype=np.int64)]
            groups = keys[train]
            for name, unique_matrix in matrices.items():
                matrix = unique_matrix[features["row_feature_index"]]
                alpha, prediction, scores = _fit_ridge(
                    matrix[train], labels[train], matrix[validation], labels[validation], groups,
                )
                row = {
                    "probe": name, "task": task, "fold": fold_id,
                    "validation_r2": float(r2_score(labels[validation], prediction)),
                    "validation_mae": float(mean_absolute_error(labels[validation], prediction)),
                    "validation_rmse": float(np.sqrt(mean_squared_error(labels[validation], prediction))),
                    "selected_alpha": alpha,
                    "inner_alpha_scores": {str(k): float(v) for k, v in scores.items()},
                    "feature_dim": int(matrix.shape[1]),
                    "train_sample_count": int(train.size), "validation_sample_count": int(validation.size),
                    "train_unique_structure_count": int(np.unique(groups).size),
                    "validation_unique_structure_count": int(np.unique(keys[validation]).size),
                    "train_geometry_valid_fraction": float(np.mean(features["geometry_valid"][features["row_feature_index"][train]])),
                    "validation_geometry_valid_fraction": float(np.mean(features["geometry_valid"][features["row_feature_index"][validation]])),
                }
                if not np.isfinite([row["validation_r2"], row["validation_mae"], row["validation_rmse"]]).all():
                    raise FloatingPointError(f"nonfinite probe metric {task} fold {fold_id} {name}")
                records.append(row)
            print(json.dumps({"phase": "probe", "task": task, "fold": fold_id}, ensure_ascii=False), flush=True)
    return records


def _summary(records):
    by = {name: [row for row in records if row["probe"] == name] for name in PROBES}
    task_means = {name: {task: float(np.mean([row["validation_r2"] for row in by[name] if row["task"] == task]))
                         for task in TASKS} for name in PROBES}
    macro = {name: float(np.mean(list(task_means[name].values()))) for name in PROBES}
    delta_tasks = {task: task_means["T1"][task] - task_means["T0"][task] for task in TASKS}
    fold_delta = []
    for task in TASKS:
        t0 = {row["fold"]: row["validation_r2"] for row in by["T0"] if row["task"] == task}
        t1 = {row["fold"]: row["validation_r2"] for row in by["T1"] if row["task"] == task}
        fold_delta.extend(t1[i] - t0[i] for i in range(5))
    delta = macro["T1"] - macro["T0"]
    return {
        "per_probe_task_mean_validation_r2": task_means,
        "per_probe_macro7_validation_r2": macro,
        "delta_t1_minus_t0": {"macro7_validation_r2_delta": float(delta),
                               "task_mean_delta": delta_tasks,
                               "positive_task_count": int(sum(value > 0 for value in delta_tasks.values())),
                               "positive_fold_count": int(sum(value > 0 for value in fold_delta)),
                               "fold_count": len(fold_delta)},
        "teacher_residual_gate": {
            "status": "PASS" if delta >= 0.005 and sum(value > 0 for value in delta_tasks.values()) >= 4 else "FAIL",
            "threshold_macro7_delta": 0.005, "threshold_positive_tasks": 4,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher-deployment", "o8-deployment", "raw-root", "cohort-root", "cache-root", "downstream-static-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--split-root", default="data/splits/mips_outer5_inner20")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    require_tmux()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"probe output is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested probe CUDA device is unavailable")
    split_identities, manifests = _split_identities(args.split_root, args.raw_root)
    teacher_package = torch.load(args.teacher_deployment, map_location="cpu", weights_only=False)
    teacher_hyper = teacher_package.get("hyperparameters", {})
    teacher = PolyPaiNNTeacher(**{key: teacher_hyper[key] for key in (
        "hidden_channels", "num_layers", "cutoff", "max_num_neighbors", "rbf_dim", "max_atomic_number"
    ) if key in teacher_hyper}).to(device).eval()
    load_teacher_deployment(teacher, teacher_package, expected_step=int(teacher_package.get("step", -1)))
    o8_package = torch.load(args.o8_deployment, map_location="cpu", weights_only=False)
    o8 = O8ControlModel().to(device).eval()
    load_o8_deployment(o8, o8_package, expected_step=5000)
    for parameter in list(teacher.parameters()) + list(o8.parameters()):
        parameter.requires_grad_(False)
    source, frame = open_source(
        args.cohort_root, args.cache_root, dual_static_root=args.downstream_static_root,
    )
    try:
        features = _extract_features(source, teacher, o8, device, args.batch_size)
    finally:
        source.close()
    if len(features["keys"]) != 3655 or len(frame) != 6265:
        raise ValueError("downstream feature/row count differs from frozen cohort")
    np.savez_compressed(output / "feature_sidecar.npz", **features)
    records = _probe(features, frame, manifests)
    if len(records) != 70:
        raise ValueError("probe did not produce exactly 35 folds x 2 probes")
    summary = {
        "schema": "eq3d-dnd-teacher-probe-v1", "plan_id": "EQ3D-DND-20260917-01",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "teacher_deployment": str(Path(args.teacher_deployment).resolve()),
        "teacher_deployment_sha256": _sha256(args.teacher_deployment),
        "teacher_step": int(teacher_package.get("step", -1)),
        "reference_o8_deployment": str(Path(args.o8_deployment).resolve()),
        "reference_o8_deployment_sha256": _sha256(args.o8_deployment),
        "tasks": list(TASKS), "feature_sidecar": str(output / "feature_sidecar.npz"),
        "exact_split_identities": split_identities, "outer_test_accessed": False,
        "outer_test_policy": "No outer-test labels, coordinates, predictions or metrics were loaded.",
        "feature_definition": {"T0": "Arm B O8 h2", "T1": "h2 + teacher RU(0) central scalar mean"},
        "records": records, "fold_count": 35,
        "diagnostic_statistics": {
            "unique_structure_count": int(len(features["keys"])),
            "property_row_count": int(len(frame)),
            "geometry_valid_count": int(features["geometry_valid"].sum()),
            "geometry_invalid_count": int((~features["geometry_valid"]).sum()),
        },
    }
    summary.update(_summary(records))
    _atomic_json(output / "summary.json", summary)
    print(json.dumps({"status": "PASS", "summary": str(output / "summary.json"),
                      "fold_count": 35, "outer_test_accessed": False,
                      "teacher_residual_gate": summary["teacher_residual_gate"]["status"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
