"""Frozen representation probes for the C0/C1/C2 repaired deployments."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.dataset import UniDataset
from src.modules.mts_glt_distill import DistillStudent
from src.utils import TargetScaler, get_data_loader, set_global_seed


VERSIONS = {"c0": "none", "c1": "n_plus_1", "c2": "n_plus_2"}
ALPHAS = (0.1, 1.0, 10.0, 100.0)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path, **payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_split_manifest(root, task, sample_count):
    path = Path(root) / f"{task}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    csv_path = Path("data/raw") / f"smi_{task}.csv"
    smiles = pd.read_csv(csv_path, usecols=[0]).iloc[:, 0].astype(str).str.strip().tolist()
    order_hash = hashlib.sha256("\n".join(smiles).encode()).hexdigest()
    if (
        payload.get("schema") != "mips-outer5-inner20-fold-v1"
        or payload.get("protocol") != "outer5_inner20"
        or bool(payload.get("validation_is_test", True))
        or int(payload.get("sample_count", -1)) != int(sample_count)
        or payload.get("sample_order_sha256") != order_hash
    ):
        raise RuntimeError(f"outer5_inner20 manifest mismatch: {path}")
    return payload, sha256_file(path)


def build_task_dataset(task):
    return UniDataset(
        root="./data", dataset=f"smi_{task}", smiles_model_name="",
        graph_encoder_type="mips_trimer_scage", graph_input="star_linking",
        use_feature_cache=True, feature_source_dataset="smi_all",
        rebuild_feature_cache=False, max_smiles_length=None,
        max_smiles_length_cap=256, fp_mode="disabled",
        feature_cache_workers=0, feature_cache_chunksize=4,
        feature_cache_partial_every=200, feature_cache_item_timeout=45,
        cache_layers="ru_base,topology,trimer,md200", cache_validate="sample",
        cache_commit_size=128, embed_tries_multiplier=8,
        conformer_3d_count=4, conformer_keep_count=4,
        conformer_profile="full", scage_distance_mode="mips_dual",
        scage_distance_rbf=32, scage_distance_cutoff=12.0,
        mips_core="paper_corrected", mips_max_hops=2,
        mips_use_descriptors=True, mips_descriptor_protocol="source_star_sub",
        spatial_mode="trimer_scage", graph_geometry_mode="trimer_scage_mcl",
        topology_representation="canonical_lifted", trimer_num_candidates=4,
        trimer_max_heavy_atoms=384, mips_variant="O8",
        finite_variant="none", conformer_mode="none", field_layout="none",
        field_channels="none", experiment_id="mts_c0_transfer_optimization",
        feature_config_hash="manual", modalities=("graph",),
        periodic_line_glt_sidecar=None,
    )


def load_deployment(path, expected_group, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected_version = VERSIONS[expected_group]
    expected_revision = None if expected_group == "c0" else 2
    if (
        checkpoint.get("schema") != "mts-glt-distill-repair-student-deploy-v1"
        or checkpoint.get("version") != expected_version
        or checkpoint.get("geometry_revision") != expected_revision
        or int(checkpoint.get("step", -1)) != 20000
    ):
        raise RuntimeError(f"invalid {expected_group} 20k deployment: {path}")
    encoder = DistillStudent()
    encoder.load_state_dict(checkpoint["state_dict"], strict=True)
    encoder.requires_grad_(False).eval().to(device)
    return encoder, {
        "schema": checkpoint["schema"], "version": checkpoint["version"],
        "geometry_revision": checkpoint.get("geometry_revision"),
        "step": int(checkpoint["step"]), "sha256": sha256_file(path),
        "path": str(Path(path).resolve()),
    }


def extract_frozen_features(encoder, dataset, device, batch_size=64, workers=2):
    encoder.eval()
    loader = get_data_loader(
        dataset, indices=np.arange(len(dataset)), batch_size=batch_size,
        shuffle=False, drop_last=False, num_workers=workers, pin_memory=True,
        persistent_workers=workers > 0,
    )
    chunks = []
    with torch.inference_mode():
        for batch in loader:
            features = encoder(batch.to(device, non_blocking=True))
            if features.ndim != 2 or features.size(1) != 512 or not torch.isfinite(features).all():
                raise RuntimeError("invalid frozen pooled O8+MD200 representation")
            chunks.append(features.float().cpu())
    values = torch.cat(chunks).numpy()
    if values.shape != (len(dataset), 512):
        raise RuntimeError("frozen feature/sample order length mismatch")
    return values, np.arange(len(dataset), dtype=np.int64)


def fit_probe_fold(features, raw_targets, split, task, alphas=ALPHAS):
    train = np.asarray(split["train_indices"], dtype=np.int64)
    validation = np.asarray(split["validation_indices"], dtype=np.int64)
    test = np.asarray(split["test_indices"], dtype=np.int64)
    feature_scaler = StandardScaler().fit(np.asarray(features[train], dtype=np.float64))
    x_train = feature_scaler.transform(features[train]).astype(np.float64, copy=False)
    x_validation = feature_scaler.transform(features[validation]).astype(np.float64, copy=False)
    x_test = feature_scaler.transform(features[test]).astype(np.float64, copy=False)
    label_scaler = TargetScaler(task, StandardScaler(), transform_mode="recommended")
    label_scaler.scaler.fit(label_scaler._pre_transform(raw_targets[train]))
    y_train = label_scaler.transform(raw_targets[train]).reshape(-1).astype(np.float64)

    candidates = []
    best = None
    for alpha in sorted(float(v) for v in alphas):
        ridge = Ridge(alpha=alpha, fit_intercept=True, solver="svd")
        ridge.fit(x_train, y_train)
        validation_prediction = label_scaler.inverse_transform(
            ridge.predict(x_validation)
        ).reshape(-1)
        validation_r2 = float(r2_score(raw_targets[validation], validation_prediction))
        candidate = (validation_r2, alpha, ridge)
        candidates.append({"alpha": alpha, "validation_r2": validation_r2})
        if best is None or (validation_r2, alpha) > (best[0], best[1]):
            best = candidate
    validation_r2, alpha, ridge = best
    prediction = label_scaler.inverse_transform(ridge.predict(x_test)).reshape(-1)
    truth = np.asarray(raw_targets[test], dtype=np.float64).reshape(-1)
    if not np.isfinite(prediction).all():
        raise RuntimeError("probe inverse-transformed predictions contain NaN/Inf")
    metrics = {
        "selected_alpha": float(alpha), "best_validation_r2": float(validation_r2),
        "test_r2": float(r2_score(truth, prediction)),
        "test_mae": float(mean_absolute_error(truth, prediction)),
        "test_rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "alpha_candidates": candidates,
    }
    state = {
        "feature_mean": feature_scaler.mean_, "feature_scale": feature_scaler.scale_,
        "feature_var": feature_scaler.var_, "label_mean": label_scaler.scaler.mean_,
        "label_scale": label_scaler.scaler.scale_, "label_var": label_scaler.scaler.var_,
        "ridge_coef": np.asarray(ridge.coef_, dtype=np.float64),
        "ridge_intercept": np.asarray(ridge.intercept_, dtype=np.float64),
    }
    return metrics, state, truth, prediction, train, validation, test


def run_probe_group_task(group, task, checkpoint, output_root, *, folds=range(5), workers=2):
    started = time.monotonic()
    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("formal frozen probes require CUDA")
    dataset = build_task_dataset(task)
    manifest, manifest_sha = load_split_manifest("data/splits/mips_outer5_inner20", task, len(dataset))
    encoder, checkpoint_meta = load_deployment(checkpoint, group, device)
    features, indices = extract_frozen_features(encoder, dataset, device, workers=workers)
    # Repeat extraction in one batch partitioning to prove eval-mode determinism.
    repeat, repeat_indices = extract_frozen_features(encoder, dataset, device, workers=workers)
    if not np.array_equal(indices, repeat_indices) or not np.array_equal(features, repeat):
        raise RuntimeError("frozen representation extraction is not repeatable")
    cache = Path(output_root) / "cache" / group / f"{task}.npz"
    atomic_npz(
        cache, features=features, sample_indices=indices,
        metadata=np.asarray(json.dumps({
            "representation": "DistillStudent.pool(md_residual(canonical_o8))",
            "checkpoint": checkpoint_meta, "task": task,
            "split_manifest_sha256": manifest_sha,
        }, sort_keys=True)),
    )
    raw = np.asarray(dataset.raw_targets, dtype=np.float64)
    outputs = []
    for fold in folds:
        split = manifest["folds"][int(fold)]
        metrics, state, truth, prediction, train, validation, test = fit_probe_fold(features, raw, split, task)
        unit = Path(output_root) / "units" / group / task / f"fold_{int(fold)}"
        metadata = {
            "schema": "mts-c0-frozen-ridge-probe-v1", "group": group,
            "task": task, "fold": int(fold), "seed": 42,
            "evaluation_protocol": "outer5_inner20",
            "split_manifest_sha256": manifest_sha, "checkpoint": checkpoint_meta,
            "feature_dtype_for_fit": "float64", "ridge_solver": "svd",
            "fit_intercept": True, "target_transform": "recommended",
        }
        atomic_npz(unit / "predictions.npz", y_true=truth, y_pred=prediction, sample_indices=test, metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
        atomic_npz(unit / "fit_state.npz", train_indices=train, validation_indices=validation, test_indices=test, **state)
        record = {**metadata, **metrics, "prediction_path": str((unit / "predictions.npz").resolve()), "feature_cache": str(cache.resolve())}
        atomic_json(unit / "metrics.json", record)
        outputs.append(record)
    atomic_json(Path(output_root) / "units" / group / task / "complete.json", {
        "schema": "mts-c0-frozen-ridge-task-complete-v1", "group": group,
        "task": task, "folds": len(outputs), "wall_seconds": time.monotonic() - started,
        "checkpoint": checkpoint_meta, "split_manifest_sha256": manifest_sha,
    })
    return outputs

