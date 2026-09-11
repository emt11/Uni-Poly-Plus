#!/usr/bin/env python3
"""Ridge probes for the historical MD tiers and the new no-MD 5k bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.dataset import UniDataset
from src.modules.mts_glt_distill import DistillStudent
from src.utils import TargetScaler, get_data_loader, set_global_seed


TASKS = ("ei", "xc", "eps", "nc")
ALPHAS = (0.1, 1.0, 10.0, 100.0)
TIERS = {
    "old_005k": (ROOT / "results/glt_v2_r2_c0_gelu138_mipshead/student/student_deploy_005k.pt", True, 5000),
    "old_010k": (ROOT / "results/glt_v2_r2_c0_gelu138_mipshead/student/student_deploy_010k.pt", True, 10000),
    "old_020k": (ROOT / "results/glt_v2_r2_c0_gelu138_mipshead/student/student_deploy_020k.pt", True, 20000),
    "nomd_005k": (ROOT / "results/glt_v2_r2_o8_nomd_mipsloss_005k/student/student_deploy_005k.pt", False, 5000),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_task_dataset(task: str, use_md200: bool):
    # Old tiers read the existing MD200 layer; no-MD only requests topology.
    layers = "ru_base,topology,trimer,md200" if use_md200 else "ru_base,topology"
    return UniDataset(
        root="./data", dataset=f"smi_{task}", smiles_model_name="",
        graph_encoder_type="mips_trimer_scage", graph_input="star_linking",
        use_feature_cache=True, feature_source_dataset="smi_all",
        rebuild_feature_cache=False, max_smiles_length=None,
        max_smiles_length_cap=256, fp_mode="disabled",
        feature_cache_workers=0, feature_cache_chunksize=4,
        feature_cache_partial_every=200, feature_cache_item_timeout=45,
        cache_layers=layers, cache_validate="sample", cache_commit_size=128,
        embed_tries_multiplier=8, conformer_3d_count=4,
        conformer_keep_count=4, conformer_profile="full",
        scage_distance_mode="mips_dual", scage_distance_rbf=32,
        scage_distance_cutoff=12.0, mips_core="paper_corrected",
        mips_max_hops=2, mips_use_descriptors=True,
        mips_descriptor_protocol="source_star_sub", spatial_mode="trimer_scage",
        graph_geometry_mode="trimer_scage_mcl", topology_representation="canonical_lifted",
        trimer_num_candidates=4, trimer_max_heavy_atoms=384, mips_variant="O8",
        finite_variant="none", conformer_mode="none", field_layout="none",
        field_channels="none", experiment_id="glt_v2_r2_o8_nomd_mipsloss_005k_probe",
        feature_config_hash="manual", modalities=("graph",), periodic_line_glt_sidecar=None,
    )


def load_deployment(path: Path, use_md200: bool, expected_step: int, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_schema = (
        "mts-glt-distill-repair-student-deploy-v1" if use_md200
        else "mts-glt-v2-r2-o8-nomd-student-deploy-v1"
    )
    if payload.get("schema") != expected_schema or int(payload.get("step", -1)) != expected_step:
        raise RuntimeError(f"deployment identity mismatch: {path}")
    if use_md200 and payload.get("use_md200", True) is False:
        raise RuntimeError("historical MD tier is marked no-MD")
    if not use_md200 and payload.get("use_md200", True) is not False:
        raise RuntimeError("no-MD deployment contains MD200")
    encoder = DistillStudent(use_md200=use_md200)
    encoder.load_state_dict(payload["state_dict"], strict=True)
    encoder.eval().requires_grad_(False).to(device)
    return encoder, {
        "path": str(path.resolve()), "sha256": sha256_file(path),
        "schema": payload["schema"], "step": int(payload["step"]),
        "use_md200": bool(use_md200),
    }


@torch.inference_mode()
def extract_features(encoder, dataset, device, batch_size=64):
    loader = get_data_loader(dataset, indices=np.arange(len(dataset)), batch_size=batch_size,
                             shuffle=False, drop_last=False, num_workers=2,
                             pin_memory=True, persistent_workers=True)
    chunks = []
    for batch in loader:
        value = encoder(batch.to(device, non_blocking=True))
        if value.ndim != 2 or value.size(1) != 512 or not torch.isfinite(value).all():
            raise RuntimeError("invalid pooled representation")
        chunks.append(value.float().cpu())
    result = torch.cat(chunks).numpy().astype(np.float64, copy=False)
    if result.shape != (len(dataset), 512) or not np.isfinite(result).all():
        raise RuntimeError("feature/sample order or finiteness mismatch")
    return result


def manifest(task: str):
    path = ROOT / "data/splits/mips_outer5_inner20" / f"{task}.json"
    payload = json.loads(path.read_text())
    ordered = pd.read_csv(ROOT / "data/raw" / f"smi_{task}.csv", usecols=[0]).iloc[:, 0].astype(str).str.strip().tolist()
    expected = hashlib.sha256("\n".join(ordered).encode()).hexdigest()
    if payload.get("schema") != "mips-outer5-inner20-fold-v1" or payload.get("sample_order_sha256") != expected:
        raise RuntimeError(f"split manifest mismatch: {path}")
    return payload, sha256_file(path)


def fit_fold(features, raw, split, task):
    train = np.asarray(split["train_indices"], dtype=np.int64)
    validation = np.asarray(split["validation_indices"], dtype=np.int64)
    test = np.asarray(split["test_indices"], dtype=np.int64)
    x_scaler = StandardScaler().fit(features[train].astype(np.float64))
    x_train, x_val, x_test = (x_scaler.transform(features[idx]).astype(np.float64) for idx in (train, validation, test))
    y_scaler = TargetScaler(task, StandardScaler(), transform_mode="standard")
    y_scaler.scaler.fit(y_scaler._pre_transform(raw[train]))
    y_train = y_scaler.transform(raw[train]).reshape(-1).astype(np.float64)
    candidates, best = [], None
    for alpha in ALPHAS:
        ridge = Ridge(alpha=float(alpha), fit_intercept=True, solver="svd")
        ridge.fit(x_train, y_train)
        pred = y_scaler.inverse_transform(ridge.predict(x_val)).reshape(-1)
        r2 = float(r2_score(raw[validation], pred))
        candidates.append({"alpha": float(alpha), "validation_r2": r2})
        if best is None or (r2, float(alpha)) > (best[0], best[1]):
            best = (r2, float(alpha), ridge)
    val_r2, alpha, ridge = best
    prediction = y_scaler.inverse_transform(ridge.predict(x_test)).reshape(-1)
    truth = raw[test].reshape(-1)
    if not np.isfinite(prediction).all():
        raise RuntimeError("non-finite probe prediction")
    return {
        "selected_alpha": alpha, "best_validation_r2": val_r2,
        "test_r2": float(r2_score(truth, prediction)),
        "test_mae": float(mean_absolute_error(truth, prediction)),
        "test_rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "alpha_candidates": candidates,
    }, prediction, truth, train, validation, test, {
        "feature_mean": x_scaler.mean_, "feature_scale": x_scaler.scale_,
        "label_mean": y_scaler.scaler.mean_, "label_scale": y_scaler.scaler.scale_,
        "ridge_coef": np.asarray(ridge.coef_, dtype=np.float64),
        "ridge_intercept": np.asarray(ridge.intercept_, dtype=np.float64),
    }


def run(output_root: Path, selected_tiers, selected_tasks):
    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("probe requires CUDA")
    rows = []
    for tier in selected_tiers:
        checkpoint, use_md, step = TIERS[tier]
        encoder, checkpoint_meta = load_deployment(checkpoint, use_md, step, device)
        for task in selected_tasks:
            dataset = build_task_dataset(task, use_md)
            split_payload, split_sha = manifest(task)
            raw = np.asarray(dataset.raw_targets, dtype=np.float64)
            feature_path = output_root / "cache" / f"{tier}_{task}.npz"
            features = extract_features(encoder, dataset, device)
            # A second extraction checks eval/dropout determinism and ordering.
            repeat = extract_features(encoder, dataset, device)
            if not np.array_equal(features, repeat):
                raise RuntimeError(f"non-repeatable frozen representation: {tier}/{task}")
            atomic_npz(feature_path, features=features, sample_indices=np.arange(len(dataset), dtype=np.int64))
            for fold, split in enumerate(split_payload["folds"]):
                metrics, prediction, truth, train, validation, test, state = fit_fold(features, raw, split, task)
                unit = output_root / "units" / tier / task / f"fold_{fold}"
                metadata = {
                    "schema": "glt-v2-r2-ridge-probe-v1", "tier": tier, "task": task,
                    "fold": fold, "seed": 42, "evaluation_protocol": "outer5_inner20",
                    "split_manifest_sha256": split_sha, "target_transform": "standard",
                    "ridge_solver": "svd", "feature_dtype_for_fit": "float64",
                    "representation": "DistillStudent.pool(canonical_o8)" if not use_md else "DistillStudent.pool(md_residual(canonical_o8))",
                    "checkpoint": checkpoint_meta,
                }
                atomic_npz(unit / "predictions.npz", y_true=truth, y_pred=prediction,
                           sample_indices=test, metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
                atomic_npz(unit / "fit_state.npz", train_indices=train, validation_indices=validation,
                           test_indices=test, **state)
                row = {**metadata, **metrics, "prediction_path": str((unit / "predictions.npz").resolve()),
                       "feature_cache": str(feature_path.resolve())}
                atomic_json(unit / "metrics.json", row)
                rows.append(row)
            atomic_json(output_root / "units" / tier / task / "complete.json", {
                "schema": "glt-v2-r2-ridge-task-complete-v1", "tier": tier,
                "task": task, "folds": 5, "split_manifest_sha256": split_sha,
                "checkpoint": checkpoint_meta,
            })
            print(f"probe {tier}/{task} complete", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "probe_fold_metrics.csv", index=False, float_format="%.17g")
    summary = frame.groupby(["tier", "task"], sort=False).agg(
        r2_mean=("test_r2", "mean"), r2_std=("test_r2", lambda x: np.std(x, ddof=0)),
        mae_mean=("test_mae", "mean"), rmse_mean=("test_rmse", "mean"),
    ).reset_index()
    summary.to_csv(output_root / "probe_task_summary.csv", index=False, float_format="%.17g")
    atomic_json(output_root / "probe_summary.json", {
        "schema": "glt-v2-r2-ridge-probe-report-v1", "units": len(rows),
        "tiers": list(selected_tiers), "tasks": list(selected_tasks),
        "target_transform": "standard", "alphas": list(ALPHAS),
    })


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="results/glt_v2_r2_o8_nomd_mipsloss_005k/probes")
    parser.add_argument("--tiers", nargs="+", default=list(TIERS))
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    args = parser.parse_args(argv)
    invalid = set(args.tiers) - set(TIERS)
    if invalid or set(args.tasks) - set(TASKS):
        raise SystemExit(f"invalid tiers/tasks: {invalid}")
    output = (ROOT / args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite probe output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    run(output, args.tiers, args.tasks)


if __name__ == "__main__":
    main()
