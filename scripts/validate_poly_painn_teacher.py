#!/usr/bin/env python3
"""Validate a PolyPaiNN deployment on one frozen Trimer without new geometry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch_geometric.data import Data

from src.dataset.poly_painn_teacher import build_teacher_sample
from src.modules.poly_painn_teacher import (
    PolyPaiNNTeacher, load_teacher_deployment,
)
from src.training.glt_dual_runtime import open_source, require_tmux, write_json


def _device():
    if torch.cuda.is_available():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible or "0" in {part.strip() for part in visible.split(",") if part.strip()}:
            raise RuntimeError("teacher validation requires explicit non-GPU0 CUDA_VISIBLE_DEVICES")
        return torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    return torch.device("cpu")


def _clean_data(data, clean_pos):
    return Data(
        z=data.z.clone(), pos=clean_pos.clone(), batch=data.batch.clone(),
        central_index=data.central_index.clone(), central_batch=data.central_batch.clone(),
    )


def _rotation():
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--cohort-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    require_tmux()
    device = _device()
    package = torch.load(args.deployment, map_location="cpu", weights_only=False)
    hyper = package.get("hyperparameters", {})
    model = PolyPaiNNTeacher(**{
        key: hyper[key] for key in (
            "hidden_channels", "num_layers", "cutoff", "max_num_neighbors",
            "rbf_dim", "max_atomic_number",
        ) if key in hyper
    }).to(device).eval()
    load_teacher_deployment(model, package, expected_step=int(package.get("step", -1)))
    source, _ = open_source(args.cohort_root, args.cache_root)
    try:
        if not 0 <= int(args.index) < len(source):
            raise IndexError("validation index outside frozen cohort")
        topology, trimer, _ = source[int(args.index)]
        noisy, target = build_teacher_sample(
            topology, trimer, seed=42, key=source.samples[int(args.index)][0].hex(),
            position=0, sigma=0.03,
        )
        clean = _clean_data(noisy, target["clean_pos"])
        clean = clean.to(device)
        with torch.no_grad():
            reference = model(clean)
            translated = _clean_data(noisy, target["clean_pos"] + torch.tensor([3.1, -1.7, 2.2])).to(device)
            moved = model(translated)
            q = _rotation().to(device)
            rotated = _clean_data(noisy, target["clean_pos"] @ q.T).to(device)
            turned = model(rotated)
            permutation = torch.randperm(clean.z.numel(), device=device)
            inverse = torch.empty_like(permutation)
            inverse[permutation] = torch.arange(permutation.numel(), device=device)
            permuted = Data(
                z=clean.z[permutation], pos=clean.pos[permutation], batch=clean.batch[permutation],
                central_index=inverse[clean.central_index], central_batch=clean.central_batch,
            )
            shuffled = model(permuted)
        translation_scalar = float((reference["central_scalar_states"] - moved["central_scalar_states"]).abs().max())
        translation_noise = float((reference["predicted_noise"] - moved["predicted_noise"]).abs().max())
        rotation_scalar = float((reference["central_scalar_states"] - turned["central_scalar_states"]).abs().max())
        rotation_vector = float((turned["central_vector_states"] - reference["central_vector_states"] @ q.T).abs().max())
        rotation_noise = float((turned["predicted_noise"] - reference["predicted_noise"] @ q.T).abs().max())
        permutation_scalar = float((reference["central_scalar_states"] - shuffled["central_scalar_states"]).abs().max())
        permutation_vector = float((reference["central_vector_states"] - shuffled["central_vector_states"]).abs().max())
        finite = all(torch.isfinite(value).all() for value in reference.values() if torch.is_tensor(value))
        report = {
            "schema": "poly-painn-teacher-validation-v1", "deployment": str(Path(args.deployment).resolve()),
            "step": int(package.get("step", -1)), "sample_index": int(args.index),
            "translation_scalar_max_abs": translation_scalar,
            "translation_noise_max_abs": translation_noise,
            "rotation_scalar_max_abs": rotation_scalar,
            "rotation_vector_max_abs": rotation_vector,
            "rotation_noise_max_abs": rotation_noise,
            "permutation_scalar_max_abs": permutation_scalar,
            "permutation_vector_max_abs": permutation_vector,
            "finite": bool(finite), "deployment_excludes_noise_head": not any(
                str(name).startswith("noise_head.") for name in package.get("encoder", {})
            ),
            "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu0_used": False,
            "status": "PASS" if finite and max(
                translation_scalar, translation_noise, rotation_scalar, rotation_vector,
                rotation_noise, permutation_scalar, permutation_vector,
            ) <= 2e-3 else "FAIL",
        }
        write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if report["status"] != "PASS":
            raise RuntimeError("PolyPaiNN equivariance validation failed")
    finally:
        source.close()


if __name__ == "__main__":
    main()
