#!/usr/bin/env python3
"""Phase 8 strict validation of one formal dual-GLT deploy checkpoint.

Checks, in order:

1. checkpoint metadata (architecture, fusion_mode, step, use_md200)
2. the state dict is exactly the downstream encoder/fusion tensor set: no
   chemistry/geometry/fingerprint head, no geometry-normalization module, no
   optimizer state, all tensors finite
3. the deploy tensors are bitwise identical to the encoder subset of the same
   step's resume checkpoint (identity, not just shape)
4. the package strict-loads into a freshly built downstream model at that step
5. a real-record read-only forward returns finite predictions of shape [n, 1]

Read-only for the frozen cache; writes only the JSON report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.dataset.cache_lifecycle import snapshot_tree
from src.dataset.glt_dual import build_dual_sample, dual_glt_collate
from src.modules.glt_dual import build_dual_glt_model
from src.modules.glt_dual_pretrain import load_deployment
from src.training.glt_dual_runtime import open_source, write_json


FORBIDDEN_KEY_PARTS = (
    "atom_head", "length_head", "angle_head", "fp_head", "geometry_norm",
    "predictor", "optimizer", "exp_avg",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_stats(state_dict):
    total = sum(int(value.numel()) for value in state_dict.values())
    nonfinite = sorted(name for name, value in state_dict.items()
                       if not bool(torch.isfinite(value).all()))
    return {"tensors": len(state_dict), "parameters": total,
            "nonfinite_tensors": nonfinite,
            "dtypes": sorted({str(value.dtype) for value in state_dict.values()})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--fusion-mode", default="concat")
    parser.add_argument("--cohort-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--index", action="append", type=int)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    checkpoint_path = Path(args.checkpoint).resolve()
    package = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    deploy = package["state_dict"]

    model = build_dual_glt_model(args.fusion_mode, dropout=0).eval()
    expected_keys = {name for name in model.state_dict() if not name.startswith("predictor.")}
    metadata = {
        "architecture": package.get("architecture"),
        "expected_architecture": model.architecture_name,
        "fusion_mode": package.get("fusion_mode"),
        "step": package.get("step"),
        "use_md200": package.get("use_md200"),
    }
    metadata_ok = (
        metadata["architecture"] == model.architecture_name
        and metadata["fusion_mode"] == args.fusion_mode
        and int(metadata["step"]) == args.expected_step
        and metadata["use_md200"] is False
    )

    forbidden = sorted(name for name in deploy
                       if any(part in name for part in FORBIDDEN_KEY_PARTS))
    key_set_ok = set(deploy) == expected_keys
    stats = _tensor_stats(deploy)

    resume = torch.load(Path(args.resume).resolve(), map_location="cpu", weights_only=False)
    resume_style = {name[len("encoder."):]: value for name, value in resume["model"].items()
                    if name.startswith("encoder.")}
    identity = {
        "resume_step": int(resume["step"]),
        "resume_encoder_tensors": len(resume_style),
        "missing_in_deploy": sorted(set(resume_style) - set(deploy)),
        "extra_in_deploy": sorted(set(deploy) - set(resume_style)),
    }
    mismatched = sorted(
        name for name in set(deploy) & set(resume_style)
        if not bool(torch.equal(deploy[name].cpu(), resume_style[name].cpu()))
    )
    identity["mismatched_tensors"] = mismatched
    identity["bitwise_identical"] = (
        not mismatched and not identity["missing_in_deploy"] and not identity["extra_in_deploy"]
        and int(resume["step"]) == args.expected_step)

    load_error = None
    try:
        downstream = build_dual_glt_model(args.fusion_mode, dropout=0).eval()
        load_deployment(downstream, package, expected_step=args.expected_step)
    except Exception as error:  # recorded, not raised: the report carries the reason
        load_error = f"{type(error).__name__}: {error}"
        downstream = None

    cache_root = Path(args.cache_root).resolve()
    before = snapshot_tree(cache_root)
    forward = {"status": "NOT_RUN"}
    source, _ = open_source(args.cohort_root, cache_root)
    try:
        if downstream is not None:
            device = torch.device(args.device)
            downstream = downstream.to(device)
            indices = args.index
            if indices is None:
                indices = []
                for index in range(min(len(source), 256)):
                    sample = build_dual_sample(*source[index])
                    if sample.geometry_valid and bool(sample.bond_center.any()):
                        indices.append(index)
                    if len(indices) == 2:
                        break
            records = [source[index] for index in indices]
            batch = dual_glt_collate([build_dual_sample(*record) for record in records]).to(device)
            with torch.no_grad():
                prediction = downstream(batch)
            finite = bool(torch.isfinite(prediction).all())
            forward = {
                "status": "PASS" if prediction.shape == (len(records), 1) and finite else "FAIL",
                "sample_indices": indices,
                "shape": list(prediction.shape), "finite": finite,
                "prediction": prediction.float().flatten().tolist(),
            }
    finally:
        source.close()
    zero_write = snapshot_tree(cache_root) == before

    valid = (metadata_ok and key_set_ok and not forbidden and not stats["nonfinite_tensors"]
             and identity["bitwise_identical"] and load_error is None
             and forward.get("status") == "PASS" and zero_write)
    report = {
        "scope": "Phase 8 strict deploy validation; read-only for the frozen cache",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checks": {
            "metadata_ok": metadata_ok, "metadata": metadata,
            "key_set_matches_downstream": key_set_ok,
            "forbidden_keys": forbidden,
            "tensor_stats": stats,
            "resume_identity": identity,
            "strict_load_error": load_error,
            "real_record_forward": forward,
            "cache_zero_write": zero_write,
        },
        "FIXED_CONCAT_DEPLOY_VALID": "YES" if valid else "NO",
    }
    write_json(args.report_json, report)
    print(json.dumps({"FIXED_CONCAT_DEPLOY_VALID": report["FIXED_CONCAT_DEPLOY_VALID"],
                      "checks": report["checks"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
