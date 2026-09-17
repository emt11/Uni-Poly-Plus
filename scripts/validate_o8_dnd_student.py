#!/usr/bin/env python3
"""Strictly validate the O8-only DND Student deployment package."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.dataset.glt_o8_control import O8OnlySource, build_o8_sample, o8_collate
from src.modules.glt_o8_control import O8ControlModel, load_o8_deployment
from src.training.glt_dual_runtime import require_tmux, write_json


def _device():
    if torch.cuda.is_available():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible or "0" in {part.strip() for part in visible.split(",") if part.strip()}:
            raise RuntimeError("DND deployment validation requires explicit non-GPU0 CUDA_VISIBLE_DEVICES")
        return torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--cohort-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--dual-static-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    require_tmux()
    device = _device()
    package = torch.load(args.deployment, map_location="cpu", weights_only=False)
    forbidden_metadata = (
        "teacher", "teacher_deployment_sha256", "distill_adapter", "coordinates", "trimer",
        "atom_head", "fp_head",
    )
    state = package.get("state_dict") or {}
    if any(any(token in str(name).lower() for token in forbidden_metadata) for name in state):
        raise ValueError("DND deployment contains forbidden teacher/head/geometry tensor")
    model = O8ControlModel().to(device).eval()
    load_o8_deployment(model, package, expected_step=int(package.get("step", -1)))
    source = O8OnlySource(
        args.cohort_root, args.cache_root, static_root=args.dual_static_root,
    )
    try:
        if not 0 <= int(args.index) < len(source):
            raise IndexError("deployment validation index is outside frozen cohort")
        topology = source[int(args.index)]
        sample = build_o8_sample(
            topology, source.static_for(int(args.index)),
            key=source.samples[int(args.index)][0],
        )
        with torch.no_grad():
            prediction, auxiliary = model(o8_collate([sample]).to(device))
        finite = bool(torch.isfinite(prediction).all())
        report = {
            "schema": "eq3d-dnd-o8-student-validation-v1",
            "deployment": str(Path(args.deployment).resolve()),
            "step": int(package.get("step", -1)), "sample_index": int(args.index),
            "architecture": package.get("architecture"), "arm": package.get("arm"),
            "strict_load": True, "finite": finite, "auxiliary_none": auxiliary is None,
            "state_key_count": len(state),
            "deployment_excludes_teacher_adapter_heads": not any(
                any(token in str(name).lower() for token in forbidden_metadata) for name in state
            ),
            "trimer_access": "not opened by O8OnlySource",
            "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu0_used": False,
            "status": "PASS" if finite and auxiliary is None else "FAIL",
        }
        write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if report["status"] != "PASS":
            raise RuntimeError("DND student deployment validation failed")
    finally:
        source.close()


if __name__ == "__main__":
    main()
