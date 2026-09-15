#!/usr/bin/env python3
"""Bounded real-record dual-GLT forward/backward and deploy-load smoke."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.dataset.cache_lifecycle import snapshot_tree
from src.dataset.glt_dual import build_dual_sample, dual_glt_collate
from src.dataset.glt_dual_pretrain import prepare_pretrain_sample, pretrain_collate
from src.modules.glt_dual import build_dual_glt_model
from src.modules.glt_dual_pretrain import (
    DualPretrainer, deployment_package, global_objective, load_deployment,
)
from src.training.glt_dual_runtime import move_labels, open_source, write_json


def _required_gradients(model, mode):
    groups = {
        "o8_encoder": ("encoder.o8.atom_embedding.projection.weight",),
        "glt_encoder": ("encoder.glt.endpoint.weight",),
        "chemistry_head": ("atom_head.head.weight",),
        "geometry_heads": ("length_head.2.weight", "angle_head.2.weight"),
        "fingerprint_head": ("fp_head.3.weight",),
    }
    if mode == "concat":
        groups["fusion"] = ("encoder.norm2.weight", "encoder.norm3.weight")
    else:
        # One knowledge source makes Q/K softmax identically one; Value is the
        # scientifically required trainable update in this configuration.
        groups["kfuse_value"] = ("encoder.kfuse.v_proj.glt3d.weight",)
    report = {}
    for group, names in groups.items():
        values = []
        for name in names:
            grad = model.get_parameter(name).grad
            valid = grad is not None and bool(torch.isfinite(grad).all())
            norm = float(grad.float().norm()) if valid else 0.0
            values.append({"name": name, "finite": valid, "norm": norm})
        if not any(item["finite"] and item["norm"] > 0 for item in values):
            raise RuntimeError(f"missing required gradient group: {group}")
        report[group] = values
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--cohort-root", required=True)
    parser.add_argument("--index", action="append", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA smoke requested without CUDA")
        torch.cuda.set_device(device)
    cache_root = Path(args.cache_root).resolve()
    before = snapshot_tree(cache_root)
    source, _ = open_source(args.cohort_root, cache_root)
    try:
        indices = args.index
        if indices is None:
            indices = []
            for index in range(min(len(source), 256)):
                sample = build_dual_sample(*source[index])
                if sample.geometry_valid and bool(sample.bond_center.any()):
                    indices.append(index)
                if len(indices) == 2:
                    break
        if not 1 <= len(indices) <= 2 or len(set(indices)) != len(indices):
            raise ValueError("smoke requires one or two distinct real records")
        if min(indices) < 0 or max(indices) >= len(source):
            raise IndexError("smoke cohort index is out of range")
        records = [source[index] for index in indices]
        clean_cpu = dual_glt_collate([build_dual_sample(*record) for record in records])
        centers = torch.bincount(
            clean_cpu.bond_batch[clean_cpu.bond_center], minlength=len(records)
        )
        if not bool(clean_cpu.geometry_valid.all()) or not bool((centers > 0).all()):
            raise ValueError("selected real records require valid center geometry")
        prepared = [
            prepare_pretrain_sample(
                *record, seed=42,
                key=source.samples[index][0].hex(), position=position,
            )
            for position, (index, record) in enumerate(zip(indices, records))
        ]
        batch_cpu, labels_cpu = pretrain_collate(prepared)
        keys = [source.samples[index][0].hex() for index in indices]
        outputs = []
        for mode in ("concat", "kfuse"):
            torch.manual_seed(42)
            model = DualPretrainer(mode).to(device).train()
            batch = batch_cpu.clone().to(device)
            labels = move_labels(labels_cpu, device)
            with torch.autocast(
                device.type, dtype=torch.bfloat16,
                enabled=args.amp_dtype == "bf16",
            ):
                output = model(batch, labels)
                loss = global_objective(output["sums"], output["counts"])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite three-task loss")
            loss.backward()
            gradients = _required_gradients(model, mode)
            for parameter in model.parameters():
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError("nonfinite gradient")
            package = deployment_package(model, 0)
            downstream = build_dual_glt_model(mode, dropout=0).to(device).eval()
            load_deployment(downstream, package, expected_step=0)
            clean = clean_cpu.clone().to(device)
            with torch.no_grad(), torch.autocast(
                device.type, dtype=torch.bfloat16,
                enabled=args.amp_dtype == "bf16",
            ):
                prediction = downstream(clean)
            if prediction.shape != (len(records), 1) or not bool(torch.isfinite(prediction).all()):
                raise RuntimeError("invalid downstream deploy forward")
            outputs.append({
                "mode": mode, "status": "PASS", "loss": float(loss.detach()),
                "valid_graphs": output["counts"].tolist(),
                "targets": output["targets"].tolist(), "gradients": gradients,
                "deploy_step": 0,
            })
            del model, downstream, output, loss, package, batch, labels, clean
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        parent_hash = source.cohort["manifest"]["main_bundle_hash"]
        cohort_hash = source.cohort["manifest_hash"]
    finally:
        source.close()
    if snapshot_tree(cache_root) != before:
        raise RuntimeError("runtime smoke modified the frozen cache")
    report = {
        "status": "PASS", "scope": "real-record local runtime smoke",
        "device": str(device), "amp_dtype": args.amp_dtype,
        "sample_indices": indices, "sample_keys": keys,
        "main_bundle_hash": parent_hash, "cohort_manifest_hash": cohort_hash,
        "modes": outputs, "cache_zero_write": True,
    }
    write_json(args.report_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
