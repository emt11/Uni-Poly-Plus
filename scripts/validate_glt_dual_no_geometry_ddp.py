#!/usr/bin/env python3
"""Bounded three-rank DDP forward/backward check with missing geometry targets.

This is a communication/objective boundary smoke, not a training run.  It
loads one real frozen, geometry-valid sample, builds the current static/target
candidate in memory, and toggles only the collated ``geometry_valid`` flag for
the rank-local cases.  No cache or checkpoint is written other than the small
JSON report requested by the caller.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.dataset.glt_dual_pretrain import prepare_pretrain_sample, pretrain_collate
from src.modules.glt_dual_pretrain import DualPretrainer, global_objective
from src.training.glt_dual_runtime import move_labels, open_source, require_tmux, write_json


def _finite_gradients(module):
    """Return finite/presence summaries without requiring unused heads."""

    present, finite, nonzero = 0, 0, 0
    groups = {"chemistry": [], "fingerprint": [], "geometry": []}
    for name, parameter in module.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        present += 1
        is_finite = bool(torch.isfinite(gradient).all())
        finite += int(is_finite)
        nonzero += int(bool(gradient.abs().sum() > 0))
        if name.startswith("atom_head") or name.startswith("encoder.o8") or name.startswith("encoder.glt"):
            groups["chemistry"].append((name, is_finite, float(gradient.abs().sum())))
        elif name.startswith("fp_head"):
            groups["fingerprint"].append((name, is_finite, float(gradient.abs().sum())))
        elif name.startswith("length_head") or name.startswith("angle_head"):
            groups["geometry"].append((name, is_finite, float(gradient.abs().sum())))
    return {
        "present": present,
        "finite": finite == present,
        "nonzero": nonzero,
        "groups": groups,
    }


def _load_real_sample(args):
    source, _ = open_source(
        args.cohort_root,
        args.cache_root,
        dual_static_root=args.dual_static_root,
        pretrain_target_root=args.pretrain_target_root,
    )
    try:
        limit = min(int(args.probe_limit), len(source))
        selected = None
        for index in range(limit):
            topology, trimer, smiles = source[index]
            if bool(getattr(topology, "graph_available", False)) and bool(
                getattr(trimer, "trimer_geometry_valid", False)
            ):
                selected = (index, topology, trimer, smiles)
                break
        if selected is None:
            raise RuntimeError(
                f"no geometry-valid real sample in first {limit} frozen cohort entries"
            )
        index, topology, trimer, smiles = selected
        key = source.samples[index][0].hex()
        static = source.static_for(index)
        target = source.target_for(index)
        if static is None or target is None:
            raise RuntimeError("DDP boundary smoke requires the current static and target candidates")
        prepared = prepare_pretrain_sample(
            topology,
            trimer,
            smiles,
            seed=42,
            key=key,
            position=0,
            static=static,
            target=target,
        )
        return source, {
            "index": int(index),
            "sample_key": key,
            "source_smiles": str(smiles),
            "cohort_manifest_hash": source.cohort["manifest_hash"],
            "main_bundle_hash": source.bundle.bundle_hash,
            "prepared": prepared,
        }
    except BaseException:
        source.close()
        raise


def _run_case(case, prepared, device, rank, world):
    data, labels = pretrain_collate([prepared])
    if case == "partial_zero":
        geometry_valid = rank == 0
    elif case == "all_zero":
        geometry_valid = False
    else:
        raise ValueError(f"unknown case {case}")
    if not geometry_valid:
        # This is the only per-case mutation: the real static row, coordinates,
        # topology and target tensors remain untouched in the frozen source.
        data.geometry_valid = torch.zeros_like(data.geometry_valid)
    data = data.to(device)
    labels = move_labels(labels, device)
    model = DualPretrainer("concat").to(device)
    module = DistributedDataParallel(
        model, device_ids=[device.index], find_unused_parameters=True
    )
    result = module(data, labels)
    local_counts = result["counts"].detach().clone()
    global_counts = local_counts.clone()
    dist.all_reduce(global_counts)
    loss = global_objective(result["sums"], global_counts, world)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError(f"{case}: non-finite loss")
    loss.backward()
    gradients = _finite_gradients(module.module)
    if not gradients["finite"]:
        raise RuntimeError(f"{case}: non-finite gradient")
    chemistry = gradients["groups"]["chemistry"]
    fingerprint = gradients["groups"]["fingerprint"]
    if not chemistry or any(not item[1] for item in chemistry):
        raise RuntimeError(f"{case}: chemistry gradient is missing/non-finite")
    if not fingerprint or any(not item[1] for item in fingerprint):
        raise RuntimeError(f"{case}: fingerprint gradient is missing/non-finite")
    rows = [None] * world
    dist.all_gather_object(rows, {
        "rank": rank,
        "local_counts": [int(value) for value in local_counts.cpu().tolist()],
        "global_counts": [int(value) for value in global_counts.cpu().tolist()],
        "loss": float(loss.detach().cpu()),
        "gradient": gradients,
        "geometry_valid": bool(geometry_valid),
    })
    # Explicit collective boundary before tearing down the DDP module.
    dist.barrier()
    del module, model, data, labels
    return {
        "local_counts_each_rank": rows,
        "global_counts": [int(value) for value in global_counts.cpu().tolist()],
        "finite_loss": all(torch.isfinite(torch.tensor(row["loss"])) for row in rows),
        "backward_completed": True,
        "gradient_finite": all(row["gradient"]["finite"] for row in rows),
        "chemistry_gradient_finite": all(
            all(item[1] for item in row["gradient"]["groups"]["chemistry"])
            for row in rows
        ),
        "fingerprint_gradient_finite": all(
            all(item[1] for item in row["gradient"]["groups"]["fingerprint"])
            for row in rows
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--dual-static-root", required=True)
    parser.add_argument("--pretrain-target-root", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--probe-limit", type=int, default=16)
    args = parser.parse_args()
    require_tmux()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != 3 or not torch.cuda.is_available():
        raise RuntimeError("this validation requires exactly three CUDA ranks")
    if args.probe_limit <= 0:
        raise ValueError("--probe-limit must be positive")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    source = None
    try:
        source, sample = _load_real_sample(args)
        prepared = sample.pop("prepared")
        partial = _run_case("partial_zero", prepared, device, rank, world)
        all_zero = _run_case("all_zero", prepared, device, rank, world)
        if partial["global_counts"][1] <= 0:
            raise RuntimeError("partial_zero geometry denominator is unexpectedly zero")
        if all_zero["global_counts"][1] != 0:
            raise RuntimeError("all_zero geometry denominator is unexpectedly non-zero")
        if rank == 0:
            report = {
                "status": "PASS",
                "world_size": world,
                "backend": dist.get_backend(),
                "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "sample": sample,
                "partial_zero": partial,
                "all_zero": all_zero,
                "optimizer_updates": 0,
            }
            # ``prepared`` contains tensors and is intentionally excluded from
            # the report; only frozen identity and aggregate diagnostics remain.
            write_json(args.report_json, report)
            print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        if source is not None:
            source.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
