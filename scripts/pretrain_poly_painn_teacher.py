#!/usr/bin/env python3
"""Pretrain the independent PolyPaiNN teacher with deterministic denoising."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.dataset.poly_painn_teacher import TeacherRankMicrobatchStream, teacher_collate, build_teacher_sample
from src.modules.poly_painn_teacher import (
    PolyPaiNNTeacher, teacher_deployment_package, teacher_global_objective,
)
from src.training.glt_dual_runtime import (
    OrderedSampleStream, restore_rng, rng_state, save_checkpoint,
    require_tmux, scheduled_lr, write_json,
)
from src.utils import set_global_seed


def _hash_keys(source):
    digest = hashlib.sha256()
    for key, _ in source.samples:
        digest.update(bytes(key))
    return digest.hexdigest()


def _rank_device():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible:
            raise RuntimeError("GPU teacher runs must explicitly set CUDA_VISIBLE_DEVICES (GPU0 is forbidden)")
        ids = [part.strip() for part in visible.split(",") if part.strip()]
        if "0" in ids:
            raise RuntimeError("GPU0 is forbidden for EQ3D teacher runs")
        if local_rank >= len(ids):
            raise RuntimeError("LOCAL_RANK is outside CUDA_VISIBLE_DEVICES")
        torch.cuda.set_device(device)
    return rank, world, device


def _per_graph_stats(result, labels, graph_count):
    error = (result["predicted_noise"].float() - labels["epsilon"].float()).square().mean(dim=-1)
    graph = result["predicted_noise"].new_zeros((graph_count,), dtype=torch.float32)
    graph.index_add_(0, labels["central_batch"], error)
    atoms = torch.bincount(labels["central_batch"], minlength=graph_count).to(graph.dtype)
    graph = graph / atoms.clamp_min(1)
    return graph.sum(), torch.tensor(float(graph_count), device=graph.device), int(error.numel())


def _module_grad_norms(model):
    values = [p.grad.detach().float().reshape(-1) for p in model.parameters()
              if p.grad is not None]
    return float(torch.cat(values).norm()) if values else 0.0


def _load_resume(path, rank, identity, ordered_keys, base, optimizer):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "poly-painn-teacher-resume-v1":
        raise ValueError("unexpected teacher resume schema")
    if payload.get("identity") != identity or payload.get("ordered_keys") != ordered_keys:
        raise ValueError("teacher resume data/config/identity differs")
    step = int(payload.get("step", -1))
    if int(payload.get("next_position", -1)) != step * int(identity["global_batch"]):
        raise ValueError("teacher resume next_position mismatch")
    base.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    states = payload.get("rng")
    if not isinstance(states, list) or rank >= len(states):
        raise ValueError("teacher resume lacks per-rank RNG state")
    return step, states[rank]


def _prepare_step(source, stream, *, step, rank, world, micro, accumulation, global_batch,
                  seed, sigma):
    batches = []
    for offset in range(accumulation):
        records = []
        for local in range(micro):
            position = step * global_batch + offset * world * micro + rank * micro + local
            index = stream.index_at(position)
            key = source.samples[index][0].hex()
            topology, trimer, _ = source[index]
            records.append(build_teacher_sample(
                topology, trimer, seed=seed, key=key, position=position, sigma=sigma,
            ))
        batches.append(teacher_collate(records))
    return batches


def _autocast(device, amp_dtype):
    return (torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and amp_dtype == "bf16" else nullcontext())


def _hist_quantile(histogram, quantile):
    total = int(histogram.sum().item())
    if total <= 0:
        return 0
    threshold = max(1, int(math.ceil(float(quantile) * total)))
    cumulative = torch.cumsum(histogram, dim=0)
    return int(torch.searchsorted(cumulative, torch.tensor(threshold, device=histogram.device)).item())


def _deterministic_average_gradients(model, rank, world):
    """Average gradients in a fixed rank order for the bounded exact-resume smoke.

    NCCL bucket all-reduce is numerically reproducible for most runs but may
    choose a different reduction tree after a fresh launch.  This opt-in path
    is used only by the 4-update smoke; formal training keeps normal DDP.
    """
    parameters = [parameter for parameter in model.parameters()
                  if parameter.requires_grad]
    flat = torch.cat([
        (parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).reshape(-1)
        for parameter in parameters
    ])
    gathered = [torch.empty_like(flat) for _ in range(int(world))] if int(rank) == 0 else None
    dist.gather(flat, gather_list=gathered, dst=0)
    if int(rank) == 0:
        average = torch.zeros_like(flat)
        for value in gathered:
            average = average + value
        average = average / int(world)
    else:
        average = torch.empty_like(flat)
    dist.broadcast(average, src=0)
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        if parameter.grad is None:
            parameter.grad = torch.empty_like(parameter)
        parameter.grad.copy_(average[offset:offset + size].view_as(parameter))
        offset += size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "cohort-root", "cache-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--prep-workers", type=int, default=0)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-step", type=int, default=0)
    parser.add_argument("--no-deploy", action="store_true")
    args = parser.parse_args()
    require_tmux()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config.get("schema") != "eq3d-dnd-teacher-v1":
        raise ValueError("unexpected PolyPaiNN teacher config schema")
    if args.prep_workers < 0:
        raise ValueError("--prep-workers must be non-negative")
    rank, world, device = _rank_device()
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    output = Path(args.output).resolve()
    error = [None]
    if rank == 0:
        try:
            if output.exists() and not args.resume:
                raise FileExistsError("new teacher run requires a new output directory")
            output.mkdir(parents=True, exist_ok=bool(args.resume))
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if world > 1:
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(error[0])

    from src.training.glt_dual_runtime import open_source
    source, _ = open_source(args.cohort_root, args.cache_root)
    try:
        micro = int(config["microbatch"])
        global_batch = int(config["global_batch"])
        if micro <= 0 or global_batch <= 0 or global_batch % (micro * world):
            raise ValueError("global_batch must be divisible by microbatch*world")
        accumulation = global_batch // (micro * world)
        max_steps = int(config["max_optimizer_steps"])
        save_every = int(config["save_every"])
        if max_steps <= 0 or save_every <= 0:
            raise ValueError("invalid teacher training budget")
        # Construct all ranks from an identical seed, then diverge only their
        # runtime RNG streams after the module has been initialized.
        set_global_seed(int(config["seed"]))
        base = PolyPaiNNTeacher(
            hidden_channels=int(config["hidden_channels"]),
            num_layers=int(config["num_layers"]),
            cutoff=float(config["cutoff"]),
            max_num_neighbors=int(config["max_num_neighbors"]),
            rbf_dim=int(config["rbf_dim"]),
        ).to(device)
        optimizer = torch.optim.AdamW(
            base.parameters(), lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )
        deterministic_reduction = bool(config.get("deterministic_gradient_reduction", False))
        module = (DistributedDataParallel(
            base, device_ids=[device.index], find_unused_parameters=False
        ) if world > 1 and not deterministic_reduction else base)
        ordered_keys = [key.hex() for key, _ in source.samples]
        identity = {
            "schema": config["schema"], "config": config, "world_size": world,
            "global_batch": global_batch, "accumulation": accumulation,
            "sample_count": len(source), "cohort_hash": source.cohort["manifest_hash"],
            "main_bundle_hash": source.bundle.bundle_hash,
            "ordered_key_hash": source.cohort["manifest"].get("ordered_sample_key_hash"),
            "architecture": base.architecture_name,
        }
        start, resume_rng = 0, None
        if args.resume:
            start, resume_rng = _load_resume(args.resume, rank, identity, ordered_keys, base, optimizer)
            if start < 0 or start >= max_steps:
                raise ValueError("teacher resume step is outside configured budget")
        else:
            set_global_seed(int(config["seed"]) + rank)
        stop = max_steps
        if args.stop_after_step:
            if not start < int(args.stop_after_step) <= max_steps:
                raise ValueError("--stop-after-step must be after start and within budget")
            stop = int(args.stop_after_step)
        if args.resume and any(int(path.stem.rsplit("_", 1)[-1]) > start
                               for path in output.glob("resume_*.pt")):
            raise FileExistsError("resume would overwrite a later teacher checkpoint")
        if rank == 0:
            write_json(output / "run.json", {
                "schema": "eq3d-dnd-teacher-run-v1", "command": sys.argv,
                "identity": identity, "ordered_key_hash": identity["ordered_key_hash"],
                "world_size": world, "start_step": start, "stop_after_step": stop,
                "prep_workers": int(args.prep_workers), "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "uses_trimer": True, "uses_geometry": True, "gpu0_used": False,
            })
        # Restore after DataLoader iterator creation: worker prefetch setup may
        # consume the parent torch RNG, while sample noise itself is hash-based.
        prefetch = None
        if args.prep_workers:
            dataset = TeacherRankMicrobatchStream(
                source, seed=config["seed"], world=world, rank=rank,
                microbatch=micro, accumulation=accumulation,
                start_step=start, max_steps=stop, sigma=float(config["sigma"]),
            )
            prefetch = iter(torch.utils.data.DataLoader(
                dataset, batch_size=None, num_workers=int(args.prep_workers),
                prefetch_factor=4, persistent_workers=False, pin_memory=False,
            ))
        if resume_rng is not None:
            restore_rng(resume_rng)
        stream = OrderedSampleStream(len(source), config["seed"])
        threshold = float(config.get("neighbor_hit_max_stop_fraction", 0.1))
        for step in range(start, stop):
            prepared = ([next(prefetch) for _ in range(accumulation)] if prefetch is not None
                        else _prepare_step(
                            source, stream, step=step, rank=rank, world=world,
                            micro=micro, accumulation=accumulation,
                            global_batch=global_batch, seed=config["seed"],
                            sigma=float(config["sigma"]),
                        ))
            local_graphs = sum(int(labels["positions"].numel()) for _, labels in prepared)
            counts = torch.tensor([float(local_graphs)], device=device)
            if world > 1:
                dist.all_reduce(counts)
            if float(counts.item()) <= 0:
                raise ValueError("teacher step contains no graphs")
            lr = scheduled_lr(step, **{key: config[key] for key in ("lr", "warmup_steps", "schedule_total_steps", "end_lr")})
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            total_sum = torch.zeros((), device=device)
            total_atoms = 0
            total_hist = torch.zeros(int(config["max_num_neighbors"]) + 1, device=device, dtype=torch.long)
            scalar_values, vector_values = [], []
            for offset, (data, labels) in enumerate(prepared):
                # Attach central batch to the labels only to keep Data's index
                # offset logic explicit and avoid relying on PyG custom batching.
                labels["central_batch"] = data.central_batch
                sync = (module.no_sync()
                        if world > 1 and not deterministic_reduction and offset + 1 < accumulation
                        else nullcontext())
                with sync:
                    moved = data.to(device, non_blocking=True)
                    moved_labels = {key: value.to(device) if torch.is_tensor(value) else value
                                    for key, value in labels.items()}
                    with _autocast(device, config["amp_dtype"]):
                        result = module(moved)
                        error_values = (result["predicted_noise"].float()
                                        - moved_labels["epsilon"].float()).square().mean(dim=-1)
                        graph_sum = result["predicted_noise"].new_zeros(
                            (int(moved.central_batch.max().item()) + 1,), dtype=torch.float32
                        )
                        graph_sum.index_add_(0, moved.central_batch, error_values)
                        graph_atoms = torch.bincount(
                            moved.central_batch,
                            minlength=graph_sum.numel(),
                        ).to(graph_sum.dtype)
                        local_sum = (graph_sum / graph_atoms.clamp_min(1)).sum()
                        loss = teacher_global_objective(
                            graph_sum / graph_atoms.clamp_min(1), counts, world
                        )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite teacher denoising loss")
                    loss.backward()
                total_sum += local_sum.detach()
                total_atoms += int(error_values.numel())
                total_hist += result["neighbor_hist"].detach().to(total_hist.dtype)
                scalar_values.append(result["central_scalar_states"].detach().float().pow(2).mean().sqrt())
                vector_values.append(result["central_vector_states"].detach().float().pow(2).mean().sqrt())
            if world > 1 and deterministic_reduction:
                _deterministic_average_gradients(base, rank, world)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0, error_if_nonfinite=True))
            optimizer.step()
            if world > 1:
                dist.all_reduce(total_sum)
                dist.all_reduce(total_hist)
            total_nodes = total_hist.sum().clamp_min(1)
            hit_fraction = float(total_hist[-1].float().div(total_nodes).item())
            if hit_fraction > threshold:
                raise RuntimeError(
                    f"radius max-neighbor cap hit fraction {hit_fraction:.6f} exceeds {threshold:.6f}"
                )
            record = {
                "step": step + 1, "rank": rank, "lr": float(lr),
                "noise_mse": float(total_sum.div(counts.clamp_min(1)).item()),
                "central_graph_count": int(counts.item()), "central_atom_count": int(total_atoms * world),
                "grad_norm_preclip": grad_norm,
                "scalar_rms": float(torch.stack(scalar_values).mean().item()),
                "vector_rms": float(torch.stack(vector_values).mean().item()),
                "neighbor_hist": total_hist.cpu().tolist(),
                "neighbor_p50": _hist_quantile(total_hist, 0.50),
                "neighbor_p95": _hist_quantile(total_hist, 0.95),
                "neighbor_max": int(torch.where(total_hist > 0)[0].max().item()),
                "neighbor_hit_max_fraction": hit_fraction,
                "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu0_used": False,
            }
            print(json.dumps(record), flush=True)
            should_save = ((step + 1) % save_every == 0 or step + 1 == stop)
            if should_save:
                states = [None] * world
                if world > 1:
                    dist.all_gather_object(states, rng_state())
                else:
                    states[0] = rng_state()
                if rank == 0:
                    payload = {
                        "schema": "poly-painn-teacher-resume-v1", "identity": identity,
                        "ordered_keys": ordered_keys, "step": step + 1,
                        "next_position": (step + 1) * global_batch,
                        "model": base.state_dict(), "optimizer": optimizer.state_dict(),
                        "rng": states, "scheduler": {"step": step + 1, "lr": float(lr)},
                    }
                    save_checkpoint(output / f"resume_{step + 1:05d}.pt", payload)
                    if not args.no_deploy:
                        save_checkpoint(output / f"teacher_deploy_{step + 1:05d}.pt",
                                        teacher_deployment_package(base, step + 1))
                if world > 1:
                    dist.barrier()
    finally:
        source.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
