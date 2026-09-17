#!/usr/bin/env python3
"""Pretrain the matched O8 student with a frozen PolyPaiNN target."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.dataset.o8_dnd_student import StudentRankMicrobatchStream, student_pretrain_collate, build_student_pretrain_sample
from src.dataset.poly_painn_teacher import build_teacher_sample
from src.modules.o8_dnd_student import (
    O8DNDStudentPretrainer, common_initialized_o8_dnd_student,
    student_deployment_package, student_global_objective,
)
from src.modules.poly_painn_teacher import PolyPaiNNTeacher, load_teacher_deployment
from src.training.glt_dual_runtime import (
    OrderedSampleStream, open_source, restore_rng, rng_state, save_checkpoint,
    require_tmux, scheduled_lr, write_json,
)
from src.utils import set_global_seed


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank_device():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        ids = [part.strip() for part in visible.split(",") if part.strip()]
        if not ids or "0" in ids:
            raise RuntimeError("Student pretraining requires physical GPUs 1,2,3 only")
        if local_rank >= len(ids):
            raise RuntimeError("LOCAL_RANK is outside CUDA_VISIBLE_DEVICES")
        if world != 3 or len(ids) != 3:
            raise RuntimeError("Student pretraining requires exactly three visible GPUs")
        torch.cuda.set_device(device)
    return rank, world, device


def _deterministic_average_gradients(model, rank, world):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
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


def _module_grad_norms(model):
    groups = {
        "o8": model.encoder.o8,
        "atom_head": model.atom_head,
        "fp_head": model.fp_head,
        "distill_adapter": model.distill_adapter,
    }
    report = {}
    for name, module in groups.items():
        values = [p.grad.detach().float().reshape(-1) for p in module.parameters()
                  if p.grad is not None]
        report[name] = float(torch.cat(values).norm()) if values else 0.0
    return report


def _load_resume(path, rank, identity, ordered_keys, model, optimizer):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "eq3d-dnd-o8-student-resume-v1":
        raise ValueError("unexpected DND student resume schema")
    if payload.get("identity") != identity or payload.get("ordered_keys") != ordered_keys:
        raise ValueError("DND student resume data/config/identity differs")
    step = int(payload.get("step", -1))
    if int(payload.get("next_position", -1)) != step * int(identity["global_batch"]):
        raise ValueError("DND student resume next_position mismatch")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    states = payload.get("rng")
    if not isinstance(states, list) or rank >= len(states):
        raise ValueError("DND student resume lacks per-rank RNG state")
    return step, states[rank]


def _prepare_step(source, stream, *, step, rank, world, micro, accumulation, global_batch,
                  seed, ratio):
    batches = []
    for offset in range(accumulation):
        records = []
        for local in range(micro):
            position = (step * global_batch + offset * world * micro
                        + rank * micro + local)
            index = stream.index_at(position)
            key = source.samples[index][0].hex()
            topology, trimer, smiles = source[index]
            records.append(build_student_pretrain_sample(
                topology, trimer, smiles, seed=seed, key=key, position=position,
                ratio=ratio, static=source.static_for(index), target=source.target_for(index),
            ))
        batches.append(student_pretrain_collate(records))
    return batches


def _local_counts(prepared, device):
    counts = torch.zeros(3, dtype=torch.float32, device=device)
    for data, labels, teacher_data in prepared:
        graphs = int(data.graph_available.numel())
        mask = labels["atom_mask"].bool()
        if bool(mask.any()):
            chem_valid = torch.bincount(
                data.canonical_graph_index[mask], minlength=graphs
            ).bool()
        else:
            chem_valid = torch.zeros(graphs, dtype=torch.bool)
        fp_valid = data.graph_available.bool()
        distill_valid = torch.bincount(
            teacher_data.central_batch.long(), minlength=graphs
        ).bool()
        counts += torch.stack((chem_valid.sum(), fp_valid.sum(), distill_valid.sum())).to(
            device=device, dtype=torch.float32
        )
    return counts


def _autocast(device, dtype):
    return (torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and dtype == "bf16" else nullcontext())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "teacher-deployment", "cohort-root", "cache-root",
                 "dual-static-root", "pretrain-target-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--prep-workers", type=int, default=0)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-step", type=int, default=0)
    parser.add_argument("--no-deploy", action="store_true")
    args = parser.parse_args()
    require_tmux()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config.get("schema") != "eq3d-dnd-o8-student-v1":
        raise ValueError("unexpected DND student config schema")
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
                raise FileExistsError("new DND student run requires a new output directory")
            output.mkdir(parents=True, exist_ok=bool(args.resume))
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if world > 1:
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(error[0])

    teacher_package = torch.load(args.teacher_deployment, map_location="cpu", weights_only=False)
    hyper = teacher_package.get("hyperparameters", {})
    teacher = PolyPaiNNTeacher(**{
        key: hyper[key] for key in (
            "hidden_channels", "num_layers", "cutoff", "max_num_neighbors",
            "rbf_dim", "max_atomic_number",
        ) if key in hyper
    }).to(device).eval()
    load_teacher_deployment(teacher, teacher_package, expected_step=5000)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher_sha256 = _sha256(args.teacher_deployment)

    from src.training.glt_dual_runtime import open_source
    source, _ = open_source(
        args.cohort_root, args.cache_root,
        dual_static_root=args.dual_static_root,
        pretrain_target_root=args.pretrain_target_root,
    )
    try:
        micro = int(config["microbatch"])
        global_batch = int(config["global_batch"])
        if micro <= 0 or global_batch <= 0 or global_batch % (micro * world):
            raise ValueError("global_batch must be divisible by microbatch*world")
        accumulation = global_batch // (micro * world)
        max_steps = int(config["max_optimizer_steps"])
        save_every = int(config["save_every"])
        if max_steps <= 0 or save_every <= 0:
            raise ValueError("invalid DND student training budget")
        set_global_seed(int(config["seed"]))
        base = common_initialized_o8_dnd_student(int(config["seed"]), dropout=0.1).to(device)
        optimizer = torch.optim.AdamW(
            base.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
        )
        deterministic = bool(config.get("deterministic_gradient_reduction", False))
        module = (DistributedDataParallel(base, device_ids=[device.index], find_unused_parameters=False)
                  if world > 1 and not deterministic else base)
        ordered_keys = [key.hex() for key, _ in source.samples]
        identity = {
            "schema": config["schema"], "config": config, "world_size": world,
            "global_batch": global_batch, "accumulation": accumulation,
            "sample_count": len(source), "cohort_hash": source.cohort["manifest_hash"],
            "main_bundle_hash": source.bundle.bundle_hash,
            "static_manifest_hash": source.static_cache.manifest_hash if source.static_cache else None,
            "target_manifest_hash": source.target_cache.manifest_hash if source.target_cache else None,
            "ordered_key_hash": source.cohort["manifest"].get("ordered_sample_key_hash"),
            "common_init_digest": base.common_init_digest,
            "teacher_deployment_sha256": teacher_sha256,
            "architecture": base.architecture_name,
        }
        start, resume_rng = 0, None
        if args.resume:
            start, resume_rng = _load_resume(
                args.resume, rank, identity, ordered_keys, base, optimizer
            )
            if start < 0 or start >= max_steps:
                raise ValueError("DND student resume step is outside configured budget")
        else:
            set_global_seed(int(config["seed"]) + rank)
        stop = max_steps
        if args.stop_after_step:
            if not start < int(args.stop_after_step) <= max_steps:
                raise ValueError("--stop-after-step is outside configured budget")
            stop = int(args.stop_after_step)
        if args.resume and any(int(path.stem.rsplit("_", 1)[-1]) > start
                               for path in output.glob("resume_*.pt")):
            raise FileExistsError("resume would overwrite a later DND student checkpoint")
        if rank == 0:
            write_json(output / "run.json", {
                "schema": "eq3d-dnd-o8-student-run-v1", "command": sys.argv,
                "identity": identity, "ordered_key_hash": identity["ordered_key_hash"],
                "world_size": world, "start_step": start, "stop_after_step": stop,
                "prep_workers": int(args.prep_workers), "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu0_used": False, "teacher_deployment_sha256": teacher_sha256,
            })

        prefetch = None
        if args.prep_workers:
            dataset = StudentRankMicrobatchStream(
                source, seed=config["seed"], world=world, rank=rank,
                microbatch=micro, accumulation=accumulation, start_step=start,
                max_steps=stop, ratio=float(config["atom_mask_ratio"]),
            )
            prefetch = iter(torch.utils.data.DataLoader(
                dataset, batch_size=None, num_workers=int(args.prep_workers),
                prefetch_factor=4, persistent_workers=False, pin_memory=False,
            ))
        if resume_rng is not None:
            restore_rng(resume_rng)
        stream = OrderedSampleStream(len(source), config["seed"])
        weights = (
            float(config["chemistry_weight"]),
            float(config["fingerprint_weight"]),
            float(config["distillation_weight"]),
        )
        for step in range(start, stop):
            prepared = ([next(prefetch) for _ in range(accumulation)] if prefetch is not None
                        else _prepare_step(
                            source, stream, step=step, rank=rank, world=world,
                            micro=micro, accumulation=accumulation,
                            global_batch=global_batch, seed=config["seed"],
                            ratio=float(config["atom_mask_ratio"]),
                        ))
            counts = _local_counts(prepared, device)
            if world > 1:
                dist.all_reduce(counts)
            if bool((counts <= 0).any()):
                raise ValueError("DND student step has an empty objective component")
            lr = scheduled_lr(
                step, **{key: config[key] for key in ("lr", "warmup_steps", "schedule_total_steps", "end_lr")}
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros(3, device=device)
            targets = torch.zeros(3, device=device)
            distill_values = []
            for offset, (data, labels, teacher_data) in enumerate(prepared):
                sync = (module.no_sync() if world > 1 and not deterministic
                        and offset + 1 < accumulation else nullcontext())
                with sync:
                    moved_data = data.to(device, non_blocking=True)
                    moved_teacher = teacher_data.to(device, non_blocking=True)
                    moved_labels = {
                        key: value.to(device) if torch.is_tensor(value) else value
                        for key, value in labels.items()
                    }
                    with _autocast(device, config["amp_dtype"]):
                        result = module(moved_data, moved_labels, moved_teacher, teacher)
                        loss = student_global_objective(
                            result["sums"], counts, world, weights=weights
                        )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite DND student loss")
                    loss.backward()
                totals += result["sums"].detach()
                targets += result["targets"].detach()
                distill_values.append(result["distill_values"].detach().float())
            if world > 1 and deterministic:
                _deterministic_average_gradients(base, rank, world)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                base.parameters(), 1.0, error_if_nonfinite=True
            ))
            optimizer.step()
            if world > 1:
                dist.all_reduce(totals)
                dist.all_reduce(targets)
            component_losses = (totals / counts.clamp_min(1)).detach().cpu().tolist()
            norms = _module_grad_norms(base)
            record = {
                "step": step + 1, "rank": rank, "lr": float(lr),
                "chemistry_loss": float(component_losses[0]),
                "fingerprint_loss": float(component_losses[1]),
                "distillation_loss": float(component_losses[2]),
                "loss": float(sum(value * weight for value, weight in zip(component_losses, weights))),
                "objective_counts": counts.detach().cpu().tolist(),
                "target_counts": targets.detach().cpu().tolist(),
                "grad_norm_preclip": grad_norm, "module_grad_norms": norms,
                "teacher_grad_none": all(parameter.grad is None for parameter in teacher.parameters()),
                "teacher_deployment_sha256": teacher_sha256,
                "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu0_used": False,
            }
            if not all(torch.isfinite(torch.tensor(value, dtype=torch.float64))
                       for value in component_losses + [record["loss"], grad_norm]):
                raise FloatingPointError("nonfinite DND student diagnostics")
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
                        "schema": "eq3d-dnd-o8-student-resume-v1", "identity": identity,
                        "ordered_keys": ordered_keys, "step": step + 1,
                        "next_position": (step + 1) * global_batch,
                        "model": base.state_dict(), "optimizer": optimizer.state_dict(),
                        "rng": states, "scheduler": {"step": step + 1, "lr": float(lr)},
                    }
                    save_checkpoint(output / f"resume_{step + 1:05d}.pt", payload)
                    if not args.no_deploy:
                        save_checkpoint(
                            output / f"deploy_{step + 1:05d}.pt",
                            student_deployment_package(base, step + 1,
                                                       teacher_sha256=teacher_sha256),
                        )
                if world > 1:
                    dist.barrier()
    finally:
        source.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
