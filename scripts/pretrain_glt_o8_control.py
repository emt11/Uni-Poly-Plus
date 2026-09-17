#!/usr/bin/env python3
"""Matched O8-only masked-chemistry/Morgan pretraining entry."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.dataset.glt_o8_control import O8OnlySource, O8RankMicrobatchStream, build_o8_pretrain_sample, o8_pretrain_collate
from src.modules.glt_o8_control import common_initialized_o8_pretrainer, o8_deployment_package, o8_global_objective
from src.training.glt_dual_runtime import require_tmux, save_checkpoint, write_json, rng_state, restore_rng, scheduled_lr
from src.utils import set_global_seed


def _module_grad_norms(model):
    groups = {"o8": model.encoder.o8, "atom_head": model.atom_head, "fp_head": model.fp_head}
    report = {}
    for name, module in groups.items():
        values = [p.grad.detach().reshape(-1) for p in module.parameters()
                  if p.grad is not None]
        report[name] = float(torch.cat(values).norm()) if values else 0.0
    return report


def _load_rank_state(path, rank, identity, ordered_keys, base, optimizer):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("identity") != identity or state.get("ordered_keys") != ordered_keys:
        raise ValueError("O8 resume data/config/identity differs")
    step = int(state.get("step", -1))
    if int(state.get("next_position", -1)) != step * int(identity["global_batch"]):
        raise ValueError("O8 resume next_position mismatch")
    base.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    rng_states = state.get("rng")
    if not isinstance(rng_states, list) or len(rng_states) <= rank:
        raise ValueError("O8 resume does not contain per-rank RNG state")
    return step, rng_states[rank]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "cohort-root", "cache-root", "dual-static-root", "pretrain-target-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--prep-workers", type=int, default=0)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-step", type=int, default=0,
                        help="bounded stop for smoke/resume; final checkpoint is written")
    parser.add_argument("--no-deploy", action="store_true")
    args = parser.parse_args()
    require_tmux()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config.get("schema") != "glt-sci-o8ctrl-arm-b-v1":
        raise ValueError("unexpected O8 control config schema")
    if config.get("use_md200") is not False or config.get("use_geometry") is not False:
        raise ValueError("O8 control cannot use MD200 or geometry")
    if args.prep_workers < 0:
        raise ValueError("--prep-workers must be non-negative")
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    output = Path(args.output).resolve()
    error = [None]
    if rank == 0:
        try:
            if output.exists() and not args.resume:
                raise FileExistsError("new O8 training requires a new output directory")
            output.mkdir(parents=True, exist_ok=bool(args.resume))
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if world > 1:
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(error[0])
    source = O8OnlySource(
        args.cohort_root, args.cache_root, static_root=args.dual_static_root,
        target_root=args.pretrain_target_root,
    )
    try:
        micro = int(config["microbatch"])
        global_batch = int(config["global_batch"])
        if micro <= 0 or global_batch <= 0 or global_batch % (micro * world):
            raise ValueError("global_batch must be divisible by microbatch*world")
        accumulation = global_batch // (micro * world)
        if accumulation <= 0:
            raise ValueError("invalid accumulation")
        if int(config["max_optimizer_steps"]) <= 0 or int(config["save_every"]) <= 0:
            raise ValueError("invalid O8 training budget")
        common = common_initialized_o8_pretrainer(int(config["seed"]))
        common_digest = common.common_init_digest
        base = common.to(device)
        optimizer = torch.optim.AdamW(base.parameters(), lr=float(config["lr"]),
                                      weight_decay=float(config["weight_decay"]))
        module = (DistributedDataParallel(base, device_ids=[device.index], find_unused_parameters=True)
                  if world > 1 else base)
        ordered_keys = [key.hex() for key, _ in source.samples]
        identity = {
            "schema": config["schema"], "config": config, "world_size": world,
            "global_batch": global_batch, "accumulation": accumulation,
            "sample_count": len(source), "cohort_hash": source.cohort["manifest_hash"],
            "main_bundle_hash": source.bundle_hash,
            "static_manifest_hash": source.static_cache.manifest_hash,
            "target_manifest_hash": source.target_cache.manifest_hash if source.target_cache else None,
            "ordered_key_hash": source.cohort["manifest"].get("ordered_sample_key_hash"),
            "common_init_digest": common_digest,
            "architecture": base.encoder.architecture_name,
        }
        start, resume_rng = 0, None
        if args.resume:
            start, resume_rng = _load_rank_state(args.resume, rank, identity, ordered_keys, base, optimizer)
            if start < 0 or start >= int(config["max_optimizer_steps"]):
                raise ValueError("O8 resume step is outside configured budget")
        else:
            # The common initializer makes all ranks start from identical
            # parameters; rank-local dropout/worker RNG streams diverge here.
            set_global_seed(int(config["seed"]) + rank)
        stop = int(config["max_optimizer_steps"])
        if args.stop_after_step:
            if not start < int(args.stop_after_step) <= stop:
                raise ValueError("--stop-after-step must be after start and within budget")
            stop = int(args.stop_after_step)
        if args.resume and any(int(p.stem.rsplit("_", 1)[-1]) > start
                               for p in output.glob("resume_*.pt")):
            raise FileExistsError("resume would overwrite a later O8 checkpoint")
        if rank == 0:
            write_json(output / "run.json", {
                "schema": "glt-sci-o8ctrl-arm-b-run-v1", "command": sys.argv,
                "identity": identity,
                "ordered_key_hash": identity["ordered_key_hash"],
                "common_init_digest": common_digest, "world_size": world,
                "start_step": start, "stop_after_step": stop,
                "prep_workers": int(args.prep_workers), "uses_trimer": False,
                "uses_geometry": False,
            })
        prefetch = None
        if args.prep_workers:
            dataset = O8RankMicrobatchStream(
                source, seed=config["seed"], world=world, rank=rank,
                microbatch=micro, accumulation=accumulation,
                start_step=start, max_steps=stop, ratio=config["atom_mask_ratio"],
            )
            prefetch = iter(torch.utils.data.DataLoader(
                dataset, batch_size=None, num_workers=int(args.prep_workers),
                prefetch_factor=4, persistent_workers=False, pin_memory=False,
            ))
        if resume_rng is not None:
            restore_rng(resume_rng)
        stream = __import__("src.training.glt_dual_runtime", fromlist=["OrderedSampleStream"]).OrderedSampleStream(len(source), config["seed"])
        for step in range(start, stop):
            prepared = []
            if prefetch is not None:
                prepared = [next(prefetch) for _ in range(accumulation)]
            else:
                for offset in range(accumulation):
                    records = []
                    for local in range(micro):
                        position = step * global_batch + offset * world * micro + rank * micro + local
                        index = stream.index_at(position)
                        key = source.samples[index][0]
                        records.append(build_o8_pretrain_sample(
                            source[index], source.static_for(index), source.target_for(index),
                            seed=config["seed"], key=key, position=position,
                            ratio=config["atom_mask_ratio"],
                        ))
                    prepared.append(o8_pretrain_collate(records))
            counts = torch.zeros(2, device=device)
            for data, labels in prepared:
                masked = torch.bincount(
                    data.canonical_graph_index[labels["atom_mask"]],
                    minlength=data.graph_available.numel(),
                )
                valid = data.graph_available.bool()
                counts += torch.tensor([(masked > 0).sum(), valid.sum()], device=device)
            if world > 1:
                dist.all_reduce(counts)
            lr = scheduled_lr(step, **{key: config[key] for key in ("lr", "warmup_steps", "schedule_total_steps", "end_lr")})
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros(2, device=device)
            targets = torch.zeros(2, device=device)
            for offset, (data, labels) in enumerate(prepared):
                sync = module.no_sync() if world > 1 and offset + 1 < accumulation else nullcontext()
                with sync:
                    moved_data = data.to(device, non_blocking=True)
                    moved_labels = {key: value.to(device) if torch.is_tensor(value) else value
                                    for key, value in labels.items()}
                    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                                if device.type == "cuda" and config["amp_dtype"] == "bf16"
                                else nullcontext())
                    with autocast:
                        result = module(moved_data, moved_labels)
                        loss = o8_global_objective(result["sums"], counts, world,
                                                   config["loss_weights"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite O8 pretraining loss")
                    loss.backward()
                totals += result["sums"].detach()
                targets += result["targets"].detach()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(base.parameters(), 1., error_if_nonfinite=True))
            optimizer.step()
            if world > 1:
                dist.all_reduce(totals)
                dist.all_reduce(targets)
            record = {
                "step": step + 1, "rank": rank, "lr": float(lr),
                "losses": (totals / counts.clamp_min(1)).tolist(),
                "valid_graphs": counts.tolist(), "target_counts": targets.tolist(),
                "grad_norm_preclip": grad_norm,
                "fallbacks": int(sum(labels["fallback_count"] for _, labels in prepared)),
                "uses_trimer": False, "uses_geometry": False,
            }
            print(json.dumps(record), flush=True)
            should_save = ((step + 1) % int(config["save_every"]) == 0
                           or step + 1 == stop)
            if should_save:
                states = [None] * world
                if world > 1:
                    dist.all_gather_object(states, rng_state())
                else:
                    states[0] = rng_state()
                if rank == 0:
                    payload = {
                        "schema": "glt-sci-o8ctrl-arm-b-resume-v1",
                        "identity": identity, "ordered_keys": ordered_keys,
                        "step": step + 1, "next_position": (step + 1) * global_batch,
                        "model": base.state_dict(), "optimizer": optimizer.state_dict(),
                        "rng": states, "scheduler": {"step": step + 1, "lr": float(lr)},
                    }
                    save_checkpoint(output / f"resume_{step + 1:05d}.pt", payload)
                    if not args.no_deploy:
                        save_checkpoint(output / f"deploy_{step + 1:05d}.pt",
                                        o8_deployment_package(base, step + 1))
                if world > 1:
                    dist.barrier()
    finally:
        source.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
