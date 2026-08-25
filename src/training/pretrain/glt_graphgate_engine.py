"""Independent GraphGate-v1 pretraining loop and checkpoint namespaces."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch import nn


def _projection(output_dim):
    return nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, int(output_dim)))


class MTSGraphGatePretrainContainer(nn.Module):
    def __init__(self, model, line_label_frequencies, projection_dim=256):
        super().__init__()
        self.o8_encoder = model.o8_encoder
        self.glt_line_encoder = model.glt_line_encoder
        self.masked_atom_head = nn.Linear(512, int(model.o8_encoder.masked_atom_classes))
        from src.modules.periodic_line_glt_graphgate import GLTMaskedLineHeadGraphGate
        self.masked_line_head = GLTMaskedLineHeadGraphGate(512)
        self.o8_contrastive_head = _projection(projection_dim)
        self.glt_contrastive_head = _projection(projection_dim)
        self.register_buffer("line_label_frequencies", line_label_frequencies.long(), persistent=True)

    def forward(self, data, args, micro_step, generator=None):
        from src.training.pretrain.engine import _joint_canonical_mask, _joint_masked_atom_terms
        from src.training.pretrain.glt_graphgate_objectives import (
            distributed_bidirectional_infonce, make_masked_line_states, masked_line_loss,
        )
        atom_mask = _joint_canonical_mask(data, int(args.seed), int(micro_step), float(args.graph_mask_ratio))
        clean = self.glt_line_encoder.clean_line_inputs(
            data, geometry_mode=args.glt_geometry_mode
        )
        corruption = make_masked_line_states(
            data, clean, self.glt_line_encoder.mask_embedding,
            float(args.glt_line_mask_ratio), self.line_label_frequencies,
            generator=generator,
        )
        z_o8, atom_states = self.o8_encoder._forward_impl(
            data, atom_mask=atom_mask, use_star=False, use_md=False
        )
        glt = self.glt_line_encoder(
            data, line_inputs=corruption["line_states"],
            geometry_mode=args.glt_geometry_mode,
        )
        atom_sum, atom_detached, atom_count, atom_correct = _joint_masked_atom_terms(
            data, atom_states, self.masked_atom_head, atom_mask
        )
        line_sum, line_count, line_correct = masked_line_loss(
            glt["line_states"], data.glt_token_label,
            corruption["selected"], self.masked_line_head,
        )
        contrastive, pool_size = distributed_bidirectional_infonce(
            self.o8_contrastive_head(z_o8),
            self.glt_contrastive_head(glt["graph_geometry"]),
            glt["query_valid"],
            temperature=float(args.glt_infonce_temperature),
        )
        return {
            "atom_sum": atom_sum, "atom_detached": atom_detached,
            "atom_count": atom_count, "atom_correct": atom_correct,
            "line_sum": line_sum, "line_count": line_count, "line_correct": line_correct,
            "infonce": contrastive, "infonce_pool_size": pool_size,
            "valid_graphs": int(glt["query_valid"].sum()),
        }


def _trained_state(container):
    return {key: value.detach().cpu().clone() for key, value in container.state_dict().items()}


def _checkpoint_payload(container, optimizer, scheduler, step, args, world_size, epoch, batch_in_epoch, rng_state_by_rank):
    return {
        "schema": "mts-glt-graphgate-v1-train-state-v1",
        "trained_state": _trained_state(container),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "optimizer_steps_completed": int(step), "world_size": int(world_size),
        "gradient_accumulation": int(args.gradient_accumulation_steps),
        "sampler_epoch": int(epoch), "batches_consumed_in_epoch": int(batch_in_epoch),
        "rng_state_by_rank": rng_state_by_rank,
    }


def _probe_payload(container, step):
    full = _trained_state(container)
    full = {
        key: value for key, value in full.items()
        if not key.startswith("o8_encoder.md_residual.")
        and not key.startswith("o8_encoder.star_distance_bias.")
    }
    forbidden = ("fusion_norm", "fusion_projection", "channel_gate", "md_residual", "regression")
    if any(any(fragment in key for fragment in forbidden) for key in full):
        raise RuntimeError("GraphGate pretrain checkpoint contains downstream-only parameters")
    prefixes = {
        "o8_encoder": "o8_encoder.",
        "glt_line_encoder": "glt_line_encoder.",
        "query_pool": "glt_line_encoder.query_pool.",
        "masked_atom_head": "masked_atom_head.",
        "masked_line_head": "masked_line_head.",
        "o8_contrastive_head": "o8_contrastive_head.",
        "glt_contrastive_head": "glt_contrastive_head.",
    }
    namespaces = {}
    for name, prefix in prefixes.items():
        values = {
            key[len(prefix):]: value for key, value in full.items()
            if key.startswith(prefix)
        }
        if name == "glt_line_encoder":
            values = {key: value for key, value in values.items() if not key.startswith("query_pool.")}
        namespaces[name] = values
    return {"schema": "mts-glt-graphgate-v1-probe-v1", "namespaces": namespaces, "step": int(step)}


def run_glt_graphgate_pretrain(args):
    from src.dataset import UniDataset
    from src.modules import MTSGraphGateModel
    from src.modules.periodic_line_glt_graphgate import NUM_LINE_LABELS
    from src.training.pretrain.config import dataset_kwargs_from_args
    from src.training.pretrain.engine import _atomic_torch_save, _b0_differentiable_mean
    from src.training.pretrain.glt_graphgate_objectives import load_line_label_counts
    from src.utils import get_data_loader, set_global_seed

    if str(args.config_schema) != "mts-glt-graphgate-v1":
        raise ValueError("GraphGate pretraining requires its explicit schema")
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = rank = 0
    world_size = 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=24), device_id=torch.device("cuda", local_rank))
        rank, world_size = dist.get_rank(), dist.get_world_size()
        if world_size != 3:
            raise ValueError("GraphGate execution requires three ranks")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("GraphGate pretraining requires CUDA")
        set_global_seed(int(args.seed))
        device = torch.device("cuda", local_rank)
        dataset = UniDataset(**dataset_kwargs_from_args(args))
        if len(dataset) != 995799:
            raise RuntimeError(f"GraphGate requires full PI1M_v2, got {len(dataset)}")
        sidecar = getattr(dataset, "_periodic_line_glt_sidecar", None)
        if sidecar is None or sidecar.metadata.get("schema") != "mts-periodic-line-glt-central-v1":
            raise RuntimeError("GraphGate requires the central-policy sidecar")
        if int(args.batch_size) * world_size * int(args.gradient_accumulation_steps) != int(args.global_batch_size):
            raise ValueError("GraphGate optimizer batch mismatch")
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True,
            seed=int(args.seed), drop_last=True,
        ) if distributed else None
        loader_generator = torch.Generator().manual_seed(int(args.seed) + rank)
        loader = get_data_loader(
            dataset, indices=None, batch_size=int(args.batch_size),
            shuffle=sampler is None, drop_last=True, num_workers=int(args.loader_workers),
            pin_memory=True, persistent_workers=int(args.loader_workers) > 0,
            sampler=sampler, generator=loader_generator,
            prefetch_factor=int(args.loader_prefetch_factor),
        )
        model = MTSGraphGateModel(layers=6)
        for parameter in model.o8_encoder.md_residual.parameters():
            parameter.requires_grad = False
        frequencies = load_line_label_counts(args.glt_line_label_counts, label_count=NUM_LINE_LABELS)
        container = MTSGraphGatePretrainContainer(model, frequencies, args.glt_projection_dim).to(device)
        if args.glt_initial_state:
            initial = torch.load(args.glt_initial_state, map_location="cpu", weights_only=True)
            if initial.get("schema") != "mts-glt-graphgate-v1-shared-step0-v1":
                raise RuntimeError("GraphGate shared step-0 schema mismatch")
            container.load_state_dict(initial["trained_state"], strict=True)
            if rank == 0:
                print(
                    f"GraphGate loaded shared step-0: {args.glt_initial_state} "
                    f"geometry_mode={args.glt_geometry_mode}", flush=True,
                )
        train_module = container
        if distributed:
            from torch.nn.parallel import DistributedDataParallel
            train_module = DistributedDataParallel(
                container, device_ids=[local_rank], output_device=local_rank,
                broadcast_buffers=True, find_unused_parameters=False,
            )
        parameters = [parameter for parameter in container.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=float(args.lr), betas=(0.9, 0.98), eps=1e-8, weight_decay=float(args.weight_decay))
        total_steps, warmup = int(args.max_optimizer_steps), int(args.warmup_steps)

        def lr_scale(step):
            if warmup and step < warmup:
                return float(step + 1) / warmup
            progress = min(1.0, max(0.0, (step - warmup) / max(1, total_steps - warmup)))
            floor = float(args.end_lr) / max(float(args.lr), 1e-12)
            return floor + (1.0 - floor) * (1.0 - progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)
        optimizer_step = 0
        resume_epoch = 0
        resume_batches = 0
        resume_payload = None
        if args.resume_state:
            payload = torch.load(args.resume_state, map_location="cpu", weights_only=False)
            resume_payload = payload
            if payload.get("schema") != "mts-glt-graphgate-v1-train-state-v1":
                raise RuntimeError("GraphGate resume schema mismatch")
            if int(payload["world_size"]) != world_size or int(payload["gradient_accumulation"]) != int(args.gradient_accumulation_steps):
                raise RuntimeError("GraphGate resume execution mismatch")
            container.load_state_dict(payload["trained_state"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            optimizer_step = int(payload["optimizer_steps_completed"])
            resume_epoch = int(payload.get("sampler_epoch", 0))
            resume_batches = int(payload.get("batches_consumed_in_epoch", 0))
            states = payload.get("rng_state_by_rank")
            if not isinstance(states, list) or len(states) != world_size:
                raise RuntimeError("GraphGate resume RNG states are missing")
            local_state = states[rank]
            torch.set_rng_state(local_state["cpu"])
            torch.cuda.set_rng_state(local_state["cuda"], device=device)
            loader_generator.set_state(local_state["loader_generator"])
        result_root = Path(args.glt_result_root)
        result_root.mkdir(parents=True, exist_ok=True)
        metrics_path = result_root / "training_metrics.jsonl"
        if rank == 0 and metrics_path.exists() and not args.resume_state:
            raise RuntimeError(f"refusing to append to existing GraphGate trajectory: {metrics_path}")
        epoch = resume_epoch
        batch_in_epoch = 0
        if sampler is not None: sampler.set_epoch(epoch)
        iterator = iter(loader)
        while batch_in_epoch < resume_batches:
            try:
                next(iterator)
            except StopIteration as exc:
                raise RuntimeError("GraphGate resume sampler position is out of range") from exc
            batch_in_epoch += 1

        def next_batch():
            nonlocal iterator, epoch, batch_in_epoch
            try:
                batch = next(iterator)
                batch_in_epoch += 1
                return batch
            except StopIteration:
                epoch += 1
                batch_in_epoch = 0
                if sampler is not None: sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
                batch_in_epoch = 1
                return batch

        run_steps = int(args.glt_stop_after_steps)
        while optimizer_step < run_steps:
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            sums = {"atom": 0.0, "line": 0.0, "infonce": 0.0}
            counts = {"atom": 0, "line": 0, "pool": 0, "valid": 0}
            for accumulation_index in range(int(args.gradient_accumulation_steps)):
                data = next_batch().to(device, non_blocking=True)
                if len(data.smiles) != int(args.batch_size):
                    raise RuntimeError("InfoNCE requires fixed local microbatch")
                micro_step = optimizer_step * int(args.gradient_accumulation_steps) + accumulation_index
                generator = torch.Generator(device=device).manual_seed(int(args.seed) + micro_step * world_size + rank)
                amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=str(args.amp_dtype) == "bf16")
                sync = train_module.no_sync() if distributed and accumulation_index + 1 < int(args.gradient_accumulation_steps) else nullcontext()
                with sync, amp:
                    output = train_module(data, args, micro_step, generator=generator)
                    atom_loss, atom_count, atom_sum = _b0_differentiable_mean(output["atom_sum"], output["atom_count"], device)
                    line_loss, line_count, line_sum = _b0_differentiable_mean(output["line_sum"], output["line_count"], device)
                    total = (float(args.glt_atom_loss_weight) * atom_loss + float(args.glt_line_loss_weight) * line_loss + float(args.glt_infonce_loss_weight) * output["infonce"]) / int(args.gradient_accumulation_steps)
                if not bool(torch.isfinite(total)):
                    raise FloatingPointError(f"non-finite GraphGate loss at micro_step={micro_step}")
                total.backward()
                sums["atom"] += float(atom_sum); sums["line"] += float(line_sum); sums["infonce"] += float(output["infonce"].detach())
                counts["atom"] += int(atom_count); counts["line"] += int(line_count); counts["pool"] += int(output["infonce_pool_size"]); counts["valid"] += int(output["valid_graphs"])
            if optimizer_step == 0:
                glt_gradients = [
                    parameter.grad for parameter in container.glt_line_encoder.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                if not glt_gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in glt_gradients):
                    raise FloatingPointError("GraphGate GLT gradients are missing or non-finite")
                if not any(bool((gradient != 0).any()) for gradient in glt_gradients):
                    raise RuntimeError("GraphGate GLT gradients are all zero")
            if float(args.max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(parameters, float(args.max_grad_norm))
            optimizer.step(); scheduler.step(); optimizer_step += 1
            record = {
                "step": optimizer_step,
                "masked_atom_loss": sums["atom"] / max(1, counts["atom"]),
                "masked_line_loss": sums["line"] / max(1, counts["line"]),
                "infonce_loss": sums["infonce"] / int(args.gradient_accumulation_steps),
                "optimizer_batch": int(args.global_batch_size),
                "mean_infonce_pool_size": counts["pool"] / int(args.gradient_accumulation_steps),
                "valid_3d_local_graphs": counts["valid"],
                "samples_per_second": int(args.global_batch_size) / max(time.perf_counter() - started, 1e-9),
                "lr": optimizer.param_groups[0]["lr"],
                "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
            }
            checkpoint_due = (
                optimizer_step % int(args.checkpoint_interval_steps) == 0
                or optimizer_step == run_steps
            )
            rng_state_by_rank = None
            if checkpoint_due:
                local_rng = {
                    "cpu": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state(device),
                    "loader_generator": loader_generator.get_state(),
                }
                if distributed:
                    rng_state_by_rank = [None for _ in range(world_size)]
                    dist.all_gather_object(rng_state_by_rank, local_rng)
                else:
                    rng_state_by_rank = [local_rng]
            if rank == 0:
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(f"GraphGate step={optimizer_step}/{run_steps} atom={record['masked_atom_loss']:.5f} line={record['masked_line_loss']:.5f} nce={record['infonce_loss']:.5f} pool={record['mean_infonce_pool_size']:.1f}", flush=True)
                if checkpoint_due:
                    _atomic_torch_save(_checkpoint_payload(container, optimizer, scheduler, optimizer_step, args, world_size, epoch, batch_in_epoch, rng_state_by_rank), str(args.save_path) + ".last.pt")
                if optimizer_step in tuple(args.glt_probe_steps):
                    _atomic_torch_save(_probe_payload(container, optimizer_step), result_root / f"mts_glt_graphgate_probe_{optimizer_step // 1000:03d}k.pth")
        if rank == 0:
            _atomic_torch_save(_probe_payload(container, optimizer_step), args.save_path)
        if distributed: dist.barrier()
    finally:
        if distributed and dist.is_initialized(): dist.destroy_process_group()


__all__ = ["MTSGraphGatePretrainContainer", "run_glt_graphgate_pretrain"]
