"""Independent MTS-GLT-v2 pretraining loop."""

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


class MTSGLTV2PretrainContainer(nn.Module):
    def __init__(
        self, model, atom_head, line_head, o8_projection, glt_projection,
        line_label_frequencies,
    ):
        super().__init__()
        self.model = model
        self.atom_head = atom_head
        self.line_head = line_head
        self.o8_projection = o8_projection
        self.glt_projection = glt_projection
        self.register_buffer(
            "line_label_frequencies", line_label_frequencies.long(), persistent=True
        )

    def forward(self, data, args, micro_step, generator=None):
        from src.training.pretrain.engine import (
            _joint_canonical_mask,
            _joint_masked_atom_terms,
        )
        from src.training.pretrain.glt_v2_objectives import (
            distributed_bidirectional_infonce,
            make_masked_line_inputs_v2,
            masked_line_loss,
        )

        atom_mask = _joint_canonical_mask(
            data, int(args.seed), int(micro_step), float(args.graph_mask_ratio)
        )
        line_inputs = make_masked_line_inputs_v2(
            data,
            float(args.glt_line_mask_ratio),
            self.line_label_frequencies,
            generator=generator,
        )
        output = self.model.forward_pretrain(
            data,
            atom_mask=atom_mask,
            line_endpoint_z_a=line_inputs["endpoint_z_a"],
            line_endpoint_z_b=line_inputs["endpoint_z_b"],
            line_observation_distances=line_inputs["observation_distances"],
            line_observation_counts=line_inputs["observation_counts"],
            line_token_shifts=line_inputs["token_shifts"],
        )
        atom_sum, atom_detached, atom_count, atom_correct = _joint_masked_atom_terms(
            data, output["atom_states"], self.atom_head, atom_mask
        )
        line_sum, line_count, line_correct = masked_line_loss(
            output["line_states"], data.glt_token_label,
            line_inputs["selected"], self.line_head,
        )
        contrastive, pool_size = distributed_bidirectional_infonce(
            self.o8_projection(output["z_o8"]),
            self.glt_projection(output["z_glt"]),
            data.glt_geometry_valid,
            temperature=float(args.glt_infonce_temperature),
        )
        return {
            "atom_sum": atom_sum,
            "atom_detached": atom_detached,
            "atom_count": atom_count,
            "atom_correct": atom_correct,
            "line_sum": line_sum,
            "line_count": line_count,
            "line_correct": line_correct,
            "infonce": contrastive,
            "infonce_pool_size": pool_size,
            "valid_graphs": int(data.glt_geometry_valid.sum()),
        }


def _checkpoint_payload(container, optimizer, scheduler, step, args, world_size):
    return {
        "schema": "mts-glt-v2-train-state-v1",
        "model_state": {
            key: value.detach().cpu().clone()
            for key, value in container.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "optimizer_steps_completed": int(step),
        "gradient_accumulation": int(args.gradient_accumulation_steps),
        "world_size": int(world_size),
        "glt_layers": int(args.glt_layers),
        "glt_attention_variant": str(args.glt_attention_variant),
    }


def _projection(output_dim):
    return nn.Sequential(
        nn.LayerNorm(512),
        nn.Linear(512, 512),
        nn.GELU(),
        nn.Linear(512, int(output_dim)),
    )


def run_glt_v2_pretrain(args):
    from src.dataset import UniDataset
    from src.modules import GLTMaskedLineHeadV2, MTSGraphLineModelV2
    from src.modules.periodic_line_glt_v2 import NUM_LINE_LABELS
    from src.training.pretrain.config import dataset_kwargs_from_args
    from src.training.pretrain.engine import _atomic_torch_save, _differentiable_mean
    from src.training.pretrain.glt_v2_objectives import load_line_label_counts
    from src.utils import get_data_loader, set_global_seed

    if str(args.config_schema) != "mts-glt-v2":
        raise ValueError("MTS-GLT-v2 requires its explicit config schema")
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", timeout=timedelta(hours=24),
            device_id=torch.device("cuda", local_rank),
        )
        rank, world_size = dist.get_rank(), dist.get_world_size()
        if world_size != 3:
            raise ValueError("MTS-GLT-v2 execution requires three ranks")
    else:
        local_rank, rank, world_size = 0, 0, 1
    try:
        set_global_seed(int(args.seed))
        if not torch.cuda.is_available():
            raise RuntimeError("MTS-GLT-v2 training requires CUDA")
        device = torch.device("cuda", local_rank)
        dataset = UniDataset(**dataset_kwargs_from_args(args))
        if len(dataset) != 995799:
            raise RuntimeError(f"MTS-GLT-v2 requires full PI1M_v2, got {len(dataset)}")
        if getattr(dataset, "_periodic_line_glt_sidecar", None) is None:
            raise RuntimeError("MTS-GLT-v2 requires the periodic line sidecar")
        effective_batch = (
            int(args.batch_size) * world_size * int(args.gradient_accumulation_steps)
        )
        if effective_batch != int(args.global_batch_size):
            raise ValueError(
                f"optimizer batch mismatch: {effective_batch} != {args.global_batch_size}"
            )
        sampler = None
        if distributed:
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank,
                shuffle=True, seed=int(args.seed), drop_last=True,
            )
        loader_generator = torch.Generator().manual_seed(int(args.seed) + rank)
        loader = get_data_loader(
            dataset, indices=None, batch_size=int(args.batch_size),
            shuffle=sampler is None, drop_last=True,
            num_workers=int(args.loader_workers), pin_memory=True,
            persistent_workers=int(args.loader_workers) > 0,
            sampler=sampler, generator=loader_generator,
            prefetch_factor=int(args.loader_prefetch_factor),
        )

        model = MTSGraphLineModelV2(
            glt_layers=int(args.glt_layers),
            glt_attention_variant=str(args.glt_attention_variant),
            use_compact19=False,
        )
        # These branches are downstream-only.  Atom incidence remains trainable
        # because it defines the GLT graph representation used by InfoNCE.
        for module in (
            model.atom_fusion_norm, model.atom_fusion_projection,
            model.compact19_residual, model.o8.md_residual,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = False
        model.atom_channel_gate.requires_grad = False
        frequencies = load_line_label_counts(
            args.glt_line_label_counts, label_count=NUM_LINE_LABELS
        )
        container = MTSGLTV2PretrainContainer(
            model,
            nn.Linear(512, int(model.o8.masked_atom_classes)),
            GLTMaskedLineHeadV2(512),
            _projection(args.glt_projection_dim),
            _projection(args.glt_projection_dim),
            frequencies,
        ).to(device)
        train_module = container
        if distributed:
            from torch.nn.parallel import DistributedDataParallel
            train_module = DistributedDataParallel(
                container, device_ids=[local_rank], output_device=local_rank,
                broadcast_buffers=True, find_unused_parameters=False,
            )
        parameters = [p for p in container.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            parameters, lr=float(args.lr), betas=(0.9, 0.98), eps=1e-8,
            weight_decay=float(args.weight_decay),
        )
        total_steps = int(args.max_optimizer_steps)
        warmup = int(args.warmup_steps)

        def lr_scale(step):
            if warmup and step < warmup:
                return float(step + 1) / float(warmup)
            progress = min(1.0, max(0.0, (step - warmup) / max(1, total_steps - warmup)))
            floor = float(args.end_lr) / max(float(args.lr), 1e-12)
            return floor + (1.0 - floor) * (1.0 - progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)
        optimizer_step = 0
        if args.resume_state:
            payload = torch.load(args.resume_state, map_location="cpu", weights_only=False)
            if payload.get("schema") != "mts-glt-v2-train-state-v1":
                raise RuntimeError("MTS-GLT-v2 resume schema mismatch")
            if int(payload["world_size"]) != world_size:
                raise RuntimeError("MTS-GLT-v2 resume world-size mismatch")
            if int(payload["gradient_accumulation"]) != int(args.gradient_accumulation_steps):
                raise RuntimeError("MTS-GLT-v2 resume accumulation mismatch")
            if int(payload["glt_layers"]) != int(args.glt_layers):
                raise RuntimeError("MTS-GLT-v2 resume layer mismatch")
            if str(payload["glt_attention_variant"]) != str(args.glt_attention_variant):
                raise RuntimeError("MTS-GLT-v2 resume attention mismatch")
            container.load_state_dict(payload["model_state"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            optimizer_step = int(payload["optimizer_steps_completed"])

        result_root = Path(args.glt_result_root)
        result_root.mkdir(parents=True, exist_ok=True)
        metrics_path = result_root / "training_metrics.jsonl"
        if rank == 0 and metrics_path.exists() and not args.resume_state:
            raise RuntimeError(f"refusing to append to existing GLT-v2 trajectory: {metrics_path}")
        epoch = 0
        if sampler is not None:
            sampler.set_epoch(epoch)
        iterator = iter(loader)

        def next_batch():
            nonlocal iterator, epoch
            try:
                return next(iterator)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                iterator = iter(loader)
                return next(iterator)

        run_steps = int(args.glt_stop_after_steps)
        while optimizer_step < run_steps:
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            sums = {"atom": 0.0, "line": 0.0, "infonce": 0.0}
            counts = {"atom": 0, "line": 0, "pool": 0, "valid": 0}
            for accumulation_index in range(int(args.gradient_accumulation_steps)):
                data = next_batch().to(device, non_blocking=True)
                if len(data.smiles) != int(args.batch_size):
                    raise RuntimeError("InfoNCE requires a fixed local microbatch")
                micro_step = (
                    optimizer_step * int(args.gradient_accumulation_steps)
                    + accumulation_index
                )
                generator = torch.Generator(device=device).manual_seed(
                    int(args.seed) + micro_step * world_size + rank
                )
                amp = torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16,
                    enabled=str(args.amp_dtype) == "bf16",
                )
                sync = (
                    train_module.no_sync()
                    if distributed and accumulation_index + 1 < int(args.gradient_accumulation_steps)
                    else nullcontext()
                )
                with sync, amp:
                    output = train_module(data, args, micro_step, generator=generator)
                    atom_loss, atom_count, atom_sum = _differentiable_mean(
                        output["atom_sum"], output["atom_count"], device
                    )
                    line_loss, line_count, line_sum = _differentiable_mean(
                        output["line_sum"], output["line_count"], device
                    )
                    total = (
                        float(args.glt_atom_loss_weight) * atom_loss
                        + float(args.glt_line_loss_weight) * line_loss
                        + float(args.glt_infonce_loss_weight) * output["infonce"]
                    ) / float(args.gradient_accumulation_steps)
                total.backward()
                sums["atom"] += float(atom_sum)
                sums["line"] += float(line_sum)
                sums["infonce"] += float(output["infonce"].detach())
                counts["atom"] += int(atom_count)
                counts["line"] += int(line_count)
                counts["pool"] += int(output["infonce_pool_size"])
                counts["valid"] += int(output["valid_graphs"])
            if float(args.max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(parameters, float(args.max_grad_norm))
            optimizer.step()
            scheduler.step()
            optimizer_step += 1
            elapsed = time.perf_counter() - started
            record = {
                "step": optimizer_step,
                "masked_atom_loss": sums["atom"] / max(1, counts["atom"]),
                "masked_line_loss": sums["line"] / max(1, counts["line"]),
                "infonce_loss": sums["infonce"] / int(args.gradient_accumulation_steps),
                "infonce_weight": float(args.glt_infonce_loss_weight),
                "optimizer_batch": int(args.global_batch_size),
                "mean_infonce_pool_size": counts["pool"] / int(args.gradient_accumulation_steps),
                "valid_3d_local_graphs": counts["valid"],
                "samples_per_second": int(args.global_batch_size) / max(elapsed, 1e-9),
                "step_seconds": elapsed,
                "lr": optimizer.param_groups[0]["lr"],
                "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
            }
            if rank == 0:
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(
                    f"MTS-GLT-v2 step={optimizer_step}/{run_steps} "
                    f"atom={record['masked_atom_loss']:.5f} "
                    f"line={record['masked_line_loss']:.5f} "
                    f"nce={record['infonce_loss']:.5f} "
                    f"pool={record['mean_infonce_pool_size']:.1f}", flush=True,
                )
                checkpoint_boundary = optimizer_step % int(args.checkpoint_interval_steps) == 0
                if checkpoint_boundary or optimizer_step == run_steps:
                    _atomic_torch_save(
                        _checkpoint_payload(
                            container, optimizer, scheduler, optimizer_step, args, world_size
                        ),
                        str(args.save_path) + ".last.pt",
                    )
                if optimizer_step in tuple(args.glt_probe_steps):
                    _atomic_torch_save(
                        {
                            "schema": "mts-glt-v2-probe-v1",
                            "state_dict": container.state_dict(),
                            "step": optimizer_step,
                            "glt_layers": int(args.glt_layers),
                            "glt_attention_variant": str(args.glt_attention_variant),
                            "infonce_weight": float(args.glt_infonce_loss_weight),
                        },
                        result_root / f"mts_glt_v2_probe_{optimizer_step // 1000:03d}k.pth",
                    )
        if distributed:
            dist.barrier()
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


__all__ = ["MTSGLTV2PretrainContainer", "run_glt_v2_pretrain"]
