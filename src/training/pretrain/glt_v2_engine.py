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
import math


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
        if bool(getattr(model, "use_spatial_contact", False)):
            self.spatial_nce_norm = nn.LayerNorm(512)
            self.spatial_nce_projection = nn.Linear(512, 512)
            self.spatial_nce_gate = nn.Parameter(
                torch.full((512,), math.atanh(0.05))
            )
        else:
            self.spatial_nce_norm = None
            self.spatial_nce_projection = None
            self.register_parameter("spatial_nce_gate", None)
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
        z_3d = output["z_glt"]
        if self.spatial_nce_projection is not None:
            z_3d = z_3d + torch.tanh(self.spatial_nce_gate) * self.spatial_nce_projection(
                self.spatial_nce_norm(output["graph_spatial"])
            )
        contrastive, pool_size = distributed_bidirectional_infonce(
            self.o8_projection(output["z_o8"]),
            self.glt_projection(z_3d),
            data.glt_geometry_valid,
            temperature=float(args.glt_infonce_temperature),
        )
        result = {
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
        if self.spatial_nce_projection is not None:
            spatial_atom = output["atom_spatial_states"]
            o8_atom = output["atom_states"]
            downstream_residual = torch.tanh(
                self.model.spatial_channel_gate
            ) * spatial_atom
            result.update({
                "spatial_residual_norm": float(
                    downstream_residual.detach().float().norm(dim=-1).mean()
                ),
                "spatial_to_o8_ratio": float(
                    downstream_residual.detach().float().norm(dim=-1).mean()
                    / o8_atom.detach().float().norm(dim=-1).mean().clamp_min(1e-8)
                ),
                "spatial_core_norm": float(
                    output["spatial_core_states"].detach().float().norm(dim=-1).mean()
                ),
                "spatial_outer_norm": float(
                    output["spatial_outer_states"].detach().float().norm(dim=-1).mean()
                ),
                "spatial_nce_gate_mean_abs": float(
                    torch.tanh(self.spatial_nce_gate.detach()).abs().mean()
                ),
                "spatial_downstream_gate_mean_abs": float(
                    torch.tanh(
                        self.model.spatial_channel_gate.detach()
                    ).abs().mean()
                ),
            })
        return result


def _checkpoint_payload(
    container, optimizer, scheduler, step, args, world_size,
    *, sampler_epoch=0, batches_in_epoch=0,
):
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
        "glt_metadata_mode": str(getattr(args, "glt_metadata_mode", "full")),
        "use_spatial_contact": bool(getattr(args, "use_spatial_contact", False)),
        "spatial_shell_mode": str(getattr(args, "spatial_shell_mode", "s4")),
        "sampler_epoch": int(sampler_epoch),
        "batches_in_epoch": int(batches_in_epoch),
    }


def _projection(output_dim):
    return nn.Sequential(
        nn.LayerNorm(512),
        nn.Linear(512, 512),
        nn.GELU(),
        nn.Linear(512, int(output_dim)),
    )


def _probe_state_dict(container):
    """Publish trained pretrain namespaces, excluding downstream-only modules."""
    downstream_only_prefixes = (
        "model.atom_fusion_norm.",
        "model.atom_fusion_projection.",
        "model.compact19_residual.",
        "model.o8.md_residual.",
    )
    return {
        key: value
        for key, value in container.state_dict().items()
        if key not in {"model.atom_channel_gate", "model.spatial_channel_gate"}
        and not key.startswith(downstream_only_prefixes)
    }


def run_glt_v2_pretrain(args):
    from src.dataset import UniDataset
    from src.modules import GLTMaskedLineHeadV2, MTSGraphLineModelV2
    from src.modules.periodic_line_glt_v2 import NUM_LINE_LABELS
    from src.training.pretrain.config import dataset_kwargs_from_args
    from src.training.pretrain.engine import _atomic_torch_save, _b0_differentiable_mean
    from src.training.pretrain.glt_v2_objectives import load_line_label_counts
    from src.training.pretrain.metadata_dedup import (
        build_matched_pretrain_containers,
    )
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
        if bool(getattr(args, "use_spatial_contact", False)) and getattr(
            dataset, "_periodic_spatial_contact_sidecar", None
        ) is None:
            raise RuntimeError("MSContact requires the periodic spatial sidecar")
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

        frequencies = load_line_label_counts(
            args.glt_line_label_counts, label_count=NUM_LINE_LABELS
        )
        def make_container(metadata_mode):
            model = MTSGraphLineModelV2(
                glt_layers=int(args.glt_layers),
                glt_attention_variant=str(args.glt_attention_variant),
                glt_metadata_mode=str(metadata_mode),
                use_compact19=False,
                use_spatial_contact=bool(
                    getattr(args, "use_spatial_contact", False)
                ),
                spatial_shell_mode=str(getattr(args, "spatial_shell_mode", "s4")),
            )
            # These branches are downstream-only.  Atom incidence remains
            # trainable because it defines the GLT graph representation used
            # by InfoNCE.
            for module in (
                model.atom_fusion_norm, model.atom_fusion_projection,
                model.compact19_residual, model.o8.md_residual,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad = False
            model.atom_channel_gate.requires_grad = False
            if model.spatial_channel_gate is not None:
                model.spatial_channel_gate.requires_grad = False
            return MTSGLTV2PretrainContainer(
                model,
                nn.Linear(512, int(model.o8.masked_atom_classes)),
                GLTMaskedLineHeadV2(512),
                _projection(args.glt_projection_dim),
                _projection(args.glt_projection_dim),
                frequencies,
            )

        loader_generator_state_before_init = loader_generator.get_state()
        cuda_rng_before_init = [
            state.cpu().clone() for state in torch.cuda.get_rng_state_all()
        ]
        container, reference_container, matched_init_report = (
            build_matched_pretrain_containers(
                make_container,
                str(getattr(args, "glt_metadata_mode", "full")),
                seed=int(args.seed),
            )
        )
        loader_generator_unchanged = torch.equal(
            loader_generator_state_before_init, loader_generator.get_state()
        )
        cuda_rng_unchanged = all(
            torch.equal(before, after.cpu())
            for before, after in zip(
                cuda_rng_before_init, torch.cuda.get_rng_state_all()
            )
        )
        matched_init_report.update({
            "loader_generator_unchanged": bool(loader_generator_unchanged),
            "cuda_rng_unchanged": bool(cuda_rng_unchanged),
            "rank": int(rank),
            "world_size": int(world_size),
            "step": 0,
        })
        del reference_container
        container = container.to(device)
        result_root = Path(args.glt_result_root)
        result_root.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            (result_root / "step0_matched_init.json").write_text(
                json.dumps(matched_init_report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
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
        resume_epoch = 0
        resume_batches_in_epoch = 0
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
            if str(payload.get("glt_metadata_mode", "full")) != str(
                getattr(args, "glt_metadata_mode", "full")
            ):
                raise RuntimeError("MTS-GLT-v2 resume metadata mode mismatch")
            if bool(payload.get("use_spatial_contact", False)) != bool(
                getattr(args, "use_spatial_contact", False)
            ):
                raise RuntimeError("MTS-GLT-v2 resume spatial switch mismatch")
            if str(payload.get("spatial_shell_mode", "s4")) != str(
                getattr(args, "spatial_shell_mode", "s4")
            ):
                raise RuntimeError("MTS-GLT-v2 resume spatial shell mismatch")
            container.load_state_dict(payload["model_state"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            optimizer_step = int(payload["optimizer_steps_completed"])
            resume_epoch = int(payload.get("sampler_epoch", 0))
            resume_batches_in_epoch = int(payload.get("batches_in_epoch", 0))

        metrics_path = result_root / "training_metrics.jsonl"
        if rank == 0 and metrics_path.exists() and not args.resume_state:
            raise RuntimeError(f"refusing to append to existing GLT-v2 trajectory: {metrics_path}")
        epoch = resume_epoch
        batches_in_epoch = 0
        if sampler is not None:
            sampler.set_epoch(epoch)
        iterator = iter(loader)
        for _ in range(resume_batches_in_epoch):
            try:
                next(iterator)
            except StopIteration as error:
                raise RuntimeError(
                    "MTS-GLT-v2 resume batch position exceeds sampler epoch"
                ) from error
        batches_in_epoch = resume_batches_in_epoch

        def next_batch():
            nonlocal iterator, epoch, batches_in_epoch
            try:
                batch = next(iterator)
                batches_in_epoch += 1
                return batch
            except StopIteration:
                epoch += 1
                batches_in_epoch = 0
                if sampler is not None:
                    sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
                batches_in_epoch += 1
                return batch

        run_steps = int(args.glt_stop_after_steps)
        while optimizer_step < run_steps:
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            sums = {"atom": 0.0, "line": 0.0, "infonce": 0.0}
            spatial_sums = {
                "spatial_residual_norm": 0.0, "spatial_to_o8_ratio": 0.0,
                "spatial_core_norm": 0.0, "spatial_outer_norm": 0.0,
                "spatial_nce_gate_mean_abs": 0.0,
                "spatial_downstream_gate_mean_abs": 0.0,
            }
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
                    atom_loss, atom_count, atom_sum = _b0_differentiable_mean(
                        output["atom_sum"], output["atom_count"], device
                    )
                    line_loss, line_count, line_sum = _b0_differentiable_mean(
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
                for name in spatial_sums:
                    spatial_sums[name] += float(output.get(name, 0.0))
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
            if bool(getattr(args, "use_spatial_contact", False)):
                record.update({
                    name: value / int(args.gradient_accumulation_steps)
                    for name, value in spatial_sums.items()
                })
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
                            container, optimizer, scheduler, optimizer_step,
                            args, world_size, sampler_epoch=epoch,
                            batches_in_epoch=batches_in_epoch,
                        ),
                        str(args.save_path) + ".last.pt",
                    )
                if optimizer_step in tuple(args.glt_probe_steps):
                    _atomic_torch_save(
                        {
                            "schema": "mts-glt-v2-probe-v1",
                            "state_dict": _probe_state_dict(container),
                            "step": optimizer_step,
                            "glt_layers": int(args.glt_layers),
                            "glt_attention_variant": str(args.glt_attention_variant),
                            "glt_metadata_mode": str(
                                getattr(args, "glt_metadata_mode", "full")
                            ),
                            "infonce_weight": float(args.glt_infonce_loss_weight),
                            "use_spatial_contact": bool(
                                getattr(args, "use_spatial_contact", False)
                            ),
                            "spatial_shell_mode": str(
                                getattr(args, "spatial_shell_mode", "s4")
                            ),
                            "spatial_sidecar_schema": (
                                "mts-periodic-spatial-contact-v1"
                                if bool(getattr(args, "use_spatial_contact", False))
                                else None
                            ),
                        },
                        result_root / f"mts_glt_v2_probe_{optimizer_step // 1000:03d}k.pth",
                    )
        if distributed:
            dist.barrier()
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


__all__ = ["MTSGLTV2PretrainContainer", "run_glt_v2_pretrain"]
