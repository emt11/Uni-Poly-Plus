"""Joint MTS-GLT-v3 pretraining container and fixed 20k training loop."""

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

from src.dataset.periodic_line_glt_image import NUM_BOND_TYPE_TARGETS, STEREO_MASK
from src.modules.periodic_line_glt_v3 import ATOM_PAIR_TARGETS


def _projection():
    return nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 256))


class MTSGLTV3PretrainContainer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.instantiate_pretraining_glt()
        self.atom_head = nn.Sequential(nn.Linear(512, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 101))
        self.line_heads = nn.ModuleDict({
            "atom_pair": nn.Linear(512, ATOM_PAIR_TARGETS),
            "bond_type": nn.Linear(512, NUM_BOND_TYPE_TARGETS),
            "stereo": nn.Linear(512, STEREO_MASK),
            "conjugated": nn.Linear(512, 2),
        })
        self.o8_projection = _projection()
        self.glt_projection = _projection()

    def forward(self, data, *, stream_step, generator=None):
        from src.training.pretrain.engine import _joint_canonical_mask, _joint_masked_atom_terms
        from src.training.pretrain.glt_v3_objectives import (
            disturb_md200, distributed_bidirectional_infonce,
            factorized_line_loss, make_masked_line_inputs_v3,
        )
        atom_mask = _joint_canonical_mask(data, 42, int(stream_step), 0.30)
        md, md_mask = disturb_md200(data.mips_md, 0.30, generator)
        line = make_masked_line_inputs_v3(data, 0.40, generator)
        output = self.model.forward_joint(
            data, atom_mask=atom_mask, md200=md,
            token_overrides={key: line[key] for key in (
                "z_a", "z_b", "distance", "bond_type", "stereo",
                "conjugated", "distance_mask",
            )},
        )
        atom_sum, _, atom_count, atom_correct = _joint_masked_atom_terms(
            data, output["o8_md_node_states"], self.atom_head, atom_mask
        )
        line_loss, line_parts, line_count = factorized_line_loss(
            output["line_states"], data, line["selected"], self.line_heads
        )
        contrastive, pool_size = distributed_bidirectional_infonce(
            self.o8_projection(output["z_o8"]),
            self.glt_projection(output["graph_geometry"]),
            data.glt3_geometry_valid, temperature=0.1,
        )
        return {
            **output, "atom_sum": atom_sum, "atom_count": atom_count,
            "atom_correct": atom_correct, "line_loss": line_loss,
            "line_sums": line_parts, "line_count": line_count,
            "infonce": contrastive, "infonce_pool_size": pool_size,
            "atom_mask": atom_mask, "line_mask": line["selected"],
            "line_policy": {name: line[name] for name in ("masked", "replaced", "kept")},
            "md_disturbance_fraction": md_mask.float().mean(),
        }


def deployable_bundle(container, step, mode="galformer"):
    if mode not in {"galformer", "mips_concat"}:
        raise ValueError("invalid deployable v3 mode")
    model = container.model
    state = {}
    for prefix, module in (("o8", model.o8), ("md_residual", model.md_residual)):
        state.update({f"{prefix}.{key}": value.detach().cpu() for key, value in module.state_dict().items()})
    if mode == "mips_concat":
        for prefix, module in (
            ("glt", model.glt), ("concat_norm", model.concat_norm),
            ("concat_projection", model.concat_projection),
        ):
            if module is None:
                raise RuntimeError("mips_concat deployment modules are absent")
            state.update({f"{prefix}.{key}": value.detach().cpu() for key, value in module.state_dict().items()})
    return {"schema": "mts-glt-v3-deploy-v1", "mode": mode, "step": int(step), "state_dict": state}


def run_glt_v3_pretrain(args):
    from src.dataset import UniDataset
    from src.dataset.md200_sidecar import DatasetWithMD200
    from src.modules import MTSGraphLineModelV3
    from src.training.pretrain.config import dataset_kwargs_from_args
    from src.training.pretrain.engine import _atomic_torch_save, _differentiable_mean
    from src.utils import get_data_loader, set_global_seed

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=timedelta(hours=24), device_id=torch.device("cuda", local_rank))
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        local_rank, rank, world_size = 0, 0, 1
    try:
        if distributed and world_size != 3:
            raise ValueError("MTS-GLT-v3 formal pretraining requires exactly three ranks")
        if not torch.cuda.is_available():
            raise RuntimeError("MTS-GLT-v3 pretraining requires CUDA")
        set_global_seed(int(args.seed))
        device = torch.device("cuda", local_rank)
        dataset = DatasetWithMD200(UniDataset(**dataset_kwargs_from_args(args)), args.md200_sidecar_root)
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed, drop_last=True) if distributed else None
        loader = get_data_loader(
            dataset, indices=None, batch_size=args.batch_size, shuffle=sampler is None,
            sampler=sampler, drop_last=True, num_workers=args.loader_workers,
            pin_memory=True, persistent_workers=args.loader_workers > 0,
            prefetch_factor=args.loader_prefetch_factor,
            generator=torch.Generator().manual_seed(args.seed + rank),
        )
        container = MTSGLTV3PretrainContainer(
            MTSGraphLineModelV3(glt_readout_mode="mips_concat")
        ).to(device)
        # O8 is fixed to Star-RBF off for this route.  Keep the disabled
        # branch out of both the optimizer and DDP reducer, matching the
        # established GLT-v2 contract.
        for parameter in container.model.o8.star_distance_bias.parameters():
            parameter.requires_grad = False
        for parameter in container.model.o8.md_residual.parameters():
            parameter.requires_grad = False
        for module in (container.model.concat_norm, container.model.concat_projection):
            for parameter in module.parameters():
                parameter.requires_grad = False
        train_module = container
        if distributed:
            train_module = torch.nn.parallel.DistributedDataParallel(container, device_ids=[local_rank], find_unused_parameters=False)
        parameters = [value for value in container.parameters() if value.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.98), weight_decay=0.0)
        warmup, total = int(args.warmup_steps), int(args.max_optimizer_steps)
        floor = float(args.end_lr) / float(args.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (step + 1) / warmup if step < warmup else floor + (1 - floor) * (1 - min(1.0, (step - warmup) / max(1, total - warmup))))
        root = Path(args.glt_result_root)
        metrics = root / "training_metrics.jsonl"
        if rank == 0:
            if root.exists():
                raise FileExistsError(f"refusing existing MTS-GLT-v3 output: {root}")
            root.mkdir(parents=True)
        if distributed:
            dist.barrier()
        epoch, step, iterator = 0, 0, iter(loader)
        while step < int(args.glt_stop_after_steps):
            try:
                data = next(iterator)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                iterator = iter(loader)
                data = next(iterator)
            started = time.perf_counter()
            data = data.to(device)
            generator = torch.Generator(device=device).manual_seed(args.seed + step * world_size + rank)
            optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp_dtype == "bf16")
            with amp:
                output = train_module(data, stream_step=step, generator=generator)
                atom_loss, atom_count, atom_sum = _differentiable_mean(output["atom_sum"], output["atom_count"], device)
                line_losses = []
                line_statistics = {}
                for name, local_sum in output["line_sums"].items():
                    value, count, detached_sum = _differentiable_mean(
                        local_sum, output["line_count"], device
                    )
                    line_losses.append(value)
                    line_statistics[name] = detached_sum / max(1, count)
                line_loss = sum(line_losses) / 4.0
                total_loss = atom_loss + line_loss + output["infonce"]
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step(); scheduler.step(); step += 1
            if rank == 0:
                record = {
                    "step": step, "mask2d": atom_sum / max(1, atom_count),
                    "mask3d": float(line_loss.detach()),
                    "mask3d_parts": line_statistics,
                    "cl": float(output["infonce"].detach()),
                    "line_targets": output["line_count"],
                    "md_disturbance_fraction": float(output["md_disturbance_fraction"]),
                    "lr": optimizer.param_groups[0]["lr"],
                    "seconds": time.perf_counter() - started,
                    "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
                }
                with metrics.open("a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(f"MTS-GLT-v3 step={step}/{args.glt_stop_after_steps} 2d={record['mask2d']:.5f} 3d={record['mask3d']:.5f} cl={record['cl']:.5f}", flush=True)
                if step in args.glt_probe_steps:
                    _atomic_torch_save({
                        "schema": "mts-glt-v3-pretrain-probe-v1",
                        "step": step,
                        "state_dict": {
                            key: value.detach().cpu()
                            for key, value in container.state_dict().items()
                        },
                        "config": {
                            "objectives": ["mask2d", "mask3d", "cl"],
                            "objective_weights": [1.0, 1.0, 1.0],
                            "glt_readout_mode": "galformer",
                            "optimizer": "Adam",
                            "lr": float(args.lr),
                            "warmup_steps": int(args.warmup_steps),
                            "end_lr": float(args.end_lr),
                            "global_batch_size": int(args.global_batch_size),
                        },
                    }, root / f"mts_glt_v3_pretrain_{step // 1000:02d}k.pt")
                    _atomic_torch_save(deployable_bundle(container, step, "galformer"), root / f"mts_glt_v3_galformer_{step // 1000:02d}k.pt")
                    _atomic_torch_save(deployable_bundle(container, step, "mips_concat"), root / f"mts_glt_v3_mips_concat_{step // 1000:02d}k.pt")
        if distributed:
            dist.barrier()
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


__all__ = ["MTSGLTV3PretrainContainer", "deployable_bundle", "run_glt_v3_pretrain"]
