import os
import sys
import argparse
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import json
import hashlib
import math
import subprocess
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.mips_trimer_contract import (
    ROUTE_INTERNAL as MTS_ROUTE_INTERNAL,
)
from src.training.pretrain.config import (
    dataset_kwargs_from_args,
    parse_arguments,
)


def _gather_rng_states(local_state, distributed, rank, world_size):
    """Collect rank-local stochastic state before a distributed checkpoint."""
    if not distributed:
        return [local_state]
    gathered = [None for _ in range(int(world_size))]
    # All ranks call this function at the same optimizer boundary.  Object
    # gather is intentional: Python/NumPy RNG tuples are not tensors.
    dist.all_gather_object(gathered, local_state)
    return gathered if rank == 0 else None


def _gather_rank_states(local_state, distributed, rank, world_size):
    """Gather a rank-local DataLoader/sampler state at a checkpoint barrier."""
    if not distributed:
        return [local_state]
    gathered = [None for _ in range(int(world_size))]
    dist.all_gather_object(gathered, local_state)
    return gathered if rank == 0 else None


def _atomic_torch_save(payload, path):
    from src.training.common.checkpoint import atomic_torch_save

    atomic_torch_save(payload, path)

def _base_model(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    if isinstance(model, MIPSPretrainContainer):
        return model.model
    return model


def _distributed_enabled():
    return dist.is_available() and dist.is_initialized()


def _distributed_sum_count_mean(local_sum, local_count, device):
    """Convert a local sum into a global sample-weighted mean.

    DDP averages gradients across ranks, so multiplying the local sum by
    ``world_size / global_count`` yields the same gradient as a single global
    mean even when ranks contain different numbers of valid targets.
    """
    count = torch.tensor(float(local_count), device=device, dtype=torch.float32)
    global_count = count.clone()
    world_size = 1
    if _distributed_enabled():
        world_size = dist.get_world_size()
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    if float(global_count.item()) <= 0.0:
        return local_sum * 0.0, 0
    return local_sum * (float(world_size) / global_count), int(global_count.item())


def _b0_differentiable_mean(local_sum, local_count, device):
    """Return a DDP-correct differentiable mean plus detached global stats."""
    stats = torch.stack([
        local_sum.detach().to(device=device, dtype=torch.float64),
        torch.as_tensor(float(local_count), device=device, dtype=torch.float64),
    ])
    if _distributed_enabled():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    else:
        world_size = 1
    global_sum, global_count = stats[0], stats[1]
    if float(global_count.item()) <= 0.0:
        return local_sum * 0.0, 0, 0.0
    return (
        local_sum * (float(world_size) / global_count),
        int(global_count.item()),
        float(global_sum.item()),
    )


def _distributed_sum_int(local_value, device):
    """All-reduce a small integer statistic without synchronizing gradients."""
    value = torch.tensor(int(local_value), device=device, dtype=torch.long)
    if _distributed_enabled():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return int(value.item())


def _distributed_joint_counts(counts, device):
    """Reduce all joint-pretraining counters in one collective."""
    names = (
        "masked_atoms", "angle_graphs", "masked_correct", "angle_correct",
        "angle_targets", "mcl_valid_graphs", "graphs",
    )
    packed = torch.tensor(
        [int(counts[name]) for name in names], device=device, dtype=torch.long
    )
    if _distributed_enabled():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    values = packed.cpu().tolist()
    return dict(zip(names, (int(value) for value in values)))


def _global_mean_from_count(local_sum, global_count):
    """DDP-correct global mean after counts have already been reduced."""
    if int(global_count) <= 0:
        return local_sum * 0.0
    world_size = dist.get_world_size() if _distributed_enabled() else 1
    return local_sum * (float(world_size) / float(global_count))


def _b0_reduce_metrics(payload, device):
    """Reduce B0 diagnostic sums/counts once per optimizer microbatch."""
    detached = payload["detached_terms"]
    relation_counts = detached["valid_relation_by_abs_shift"].to(
        device=device, dtype=torch.float64
    )
    values = torch.cat([
        detached["coordinate_sum"].reshape(1).to(device=device, dtype=torch.float64),
        detached["zero_sum"].reshape(1).to(device=device, dtype=torch.float64),
        detached["displacement_squared_sum"].reshape(1).to(device=device, dtype=torch.float64),
        detached["valid_graph_count"].reshape(1).to(device=device, dtype=torch.float64),
        detached["displacement_element_count"].reshape(1).to(device=device, dtype=torch.float64),
        relation_counts.reshape(3),
    ])
    if _distributed_enabled():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {
        "coordinate_sum": values[0],
        "zero_sum": values[1],
        "displacement_squared_sum": values[2],
        "valid_graph_count": int(values[3].item()),
        "displacement_element_count": int(values[4].item()),
        "valid_relation_by_abs_shift": [int(value.item()) for value in values[5:8]],
    }


def _all_reduce_gradients(modules):
    """Legacy synchronization used only by non-MIPS training routes.

    The non-PBC MIPS Stage 1/2 route is wrapped in real
    ``DistributedDataParallel`` and must never call this helper.
    """
    if not _distributed_enabled():
        return
    world_size = dist.get_world_size()
    for module in modules:
        for parameter in module.parameters():
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)


@torch.no_grad()
def _assert_ddp_parameters_synced(module, step):
    """Full-parameter DDP divergence detector (broadcast-compare, no sampling).

    Every trainable parameter is compared against the rank-0 value bitwise;
    a single divergent parameter aborts the run.
    """
    if not dist.is_available() or not dist.is_initialized():
        return
    inner = (
        module.module
        if isinstance(module, torch.nn.parallel.DistributedDataParallel)
        else module
    )
    divergent = []
    for name, parameter in inner.named_parameters():
        if not parameter.requires_grad:
            continue
        reference = parameter.detach().clone()
        dist.broadcast(reference, src=0)
        diff = (parameter.detach() - reference).abs().max()
        dist.all_reduce(diff, op=dist.ReduceOp.MAX)
        if float(diff.item()) != 0.0:
            divergent.append((name, float(diff.item())))
    if divergent:
        divergent.sort(key=lambda item: -item[1])
        raise RuntimeError(
            f"DDP parameter divergence detected after optimizer step {step}: "
            f"{len(divergent)} parameters; top: {divergent[:10]}"
        )


def _distributed_sum_mapping(values, device):
    """Sum a scalar mapping across ranks, including keys absent on some ranks."""
    if not _distributed_enabled():
        return dict(values)
    gathered_keys = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_keys, sorted(values))
    keys = sorted({key for rank_keys in gathered_keys for key in rank_keys})
    if not keys:
        return {}
    tensor = torch.tensor(
        [float(values.get(key, 0.0)) for key in keys],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {key: float(tensor[idx].item()) for idx, key in enumerate(keys)}


def _joint_canonical_mask(data, seed, stream_step, mask_ratio):
    """Vectorized stateless mask over canonical atoms.

    Keys depend only on sample identity, local canonical atom id, optimizer
    stream step and seed, so rank assignment and batch composition cannot
    change the selected atoms.
    """
    device = data.mips_x.device
    canonical_periodic = bool(
        getattr(data, "mts_canonical_periodic", False)
        or getattr(data, "mips_local_lga_schema_version", 0) == 2
    )
    if canonical_periodic:
        # One mask decision per canonical node.  The Trimer collator lifts
        # these states to all three RU copies downstream.
        canonical = torch.arange(
            int(data.mips_x.size(0)), device=data.mips_x.device,
            dtype=torch.long,
        )
    else:
        canonical = data.canonical_ru_atom_index.long()
    graph_available = torch.as_tensor(
        getattr(data, "graph_available", torch.ones(len(data.smiles))),
        device=device,
    ).bool().flatten()
    required = (
        "canonical_graph_index", "canonical_local_index",
        "canonical_first_node_index", "mts_sample_hash64",
    )
    if not all(hasattr(data, name) for name in required):
        raise ValueError(
            "MTS joint pretraining requires vectorized canonical metadata "
            "from mips_trimer_collate"
        )
    graph_ids = data.canonical_graph_index.long()
    local_ids = data.canonical_local_index.long()
    sample_hash = data.mts_sample_hash64.long()[graph_ids]
    values = sample_hash.clone()
    values ^= local_ids * 6364136223846793005
    values ^= (int(stream_step) * 1442695040888963407) & ((1 << 63) - 1)
    values ^= (int(seed) * 2862933555777941757) & ((1 << 63) - 1)
    values ^= values >> 30
    values *= 3935559000370003845
    values ^= values >> 27
    values *= 2691343689449507681
    values ^= values >> 31
    scores = (values & ((1 << 53) - 1)).to(torch.float64) / float(1 << 53)
    canonical_selected = (scores < float(mask_ratio)) & graph_available[graph_ids]

    selected_counts = torch.zeros(
        graph_available.numel(), dtype=torch.long, device=device
    )
    selected_counts.index_add_(0, graph_ids, canonical_selected.long())
    minimum = torch.full(
        (graph_available.numel(),), float("inf"), dtype=scores.dtype, device=device
    )
    minimum.scatter_reduce_(0, graph_ids, scores, reduce="amin", include_self=True)
    empty = graph_available & (selected_counts == 0)
    canonical_selected |= empty[graph_ids] & (scores == minimum[graph_ids])
    return canonical_selected[canonical]


def _joint_masked_atom_terms(data, node_rep, prediction_head, atom_mask):
    if not hasattr(data, "canonical_first_node_index"):
        raise ValueError("Missing canonical_first_node_index in MTS batch")
    representatives = data.canonical_first_node_index.long()
    # Canonical topology has one node per atom, so representatives are the
    # nodes themselves.  Explicit test-reference batches still use the first
    # copy mapping supplied by the legacy collator.
    if atom_mask.numel() == node_rep.size(0) and representatives.numel() == node_rep.size(0):
        target_indices = torch.nonzero(atom_mask, as_tuple=False).flatten()
    else:
        target_indices = representatives[atom_mask[representatives]]
    if target_indices.numel() == 0:
        zero = node_rep.sum() * 0.0
        return zero, zero.detach(), 0, 0
    targets = data.mips_x[target_indices, :101].argmax(dim=-1).long()
    logits = prediction_head(node_rep[target_indices])
    per_target = F.cross_entropy(logits.float(), targets, reduction="none")
    correct = int((logits.detach().argmax(dim=-1) == targets).sum().item())
    return (
        per_target.sum(), per_target.detach().sum(), int(per_target.numel()), correct
    )


class MIPSPretrainContainer(nn.Module):
    """One DDP-visible module containing the encoder and every Stage 1 head.

    Calling encoder submodules behind DDP bypasses reducer preparation.  This
    container keeps the existing loss helpers while ensuring every trainable
    tensor is reached from one real DDP forward.
    """

    def __init__(self, model, heads=None):
        super().__init__()
        self.model = model
        self.heads = nn.ModuleDict(dict(heads or {}))

    def forward(self, stage, data, args, epoch=0, noise_generator=None):
        if stage != "b0_periodic_coordinate_denoising":
            raise ValueError("B0-v2 is the only active MTS pretraining stage")
        return self._b0_periodic_coordinate_denoising(
            data, args, epoch, noise_generator=noise_generator
        )

    def _b0_periodic_coordinate_denoising(
        self, data, args, stream_step, noise_generator=None
    ):
        """Shared masked-atom + periodic-coordinate B0-v2 forward."""
        from src.training.pretrain.periodic_denoising import (
            add_canonical_correlated_noise,
            kabsch_align_clean_to_noisy,
        )

        if not bool(getattr(args, "b0_coordinate_denoising", True)):
            raise ValueError("B0-v2 requires coordinate denoising")
        mask = _joint_canonical_mask(
            data, args.seed, stream_step, args.graph_mask_ratio
        )
        noisy_pos, noisy_distances, noisy_mask = add_canonical_correlated_noise(
            data, float(args.b0_noise_sigma), generator=noise_generator
        )
        _, node_states, _ = self.model.encoders["graph"].encoder.forward_b0_pretrain(
            data,
            canonical_atom_mask=mask,
            noisy_pair_observation_distances=noisy_distances,
            noisy_pair_observation_mask=noisy_mask,
        )
        atom_sum, atom_detached, atom_count, atom_correct = _joint_masked_atom_terms(
            data, node_states, self.heads["mips_atom"], mask
        )
        decoder = self.heads["coordinate"]
        displacement, relation_valid = decoder(data, node_states, noisy_pos)
        central_index = data.mips_to_trimer_central_index.long()
        # Invalid/padded MIPS rows can carry ``-1`` central indices.  Clamp
        # only for the gather and retain the original validity mask below so
        # those rows never contribute to the objective or metrics.
        safe_noisy_index = central_index.clamp(0, max(0, noisy_pos.size(0) - 1))
        central_noisy_valid = (
            (central_index >= 0) & (central_index < noisy_pos.size(0))
        )
        predicted = noisy_pos[safe_noisy_index] + displacement
        graph_valid = (
            data.graph_available.bool().flatten()
            & data.trimer_geometry_valid.bool().flatten()
            & data.trimer_geometry_is_3d.bool().flatten()
            & ~data.trimer_2d_fallback.bool().flatten()
        )
        target_trimer = kabsch_align_clean_to_noisy(
            data.trimer_pos, noisy_pos, data.trimer_batch, graph_valid
        )
        safe_central_index = central_index.clamp(
            0, max(0, target_trimer.size(0) - 1)
        )
        central_mapping_valid = (
            (central_index >= 0) & (central_index < target_trimer.size(0))
        ) & central_noisy_valid
        target = target_trimer[safe_central_index]
        per_atom = (predicted - target).abs().mean(dim=-1)
        graph_ids = data.canonical_graph_index.long()
        coord_sum = per_atom.new_zeros(())
        zero_sum = per_atom.new_zeros(())
        displacement_sq_sum = per_atom.new_zeros(())
        displacement_element_count = 0
        coord_count = 0
        for graph_id in range(int(graph_valid.numel())):
            selected = (graph_ids == graph_id) & central_mapping_valid
            if bool(selected.any()) and bool(graph_valid[graph_id]):
                coord_sum = coord_sum + per_atom[selected].mean()
                zero_atom = (
                    noisy_pos[safe_central_index[selected]]
                    - target[selected]
                ).abs().mean(dim=-1)
                zero_sum = zero_sum + zero_atom.mean()
                coord_count += 1
                # Decoder outputs one vector per canonical node; only the
                # noisy/target endpoint lookup uses the Trimer central index.
                selected_displacement = displacement[selected]
                displacement_sq_sum = (
                    displacement_sq_sum
                    + selected_displacement.float().square().sum()
                )
                displacement_element_count += int(selected_displacement.numel())
        coord_loss = coord_sum / max(1, coord_count)
        zero_reference = atom_sum * 0.0 + coord_loss * 0.0
        for head_name in ("mips_atom", "coordinate"):
            for parameter in self.heads[head_name].parameters():
                zero_reference = zero_reference + parameter.reshape(-1).sum() * 0.0
        return {
            "loss_terms": {
                "masked_atom_sum": atom_sum,
                "coordinate_sum": coord_sum,
            },
            "detached_terms": {
                "masked_atom_sum": atom_detached,
                "coordinate_sum": coord_sum.detach(),
                "zero_sum": zero_sum.detach(),
                "displacement_squared_sum": displacement_sq_sum.detach(),
                "coordinate_relation_count": torch.as_tensor(
                    int(relation_valid.sum().item()), device=coord_sum.device
                ),
                "displacement_element_count": torch.as_tensor(
                    int(displacement_element_count), device=coord_sum.device
                ),
                "valid_graph_count": torch.as_tensor(
                    int(coord_count), device=coord_sum.device
                ),
                "valid_relation_by_abs_shift": torch.bincount(
                    data.lga_source_image_shift.long()[relation_valid]
                    .abs().clamp_max(2), minlength=3
                ).to(device=coord_sum.device, dtype=torch.long),
            },
            "counts": {
                "masked_atoms": int(atom_count),
                "masked_correct": int(atom_correct),
                "coordinate_graphs": int(coord_count),
                "graphs": int(graph_valid.numel()),
            },
            "zero_reference": zero_reference,
        }

def _build_b0_model(args):
    """Construct the B0 graph-only model with MCL disabled."""
    from src.modules import UniEncoderAttention

    return UniEncoderAttention(
        joint_embedding_dim=args.joint_embedding_dim,
        smiles_model_name=args.smiles_model_name,
        gnn_model_name=args.gnn_model_name,
        modality_list=["graph"],
        freeze_encoder=False,
        graph_num_layers=args.graph_num_layers,
        graph_emb_dim=args.graph_emb_dim,
        graph_dropout=args.graph_dropout,
        graph_encoder_type=args.graph_encoder_type,
        scage_dist_bar=args.scage_dist_bar,
        scage_num_heads=args.scage_num_heads,
        scage_ffn_hidden_dim=args.scage_ffn_hidden_dim,
        scage_num_kernels=args.scage_num_kernels,
        scage_attention_dropout=args.scage_attention_dropout,
        scage_use_pbc_distance=args.scage_use_pbc_distance,
        scage_use_descriptors=args.scage_use_descriptors,
        scage_distance_mode=args.scage_distance_mode,
        scage_distance_rbf=args.scage_distance_rbf,
        scage_distance_cutoff=args.scage_distance_cutoff,
        scage_distance_scales=args.scage_distance_scales,
        scage_distance_taus=args.scage_distance_taus,
        scage_topology_bias=args.scage_topology_bias,
        scage_topology_max_distance=args.scage_topology_max_distance,
        scage_topology_locality_mode=args.scage_topology_locality_mode,
        scage_topology_locality_threshold=args.scage_topology_locality_threshold,
        scage_topology_locality_tau=args.scage_topology_locality_tau,
        scage_periodic_image_mode=args.scage_periodic_image_mode,
        scage_periodic_image_cap=args.scage_periodic_image_cap,
        scage_periodic_image_temperature=args.scage_periodic_image_temperature,
        scage_force_topology_only=args.scage_force_topology_only,
        mips_core=args.mips_core,
        mips_max_hops=(2 if args.mips_max_hops is None else args.mips_max_hops),
        mips_use_descriptors=args.mips_use_descriptors,
        spatial_mode=args.spatial_mode,
        graph_geometry_mode=args.graph_geometry_mode,
        trimer_num_candidates=args.trimer_num_candidates,
        trimer_max_heavy_atoms=args.trimer_max_heavy_atoms,
        mips_variant=args.mips_variant,
        mips_fusion_mode="none",
        projection_mode="plain",
        modality_control="real",
        mips_atom_feature_mode=args.mips_atom_feature_mode,
        mips_attention_scale=args.mips_attention_scale,
        mips_norm_mode=args.mips_norm_mode,
        mips_activation=args.mips_activation,
        mips_spd_bias_mode=args.mips_spd_bias_mode,
        mips_path_bias_mode=args.mips_path_bias_mode,
        mips_multi_scale_hop_gate=args.mips_multi_scale_hop_gate,
        mips_semantics=args.mips_semantics,
        mips_descriptor_fusion_mode=args.mips_descriptor_fusion_mode,
        mips_descriptor_components=args.mips_descriptor_components,
        mips_descriptor_disturbance=args.mips_descriptor_disturbance,
        mips_backbone_mode=args.mips_backbone_mode,
        mips_input_norm=args.mips_input_norm,
        mips_mask_mode=args.mips_mask_mode,
        mips_mask_policy=args.mips_mask_policy,
        mips_masked_loss_reduction=args.mips_masked_loss_reduction,
        use_star_rbf=bool(getattr(args, "b0_use_star_rbf", True)),
        star_rbf_upper=args.star_rbf_upper,
        use_mcl=False,
        topology_attention_variant=args.topology_attention_variant,
        fusion_type="none",
        fp_mode=args.fp_mode,
        alignment_projection_dim=args.alignment_projection_dim,
    )


_B0_V2_TRAIN_STATE_SCHEMA = "mts-b0-v2-train-state-v1"


def _b0_state_payload(
    train_module, optimizer, scheduler, optimizer_steps_completed,
    sampler_epoch, next_batch_index, args, world_size,
):
    """Minimal B0-v2 resume state; stochastic RNG is intentionally absent."""
    module = train_module.module if isinstance(
        train_module, torch.nn.parallel.DistributedDataParallel
    ) else train_module
    model_state = {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }
    return {
        "schema": _B0_V2_TRAIN_STATE_SCHEMA,
        "model_state": model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "optimizer_steps_completed": int(optimizer_steps_completed),
        "sampler_epoch": int(sampler_epoch),
        "next_batch_index": int(next_batch_index),
        "gradient_accumulation": int(args.gradient_accumulation_steps),
        "world_size": int(world_size),
    }


def _run_b0_pretrain(args):
    """Run the single B0-v2 trajectory without publishing a final checkpoint."""
    if args.graph_encoder_type != MTS_ROUTE_INTERNAL:
        raise ValueError("B0 requires graph_encoder_type=mips_trimer_scage")
    if args.topology_attention_variant != "o8":
        raise ValueError("B0-v2 requires topology_attention_variant='o8'")
    if args.checkpoint_interval_steps != 2000:
        raise ValueError("B0 checkpoint interval must remain 2000")
    if str(getattr(args, "config_schema", "")) != "mts-b0-v2":
        raise ValueError("B0-v2 requires config_schema=mts-b0-v2")
    from src.dataset import UniDataset
    from src.modules.periodic_coordinate_decoder import PeriodicCoordinateDecoder
    from src.utils import get_data_loader, set_global_seed

    requested_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    debug_ddp = bool(os.environ.get("B0_DEBUG_DDP"))
    if requested_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", timeout=timedelta(hours=24),
            device_id=torch.device("cuda", local_rank),
        )
        rank, world_size = dist.get_rank(), dist.get_world_size()
        if world_size != 3:
            raise ValueError("B0 DDP requires exactly three ranks on GPUs 1,2,3")
    else:
        local_rank, rank, world_size = 0, 0, 1
    try:
        set_global_seed(args.seed)
        if not torch.cuda.is_available():
            raise RuntimeError("B0 training requires CUDA")
        device = torch.device("cuda", local_rank)
        dataset_kwargs = dataset_kwargs_from_args(args)
        dataset_kwargs["rebuild_feature_cache"] = False
        dataset = UniDataset(**dataset_kwargs)
        if len(dataset) != 995799:
            raise RuntimeError(
                f"B0 requires the full PI1M_v2 cohort (995799), got {len(dataset)}"
            )
        if getattr(dataset, "_star_rbf_v2_sidecar", None) is None and bool(
            args.b0_use_star_rbf
        ):
            raise RuntimeError("B0 requires the frozen Star-RBF v2 sidecar")
        indices = np.arange(len(dataset), dtype=np.int64)
        sampler = None
        sampler_dataset = torch.utils.data.Subset(dataset, indices.tolist())
        if requested_distributed:
            sampler = torch.utils.data.DistributedSampler(
                sampler_dataset,
                num_replicas=world_size, rank=rank, shuffle=True,
                seed=args.seed, drop_last=False,
            )
        effective_global_batch = (
            int(args.batch_size) * int(world_size)
            * int(args.gradient_accumulation_steps)
        )
        if effective_global_batch != int(args.global_batch_size):
            raise ValueError(
                "B0 global batch mismatch: "
                f"batch_size*world_size*accumulation={effective_global_batch}, "
                f"configured={args.global_batch_size}"
            )
        loader_generator = torch.Generator()
        loader_generator.manual_seed(int(args.seed) + 9176 * rank)
        dataloader = get_data_loader(
            dataset, indices=indices, batch_size=args.batch_size, shuffle=True,
            drop_last=False, random_conformer=True,
            num_workers=args.loader_workers, pin_memory=True,
            prefetch_factor=args.loader_prefetch_factor,
            persistent_workers=args.loader_workers > 0, sampler=sampler,
            generator=loader_generator,
        )
        model = _build_b0_model(args)
        graph_encoder = model.encoders["graph"].encoder
        if bool(args.b0_use_star_rbf):
            star_bias = graph_encoder.star_distance_bias
            sidecar_upper = float(dataset._star_rbf_v2_sidecar.rbf_upper)
            expected_upper = 3.75
            if abs(float(args.star_rbf_upper) - expected_upper) > 1e-9:
                raise RuntimeError(
                    f"B0-v2 requires star_rbf_upper={expected_upper}, "
                    f"got config {args.star_rbf_upper}"
                )
            if abs(sidecar_upper - float(args.star_rbf_upper)) > 1e-9:
                raise RuntimeError(
                    f"B0-v2 RBF upper mismatch: sidecar={sidecar_upper}, "
                    f"config={args.star_rbf_upper}"
                )
            centers = star_bias.centers
            if (
                int(centers.numel()) != 32
                or abs(float(centers[0])) > 1e-9
                or abs(float(centers[-1]) - float(args.star_rbf_upper)) > 1e-6
            ):
                raise RuntimeError(
                    "B0-v2 RBF definition mismatch: expected 32 centers in "
                    f"[0.0, {args.star_rbf_upper}], got num={centers.numel()} "
                    f"first={float(centers[0])} last={float(centers[-1])}"
                )
            rbf_spacing = float(centers[1] - centers[0])
            if abs(
                float(star_bias.gamma)
                - 0.5 / max(rbf_spacing * rbf_spacing, 1e-12)
            ) > 1e-4:
                raise RuntimeError(
                    "B0-v2 RBF gamma must be 0.5/spacing^2 from the actual "
                    f"centers, got gamma={float(star_bias.gamma)} "
                    f"spacing={rbf_spacing}"
                )
        graph_dim = int(graph_encoder.emb_dim)
        atom_head = nn.Linear(
            graph_dim, int(graph_encoder.masked_atom_classes)
        )
        heads = {"mips_atom": atom_head}
        if bool(getattr(args, "b0_coordinate_denoising", True)):
            coordinate_decoder = PeriodicCoordinateDecoder(graph_dim)
            heads["coordinate"] = coordinate_decoder
        train_container = MIPSPretrainContainer(model, heads).to(device)
        for parameter in train_container.parameters():
            parameter.requires_grad = False
        trainable_modules = [
            graph_encoder.atom_embedding, graph_encoder.spd_embedding,
            graph_encoder.path_bias, graph_encoder.layers,
            atom_head,
        ]
        if bool(getattr(args, "b0_use_star_rbf", True)):
            trainable_modules.append(graph_encoder.star_distance_bias)
        if bool(getattr(args, "b0_coordinate_denoising", True)):
            trainable_modules.append(coordinate_decoder)
        for module in trainable_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
        train_module = train_container
        if requested_distributed:
            # Parameter registration order must be identical across ranks
            # before DDP wrapping; a mismatch means the model construction is
            # non-deterministic (unordered set/dict iteration) and would make
            # the all-reduce parameter buckets disagree.
            signatures = [
                (name, tuple(parameter.shape), str(parameter.dtype))
                for name, parameter in train_container.named_parameters()
            ]
            gathered_signatures = [None] * world_size
            dist.all_gather_object(gathered_signatures, signatures)
            if any(item != gathered_signatures[0] for item in gathered_signatures[1:]):
                raise RuntimeError(
                    "DDP parameter registration order differs across ranks"
                )
            from torch.nn.parallel import DistributedDataParallel
            train_module = DistributedDataParallel(
                train_container, device_ids=[local_rank],
                output_device=local_rank, broadcast_buffers=True,
                find_unused_parameters=False,
            )
            _assert_ddp_parameters_synced(train_module, step=0)

        optimizer = optim.Adam(
            [p for p in train_container.parameters() if p.requires_grad],
            lr=float(args.lr), betas=(0.9, 0.98), eps=1e-8, weight_decay=0.0,
        )
        total_steps = int(args.max_optimizer_steps)
        warmup = min(max(0, total_steps - 1), int(args.warmup_steps))
        def lr_scale(step):
            if warmup and step < warmup:
                return float(step + 1) / float(warmup)
            progress = min(1.0, max(0.0, (step - warmup) / max(1, total_steps - warmup)))
            return max(float(args.end_lr) / max(float(args.lr), 1e-12), 0.0) + (
                1.0 - max(float(args.end_lr) / max(float(args.lr), 1e-12), 0.0)
            ) * (1.0 - progress)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)
        result_root = Path(args.b0_result_root)
        result_root.mkdir(parents=True, exist_ok=True)
        metrics_path = result_root / "b0_training_metrics.jsonl"
        if rank == 0 and metrics_path.exists() and not args.resume_state:
            raise RuntimeError(
                f"B0-v2 refuses to append to an existing trajectory: {metrics_path}"
            )

        optimizer_steps = 0
        sampler_epoch = 0
        next_batch_index = 0
        if args.resume_state:
            resume_path = Path(args.resume_state).resolve()
            if not resume_path.is_file():
                raise RuntimeError(f"B0-v2 resume state does not exist: {resume_path}")
            payload = torch.load(resume_path, map_location="cpu", weights_only=False)
            if payload.get("schema") != _B0_V2_TRAIN_STATE_SCHEMA:
                raise RuntimeError("B0-v2 resume state schema mismatch")
            if int(payload.get("world_size", -1)) != int(world_size):
                raise RuntimeError("B0-v2 resume world_size mismatch")
            if int(payload.get("gradient_accumulation", -1)) != int(args.gradient_accumulation_steps):
                raise RuntimeError("B0-v2 resume gradient_accumulation mismatch")
            train_container.load_state_dict(payload["model_state"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            optimizer_steps = int(payload["optimizer_steps_completed"])
            sampler_epoch = int(payload["sampler_epoch"])
            next_batch_index = int(payload["next_batch_index"])

        def _epoch_iterator(epoch, skip_batches=0):
            if sampler is not None:
                sampler.set_epoch(int(epoch))
            iterator_value = iter(dataloader)
            for _ in range(int(skip_batches)):
                try:
                    next(iterator_value)
                except StopIteration as exc:
                    raise RuntimeError("B0-v2 resume batch position exceeds sampler epoch") from exc
            return iterator_value

        iterator = _epoch_iterator(sampler_epoch, next_batch_index)

        def _next_batch():
            nonlocal iterator, sampler_epoch, next_batch_index
            try:
                data_value = next(iterator)
            except StopIteration:
                sampler_epoch += 1
                next_batch_index = 0
                iterator = _epoch_iterator(sampler_epoch, 0)
                data_value = next(iterator)
            next_batch_index += 1
            return data_value

        stop_steps = int(getattr(args, "resume_smoke_stop_steps", 0) or 0)
        run_steps = stop_steps if stop_steps else total_steps
        if run_steps < optimizer_steps:
            raise RuntimeError("B0-v2 resume target is behind checkpoint step")
        while optimizer_steps < run_steps:
            step_started_at = time.perf_counter()
            data_wait_seconds = 0.0
            forward_seconds = 0.0
            backward_seconds = 0.0
            rank_wait_seconds = 0.0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            global_micro_stats = {
                "atom_sum": 0.0, "atom_count": 0, "coord_sum": 0.0,
                "coord_count": 0, "zero_sum": 0.0,
                "disp_sq": 0.0, "disp_count": 0,
                "relation_counts": [0, 0, 0],
            }
            last_payload = None
            for accumulation_index in range(int(args.gradient_accumulation_steps)):
                data_started_at = time.perf_counter()
                data = _next_batch().to(device)
                data_wait_seconds += time.perf_counter() - data_started_at
                global_micro_step = (
                    int(optimizer_steps) * int(args.gradient_accumulation_steps)
                    + int(accumulation_index)
                )
                noise_generator = torch.Generator(device=device)
                noise_generator.manual_seed(
                    int(args.seed)
                    + global_micro_step * int(world_size)
                    + int(rank)
                )
                amp_enabled = args.amp_dtype == "bf16" and device.type == "cuda"
                context = torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled
                )
                forward_started_at = time.perf_counter()
                with context:
                    payload = train_module(
                        "b0_periodic_coordinate_denoising", data, args,
                        global_micro_step, noise_generator=noise_generator,
                    )
                    atom_loss, atom_count, atom_sum_global = _b0_differentiable_mean(
                        payload["loss_terms"]["masked_atom_sum"],
                        payload["counts"]["masked_atoms"], device,
                    )
                    coord_loss, coord_count, coord_sum_global = _b0_differentiable_mean(
                        payload["loss_terms"]["coordinate_sum"],
                        payload["counts"]["coordinate_graphs"], device,
                    )
                    loss = (
                        atom_loss
                        + float(args.b0_coordinate_loss_weight) * coord_loss
                        + payload["zero_reference"]
                    ) / float(args.gradient_accumulation_steps)
                forward_seconds += time.perf_counter() - forward_started_at
                backward_started_at = time.perf_counter()
                loss.backward()
                backward_seconds += time.perf_counter() - backward_started_at
                metrics = _b0_reduce_metrics(payload, device)
                global_micro_stats["atom_sum"] += float(atom_sum_global)
                global_micro_stats["atom_count"] += int(atom_count)
                global_micro_stats["coord_sum"] += float(coord_sum_global)
                global_micro_stats["coord_count"] += int(coord_count)
                global_micro_stats["zero_sum"] += float(metrics["zero_sum"].item())
                global_micro_stats["disp_sq"] += float(metrics["displacement_squared_sum"].item())
                global_micro_stats["disp_count"] += int(metrics["displacement_element_count"])
                global_micro_stats["relation_counts"] = [
                    left + right for left, right in zip(
                        global_micro_stats["relation_counts"],
                        metrics["valid_relation_by_abs_shift"],
                    )
                ]
                last_payload = payload
            if float(args.max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in train_container.parameters() if p.requires_grad],
                    float(args.max_grad_norm),
                )
            if requested_distributed and os.environ.get("TOPO_DEBUG_GRAD_CHECK", "0") == "1":
                bad_grads = []
                for name, parameter in train_container.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    if parameter.grad is None:
                        none_flag = torch.tensor([1.0], device=device)
                    else:
                        none_flag = torch.tensor([0.0], device=device)
                    torch.distributed.all_reduce(none_flag, op=torch.distributed.ReduceOp.MAX)
                    if bool(none_flag.item() > 0.5):
                        bad_grads.append((name, float("inf")))
                        continue
                    reference = parameter.grad.detach().clone()
                    torch.distributed.broadcast(reference, src=0)
                    diff = (parameter.grad.detach() - reference).abs().max()
                    torch.distributed.all_reduce(diff, op=torch.distributed.ReduceOp.MAX)
                    if float(diff.item()) != 0.0:
                        bad_grads.append((name, float(diff.item())))
                if rank == 0:
                    bad_grads.sort(key=lambda item: -item[1])
                    print(
                        f"[GRAD-CHECK] before step {optimizer_steps}: "
                        f"divergent grads = {len(bad_grads)} top={bad_grads[:5]}",
                        flush=True,
                    )
            optimizer.step()
            scheduler.step()
            optimizer_steps += 1
            if requested_distributed:
                rank_wait_started_at = time.perf_counter()
                if os.environ.get("TOPO_SKIP_STEP_CHECK", "0") != "1":
                    _assert_ddp_parameters_synced(train_module, optimizer_steps)
                rank_wait_seconds += time.perf_counter() - rank_wait_started_at
                if debug_ddp:
                    print(
                        f"B0-v2 rank={rank} passed parameter sync step={optimizer_steps}",
                        flush=True,
                    )
            coord_mean = (
                global_micro_stats["coord_sum"] / global_micro_stats["coord_count"]
                if global_micro_stats["coord_count"] else 0.0
            )
            atom_mean = (
                global_micro_stats["atom_sum"] / global_micro_stats["atom_count"]
                if global_micro_stats["atom_count"] else 0.0
            )
            zero_mean = (
                global_micro_stats["zero_sum"] / global_micro_stats["coord_count"]
                if global_micro_stats["coord_count"] else 0.0
            )
            r_denoise = coord_mean / max(zero_mean, 1e-8) if zero_mean > 0 else float("nan")
            disp_rms = (
                math.sqrt(global_micro_stats["disp_sq"] / global_micro_stats["disp_count"])
                if global_micro_stats["disp_count"] else 0.0
            )
            record = {
                "step": optimizer_steps,
                "global_micro_step": optimizer_steps * int(args.gradient_accumulation_steps),
                "loss": float((atom_mean + float(args.b0_coordinate_loss_weight) * coord_mean)),
                "masked_atom_loss": float(atom_mean),
                "coordinate_loss": float(coord_mean),
                "zero_displacement_loss": float(zero_mean),
                "r_denoise": float(r_denoise),
                "displacement_rms": float(disp_rms),
                "coordinate_graphs": int(global_micro_stats["coord_count"]),
                "masked_atoms": int(global_micro_stats["atom_count"]),
                "coordinate_relations": int(sum(global_micro_stats["relation_counts"])),
                "valid_relation_by_abs_shift": global_micro_stats["relation_counts"],
                "lr": float(optimizer.param_groups[0]["lr"]),
                "sampler_epoch": int(sampler_epoch),
                "next_batch_index": int(next_batch_index),
                "step_wall_seconds": float(time.perf_counter() - step_started_at),
                "data_wait_seconds": float(data_wait_seconds),
                "forward_seconds": float(forward_seconds),
                "backward_seconds": float(backward_seconds),
                "rank_wait_seconds": float(rank_wait_seconds),
                "samples_per_second": float(
                    int(args.global_batch_size)
                    / max(time.perf_counter() - step_started_at, 1e-9)
                ),
                "optimizer_steps_per_second": float(
                    1.0 / max(time.perf_counter() - step_started_at, 1e-9)
                ),
                "peak_memory_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda" else 0
                ),
            }
            if rank == 0:
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(
                    f"B0-v2 step={optimizer_steps}/{total_steps} "
                    f"loss={record['loss']:.6f} atom={record['masked_atom_loss']:.6f} "
                    f"coord={record['coordinate_loss']:.6f} "
                    f"R={record['r_denoise']:.6f} disp={record['displacement_rms']:.6f}",
                    flush=True,
                )
                checkpoint_boundary = (
                    optimizer_steps % int(args.checkpoint_interval_steps) == 0
                )
                smoke_boundary = bool(args.resume_smoke and optimizer_steps == run_steps)
                if checkpoint_boundary or smoke_boundary:
                    _atomic_torch_save(
                        _b0_state_payload(
                            train_module, optimizer, scheduler, optimizer_steps,
                            sampler_epoch, next_batch_index, args, world_size,
                        ),
                        str(args.save_path) + ".last.pt",
                    )
                if optimizer_steps in tuple(args.b0_probe_steps):
                    probe = result_root / (
                        f"b0_v2_probe_{optimizer_steps // 1000:03d}k.pth"
                    )
                    state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value in train_container.state_dict().items()
                    }
                    train_container.load_state_dict(state_dict, strict=True)
                    _atomic_torch_save(
                        {"state_dict": state_dict, "step": optimizer_steps}, probe
                    )
        if rank == 0:
            print(
                "B0-v2 trajectory complete; no final checkpoint or completion "
                "marker is published before downstream probe selection.", flush=True
            )
        if requested_distributed:
            if debug_ddp:
                print(f"B0-v2 rank={rank} entering final barrier", flush=True)
            dist.barrier()
    finally:
        if requested_distributed and dist.is_initialized():
            dist.destroy_process_group()
    return

def run_pretrain(args=None):
    args = args or parse_arguments()
    if str(getattr(args, "config_schema", "")) == "mts-glt-graphgate-v1":
        from src.training.pretrain.glt_graphgate_engine import run_glt_graphgate_pretrain
        return run_glt_graphgate_pretrain(args)
    if str(getattr(args, "config_schema", "")) == "mts-glt-v2":
        from src.training.pretrain.glt_v2_engine import run_glt_v2_pretrain
        return run_glt_v2_pretrain(args)
    if str(getattr(args, "config_schema", "")) == "mts-glt-v1":
        from src.training.pretrain.glt_engine import run_glt_pretrain
        return run_glt_pretrain(args)
    if str(getattr(args, "config_schema", "")) == "mts-b0-v2":
        return _run_b0_pretrain(args)
    raise ValueError("pretraining requires an explicit active MTS experiment config")


def main():
    return run_pretrain(parse_arguments())


if __name__ == "__main__":
    main()
