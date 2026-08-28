"""Baseline MTS-GLT-v2 pretraining entry point and shared loss helpers.

The baseline keeps one explicit route: masked atom prediction, masked line
prediction, and bidirectional O8/GLT InfoNCE.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.training.common.checkpoint import atomic_torch_save
from src.training.pretrain.config import parse_arguments


def _atomic_torch_save(payload, path):
    """Write a checkpoint through the shared atomic checkpoint helper."""
    atomic_torch_save(payload, path)


def _distributed_enabled():
    return dist.is_available() and dist.is_initialized()


def _differentiable_mean(local_sum, local_count, device):
    """Return a DDP-correct differentiable mean and detached global statistics.

    It is used by the MTS-GLT-v2 objective loop for local/global statistics.
    """
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


def _joint_canonical_mask(data, seed, stream_step, mask_ratio):
    """Build a stateless, identity-stable mask over canonical atoms."""
    device = data.mips_x.device
    canonical_periodic = bool(
        getattr(data, "mts_canonical_periodic", False)
        or getattr(data, "mips_local_lga_schema_version", 0) == 2
    )
    if canonical_periodic:
        canonical = torch.arange(
            int(data.mips_x.size(0)), device=device, dtype=torch.long
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
            "MTS joint pretraining requires canonical metadata from "
            "mips_trimer_collate"
        )
    graph_ids = data.canonical_graph_index.long()
    local_ids = data.canonical_local_index.long()
    sample_hash = data.mts_sample_hash64.long()[graph_ids]
    values = sample_hash.clone()
    values ^= local_ids * 6364136223846793005
    values ^= (int(stream_step) * 1442690888960403407) & ((1 << 63) - 1)
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
    """Return summed masked-atom loss and detached counters."""
    if not hasattr(data, "canonical_first_node_index"):
        raise ValueError("Missing canonical_first_node_index in MTS batch")
    representatives = data.canonical_first_node_index.long()
    if (
        atom_mask.numel() == node_rep.size(0)
        and representatives.numel() == node_rep.size(0)
    ):
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
        per_target.sum(),
        per_target.detach().sum(),
        int(per_target.numel()),
        correct,
    )


def run_pretrain(args=None):
    """Dispatch only the retained MTS-GLT-v2 baseline pretraining route."""
    args = parse_arguments() if args is None else args
    if str(getattr(args, "config_schema", "")) != "mts-glt-v2":
        raise ValueError(
            "pretraining requires the explicit mts-glt-v2 baseline config"
        )
    from src.training.pretrain.glt_v2_engine import run_glt_v2_pretrain
    return run_glt_v2_pretrain(args)


def main():
    return run_pretrain(parse_arguments())


if __name__ == "__main__":
    main()
