"""Masked-line and distributed contrastive objectives for MTS-GLT-v1."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.modules.periodic_line_glt import MASK_ATOMIC_NUMBER


def make_masked_line_inputs(data, ratio, generator=None):
    """Apply 80/10/10 masked-line corruption without exposing bond type."""
    device = data.glt_token_label.device
    eligible = data.glt_token_valid.bool()
    draws = torch.rand(eligible.shape, device=device, generator=generator)
    selected = eligible & (draws < float(ratio))
    # Ensure a non-empty objective when the batch contains valid geometry.
    if bool(eligible.any()) and not bool(selected.any()):
        selected[torch.nonzero(eligible, as_tuple=False).flatten()[0]] = True

    policy = torch.rand(eligible.shape, device=device, generator=generator)
    mask_replace = selected & (policy < 0.8)
    random_replace = selected & (policy >= 0.8) & (policy < 0.9)
    z_a = data.glt_token_endpoint_z_a.clone()
    z_b = data.glt_token_endpoint_z_b.clone()
    distances = data.glt_token_observation_distances.clone()
    counts = data.glt_token_observation_count.clone()
    z_a[mask_replace] = MASK_ATOMIC_NUMBER
    z_b[mask_replace] = MASK_ATOMIC_NUMBER
    distances[mask_replace] = 0.0
    counts[mask_replace] = 0
    if bool(random_replace.any()):
        number = int(random_replace.sum())
        replacements = torch.randint(
            1, MASK_ATOMIC_NUMBER, (number, 2), device=device, generator=generator
        )
        z_a[random_replace], z_b[random_replace] = replacements[:, 0], replacements[:, 1]
        valid_indices = torch.nonzero(eligible, as_tuple=False).flatten()
        chosen = valid_indices[torch.randint(
            0, int(valid_indices.numel()), (number,), device=device, generator=generator
        )]
        distances[random_replace] = data.glt_token_observation_distances[chosen]
        counts[random_replace] = data.glt_token_observation_count[chosen]
    return {
        "selected": selected,
        "endpoint_z_a": z_a,
        "endpoint_z_b": z_b,
        "observation_distances": distances,
        "observation_counts": counts,
    }


def masked_line_loss(line_states, labels, selected, prediction_head):
    indices = torch.nonzero(selected, as_tuple=False).flatten()
    if not int(indices.numel()):
        zero = line_states.sum() * 0.0
        return zero, 0, 0
    logits = prediction_head(line_states[indices])
    loss = F.cross_entropy(logits.float(), labels[indices].long(), reduction="sum")
    correct = int((logits.detach().argmax(dim=-1) == labels[indices]).sum())
    return loss, int(indices.numel()), correct


def _gather_embeddings_fixed_shape(values):
    if not (dist.is_available() and dist.is_initialized()):
        return values
    from torch.distributed.nn.functional import all_gather
    return torch.cat(tuple(all_gather(values)), dim=0)


@torch.no_grad()
def _gather_valid_fixed_shape(valid):
    if not (dist.is_available() and dist.is_initialized()):
        return valid
    gathered = [torch.empty_like(valid) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, valid)
    return torch.cat(gathered, dim=0)


def distributed_bidirectional_infonce(z_o8, z_glt, valid, temperature=0.1):
    """Gather fixed local batches first, then filter a shared valid mask."""
    if z_o8.shape != z_glt.shape or z_o8.ndim != 2:
        raise ValueError("InfoNCE branch embeddings must have identical [batch, dim] shape")
    valid = valid.bool().reshape(-1)
    if int(valid.numel()) != int(z_o8.size(0)):
        raise ValueError("InfoNCE valid mask must match local batch")
    gathered_o8 = _gather_embeddings_fixed_shape(z_o8)
    gathered_glt = _gather_embeddings_fixed_shape(z_glt)
    gathered_valid = _gather_valid_fixed_shape(valid)
    filtered_o8 = gathered_o8[gathered_valid]
    filtered_glt = gathered_glt[gathered_valid]
    pool_size = int(filtered_o8.size(0))
    if pool_size < 2:
        return (gathered_o8.sum() + gathered_glt.sum()) * 0.0, pool_size
    first = F.normalize(filtered_o8.float(), dim=-1)
    second = F.normalize(filtered_glt.float(), dim=-1)
    logits = first @ second.T / float(temperature)
    labels = torch.arange(pool_size, device=logits.device)
    loss = 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )
    return loss, pool_size


__all__ = [
    "distributed_bidirectional_infonce", "make_masked_line_inputs",
    "masked_line_loss",
]
