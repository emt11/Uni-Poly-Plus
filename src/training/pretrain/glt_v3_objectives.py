"""Masking and loss functions for MTS-GLT-v3-Galformer."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.dataset.periodic_line_glt_image import BOND_TYPE_MASK, STEREO_MASK
from src.modules.periodic_line_glt_v3 import (
    CONJUGATED_MASK,
    ELEMENT_MASK,
    atom_pair_label,
)
from .glt_v2_objectives import distributed_bidirectional_infonce


def select_line_mask(data, ratio=0.4, generator=None):
    selected = torch.zeros_like(data.glt3_token_valid, dtype=torch.bool)
    for graph_id in range(int(data.glt3_geometry_valid.numel())):
        eligible = torch.nonzero(
            data.glt3_token_valid.bool()
            & (data.glt3_token_batch.long() == graph_id), as_tuple=False,
        ).flatten()
        if not int(eligible.numel()):
            continue
        count = min(int(eligible.numel()), max(1, round(float(ratio) * int(eligible.numel()))))
        order = torch.randperm(int(eligible.numel()), device=eligible.device, generator=generator)
        selected[eligible[order[:count]]] = True
    return selected


def make_masked_line_inputs_v3(data, ratio=0.4, generator=None):
    selected = select_line_mask(data, ratio, generator)
    policy = torch.rand(selected.shape, device=selected.device, generator=generator)
    masked = selected & (policy < 0.8)
    replaced = selected & (policy >= 0.8) & (policy < 0.9)
    kept = selected & (policy >= 0.9)
    values = {
        "z_a": data.glt3_token_endpoint_z_a.clone(),
        "z_b": data.glt3_token_endpoint_z_b.clone(),
        "distance": data.glt3_token_distance.clone(),
        "bond_type": data.glt3_token_bond_type.clone(),
        "stereo": data.glt3_token_stereo.clone(),
        "conjugated": data.glt3_token_conjugated.clone(),
        "distance_mask": masked.clone(),
    }
    values["z_a"][masked] = ELEMENT_MASK
    values["z_b"][masked] = ELEMENT_MASK
    values["bond_type"][masked] = BOND_TYPE_MASK
    values["stereo"][masked] = STEREO_MASK
    values["conjugated"][masked] = CONJUGATED_MASK
    targets = torch.nonzero(replaced, as_tuple=False).flatten()
    eligible = torch.nonzero(data.glt3_token_valid.bool(), as_tuple=False).flatten()
    for target in targets.tolist():
        candidates = eligible[eligible != target]
        if not int(candidates.numel()):
            values["z_a"][target] = ELEMENT_MASK
            values["z_b"][target] = ELEMENT_MASK
            values["bond_type"][target] = BOND_TYPE_MASK
            values["stereo"][target] = STEREO_MASK
            values["conjugated"][target] = CONJUGATED_MASK
            values["distance_mask"][target] = True
            replaced[target] = False
            masked[target] = True
            continue
        draw = torch.randint(int(candidates.numel()), (1,), device=eligible.device, generator=generator)
        source = int(candidates[draw].item())
        for name, field in (
            ("z_a", "glt3_token_endpoint_z_a"), ("z_b", "glt3_token_endpoint_z_b"),
            ("distance", "glt3_token_distance"), ("bond_type", "glt3_token_bond_type"),
            ("stereo", "glt3_token_stereo"), ("conjugated", "glt3_token_conjugated"),
        ):
            values[name][target] = getattr(data, field)[source]
    return {"selected": selected, "masked": masked, "replaced": replaced, "kept": kept, **values}


def factorized_line_loss(states, data, selected, heads):
    indices = torch.nonzero(selected, as_tuple=False).flatten()
    if not int(indices.numel()):
        zero = states.sum() * 0.0
        return zero, {name: zero for name in heads}, 0
    targets = {
        "atom_pair": atom_pair_label(data.glt3_token_endpoint_z_a, data.glt3_token_endpoint_z_b),
        "bond_type": data.glt3_token_bond_type.long(),
        "stereo": data.glt3_token_stereo.long(),
        "conjugated": data.glt3_token_conjugated.long(),
    }
    losses = {
        name: F.cross_entropy(head(states[indices]).float(), targets[name][indices], reduction="sum")
        for name, head in heads.items()
    }
    count = int(indices.numel())
    return sum(losses.values()) / (4.0 * count), losses, count


def disturb_md200(values, ratio=0.3, generator=None):
    selected = torch.rand(values.shape, device=values.device, generator=generator) < float(ratio)
    random_values = torch.rand(values.shape, device=values.device, generator=generator)
    return torch.where(selected, random_values.to(values.dtype), values), selected


__all__ = [
    "distributed_bidirectional_infonce", "disturb_md200",
    "factorized_line_loss", "make_masked_line_inputs_v3", "select_line_mask",
]
