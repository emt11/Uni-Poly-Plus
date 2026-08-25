"""Objectives specific to MTS-GLT-v2.

The frozen v1 sidecar remains the source of line labels and observations.
Only masking policy changes here: exact-size inverse-sqrt frequency sampling
and whole-token random replacement.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.modules.periodic_line_glt_v2 import MASK_ATOMIC_NUMBER, MASK_SHIFT_CLASS
from src.training.pretrain.glt_objectives import (
    distributed_bidirectional_infonce,
    masked_line_loss,
)


def load_line_label_counts(path, *, label_count):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    counts = payload.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("line label counts must contain an object named counts")
    result = torch.zeros(int(label_count), dtype=torch.long)
    for key, value in counts.items():
        index = int(key)
        if not 0 <= index < int(label_count):
            raise ValueError(f"line label count index is out of range: {index}")
        if int(value) < 0:
            raise ValueError("line label counts must be non-negative")
        result[index] = int(value)
    return result


def _select_balanced(labels, eligible, ratio, frequencies, generator):
    valid_indices = torch.nonzero(eligible, as_tuple=False).flatten()
    selected = torch.zeros_like(eligible)
    valid_count = int(valid_indices.numel())
    if not valid_count:
        return selected
    sample_count = min(valid_count, max(1, int(round(float(ratio) * valid_count))))
    label_frequency = frequencies.to(labels.device)[labels[valid_indices].long()]
    if bool((label_frequency <= 0).any()):
        missing = labels[valid_indices][label_frequency <= 0].unique().tolist()
        raise ValueError(f"missing positive global frequency for line labels: {missing}")
    weights = label_frequency.float().rsqrt()
    chosen = torch.multinomial(
        weights, sample_count, replacement=False, generator=generator
    )
    selected[valid_indices[chosen]] = True
    return selected


def _different_label_replacements(labels, eligible, targets, generator):
    """Choose real valid tokens with a different label, or return -1."""
    valid_indices = torch.nonzero(eligible, as_tuple=False).flatten()
    output = torch.full_like(targets, -1)
    if not int(valid_indices.numel()):
        return output
    valid_labels = labels[valid_indices]
    for position, target in enumerate(targets.tolist()):
        candidates = valid_indices[valid_labels != labels[target]]
        if int(candidates.numel()):
            draw = torch.randint(
                int(candidates.numel()), (1,), device=labels.device,
                generator=generator,
            )
            output[position] = candidates[draw]
    return output


def make_masked_line_inputs_v2(data, ratio, frequencies, generator=None):
    """Apply exact-count balanced 80/10/10 whole-token corruption."""
    eligible = data.glt_token_valid.bool()
    labels = data.glt_token_label.long()
    selected = _select_balanced(
        labels, eligible, float(ratio), frequencies, generator
    )
    policy = torch.rand(eligible.shape, device=labels.device, generator=generator)
    mask_replace = selected & (policy < 0.8)
    random_replace = selected & (policy >= 0.8) & (policy < 0.9)

    z_a = data.glt_token_endpoint_z_a.clone()
    z_b = data.glt_token_endpoint_z_b.clone()
    distances = data.glt_token_observation_distances.clone()
    counts = data.glt_token_observation_count.clone()
    shifts = data.glt_token_shift.clone()

    def apply_mask(mask):
        z_a[mask] = MASK_ATOMIC_NUMBER
        z_b[mask] = MASK_ATOMIC_NUMBER
        distances[mask] = 0.0
        counts[mask] = 0
        shifts[mask] = MASK_SHIFT_CLASS

    apply_mask(mask_replace)
    random_indices = torch.nonzero(random_replace, as_tuple=False).flatten()
    if int(random_indices.numel()):
        replacements = _different_label_replacements(
            labels, eligible, random_indices, generator
        )
        usable = replacements >= 0
        if bool(usable.any()):
            target = random_indices[usable]
            source = replacements[usable]
            z_a[target] = data.glt_token_endpoint_z_a[source]
            z_b[target] = data.glt_token_endpoint_z_b[source]
            distances[target] = data.glt_token_observation_distances[source]
            counts[target] = data.glt_token_observation_count[source]
            shifts[target] = data.glt_token_shift[source]
        if bool((~usable).any()):
            fallback = torch.zeros_like(selected)
            fallback[random_indices[~usable]] = True
            apply_mask(fallback)

    return {
        "selected": selected,
        "endpoint_z_a": z_a,
        "endpoint_z_b": z_b,
        "observation_distances": distances,
        "observation_counts": counts,
        "token_shifts": shifts,
    }


__all__ = [
    "distributed_bidirectional_infonce",
    "load_line_label_counts",
    "make_masked_line_inputs_v2",
    "masked_line_loss",
]
