"""Masked-line corruption at the GraphGate token self-representation level."""

from __future__ import annotations

import torch

from src.training.pretrain.glt_v2_objectives import (
    _different_label_replacements,
    _select_balanced,
    distributed_bidirectional_infonce,
    load_line_label_counts,
    masked_line_loss,
)


def make_masked_line_states(data, clean_states, mask_embedding, ratio, frequencies, generator=None):
    eligible = (
        data.glt_token_valid.bool()
        & data.glt_query_valid.bool()[data.glt_token_batch.long()]
    )
    labels = data.glt_token_label.long()
    selected = _select_balanced(labels, eligible, float(ratio), frequencies, generator)
    policy = torch.rand(selected.shape, device=selected.device, generator=generator)
    mask_replace = selected & (policy < 0.8)
    random_replace = selected & (policy >= 0.8) & (policy < 0.9)
    corrupted = clean_states.clone()
    corrupted[mask_replace] = mask_embedding.to(corrupted.dtype)
    targets = torch.nonzero(random_replace, as_tuple=False).flatten()
    if int(targets.numel()):
        donors = _different_label_replacements(labels, eligible, targets, generator)
        usable = donors >= 0
        if bool(usable.any()):
            corrupted[targets[usable]] = clean_states[donors[usable]]
        if bool((~usable).any()):
            corrupted[targets[~usable]] = mask_embedding.to(corrupted.dtype)
    return {"selected": selected, "line_states": corrupted,
            "mask_replace": mask_replace, "random_replace": random_replace}


__all__ = [
    "distributed_bidirectional_infonce", "load_line_label_counts",
    "make_masked_line_states", "masked_line_loss",
]
