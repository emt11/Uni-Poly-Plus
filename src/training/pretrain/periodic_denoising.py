"""Pure B0 periodic coordinate-denoising helpers.

The helpers in this module operate on an already-collated, read-only MTS
batch.  They create transient noisy tensors and never mutate cache or sidecar
fields.  A canonical atom receives one noise vector which is shared by its
three Trimer images.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch


def _state_lookup(data):
    base = data.trimer_base_ru_atom_index.long().reshape(-1)
    offset = data.trimer_ru_offset.long().reshape(-1)
    # The production Trimer contains RU offsets -1/0/+1.  The multiplier keeps
    # every canonical atom's three states disjoint without a Python dictionary.
    code = base * 3 + (offset + 1).clamp(0, 2)
    order = torch.argsort(code)
    return code[order], order


def _lookup_state(data, base, offset, sorted_codes=None, sorted_indices=None):
    if sorted_codes is None or sorted_indices is None:
        sorted_codes, sorted_indices = _state_lookup(data)
    query = base.long() * 3 + (offset.long() + 1).clamp(0, 2)
    if sorted_codes.numel() == 0:
        return torch.full_like(query, -1), torch.zeros_like(query, dtype=torch.bool)
    at = torch.searchsorted(sorted_codes, query)
    valid = at < sorted_codes.numel()
    safe = at.clamp_max(max(0, sorted_codes.numel() - 1))
    valid &= sorted_codes[safe] == query
    result = torch.full_like(query, -1)
    result[valid] = sorted_indices[safe[valid]]
    return result, valid


def _pair_observations(data, positions):
    """Compute noisy observation distances with the audited v2 semantics."""
    src = data.mts_star_v2_pair_key_src.long().reshape(-1)
    dst = data.mts_star_v2_pair_key_dst.long().reshape(-1)
    shift = data.mts_star_v2_pair_key_shift.long().reshape(-1)
    pair_count = int(src.numel())
    if any(int(value.numel()) != pair_count for value in (dst, shift)):
        raise ValueError("B0 Star-RBF pair-key lengths do not match")
    # Sidecar identity is inversion-canonical and can store a negative shift;
    # model geometry is always evaluated in the positive-shift orientation.
    negative = shift < 0
    oriented_src = torch.where(negative, dst, src)
    oriented_dst = torch.where(negative, src, dst)
    oriented_shift = shift.abs()
    sorted_codes, sorted_indices = _state_lookup(data)

    def distance(a, a_shift, b, b_shift):
        left, left_ok = _lookup_state(
            data, a, torch.full_like(a, int(a_shift)), sorted_codes, sorted_indices
        )
        right, right_ok = _lookup_state(
            data, b, torch.full_like(b, int(b_shift)), sorted_codes, sorted_indices
        )
        valid = left_ok & right_ok
        left_safe = left.clamp_min(0)
        right_safe = right.clamp_min(0)
        value = torch.linalg.vector_norm(
            positions[left_safe] - positions[right_safe], dim=-1
        )
        return value, valid & torch.isfinite(value) & (value > 0)

    values = positions.new_zeros((pair_count, 2))
    mask = torch.zeros((pair_count, 2), dtype=torch.bool, device=positions.device)
    for relation_shift, first, second in (
        (0, (0, 0), (0, 0)),
        (1, (-1, 0), (0, 1)),
        (2, (-1, 1), (-1, 1)),
    ):
        selected = torch.nonzero(oriented_shift == relation_shift, as_tuple=False).flatten()
        if not selected.numel():
            continue
        # Shift 2 has one outer-Trimer observation.  It is written once and
        # remains a one-observation RBF input just like the clean sidecar.
        left_shift, right_shift = first
        value, valid = distance(
            oriented_src[selected], left_shift, oriented_dst[selected], right_shift
        )
        values[selected, 0] = value
        mask[selected, 0] = valid
        if relation_shift == 1:
            left_shift, right_shift = second
            value, valid = distance(
                oriented_src[selected], left_shift, oriented_dst[selected], right_shift
            )
            values[selected, 1] = value
            mask[selected, 1] = valid
    return values, mask


def add_canonical_correlated_noise(data, sigma, *, generator=None):
    """Return noisy Trimer positions and dynamic Star-RBF observations."""
    clean = data.trimer_pos.float()
    canonical_count = int(data.mips_x.size(0))
    if generator is None:
        noise = torch.randn(
            (canonical_count, 3), device=clean.device, dtype=clean.dtype
        )
    else:
        noise = torch.randn(
            (canonical_count, 3), generator=generator,
            device=clean.device, dtype=clean.dtype,
        )
    noise = noise * float(sigma)
    base = data.trimer_base_ru_atom_index.long()
    valid_base = (base >= 0) & (base < canonical_count)
    safe_base = base.clamp(0, max(0, canonical_count - 1))
    noisy = clean + noise[safe_base] * valid_base.unsqueeze(-1).to(clean.dtype)
    distances, observation_mask = _pair_observations(data, noisy)
    # Invalid/2-D records must not become a coordinate objective by accident.
    graph_valid = (
        data.trimer_geometry_valid.bool()
        & data.trimer_geometry_is_3d.bool()
        & ~data.trimer_2d_fallback.bool()
    )
    canonical_graph = data.canonical_graph_index.long().reshape(-1)
    pair_src = data.mts_star_v2_pair_key_src.long().reshape(-1)
    safe_pair_src = pair_src.clamp(0, max(0, canonical_graph.numel() - 1))
    pair_graph = canonical_graph[safe_pair_src]
    pair_graph_valid = (pair_src >= 0) & (pair_src < canonical_graph.numel())
    observation_mask &= pair_graph_valid.unsqueeze(-1)
    observation_mask &= graph_valid[pair_graph].unsqueeze(-1)
    return noisy, distances, observation_mask


def kabsch_align_clean_to_noisy(clean, noisy, graph_index, graph_valid=None):
    """Map clean coordinates into each graph's noisy frame without reflection."""
    clean = clean.detach().float()
    noisy = noisy.detach().float()
    graph_index = graph_index.long().reshape(-1)
    output = clean.clone()
    graph_count = int(graph_index.max().item()) + 1 if graph_index.numel() else 0
    valid = (
        torch.ones(graph_count, dtype=torch.bool, device=clean.device)
        if graph_valid is None else torch.as_tensor(
            graph_valid, device=clean.device, dtype=torch.bool
        ).reshape(-1)
    )
    for graph_id in range(graph_count):
        selected = graph_index == graph_id
        if not bool(selected.any()) or graph_id >= valid.numel() or not bool(valid[graph_id]):
            continue
        p, q = clean[selected], noisy[selected]
        if p.size(0) < 3:
            continue
        p_centroid, q_centroid = p.mean(dim=0), q.mean(dim=0)
        autocast_off = (
            torch.autocast(device_type="cuda", enabled=False)
            if clean.is_cuda else nullcontext()
        )
        with autocast_off:
            h = (p.float() - p_centroid.float()).t() @ (
                q.float() - q_centroid.float()
            )
            u, _, vh = torch.linalg.svd(h, full_matrices=False)
            r = u @ vh
            if bool(torch.det(r) < 0):
                u = u.clone()
                u[:, -1] *= -1
                r = u @ vh
            aligned = (p.float() - p_centroid.float()) @ r + q_centroid.float()
        output[selected] = aligned
    return output


__all__ = [
    "add_canonical_correlated_noise",
    "kabsch_align_clean_to_noisy",
]
