"""Compact 19-D descriptors computed read-only from an existing Trimer batch."""

from __future__ import annotations

import math

import torch


def _kabsch_summary(source, target):
    source_center = source.mean(dim=0)
    target_center = target.mean(dim=0)
    left = source - source_center
    right = target - target_center
    covariance = left.T @ right
    u, _, vh = torch.linalg.svd(covariance)
    correction = torch.eye(3, device=source.device, dtype=source.dtype)
    correction[-1, -1] = torch.where(
        torch.det(vh.T @ u.T) < 0,
        source.new_tensor(-1.0), source.new_tensor(1.0),
    )
    rotation = vh.T @ correction @ u.T
    aligned = left @ rotation.T
    rmsd = torch.sqrt((aligned - right).square().sum(dim=-1).mean().clamp_min(0.0))
    trace = torch.trace(rotation)
    angle = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
    translation = torch.linalg.vector_norm(target_center - source_center)
    return translation, angle, rmsd


def _mean_abs_difference(left, right):
    return torch.stack(((left + right) * 0.5, (left - right).abs()))


def _angle(first, second):
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denominator) <= 1e-12:
        return first.new_tensor(0.0)
    cosine = torch.dot(first, second) / denominator
    return torch.acos(cosine.clamp(-1.0, 1.0))


def _shape_descriptors(positions):
    centered = positions - positions.mean(dim=0)
    covariance = centered.T @ centered / max(1, int(positions.size(0)))
    values, vectors = torch.linalg.eigh(covariance)
    values = values.clamp_min(0.0)
    small, middle, large = values
    total = values.sum().clamp_min(1e-12)
    rg = torch.sqrt(total)
    npr1 = small / large.clamp_min(1e-12)
    npr2 = middle / large.clamp_min(1e-12)
    asphericity = 0.5 * (
        (large - middle).square()
        + (large - small).square()
        + (middle - small).square()
    ) / total.square()
    eccentricity = torch.sqrt(
        (1.0 - small.square() / large.square().clamp_min(1e-12)).clamp_min(0.0)
    )
    spherocity = 3.0 * small / total
    normal = vectors[:, 0]
    pbf = (centered @ normal).abs().mean()
    return torch.stack((rg, npr1, npr2, asphericity, eccentricity, spherocity, pbf))


def _backbone_path(data, graph_id):
    nodes = torch.nonzero(
        (data.canonical_graph_index.long() == graph_id)
        & data.mips_backbone_mask.bool(),
        as_tuple=False,
    ).flatten().tolist()
    if len(nodes) < 2:
        raise ValueError("backbone has fewer than two atoms")
    node_set = set(nodes)
    adjacency = {node: set() for node in nodes}
    target, source = data.lga_edge_index.long()
    relation = (
        (data.lga_spd.long() == 1)
        & (data.lga_source_image_shift.long() == 0)
    )
    for left, right in zip(target[relation].tolist(), source[relation].tolist()):
        if left in node_set and right in node_set and left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)

    best_path = []
    for start in nodes:
        queue = [start]
        parent = {start: None}
        for current in queue:
            for neighbor in sorted(adjacency[current]):
                if neighbor not in parent:
                    parent[neighbor] = current
                    queue.append(neighbor)
        for end in queue:
            path = []
            current = end
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            if len(path) > len(best_path):
                best_path = path
    if len(best_path) < 2:
        raise ValueError("backbone subgraph is disconnected")
    return torch.tensor(best_path, device=target.device, dtype=torch.long)


@torch.no_grad()
def compact_trimer_descriptors(data):
    """Return ``(values[G,19], valid[G])`` without modifying the batch."""
    graph_count = int(data.glt_geometry_valid.numel())
    output = data.trimer_pos.new_zeros((graph_count, 19))
    valid = data.glt_geometry_valid.bool().clone()
    required = (
        "trimer_pos", "trimer_batch", "trimer_ru_offset",
        "trimer_base_ru_atom_index", "mips_to_trimer_central_index",
        "mips_backbone_mask", "canonical_graph_index", "lga_edge_index",
        "lga_spd", "lga_source_image_shift",
    )
    if any(not hasattr(data, name) for name in required):
        return output, torch.zeros_like(valid)
    for graph_id in range(graph_count):
        if not bool(valid[graph_id]):
            continue
        try:
            graph_nodes = torch.nonzero(
                data.trimer_batch.long() == graph_id, as_tuple=False
            ).flatten()
            positions = data.trimer_pos[graph_nodes].float()
            offsets = data.trimer_ru_offset[graph_nodes].long()
            base_ids = data.trimer_base_ru_atom_index[graph_nodes].long()
            unit = {}
            for offset in (-1, 0, 1):
                indices = torch.nonzero(offsets == offset, as_tuple=False).flatten()
                order = torch.argsort(base_ids[indices])
                unit[offset] = positions[indices[order]]
            if min(int(unit[offset].size(0)) for offset in (-1, 0, 1)) < 2:
                raise ValueError("incomplete trimer units")
            count = min(int(unit[offset].size(0)) for offset in (-1, 0, 1))
            left_summary = _kabsch_summary(unit[0][:count], unit[-1][:count])
            right_summary = _kabsch_summary(unit[0][:count], unit[1][:count])
            kabsch = torch.cat([
                _mean_abs_difference(left_summary[index], right_summary[index])
                for index in range(3)
            ])

            centroids = {offset: unit[offset].mean(dim=0) for offset in (-1, 0, 1)}
            bend = _angle(centroids[-1] - centroids[0], centroids[1] - centroids[0])

            token_mask = (
                (data.glt_token_batch.long() == graph_id)
                & (data.glt_token_shift.long().abs() == 1)
                & data.glt_token_valid.bool()
            )
            token_indices = torch.nonzero(token_mask, as_tuple=False).flatten()
            link_values = []
            for token in token_indices.tolist():
                observation_count = int(data.glt_token_observation_count[token])
                link_values.extend(
                    data.glt_token_observation_distances[token, :observation_count].float()
                )
            if len(link_values) < 2:
                raise ValueError("periodic connection observations are missing")
            links = torch.stack(link_values)
            link_summary = torch.stack((links.mean(), links.max() - links.min()))

            path = _backbone_path(data, graph_id)
            central_indices = data.mips_to_trimer_central_index[path].long()
            if bool((central_indices < 0).any()):
                raise ValueError("backbone mapping is incomplete")
            backbone_positions = data.trimer_pos[central_indices].float()
            endpoint = torch.linalg.vector_norm(
                backbone_positions[-1] - backbone_positions[0]
            )
            contour = torch.linalg.vector_norm(
                backbone_positions[1:] - backbone_positions[:-1], dim=-1
            ).sum()
            backbone = torch.stack((endpoint, contour, endpoint / contour.clamp_min(1e-8)))
            shape = _shape_descriptors(positions)
            row = torch.cat((kabsch, bend.reshape(1), link_summary, backbone, shape))
            if int(row.numel()) != 19 or not bool(torch.isfinite(row).all()):
                raise ValueError("compact descriptor is non-finite")
            output[graph_id] = row.to(output.dtype)
        except (RuntimeError, ValueError, IndexError):
            valid[graph_id] = False
            output[graph_id].zero_()
    return output, valid


__all__ = ["compact_trimer_descriptors"]
