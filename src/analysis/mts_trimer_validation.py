"""Read-only geometry and representation helpers for Trimer validation."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch

from src.dataset.mts_star_rbf_v2 import prepare_topology, prepare_trimer


VARIANTS = (
    "full",
    "distance_only",
    "angle_only",
    "atom_line_only",
    "left_only",
    "right_only",
)


def distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not values.size:
        return {
            "count": 0, "mean": None, "std": None, "p1": None,
            "p10": None, "p50": None, "p90": None, "p99": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "p1": float(np.quantile(values, 0.01)),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def kabsch_summary(source: torch.Tensor, target: torch.Tensor):
    """Return translation, rotation angle and RMSD after proper Kabsch fit."""
    source = torch.as_tensor(source, dtype=torch.float64)
    target = torch.as_tensor(target, dtype=torch.float64)
    if source.shape != target.shape or source.ndim != 2 or source.size(1) != 3:
        raise ValueError("Kabsch inputs must have matching [N,3] shapes")
    if source.size(0) < 1 or not bool(torch.isfinite(source).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("Kabsch inputs are incomplete or non-finite")
    source_center = source.mean(dim=0)
    target_center = target.mean(dim=0)
    if source.size(0) == 1:
        return {
            "translation": float(torch.linalg.vector_norm(target_center - source_center)),
            "rotation_rad": 0.0,
            "rmsd": 0.0,
        }
    left = source - source_center
    right = target - target_center
    u, _, vh = torch.linalg.svd(left.T @ right)
    correction = torch.eye(3, dtype=source.dtype, device=source.device)
    if float(torch.det(vh.T @ u.T)) < 0:
        correction[-1, -1] = -1.0
    rotation = vh.T @ correction @ u.T
    aligned = left @ rotation.T
    rmsd = torch.sqrt((aligned - right).square().sum(dim=-1).mean().clamp_min(0.0))
    rotation_angle = torch.acos(((torch.trace(rotation) - 1.0) * 0.5).clamp(-1.0, 1.0))
    translation = torch.linalg.vector_norm(target_center - source_center)
    return {
        "translation": float(translation),
        "rotation_rad": float(rotation_angle),
        "rmsd": float(rmsd),
    }


def _angle(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
        raise ValueError("invalid angle vectors")
    value = torch.acos(torch.clamp(torch.dot(first, second) / denominator, -1.0, 1.0))
    if not bool(torch.isfinite(value)):
        raise ValueError("non-finite angle")
    return float(value)


def _mean_max(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return float("nan"), float("nan")
    return float(values.mean()), float(values.max())


def trimer_sample_metrics(topology, trimer, sidecar_row=None):
    """Compute scalar repeat-consistency metrics for one frozen Trimer."""
    prepared, topology_reason = prepare_topology(topology)
    geometry, geometry_reason = prepare_trimer(trimer, topology)
    result = {
        "geometry_valid": False,
        "failure_reason": topology_reason or geometry_reason or "",
    }
    if prepared is None or geometry is None:
        return result

    mapping = torch.as_tensor(topology.canonical_to_trimer_base_atom_id).long()
    positions = geometry["positions"].double()
    state = geometry["state_to_local"]

    def point(atom, shift):
        index = state.get((int(mapping[int(atom)]), int(shift)))
        if index is None:
            raise ValueError("missing repeat-unit atom state")
        return positions[int(index)]

    base_ids = sorted({base for base, shift in state if int(shift) == 0})
    units = {
        shift: torch.stack([positions[state[(base, shift)]] for base in base_ids])
        for shift in (-1, 0, 1)
    }
    center = units[0]
    radius = torch.sqrt(
        (center - center.mean(dim=0)).square().sum(dim=-1).mean().clamp_min(1e-16)
    )
    left_fit = kabsch_summary(center, units[-1])
    right_fit = kabsch_summary(center, units[1])

    bond_abs, bond_rel = [], []
    adjacency = {atom: set() for atom in range(int(prepared["z"].numel()))}
    for first, second in prepared["internal_edges"]:
        distances = [float(torch.linalg.vector_norm(point(first, q) - point(second, q))) for q in (-1, 0, 1)]
        for outer in (distances[0], distances[2]):
            difference = abs(outer - distances[1])
            bond_abs.append(difference)
            bond_rel.append(difference / (0.5 * (outer + distances[1]) + 1e-8))
        adjacency[first].add(second)
        adjacency[second].add(first)

    angle_abs = []
    for center_atom, neighbors in adjacency.items():
        ordered = sorted(neighbors)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                observed = []
                for q in (-1, 0, 1):
                    observed.append(_angle(
                        point(ordered[i], q) - point(center_atom, q),
                        point(ordered[j], q) - point(center_atom, q),
                    ))
                angle_abs.extend((abs(observed[0] - observed[1]), abs(observed[2] - observed[1])))

    left_boundary = int(prepared["right"])
    right_boundary = int(prepared["left"])
    d_left = float(torch.linalg.vector_norm(point(left_boundary, -1) - point(right_boundary, 0)))
    d_right = float(torch.linalg.vector_norm(point(left_boundary, 0) - point(right_boundary, 1)))
    boundary_abs = abs(d_left - d_right)
    boundary_rel = boundary_abs / (0.5 * (d_left + d_right) + 1e-8)

    centroids = {shift: values.mean(dim=0) for shift, values in units.items()}
    bend = _angle(centroids[-1] - centroids[0], centroids[1] - centroids[0])
    bond_abs_mean, bond_abs_max = _mean_max(bond_abs)
    bond_rel_mean, bond_rel_max = _mean_max(bond_rel)
    angle_mean, angle_max = _mean_max(angle_abs)

    energy = float(getattr(trimer, "trimer_conformer_energy", float("nan")))
    result.update({
        "geometry_valid": True,
        "failure_reason": "",
        "ru_atoms": int(len(base_ids)),
        "token_count": int(len(prepared["internal_edges"]) + 1),
        "relation_count": int(
            len(sidecar_row["relations"]["relation_source"])
            if sidecar_row is not None else 0
        ),
        "boundary_distance_left": d_left,
        "boundary_distance_right": d_right,
        "boundary_distance_abs_asymmetry": boundary_abs,
        "boundary_distance_relative_asymmetry": boundary_rel,
        "internal_bond_outer_abs_mean": bond_abs_mean,
        "internal_bond_outer_abs_max": bond_abs_max,
        "internal_bond_outer_relative_mean": bond_rel_mean,
        "internal_bond_outer_relative_max": bond_rel_max,
        "span0_angle_outer_abs_rad_mean": angle_mean,
        "span0_angle_outer_abs_rad_max": angle_max,
        "span0_angle_outer_abs_deg_mean": math.degrees(angle_mean),
        "span0_angle_outer_abs_deg_max": math.degrees(angle_max),
        "kabsch_left_translation": left_fit["translation"],
        "kabsch_right_translation": right_fit["translation"],
        "kabsch_translation_abs_difference": abs(left_fit["translation"] - right_fit["translation"]),
        "kabsch_left_rotation_rad": left_fit["rotation_rad"],
        "kabsch_right_rotation_rad": right_fit["rotation_rad"],
        "kabsch_rotation_abs_difference_rad": abs(left_fit["rotation_rad"] - right_fit["rotation_rad"]),
        "kabsch_left_rmsd": left_fit["rmsd"],
        "kabsch_right_rmsd": right_fit["rmsd"],
        "kabsch_rmsd_mean": 0.5 * (left_fit["rmsd"] + right_fit["rmsd"]),
        "kabsch_rmsd_max": max(left_fit["rmsd"], right_fit["rmsd"]),
        "kabsch_rmsd_over_rg_mean": 0.5 * (left_fit["rmsd"] + right_fit["rmsd"]) / float(radius),
        "kabsch_rmsd_over_rg_max": max(left_fit["rmsd"], right_fit["rmsd"]) / float(radius),
        "centroid_bend_rad": bend,
        "centroid_bend_deg": math.degrees(bend),
        "mmff_energy": energy,
        "mmff_energy_per_heavy_atom": energy / max(1, int(positions.size(0))),
    })

    if sidecar_row is not None:
        relations = sidecar_row["relations"]
        span = np.asarray(relations["relation_span"])
        fallback = np.asarray(relations["relation_is_fallback"], dtype=bool)
        valid = np.asarray(relations["relation_observation_valid"], dtype=bool)
        angles = np.asarray(relations["relation_observation_angles"], dtype=np.float64)
        selected = (span == 1) & ~fallback & valid[:, 0] & valid[:, 1]
        differences = np.abs(angles[selected, 0] - angles[selected, 1])
        mean, maximum = _mean_max(differences)
        result.update({
            "span1_angle_abs_rad_mean": mean,
            "span1_angle_abs_rad_max": maximum,
            "span1_angle_abs_deg_mean": math.degrees(mean),
            "span1_angle_abs_deg_max": math.degrees(maximum),
        })
    return result


_GLT_FIELDS = (
    "glt_token_atom_a", "glt_token_endpoint_z_a", "glt_token_endpoint_z_b",
    "glt_token_observation_distances", "glt_token_observation_valid",
    "glt_token_observation_translation", "glt_token_shift",
    "glt_token_batch", "glt_token_valid", "glt_relation_source",
    "glt_relation_target", "glt_relation_observation_angles",
    "glt_relation_observation_valid", "glt_relation_observation_translation",
    "glt_relation_span", "glt_relation_valid", "glt_relation_is_fallback",
    "glt_query_valid",
)


def graphgate_variant_view(data, variant: str):
    """Return a lightweight GLT view with only observation masks changed."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown GraphGate geometry variant: {variant}")
    values = {name: getattr(data, name) for name in _GLT_FIELDS}
    token_mask = values["glt_token_observation_valid"].clone()
    relation_mask = values["glt_relation_observation_valid"].clone()

    if variant in {"angle_only", "atom_line_only"}:
        token_mask.zero_()
    if variant in {"distance_only", "atom_line_only"}:
        relation_mask.zero_()
    if variant in {"left_only", "right_only"}:
        side_translation = -1 if variant == "left_only" else 0
        token_translation = values["glt_token_observation_translation"]
        token_shift = values["glt_token_shift"].abs().unsqueeze(-1)
        keep_token = (token_shift == 0) & (token_translation == 0)
        keep_token |= (token_shift == 1) & (token_translation == side_translation)
        token_mask &= keep_token

        relation_translation = values["glt_relation_observation_translation"]
        span = values["glt_relation_span"].unsqueeze(-1)
        keep_relation = ((span == 0) | (span == 2)) & (relation_translation == 0)
        keep_relation |= (span == 1) & (relation_translation == side_translation)
        relation_mask &= keep_relation

    values["glt_token_observation_valid"] = token_mask
    values["glt_relation_observation_valid"] = relation_mask
    return SimpleNamespace(**values)


__all__ = [
    "VARIANTS", "distribution", "graphgate_variant_view", "kabsch_summary",
    "trimer_sample_metrics",
]
