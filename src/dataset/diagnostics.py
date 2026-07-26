import json
import math
import os
from collections import Counter

import torch


def _attr(data, name, default=None):
    return getattr(data, name, default)


def _as_text(value):
    if value is None:
        return ""
    return str(value)


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _numeric_summary(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    tensor = torch.tensor(values, dtype=torch.float)
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "p05": _percentile(values, 0.05),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": float(tensor.max().item()),
    }


def _tensor_values(value):
    if value is None:
        return []
    if torch.is_tensor(value):
        return value.detach().cpu().flatten().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _active_pbc_axis(data):
    pbc = getattr(data, "pbc", None)
    if not torch.is_tensor(pbc) or pbc.numel() < 3:
        return None
    active = torch.nonzero(pbc.detach().cpu().flatten().bool(), as_tuple=False).flatten()
    return int(active[0].item()) if active.numel() == 1 else None


def _cell_t_norms(data):
    axis = _active_pbc_axis(data)
    if axis is None:
        return []
    if hasattr(data, "cell_confs"):
        cell_confs = data.cell_confs
        if torch.is_tensor(cell_confs) and cell_confs.dim() == 3 and cell_confs.size(1) > axis:
            return torch.linalg.vector_norm(cell_confs[:, axis, :], dim=-1).detach().cpu().tolist()
    if hasattr(data, "cell"):
        cell = data.cell
        if torch.is_tensor(cell) and cell.dim() == 2 and cell.size(0) > axis:
            return [float(torch.linalg.vector_norm(cell[axis]).detach().cpu().item())]
    return []


def _first_conformer_pos(data):
    if hasattr(data, "pos_confs"):
        pos_confs = data.pos_confs
        if torch.is_tensor(pos_confs) and pos_confs.dim() == 3 and pos_confs.size(0) > 0:
            return pos_confs[0].detach().cpu().float()
    if hasattr(data, "pos") and torch.is_tensor(data.pos):
        return data.pos.detach().cpu().float()
    return None


def _first_cell(data):
    if hasattr(data, "cell_confs"):
        cell_confs = data.cell_confs
        if torch.is_tensor(cell_confs) and cell_confs.dim() == 3 and cell_confs.size(0) > 0:
            return cell_confs[0].detach().cpu().float()
    if hasattr(data, "cell") and torch.is_tensor(data.cell):
        return data.cell.detach().cpu().float()
    return None


def _estimate_periodic_edge_counts(data, cutoff, eps=1e-8):
    """Estimate PaiNN periodic radius-edge counts for the first conformer.

    This mirrors PaiNNEncoder._periodic_graph_edges at a diagnostics level. It
    does not enforce max_num_neighbors, because the purpose is to measure whether
    periodic images are available within the cutoff at all.
    """
    pos = _first_conformer_pos(data)
    cell = _first_cell(data)
    pbc = getattr(data, "pbc", None)
    if pos is None or cell is None or pbc is None:
        return None
    axis = _active_pbc_axis(data)
    if axis is None:
        return None
    if cell.dim() != 2 or cell.size(0) <= axis or pos.dim() != 2 or pos.size(-1) != 3:
        return None

    vector_t = cell[axis]
    if not torch.isfinite(pos).all() or not torch.isfinite(vector_t).all():
        return None

    counts = {"minus": 0, "center": 0, "plus": 0}
    cutoff = float(cutoff)
    for local_row in range(pos.size(0)):
        target = pos[local_row].view(1, 3)
        for name, shift in (("minus", -1.0), ("center", 0.0), ("plus", 1.0)):
            shifted = pos + vector_t.view(1, 3) * shift
            dist = torch.linalg.vector_norm(target - shifted, dim=-1)
            valid = (dist < cutoff) & (dist > eps)
            if shift == 0.0:
                valid[local_row] = False
            counts[name] += int(valid.sum().item())
    total = counts["minus"] + counts["center"] + counts["plus"]
    periodic = counts["minus"] + counts["plus"]
    return {
        "minus_shift_edges": counts["minus"],
        "center_shift_edges": counts["center"],
        "plus_shift_edges": counts["plus"],
        "periodic_shift_edges": periodic,
        "total_edges": total,
        "periodic_edge_rate": float(periodic / total) if total else None,
    }


def summarize_feature_cache(features, graph_input=None, geom_input=None, painn_cutoff=10.0):
    """Summarize cached structural feature construction health.

    The feature cache stores one PyG Data object per unique SMILES. Graph and
    geometry builders already attach construction metadata to these objects; this
    function turns those fields into auditable aggregate statistics.
    """
    total = len(features)

    graph_requested = 0
    star_success = 0
    star_fallback = 0
    star_reasons = Counter()

    geom_requested = 0
    pbc_success = 0
    pbc_fallback = 0
    pbc_reasons = Counter()
    pbc_status_counts = Counter()
    euclidean_fallback_usable = 0
    t_norms = []
    t_left_norms = []
    t_right_norms = []
    t_cosines = []
    t_relative_differences = []
    attachment_lengths = []
    attachment_ratios = []
    attachment_pass = 0
    attachment_total = 0
    pbc_t_methods = Counter()
    smer_success = 0
    smer_fallback = 0
    smer_reasons = Counter()
    screw_requested = 0
    screw_success = 0
    screw_fallback = 0
    screw_reasons = Counter()
    screw_energy_per_atom = []
    screw_symmetry_rmsd = []
    screw_torsion_degrees = []
    screw_source_counts = Counter()
    screw_source_id_counts = Counter()
    screw_fit_point_counts = []
    screw_five_cell_minimum = []
    polygen_requested = 0
    polygen_success = 0
    polygen_fallback = 0
    polygen_reasons = Counter()
    polygen_seed_modes = Counter()
    polygen_ru_counts = []
    polygen_cell_lengths = []
    polygen_closure_errors = []
    polygen_optimization_losses = []
    polygen_boundary_bond_errors = []
    polygen_boundary_angle_errors = []
    polygen_boundary_torsion_errors = []
    polygen_minimum_nonbonded = []
    polygen_candidate_counts = []
    pbc_edge_counts = {
        "minus_shift_edges": 0,
        "center_shift_edges": 0,
        "plus_shift_edges": 0,
        "periodic_shift_edges": 0,
        "total_edges": 0,
    }
    pbc_edge_sample_count = 0
    pbc_edge_rates = []
    optimizer_counts = Counter()
    conformer_candidates = []
    conformer_converged = []
    conformer_energy = []
    conformer_energy_spread = []
    context_counts = Counter()
    polymer_ecfp_valid = 0
    polymer_ecfp_invalid = 0
    polymer_ecfp_reasons = Counter()

    for data in features.values():
        if bool(_attr(data, "polymer_ecfp_valid", False)):
            polymer_ecfp_valid += 1
        else:
            polymer_ecfp_invalid += 1
            reason = _as_text(_attr(data, "polymer_ecfp_failed_reason", "missing")) or "missing"
            polymer_ecfp_reasons[reason] += 1
        context_counts[str(int(_attr(data, "geom_context_id", 2)))] += 1
        for name, count in (_attr(data, "geom_optimizer_counts", {}) or {}).items():
            optimizer_counts[str(name)] += int(count)
        conformer_candidates.append(int(_attr(data, "geom_conformer_candidate_count", 0)))
        conformer_converged.append(int(_attr(data, "geom_conformer_converged_count", 0)))
        energies = _attr(data, "geom_conformer_energies", None)
        if torch.is_tensor(energies) and energies.numel():
            values = energies.detach().cpu().flatten().tolist()
            conformer_energy.extend(values)
            if len(values) > 1:
                conformer_energy_spread.append(max(values) - min(values))
        requested_graph = _as_text(_attr(data, "requested_graph_input", graph_input)).lower()
        graph_used = _as_text(_attr(data, "structure_input", _attr(data, "graph_input", ""))).lower()
        graph_ok = bool(_attr(data, "graph_build_ok", True))
        if requested_graph == "star_linking":
            graph_requested += 1
            if graph_ok and graph_used in {
                "star_linking", "polygen_periodic", "mips_periodic_supercell",
            }:
                star_success += 1
            else:
                star_fallback += 1
                reason = _as_text(_attr(data, "graph_failed_reason", "unknown")) or "unknown"
                star_reasons[reason] += 1

        requested_geom = _as_text(_attr(data, "geom_requested_input", geom_input)).lower()
        geom_used = _as_text(_attr(data, "geom_input", "")).lower()
        geom_ok = bool(_attr(data, "geom_build_ok", True))
        pbc_status = _as_text(_attr(data, "geom_pbc_status", "unknown")) or "unknown"
        pbc_status_counts[pbc_status] += 1
        if pbc_status not in {"pbc_3d_converged", "polygen_periodic_valid"} and bool(_attr(data, "geom_coordinate_ok", False)):
            euclidean_fallback_usable += 1
        has_pbc_fields = hasattr(data, "cell") and hasattr(data, "pbc")
        if requested_geom == "periodic_pbc":
            geom_requested += 1
            if geom_ok and geom_used == "periodic_pbc" and has_pbc_fields:
                pbc_success += 1
                t_norms.extend(_cell_t_norms(data))
                t_method = _as_text(_attr(data, "geom_t_method", "unknown")) or "unknown"
                pbc_t_methods[t_method] += 1
                t_left_norms.extend(_tensor_values(_attr(data, "geom_t_left_norm", None)))
                t_right_norms.extend(_tensor_values(_attr(data, "geom_t_right_norm", None)))
                t_cosines.extend(_tensor_values(_attr(data, "geom_t_cosine_similarity", None)))
                t_relative_differences.extend(_tensor_values(_attr(data, "geom_t_relative_length_difference", None)))
                lengths = _tensor_values(_attr(data, "geom_attachment_bond_lengths", None))
                ratios = _tensor_values(_attr(data, "geom_attachment_bond_ratios", None))
                attachment_lengths.extend(lengths)
                attachment_ratios.extend(ratios)
                attachment_total += len(ratios)
                attachment_pass += sum(0.75 <= float(value) <= 1.25 for value in ratios)
                edge_counts = _estimate_periodic_edge_counts(data, painn_cutoff)
                if edge_counts is not None:
                    pbc_edge_sample_count += 1
                    for key in pbc_edge_counts:
                        pbc_edge_counts[key] += int(edge_counts[key])
                    if edge_counts["periodic_edge_rate"] is not None:
                        pbc_edge_rates.append(float(edge_counts["periodic_edge_rate"]))
            else:
                pbc_fallback += 1
                reason = _as_text(_attr(data, "geom_failed_reason", "unknown")) or "unknown"
                pbc_reasons[reason] += 1
        elif requested_geom == "smer_context":
            geom_requested += 1
            if geom_ok and geom_used == "smer_context" and bool(_attr(data, "smer_valid", False)):
                smer_success += 1
            else:
                smer_fallback += 1
                reason = _as_text(_attr(data, "geom_failed_reason", "unknown")) or "unknown"
                smer_reasons[reason] += 1
        elif requested_geom == "screw_periodic":
            screw_requested += 1
            screw_source = _as_text(_attr(data, "geom_screw_source", "unknown")) or "unknown"
            screw_source_counts[screw_source] += 1
            screw_source_id_counts[str(int(_attr(data, "geometry_source_id", _attr(data, "geom_screw_source_id", -1))))] += 1
            screw_fit_point_counts.extend(_tensor_values(_attr(data, "geom_screw_fit_point_count", None)))
            if geom_ok and geom_used == "screw_periodic" and bool(_attr(data, "screw_valid", False)):
                screw_success += 1
                screw_energy_per_atom.extend(_tensor_values(_attr(data, "geom_screw_energy_per_atom", None)))
                screw_symmetry_rmsd.extend(_tensor_values(_attr(data, "geom_screw_symmetry_rmsd", None)))
                screw_torsion_degrees.extend(_tensor_values(_attr(data, "geom_screw_torsion_degrees", None)))
                screw_five_cell_minimum.extend(_tensor_values(
                    _attr(data, "geom_five_cell_minimum_distance", None)
                ))
            else:
                screw_fallback += 1
                reason = _as_text(_attr(data, "geom_failed_reason", "unknown")) or "unknown"
                screw_reasons[reason] += 1
        elif requested_geom == "polygen_periodic":
            polygen_requested += 1
            if geom_ok and geom_used == "polygen_periodic" and bool(_attr(data, "polygen_periodic_valid", False)):
                polygen_success += 1
                seed_mode = _as_text(
                    _attr(data, "geom_polygen_seed_mode", "unknown")
                ) or "unknown"
                polygen_seed_modes[seed_mode] += 1
                t_norms.extend(_cell_t_norms(data))
                polygen_ru_counts.extend(_tensor_values(_attr(data, "periodic_ru_count", None)))
                polygen_cell_lengths.extend(_tensor_values(_attr(data, "periodic_cell_length", None)))
                polygen_closure_errors.extend(_tensor_values(_attr(data, "periodic_closure_error", None)))
                polygen_optimization_losses.extend(_tensor_values(_attr(data, "periodic_optimization_loss", None)))
                polygen_boundary_bond_errors.extend(_tensor_values(_attr(data, "periodic_boundary_bond_error", None)))
                polygen_boundary_angle_errors.extend(_tensor_values(_attr(data, "periodic_boundary_angle_error_deg", None)))
                polygen_boundary_torsion_errors.extend(_tensor_values(_attr(data, "periodic_boundary_torsion_error_deg", None)))
                polygen_minimum_nonbonded.extend(_tensor_values(_attr(data, "periodic_minimum_nonbonded_distance", None)))
                polygen_candidate_counts.extend(_tensor_values(_attr(data, "periodic_candidate_count", None)))
                pbc_t_methods["polygen_fractional_cell"] += 1
                edge_counts = _estimate_periodic_edge_counts(data, painn_cutoff)
                if edge_counts is not None:
                    pbc_edge_sample_count += 1
                    for key in pbc_edge_counts:
                        pbc_edge_counts[key] += int(edge_counts[key])
                    if edge_counts["periodic_edge_rate"] is not None:
                        pbc_edge_rates.append(float(edge_counts["periodic_edge_rate"]))
            else:
                polygen_fallback += 1
                reason = _as_text(_attr(data, "geom_failed_reason", "unknown")) or "unknown"
                polygen_reasons[reason] += 1

    t_summary = _numeric_summary(t_norms)
    cutoff = float(painn_cutoff)
    t_gt_cutoff = sum(1 for value in t_norms if float(value) > cutoff)
    t_gt_2cutoff = sum(1 for value in t_norms if float(value) > 2.0 * cutoff)
    total_edges = pbc_edge_counts["total_edges"]
    periodic_edges = pbc_edge_counts["periodic_shift_edges"]
    pbc_edge_summary = {
        **{key: int(value) for key, value in pbc_edge_counts.items()},
        "sample_count": int(pbc_edge_sample_count),
        "periodic_edge_rate": float(periodic_edges / total_edges) if total_edges else None,
        "per_sample_periodic_edge_rate": _numeric_summary(pbc_edge_rates),
    }

    return {
        "total_unique_smiles": int(total),
        "graph": {
            "requested_graph_input": graph_input,
            "star_linking_requested_count": int(graph_requested),
            "star_linking_success_count": int(star_success),
            "star_linking_fallback_count": int(star_fallback),
            "star_linking_success_rate": float(star_success / graph_requested) if graph_requested else None,
            "star_linking_failed_reason_counts": dict(star_reasons.most_common()),
        },
        "geom": {
            "requested_geom_input": geom_input,
            "periodic_pbc_requested_count": int(geom_requested),
            "periodic_pbc_success_count": int(pbc_success),
            "periodic_pbc_fallback_count": int(pbc_fallback),
            "periodic_pbc_success_rate": float(pbc_success / geom_requested) if geom_requested else None,
            "periodic_pbc_failed_reason_counts": dict(pbc_reasons.most_common()),
            "smer_context_success_count": int(smer_success),
            "smer_context_fallback_count": int(smer_fallback),
            "smer_context_success_rate": float(smer_success / geom_requested) if requested_geom == "smer_context" and geom_requested else None,
            "smer_context_failed_reason_counts": dict(smer_reasons.most_common()),
            "screw_periodic_requested_count": int(screw_requested),
            "screw_periodic_success_count": int(screw_success),
            "screw_periodic_fallback_count": int(screw_fallback),
            "screw_periodic_success_rate": float(screw_success / screw_requested) if screw_requested else None,
            "screw_periodic_failed_reason_counts": dict(screw_reasons.most_common()),
            "polygen_periodic_requested_count": int(polygen_requested),
            "polygen_periodic_success_count": int(polygen_success),
            "polygen_periodic_fallback_count": int(polygen_fallback),
            "polygen_periodic_success_rate": float(polygen_success / polygen_requested) if polygen_requested else None,
            "polygen_periodic_failed_reason_counts": dict(polygen_reasons.most_common()),
            "pbc_status_counts": dict(pbc_status_counts.most_common()),
            "pbc_3d_converged_count": int(pbc_status_counts["pbc_3d_converged"]),
            "pbc_3d_unconverged_count": int(pbc_status_counts["pbc_3d_unconverged"]),
            "pbc_2d_rejected_count": int(pbc_status_counts["pbc_2d_rejected"]),
            "pbc_quality_pass_count": int(pbc_status_counts["pbc_3d_converged"]),
            "pbc_quality_pass_rate": (
                float(pbc_status_counts["pbc_3d_converged"] / geom_requested)
                if geom_requested else None
            ),
            "euclidean_fallback_usable_count": int(euclidean_fallback_usable),
            "geom_context_counts": dict(context_counts.most_common()),
            "optimizer_counts": dict(optimizer_counts.most_common()),
            "conformer_candidate_count": _numeric_summary(conformer_candidates),
            "conformer_converged_count": _numeric_summary(conformer_converged),
            "selected_conformer_energy": _numeric_summary(conformer_energy),
            "selected_conformer_energy_spread": _numeric_summary(conformer_energy_spread),
        },
        "painn_pbc": {
            "t_norm": t_summary,
            "t_method_counts": dict(pbc_t_methods.most_common()),
            "t_left_norm": _numeric_summary(t_left_norms),
            "t_right_norm": _numeric_summary(t_right_norms),
            "t_cosine_similarity": _numeric_summary(t_cosines),
            "t_relative_length_difference": _numeric_summary(t_relative_differences),
            "attachment_bond_length": _numeric_summary(attachment_lengths),
            "attachment_bond_ratio": _numeric_summary(attachment_ratios),
            "attachment_bond_pass_rate": float(attachment_pass / attachment_total) if attachment_total else None,
            "t_norm_gt_cutoff_count": int(t_gt_cutoff),
            "t_norm_gt_2cutoff_count": int(t_gt_2cutoff),
            "edge_counts": pbc_edge_summary,
            "cutoff": cutoff,
        },
        "screw_periodic": {
            "energy_per_atom": _numeric_summary(screw_energy_per_atom),
            "symmetry_rmsd": _numeric_summary(screw_symmetry_rmsd),
            "torsion_degrees": _numeric_summary(screw_torsion_degrees),
            "source_counts": dict(screw_source_counts.most_common()),
            "source_id_counts": dict(screw_source_id_counts.most_common()),
            "fit_point_count": _numeric_summary(screw_fit_point_counts),
            "five_cell_minimum_distance": _numeric_summary(screw_five_cell_minimum),
        },
        "polygen_periodic": {
            "seed_mode_counts": dict(polygen_seed_modes.most_common()),
            "periodic_ru_count": _numeric_summary(polygen_ru_counts),
            "cell_length": _numeric_summary(polygen_cell_lengths),
            "closure_error": _numeric_summary(polygen_closure_errors),
            "optimization_loss": _numeric_summary(polygen_optimization_losses),
            "boundary_bond_error": _numeric_summary(polygen_boundary_bond_errors),
            "boundary_angle_error": _numeric_summary(polygen_boundary_angle_errors),
            "boundary_torsion_error": _numeric_summary(polygen_boundary_torsion_errors),
            "minimum_nonbonded_distance": _numeric_summary(polygen_minimum_nonbonded),
            "candidate_count": _numeric_summary(polygen_candidate_counts),
        },
        "polymer_ecfp": {
            "target": "capped_3mer_morgan_r2_2048",
            "valid_count": int(polymer_ecfp_valid),
            "invalid_count": int(polymer_ecfp_invalid),
            "valid_rate": float(polymer_ecfp_valid / total) if total else None,
            "failed_reason_counts": dict(polymer_ecfp_reasons.most_common()),
        },
    }


def print_dataset_diagnostics(summary):
    graph = summary["graph"]
    geom = summary["geom"]
    pbc = summary["painn_pbc"]

    graph_total = graph["star_linking_requested_count"]
    graph_rate = graph["star_linking_success_rate"]
    if graph_rate is not None:
        print(
            "[diagnostics] star_linking success: "
            f"{graph['star_linking_success_count']}/{graph_total} = {graph_rate:.2%}, "
            f"fallback={graph['star_linking_fallback_count']}"
        )

    geom_total = geom["periodic_pbc_requested_count"]
    geom_rate = geom["periodic_pbc_success_rate"]
    if geom_rate is not None:
        print(
            "[diagnostics] periodic_pbc success: "
            f"{geom['periodic_pbc_success_count']}/{geom_total} = {geom_rate:.2%}, "
            f"fallback={geom['periodic_pbc_fallback_count']}"
        )
        print(
            "[diagnostics] PBC quality: "
            f"pass={geom['pbc_quality_pass_count']}, "
            f"3d_unconverged={geom['pbc_3d_unconverged_count']}, "
            f"2d_rejected={geom['pbc_2d_rejected_count']}"
        )

    screw_total = geom.get("screw_periodic_requested_count", 0)
    screw_rate = geom.get("screw_periodic_success_rate")
    if screw_rate is not None:
        screw = summary.get("screw_periodic", {})
        energy = screw.get("energy_per_atom", {})
        residual = screw.get("symmetry_rmsd", {})
        print(
            "[diagnostics] screw_periodic success: "
            f"{geom['screw_periodic_success_count']}/{screw_total} = {screw_rate:.2%}, "
            f"fallback={geom['screw_periodic_fallback_count']}"
        )
        if energy.get("count", 0) > 0:
            print(
                "[diagnostics] screw quality: "
                f"energy/atom p95={energy['p95']:.3f}, "
                f"symmetry_rmsd max={residual['max']:.3e}"
            )
    polygen_total = geom.get("polygen_periodic_requested_count", 0)
    polygen_rate = geom.get("polygen_periodic_success_rate")
    if polygen_rate is not None:
        quality = summary.get("polygen_periodic", {})
        print(
            "[diagnostics] polygen_periodic success: "
            f"{geom['polygen_periodic_success_count']}/{polygen_total} = {polygen_rate:.2%}, "
            f"fallback={geom['polygen_periodic_fallback_count']}"
        )
        cell_length = quality.get("cell_length", {})
        ru_count = quality.get("periodic_ru_count", {})
        if cell_length.get("count", 0) > 0 and cell_length.get("p50") is not None:
            print(
                "[diagnostics] polygen periodic cell: "
                f"m-RU p50={ru_count.get('p50')}, "
                f"L p50={cell_length['p50']:.3f} A"
            )
        else:
            print("[diagnostics] polygen periodic cell: no valid periodic cells")
    ecfp = summary.get("polymer_ecfp", {})
    if ecfp.get("valid_rate") is not None:
        print(
            "[diagnostics] Polymer ECFP target success: "
            f"{ecfp['valid_count']}/{summary['total_unique_smiles']} = {ecfp['valid_rate']:.2%}, "
            f"invalid={ecfp['invalid_count']}"
        )
    t_norm = pbc["t_norm"]
    if t_norm["count"] > 0:
        print(
            "[diagnostics] PaiNN PBC |T|: "
            f"count={t_norm['count']}, mean={t_norm['mean']:.3f}, "
            f"p50={t_norm['p50']:.3f}, p95={t_norm['p95']:.3f}, "
            f"max={t_norm['max']:.3f}, "
            f">cutoff={pbc['t_norm_gt_cutoff_count']}, "
            f">2cutoff={pbc['t_norm_gt_2cutoff_count']}"
        )
        if pbc.get("t_method_counts"):
            print(f"[diagnostics] PaiNN PBC T methods: {pbc['t_method_counts']}")
        edge_counts = pbc.get("edge_counts", {})
        if edge_counts.get("sample_count", 0) > 0:
            rate = edge_counts.get("periodic_edge_rate")
            rate_text = f"{rate:.2%}" if rate is not None else "n/a"
            print(
                "[diagnostics] PaiNN PBC periodic edges: "
                f"samples={edge_counts['sample_count']}, "
                f"minus={edge_counts['minus_shift_edges']}, "
                f"center={edge_counts['center_shift_edges']}, "
                f"plus={edge_counts['plus_shift_edges']}, "
                f"periodic_rate={rate_text}"
            )


def write_dataset_diagnostics(summary, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"[diagnostics] wrote {output_path}")
