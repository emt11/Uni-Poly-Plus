#!/usr/bin/env python3
"""Run the bounded, CPU-only Trimer-v9 300-record real-data pilot."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import heapq
import json
import math
import platform
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import Lipinski
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_dual_glt import audit_frozen_stereo
from src.dataset.dataset import _compute_ru_base_layer, _compute_topology_layer
from src.dataset.graph_data import build_periodic_multimer_mol
from src.dataset.lmdb_cache import LmdbLayerStore, sample_key_from_smiles
from src.dataset.mips_trimer_contract import (
    TRIMER_CONTENT_SCHEMA,
    TRIMER_LMDB_SCHEMA,
    TRIMER_PROTOCOL,
)
from src.dataset.trimer_mcl import (
    TRIMER_CANDIDATES_PER_ROUND,
    TRIMER_ETKDG_MAX_ITERATIONS,
    TRIMER_ETKDG_TIMEOUT_SECONDS,
    TRIMER_MAX_ROUNDS,
    TRIMER_MMFF_RELAX_MAX_ITERATIONS,
    TRIMER_SAMPLE_TIMEOUT_SECONDS,
    TrimerContractError,
    attach_finite_trimer_mcl,
)


SEED = 20260913
KNOWN_CONFLICT = "*C1=C(/C=C/c2ccc(*)cc2)C=C1"
OLD_ROOT = ROOT / "data/processed/trimer_ensemble_stage_a_20260912/pilot256"
DOUBLE_STEREO = {
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOCIS,
    Chem.BondStereo.STEREOTRANS,
}
TETRA_STEREO = {
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
}
SPECIAL_ELEMENTS = {"S", "P", "F", "Cl", "Br", "I"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=300, choices=[300])
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw/PI1M_v2.csv")
    parser.add_argument("--downstream", type=Path, default=ROOT / "data/raw/smi_all.csv")
    return parser.parse_args()


def json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def write_json(path, value):
    path.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), sort_keys=True, allow_nan=False) + "\n")


def append_jsonl(path, rows):
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), sort_keys=True, allow_nan=False) + "\n")


def rank(smiles, label="general"):
    return int.from_bytes(
        hashlib.sha256(f"{SEED}:{label}:{smiles}".encode()).digest(), "big"
    )


def offer(heap, limit, item_rank, source, source_row, smiles):
    entry = (-item_rank, str(source), int(source_row), str(smiles))
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif item_rank < -heap[0][0]:
        heapq.heapreplace(heap, entry)


def iter_source(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        field = "SMILES" if "SMILES" in reader.fieldnames else "smiles"
        for source_row, row in enumerate(reader):
            smiles = str(row.get(field, "")).strip()
            if smiles:
                yield source_row, smiles


def molecule_profile(smiles):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    normalized = Chem.MolFromSmiles(canonical)
    if normalized is None:
        return None
    declared_double = sum(bond.GetStereo() in DOUBLE_STEREO for bond in normalized.GetBonds())
    declared_tetra = sum(atom.GetChiralTag() in TETRA_STEREO for atom in normalized.GetAtoms())
    heavy = sum(atom.GetAtomicNum() > 1 for atom in normalized.GetAtoms())
    aromatic = sum(atom.GetIsAromatic() for atom in normalized.GetAtoms())
    rotatable = int(Lipinski.NumRotatableBonds(normalized))
    elements = sorted({atom.GetSymbol() for atom in normalized.GetAtoms()} & SPECIAL_ELEMENTS)
    categories = []
    if not declared_double and not declared_tetra:
        categories.append("A_no_declared_stereo")
    if declared_double and not declared_tetra:
        categories.append("B_declared_double_only")
    if declared_tetra and not declared_double:
        categories.append("C_declared_tetra_only")
    if declared_double and declared_tetra:
        categories.append("D_double_and_tetra")
    if rotatable >= 10:
        categories.append("E_high_flexibility")
    if heavy and aromatic / heavy >= 0.5 and rotatable <= 5:
        categories.append("F_aromatic_rigid")
    if heavy >= 40:
        categories.append("G_large_base_heavy_count")
    if elements:
        categories.append("H_selected_elements")
    return {
        "canonical_smiles": canonical,
        "source_declared_double_count": declared_double,
        "source_declared_tetra_count": declared_tetra,
        "base_heavy_atom_count": heavy,
        "rotatable_bond_count": rotatable,
        "selected_elements": elements,
        "categories": categories,
    }


def declared_stereo_info(canonical_smiles):
    heavy, metadata = build_periodic_multimer_mol(
        Chem.MolFromSmiles(canonical_smiles), 3, close_periodic=False
    )
    molecule = Chem.AddHs(heavy)
    heavy_count = heavy.GetNumAtoms()
    offsets = list(np.asarray(metadata["atom_ru_index"], dtype=int) - 1)
    parents = [-1] * molecule.GetNumAtoms()
    for atom_index in range(heavy_count, molecule.GetNumAtoms()):
        parent = int(next(iter(molecule.GetAtomWithIdx(atom_index).GetNeighbors())).GetIdx())
        parents[atom_index] = parent
        offsets.append(offsets[parent])
    doubles = []
    for bond in heavy.GetBonds():
        if bond.GetStereo() in DOUBLE_STEREO:
            doubles.append({
                "bond_atoms": [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()],
                "reference_atoms": list(bond.GetStereoAtoms()),
                "stereo": str(bond.GetStereo()),
                "ru_offsets": [offsets[bond.GetBeginAtomIdx()], offsets[bond.GetEndAtomIdx()]],
            })
    tetra = []
    for atom in molecule.GetAtoms():
        if atom.GetChiralTag() in TETRA_STEREO:
            neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
            tetra.append({
                "center_atom": atom.GetIdx(),
                "chiral_tag": str(atom.GetChiralTag()),
                "neighbor_atoms": neighbors,
                "center_ru_offset": offsets[atom.GetIdx()],
                "neighbor_ru_offsets": [offsets[index] for index in neighbors],
                "neighbor_h_parent_heavy_index": [parents[index] for index in neighbors],
            })
    return {"double_bonds": doubles, "tetrahedral_centers": tetra}


def select_samples(source, downstream):
    history_rows = json.loads((OLD_ROOT / "selection.json").read_text())
    old_store = LmdbLayerStore(OLD_ROOT / "cache")
    history = []
    try:
        for row in history_rows:
            profile = molecule_profile(row["smiles"])
            if profile is None:
                raise RuntimeError(f"historical real sample no longer parses: {row['smiles']}")
            record = old_store[sample_key_from_smiles(row["smiles"])]
            reason = str(record.search_stop_reason)
            categories = list(profile["categories"])
            if reason == "MMFF_UNSUPPORTED":
                categories.append("I_historical_mmff_unsupported")
            if row["smiles"] == KNOWN_CONFLICT:
                categories.append("J_historical_stereo_conflict")
            if reason == "ETKDG_NO_VALID_CONFORMER":
                categories.append("K_historical_etkdg_difficult")
            history.append({
                **row,
                **profile,
                "categories": categories,
                "historical_v8_stop_reason": reason,
                "selection_stratum": "historical_deterministic_pilot256",
            })
    finally:
        old_store.close()
    if len(history) != 256 or len({row["canonical_smiles"] for row in history}) != 256:
        raise RuntimeError("historical deterministic 256 selection is incomplete or duplicated")

    downstream_heap = []
    for source_row, smiles in iter_source(downstream):
        offer(downstream_heap, 1000, rank(smiles, "downstream"), downstream, source_row, smiles)

    selected = list(history)
    selected_ids = {row["canonical_smiles"] for row in selected}

    def materialize(heap, stratum):
        output = []
        for _, path, source_row, smiles in sorted(heap, key=lambda item: (-item[0], item[3])):
            profile = molecule_profile(smiles)
            if profile is None or profile["canonical_smiles"] in selected_ids:
                continue
            output.append({
                "smiles": smiles,
                "source": path,
                "source_row": source_row,
                **profile,
                "historical_v8_stop_reason": None,
                "selection_stratum": stratum,
            })
        return output

    downstream_rows = materialize(downstream_heap, "downstream_diversity")

    def take(rows, count, predicate):
        added = 0
        for row in rows:
            identity = row["canonical_smiles"]
            if identity in selected_ids or not predicate(row):
                continue
            selected.append(row)
            selected_ids.add(identity)
            added += 1
            if added == count:
                return
        raise RuntimeError(f"selection stratum shortfall: requested {count}, found {added}")

    take(downstream_rows, 44, lambda row: True)
    if len(selected) != 300:
        raise RuntimeError(f"expected 300 selected samples, got {len(selected)}")
    for index, row in enumerate(selected):
        row["pilot_index"] = index
        row["sample_id"] = sample_key_from_smiles(row["canonical_smiles"]).hex()
    return selected


def best_candidate(rows):
    valid = [row for row in rows if row.get("final_valid")]
    if not valid:
        return None
    converged = [row for row in valid if row.get("mmff_converged")]
    pool = converged or valid
    return min(pool, key=lambda row: (float(row["mmff_energy"]), int(row["candidate_index"])))


def expanded_candidates(sample, diagnostics):
    output = []
    rounds = {int(row["round_id"]): row for row in diagnostics.get("rounds", [])}
    for round_id, round_row in sorted(rounds.items()):
        observed = {int(row["candidate_id"]): row for row in round_row["candidates"]}
        for candidate_index in range(int(round_row["requested"])):
            raw = observed.get(candidate_index)
            base = {
                "sample_id": sample["sample_id"],
                "canonical_smiles": sample["canonical_smiles"],
                "round_id": round_id,
                "candidate_index": candidate_index,
                "requested": True,
                "embedded": raw is not None,
            }
            if raw is None:
                base.update({
                    "conformer_id": None,
                    "finite_pre_mmff": False,
                    "double_bond_stereo_pre_pass": None,
                    "tetra_stereo_pre_pass": None,
                    "mmff_attempted": False,
                    "mmff_status": None,
                    "mmff_converged": None,
                    "mmff_energy_finite": False,
                    "mmff_energy": None,
                    "finite_post_mmff": False,
                    "double_bond_stereo_post_f64_pass": None,
                    "tetra_stereo_post_f64_pass": None,
                    "double_bond_stereo_post_f32_pass": None,
                    "tetra_stereo_post_f32_pass": None,
                    "final_valid": False,
                    "rejection_reason": "ETKDG_NO_CONFORMER",
                    "stereo_error": None,
                })
            else:
                for key in (
                    "conformer_id", "finite_pre_mmff", "double_bond_stereo_pre_pass",
                    "tetra_stereo_pre_pass", "mmff_attempted", "mmff_status",
                    "mmff_converged", "mmff_energy_finite", "mmff_energy",
                    "finite_post_mmff", "double_bond_stereo_post_f64_pass",
                    "tetra_stereo_post_f64_pass", "double_bond_stereo_post_f32_pass",
                    "tetra_stereo_post_f32_pass", "final_valid", "stereo_error",
                ):
                    base[key] = raw.get(key)
                base["rejection_reason"] = raw.get("rejection") or None
            output.append(base)
    return output


def round_summary(round_rows, round_id):
    rows = [row for row in round_rows if row["round_id"] == round_id]
    if not rows:
        return {name: None for name in (
            "requested", "embedded", "finite", "pre_stereo_valid", "mmff_finite",
            "mmff_converged", "final_valid", "first4_final_valid_count",
            "last4_final_valid_count", "best_energy_first4", "best_energy_all8",
        )}
    first = [row for row in rows if row["candidate_index"] < 4]
    last = [row for row in rows if row["candidate_index"] >= 4]
    best4, best8 = best_candidate(first), best_candidate(rows)
    return {
        "requested": len(rows),
        "embedded": sum(bool(row["embedded"]) for row in rows),
        "finite": sum(bool(row["finite_pre_mmff"]) for row in rows),
        "pre_stereo_valid": sum(
            row["double_bond_stereo_pre_pass"] is True
            and row["tetra_stereo_pre_pass"] is True for row in rows
        ),
        "mmff_finite": sum(bool(row["mmff_energy_finite"]) for row in rows),
        "mmff_converged": sum(row["mmff_converged"] is True for row in rows),
        "final_valid": sum(bool(row["final_valid"]) for row in rows),
        "first4_final_valid_count": sum(bool(row["final_valid"]) for row in first),
        "last4_final_valid_count": sum(bool(row["final_valid"]) for row in last),
        "best_energy_first4": None if best4 is None else float(best4["mmff_energy"]),
        "best_energy_all8": None if best8 is None else float(best8["mmff_energy"]),
    }


def fixed_identity_rmsd(left, right):
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("RMSD identity arrays differ")
    x, y = x - x.mean(0), y - y.mean(0)
    u, _, vh = np.linalg.svd(x.T @ y)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return float(np.sqrt(np.mean(np.sum((x @ rotation - y) ** 2, axis=1))))


def old_geometry_comparison(sample, old_record, new_record):
    if int(getattr(old_record, "num_conformers", 0)) == 0 or not new_record.trimer_geometry_valid:
        return None
    energies = torch.as_tensor(old_record.conformer_energies, dtype=torch.float64)
    old_index = int(torch.argmin(energies))
    old_pos = torch.as_tensor(old_record.conformer_positions[old_index]).double()
    old_states = [
        (int(base), int(offset)) for base, offset in zip(
            old_record.trimer_base_ru_atom_id, old_record.trimer_ru_offset
        )
    ]
    heavy_indices = torch.as_tensor(new_record.trimer_heavy_indices).long()
    new_pos = new_record.trimer_pos[heavy_indices].double()
    new_states = [
        (int(base), int(offset)) for base, offset in zip(
            new_record.trimer_base_ru_atom_id[heavy_indices],
            new_record.trimer_ru_offset[heavy_indices],
        )
    ]
    if len(set(old_states)) != len(old_states) or set(old_states) != set(new_states):
        raise TrimerContractError("old_new_heavy_identity_mapping_failed")
    old_lookup = {state: index for index, state in enumerate(old_states)}
    old_aligned = old_pos[[old_lookup[state] for state in new_states]]
    old_dist = torch.cdist(old_aligned, old_aligned)
    new_dist = torch.cdist(new_pos, new_pos)
    upper = torch.triu(torch.ones_like(old_dist, dtype=torch.bool), diagonal=1)
    pair_delta = torch.abs(old_dist - new_dist)
    heavy_inverse = torch.full((new_record.trimer_pos.size(0),), -1, dtype=torch.long)
    heavy_inverse[heavy_indices] = torch.arange(heavy_indices.numel())
    edge = new_record.trimer_edge_index.long()
    keep = (heavy_inverse[edge[0]] >= 0) & (heavy_inverse[edge[1]] >= 0)
    heavy_edge = heavy_inverse[edge[:, keep]]
    bonded = torch.zeros_like(upper)
    bonded[heavy_edge[0], heavy_edge[1]] = True
    nonbonded = upper & ~bonded
    old_view = copy.copy(old_record)
    old_view.trimer_pos = old_pos.float()
    old_view.trimer_geometry_valid = True
    old_view.trimer_geometry_is_3d = True
    old_view.trimer_2d_fallback = False
    old_conflict = False
    old_checked = 0
    try:
        old_checked = int(audit_frozen_stereo(old_view, sample["canonical_smiles"]))
    except ValueError:
        old_conflict = True
    new_counts = audit_frozen_stereo(
        new_record, sample["canonical_smiles"], return_details=True
    )
    return {
        "sample_id": sample["sample_id"],
        "canonical_smiles": sample["canonical_smiles"],
        "old_conformer_index": old_index,
        "old_energy": float(energies[old_index]),
        "new_energy": float(new_record.trimer_conformer_energy),
        "heavy_atom_rmsd": fixed_identity_rmsd(old_aligned, new_pos),
        "mean_absolute_pairwise_distance_difference": float(pair_delta[upper].mean()),
        "mean_absolute_nonbonded_distance_difference": (
            float(pair_delta[nonbonded].mean()) if bool(nonbonded.any()) else None
        ),
        "old_declared_stereo_checked": old_checked,
        "old_declared_coordinate_stereo_conflict": old_conflict,
        "new_declared_stereo": new_counts,
    }


def distribution(values):
    values = np.asarray([value for value in values if value is not None and math.isfinite(value)], dtype=float)
    if not values.size:
        return {key: None for key in ("mean", "median", "p90", "p95", "max")}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def safe_spearman(left, right):
    pairs = [(a, b) for a, b in zip(left, right) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    value = float(spearmanr([a for a, _ in pairs], [b for _, b in pairs]).statistic)
    return value if math.isfinite(value) else None


def classify_failure(summary, rows):
    stop = summary["stop_reason"]
    if stop == "MMFF_UNSUPPORTED":
        return "MMFF_UNSUPPORTED"
    if stop == "timeout":
        return "TIMEOUT"
    reasons = Counter(row["rejection_reason"] for row in rows if row["rejection_reason"])
    if rows and not any(row["embedded"] for row in rows):
        return "ETKDG_NO_CONFORMER"
    if any("TETRA_STEREO" in reason for reason in reasons):
        return "TETRA_STEREO_REJECT"
    if any("DOUBLE_BOND_STEREO" in reason for reason in reasons):
        return "DOUBLE_BOND_STEREO_REJECT"
    if any("POST_F64" in reason for reason in reasons):
        return "POST_MMFF_STEREO_REJECT"
    if any("POST_F32" in reason for reason in reasons):
        return "FLOAT32_STEREO_REJECT"
    if rows and not any(row["finite_pre_mmff"] for row in rows if row["embedded"]):
        return "NO_FINITE_3D"
    if rows and not any(row["mmff_energy_finite"] for row in rows):
        return "MMFF_NO_FINITE_CANDIDATE"
    return "NO_FINAL_VALID_CONFORMER"


def analyze(samples, summaries, candidates, comparisons):
    eligible = [row for row in summaries if row["stop_reason"] != "MMFF_UNSUPPORTED"]
    round0_success = [row for row in eligible if row["round0_final_valid"] > 0]
    retry = [row for row in eligible if row["retry_used"]]
    recovered = [row for row in retry if row["round1_final_valid"] > 0]
    successes = [row for row in summaries if row["geometry_valid"]]
    first4 = [row for row in eligible if row["round0_first4_final_valid_count"] > 0]
    all8 = [row for row in eligible if row["round0_final_valid"] > 0]
    deltas = [
        row["round0_best_energy_first4"] - row["round0_best_energy_all8"]
        for row in eligible
        if row["round0_best_energy_first4"] is not None
        and row["round0_best_energy_all8"] is not None
    ]
    rejection_counts = Counter(
        row["rejection_reason"] for row in candidates if row["rejection_reason"]
    )
    final_failures = Counter(
        row["failure_category"] for row in summaries if not row["geometry_valid"]
    )
    stereo = {
        "samples_with_declared_double_bond_stereo": sum(row["declared_double_bond_stereo_count"] > 0 for row in summaries),
        "samples_with_declared_tetrahedral_stereo": sum(row["declared_tetrahedral_stereo_count"] > 0 for row in summaries),
        "declared_double_bond_centers": sum(row["declared_double_bond_stereo_count"] for row in summaries),
        "declared_tetrahedral_centers": sum(row["declared_tetrahedral_stereo_count"] for row in summaries),
        "candidate_double_bond_pre_rejections": sum(row["rejection_reason"] == "DOUBLE_BOND_STEREO_PRE" for row in candidates),
        "candidate_double_bond_post_rejections": sum(str(row["rejection_reason"]).startswith("DOUBLE_BOND_STEREO_POST") for row in candidates),
        "candidate_tetra_pre_rejections": sum(row["rejection_reason"] == "TETRA_STEREO_PRE" for row in candidates),
        "candidate_tetra_post_rejections": sum(str(row["rejection_reason"]).startswith("TETRA_STEREO_POST") for row in candidates),
        "sample_final_failure_due_to_double_bond_stereo": final_failures["DOUBLE_BOND_STEREO_REJECT"],
        "sample_final_failure_due_to_tetra_stereo": final_failures["TETRA_STEREO_REJECT"],
    }
    any_converged, only_nonconverged, lower_nonconv = 0, 0, 0
    for sample in summaries:
        rows = [row for row in candidates if row["sample_id"] == sample["sample_id"] and row["final_valid"]]
        converged = [row for row in rows if row["mmff_converged"]]
        nonconverged = [row for row in rows if not row["mmff_converged"]]
        any_converged += bool(converged)
        only_nonconverged += bool(rows and not converged)
        lower_nonconv += bool(
            converged and nonconverged
            and min(row["mmff_energy"] for row in nonconverged)
            < min(row["mmff_energy"] for row in converged)
        )
    timing = {
        name: distribution([row[name] for row in summaries])
        for name in ("embedding_time", "mmff_time", "stereo_time", "sample_total_time")
    }
    timing["round0_success_total_time"] = distribution([row["sample_total_time"] for row in round0_success])
    timing["retry_used_total_time"] = distribution([row["sample_total_time"] for row in retry])
    slowest = sorted(summaries, key=lambda row: row["sample_total_time"], reverse=True)[:20]
    categories = Counter(category for sample in samples for category in sample["categories"])
    result = {
        "scope": "300 explicitly selected real records only; not full-cache coverage",
        "sample_count": len(summaries),
        "category_counts": dict(sorted(categories.items())),
        "source_counts": dict(sorted(Counter(row["source"] for row in samples).items())),
        "basic_success": {
            "mmff_eligible_count": len(eligible),
            "round0_success_count": len(round0_success),
            "round0_success_rate": len(round0_success) / len(eligible),
            "retry_invoked_count": len(retry),
            "retry_invocation_rate": len(retry) / len(eligible),
            "retry_recovered_count": len(recovered),
            "retry_recovery_rate_among_retry_used": len(recovered) / len(retry) if retry else None,
            "final_success_count": len(successes),
            "final_success_rate": len(successes) / len(summaries),
            "final_failure_count": len(summaries) - len(successes),
            "final_failure_rate": 1 - len(successes) / len(summaries),
        },
        "first4_vs_all8": {
            "denominator_mmff_eligible": len(eligible),
            "success4_count": len(first4),
            "success4_rate": len(first4) / len(eligible),
            "success8_count": len(all8),
            "success8_rate": len(all8) / len(eligible),
            "last4_only_recovered_count": sum(row["round0_first4_final_valid_count"] == 0 and row["round0_final_valid"] > 0 for row in eligible),
            "last4_only_recovered_rate": sum(row["round0_first4_final_valid_count"] == 0 and row["round0_final_valid"] > 0 for row in eligible) / len(eligible),
            "energy_unit": "kcal/mol (RDKit MMFF force-field convention)",
            "energy_unit_reference": "https://rdkit.org/docs/source/rdkit.ForceField.rdForceField.html",
            "delta_energy_count": len(deltas),
            "delta_energy_mean": statistics.fmean(deltas) if deltas else None,
            "delta_energy_median": statistics.median(deltas) if deltas else None,
            "delta_energy_p90": float(np.percentile(deltas, 90)) if deltas else None,
            "delta_energy_max": max(deltas) if deltas else None,
            "fraction_delta_gt_0": sum(value > 0 for value in deltas) / len(deltas) if deltas else None,
            "fraction_delta_gt_0_5": sum(value > 0.5 for value in deltas) / len(deltas) if deltas else None,
            "fraction_delta_gt_1_0": sum(value > 1.0 for value in deltas) / len(deltas) if deltas else None,
            "fraction_delta_gt_3_0": sum(value > 3.0 for value in deltas) / len(deltas) if deltas else None,
        },
        "retry_value": {
            "recovered_count": len(recovered),
            "mean_total_time_increment_vs_round0_success": (
                statistics.fmean(row["sample_total_time"] for row in retry)
                - statistics.fmean(row["sample_total_time"] for row in round0_success)
                if retry and round0_success else None
            ),
            "median_total_time_increment_vs_round0_success": (
                statistics.median(row["sample_total_time"] for row in retry)
                - statistics.median(row["sample_total_time"] for row in round0_success)
                if retry and round0_success else None
            ),
        },
        "stereo": stereo,
        "mmff": {
            "unsupported_samples": final_failures["MMFF_UNSUPPORTED"],
            "samples_with_at_least_one_converged_final_valid": any_converged,
            "samples_with_only_nonconverged_final_valid": only_nonconverged,
            "samples_selected_nonconverged": sum(row["geometry_valid"] and not row["selected_converged"] for row in summaries),
            "samples_with_lower_nonconverged_energy_despite_converged_candidate": lower_nonconv,
        },
        "explicit_h": {
            "heavy_atom_count": distribution([row["heavy_atom_count"] for row in summaries]),
            "hydrogen_count": distribution([row["hydrogen_count"] for row in summaries]),
            "all_atom_count": distribution([row["all_atom_count"] for row in summaries]),
            "all_over_heavy_ratio": distribution([row["all_over_heavy_ratio"] for row in summaries]),
            "spearman_total_time_vs_heavy_atom_count": safe_spearman(
                [row["heavy_atom_count"] for row in summaries], [row["sample_total_time"] for row in summaries]
            ),
            "spearman_total_time_vs_all_atom_count": safe_spearman(
                [row["all_atom_count"] for row in summaries], [row["sample_total_time"] for row in summaries]
            ),
        },
        "timing_seconds": timing,
        "candidate_rejection_counts": dict(sorted(rejection_counts.items())),
        "sample_final_failure_counts": dict(sorted(final_failures.items())),
        "slowest_samples": [{key: row[key] for key in (
            "sample_id", "canonical_smiles", "heavy_atom_count", "all_atom_count",
            "retry_used", "embedding_time", "mmff_time", "sample_total_time", "stop_reason",
        )} for row in slowest],
        "old_geometry_comparison": {
            "count": len(comparisons),
            "heavy_atom_rmsd": distribution([row["heavy_atom_rmsd"] for row in comparisons]),
            "mean_absolute_pairwise_distance_difference": distribution([row["mean_absolute_pairwise_distance_difference"] for row in comparisons]),
            "mean_absolute_nonbonded_distance_difference": distribution([row["mean_absolute_nonbonded_distance_difference"] for row in comparisons]),
            "old_declared_coordinate_stereo_conflicts": sum(row["old_declared_coordinate_stereo_conflict"] for row in comparisons),
            "new_declared_coordinate_stereo_conflicts": 0,
        },
    }
    return result


def report_markdown(report):
    b, e, r, s, m, h, t = (
        report["basic_success"], report["first4_vs_all8"], report["retry_value"],
        report["stereo"], report["mmff"], report["explicit_h"], report["timing_seconds"],
    )
    decision = (
        "3. 当前存在必须先修复的问题"
        if report["sample_final_failure_counts"].get("TRIMER_CONTRACT_ERROR", 0)
        else "1. 推荐进入更大规模 pilot，但暂不全量重建"
    )
    return f"""# Trimer v9 有限真实记录 pilot

范围：{report['scope']}

## A. 基本成功率

- 总样本：{report['sample_count']}
- MMFF eligible：{b['mmff_eligible_count']}
- Round 0：{b['round0_success_count']} / {b['mmff_eligible_count']} ({b['round0_success_rate']:.4%})
- Retry invoked：{b['retry_invoked_count']} ({b['retry_invocation_rate']:.4%})
- Retry recovered：{b['retry_recovered_count']}；retry 内恢复率 {b['retry_recovery_rate_among_retry_used'] if b['retry_recovery_rate_among_retry_used'] is not None else 'NA'}
- Final：{b['final_success_count']} / {report['sample_count']} ({b['final_success_rate']:.4%})

## B. 前 4 vs 全 8

- success4：{e['success4_count']} / {e['denominator_mmff_eligible']} ({e['success4_rate']:.4%})
- success8：{e['success8_count']} / {e['denominator_mmff_eligible']} ({e['success8_rate']:.4%})
- 后 4 个独立恢复：{e['last4_only_recovered_count']} ({e['last4_only_recovered_rate']:.4%})
- ΔE=E4-E8（{e['energy_unit']}；[RDKit CalcEnergy 文档]({e['energy_unit_reference']})）：median={e['delta_energy_median']}，mean={e['delta_energy_mean']}，P90={e['delta_energy_p90']}，max={e['delta_energy_max']}
- ΔE>0/0.5/1/3 比例：{e['fraction_delta_gt_0']} / {e['fraction_delta_gt_0_5']} / {e['fraction_delta_gt_1_0']} / {e['fraction_delta_gt_3_0']}

## C. Retry 价值

- 第二轮恢复 {r['recovered_count']} 条。
- retry-used 相对 round0-success 的 total-time mean/median 差：{r['mean_total_time_increment_vs_round0_success']} / {r['median_total_time_increment_vs_round0_success']} 秒。

## D. Stereo

- 有明确 E/Z 的样本 {s['samples_with_declared_double_bond_stereo']}，明确双键副本 {s['declared_double_bond_centers']}。
- 有明确 tetrahedral 的样本 {s['samples_with_declared_tetrahedral_stereo']}，明确四面体副本 {s['declared_tetrahedral_centers']}。
- 双键 pre/post candidate rejection：{s['candidate_double_bond_pre_rejections']} / {s['candidate_double_bond_post_rejections']}。
- tetrahedral pre/post candidate rejection：{s['candidate_tetra_pre_rejections']} / {s['candidate_tetra_post_rejections']}。
- 因双键/tetra Stereo 最终失败样本：{s['sample_final_failure_due_to_double_bond_stereo']} / {s['sample_final_failure_due_to_tetra_stereo']}。

## E. MMFF

- unsupported：{m['unsupported_samples']}。
- 至少一个 converged final-valid：{m['samples_with_at_least_one_converged_final_valid']}。
- 只有 non-converged final-valid：{m['samples_with_only_nonconverged_final_valid']}。
- 最终选择 non-converged：{m['samples_selected_nonconverged']}。
- 有 converged 但更低数值能量来自 non-converged：{m['samples_with_lower_nonconverged_energy_despite_converged_candidate']}。

## F. 显式 H

- N_all/N_heavy：{h['all_over_heavy_ratio']}。
- Spearman(total time, N_heavy/N_all)：{h['spearman_total_time_vs_heavy_atom_count']} / {h['spearman_total_time_vs_all_atom_count']}。

## G. 时间

- embedding：{t['embedding_time']}
- MMFF：{t['mmff_time']}
- Stereo：{t['stereo_time']}
- total：{t['sample_total_time']}
- round0-success total：{t['round0_success_total_time']}
- retry-used total：{t['retry_used_total_time']}

## H. 失败和旧几何对照

- 样本级失败：{report['sample_final_failure_counts']}
- 候选级拒绝：{report['candidate_rejection_counts']}
- 旧/新共同有效且完成可靠身份对齐：{report['old_geometry_comparison']['count']} 条。
- 旧 declared-coordinate Stereo conflict：{report['old_geometry_comparison']['old_declared_coordinate_stereo_conflicts']}；新：{report['old_geometry_comparison']['new_declared_coordinate_stereo_conflicts']}。
- RMSD/距离差只证明初始化与选样协议产生不同局部构象；不能解释为新构象更正确或更接近平衡态。

## 决策

**{decision}**

这是 300 条分层真实记录的 pilot，不是全量覆盖、模型性能或物理真实性结论。8 个候选是否最优、RandomCoords 是否优于普通初始化、以及下游性能均未验证。
"""


def main():
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing pilot output: {output}")
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for the mandated Parquet output") from exc
    output.mkdir(parents=True, exist_ok=False)
    samples = select_samples(args.source, args.downstream)
    config = {
        "git_commit": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_datasets": [str(args.source), str(args.downstream)],
        "selection_seed": SEED,
        "selection_rule": "historical deterministic v8 pilot256 plus 44 downstream hash-ranked unique real records; current data/raw contains no @/@@ records",
        "unavailable_selection_categories": [
            "C_declared_tetra_only", "D_double_and_tetra"
        ],
        "sample_count": len(samples),
        "rdkit_version": rdBase.rdkitVersion,
        "python_version": platform.python_version(),
        "argv": list(sys.argv),
        "trimer_content_schema": TRIMER_CONTENT_SCHEMA,
        "trimer_lmdb_schema": TRIMER_LMDB_SCHEMA,
        "protocol": TRIMER_PROTOCOL,
        "etkdg": {
            "variant": "ETKDGv3", "candidates_per_round": TRIMER_CANDIDATES_PER_ROUND,
            "max_rounds": TRIMER_MAX_ROUNDS, "useRandomCoords": True,
            "enforceChirality": True, "maxIterations": TRIMER_ETKDG_MAX_ITERATIONS,
            "pruneRmsThresh": -1.0, "per_embed_timeout_seconds": TRIMER_ETKDG_TIMEOUT_SECONDS,
        },
        "mmff": {"variant": "MMFF94", "maxIters": TRIMER_MMFF_RELAX_MAX_ITERATIONS},
        "sample_timeout_seconds": TRIMER_SAMPLE_TIMEOUT_SECONDS,
        "old_geometry_comparison_limit": 100,
    }
    write_json(output / "pilot_config.json", config)
    write_jsonl(output / "pilot_samples.jsonl", samples)
    summaries, candidates, failures, recovered = [], [], [], []
    stereo_rejections, nonconverged, comparisons = [], [], []
    old_store = LmdbLayerStore(OLD_ROOT / "cache")
    old_rows = json.loads((OLD_ROOT / "selection.json").read_text())
    old_ids = {sample_key_from_smiles(row["smiles"]): row for row in old_rows}
    try:
        for position, sample in enumerate(samples, 1):
            started = time.monotonic()
            canonical = sample["canonical_smiles"]
            ru = _compute_ru_base_layer(canonical)
            topology = _compute_topology_layer(canonical, ru, max_hops=2)
            topology.smiles = canonical
            if not bool(getattr(topology, "graph_available", False)):
                raise TrimerContractError("pilot_topology_unavailable")
            try:
                record = attach_finite_trimer_mcl(
                    topology, canonical, num_candidates=8, max_rounds=2,
                    timeout_seconds=TRIMER_SAMPLE_TIMEOUT_SECONDS,
                    sample_key=sample["sample_id"],
                )
            except TrimerContractError as exc:
                partial = {
                    "sample_id": sample["sample_id"], "canonical_smiles": canonical,
                    "geometry_valid": False, "stop_reason": "TRIMER_CONTRACT_ERROR",
                    "failure_category": "TRIMER_CONTRACT_ERROR", "error": str(exc),
                    "sample_total_time": time.monotonic() - started,
                }
                summaries.append(partial)
                write_jsonl(output / "pilot_partial_summary.jsonl", summaries)
                raise
            diagnostics = record.generation_diagnostics
            sample_candidates = expanded_candidates(sample, diagnostics)
            candidates.extend(sample_candidates)
            append_jsonl(
                output / "pilot_partial_candidate_stats.jsonl", sample_candidates
            )
            round0 = round_summary(sample_candidates, 0)
            round1 = round_summary(sample_candidates, 1)
            retry_used = round1["requested"] is not None
            heavy_count = int(diagnostics.get("trimer_heavy_atoms", 0))
            all_count = int(diagnostics.get("trimer_atoms_with_h", 0))
            round_map = {int(row["round_id"]): row for row in diagnostics.get("rounds", [])}
            summary = {
                "sample_id": sample["sample_id"], "canonical_smiles": canonical,
                "heavy_atom_count": heavy_count,
                "hydrogen_count": all_count - heavy_count if all_count else None,
                "all_atom_count": all_count or None,
                "all_over_heavy_ratio": all_count / heavy_count if heavy_count else None,
                "declared_double_bond_stereo_count": int(diagnostics.get("declared_double_bond_stereo_count", 0)),
                "declared_tetrahedral_stereo_count": int(diagnostics.get("declared_tetrahedral_stereo_count", 0)),
                **{f"round0_{key}": value for key, value in round0.items()},
                "retry_used": retry_used,
                **{f"round1_{key}": value for key, value in round1.items()},
                "selected_round": int(record.trimer_conformer_round_id) if record.trimer_geometry_valid else None,
                "selected_candidate_index": int(record.trimer_conformer_candidate_id) if record.trimer_geometry_valid else None,
                "selected_energy": float(record.trimer_conformer_energy) if record.trimer_geometry_valid else None,
                "selected_converged": bool(record.selected_converged) if record.trimer_geometry_valid else None,
                "stop_reason": str(record.search_stop_reason),
                "geometry_valid": bool(record.trimer_geometry_valid),
                "embedding_time_round0": round_map.get(0, {}).get("embed_time"),
                "mmff_time_round0": round_map.get(0, {}).get("mmff_time"),
                "stereo_time_round0": round_map.get(0, {}).get("stereo_time"),
                "embedding_time_round1": round_map.get(1, {}).get("embed_time"),
                "mmff_time_round1": round_map.get(1, {}).get("mmff_time"),
                "stereo_time_round1": round_map.get(1, {}).get("stereo_time"),
                "embedding_time": float(diagnostics.get("embed_time", 0.0)),
                "mmff_time": float(diagnostics.get("mmff_time", 0.0)),
                "stereo_time": float(diagnostics.get("stereo_time", 0.0)),
                "sample_total_time": float(diagnostics.get("total_time", time.monotonic() - started)),
            }
            summary["failure_category"] = None if summary["geometry_valid"] else classify_failure(summary, sample_candidates)
            summaries.append(summary)
            if summary["failure_category"]:
                failures.append(summary)
            if retry_used and round1["final_valid"] and round1["final_valid"] > 0:
                recovered.append(summary)
            if summary["geometry_valid"] and summary["selected_converged"] is False:
                nonconverged.append(summary)
            stereo_info = None
            for row in sample_candidates:
                if row["rejection_reason"] and "STEREO" in row["rejection_reason"]:
                    if stereo_info is None:
                        stereo_info = declared_stereo_info(canonical)
                    stereo_rejections.append({
                        **row,
                        "source_declared_double_count": sample["source_declared_double_count"],
                        "source_declared_tetra_count": sample["source_declared_tetra_count"],
                        "heavy_indices": record.trimer_heavy_indices if record.trimer_geometry_valid else None,
                        "h_parent_heavy_index": record.h_parent_heavy_index if record.trimer_geometry_valid else None,
                        "ru_offsets": record.trimer_ru_offset if record.trimer_geometry_valid else None,
                        "declared_stereo_info": stereo_info,
                    })
            if record.trimer_geometry_valid:
                independent = audit_frozen_stereo(record, canonical, topology=topology, return_details=True)
                if independent["double_bonds"] != summary["declared_double_bond_stereo_count"] or independent["tetrahedral_centers"] != summary["declared_tetrahedral_stereo_count"]:
                    raise TrimerContractError("independent_final_stereo_count_mismatch")
            key = sample_key_from_smiles(sample["smiles"])
            if len(comparisons) < 100 and key in old_ids and record.trimer_geometry_valid:
                comparison = old_geometry_comparison(sample, old_store[key], record)
                if comparison is not None:
                    comparisons.append(comparison)
            write_jsonl(output / "pilot_partial_summary.jsonl", summaries)
            print(json.dumps({
                "completed": position, "total": len(samples), "sample_id": sample["sample_id"],
                "valid": summary["geometry_valid"], "stop": summary["stop_reason"],
                "retry": retry_used, "seconds": summary["sample_total_time"],
            }, sort_keys=True), flush=True)
    finally:
        old_store.close()

    report = analyze(samples, summaries, candidates, comparisons)
    pd.DataFrame(candidates).to_parquet(output / "pilot_candidate_stats.parquet", index=False)
    pd.DataFrame(summaries).to_csv(output / "pilot_sample_summary.csv", index=False)
    write_jsonl(output / "pilot_failures.jsonl", failures)
    write_jsonl(output / "pilot_retry_recovered.jsonl", recovered)
    write_jsonl(output / "pilot_stereo_rejections.jsonl", stereo_rejections[:20])
    write_jsonl(output / "pilot_nonconverged_selected.jsonl", nonconverged)
    write_jsonl(output / "pilot_old_geometry_comparison.jsonl", comparisons)
    write_json(output / "pilot_report.json", report)
    (output / "pilot_report.md").write_text(report_markdown(report), encoding="utf-8")
    (output / "PILOT_COMPLETE").write_text(datetime.now(timezone.utc).isoformat() + "\n")
    print(json.dumps({"pilot_complete": True, **report["basic_success"]}, sort_keys=True))


if __name__ == "__main__":
    main()
