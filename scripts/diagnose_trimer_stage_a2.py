#!/usr/bin/env python3
"""Historical v8 Stage-A2 ensemble diagnostic (retired)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, Lipinski, rdDistGeom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.dataset import _compute_ru_base_layer, _compute_topology_layer
from src.dataset.graph_data import build_periodic_multimer_mol
from src.dataset.lmdb_cache import LmdbLayerStore, sample_key_from_smiles
from src.dataset.trimer_mcl import (
    TRIMER_MMFF_RELAX_MAX_ITERATIONS,
    TrimerContractError,
    _ExpectedGeometryFailure,
    _calculate_mmff_energy,
    _conformer_coordinates_are_finite_3d,
    _coordinates,
    _round_seed,
    attach_finite_trimer_mcl,
    audit_double_bond_stereo_coordinates,
)


FAILURE_NAMES = {
    int(getattr(rdDistGeom.EmbedFailureCauses, name)): name
    for name in dir(rdDistGeom.EmbedFailureCauses) if name.isupper()
}
EXPLICIT_STEREO = {
    Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--probe-count", type=int, default=16)
    return parser.parse_args()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def embed_attempt(molecule, seed, num_candidates=8):
    candidate = Chem.Mol(molecule)
    candidate.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = 1
    params.useRandomCoords = False
    params.enforceChirality = True
    params.maxIterations = 200
    params.timeout = 60
    params.pruneRmsThresh = -1.0
    params.trackFailures = True
    embedded = AllChem.EmbedMultipleConfs(candidate, numConfs=int(num_candidates), params=params)
    ids = [int(conf_id) for conf_id in embedded if int(conf_id) >= 0]
    counts = params.GetFailureCounts()
    failures = {FAILURE_NAMES.get(index, f"UNKNOWN_{index}"): int(value)
                for index, value in enumerate(counts) if value}
    return candidate, ids, failures


def capped_ru(normalized):
    editable = Chem.RWMol(Chem.Mol(normalized))
    dummies = [atom.GetIdx() for atom in editable.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummies) != 2:
        raise TrimerContractError("normalized RU does not have exactly two attachments")
    for index in dummies:
        atom = editable.GetAtomWithIdx(index)
        if atom.GetDegree() != 1:
            raise TrimerContractError("attachment dummy is not terminal")
        atom.SetAtomicNum(1)
        atom.SetNoImplicit(True)
    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule)
    return molecule


def finite_embed_summary(molecule, seed):
    embedded, ids, failures = embed_attempt(Chem.AddHs(Chem.Mol(molecule)), seed)
    finite = sum(_conformer_coordinates_are_finite_3d(
        embedded, conf_id, embedded.GetNumAtoms()) for conf_id in ids)
    return {"requested": 8, "embedded": len(ids), "finite": int(finite),
            "success": bool(finite), "rdkit_failure_counts": failures}


def replay_full(normalized, sample_key):
    trimer, metadata = build_periodic_multimer_mol(normalized, 3, close_periodic=False)
    heavy_count = trimer.GetNumAtoms()
    mol_h_template = Chem.AddHs(Chem.Mol(trimer))
    if AllChem.MMFFGetMoleculeProperties(mol_h_template, mmffVariant="MMFF94") is None:
        raise TrimerContractError("failure cohort unexpectedly lacks MMFF94 parameters")
    rounds = []
    valid_candidates = []
    for round_id in range(4):
        seed = _round_seed(sample_key, round_id)
        mol_h, ids, rdkit_failures = embed_attempt(mol_h_template, seed)
        row = {
            "round_id": round_id, "seed": seed, "requested_candidates": 8,
            "embedded_candidates": len(ids), "finite_candidates": 0,
            "pre_stereo_pass": 0, "pre_stereo_fail": 0,
            "pre_geometry_pass": 0, "pre_geometry_fail": 0,
            "mmff_attempted_count": 0, "finite_mmff_energy_count": 0,
            "post_stereo_pass": 0, "post_stereo_fail": 0,
            "post_geometry_pass": 0, "post_geometry_fail": 0,
            "final_valid_count": 0, "rdkit_failure_counts": rdkit_failures,
        }
        pre_valid = []
        for candidate_id, conf_id in enumerate(ids):
            if not _conformer_coordinates_are_finite_3d(mol_h, conf_id, mol_h.GetNumAtoms()):
                row["pre_geometry_fail"] += 1
                continue
            row["finite_candidates"] += 1
            row["pre_geometry_pass"] += 1
            try:
                audit_double_bond_stereo_coordinates(
                    trimer, _coordinates(mol_h, conf_id, heavy_count))
            except _ExpectedGeometryFailure:
                row["pre_stereo_fail"] += 1
                continue
            row["pre_stereo_pass"] += 1
            pre_valid.append((candidate_id, conf_id))
        if pre_valid:
            valid_ids = {conf_id for _, conf_id in pre_valid}
            for conf in list(mol_h.GetConformers()):
                if conf.GetId() not in valid_ids:
                    mol_h.RemoveConformer(conf.GetId())
            row["mmff_attempted_count"] = len(pre_valid)
            AllChem.MMFFOptimizeMoleculeConfs(
                mol_h, numThreads=1, maxIters=TRIMER_MMFF_RELAX_MAX_ITERATIONS,
                mmffVariant="MMFF94")
            properties = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94")
            if properties is None:
                raise TrimerContractError("MMFF parameters changed during replay")
            heavy = Chem.RemoveHs(mol_h)
            if heavy.GetNumAtoms() != heavy_count:
                raise TrimerContractError("RemoveHs changed heavy-atom identity")
            for candidate_id, conf_id in pre_valid:
                energy = _calculate_mmff_energy(mol_h, properties, conf_id)
                if energy is None:
                    row["post_geometry_fail"] += 1
                    continue
                row["finite_mmff_energy_count"] += 1
                if not _conformer_coordinates_are_finite_3d(heavy, conf_id, heavy_count):
                    row["post_geometry_fail"] += 1
                    continue
                xyz = _coordinates(heavy, conf_id, heavy_count).to(torch.float32)
                row["post_geometry_pass"] += 1
                try:
                    audit_double_bond_stereo_coordinates(trimer, xyz)
                except _ExpectedGeometryFailure:
                    row["post_stereo_fail"] += 1
                    continue
                row["post_stereo_pass"] += 1
                row["final_valid_count"] += 1
                valid_candidates.append(_EnsembleCandidate(
                    xyz, float(energy), round_id, candidate_id))
        rounds.append(row)
    return trimer, metadata, rounds, valid_candidates


def primary_category(rounds):
    totals = Counter()
    for row in rounds:
        for key, value in row.items():
            if isinstance(value, int):
                totals[key] += value
    embedded = totals["embedded_candidates"]
    if embedded == 0:
        return "EMBED_ZERO"
    if totals["finite_candidates"] == 0:
        return "EMBEDDED_ALL_NONFINITE"
    if totals["pre_stereo_pass"] == 0 and totals["pre_stereo_fail"]:
        return "EMBEDDED_ALL_PRE_STEREO_FAIL"
    if totals["pre_geometry_pass"] == 0 and totals["pre_geometry_fail"]:
        return "EMBEDDED_ALL_PRE_GEOMETRY_FAIL"
    if (totals["post_stereo_pass"] == 0 and totals["post_stereo_fail"]
            and totals["post_geometry_pass"]):
        return "POST_MMFF_ALL_STEREO_FAIL"
    if (totals["final_valid_count"] == 0 and totals["post_geometry_fail"]
            and totals["post_stereo_fail"] == 0):
        return "POST_MMFF_ALL_GEOMETRY_FAIL"
    rejection_kinds = sum(bool(totals[key]) for key in (
        "pre_geometry_fail", "pre_stereo_fail", "post_geometry_fail", "post_stereo_fail"))
    if totals["final_valid_count"] == 0 and rejection_kinds > 1:
        return "MIXED_REJECTIONS"
    return "OTHER"


def structure_metrics(normalized):
    trimer, metadata = build_periodic_multimer_mol(normalized, 3, close_periodic=False)
    ru_atoms = [atom for atom in normalized.GetAtoms() if atom.GetAtomicNum() > 0]
    ring_sizes = [len(ring) for ring in trimer.GetRingInfo().AtomRings()]
    dummies = [atom for atom in normalized.GetAtoms() if atom.GetAtomicNum() == 0]
    attachment_atoms, attachment_bonds = [], []
    for dummy in dummies:
        if dummy.GetDegree() != 1:
            raise TrimerContractError("nonterminal attachment during structural analysis")
        neighbor = dummy.GetNeighbors()[0]
        attachment_atoms.append(neighbor.GetSymbol())
        attachment_bonds.append(str(normalized.GetBondBetweenAtoms(
            dummy.GetIdx(), neighbor.GetIdx()).GetBondType()))
    cross_types = []
    for left, right in metadata["inter_unit_edges"]:
        bond = trimer.GetBondBetweenAtoms(int(left), int(right))
        if bond is None:
            raise TrimerContractError("missing cross-RU bond during structural analysis")
        cross_types.append(str(bond.GetBondType()))
    return {
        "ru_heavy_atoms": len(ru_atoms),
        "trimer_heavy_atoms": trimer.GetNumAtoms(),
        "trimer_atoms_with_h": Chem.AddHs(trimer).GetNumAtoms(),
        "physical_bond_count": trimer.GetNumBonds(),
        "rotatable_bond_count": int(Lipinski.NumRotatableBonds(trimer)),
        "ring_count": trimer.GetRingInfo().NumRings(),
        "macrocycle": any(size >= 12 for size in ring_sizes),
        "aromatic_atom_fraction": (sum(atom.GetIsAromatic() for atom in trimer.GetAtoms())
                                    / max(1, trimer.GetNumAtoms())),
        "explicit_ez_bond_count": sum(bond.GetStereo() in EXPLICIT_STEREO
                                      for bond in trimer.GetBonds()),
        "formal_charge": sum(atom.GetFormalCharge() for atom in trimer.GetAtoms()),
        "attachment_atom_types": attachment_atoms,
        "attachment_bond_types": attachment_bonds,
        "cross_ru_bond_types": cross_types,
    }


def numeric_summary(rows, field):
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    return {"count": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)), "p25": float(np.quantile(values, .25)),
            "p75": float(np.quantile(values, .75)), "min": float(values.min()),
            "max": float(values.max())}


def size_bucket(value):
    if value < 30: return "<30"
    if value < 60: return "30-59"
    if value < 90: return "60-89"
    if value < 120: return "90-119"
    return ">=120"


def compare_groups(rows):
    failure = [row for row in rows if row["group"] == "A_ETKDG_FAILURE"]
    success = [row for row in rows if row["group"] == "B_K_GE_1"]
    numeric_fields = (
        "ru_heavy_atoms", "trimer_heavy_atoms", "trimer_atoms_with_h",
        "physical_bond_count", "rotatable_bond_count", "ring_count",
        "aromatic_atom_fraction", "explicit_ez_bond_count", "formal_charge")
    summary = {group: {field: numeric_summary(values, field) for field in numeric_fields}
               for group, values in (("failure", failure), ("success", success))}
    summary["categorical"] = {}
    for field in ("macrocycle", "attachment_atom_types", "attachment_bond_types",
                  "cross_ru_bond_types"):
        summary["categorical"][field] = {
            group: dict(Counter(json.dumps(row[field], sort_keys=True) for row in values))
            for group, values in (("failure", failure), ("success", success))}
    buckets = defaultdict(lambda: {"total": 0, "failed": 0})
    for row in rows:
        bucket = size_bucket(row["trimer_heavy_atoms"])
        buckets[bucket]["total"] += 1
        buckets[bucket]["failed"] += int(row["group"] == "A_ETKDG_FAILURE")
    summary["trimer_heavy_atom_failure_rate"] = {
        bucket: {**counts, "failure_rate": counts["failed"] / counts["total"]}
        for bucket, counts in buckets.items()}
    return summary


def deterministic_replay(rows, store):
    selections = {}
    anomaly = next(row for row in rows if row["smiles"] == "*C1=C(/C=C/c2ccc(*)cc2)C=C1")
    selections["known_stereo"] = anomaly
    for label, predicate in (
        ("k4", lambda record: record.num_conformers == 4),
        ("k1_3", lambda record: 1 <= record.num_conformers <= 3),
        ("embed_failure", lambda record: record.search_stop_reason == "ETKDG_NO_VALID_CONFORMER"),
    ):
        candidates = [row for row in rows if predicate(store[sample_key_from_smiles(row["smiles"])])
                      and row["smiles"] != anomaly["smiles"]]
        selections[label] = min(candidates, key=lambda row: hashlib.sha256(
            row["canonical"].encode()).hexdigest())
    output = []
    for role, row in selections.items():
        runs = []
        key = sample_key_from_smiles(row["smiles"])
        for _ in range(2):
            ru = _compute_ru_base_layer(row["smiles"])
            topology = _compute_topology_layer(row["smiles"], ru, max_hops=2)
            result = attach_finite_trimer_mcl(
                topology, Chem.MolFromSmiles(row["canonical"]), num_candidates=8,
                max_rounds=4, target_conformers=4, rmsd_threshold=.3,
                timeout_seconds=240, sample_key=key.hex())
            runs.append(result)
        first, second = runs
        energy_diff = (float(torch.max(torch.abs(first.conformer_energies - second.conformer_energies)))
                       if first.num_conformers == second.num_conformers and first.num_conformers else 0.0)
        coordinate_rmsd = ([fixed_identity_rmsd(first.conformer_positions[i], second.conformer_positions[i])
                            for i in range(first.num_conformers)]
                           if first.num_conformers == second.num_conformers else [])
        seeds = [[_round_seed(key.hex(), i) for i in range(4)] for _ in range(2)]
        passed = (seeds[0] == seeds[1]
                  and first.generation_diagnostics["num_candidates_embedded"]
                  == second.generation_diagnostics["num_candidates_embedded"]
                  and first.num_conformers == second.num_conformers
                  and torch.equal(first.conformer_round_ids, second.conformer_round_ids)
                  and torch.equal(first.conformer_candidate_ids, second.conformer_candidate_ids)
                  and energy_diff <= 1e-6 and all(value <= 1e-6 for value in coordinate_rmsd))
        output.append({"role": role, "smiles": row["smiles"], "round_seeds": seeds[0],
                       "embedded_counts_total": [r.generation_diagnostics["num_candidates_embedded"] for r in runs],
                       "final_k": [r.num_conformers for r in runs],
                       "round_ids_equal": torch.equal(first.conformer_round_ids, second.conformer_round_ids),
                       "candidate_ids_equal": torch.equal(first.conformer_candidate_ids, second.conformer_candidate_ids),
                       "max_energy_abs_diff": energy_diff,
                       "fixed_identity_rmsd": coordinate_rmsd, "pass": passed})
    return output


def choose_probes(failures, count):
    bucket_order = {"<30": 0, "30-59": 1, "60-89": 2, "90-119": 3, ">=120": 4}
    ordered = sorted(failures, key=lambda row: (
        bucket_order[size_bucket(row["trimer_heavy_atoms"])], row["primary_failure_category"],
        hashlib.sha256(row["canonical"].encode()).hexdigest()))
    selected, signatures = [], set()
    # Reserve at least one representative from every populated size bucket.
    for bucket in bucket_order:
        match = next((row for row in ordered
                      if size_bucket(row["trimer_heavy_atoms"]) == bucket), None)
        if match is not None:
            selected.append(match)
    for row in ordered:
        signature = (size_bucket(row["trimer_heavy_atoms"]),
                     row["primary_failure_category"], row["ring_count"] > 0,
                     row["rotatable_bond_count"] >= 20,
                     row["explicit_ez_bond_count"] > 0,
                     tuple(row["attachment_atom_types"]))
        if row not in selected and signature not in signatures:
            selected.append(row); signatures.add(signature)
            if len(selected) == count: return selected
    for row in ordered:
        if row not in selected:
            selected.append(row)
            if len(selected) == count: break
    return selected


def main():
    args = parse_args()
    raise RuntimeError(
        "retired v8 ensemble diagnostic: it is incompatible with the active "
        "single-conformer all-atom v9 contract"
    )
    pilot_root = Path(args.pilot_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    selection = json.loads((pilot_root / "selection.json").read_text())
    store = LmdbLayerStore(pilot_root / "cache")
    started = time.time()
    try:
        all_metrics, failures = [], []
        for index, row in enumerate(selection, 1):
            record = store[sample_key_from_smiles(row["smiles"])]
            normalized = Chem.MolFromSmiles(row["canonical"])
            metrics = structure_metrics(normalized)
            group = ("A_ETKDG_FAILURE" if record.search_stop_reason == "ETKDG_NO_VALID_CONFORMER"
                     else ("B_K_GE_1" if record.num_conformers >= 1 else "OTHER_K0"))
            metric_row = {**row, **metrics, "num_conformers": int(record.num_conformers),
                          "search_stop_reason": record.search_stop_reason, "group": group}
            all_metrics.append(metric_row)
            if group != "A_ETKDG_FAILURE":
                continue
            key = sample_key_from_smiles(row["smiles"]).hex()
            trimer, metadata, rounds, candidates = replay_full(normalized, key)
            category = primary_category(rounds)
            if candidates:
                raise TrimerContractError(
                    f"Stage A failure replay produced valid candidates for {row['smiles']}")
            ru_result = finite_embed_summary(capped_ru(normalized), _round_seed(key, 0))
            full_result = {"requested": 32,
                           "embedded": sum(item["embedded_candidates"] for item in rounds),
                           "finite": sum(item["finite_candidates"] for item in rounds),
                           "success": any(item["finite_candidates"] for item in rounds)}
            failure = {**metric_row, "primary_failure_category": category,
                       "rounds": rounds, "ru_embedding": ru_result,
                       "full_trimer_embedding": full_result}
            failures.append(failure)
            print(json.dumps({"failure_completed": len(failures), "failure_total": 81,
                              "sample_index": index, "category": category,
                              "ru_success": ru_result["success"]}, sort_keys=True), flush=True)
        if len(failures) != 81:
            raise RuntimeError(f"expected 81 ETKDG failures, found {len(failures)}")
        categories = Counter(row["primary_failure_category"] for row in failures)
        contingency = Counter()
        for row in failures:
            contingency[(row["ru_embedding"]["success"],
                         row["full_trimer_embedding"]["success"])] += 1
        comparison = compare_groups([
            row for row in all_metrics if row["group"] in {"A_ETKDG_FAILURE", "B_K_GE_1"}])
        probes = choose_probes(failures, int(args.probe_count))
        replay = deterministic_replay(selection, store)
        breakdown = {
            "scope": "81 Stage-A pilot records labeled ETKDG_NO_VALID_CONFORMER",
            "category_counts": dict(categories),
            "category_percentages": {key: value / 81 for key, value in categories.items()},
            "rdkit_version": rdBase.rdkitVersion,
            "rdkit_failure_tracking_available": True,
            "records": failures,
        }
        atomic_json(output / "stage_a2_failure_breakdown.json", breakdown)
        with (output / "stage_a2_failure_breakdown.csv").open("w", newline="") as handle:
            fields = ["smiles", "canonical", "source", "source_row",
                      "primary_failure_category", "ru_heavy_atoms", "trimer_heavy_atoms",
                      "trimer_atoms_with_h", "rotatable_bond_count", "ring_count",
                      "macrocycle", "explicit_ez_bond_count", "formal_charge",
                      "attachment_atom_types", "attachment_bond_types", "cross_ru_bond_types",
                      "ru_embed_success", "ru_embedded", "full_trimer_embedded"]
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for row in failures:
                writer.writerow({field: (row["ru_embedding"]["success"] if field == "ru_embed_success"
                                         else row["ru_embedding"]["embedded"] if field == "ru_embedded"
                                         else row["full_trimer_embedding"]["embedded"] if field == "full_trimer_embedded"
                                         else json.dumps(row[field]) if isinstance(row.get(field), list)
                                         else row.get(field)) for field in fields})
        atomic_json(output / "stage_a2_structure_comparison.json", comparison)
        atomic_json(output / "stage_a2_detailed_probes.json", probes)
        atomic_json(output / "stage_a2_deterministic_replay.json", replay)
        summary = {
            "elapsed_seconds": time.time() - started,
            "failure_categories": dict(categories),
            "ru_vs_full_trimer": {
                f"RU_{'PASS' if ru else 'FAIL'}__TRIMER_{'PASS' if full else 'FAIL'}": count
                for (ru, full), count in contingency.items()},
            "structure_comparison": comparison,
            "deterministic_replay_pass": all(item["pass"] for item in replay),
            "probe_count": len(probes),
            "stage_b_started": False,
        }
        atomic_json(output / "stage_a2_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    finally:
        store.close()


if __name__ == "__main__":
    main()
