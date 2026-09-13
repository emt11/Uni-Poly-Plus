#!/usr/bin/env python3
"""Build only the bounded 16/256 Stage-A multi-conformer pilot cache."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.dataset import _compute_ru_base_layer, _compute_topology_layer
from src.dataset.graph_data import build_periodic_multimer_mol
from src.dataset.lmdb_cache import LmdbLayerStore, LmdbLayerWriter, sample_key_from_smiles
from src.dataset.trimer_mcl import (
    TrimerContractError, attach_finite_trimer_mcl,
    audit_double_bond_stereo_coordinates, fixed_identity_rmsd, iter_conformers,
)

SCHEMA = "mts-trimer-ensemble-stage-a"
ANOMALY = "*C1=C(/C=C/c2ccc(*)cc2)C=C1"


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("correctness16", "pilot256", "report"), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source", default="data/raw/PI1M_v2.csv")
    return parser.parse_args()


def _canonical(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    return None if mol is None else Chem.MolToSmiles(mol, canonical=True)


def _source_rows(source):
    paths = [Path(source)] + sorted(Path("data/raw").glob("smi_*.csv"))
    seen = set()
    rows = []
    for path in paths:
        frame = pd.read_csv(path, usecols=[0])
        for row, value in enumerate(frame.iloc[:, 0].astype(str)):
            value = str(value).strip()
            if value in seen:
                continue
            seen.add(value)
            rows.append({"smiles": value, "source": str(path), "source_row": row})
    return rows


def _categories(row, *, check_mmff=False):
    smiles = row["smiles"]
    mol = Chem.MolFromSmiles(smiles)
    categories = set()
    if smiles == ANOMALY:
        categories.add("known_stereo_anomaly")
    if smiles == "*O*":
        categories.add("n0_oxygen")
    if any(b.GetStereo() in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
                             Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS}
           for b in mol.GetBonds()):
        categories.add("explicit_stereo")
    else:
        categories.add("no_explicit_stereo")
    heavy = sum(a.GetAtomicNum() > 0 for a in mol.GetAtoms())
    if heavy >= 40:
        categories.add("large")
    if check_mmff:
        try:
            trimer, _ = build_periodic_multimer_mol(mol, 3, close_periodic=False)
            if AllChem.MMFFGetMoleculeProperties(Chem.AddHs(trimer), mmffVariant="MMFF94") is None:
                categories.add("mmff_unsupported")
        except Exception:
            categories.add("contract_candidate")
    return categories


def _selection(source):
    rows = _source_rows(source)
    by_smiles = {row["smiles"]: row for row in rows}
    required = []
    for smiles in (ANOMALY, "*O*"):
        if smiles not in by_smiles:
            raise RuntimeError(f"required real record missing: {smiles}")
        required.append(by_smiles[smiles])
    scored = heapq.nsmallest(
        1024, rows,
        key=lambda row: hashlib.sha256(row["smiles"].encode()).hexdigest())
    # Canonicalise only a bounded candidate pool, not the million-row source.
    probes, probe_ids = [], set()
    for raw in required + rows[:64] + scored:
        canonical = _canonical(raw["smiles"])
        if canonical is None or canonical in probe_ids:
            continue
        row = dict(raw); row["canonical"] = canonical
        probes.append(row); probe_ids.add(canonical)
    required = [next(row for row in probes if row["smiles"] == smiles)
                for smiles in (ANOMALY, "*O*")]
    basic = [(row, _categories(row))
             for row in probes]
    unsupported = next((row for row in probes[:66]
                        if "mmff_unsupported" in _categories(row, check_mmff=True)), None)
    if unsupported is None:
        raise RuntimeError("required real category missing: mmff_unsupported")
    annotated = basic + [(unsupported, {"mmff_unsupported"})]
    chosen = list(required)
    chosen_ids = {row["canonical"] for row in chosen}
    for category in ("mmff_unsupported", "explicit_stereo", "large", "no_explicit_stereo"):
        match = next((row for row, cats in annotated
                      if category in cats and row["canonical"] not in chosen_ids), None)
        if match is None:
            raise RuntimeError(f"required real category missing: {category}")
        chosen.append(match); chosen_ids.add(match["canonical"])
    for row, _ in basic:
        if len(chosen) >= 16:
            break
        if row["canonical"] not in chosen_ids:
            chosen.append(row); chosen_ids.add(row["canonical"])
    pilot = list(chosen)
    for row, _ in basic:
        if len(pilot) >= 256:
            break
        if row["canonical"] not in {item["canonical"] for item in pilot}:
            pilot.append(row)
    if len(chosen) != 16 or len(pilot) != 256:
        raise RuntimeError("could not select 16/256 unique real structures")
    return chosen, pilot


def _meta(size):
    return {"schema": SCHEMA, "multi_conformer": True,
            "stage": "A", "protocol": "ETKDGv3-8x4-MMFF94-RMSD0.3",
            "record_count": int(size)}


def _build(rows, root, reuse_root=None):
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "selection.json"
    manifest_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    writer = LmdbLayerWriter(root / "cache", _meta(len(rows)), commit_size=8)
    reuse = (LmdbLayerStore(Path(reuse_root) / "cache")
             if reuse_root is not None else None)
    try:
        for index, row in enumerate(rows):
            key = sample_key_from_smiles(row["smiles"])
            if key in writer:
                continue
            if reuse is not None and key in reuse:
                writer.add(key, reuse[key])
                print(json.dumps({"completed": index + 1, "total": len(rows),
                                  "smiles": row["smiles"], "reused": True},
                                 sort_keys=True), flush=True)
                continue
            ru = _compute_ru_base_layer(row["smiles"])
            topology = _compute_topology_layer(row["smiles"], ru, max_hops=2)
            if not bool(getattr(topology, "graph_available", False)):
                raise TrimerContractError(f"topology unavailable: {row['smiles']}")
            topology.smiles = row["smiles"]
            record = attach_finite_trimer_mcl(
                topology, Chem.MolFromSmiles(row["canonical"]),
                num_candidates=8, max_rounds=4, target_conformers=4,
                rmsd_threshold=0.3, timeout_seconds=240,
                sample_key=key.hex())
            writer.add(key, record)
            print(json.dumps({"completed": index + 1, "total": len(rows),
                              "smiles": row["smiles"], "K": record.num_conformers,
                              "stop": record.search_stop_reason,
                              "seconds": record.generation_diagnostics["total_time"]},
                             sort_keys=True), flush=True)
        store = writer.finalize(failure_count=0)
        store.close()
    except BaseException:
        writer.close()
        raise
    finally:
        if reuse is not None:
            reuse.close()


def _greedy_count(diag, cutoff):
    matrix = diag.get("valid_candidate_rmsd", [])
    kept = []
    for i in range(len(matrix)):
        if all(float(matrix[i][j]) >= cutoff for j in kept):
            kept.append(i)
            if len(kept) == 4:
                break
    return len(kept)


def _percentile(values, q):
    if not values:
        return None
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), q / 100.0))


def _report(root, rows):
    store = LmdbLayerStore(root / "cache")
    if store.schema != SCHEMA or not bool(store.meta.get("multi_conformer")):
        store.close()
        raise RuntimeError("pilot cache metadata is not multi-conformer Stage A")
    records = []
    audited_stereo_conformers = 0
    audited_stereo_bonds = 0
    try:
        for row in rows:
            record = store[sample_key_from_smiles(row["smiles"])]
            # Read API and final float32 payload are exercised for every conformer.
            views = list(iter_conformers(record))
            assert len(views) == int(record.num_conformers)
            rebuilt, meta = build_periodic_multimer_mol(
                Chem.MolFromSmiles(row["canonical"]), 3, close_periodic=False)
            expected_z = torch.tensor(
                [atom.GetAtomicNum() for atom in rebuilt.GetAtoms()], dtype=torch.long)
            if not torch.equal(expected_z, record.trimer_atomic_number):
                raise TrimerContractError("cached atomic identity differs from independent rebuild")
            expected_edges = {tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
                              for bond in rebuilt.GetBonds()}
            cached_edges = {tuple(sorted((int(record.trimer_edge_index[0, col]),
                                          int(record.trimer_edge_index[1, col]))))
                            for col in range(record.trimer_edge_index.size(1))}
            if expected_edges != cached_edges:
                raise TrimerContractError("cached physical bonds differ from independent rebuild")
            for view in views:
                if view.trimer_pos.dtype != torch.float32:
                    raise RuntimeError("final cached coordinates are not float32")
                checked = audit_double_bond_stereo_coordinates(rebuilt, view.trimer_pos)
                audited_stereo_conformers += int(checked > 0)
                audited_stereo_bonds += checked
            records.append((row, record))
    finally:
        store.close()
    ks = [int(record.num_conformers) for _, record in records]
    times = [float(record.generation_diagnostics["total_time"]) for _, record in records]
    atom_counts = [int(record.generation_diagnostics["trimer_heavy_atoms"]) for _, record in records]
    diversities, delta_energies = [], []
    for _, record in records:
        for i in range(record.num_conformers):
            for j in range(i):
                diversities.append(fixed_identity_rmsd(record.conformer_positions[i], record.conformer_positions[j]))
        if record.num_conformers:
            energies = record.conformer_energies.double()
            delta_energies.extend((energies - energies.min()).tolist())
    anomaly = next(record for row, record in records if row["smiles"] == ANOMALY)
    result = {
        "scope": f"{len(rows)} specified real records only",
        "source_distribution": dict(sorted(Counter(row["source"] for row, _ in records).items())),
        "selection_note": "deterministic combined-source sample; not task-balanced",
        "cache_root": str((root / "cache").resolve()),
        "K_distribution": dict(sorted(Counter(ks).items())),
        "target_met": sum(k == 4 for k in ks),
        "stop_reasons": dict(sorted(Counter(r.search_stop_reason for _, r in records).items())),
        "round_target_attainment": dict(sorted(Counter(
            int(r.conformer_round_ids.max()) + 1 if r.num_conformers == 4 else 0
            for _, r in records).items())),
        "rejections": {name: sum(int(r.generation_diagnostics[name]) for _, r in records)
                       for name in ("num_pre_stereo_rejected", "num_post_stereo_rejected",
                                    "num_geometry_rejected", "num_duplicate_rejected")},
        "offline_K_distribution_within_generated_pool": {str(cutoff): dict(sorted(Counter(
            _greedy_count(r.generation_diagnostics, cutoff) for _, r in records).items()))
            for cutoff in (0.3, 0.5, 1.0)},
        "pairwise_RMSD": {"p50": _percentile(diversities, 50), "p90": _percentile(diversities, 90)},
        "delta_energy": {"p50": _percentile(delta_energies, 50), "p90": _percentile(delta_energies, 90)},
        "runtime_seconds": {f"p{q}": _percentile(times, q) for q in (50, 90, 95, 100)},
        "timeouts": sum(r.search_stop_reason == "timeout" for _, r in records),
        "cache_bytes": sum(p.stat().st_size for p in (root / "cache").rglob("*") if p.is_file()),
        "atom_count": {"min": min(atom_counts), "max": max(atom_counts),
                       "runtime_correlation": float(torch.corrcoef(torch.tensor([atom_counts, times], dtype=torch.float64))[0, 1])},
        "known_anomaly_final_stereo_conformers": int(anomaly.num_conformers),
        "independent_final_stereo_audit": {
            "conformers_with_explicit_stereo": audited_stereo_conformers,
            "stereo_bonds_checked": audited_stereo_bonds,
        },
    }
    path = root / "report.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


def main():
    args = _arguments()
    correctness, pilot = _selection(args.source)
    root = Path(args.output_root)
    if args.phase == "correctness16":
        _build(correctness, root / "correctness16")
        _report(root / "correctness16", correctness)
    elif args.phase == "pilot256":
        # Exact protocol reuse is by copying already serialized records into the
        # independent 256-record cache, never by regenerating those 16 records.
        _build(pilot, root / "pilot256", reuse_root=root / "correctness16")
        _report(root / "pilot256", pilot)
    else:
        _report(root / "pilot256", pilot)


if __name__ == "__main__":
    main()
