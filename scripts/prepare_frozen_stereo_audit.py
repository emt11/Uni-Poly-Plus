#!/usr/bin/env python3
"""Prepare an immutable accepted Stereo cohort for independent cache audit."""

from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from rdkit import Chem

from src.dataset.cache_lifecycle import (CacheLifecycleError, atomic_json,
                                          zero_write_snapshot)
from src.dataset.glt_dual_cache import load_active_dual_store


DEFINED_BOND = {
    Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS,
}
DEFINED_ATOM = {
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
}


def _worker_init():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)


def _classify(row):
    smiles = str(row["normalized_smiles"])
    # Fast conservative rejection.  Slash/backslash encode directional bond
    # information and @ encodes atom chirality in isomeric SMILES.
    if not any(marker in smiles for marker in ("/", "\\", "@")):
        return row, 0, 0, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return row, 0, 0, "normalized Stereo source cannot be parsed"
    bonds = sum(bond.GetStereo() in DEFINED_BOND for bond in mol.GetBonds())
    atoms = sum(atom.GetChiralTag() in DEFINED_ATOM for atom in mol.GetAtoms())
    return row, int(bonds), int(atoms), None


def build(cache_root: Path, output: Path, workers: int) -> dict:
    cache_root, output = cache_root.resolve(), output.resolve()
    if output.exists() or output.with_name(output.name + ".staging").exists():
        raise FileExistsError(f"Stereo audit cohort already exists: {output}")
    before = zero_write_snapshot(cache_root)
    store = load_active_dual_store(cache_root)
    bundle_root = cache_root / "builds" / store["bundle_hash"]
    accepted_array = np.load(bundle_root / "trimer" / "accepted_keys.npy", mmap_mode="r")
    accepted = {bytes(np.asarray(row, dtype=np.uint8)) for row in accepted_array}
    fallback_path = bundle_root / "trimer" / "fallbacks.jsonl"
    fallback = set()
    if fallback_path.is_file():
        for line in fallback_path.read_text(encoding="utf-8").splitlines():
            if line:
                fallback.add(bytes.fromhex(json.loads(line)["sample_key"]))
    source_path = cache_root / store["source"]["path"] / "records.jsonl"
    rows = []
    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if bytes.fromhex(row["sample_key"]) in accepted:
                rows.append(row)
    if len(rows) != len(accepted):
        raise CacheLifecycleError("Stereo source/accepted coverage mismatch")
    staging = output.with_name(output.name + ".staging")
    staging.mkdir(parents=True)
    selected, fallback_selected, errors = [], [], []
    counts = Counter()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, initializer=_worker_init
    ) as executor:
        for row, bonds, atoms, error in executor.map(
            _classify, rows, chunksize=256
        ):
            if error:
                errors.append({"sample_key": row["sample_key"], "error": error})
                continue
            if not bonds and not atoms:
                continue
            counts["structures"] += 1
            counts["defined_double_bonds"] += bonds
            counts["defined_tetrahedral_centers"] += atoms
            entry = {
                "sample_key": row["sample_key"], "status": "accepted",
                "declared_double_bonds": bonds,
                "declared_tetrahedral_centers": atoms,
            }
            if bytes.fromhex(row["sample_key"]) in fallback:
                fallback_selected.append({**entry, "fallback": True})
            else:
                selected.append(entry)
    if errors:
        raise CacheLifecycleError(f"Stereo cohort classification errors: {len(errors)}")
    selected_by_key = {row["sample_key"]: row for row in selected}
    runtime_path = bundle_root / "trimer" / "runtime.jsonl"
    runtime_found = set()
    with runtime_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            runtime = json.loads(line)
            key_hex = runtime["sample_key"]
            if key_hex not in selected_by_key:
                continue
            if runtime.get("status") != "accepted":
                raise CacheLifecycleError(
                    "coordinate Stereo cohort contains a non-accepted runtime row"
                )
            selected_by_key[key_hex].update(runtime)
            runtime_found.add(key_hex)
    if runtime_found != set(selected_by_key):
        raise CacheLifecycleError("Stereo cohort/runtime ledger coverage mismatch")
    def write_jsonl(name, values):
        with (staging / name).open("w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, sort_keys=True) + "\n")
    selected_keys = {row["sample_key"] for row in selected}
    write_jsonl("accepted_keys.jsonl", selected)
    write_jsonl("source_rows.jsonl", [
        row for row in rows if row["sample_key"] in selected_keys
    ])
    write_jsonl("stereo_geometry_fallbacks.jsonl", fallback_selected)
    snapshot = {
        "scope": "all accepted structures with explicit source Stereo and usable geometry",
        "staging_path": str(bundle_root.resolve()),
        "staging_bundle_hash": store["bundle_hash"],
        "main_bundle_hash": store["bundle_hash"],
        "source_manifest_hash": store["source"]["source_manifest_hash"],
        "accepted_structure_count": len(rows),
        "stereo_structure_count": int(counts["structures"]),
        "coordinate_audit_count": len(selected),
        "geometry_fallback_stereo_count": len(fallback_selected),
        "declared_double_bonds": int(counts["defined_double_bonds"]),
        "declared_tetrahedral_centers": int(counts["defined_tetrahedral_centers"]),
        "workers": int(workers),
        "cache_zero_write": zero_write_snapshot(cache_root) == before,
    }
    if not snapshot["cache_zero_write"]:
        raise CacheLifecycleError("Stereo cohort preparation modified frozen cache")
    atomic_json(staging / "snapshot.json", snapshot)
    os.replace(staging, output)
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    result = build(Path(args.cache_root), Path(args.output), int(args.workers))
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
