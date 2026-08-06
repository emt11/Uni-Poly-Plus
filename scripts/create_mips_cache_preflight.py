#!/usr/bin/env python3
"""Create deterministic, nested correctness/performance cache cohorts.

The 500-row cohort deliberately covers attachment corner cases.  The 10k
cohort contains all 500 rows and is filled by a stable SHA256 ordering, so the
same source CSV always produces the same preflight inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, rdBase


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/raw/PI1M_v2.csv")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--correctness-size", type=int, default=500)
    parser.add_argument("--performance-size", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=48)
    return parser.parse_args()


def stable_score(smiles):
    return hashlib.sha256(str(smiles).strip().encode("utf-8")).hexdigest()


def classify(smiles):
    molecule = Chem.MolFromSmiles(str(smiles).strip())
    if molecule is None:
        return {"invalid"}
    dummy = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy) != 2 or any(atom.GetDegree() != 1 for atom in dummy):
        return {"attachment_invalid"}
    neighbors = [atom.GetNeighbors()[0].GetIdx() for atom in dummy]
    bond_types = [
        molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbors[index]).GetBondType()
        for index, atom in enumerate(dummy)
    ]
    categories = {"valid"}
    if neighbors[0] == neighbors[1]:
        categories.add("shared_boundary")
    elif bond_types[0] != bond_types[1]:
        categories.add("bond_mismatch")
    heavy = sum(atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms())
    if heavy >= 64:
        categories.add("large_ru")
    if any(atom.GetIsAromatic() for atom in molecule.GetAtoms()):
        categories.add("aromatic")
    if any(
        atom.GetAtomicNum() > 0 and atom.GetDegree() >= 3
        for atom in molecule.GetAtoms()
    ):
        categories.add("branched")
    return categories


def classify_entry(item):
    index, smiles = item
    return stable_score(smiles), index, classify(smiles)


def initialize_worker():
    rdBase.DisableLog("rdApp.*")


def main():
    args = parse_args()
    source = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source)
    smiles_column = frame.columns[0]
    rows = []
    counts = Counter()
    started = time.monotonic()
    last_report = started
    try:
        context = mp.get_context("spawn")
        values = enumerate(frame[smiles_column].astype(str))
        with context.Pool(
            processes=max(1, int(args.workers)),
            initializer=initialize_worker,
        ) as pool:
            for completed, result in enumerate(
                pool.imap_unordered(classify_entry, values, chunksize=256),
                start=1,
            ):
                score, index, categories = result
                counts.update(categories)
                rows.append((score, index, categories))
                now = time.monotonic()
                if now - last_report >= 5.0:
                    rate = completed / max(now - started, 1e-9)
                    eta = (len(frame) - completed) / max(rate, 1e-9)
                    print(
                        "[preflight_cohort] "
                        f"completed={completed}/{len(frame)} "
                        f"rate={rate:.1f}/s eta_min={eta / 60.0:.1f}",
                        flush=True,
                    )
                    last_report = now
    finally:
        rdBase.EnableLog("rdApp.*")
    rows.sort()

    quotas = (
        ("invalid", 25),
        ("attachment_invalid", 25),
        ("shared_boundary", 75),
        ("bond_mismatch", 75),
        ("large_ru", 50),
        ("aromatic", 75),
        ("branched", 75),
    )
    selected = []
    selected_set = set()
    quota_counts = {}
    for category, quota in quotas:
        matches = [
            row_index for _, row_index, categories in rows
            if category in categories and row_index not in selected_set
        ][:quota]
        selected.extend(matches)
        selected_set.update(matches)
        quota_counts[category] = len(matches)
    for _, row_index, _ in rows:
        if len(selected) >= args.correctness_size:
            break
        if row_index not in selected_set:
            selected.append(row_index)
            selected_set.add(row_index)
    if len(selected) != args.correctness_size:
        raise RuntimeError("source does not contain enough unique correctness rows")

    performance = list(selected)
    performance_set = set(selected)
    for _, row_index, _ in rows:
        if len(performance) >= args.performance_size:
            break
        if row_index not in performance_set:
            performance.append(row_index)
            performance_set.add(row_index)
    if len(performance) != args.performance_size:
        raise RuntimeError("source does not contain enough performance rows")

    # Stable hash order, not category assembly order, makes output diffs clear.
    rank = {row_index: position for position, (_, row_index, _) in enumerate(rows)}
    selected.sort(key=rank.__getitem__)
    performance.sort(key=rank.__getitem__)
    outputs = {
        "PI1M_preflight500.csv": selected,
        "PI1M_preflight10k.csv": performance,
    }
    for name, indices in outputs.items():
        temporary = output_dir / f"{name}.tmp"
        frame.iloc[indices].to_csv(temporary, index=False)
        temporary.replace(output_dir / name)

    metadata = {
        "schema": "mips-cache-preflight-cohorts-v1",
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_rows": len(frame),
        "category_counts": dict(sorted(counts.items())),
        "correctness_quotas_realized": quota_counts,
        "correctness_rows": len(selected),
        "performance_rows": len(performance),
        "correctness_is_subset_of_performance": set(selected).issubset(performance),
        "correctness_ordered_hash": hashlib.sha256(
            ",".join(map(str, selected)).encode("utf-8")
        ).hexdigest(),
        "performance_ordered_hash": hashlib.sha256(
            ",".join(map(str, performance)).encode("utf-8")
        ).hexdigest(),
    }
    metadata_path = output_dir / "PI1M_preflight.metadata.json"
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(metadata_path)
    print(json.dumps(metadata, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
