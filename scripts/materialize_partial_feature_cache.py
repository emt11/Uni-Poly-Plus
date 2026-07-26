#!/usr/bin/env python3
"""Materialize a deterministic training subset from a partial feature cache."""

import argparse
import copy
import json
import pickle
import random
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.dataset import (
    _attach_periodic_aug_views,
    _attach_scage_3d_descriptors,
    _standardize_scage_descriptors,
)
from src.dataset.graph_data import (
    annotate_structure_fields,
    build_structure_for_input,
    mol_to_graph_data_obj_simple,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a dataset CSV and complete feature cache from a read-only "
            "snapshot of an in-progress cache."
        )
    )
    parser.add_argument("--partial-cache", required=True)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timing-json", default=None)
    parser.add_argument(
        "--migrate-scage-mips", action="store_true",
        help="Reuse cached geometry but rebuild scage-mips-v1 graph/path/descriptor fields.",
    )
    return parser.parse_args()


def _migrate_feature(payload):
    smiles, old = payload
    if isinstance(old, (bytes, bytearray)):
        old = pickle.loads(old)
    structure = build_structure_for_input(smiles, "star_linking")
    new = mol_to_graph_data_obj_simple(
        structure["structure_mol"], backbone_info=structure.get("backbone_info")
    )
    annotate_structure_fields(new, structure, prefix="graph")
    graph_keys = {
        "x", "edge_index", "edge_attr", "atomic_num", "chiral_tag", "degree",
        "explicit_valence", "formal_charge", "hybridization", "is_aromatic",
        "total_numHs", "atom_is_in_ring", "mass", "van_der_waals_radius",
        "partial_charge", "scage_backbone_role", "scage_spd",
        "scage_path_bond_fields", "scage_topology_schema_version",
        "attachment_pair", "star_link_edge", "ordered_backbone_path",
        "star_link_metadata_valid", "graph_build_ok", "graph_failed_reason",
        "graph_smiles", "graph_input", "structure_smiles", "structure_input",
        "requested_graph_input", "attachment_count", "has_backbone_features",
    }
    for key in old.keys():
        if key not in graph_keys and not key.startswith("scage_descriptor_"):
            setattr(new, key, copy.deepcopy(old[key]))
    new.smiles = smiles
    _attach_periodic_aug_views(new, smiles)
    _attach_scage_3d_descriptors(new, structure["structure_mol"])
    # ProcessPool's default PyTorch reducer places returned tensor storages in
    # /dev/shm. Containers commonly expose only 64 MB there, so return an
    # ordinary byte stream and deserialize it in the parent process instead.
    return smiles, pickle.dumps(new, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    args = parse_args()
    partial_path = Path(args.partial_cache)
    if not partial_path.is_file():
        raise FileNotFoundError(partial_path)
    if not partial_path.name.endswith(".pt.partial"):
        raise ValueError("--partial-cache must end in .pt.partial")

    print(f"[materialize] loading cache: {partial_path}", flush=True)
    load_started = time.monotonic()
    cache = torch.load(partial_path, map_location="cpu", weights_only=False, mmap=True)
    load_seconds = time.monotonic() - load_started
    features = cache.get("features", {})
    if not features:
        raise ValueError("Partial cache contains no completed features")
    print(
        f"[materialize] loaded {len(features)} completed entries in {load_seconds:.1f}s",
        flush=True,
    )
    sample_size = int(args.sample_size)
    if sample_size <= 0 or sample_size > len(features):
        raise ValueError(
            f"sample-size must be in [1, {len(features)}], got {sample_size}"
        )

    candidates = []
    for smiles, data in features.items():
        if args.migrate_scage_mips and not (
            bool(getattr(data, "graph_build_ok", False))
            and bool(getattr(data, "screw_valid", False))
            and bool(getattr(data, "geom_coordinate_ok", False))
        ):
            continue
        candidates.append(smiles)
    if sample_size > len(candidates):
        raise ValueError(f"Only {len(candidates)} valid candidates are available")
    print(
        f"[materialize] valid candidates={len(candidates)}; selecting {sample_size}",
        flush=True,
    )
    # Stable size-stratified sampling preserves small/medium/large RU coverage.
    buckets = {0: [], 1: [], 2: [], 3: []}
    for smiles in candidates:
        atom_count = int(features[smiles].x.size(0))
        bucket = 0 if atom_count < 20 else (1 if atom_count < 40 else (2 if atom_count < 80 else 3))
        buckets[bucket].append(smiles)
    rng = random.Random(int(args.seed))
    selected_list = []
    remaining = sample_size
    nonempty = [key for key, values in buckets.items() if values]
    for position, key in enumerate(nonempty):
        quota = remaining if position == len(nonempty) - 1 else round(
            sample_size * len(buckets[key]) / len(candidates)
        )
        quota = min(quota, len(buckets[key]), remaining)
        selected_list.extend(rng.sample(buckets[key], quota))
        remaining -= quota
    if remaining:
        pool = [smiles for smiles in candidates if smiles not in set(selected_list)]
        selected_list.extend(rng.sample(pool, remaining))
    selected = set(selected_list)
    source_df = pd.read_csv(args.source_csv)
    smiles_column = source_df.columns[0]
    smiles_values = source_df[smiles_column].astype(str).str.strip()
    output_df = source_df.loc[smiles_values.isin(selected)].copy()
    output_df[smiles_column] = output_df[smiles_column].astype(str).str.strip()
    output_df = output_df.drop_duplicates(subset=[smiles_column], keep="first")
    if len(output_df) != sample_size:
        missing = sample_size - len(output_df)
        raise ValueError(
            f"Source CSV did not contain every selected feature key; missing={missing}"
        )

    data_root = Path(args.data_root)
    csv_path = data_root / "raw" / f"{args.output_dataset}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(csv_path, index=False)

    output_name = partial_path.name[:-len(".partial")].replace(
        "feature_cache_" + str(cache.get("meta", {}).get("feature_source_dataset")),
        "feature_cache_" + args.output_dataset,
        1,
    )
    if output_name == partial_path.name[:-len(".partial")]:
        raise ValueError("Could not derive output cache name from cache metadata")
    if args.migrate_scage_mips:
        output_name = output_name.replace(
            "scage-starlink-backbone-input-v1-m4p-v1",
            "scage-starlink-backbone-scage-mips-v1",
        )
    output_path = partial_path.with_name(output_name)

    selected_features = {}
    if args.migrate_scage_mips:
        workers = max(1, int(args.workers))
        migration_started = time.monotonic()
        print(
            f"[materialize] rebuilding MIPS fields with workers={workers}", flush=True
        )
        if workers == 1:
            for smiles in selected_list:
                key, serialized_value = _migrate_feature((smiles, features[smiles]))
                selected_features[key] = pickle.loads(serialized_value)
        else:
            pending = iter(selected_list)
            max_in_flight = workers * 2
            with ProcessPoolExecutor(max_workers=workers) as executor:
                in_flight = {}

                def submit_one():
                    try:
                        smiles = next(pending)
                    except StopIteration:
                        return False
                    serialized = pickle.dumps(
                        features[smiles], protocol=pickle.HIGHEST_PROTOCOL
                    )
                    future = executor.submit(_migrate_feature, (smiles, serialized))
                    in_flight[future] = smiles
                    return True

                for _ in range(min(max_in_flight, len(selected_list))):
                    submit_one()
                completed = 0
                while in_flight:
                    done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for future in done:
                        smiles = in_flight.pop(future)
                        key, serialized_value = future.result()
                        selected_features[key] = pickle.loads(serialized_value)
                        completed += 1
                        if completed % 100 == 0 or completed == len(selected_list):
                            print(
                                f"Migrating SCAGE MIPS fields: {completed}/{len(selected_list)}",
                                flush=True,
                            )
                        submit_one()
            selected_features = {
                smiles: selected_features[smiles] for smiles in selected_list
            }
        descriptor_statistics = _standardize_scage_descriptors(selected_features)
        migration_seconds = time.monotonic() - migration_started
    else:
        selected_features = {smiles: features[smiles] for smiles in selected_list}
        descriptor_statistics = None
        migration_seconds = 0.0

    output_meta = copy.deepcopy(cache.get("meta", {}))
    output_meta.update({
        "feature_source_dataset": args.output_dataset,
        "materialized_from_partial": str(partial_path),
        "materialized_sample_size": sample_size,
        "materialized_seed": int(args.seed),
        "materialized_valid_candidates": len(candidates),
        "materialized_workers": max(1, int(args.workers)),
        "materialized_load_seconds": load_seconds,
        "materialized_migration_seconds": migration_seconds,
        "scage_data_schema": "scage-mips-v1" if args.migrate_scage_mips else output_meta.get("scage_data_schema"),
        "scage_topology_schema_version": 1 if args.migrate_scage_mips else output_meta.get("scage_topology_schema_version"),
        "scage_descriptor_schema_version": 1 if args.migrate_scage_mips else output_meta.get("scage_descriptor_schema_version"),
        "scage_descriptor_statistics": descriptor_statistics,
    })
    output_cache = {
        "meta": output_meta,
        "features": selected_features,
        "failures": [],
    }
    torch.save(output_cache, output_path)
    if args.timing_json:
        timing_path = Path(args.timing_json)
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text(json.dumps({
            "load_seconds": load_seconds,
            "migration_seconds": migration_seconds,
            "sample_size": sample_size,
        }, indent=2) + "\n")
    print(f"Materialized {sample_size} features")
    print(f"Load seconds: {load_seconds:.2f}; migration seconds: {migration_seconds:.2f}")
    print(f"CSV: {csv_path}")
    print(f"Cache: {output_path}")


if __name__ == "__main__":
    main()
