#!/usr/bin/env python3
"""Build a deterministic, conditionally stratified G3 relation permutation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.dataset.lmdb_cache import LmdbLayerStore, load_cohort
from src.dataset.mts_relation_geometry import RelationGeometrySidecar


SCHEMA = "mts-relation-geometry-permutation-v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _strata_for_sidecar(sidecar, *, topology_store=None, sample_keys=None):
    """Return one chemistry stratum key per global relation row."""
    strata = []
    sample_offsets = sidecar.arrays["sample_relation_offsets"]
    if topology_store is None:
        # This fallback is useful for isolated unit tests with synthetic
        # sidecars.  Production generation passes a topology store and binds
        # atomic-number provenance in metadata.
        atomic_source = "canonical_id_fallback"
        for sample_index in range(len(sidecar)):
            a, b = int(sample_offsets[sample_index]), int(sample_offsets[sample_index + 1])
            relation_rows = sidecar.row(sample_index)["relations"]
            paths = sidecar.row(sample_index)["paths"]
            path_base = int(sidecar.arrays["relation_path_offsets"][a])
            for local in range(b - a):
                pa, pb = int(sidecar.arrays["relation_path_offsets"][a + local]) - path_base, int(sidecar.arrays["relation_path_offsets"][a + local + 1]) - path_base
                chemistry = tuple(sorted(zip(
                    paths["path_intermediate_canonical_id"][pa - int(sidecar.arrays["relation_path_offsets"][a]): pb - int(sidecar.arrays["relation_path_offsets"][a])].tolist(),
                    paths["path_bond_type_left"][pa - int(sidecar.arrays["relation_path_offsets"][a]): pb - int(sidecar.arrays["relation_path_offsets"][a])].tolist(),
                    paths["path_bond_type_right"][pa - int(sidecar.arrays["relation_path_offsets"][a]): pb - int(sidecar.arrays["relation_path_offsets"][a])].tolist(),
                )))
                strata.append((
                    int(relation_rows["relation_source_canonical_id"][local]),
                    int(relation_rows["relation_target_canonical_id"][local]),
                    int(relation_rows["relation_signed_source_shift"][local]),
                    int(relation_rows["relation_num_shortest_paths"][local]),
                    chemistry,
                ))
        return strata, atomic_source

    atomic_source = "topology_atomic_numbers"
    for sample_index in range(len(sidecar)):
        key = bytes(np.asarray(sidecar.arrays["sample_keys"][sample_index], dtype=np.uint8).tobytes())
        topology = topology_store[key]
        atomic_numbers = getattr(topology, "atomic_numbers", None)
        if atomic_numbers is None:
            atomic_numbers = getattr(topology, "z", None)
        if atomic_numbers is None:
            raise RuntimeError("topology record is missing atomic_numbers/z")
        z = np.asarray(atomic_numbers, dtype=np.int64).reshape(-1)
        record = sidecar.row(sample_index)
        relation = record["relations"]
        paths = record["paths"]
        base_path = int(sidecar.arrays["relation_path_offsets"][int(sample_offsets[sample_index])])
        for local in range(len(relation["relation_row"])):
            pa, pb = int(sidecar.arrays["relation_path_offsets"][int(sample_offsets[sample_index]) + local]) - base_path, int(sidecar.arrays["relation_path_offsets"][int(sample_offsets[sample_index]) + local + 1]) - base_path
            p0, p1 = pa, pb
            chemistry = []
            for path_index in range(p0, p1):
                middle = int(paths["path_intermediate_canonical_id"][path_index])
                middle_z = int(z[middle]) if 0 <= middle < len(z) else -1
                left = int(paths["path_bond_type_left"][path_index])
                right = int(paths["path_bond_type_right"][path_index])
                chemistry.append((middle_z, min(left, right), max(left, right)))
            source = int(relation["relation_source_canonical_id"][local])
            target = int(relation["relation_target_canonical_id"][local])
            strata.append((
                int(z[source]) if 0 <= source < len(z) else -1,
                int(z[target]) if 0 <= target < len(z) else -1,
                int(relation["relation_signed_source_shift"][local]),
                int(relation["relation_num_shortest_paths"][local]),
                tuple(sorted(chemistry)),
            ))
    return strata, atomic_source


def build_permutation(sidecar_root, output_root, *, seed=42, topology_root=None):
    sidecar = RelationGeometrySidecar(sidecar_root, verify_array_hashes=True)
    topology_store = LmdbLayerStore(topology_root, require_done=True) if topology_root else None
    try:
        strata, atomic_source = _strata_for_sidecar(sidecar, topology_store=topology_store)
    finally:
        if topology_store is not None:
            topology_store.close()
    if len(strata) != int(sidecar.arrays["relation_row"].shape[0]):
        raise RuntimeError("G3 stratum count does not match sidecar relation count")
    valid = np.asarray(sidecar.arrays["relation_geometry_valid"], dtype=bool)
    path_offsets = np.asarray(sidecar.arrays["relation_path_offsets"], dtype=np.int64)
    path_valid = np.asarray(sidecar.arrays["path_geometry_valid"], dtype=bool)
    eligible = valid.copy()
    for index in np.flatnonzero(eligible):
        eligible[index] = bool(path_offsets[index + 1] > path_offsets[index] and path_valid[path_offsets[index]:path_offsets[index + 1]].all())
    groups = defaultdict(list)
    for index, key in enumerate(strata):
        if bool(eligible[index]):
            groups[key].append(index)
    permutation = np.arange(len(strata), dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    singleton_count = 0
    permuted_count = 0
    for key in sorted(groups, key=repr):
        indices = np.asarray(groups[key], dtype=np.int64)
        if len(indices) <= 1:
            singleton_count += int(len(indices))
            continue
        shuffled = indices[rng.permutation(len(indices))]
        permutation[indices] = shuffled
        permuted_count += int(np.count_nonzero(indices != shuffled))
    metadata = {
        "schema": SCHEMA,
        "version": 1,
        "seed": int(seed),
        "cohort": sidecar.metadata.get("cohort"),
        "cohort_hash": sidecar.cohort_hash,
        "source_sidecar_root": str(sidecar.root),
        "source_sidecar_artifact": sidecar.artifact_hash,
        "atomic_number_source": atomic_source,
        "stratum_fields": [
            "ordered_endpoint_atomic_numbers", "signed_source_ru_shift",
            "num_shortest_paths", "path_intermediate_atomic_number_bond_pair_multiset",
        ],
        "relation_count": int(len(strata)),
        "eligible_count": int(eligible.sum()),
        "permuted_count": int(permuted_count),
        "singleton_count": int(singleton_count),
        "strata_count": int(len(groups)),
        "invalid_identity_count": int((~eligible).sum()),
    }
    metadata["identity_hash"] = _digest({key: value for key, value in metadata.items() if key != "identity_hash"})
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite G3 permutation artifact: {output}")
    output.mkdir(parents=True, exist_ok=True)
    payload = output / "relation_permutation.npy"
    np.save(payload, permutation, allow_pickle=False)
    metadata["array"] = {"path": payload.name, "dtype": np.dtype(permutation.dtype).str, "shape": list(permutation.shape), "sha256": _sha256(payload)}
    metadata["artifact_hash"] = _digest({key: value for key, value in metadata.items() if key != "artifact_hash"})
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / ".done").write_text(metadata["artifact_hash"] + "\n", encoding="utf-8")
    (output / ".frozen").write_text(json.dumps({"schema": SCHEMA, "artifact_hash": metadata["artifact_hash"], "source_sidecar_artifact": sidecar.artifact_hash}, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--topology-root", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print(json.dumps(build_permutation(args.sidecar, args.output, seed=args.seed, topology_root=args.topology_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
