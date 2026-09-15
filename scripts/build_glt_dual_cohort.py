#!/usr/bin/env python3
"""Build the ordered accepted-only cohort for the current dual GLT route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.dataset.cache_lifecycle import CacheLifecycleError, atomic_json, json_hash, snapshot_tree
from src.dataset.canonical_periodic import resolve_normalized_identity
from src.dataset.glt_dual_cache import (
    DualFrozenBundle, load_active_dual_store, load_dual_cohort,
    ordered_key_hash, sha256_file,
)
from src.dataset.lmdb_cache import sample_key_from_normalized


ORDERING_POLICY = "source_manifest_order_filtered_by_topology_and_trimer_acceptance"


def _git_identity(root: Path) -> dict:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD"], cwd=root
    )
    return {
        "code_commit": commit,
        "code_dirty": bool(diff),
        "code_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _key_set(path: Path) -> set[bytes]:
    array = np.load(path, mmap_mode="r")
    if array.dtype != np.uint8 or array.ndim != 2 or array.shape[1] != 32:
        raise CacheLifecycleError(f"invalid accepted key array: {path}")
    return {bytes(np.asarray(row, dtype=np.uint8)) for row in array}


def _write_records(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(
                row, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _strict_read_audit(bundle: DualFrozenBundle, records: list[dict]) -> dict:
    count = len(records)
    selected = list(range(min(1000, count)))
    rng = np.random.default_rng(20260915)
    selected.extend(int(value) for value in rng.choice(
        count, size=min(1000, count), replace=False
    ))
    selected.extend([0, count // 2, count - 1])
    checked = set()
    for index in selected:
        if index in checked:
            continue
        checked.add(index)
        row = records[index]
        key = bytes.fromhex(row["sample_key"])
        topology = bundle.topology[key]
        trimer = bundle.trimer[key]
        identity = resolve_normalized_identity(
            topology, row["source_smiles"], require_fields=True
        )
        if identity["normalized_smiles"] != row["normalized_smiles"]:
            raise CacheLifecycleError("cohort/source/topology identity mismatch")
        mapping = np.asarray(trimer.mips_to_trimer_central_index)
        if mapping.ndim != 1 or mapping.size != int(topology.mips_x.size(0)):
            raise CacheLifecycleError("cohort topology/Trimer mapping mismatch")
    return {
        "strict_read_count": len(checked),
        "sequential_1k": min(1000, count),
        "deterministic_random_1k": min(1000, count),
        "first_middle_last": True,
    }


def build(cache_root: Path, output: Path, expected_count: int) -> dict:
    cache_root = cache_root.resolve()
    output = output.resolve()
    before = snapshot_tree(cache_root)
    store = load_active_dual_store(cache_root)
    if output.exists():
        cohort = load_dual_cohort(output, cache_root)
        if expected_count and len(cohort["records"]) != expected_count:
            raise CacheLifecycleError("existing cohort count differs from expectation")
        if snapshot_tree(cache_root) != before:
            raise CacheLifecycleError("repeat cohort read modified the main cache")
        return {
            "status": "PASS", "repeat_zero_write": True,
            "sample_count": len(cohort["records"]),
            "cohort_manifest_hash": cohort["manifest_hash"],
            "output": str(output),
        }

    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"cohort staging path already exists: {staging}")
    staging.mkdir(parents=True)
    bundle = DualFrozenBundle(cache_root, expected_bundle_hash=store["bundle_hash"])
    try:
        topology_keys = _key_set(bundle.topology.root / "accepted_keys.npy")
        trimer_keys = _key_set(bundle.trimer.root / "accepted_keys.npy")
        accepted = topology_keys & trimer_keys
        records = []
        source_path = (
            cache_root / store["source"]["path"] / "records.jsonl"
        ).resolve()
        with source_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = bytes.fromhex(row["sample_key"])
                if key not in accepted:
                    continue
                if key != sample_key_from_normalized(row["normalized_smiles"]):
                    raise CacheLifecycleError("source normalized identity corruption")
                records.append(row)
        if len(records) != len(accepted):
            raise CacheLifecycleError("accepted key/source manifest coverage mismatch")
        if expected_count and len(records) != expected_count:
            raise CacheLifecycleError(
                f"dual cohort count {len(records)} != expected {expected_count}"
            )
        keys = np.frombuffer(
            b"".join(bytes.fromhex(row["sample_key"]) for row in records),
            dtype=np.uint8,
        ).reshape(-1, 32)
        np.save(staging / "keys.npy", keys)
        _write_records(staging / "records.jsonl", records)
        audit = _strict_read_audit(bundle, records)
        repo_root = Path(__file__).resolve().parents[1]
        manifest = {
            "artifact_type": "glt_dual_formal_cohort",
            "identity_policy": "unique_structure",
            "ordering_policy": ORDERING_POLICY,
            "sample_count": len(records),
            "duplicate_count": len(records) - len(set(keys.tobytes()[i:i+32]
                for i in range(0, keys.nbytes, 32))),
            "ordered_sample_key_hash": ordered_key_hash(keys),
            "main_bundle_hash": store["bundle_hash"],
            "source_manifest_hash": store["source"]["source_manifest_hash"],
            "topology_manifest_hash": store["artifacts"]["topology"]["manifest_hash"],
            "trimer_manifest_hash": store["artifacts"]["trimer"]["manifest_hash"],
            "keys_file_sha256": sha256_file(staging / "keys.npy"),
            "records_file_sha256": sha256_file(staging / "records.jsonl"),
            **_git_identity(repo_root),
            "audit": audit,
        }
        atomic_json(staging / "manifest.json", manifest)
        atomic_json(staging / ".frozen", {"manifest_hash": json_hash(manifest)})
        if snapshot_tree(cache_root) != before:
            raise CacheLifecycleError(
                "cohort construction observed a main-cache change before publish"
            )
        os.replace(staging, output)
    finally:
        bundle.close()
    cohort = load_dual_cohort(output, cache_root)
    zero_write = snapshot_tree(cache_root) == before
    if not zero_write:
        raise CacheLifecycleError("cohort publish observed a main-cache change")
    return {
        "status": "PASS", "sample_count": len(cohort["records"]),
        "duplicate_count": cohort["manifest"]["duplicate_count"],
        "cohort_manifest_hash": cohort["manifest_hash"],
        "main_bundle_hash": store["bundle_hash"],
        "zero_write": True, "audit": cohort["manifest"]["audit"],
        "output": str(output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-count", type=int, default=959588)
    parser.add_argument("--report-json")
    args = parser.parse_args()
    result = build(Path(args.cache_root), Path(args.output), args.expected_count)
    if args.report_json:
        path = Path(args.report_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, result)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
