#!/usr/bin/env python3
"""Bind all fixed downstream dataset rows to one published downstream bundle."""

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
from src.dataset.glt_dual_cache import (
    DualFrozenBundle, load_active_dual_store, load_dual_cohort,
    ordered_key_hash, sha256_file,
)


def _git_identity(root: Path) -> dict:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
    return {
        "code_commit": commit, "code_dirty": bool(diff),
        "code_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _load_union(root: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    frozen = json.loads((root / ".frozen").read_text(encoding="utf-8"))
    if frozen != {"manifest_hash": json_hash(manifest)}:
        raise CacheLifecycleError("downstream union manifest is not frozen")
    if sha256_file(root / "rows.jsonl") != manifest["rows_file_sha256"]:
        raise CacheLifecycleError("downstream union row hash mismatch")
    for task, expected in manifest["split_manifest_sha256"].items():
        path = Path("data/splits/mips_outer5_inner20") / f"{task}.json"
        if sha256_file(path) != expected:
            raise CacheLifecycleError(f"fixed split changed after union freeze: {task}")
    rows = [json.loads(line) for line in (root / "rows.jsonl").read_text(
        encoding="utf-8"
    ).splitlines() if line]
    if len(rows) != int(manifest["row_count"]):
        raise CacheLifecycleError("downstream union row count mismatch")
    return manifest, rows


def build(union_root: Path, cache_root: Path, output: Path) -> dict:
    union_root, cache_root, output = (
        union_root.resolve(), cache_root.resolve(), output.resolve()
    )
    if output.exists() or output.with_name(output.name + ".staging").exists():
        raise FileExistsError(f"downstream cohort output already exists: {output}")
    before = snapshot_tree(cache_root)
    union, rows = _load_union(union_root)
    store = load_active_dual_store(cache_root)
    bundle = DualFrozenBundle(cache_root, expected_bundle_hash=store["bundle_hash"])
    try:
        unique_keys = {bytes.fromhex(row["sample_key"]) for row in rows}
        for key in unique_keys:
            if key not in bundle.topology or key not in bundle.trimer:
                raise CacheLifecycleError(
                    "DROPPED_DOWNSTREAM_ROWS: union identity missing from frozen cache"
                )
    finally:
        bundle.close()
    if len(unique_keys) != int(union["unique_normalized_structure_count"]):
        raise CacheLifecycleError("downstream unique identity count mismatch")
    keys = np.frombuffer(
        b"".join(bytes.fromhex(row["sample_key"]) for row in rows),
        dtype=np.uint8,
    ).reshape(-1, 32)
    staging = output.with_name(output.name + ".staging")
    staging.mkdir(parents=True)
    np.save(staging / "keys.npy", keys)
    with (staging / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(
                row, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    manifest = {
        "artifact_type": "glt_dual_downstream_rows",
        "identity_policy": "dataset_rows_with_reuse",
        "ordering_policy": "task_order_then_original_row_preserving_fixed_property_rows",
        "sample_count": len(rows),
        "unique_structure_count": len(unique_keys),
        "duplicate_count": len(rows) - len(unique_keys),
        "dropped_downstream_rows": 0,
        "ordered_sample_key_hash": ordered_key_hash(keys),
        "main_bundle_hash": store["bundle_hash"],
        "source_manifest_hash": store["source"]["source_manifest_hash"],
        "topology_manifest_hash": store["artifacts"]["topology"]["manifest_hash"],
        "trimer_manifest_hash": store["artifacts"]["trimer"]["manifest_hash"],
        "union_manifest_hash": json_hash(union),
        "split_manifest_sha256": union["split_manifest_sha256"],
        "task_counts": union["task_counts"],
        "keys_file_sha256": sha256_file(staging / "keys.npy"),
        "records_file_sha256": sha256_file(staging / "records.jsonl"),
        **_git_identity(Path(__file__).resolve().parents[1]),
    }
    atomic_json(staging / "manifest.json", manifest)
    atomic_json(staging / ".frozen", {"manifest_hash": json_hash(manifest)})
    os.replace(staging, output)
    cohort = load_dual_cohort(output, cache_root)
    if snapshot_tree(cache_root) != before:
        raise CacheLifecycleError("downstream cohort reader modified frozen bundle")
    return {
        "status": "PASS", "sample_count": len(cohort["records"]),
        "unique_structure_count": len(unique_keys), "dropped_downstream_rows": 0,
        "cohort_manifest_hash": cohort["manifest_hash"],
        "main_bundle_hash": store["bundle_hash"], "zero_write": True,
        "output": str(output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--union-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-json")
    args = parser.parse_args()
    result = build(Path(args.union_root), Path(args.cache_root), Path(args.output))
    if args.report_json:
        path = Path(args.report_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, result)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
