#!/usr/bin/env python3
"""Stream a frozen cohort into dual_static_v1 and optional target artifacts.

The builder is deliberately offline and single-writer: workers only read the
published Topology/Trimer LMDBs, while the parent writes immutable NumPy
chunks.  A completed artifact is never overwritten; an interrupted staging
directory can be resumed chunk by chunk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.dataset.cache_lifecycle import CacheLifecycleError, atomic_json, json_hash, zero_write_snapshot
from src.dataset.glt_dual_cache import DualFrozenBundle, load_active_dual_store, load_dual_cohort, ordered_key_hash
from src.dataset.glt_dual_static import (STATIC_FORMAT, TARGET_FORMAT, build_dual_static,
                                          build_pretrain_target, write_chunk)


_WORKER_BUNDLE = None


def _worker_init(cache_root, bundle_hash):
    global _WORKER_BUNDLE
    _WORKER_BUNDLE = DualFrozenBundle(cache_root, expected_bundle_hash=bundle_hash)


def _worker_close():
    global _WORKER_BUNDLE
    if _WORKER_BUNDLE is not None:
        _WORKER_BUNDLE.close()
        _WORKER_BUNDLE = None


def _build_one(payload):
    index, key_hex, source_smiles, normalized_smiles, with_target = payload
    key = bytes.fromhex(key_hex)
    if _WORKER_BUNDLE is None:
        raise RuntimeError("static worker is not initialized")
    topology = _WORKER_BUNDLE.topology[key]
    trimer = _WORKER_BUNDLE.trimer[key]
    static = build_dual_static(topology, trimer, source_smiles)
    target = build_pretrain_target(normalized_smiles) if with_target else None
    return int(index), key, static, target


def _git_identity(root):
    root = Path(root)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
    return {
        "code_commit": commit,
        "code_dirty": bool(diff),
        "code_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _selected_records(cohort, limit, unique):
    records = cohort["records"]
    if unique:
        seen, selected = set(), []
        for row in records:
            key = str(row["sample_key"])
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
    else:
        selected = list(records)
    if int(limit) > 0:
        selected = selected[:int(limit)]
    if not selected:
        raise CacheLifecycleError("selected cohort is empty")
    return selected


def _prepare_artifact(root, fmt, keys, cohort, params):
    root = Path(root).resolve()
    staging = root.with_name(root.name + ".staging")
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing static artifact: {root}")
    staging.mkdir(parents=True, exist_ok=True)
    key_array = np.frombuffer(b"".join(keys), dtype=np.uint8).reshape(-1, 32)
    keys_path = staging / "sample_keys.npy"
    if keys_path.exists():
        old = np.load(keys_path, mmap_mode="r")
        if old.shape != key_array.shape or not np.array_equal(old, key_array):
            raise CacheLifecycleError(f"staging sample keys differ: {staging}")
    else:
        np.save(keys_path, key_array)
    return root, staging


def _existing_chunk(staging, start, count, target):
    chunk = Path(staging) / "chunks" / f"chunk_{start:08d}"
    marker = chunk / ".complete"
    manifest = chunk / "manifest.json"
    if not (chunk.is_dir() and marker.is_file() and manifest.is_file()):
        return False
    observed = json.loads(manifest.read_text(encoding="utf-8"))
    return int(observed.get("start", -1)) == int(start) and int(observed.get("count", -1)) == int(count) and bool(observed.get("target")) == bool(target)


def _artifact_manifest(root, staging, fmt, selected, cohort, params, chunk_manifests):
    keys = np.load(Path(staging) / "sample_keys.npy", mmap_mode="r")
    spec = {
        "format": fmt,
        "parent_bundle_hash": cohort["manifest"]["main_bundle_hash"],
        "cohort_manifest_hash": cohort["manifest_hash"],
        "parameters": params,
    }
    manifest = {
        "format": fmt,
        "sample_count": int(len(selected)),
        "ordered_sample_key_hash": ordered_key_hash(keys),
        "cohort_ordered_sample_key_hash": cohort["manifest"]["ordered_sample_key_hash"],
        "cohort_manifest_hash": cohort["manifest_hash"],
        "parent_bundle_hash": cohort["manifest"]["main_bundle_hash"],
        "chunks": chunk_manifests,
        "build_parameters": params,
        "build_spec_hash": json_hash(spec),
        **_git_identity(Path(__file__).resolve().parents[1]),
    }
    atomic_json(Path(staging) / "manifest.json", manifest)
    atomic_json(Path(staging) / ".frozen", {"manifest_hash": json_hash(manifest)})
    return manifest


def build(args):
    cache_root = Path(args.cache_root).resolve()
    cohort = load_dual_cohort(args.cohort_root, cache_root)
    store = load_active_dual_store(cache_root)
    selected = _selected_records(cohort, args.limit, args.unique)
    keys = [bytes.fromhex(str(row["sample_key"])) for row in selected]
    if any(len(key) != 32 for key in keys):
        raise CacheLifecycleError("selected sample key is not 32 bytes")
    static_root = Path(args.output_root).resolve()
    target_root = Path(args.target_root).resolve() if args.target_root else None
    if target_root is not None and not args.build_targets:
        raise ValueError("--target-root requires --build-targets")
    if args.build_targets and target_root is None:
        raise ValueError("--build-targets requires --target-root")
    before = zero_write_snapshot(cache_root)
    static_root, static_staging = _prepare_artifact(
        static_root, STATIC_FORMAT, keys, cohort,
        {"chunk_size": int(args.chunk_size), "unique": bool(args.unique), "limit": int(args.limit), "targets": bool(args.build_targets)},
    )
    target_staging = None
    if target_root is not None:
        target_root, target_staging = _prepare_artifact(
            target_root, TARGET_FORMAT, keys, cohort,
            {"chunk_size": int(args.chunk_size), "unique": bool(args.unique), "limit": int(args.limit), "targets": True},
        )
    workers = max(1, int(args.workers))
    chunk_size = max(1, int(args.chunk_size))
    started = time.time()
    chunks_static, chunks_target = [], []
    try:
        context = mp.get_context("fork")
        with context.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(str(cache_root), store["bundle_hash"]),
        ) as pool:
            for start in range(0, len(selected), chunk_size):
                rows = selected[start:start + chunk_size]
                end = start + len(rows)
                static_done = _existing_chunk(static_staging, start, len(rows), False)
                target_done = target_staging is None or _existing_chunk(target_staging, start, len(rows), True)
                if static_done and target_done:
                    observed_static = json.loads((static_staging / "chunks" / f"chunk_{start:08d}" / "manifest.json").read_text())
                    chunks_static.append({"start": start, "count": len(rows), "path": f"chunks/chunk_{start:08d}", "arrays": observed_static["arrays"]})
                    if target_staging is not None:
                        observed_target = json.loads((target_staging / "chunks" / f"chunk_{start:08d}" / "manifest.json").read_text())
                        chunks_target.append({"start": start, "count": len(rows), "path": f"chunks/chunk_{start:08d}", "arrays": observed_target["arrays"]})
                    continue
                payloads = [
                    (index, str(row["sample_key"]), str(row["source_smiles"]),
                     str(row["normalized_smiles"]), bool(args.build_targets))
                    for index, row in enumerate(rows, start=start)
                ]
                built = list(pool.imap(_build_one, payloads, chunksize=1))
                built.sort(key=lambda value: value[0])
                if [value[0] for value in built] != list(range(start, end)):
                    raise CacheLifecycleError("static worker result order mismatch")
                static_rows = [value[2] for value in built]
                if static_done:
                    static_manifest = json.loads((static_staging / "chunks" / f"chunk_{start:08d}" / "manifest.json").read_text())
                else:
                    static_manifest = write_chunk(static_staging, start, static_rows, targets=False)
                chunks_static.append({"start": start, "count": len(rows), "path": f"chunks/chunk_{start:08d}", "arrays": static_manifest["arrays"]})
                if target_staging is not None:
                    target_rows = [value[3] for value in built]
                    if target_done:
                        target_manifest = json.loads((target_staging / "chunks" / f"chunk_{start:08d}" / "manifest.json").read_text())
                    else:
                        target_manifest = write_chunk(target_staging, start, target_rows, targets=True)
                    chunks_target.append({"start": start, "count": len(rows), "path": f"chunks/chunk_{start:08d}", "arrays": target_manifest["arrays"]})
                if (len(chunks_static) % max(1, int(args.progress_chunks))) == 0:
                    rate = (end / max(1e-6, time.time() - started))
                    print(json.dumps({"progress": end, "total": len(selected), "samples_per_second": rate}), flush=True)
        chunks_static.sort(key=lambda item: int(item["start"]))
        static_manifest = _artifact_manifest(
            static_root, static_staging, STATIC_FORMAT, selected, cohort,
            {"chunk_size": chunk_size, "unique": bool(args.unique), "limit": int(args.limit), "targets": bool(args.build_targets)},
            chunks_static,
        )
        target_manifest = None
        if target_staging is not None:
            chunks_target.sort(key=lambda item: int(item["start"]))
            target_manifest = _artifact_manifest(
                target_root, target_staging, TARGET_FORMAT, selected, cohort,
                {"chunk_size": chunk_size, "unique": bool(args.unique), "limit": int(args.limit), "targets": True},
                chunks_target,
            )
        if zero_write_snapshot(cache_root) != before:
            raise CacheLifecycleError("static build modified the frozen main bundle")
        os.replace(static_staging, static_root)
        if target_staging is not None:
            os.replace(target_staging, target_root)
    except BaseException:
        # Keep staging for explicit, chunk-level resume.  It is never used by
        # readers until the final .frozen manifest is atomically published.
        raise
    result = {
        "status": "PASS", "sample_count": len(selected),
        "geometry_valid_count": sum(1 for _ in ()),
        "static_root": str(static_root),
        "target_root": str(target_root) if target_root is not None else None,
        "static_manifest_hash": json_hash(static_manifest),
        "target_manifest_hash": json_hash(target_manifest) if target_manifest else None,
        "elapsed_seconds": time.time() - started,
        "main_bundle_hash": store["bundle_hash"],
        "zero_write": True,
    }
    # Count geometry flags from immutable chunks without retaining rows.
    geometry = []
    for item in chunks_static:
        path = static_root / item["path"] / "geometry_valid.npy"
        geometry.append(np.load(path, mmap_mode="r"))
    result["geometry_valid_count"] = int(sum(int(np.asarray(values, dtype=bool).sum()) for values in geometry))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--cohort-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--target-root")
    parser.add_argument("--build-targets", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--unique", action="store_true")
    parser.add_argument("--progress-chunks", type=int, default=1)
    parser.add_argument("--report-json")
    args = parser.parse_args()
    result = build(args)
    if args.report_json:
        atomic_json(Path(args.report_json), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
