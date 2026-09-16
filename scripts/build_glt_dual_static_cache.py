#!/usr/bin/env python3
"""Stream a frozen cohort into dual_static_v1 and optional target artifacts.

The builder is deliberately offline and single-writer: workers only read the
published Topology/Trimer LMDBs, while the parent writes immutable NumPy
chunks.  A completed artifact is never overwritten; an interrupted staging
directory can be resumed chunk by chunk.
"""

from __future__ import annotations

import argparse
import fcntl
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
                                          build_pretrain_target, load_chunk_payload,
                                          write_chunk)


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


BUILD_CONTEXT_NAME = "build_context.json"


def _build_context(fmt, keys, cohort, params):
    """Identity of this build, recorded in staging before the first chunk.

    Matching sample keys alone must never authorise a resume: a staging tree
    left by a different parent bundle, cohort or parameter set has to be
    rejected even when its keys coincide.
    """

    key_array = np.frombuffer(b"".join(keys), dtype=np.uint8).reshape(-1, 32)
    return {
        "format": fmt,
        "parent_bundle_hash": cohort["manifest"]["main_bundle_hash"],
        "cohort_manifest_hash": cohort["manifest_hash"],
        "build_parameters": params,
        "ordered_sample_key_hash": ordered_key_hash(key_array),
    }


def _published_identity(root):
    """Read a published artifact's own build identity, or None if unpublished."""

    root = Path(root).resolve()
    if not root.exists():
        return None
    manifest_path = root / "manifest.json"
    frozen_path = root / ".frozen"
    if not manifest_path.is_file() or not frozen_path.is_file():
        raise CacheLifecycleError(f"published path is incomplete (needs manifest.json and .frozen): {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen != {"manifest_hash": json_hash(manifest)}:
        raise CacheLifecycleError(f"published .frozen does not bind its manifest: {root}")
    return {name: manifest.get(name) for name in
            ("format", "parent_bundle_hash", "cohort_manifest_hash",
             "build_parameters", "ordered_sample_key_hash")}


def _prepare_artifact(root, fmt, keys, cohort, params):
    root = Path(root).resolve()
    staging = root.with_name(root.name + ".staging")
    context = _build_context(fmt, keys, cohort, params)
    published = _published_identity(root)
    if published is not None:
        for name, expected in context.items():
            if published.get(name) != expected:
                raise CacheLifecycleError(
                    f"published artifact {root} came from a different build context: {name}")
    key_array = np.frombuffer(b"".join(keys), dtype=np.uint8).reshape(-1, 32)
    if staging.exists() and not staging.is_dir():
        raise CacheLifecycleError(f"staging path is not a directory: {staging}")
    context_path = staging / BUILD_CONTEXT_NAME
    keys_path = staging / "sample_keys.npy"
    if staging.exists():
        entries = [path for path in staging.iterdir() if path.name != "build.lock"]
        if not context_path.is_file() and entries:
            raise CacheLifecycleError(
                f"staging contains data but no build context: {staging}")
        if context_path.is_file():
            try:
                recorded = json.loads(context_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise CacheLifecycleError(f"staging build context is unreadable: {staging}") from exc
            if recorded != context:
                differing = sorted(name for name in context if recorded.get(name) != context[name])
                raise CacheLifecycleError(
                    f"staging belongs to a different build context: {staging} ({','.join(differing)})")
        if keys_path.exists():
            try:
                old = np.load(keys_path, mmap_mode="r")
            except Exception as exc:
                raise CacheLifecycleError(f"staging sample keys are unreadable: {staging}") from exc
            if old.shape != key_array.shape or not np.array_equal(old, key_array):
                raise CacheLifecycleError(f"staging sample keys differ: {staging}")
    return root, staging, published


def _atomic_numpy(path, value):
    """Write a small identity array before exposing it in staging."""

    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npy")
    try:
        np.save(temporary, value)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _initialize_staging(staging, keys, context):
    """Create/validate staging identity while the caller holds its writer lock."""

    staging = Path(staging)
    staging.mkdir(parents=True, exist_ok=True)
    keys_array = np.frombuffer(b"".join(keys), dtype=np.uint8).reshape(-1, 32)
    keys_path = staging / "sample_keys.npy"
    if keys_path.exists():
        try:
            old = np.load(keys_path, mmap_mode="r")
        except Exception as exc:
            raise CacheLifecycleError(f"staging sample keys are unreadable: {staging}") from exc
        if old.shape != keys_array.shape or not np.array_equal(old, keys_array):
            raise CacheLifecycleError(f"staging sample keys differ: {staging}")
    else:
        _atomic_numpy(keys_path, keys_array)
    context_path = staging / BUILD_CONTEXT_NAME
    if context_path.exists():
        try:
            recorded = json.loads(context_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise CacheLifecycleError(f"staging build context is unreadable: {staging}") from exc
        if recorded != context:
            differing = sorted(name for name in context if recorded.get(name) != context[name])
            raise CacheLifecycleError(
                f"staging belongs to a different build context: {staging} ({','.join(differing)})")
    else:
        atomic_json(context_path, context)


class _BuildLock:
    """An advisory flock whose file descriptor remains held for the build."""

    def __init__(self, path, handle):
        self.path = Path(path)
        self.handle = handle

    def is_file(self):
        return self.path.is_file()


def _acquire_build_lock(staging):
    """Explicit single-writer guard for one staging root.

    A concurrent writer must not adopt the same staging.  Kernel flock release
    makes a process-dead lock immediately reclaimable without PID probing or
    deleting a lock file that may belong to another writer.
    """

    staging = Path(staging)
    staging.mkdir(parents=True, exist_ok=True)
    lock = staging / "build.lock"
    handle = lock.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        holder = ""
        try:
            handle.seek(0)
            holder = handle.read().strip()
        except OSError:
            pass
        handle.close()
        raise CacheLifecycleError(
            f"another writer holds this staging: {staging} (pid {holder or 'unknown'})") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    os.fsync(handle.fileno())
    return _BuildLock(lock, handle)


def _release_build_lock(lock):
    if not isinstance(lock, _BuildLock):
        raise TypeError("build lock must be the handle returned by _acquire_build_lock")
    try:
        lock.handle.seek(0)
        lock.handle.truncate()
        lock.handle.flush()
        os.fsync(lock.handle.fileno())
        fcntl.flock(lock.handle.fileno(), fcntl.LOCK_UN)
    finally:
        lock.handle.close()


def _publish_plan(*, static_published, targets_requested, targets_published):
    """Classify the publish state and decide which sides still need building.

    Static and targets are separate roots, so two renames cannot be atomic
    together.  A published side is reused only after its build context matched,
    and only the missing side is built and published.
    """

    if static_published and (not targets_requested or targets_published):
        state = "both" if targets_requested else "static_only"
    elif static_published:
        state = "static_only"
    elif targets_published:
        state = "target_only"
    else:
        state = "none"
    return state, (not static_published), (bool(targets_requested) and not targets_published)


def _existing_chunk(staging, start, count, target):
    chunk = Path(staging) / "chunks" / f"chunk_{start:08d}"
    marker = chunk / ".complete"
    manifest = chunk / "manifest.json"
    if not chunk.exists():
        return False
    if not chunk.is_dir() or not marker.is_file() or not manifest.is_file():
        if marker.is_file():
            raise CacheLifecycleError(f"completed chunk lacks a readable manifest: {chunk}")
        return False
    try:
        observed = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CacheLifecycleError(f"completed chunk manifest is unreadable: {chunk}") from exc
    if (int(observed.get("start", -1)) != int(start)
            or int(observed.get("count", -1)) != int(count)
            or bool(observed.get("target")) != bool(target)):
        raise CacheLifecycleError(f"completed chunk identity mismatch: {chunk}")
    load_chunk_payload(chunk, observed, targets=target)
    return True


def _artifact_manifest(root, staging, fmt, selected, cohort, params, chunk_manifests):
    keys = np.load(Path(staging) / "sample_keys.npy", mmap_mode="r")
    expected_start = 0
    for item in sorted(chunk_manifests, key=lambda value: int(value.get("start", -1))):
        start, count = int(item.get("start", -1)), int(item.get("count", -1))
        if start != expected_start or count < 0:
            raise CacheLifecycleError("artifact chunks are not contiguous")
        if bool(item.get("target")) != (fmt == TARGET_FORMAT):
            raise CacheLifecycleError("artifact chunk target flag does not match format")
        chunk = Path(staging) / str(item.get("path", ""))
        if not (chunk / ".complete").is_file():
            raise CacheLifecycleError(f"artifact chunk is not complete: {chunk}")
        load_chunk_payload(chunk, item, targets=fmt == TARGET_FORMAT)
        expected_start += count
    if expected_start != len(selected):
        raise CacheLifecycleError("artifact chunk count does not match selected cohort")
    if keys.shape != (len(selected), 32):
        raise CacheLifecycleError("artifact sample key shape does not match selected cohort")
    if ordered_key_hash(keys) != ordered_key_hash(
            bytes.fromhex(str(row["sample_key"])) for row in selected):
        raise CacheLifecycleError("artifact sample key order does not match selected cohort")
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
    manifest_path = Path(staging) / "manifest.json"
    frozen_path = Path(staging) / ".frozen"
    if frozen_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise CacheLifecycleError(f"frozen staging manifest is unreadable: {staging}") from exc
        if frozen != {"manifest_hash": json_hash(existing)}:
            raise CacheLifecycleError(f"frozen staging does not bind its manifest: {staging}")
        if existing != manifest:
            raise CacheLifecycleError(f"frozen staging manifest would change: {staging}")
        return existing
    atomic_json(manifest_path, manifest)
    atomic_json(frozen_path, {"manifest_hash": json_hash(manifest)})
    return manifest


def build(args):
    started = time.time()
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
    static_params = {
        "chunk_size": int(args.chunk_size), "unique": bool(args.unique),
        "limit": int(args.limit), "targets": False,
    }
    target_params = {
        "chunk_size": int(args.chunk_size), "unique": bool(args.unique),
        "limit": int(args.limit), "targets": True,
    }
    static_root, static_staging, static_published = _prepare_artifact(
        static_root, STATIC_FORMAT, keys, cohort,
        static_params,
    )
    target_staging, target_published = None, False
    if target_root is not None:
        target_root, target_staging, target_published = _prepare_artifact(
            target_root, TARGET_FORMAT, keys, cohort,
            target_params,
        )
    # A published side is never rewritten.  When both sides are already
    # published with a matching context the call is an idempotent no-op; when a
    # single side is published only the missing side is resumed and published.
    publish_state, resume_static, resume_target = _publish_plan(
        static_published=bool(static_published),
        targets_requested=target_staging is not None,
        targets_published=bool(target_published),
    )
    if not resume_static and not resume_target:
        return {
            "status": "IDEMPOTENT", "publish_state": publish_state,
            "sample_count": len(selected),
            "static_root": str(static_root),
            "target_root": str(target_root) if target_root is not None else None,
            "static_manifest_hash": json_hash(json.loads((static_root / "manifest.json").read_text(encoding="utf-8"))),
            "target_manifest_hash": (json_hash(json.loads((target_root / "manifest.json").read_text(encoding="utf-8")))
                                     if target_root is not None else None),
            "elapsed_seconds": time.time() - started,
            "main_bundle_hash": store["bundle_hash"],
            "zero_write": True,
            "note": "both requested sides were already published with a matching build context; nothing was rewritten",
        }
    workers = max(1, int(args.workers))
    chunk_size = max(1, int(args.chunk_size))
    chunks_static, chunks_target = [], []
    quarantine_static = static_staging / ".interrupted"
    quarantine_target = target_staging / ".interrupted" if target_staging is not None else None
    locks = []
    try:
        # Identity files and all chunk/finalization writes happen only after
        # every side that will be resumed has acquired its process-held lock.
        if resume_static:
            locks.append(_acquire_build_lock(static_staging))
            _initialize_staging(static_staging, keys, _build_context(
                STATIC_FORMAT, keys, cohort, static_params))
        if resume_target:
            locks.append(_acquire_build_lock(target_staging))
            _initialize_staging(target_staging, keys, _build_context(
                TARGET_FORMAT, keys, cohort, target_params))
        context = mp.get_context("fork")
        with context.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(str(cache_root), store["bundle_hash"]),
        ) as pool:
            for start in range(0, len(selected), chunk_size):
                rows = selected[start:start + chunk_size]
                end = start + len(rows)
                static_done = (not resume_static) or _existing_chunk(static_staging, start, len(rows), False)
                target_done = (not resume_target) or _existing_chunk(target_staging, start, len(rows), True)
                if static_done and target_done:
                    if resume_static:
                        observed_static = json.loads((static_staging / "chunks" / f"chunk_{start:08d}" / "manifest.json").read_text())
                        chunks_static.append({"start": start, "count": len(rows),
                                              "target": False,
                                              "path": f"chunks/chunk_{start:08d}",
                                              "arrays": observed_static["arrays"]})
                    if resume_target:
                        observed_target = json.loads((target_staging / "chunks" / f"chunk_{start:08d}" / "manifest.json").read_text())
                        chunks_target.append({"start": start, "count": len(rows),
                                              "target": True,
                                              "path": f"chunks/chunk_{start:08d}",
                                              "arrays": observed_target["arrays"]})
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
                if resume_static:
                    static_rows = [value[2] for value in built]
                    if static_done:
                        static_manifest = json.loads((static_staging / "chunks" / f"chunk_{start:08d}" / "manifest.json").read_text())
                    else:
                        static_manifest = write_chunk(static_staging, start, static_rows, targets=False,
                                                      quarantine_root=quarantine_static)
                    chunks_static.append({"start": start, "count": len(rows),
                                          "target": False,
                                          "path": f"chunks/chunk_{start:08d}",
                                          "arrays": static_manifest["arrays"]})
                if resume_target:
                    target_rows = [value[3] for value in built]
                    if target_done:
                        target_manifest = json.loads((target_staging / "chunks" / f"chunk_{start:08d}" / "manifest.json").read_text())
                    else:
                        target_manifest = write_chunk(target_staging, start, target_rows, targets=True,
                                                      quarantine_root=quarantine_target)
                    chunks_target.append({"start": start, "count": len(rows),
                                          "target": True,
                                          "path": f"chunks/chunk_{start:08d}",
                                          "arrays": target_manifest["arrays"]})
                if (len(chunks_static) % max(1, int(args.progress_chunks))) == 0:
                    rate = (end / max(1e-6, time.time() - started))
                    print(json.dumps({"progress": end, "total": len(selected), "samples_per_second": rate}), flush=True)
        chunks_static.sort(key=lambda item: int(item["start"]))
        if resume_static:
            static_manifest = _artifact_manifest(
                static_root, static_staging, STATIC_FORMAT, selected, cohort,
                static_params,
                chunks_static,
            )
        else:
            static_manifest = json.loads((static_root / "manifest.json").read_text(encoding="utf-8"))
            chunks_static = static_manifest["chunks"]
        target_manifest = None
        if resume_target:
            chunks_target.sort(key=lambda item: int(item["start"]))
            target_manifest = _artifact_manifest(
                target_root, target_staging, TARGET_FORMAT, selected, cohort,
                target_params,
                chunks_target,
            )
        elif target_root is not None:
            target_manifest = json.loads((target_root / "manifest.json").read_text(encoding="utf-8"))
        if zero_write_snapshot(cache_root) != before:
            raise CacheLifecycleError("static build modified the frozen main bundle")
        if resume_static:
            os.replace(static_staging, static_root)
        if resume_target:
            os.replace(target_staging, target_root)
    except BaseException:
        # Keep staging for explicit, chunk-level resume.  It is never used by
        # readers until the final .frozen manifest is atomically published.
        raise
    finally:
        for lock in locks:
            _release_build_lock(lock)
    result = {
        "status": "PASS", "sample_count": len(selected),
        "geometry_valid_count": sum(1 for _ in ()),
        "publish_state_at_start": publish_state,
        "published_at_start": {"static": bool(static_published), "targets": bool(target_published)},
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
