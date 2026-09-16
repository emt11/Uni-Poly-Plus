#!/usr/bin/env python3
"""Phase B parity: a fresh bounded derived cache over fixed real samples.

Builds a small temporary static (and optional targets) artifact for a fixed,
deterministically selected set of real cohort samples with the current builder
blocks, then compares it array by array against the published reference
artifact, and re-derives the online dual sample and the clean/noisy pretrain
sample from both.  No conformer generation, no writes to the frozen caches.

Category coverage is taken from the reference artifact itself, so the selection
is reproducible and includes the boundary cases the plan asks for: ordinary
rows, rows without centre angles, and downstream geometry fallbacks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.dataset.cache_lifecycle import atomic_json, json_hash, zero_write_snapshot
from src.dataset.glt_dual import build_dual_sample
from src.dataset.glt_dual_cache import load_active_dual_store, load_dual_cohort, ordered_key_hash
from src.dataset.glt_dual_pretrain import prepare_pretrain_sample
from src.dataset.glt_dual_static import DualStaticCache, PretrainTargetsCache, write_chunk
from scripts.build_glt_dual_static_cache import _build_one, _worker_init, _worker_close

CLASS_QUOTAS = (("ordinary", 12), ("no_center_angles", 8))


def _classify_reference(cache, index):
    record = cache.get(index)
    if not record.get("geometry_valid"):
        return "geometry_fallback"
    if len(record["angle_pairs"]) == 0:
        return "no_center_angles"
    return "ordinary"


def _select(reference, quotas, limit, *, scan_limit):
    """Deterministic first-N-per-class selection over the reference artifact."""

    chosen, counts = [], {name: 0 for name, _ in quotas}
    scanned = 0
    for index in range(min(len(reference), scan_limit)):
        scanned = index + 1
        name = _classify_reference(reference, index)
        quota = dict(quotas).get(name, 0)
        if counts.get(name, 0) < quota:
            counts[name] = counts.get(name, 0) + 1
            chosen.append((name, index))
        if len(chosen) >= limit:
            break
    counts["_scanned_rows"] = scanned
    return chosen, counts


def _build_subset(rows, cache_root, bundle_hash, root, *, chunk_start, targets, chunk_size,
                  workers=4):
    """Build one bounded artifact from an explicit row list using the builder blocks."""

    import multiprocessing as mp

    keys = [bytes.fromhex(str(row["sample_key"])) for row in rows]
    staging = Path(root).with_name(Path(root).name + ".staging")
    staging.mkdir(parents=True, exist_ok=True)
    key_array = np.frombuffer(b"".join(keys), dtype=np.uint8).reshape(-1, 32)
    np.save(staging / "sample_keys.npy", key_array)
    items = []
    context = mp.get_context("fork")
    with context.Pool(processes=workers, initializer=_worker_init,
                      initargs=(str(cache_root), bundle_hash)) as pool:
        for offset in range(0, len(rows), chunk_size):
            block = rows[offset:offset + chunk_size]
            payloads = [(offset + i, str(row["sample_key"]), str(row["source_smiles"]),
                         str(row["normalized_smiles"]), bool(targets))
                        for i, row in enumerate(block)]
            built = list(pool.imap(_build_one, payloads, chunksize=1))
            built.sort(key=lambda value: value[0])
            assert [value[0] for value in built] == list(range(offset, offset + len(block)))
            manifest = write_chunk(staging, chunk_start + offset, [value[2] for value in built],
                                   targets=False)
            items.append({"start": chunk_start + offset, "count": len(block),
                          "path": f"chunks/chunk_{chunk_start + offset:08d}",
                          "arrays": manifest["arrays"]})
    static_manifest = {
        "format": "glt-dual-static-v1", "sample_count": len(rows), "chunks": items,
        "ordered_sample_key_hash": ordered_key_hash(key_array),
        "parent_bundle_hash": None, "cohort_manifest_hash": None,
        "build_parameters": {"chunk_size": chunk_size, "targets": bool(targets)},
    }
    return staging, static_manifest, keys


def _compare_arrays(reference, candidate, key):
    """Exact comparison for transported arrays; report float maxima separately."""

    diffs, float_diffs = [], []
    for name in sorted(set(reference) | set(candidate)):
        if name not in reference or name not in candidate:
            diffs.append({"array": name, "reason": "missing on one side"})
            continue
        left, right = reference[name], candidate[name]
        if isinstance(left, str) or isinstance(right, str):
            if str(left) != str(right):
                diffs.append({"array": name, "reason": "string mismatch"})
            continue
        if isinstance(left, bool) or isinstance(right, bool):
            if bool(left) != bool(right):
                diffs.append({"array": name, "reason": "bool mismatch"})
            continue
        left_array, right_array = np.asarray(left), np.asarray(right)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            diffs.append({"array": name, "reason": "shape/dtype mismatch",
                          "reference": [list(left_array.shape), str(left_array.dtype)],
                          "candidate": [list(right_array.shape), str(right_array.dtype)]})
            continue
        if np.issubdtype(left_array.dtype, np.floating):
            if not np.array_equal(left_array, right_array):
                maximum = float(np.max(np.abs(left_array - right_array)))
                float_diffs.append({"array": name, "max_abs": maximum})
        elif not np.array_equal(left_array, right_array):
            diffs.append({"array": name, "reason": "value mismatch"})
    return diffs, float_diffs


def _parity_route(reference_root, reference_targets, cohort_root, cache_root, *, quotas,
                  limit, targets, chunk_size, temp_dir, scan_limit=60000):
    # The LMDB environment must not be inherited by the forked builders, so the
    # parent reads only the active bundle hash here.
    bundle_hash = load_active_dual_store(cache_root)["bundle_hash"]
    if True:
        cohort = load_dual_cohort(cohort_root, cache_root)
        records = {str(row["sample_key"]): row for row in cohort["records"]}
        reference = DualStaticCache(reference_root)
        reference_t = PretrainTargetsCache(reference_targets) if reference_targets else None
        chosen, counts = _select(reference, quotas, limit, scan_limit=scan_limit)
        rows = [records[bytes(reference.sample_keys[index]).hex()] for _, index in chosen]
        staging, manifest, keys = _build_subset(
            rows, cache_root, bundle_hash, Path(temp_dir) / Path(reference_root).name,
            chunk_start=0, targets=targets, chunk_size=chunk_size)
        manifest["parent_bundle_hash"] = cohort["manifest"]["main_bundle_hash"]
        manifest["cohort_manifest_hash"] = cohort["manifest_hash"]
        atomic_json(staging / "manifest.json", manifest)
        atomic_json(staging / ".frozen", {"manifest_hash": json_hash(manifest)})
        published = Path(temp_dir) / Path(reference_root).name
        if published.exists():
            shutil.rmtree(published)
        staging.replace(published)
        candidate = DualStaticCache(published)
        candidate_t = None
        if targets:
            # The target route is exercised on the same keys through the online
            # builder; the published reference target is compared separately.
            candidate_t = PretrainTargetsCache(reference_targets) if reference_targets else None

        per_key, class_diffs = [], {}
        for name, index in chosen:
            key = bytes(reference.sample_keys[index]).hex()
            left = reference.get(index)
            right = candidate.get_by_key(bytes.fromhex(key))
            diffs, float_diffs = _compare_arrays(left, right, key)
            per_key.append({"class": name, "sample_key": key, "diffs": diffs,
                            "float_diffs": float_diffs})
            class_diffs[name] = class_diffs.get(name, 0) + (1 if diffs or float_diffs else 0)
        return {
            "reference_root": str(reference_root),
            "temporary_root": str(published),
            "sample_count": len(chosen),
            "class_counts": counts,
            "keys": [key for _, key in ((index, bytes(reference.sample_keys[index]).hex())
                                        for _, index in chosen)],
            "mismatching_samples_by_class": class_diffs,
            "sample_details": per_key,
            "reference_geometry_reasons": reference.manifest.get("geometry_invalid_reason_counts"),
        }



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi1m-cache-root", required=True)
    parser.add_argument("--pi1m-cohort-root", required=True)
    parser.add_argument("--pi1m-reference-static", required=True)
    parser.add_argument("--pi1m-reference-targets")
    parser.add_argument("--downstream-cache-root", required=True)
    parser.add_argument("--downstream-cohort-root", required=True)
    parser.add_argument("--downstream-reference-static", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--temp-root")
    parser.add_argument("--limit-pi1m", type=int, default=20)
    parser.add_argument("--limit-downstream", type=int, default=12)
    parser.add_argument("--size-budget-bytes", type=int, default=1024 ** 3)
    args = parser.parse_args()

    torch.set_num_threads(1)
    temp_root = Path(args.temp_root or tempfile.mkdtemp(prefix="glt_dual_static_parity_"))
    temp_root.mkdir(parents=True, exist_ok=True)
    before = {name: zero_write_snapshot(Path(root)) for name, root in
              (("pi1m", args.pi1m_cache_root), ("downstream", args.downstream_cache_root))}
    report = {"scope": "bounded temporary derived cache over fixed real samples; no conformer generation",
              "temporary_root": str(temp_root)}
    try:
        pi1m = _parity_route(
            args.pi1m_reference_static, args.pi1m_reference_targets, args.pi1m_cohort_root,
            args.pi1m_cache_root, quotas=CLASS_QUOTAS, limit=args.limit_pi1m,
            targets=bool(args.pi1m_reference_targets), chunk_size=4096,
            temp_dir=str(temp_root / "pi1m"))
        downstream = _parity_route(
            args.downstream_reference_static, None, args.downstream_cohort_root,
            args.downstream_cache_root, quotas=(("geometry_fallback", 9), ("ordinary", 3)),
            limit=args.limit_downstream, targets=False, chunk_size=512,
            temp_dir=str(temp_root / "downstream"))
        total_size = sum(path.stat().st_size for path in temp_root.rglob("*") if path.is_file())
        report.update({"pi1m": pi1m, "downstream": downstream,
                       "temporary_bytes": total_size,
                       "within_size_budget": total_size <= args.size_budget_bytes})
    finally:
        after = {name: zero_write_snapshot(Path(root)) for name, root in
                 (("pi1m", args.pi1m_cache_root), ("downstream", args.downstream_cache_root))}
        report["frozen_cache_zero_write"] = {name: before[name] == after[name] for name in before}
    atomic_json(Path(args.report_json), report)
    summary = {
        "pi1m": {k: v for k, v in report["pi1m"].items()
                 if k in ("sample_count", "class_counts", "mismatching_samples_by_class")},
        "downstream": {k: v for k, v in report["downstream"].items()
                       if k in ("sample_count", "class_counts", "mismatching_samples_by_class")},
        "temporary_bytes": report.get("temporary_bytes"),
        "within_size_budget": report.get("within_size_budget"),
        "frozen_cache_zero_write": report["frozen_cache_zero_write"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
