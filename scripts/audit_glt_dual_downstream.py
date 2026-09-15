#!/usr/bin/env python3
"""Exhaustively resolve frozen downstream rows into current dual model inputs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.dataset.cache_lifecycle import CacheLifecycleError, zero_write_snapshot
from src.dataset.glt_dual import build_dual_sample
from src.dataset.glt_dual_cache import sha256_file
from src.training.glt_dual_runtime import open_source, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--cohort-root", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cache_root = Path(args.cache_root).resolve()
    split_root = Path(args.split_root).resolve()
    before = zero_write_snapshot(cache_root)
    source, frame = open_source(args.cohort_root, cache_root)
    failures = []
    try:
        manifest = source.cohort["manifest"]
        rows = source.entries
        task_counts = Counter(str(row["task"]) for row in rows)
        if dict(sorted(task_counts.items())) != manifest["task_counts"]:
            raise CacheLifecycleError("downstream task count differs from frozen cohort")
        for task, expected_hash in manifest["split_manifest_sha256"].items():
            if sha256_file(split_root / f"{task}.json") != expected_hash:
                raise CacheLifecycleError(f"fixed split changed: {task}")
        unique_first = {}
        for index, row in enumerate(rows):
            unique_first.setdefault(row["sample_key"], index)
            memberships = row.get("fold_membership")
            if (
                not isinstance(memberships, list) or len(memberships) != 5
                or sorted(int(value["fold"]) for value in memberships) != list(range(5))
                or any(value.get("role") not in {"train", "validation", "test"}
                       for value in memberships)
                or not math.isfinite(float(row["label"]))
            ):
                raise CacheLifecycleError("invalid downstream row/fold/label provenance")

        geometry_valid = 0
        fallback_reasons = Counter()
        for ordinal, (key, index) in enumerate(unique_first.items()):
            try:
                sample = build_dual_sample(*source[index])
                required_finite = (
                    sample.mips_x, sample.bond_path_features,
                    sample.bond_distance, sample.line_angle,
                )
                if any(not bool(torch.isfinite(value).all()) for value in required_finite):
                    raise FloatingPointError("nonfinite model input")
                if sample.geometry_valid:
                    geometry_valid += 1
                    if sample.bond_distance.numel() == 0:
                        raise ValueError("valid geometry has no physical bond tokens")
                else:
                    fallback_reasons[sample.geometry_invalid_reason] += 1
                    if sample.bond_distance.numel() != 0 or sample.line_source.numel() != 0:
                        raise ValueError("geometry fallback leaked partial 3D tokens")
                if sample.mips_x.size(0) == 0 or not bool(sample.graph_available):
                    raise ValueError("downstream row has no usable 2D input")
            except BaseException as exc:
                failures.append({
                    "sample_key": key, "row_index": index,
                    "exception_type": type(exc).__name__, "message": str(exc),
                })
            if failures:
                break
    finally:
        source.close()
    zero_write = zero_write_snapshot(cache_root) == before
    if not zero_write:
        raise CacheLifecycleError("downstream exhaustive audit modified cache")
    report = {
        "status": "PASS" if not failures else "FAIL",
        "row_count": len(rows),
        "unique_structure_count": len(unique_first),
        "resolved_unique_structure_count": len(unique_first) if not failures else None,
        "dropped_downstream_rows": 0 if not failures else None,
        "geometry_valid_count": geometry_valid,
        "geometry_fallback_count": sum(fallback_reasons.values()),
        "geometry_fallback_distribution": dict(sorted(fallback_reasons.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "fixed_split_hashes_verified": not failures,
        "cache_zero_write": True,
        "failures": failures,
    }
    write_json(args.report_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
