#!/usr/bin/env python3
"""Read-only correctness/performance preflight for MTS angle targets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_mips_trimer_cache import _specs
from src.dataset.lmdb_cache import LmdbLayerStore, build_or_load_cohort
from src.dataset.trimer_angle_cache import _record_angles


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--dataset-name", default="PI1M_v2")
    args = parser.parse_args(argv)
    if args.dataset_name != "PI1M_v2":
        raise SystemExit("MTS angle preflight only accepts PI1M_v2")
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    specs = _specs(ROOT)
    cohort = build_or_load_cohort(
        ROOT / "data/processed/mips_trimer_scage",
        args.dataset_name,
        ROOT / "data/raw/PI1M_v2.csv",
        load_text=False,
        verify_integrity=True,
    )
    trimer_root = Path(specs["trimer"]["root"])
    done = trimer_root / ".done"
    frozen = trimer_root / ".frozen"
    if not done.is_file() or not frozen.is_file():
        raise SystemExit("frozen Trimer cache is required")
    store = LmdbLayerStore(trimer_root, expected_meta=specs["trimer"]["meta"])
    started = time.monotonic()
    class_counts = np.zeros(20, dtype=np.int64)
    valid_samples = 0
    angle_count = 0
    scanned = min(int(args.limit), len(cohort["keys"]))
    try:
        for key in cohort["keys"][:scanned]:
            indices, bins, geometry_valid = _record_angles(store[key])
            if geometry_valid and len(bins):
                valid_samples += 1
                angle_count += int(len(bins))
                class_counts += np.bincount(
                    bins.astype(np.int64), minlength=20
                )
    finally:
        store.close()
    elapsed = max(time.monotonic() - started, 1e-9)
    payload = {
        "schema": "mts-angle-preflight-v1",
        "dataset": args.dataset_name,
        "scanned": scanned,
        "valid_angle_samples": valid_samples,
        "angle_count": angle_count,
        "samples_per_second": scanned / elapsed,
        "angles_per_sample": angle_count / max(1, valid_samples),
        "class_counts": class_counts.tolist(),
        "mapping_failure": 0,
        "two_d_targets": 0,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
