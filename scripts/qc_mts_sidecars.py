#!/usr/bin/env python
"""Explicit, read-only quality checks for frozen MTS sidecars.

Training workers use lightweight mmap readers.  This command is the opt-in
place for full-array semantic checks when an operator needs a global QC
answer; it never writes a receipt or changes the sidecar.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.mts_star_rbf_v2 import StarRBFV2Sidecar


def _check_star(root: Path) -> dict:
    reader = StarRBFV2Sidecar(root)
    arrays = reader.arrays
    sample_offsets = np.asarray(arrays["sample_relation_offsets"], dtype=np.int64)
    pair_offsets = np.asarray(arrays["sample_pair_offsets"], dtype=np.int64)
    if np.any(sample_offsets[1:] < sample_offsets[:-1]) or np.any(pair_offsets[1:] < pair_offsets[:-1]):
        raise RuntimeError("Star-RBF offsets are not monotonic")
    relation_count = int(arrays["relation_row"].shape[0])
    pair_count = int(arrays["pair_valid"].shape[0])
    pair_index = np.asarray(arrays["relation_pair_index"], dtype=np.int64)
    if pair_index.size and (pair_index.min() < 0 or pair_index.max() >= pair_count):
        raise RuntimeError("Star-RBF relation pair index out of bounds")
    counts = np.asarray(arrays["pair_observation_count"], dtype=np.int64)
    sources = np.asarray(arrays["pair_geometry_source"], dtype=np.int64)
    valid = np.asarray(arrays["pair_valid"], dtype=bool)
    if np.any(counts > 2):
        raise RuntimeError("Star-RBF observation count exceeds two")
    if np.any((sources == 1) & ((counts != 0) | (~valid))):
        raise RuntimeError("Star-RBF trivial-self contract mismatch")
    if np.any(valid & (sources != 1) & (counts == 0)):
        raise RuntimeError("Star-RBF valid pair has no observation")
    for name, value in arrays.items():
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise RuntimeError(f"non-finite Star-RBF array: {name}")
    return {"type": "star_rbf_v2", "samples": len(reader), "relations": relation_count, "pairs": pair_count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--type", choices=("star_rbf_v2",), required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result = _check_star(root)
    print(json.dumps({"status": "ok", **result}, sort_keys=True))


if __name__ == "__main__":
    main()
