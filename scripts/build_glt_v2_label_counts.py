#!/usr/bin/env python3
"""Count frozen GLT-v1 line labels in bounded mmap chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modules.periodic_line_glt_v2 import NUM_LINE_LABELS  # noqa: E402


def build_counts(sidecar_root, output, chunk_size=1_000_000):
    sidecar_root = Path(sidecar_root).resolve()
    labels = np.load(sidecar_root / "token_label.npy", mmap_mode="r")
    valid_path = sidecar_root / "token_runtime_valid.npy"
    if not valid_path.is_file():
        valid_path = sidecar_root / "token_valid.npy"
    valid = np.load(valid_path, mmap_mode="r")
    if labels.ndim != 1 or valid.shape != labels.shape:
        raise RuntimeError("line label and valid arrays have incompatible shapes")
    counts = np.zeros(NUM_LINE_LABELS, dtype=np.int64)
    for start in range(0, int(labels.size), int(chunk_size)):
        stop = min(int(labels.size), start + int(chunk_size))
        chunk_labels = np.asarray(labels[start:stop])
        chunk_valid = np.asarray(valid[start:stop], dtype=bool)
        selected = chunk_labels[chunk_valid]
        if selected.size:
            if int(selected.min()) < 0 or int(selected.max()) >= NUM_LINE_LABELS:
                raise RuntimeError("sidecar contains an out-of-range line label")
            counts += np.bincount(selected, minlength=NUM_LINE_LABELS)
    payload = {
        "schema": "mts-glt-v2-line-label-counts-v1",
        "source_sidecar": str(sidecar_root),
        "valid_token_count": int(counts.sum()),
        "nonzero_label_count": int(np.count_nonzero(counts)),
        "counts": {
            str(index): int(value)
            for index, value in enumerate(counts)
            if int(value) > 0
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sidecar-root",
        default="data/processed/mips_trimer_scage/periodic_line_glt_v1/PI1M_v2",
    )
    parser.add_argument(
        "--output", default="results/mts_glt_v2/setup/line_label_counts.json"
    )
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    args = parser.parse_args()
    payload = build_counts(args.sidecar_root, args.output, args.chunk_size)
    print(json.dumps({
        "valid_token_count": payload["valid_token_count"],
        "nonzero_label_count": payload["nonzero_label_count"],
        "output": str(Path(args.output).resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
