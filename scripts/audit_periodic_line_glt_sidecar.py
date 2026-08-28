#!/usr/bin/env python3
"""Explicit offline QC for the retained MTS-GLT-v2 line sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.periodic_line_glt import PeriodicLineGLTSidecar  # noqa: E402


def audit(root):
    sidecar = PeriodicLineGLTSidecar(root)
    arrays = sidecar.arrays
    graph_valid = np.asarray(arrays["graph_geometry_valid"], dtype=bool)
    token_valid = np.asarray(arrays["token_valid"], dtype=bool)
    relation_valid = np.asarray(arrays["relation_valid"], dtype=bool)
    token_counts = np.asarray(arrays["token_observation_count"], dtype=np.int64)
    relation_counts = np.asarray(arrays["relation_observation_count"], dtype=np.int64)
    if token_counts.size and not np.isin(token_counts, [0, 2, 3]).all():
        raise RuntimeError("unexpected line-token observation count")
    if relation_counts.size and not np.isin(relation_counts, [0, 1, 2, 3]).all():
        raise RuntimeError("unexpected line-relation observation count")
    return {
        "schema": sidecar.metadata["schema"],
        "records": len(sidecar),
        "geometry_valid_records": int(graph_valid.sum()),
        "geometry_valid_fraction": float(graph_valid.mean()) if graph_valid.size else 0.0,
        "tokens": int(token_valid.size),
        "valid_tokens": int(token_valid.sum()),
        "relations": int(relation_valid.size),
        "valid_relations": int(relation_valid.sum()),
        "distance_observation_count_histogram": {
            str(value): int((token_counts == value).sum())
            for value in sorted(set(token_counts.tolist()))
        },
        "angle_observation_count_histogram": {
            str(value): int((relation_counts == value).sum())
            for value in sorted(set(relation_counts.tolist()))
        },
        "bond_type_model_input": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit(args.root)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
