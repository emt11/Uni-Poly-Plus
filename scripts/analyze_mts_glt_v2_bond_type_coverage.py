#!/usr/bin/env python3
"""Read-only bond-category coverage for the fixed PI1M-v2 audit cohort."""

from __future__ import annotations

from collections import Counter
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.periodic_line_glt import (  # noqa: E402
    NUM_BOND_TYPES,
    PeriodicLineGLTSidecar,
)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _distribution(counter: Counter, total: int):
    return {
        str(category): {
            "count": int(counter.get(category, 0)),
            "fraction": (
                float(counter.get(category, 0) / total) if total else 0.0
            ),
        }
        for category in range(NUM_BOND_TYPES)
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sidecar",
        default=ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_v1/PI1M_v2",
        type=Path,
    )
    parser.add_argument(
        "--sample-ids",
        default=ROOT / "results/mts_glt_v2/alignment_audit_v1/sampled_graph_ids.json",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=ROOT / "results/mts_glt_v2/o8_bond_type_screen_v1/bond_type_coverage.json",
        type=Path,
    )
    args = parser.parse_args(argv)
    sidecar = PeriodicLineGLTSidecar(args.sidecar)
    sampled = json.loads(args.sample_ids.read_text(encoding="utf-8"))
    rows = np.asarray(sampled["sampled_graph_ids"], dtype=np.int64)
    if rows.shape != (20000,) or len(np.unique(rows)) != 20000:
        raise RuntimeError("bond coverage requires the fixed 20,000 unique rows")

    all_counts, internal_counts, cross_counts = Counter(), Counter(), Counter()
    valid = total = 0
    for row_index in rows.tolist():
        row = sidecar.model_row(int(row_index))
        tokens = row["tokens"]
        categories = np.asarray(tokens["token_bond_type"], dtype=np.int64)
        shifts = np.abs(np.asarray(tokens["token_shift"], dtype=np.int64))
        if categories.shape != shifts.shape:
            raise RuntimeError("bond category/shift length mismatch")
        total += int(categories.size)
        category_valid = (categories >= 0) & (categories < NUM_BOND_TYPES)
        valid += int(category_valid.sum())
        all_counts.update(categories[category_valid].tolist())
        internal_counts.update(categories[category_valid & (shifts == 0)].tolist())
        cross_counts.update(categories[category_valid & (shifts != 0)].tolist())

    internal_total = int(sum(internal_counts.values()))
    cross_total = int(sum(cross_counts.values()))
    payload = {
        "schema": "mts-glt-v2-o8-bond-type-coverage-v1",
        "tensor_field": "glt_token_bond_type (sidecar token_bond_type)",
        "source_sidecar": str(args.sidecar.resolve()),
        "sample_ids": str(args.sample_ids.resolve()),
        "sample_count": int(rows.size),
        "num_categories": int(NUM_BOND_TYPES),
        "total_true_chemical_bonds": int(total),
        "valid_bond_type_count": int(valid),
        "valid_bond_type_coverage": float(valid / total) if total else 0.0,
        "all": _distribution(all_counts, valid),
        "internal_ru": _distribution(internal_counts, internal_total),
        "cross_ru": _distribution(cross_counts, cross_total),
        "internal_ru_total": internal_total,
        "cross_ru_total": cross_total,
        "category_contract": {
            "0": "unknown/fallback",
            "1": "single",
            "2": "double, and internal aromatic bonds are conflated here by round(1.5)",
            "3": "triple",
            "4": "aromatic only when the category comes from the Trimer _bond_code path",
            "5": "other",
        },
        "mapping_code": {
            "internal": "canonical_periodic.py uses int(round(BondTypeAsDouble()))",
            "cross_ru": "trimer_mcl.py::_bond_code via periodic_line_glt.py",
        },
        "warning": (
            "This experiment tests the repository's existing six-category "
            "identity. Category 2 does not separate internal double from "
            "internal aromatic bonds."
        ),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
