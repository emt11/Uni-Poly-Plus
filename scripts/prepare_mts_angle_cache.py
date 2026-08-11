#!/usr/bin/env python3
"""Materialize the frozen PI1M_v2 Trimer bond-angle target cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_mips_trimer_cache import _specs  # noqa: E402
from src.dataset.lmdb_cache import LmdbLayerStore, build_or_load_cohort  # noqa: E402
from src.dataset.trimer_angle_cache import build_angle_cache  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="PI1M_v2")
    parser.add_argument("--source-csv", default="data/raw/PI1M_v2.csv")
    parser.add_argument(
        "--cohorts",
        default=None,
        help="space/comma separated cohort names (overrides --dataset-name)",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-chunk", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    specs = _specs(PROJECT_ROOT)
    trimer_root = Path(specs["trimer"]["root"])
    trimer_done = trimer_root / ".done"
    if not trimer_done.is_file() or not (trimer_root / ".frozen").is_file():
        raise SystemExit(f"Trimer cache is not frozen: {trimer_root}")
    artifact_hash = trimer_done.read_text(encoding="utf-8").strip()
    trimer_store = LmdbLayerStore(trimer_root, expected_meta=specs["trimer"]["meta"])
    try:
        names = (
            [item for item in str(args.cohorts).replace(",", " ").split() if item]
            if args.cohorts
            else [args.dataset_name]
        )
        outputs = []
        for name in names:
            if args.cohorts and len(names) == 1 and args.source_csv != "data/raw/PI1M_v2.csv":
                source_csv = (PROJECT_ROOT / args.source_csv).resolve()
            else:
                filename = "smi_all.csv" if name == "downstream_union" else f"{name}.csv"
                source_csv = PROJECT_ROOT / "data" / "raw" / filename
            cohort = build_or_load_cohort(
                PROJECT_ROOT / "data/processed/mips_trimer_scage",
                name,
                source_csv,
                load_text=False,
                verify_integrity=True,
            )
            root, metadata = build_angle_cache(
                cohort,
                trimer_store,
                trimer_root,
                artifact_hash,
                workers=max(1, args.workers),
                chunk_size=max(1, args.batch_chunk),
            )
            outputs.append({
                "cohort": name,
                "root": str(root),
                "record_count": metadata["record_count"],
                "angle_count": metadata["angle_count"],
            })
    finally:
        trimer_store.close()
    print(json.dumps({"command": "build-angle-cache", "outputs": outputs}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
