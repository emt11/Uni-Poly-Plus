#!/usr/bin/env python3
"""Materialize the frozen PI1M_v2 Trimer bond-angle target cache."""

from __future__ import annotations

import argparse
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
    args = parser.parse_args(argv)
    if args.dataset_name != "PI1M_v2":
        raise SystemExit("MTS pretraining angle cache only accepts PI1M_v2")
    source_csv = (PROJECT_ROOT / args.source_csv).resolve()
    specs = _specs(PROJECT_ROOT)
    cohort = build_or_load_cohort(
        PROJECT_ROOT / "data/processed/mips_trimer_scage",
        args.dataset_name,
        source_csv,
        load_text=False,
        verify_integrity=True,
    )
    trimer_root = Path(specs["trimer"]["root"])
    trimer_done = trimer_root / ".done"
    if not trimer_done.is_file() or not (trimer_root / ".frozen").is_file():
        raise SystemExit(f"Trimer cache is not frozen: {trimer_root}")
    artifact_hash = trimer_done.read_text(encoding="utf-8").strip()
    trimer_store = LmdbLayerStore(trimer_root, expected_meta=specs["trimer"]["meta"])
    try:
        root, metadata = build_angle_cache(
            cohort, trimer_store, trimer_root, artifact_hash
        )
    finally:
        trimer_store.close()
    print(f"MTS angle cache ready: {root}")
    print(f"records={metadata['record_count']} angles={metadata['angle_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
