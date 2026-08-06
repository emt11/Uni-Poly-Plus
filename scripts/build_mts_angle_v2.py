#!/usr/bin/env python3
"""Build the optional Angle-v2 cosine sidecar without regenerating Trimers."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_mips_trimer_cache import _specs
from src.dataset.lmdb_cache import LmdbLayerStore, build_or_load_cohort
from src.dataset.trimer_angle_continuous_cache import build_continuous_angle_cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-name", default="PI1M_v2", choices=("PI1M_v2",)
    )
    parser.parse_args(argv)
    specs = _specs(ROOT)
    cohort = build_or_load_cohort(
        ROOT / "data/processed/mips_trimer_scage", "PI1M_v2",
        ROOT / "data/raw/PI1M_v2.csv", load_text=False, verify_integrity=True,
    )
    root = Path(specs["trimer"]["root"])
    artifact = (root / ".done").read_text().strip()
    store = LmdbLayerStore(root, expected_meta=specs["trimer"]["meta"])
    try:
        output, metadata = build_continuous_angle_cache(
            cohort, store, root, artifact
        )
    finally:
        store.close()
    print(f"Angle-v2 cache: {output} angles={metadata['angle_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
