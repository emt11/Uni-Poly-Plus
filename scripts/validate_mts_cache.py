#!/usr/bin/env python3
"""Read-only exact-union validation for the frozen MTS cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.mts_cache_integrity import validate_targets_parallel  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohorts",
        default="PI1M_v2 downstream_union",
        help="space/comma separated cohort names",
    )
    parser.add_argument("--cohort", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--cache-root",
        default="data/processed/mips_trimer_scage",
    )
    parser.add_argument(
        "--result-root",
        default="results/mts_canonical_migration",
    )
    parser.add_argument("--source-csv", default=None)
    parser.add_argument("--target-root", default=None)
    parser.add_argument("--target-topology-root", default=None)
    parser.add_argument("--target-trimer-root", default=None)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-chunk", type=int, default=256)
    parser.add_argument(
        "--full",
        action="store_true",
        help="run the complete exact-union scan (the default behavior)",
    )
    args = parser.parse_args(argv)
    report = validate_targets_parallel(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
