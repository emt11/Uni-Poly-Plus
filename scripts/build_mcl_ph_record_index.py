#!/usr/bin/env python3
"""Prepare a trusted byte-offset index for the frozen downstream cohort.

Preparation scans the complete JSONL byte stream once. Training uses the
resulting index to seek only to its train/validation rows.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.glt_dual_cache import build_dual_cohort_record_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    payload = build_dual_cohort_record_index(args.cohort_root, args.output)
    print(json.dumps({key: value for key, value in payload.items()
                      if key not in ('offsets', 'lengths')}, sort_keys=True))


if __name__ == '__main__':
    main()
