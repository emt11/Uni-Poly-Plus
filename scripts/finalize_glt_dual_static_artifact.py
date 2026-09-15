#!/usr/bin/env python3
"""Add deterministic audit counts/spec identity to a published static artifact."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.dataset.cache_lifecycle import atomic_json, json_hash


def finalize(root):
    root = Path(root).resolve()
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    valid, reasons = 0, Counter()
    if manifest['format'].endswith('static-v1'):
        for item in manifest['chunks']:
            chunk = root / item['path']
            values = np.load(chunk / 'geometry_valid.npy', mmap_mode='r')
            valid += int(np.asarray(values, dtype=bool).sum())
            reasons.update(str(value) for value in np.load(chunk / 'geometry_invalid_reason.npy', mmap_mode='r').tolist())
    spec = {
        'format': manifest['format'],
        'parent_bundle_hash': manifest['parent_bundle_hash'],
        'cohort_manifest_hash': manifest['cohort_manifest_hash'],
        'parameters': manifest.get('build_parameters', {}),
    }
    manifest['build_spec_hash'] = manifest.get('build_spec_hash', json_hash(spec))
    if manifest['format'].endswith('static-v1'):
        manifest['geometry_valid_count'] = int(valid)
        manifest['geometry_invalid_reason_counts'] = dict(sorted(reasons.items()))
    atomic_json(manifest_path, manifest)
    atomic_json(root / '.frozen', {'manifest_hash': json_hash(manifest)})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    manifest = finalize(args.root)
    print(json.dumps({
        'status': 'PASS', 'root': str(Path(args.root).resolve()),
        'sample_count': manifest['sample_count'],
        'geometry_valid_count': manifest.get('geometry_valid_count'),
        'manifest_hash': json_hash(manifest),
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
