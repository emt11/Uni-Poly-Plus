#!/usr/bin/env python3
"""Add deterministic audit counts/spec identity to a published static artifact.

The command must not silently rewrite published identity.  Aggregation and the
checks it performs belong before freezing: if the artifact is already frozen
(``.frozen`` present), this command either finds the manifest already final
(idempotent no-op) or refuses and, on request, writes a standalone diagnostic
report describing the difference without touching the artifact.

The supported workflow for an unfrozen staging root is unchanged: finalized
counts are computed there and the caller freezes by publishing it.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.dataset.cache_lifecycle import CacheLifecycleError, atomic_json, json_hash
from src.dataset.glt_dual_static import TARGET_FORMAT, load_chunk_payload


def _final_manifest(manifest, root):
    """Return the manifest this command would publish, without writing it."""

    manifest = json.loads(json.dumps(manifest))
    valid, reasons = 0, Counter()
    expected_start = 0
    targets = manifest.get("format") == TARGET_FORMAT
    for item in sorted(manifest.get("chunks", []), key=lambda value: int(value.get("start", -1))):
        start, count = int(item.get("start", -1)), int(item.get("count", -1))
        if start != expected_start or count < 0:
            raise CacheLifecycleError("artifact chunks are not contiguous")
        if "target" in item and bool(item.get("target")) != targets:
            raise CacheLifecycleError("artifact chunk target flag does not match format")
        chunk = Path(root) / str(item.get("path", ""))
        if not (chunk / ".complete").is_file():
            raise CacheLifecycleError(f"artifact chunk is not complete: {chunk}")
        load_chunk_payload(chunk, item, targets=targets)
        expected_start += count
    if expected_start != int(manifest.get("sample_count", -1)):
        raise CacheLifecycleError("artifact chunk count does not match sample_count")
    if manifest['format'].endswith('static-v1'):
        for item in manifest['chunks']:
            chunk = Path(root) / item['path']
            values = np.load(chunk / 'geometry_valid.npy', mmap_mode='r')
            valid += int(np.asarray(values, dtype=bool).sum())
            reasons.update(str(value) for value in
                           np.load(chunk / 'geometry_invalid_reason.npy', mmap_mode='r').tolist())
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
    return manifest


def finalize(root, *, diagnostic_report=None):
    root = Path(root).resolve()
    manifest_path = root / 'manifest.json'
    frozen_path = root / '.frozen'
    if not manifest_path.is_file():
        raise CacheLifecycleError(f"artifact has no manifest.json: {root}")
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    frozen = None
    if frozen_path.is_file():
        frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    candidate = _final_manifest(manifest, root)
    if frozen is None:
        atomic_json(manifest_path, candidate)
        atomic_json(frozen_path, {'manifest_hash': json_hash(candidate)})
        return candidate, 'PUBLISHED_UNFROZEN'
    current_hash = json_hash(manifest)
    if frozen != {'manifest_hash': current_hash}:
        raise CacheLifecycleError(
            f"refusing to touch an artifact whose .frozen does not bind its manifest: {root}")
    if candidate == manifest:
        return manifest, 'IDEMPOTENT_ALREADY_FINAL'
    differing = sorted(
        name for name in set(manifest) | set(candidate)
        if manifest.get(name) != candidate.get(name))
    if diagnostic_report:
        atomic_json(Path(diagnostic_report), {
            'status': 'REFUSED_FROZEN_ARTIFACT',
            'root': str(root),
            'manifest_hash': current_hash,
            'frozen_manifest_hash': frozen['manifest_hash'],
            'differing_fields': differing,
            'note': 'diagnostic only; the frozen artifact was not modified',
        })
    raise CacheLifecycleError(
        "refusing to modify a frozen artifact; aggregation and validation must happen before "
        f"freezing (differing fields: {','.join(differing) or 'none'})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--diagnostic-report',
                        help='write a standalone refusal report here instead of only failing')
    args = parser.parse_args()
    manifest, outcome = finalize(args.root, diagnostic_report=args.diagnostic_report)
    print(json.dumps({
        'status': 'PASS', 'outcome': outcome, 'root': str(Path(args.root).resolve()),
        'sample_count': manifest['sample_count'],
        'geometry_valid_count': manifest.get('geometry_valid_count'),
        'manifest_hash': json_hash(manifest),
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
