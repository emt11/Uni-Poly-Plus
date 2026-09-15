#!/usr/bin/env python3
"""Compare persistent dual static rows with the reference runtime path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.dataset.cache_lifecycle import atomic_json, zero_write_snapshot
from src.dataset.glt_dual import build_dual_sample
from src.dataset.glt_dual_cache import DualFrozenBundle, load_dual_cohort
from src.dataset.glt_dual_pretrain import prepare_pretrain_sample
from src.dataset.glt_dual_static import (DualStaticCache, PretrainTargetsCache,
                                          build_pretrain_target)


def _same(left, right, *, atol=1e-6):
    if torch.is_tensor(left) or torch.is_tensor(right):
        left, right = torch.as_tensor(left), torch.as_tensor(right)
        if left.dtype.is_floating_point or right.dtype.is_floating_point:
            return bool(torch.allclose(left, right, atol=atol, rtol=atol))
        return bool(torch.equal(left, right))
    if hasattr(left, 'shape') or hasattr(right, 'shape'):
        import numpy as np
        left, right = np.asarray(left), np.asarray(right)
        return bool(np.allclose(left, right, atol=atol, rtol=atol)) if (
            np.issubdtype(left.dtype, np.floating) or np.issubdtype(right.dtype, np.floating)
        ) else bool(np.array_equal(left, right))
    return bool(left == right)


def verify(args):
    cache_root = Path(args.cache_root).resolve()
    cohort = load_dual_cohort(args.cohort_root, cache_root)
    static = DualStaticCache(
        args.static_root,
        parent_bundle_hash=cohort['manifest']['main_bundle_hash'],
        cohort_manifest_hash=cohort['manifest_hash'],
    )
    target = None
    if args.target_root:
        target = PretrainTargetsCache(
            args.target_root,
            parent_bundle_hash=cohort['manifest']['main_bundle_hash'],
            cohort_manifest_hash=cohort['manifest_hash'],
        )
    before = zero_write_snapshot(cache_root)
    started = time.time()
    try:
        bundle = DualFrozenBundle(cache_root, expected_bundle_hash=cohort['manifest']['main_bundle_hash'])
        by_key = {bytes.fromhex(str(row['sample_key'])): row for row in cohort['records']}
        available = min(int(args.count), len(static))
        if target is not None and len(target) != len(static):
            raise ValueError('static/target sample counts differ')
        checked = []
        for index in range(available):
            key = bytes(static.sample_keys[index])
            if key not in by_key:
                raise ValueError(f'static sample key is absent from cohort: {key.hex()}')
            row = by_key[key]
            topology, trimer = bundle.topology[key], bundle.trimer[key]
            reference = build_dual_sample(topology, trimer, row['source_smiles'])
            cached = build_dual_sample(
                topology, trimer, row['source_smiles'], static=static.get(index)
            )
            for name in (
                'bond_path_features', 'bond_path_mask', 'bond_z_a', 'bond_z_b',
                'bond_distance', 'bond_type', 'bond_center', 'line_source',
                'line_target', 'line_path', 'line_angle', 'line_mask',
                'line_path_group', 'line_is_self', 'geometry_valid',
                'geometry_invalid_reason',
            ):
                if not _same(getattr(reference, name), getattr(cached, name)):
                    raise AssertionError(f'cached/reference mismatch at row {index}: {name}')
            if target is not None:
                target_row = target.get(index)
                expected_target = build_pretrain_target(row['normalized_smiles'])
                observed_groups = tuple(tuple(int(atom) for atom in group)
                                        for group in target_row['brics_groups'])
                if observed_groups != expected_target['brics_groups']:
                    raise AssertionError(f'BRICS mismatch at row {index}')
                if not _same(target_row['fingerprint_packed'], expected_target['fingerprint_packed']):
                    raise AssertionError(f'fingerprint mismatch at row {index}')
                old_input, old_labels = prepare_pretrain_sample(
                    topology, trimer, row['source_smiles'], seed=20260915,
                    key=key.hex(), position=index,
                )
                new_input, new_labels = prepare_pretrain_sample(
                    topology, trimer, row['source_smiles'], seed=20260915,
                    key=key.hex(), position=index, static=static.get(index),
                    target=target_row,
                )
                for name in old_input.keys():
                    if not _same(getattr(old_input, name), getattr(new_input, name)):
                        raise AssertionError(f'pretrain input mismatch at row {index}: {name}')
                for name in old_labels.keys():
                    if not _same(old_labels[name], new_labels[name]):
                        raise AssertionError(f'pretrain target mismatch at row {index}: {name}')
            checked.append(index)
        bundle.close()
    finally:
        static.close()
        if target is not None:
            target.close()
    if zero_write_snapshot(cache_root) != before:
        raise RuntimeError('verification modified the frozen main bundle')
    fields = set(static.manifest.get('chunks', [{}])[0].get('arrays', {})) if static.manifest.get('chunks') else set()
    if 'token_distance' in fields or 'line_angle' in fields:
        raise AssertionError('static artifact stores clean geometry forbidden for noisy encoder')
    return {
        'status': 'PASS', 'checked': len(checked), 'target_checked': target is not None,
        'static_manifest_hash': static.manifest_hash,
        'target_manifest_hash': target.manifest_hash if target is not None else None,
        'elapsed_seconds': time.time() - started,
        'scope': 'specified static artifact rows only',
        'main_cache_zero_write': True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--cohort-root', required=True)
    parser.add_argument('--static-root', required=True)
    parser.add_argument('--target-root')
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--report-json')
    args = parser.parse_args()
    try:
        result = verify(args)
    except Exception as exc:
        result = {'status': 'FAIL', 'error_type': type(exc).__name__, 'error': str(exc)}
        if args.report_json:
            atomic_json(Path(args.report_json), result)
        print(json.dumps(result, indent=2, sort_keys=True))
        raise
    if args.report_json:
        atomic_json(Path(args.report_json), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
