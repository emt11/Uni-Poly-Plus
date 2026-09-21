#!/usr/bin/env python3
"""D0 static audit for GLT-3D-GAIN-20260921-01.

Read-only: validates the frozen cohort/static bindings, then reports per
task/fold sample counts, geometry validity, centre-bond availability and label
distribution for the six development units.  No model forward, no optimizer
update, no cache write.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.dataset.glt_dual_cache import load_dual_cohort
from src.dataset.glt_dual_static import DualStaticCache, load_chunk_payload


def _describe(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {'count': 0}
    return {
        'count': int(array.size),
        'mean': float(array.mean()),
        'std': float(array.std(ddof=0)),
        'min': float(array.min()),
        'q25': float(np.quantile(array, 0.25)),
        'median': float(np.median(array)),
        'q75': float(np.quantile(array, 0.75)),
        'max': float(array.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--static-root', required=True)
    parser.add_argument('--split-root', required=True)
    parser.add_argument('--tasks', nargs='+', default=['xc', 'eps', 'eat'])
    parser.add_argument('--folds', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    cohort = load_dual_cohort(args.cohort_root, args.cache_root)
    manifest = cohort['manifest']
    records = cohort['records']

    static = DualStaticCache(
        args.static_root,
        parent_bundle_hash=manifest['main_bundle_hash'],
        cohort_manifest_hash=cohort['manifest_hash'],
    )
    # Same cohort ordered-key binding check as the training reader.
    bound_order = static.manifest.get('cohort_ordered_sample_key_hash')
    if bound_order is not None and bound_order != manifest.get('ordered_sample_key_hash'):
        raise ValueError('static cache cohort ordered-key binding mismatch')

    report = {
        'cohort_root': str(Path(args.cohort_root).resolve()),
        'cohort_manifest_hash': cohort['manifest_hash'],
        'main_bundle_hash': manifest['main_bundle_hash'],
        'cohort_sample_count': int(manifest['sample_count']),
        'cohort_unique_structures': int(manifest['unique_structure_count']),
        'cohort_duplicate_rows': int(manifest['duplicate_count']),
        'cohort_task_counts': manifest['task_counts'],
        'static_root': str(Path(args.static_root).resolve()),
        'static_manifest_hash': static.manifest_hash,
        'static_sample_count': len(static),
        'static_geometry_valid_count': int(static.manifest['geometry_valid_count']),
        'static_geometry_invalid_reason_counts': static.manifest['geometry_invalid_reason_counts'],
        'tasks': {},
    }

    # One pass over the static chunks: (geometry_valid, centre-bond count) per
    # unique structure.  Rows are addressed by the cohort sample key, so the
    # repeated property rows share one structural entry.
    structure_stats = {}
    for item in static.manifest['chunks']:
        arrays = load_chunk_payload(static.root / item['path'], item)
        offsets = np.asarray(arrays['token_offsets'])
        center = np.asarray(arrays['token_center_mask'])
        valid = np.asarray(arrays['geometry_valid'])
        start = int(item['start'])
        for local in range(int(item['count'])):
            key = bytes(static.sample_keys[start + local])
            if bool(valid[local]):
                count = int(center[int(offsets[local]):int(offsets[local + 1])].sum())
            else:
                count = 0
            structure_stats[key] = (bool(valid[local]), count)

    def static_row(key_bytes):
        stats = structure_stats.get(bytes(key_bytes))
        if stats is None:
            raise ValueError('static cache is missing a cohort sample key')
        return stats

    for task in args.tasks:
        task_rows = sorted([row for row in records if row['task'] == task],
                           key=lambda row: int(row['original_row']))
        if [int(row['original_row']) for row in task_rows] != list(range(len(task_rows))):
            raise ValueError(f'{task}: cohort rows are not a contiguous original_row range')
        split_path = Path(args.split_root) / f'{task}.json'
        split = json.loads(split_path.read_text(encoding='utf-8'))
        if int(split['sample_count']) != len(task_rows):
            raise ValueError(f'{task}: split sample_count disagrees with cohort rows')

        labels = np.asarray([float(row['label']) for row in task_rows], dtype=np.float64)
        if not np.isfinite(labels).all():
            raise ValueError(f'{task}: nonfinite label')
        keys = [bytes.fromhex(row['sample_key']) for row in task_rows]
        label_by_key = {key: float(row['label']) for key, row in zip(keys, task_rows)}

        task_entry = {
            'sample_count': len(task_rows),
            'unique_structure_count': len(set(keys)),
            'duplicate_row_count': len(task_rows) - len(set(keys)),
            'label_overall': _describe(labels),
            'label_duplicate_mismatch': int(sum(
                1 for key, count in Counter(keys).items()
                if count > 1 and len({label_by_key[key]}) > 1)),
            'folds': {},
        }

        for fold in split['folds']:
            fold_id = int(fold['fold'])
            if fold_id not in args.folds:
                continue
            fold_entry = {}
            membership = Counter()
            for row in task_rows:
                for item in row['fold_membership']:
                    if int(item['fold']) == fold_id:
                        membership[item['role']] += 1
            for split_name in ('train', 'validation', 'test'):
                indices = [int(value) for value in fold[f'{split_name}_indices']]
                if len(indices) != len(set(indices)):
                    raise ValueError(f'{task}/fold{fold_id}/{split_name}: duplicate indices')
                entries = []
                for index in indices:
                    if index < 0 or index >= len(task_rows):
                        raise ValueError(f'{task}/fold{fold_id}/{split_name}: index out of range')
                    row = task_rows[index]
                    roles = {str(item['role']) for item in row['fold_membership']
                             if int(item['fold']) == fold_id}
                    if roles != {split_name}:
                        raise ValueError(
                            f'{task}/fold{fold_id}/{split_name}: fold_membership disagrees at row {index}')
                    entries.append(row)

                selected_keys = [bytes.fromhex(row['sample_key']) for row in entries]
                invalid = 0
                center_zero_total = 0
                center_zero_valid = 0
                center_counts = []
                for key in selected_keys:
                    valid, count = static_row(key)
                    if not valid:
                        invalid += 1
                    center_counts.append(count)
                    if count == 0:
                        center_zero_total += 1
                        if valid:
                            center_zero_valid += 1
                fold_entry[split_name] = {
                    'rows': len(entries),
                    'unique_structures': len(set(selected_keys)),
                    'geometry_invalid': invalid,
                    'center_bond_zero_total': center_zero_total,
                    'center_bond_zero_valid_only': center_zero_valid,
                    'center_bond_count': _describe(center_counts),
                    'label': _describe([float(row['label']) for row in entries]),
                }
            if membership:
                fold_entry['fold_membership_counts'] = dict(membership)
            task_entry['folds'][str(fold_id)] = fold_entry
        report['tasks'][task] = task_entry

    static.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status': 'PASS', 'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
