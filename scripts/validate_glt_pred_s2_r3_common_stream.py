#!/usr/bin/env python3
"""Compare the shared sample/RNG stream across all S2 third objectives."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from src.dataset.glt_dual_pretrain import prepare_pretrain_sample
from src.training.glt_dual_runtime import OrderedSampleStream, open_source


PUBLIC_DATA_FIELDS = (
    'mips_x', 'mips_backbone_mask', 'lga_edge_index', 'lga_spd',
    'lga_path_index', 'lga_path_mask', 'lga_path_shift',
    'lga_source_image_shift', 'bond_path_features', 'bond_path_mask',
    'bond_z_a', 'bond_z_b', 'bond_distance', 'bond_type', 'bond_center',
    'line_source', 'line_target', 'line_path', 'line_angle', 'line_mask',
    'line_path_group', 'line_is_self',
)
PUBLIC_LABEL_FIELDS = (
    'atom_mask', 'atom_label', 'distance', 'angle_pairs', 'angle_cos',
    'fallback', 'skip_reasons',
)


def _value_equal(left, right):
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left, right)
    return left == right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--dual-static-root', required=True)
    parser.add_argument('--pretrain-target-root', required=True)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.samples < 32:
        raise ValueError('--samples must be at least 32')
    source, _ = open_source(
        args.cohort_root, args.cache_root,
        dual_static_root=args.dual_static_root,
        pretrain_target_root=args.pretrain_target_root,
    )
    stream = OrderedSampleStream(len(source), args.seed)
    positions = list(range(args.samples))
    records = []
    try:
        for position in positions:
            index = stream.index_at(position)
            key, _ = source.samples[index]
            rows = {}
            for task in ('fp', 'none', 'fgr', 'align'):
                rows[task] = prepare_pretrain_sample(
                    *source[index], seed=args.seed, key=key.hex(), position=position,
                    sigma=0.03, ratio=0.3, static=source.static_for(index),
                    target=(source.target_for(index) if task == 'fp' else None),
                    third_task=task, fgr_mu=0.0, fgr_sigma=1.0, fgr_max_pairs=32,
                )
            reference_data, reference_labels = rows['fp']
            mismatches = []
            for task in ('none', 'fgr', 'align'):
                data, labels = rows[task]
                for field in PUBLIC_DATA_FIELDS:
                    if not _value_equal(getattr(reference_data, field), getattr(data, field)):
                        mismatches.append(f'{task}.data.{field}')
                for field in PUBLIC_LABEL_FIELDS:
                    if not _value_equal(reference_labels[field], labels[field]):
                        mismatches.append(f'{task}.labels.{field}')
            records.append({
                'position': position, 'source_index': index, 'sample_key': key.hex(),
                'mismatch_fields': mismatches, 'exact': not mismatches,
                'third_task_fields': {
                    'fp_fingerprint_only': True,
                    'fgr_pair_target_only': True,
                    'align_identity_metadata_only': True,
                },
            })
    finally:
        source.close()
    mismatch_count = sum(not record['exact'] for record in records)
    report = {
        'status': 'PASS' if mismatch_count == 0 else 'FAIL',
        'sample_count': len(records), 'seed': int(args.seed),
        'absolute_positions': positions,
        'shared_fields_exact': mismatch_count == 0,
        'outer_test_accessed': False, 'active_cache_modified': False,
        'records': records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('status', 'sample_count', 'shared_fields_exact')}, sort_keys=True))
    if mismatch_count:
        raise AssertionError(f'common stream parity failed for {mismatch_count} records')


if __name__ == '__main__':
    main()
