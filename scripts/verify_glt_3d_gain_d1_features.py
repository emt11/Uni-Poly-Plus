#!/usr/bin/env python3
"""r2 feature-identity verification for GLT-3D-GAIN-20260921-01.

Read-only check of the d1/feat_deploy archives before they are reused as probe
input: split tag, row order against the fixed split manifest, sample_key and
label against the frozen cohort, and finiteness of every saved array.  No
encoding, no model, no GPU.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

TASKS = ('xc', 'eps', 'eat')
FOLDS = (0, 1)
FINITE_FIELDS = ('z2', 'z3', 'graph_2d', 'graph_3d', 'token_norm_mean')
EXPECTED_DIMS = {'z2': 512, 'z3': 512, 'graph_2d': 512, 'graph_3d': 512}


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def assert_zero_z3_rows(z3, center_counts, geometry_valid):
    """Both degenerate row kinds must carry an exact zero z3.

    The trained representation defines ``z3 = norm3(graph_3d)`` re-zeroed at
    every row where ``geometry_valid & (center_bond_count > 0)`` fails.  That
    covers **two** disjoint populations: rows with no centre bond and rows whose
    geometry is invalid.  Checking only their intersection would silently accept
    a corrupt geometry-invalid row, so both are asserted separately here.
    """
    z3 = np.asarray(z3, dtype=np.float64)
    counts = np.asarray(center_counts, dtype=np.int64)
    valid = np.asarray(geometry_valid, dtype=bool)
    if not (z3.shape[0] == counts.shape[0] == valid.shape[0]):
        raise ValueError('z3 / centre-count / validity rows disagree')
    zero_centre = counts == 0
    invalid = ~valid
    for mask, label in ((zero_centre, 'zero-centre'), (invalid, 'geometry-invalid')):
        if bool(mask.any()) and not bool((np.abs(z3[mask]).sum(axis=1) == 0).all()):
            raise ValueError(f'{label} rows do not carry an exact zero z3')
    return {'zero_centre': int(zero_centre.sum()), 'geometry_invalid': int(invalid.sum()),
            'union': int((zero_centre | invalid).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort-root', required=True)
    parser.add_argument('--split-root', required=True)
    parser.add_argument('--feature-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    records = [json.loads(line) for line in
               Path(args.cohort_root, 'records.jsonl').open(encoding='utf-8')]
    report = {'feature_root': str(Path(args.feature_root).resolve()), 'units': {},
              'checks': ['split tag', 'row order', 'sample_key', 'label',
                         'finiteness', 'shape', 'geometry flags']}
    for task in TASKS:
        rows = sorted([row for row in records if row['task'] == task],
                      key=lambda row: int(row['original_row']))
        if [int(row['original_row']) for row in rows] != list(range(len(rows))):
            raise ValueError(f'{task}: cohort rows are not a contiguous original_row range')
        keys = [row['sample_key'] for row in rows]
        labels = [float(row['label']) for row in rows]
        manifest = json.loads((Path(args.split_root) / f'{task}.json').read_text(encoding='utf-8'))
        for fold in FOLDS:
            entry = next(item for item in manifest['folds'] if int(item['fold']) == fold)
            unit = {}
            for split in ('train', 'validation'):
                indices = [int(value) for value in entry[f'{split}_indices']]
                path = Path(args.feature_root) / f'{task}_fold{fold}_{split}.npz'
                if not path.is_file():
                    raise FileNotFoundError(f'feature archive is missing: {path}')
                with np.load(path, allow_pickle=False) as archive:
                    data = {name: np.asarray(archive[name]) for name in archive.files}
                if str(data['split']) != split:
                    raise ValueError(f'{path}: split tag mismatch: {data["split"]}')
                if data['row_index'].tolist() != indices:
                    raise ValueError(f'{path}: row order differs from the fixed split manifest')
                observed_keys = [bytes(row).hex() for row in data['sample_key']]
                if observed_keys != [keys[index] for index in indices]:
                    raise ValueError(f'{path}: sample_key differs from the frozen cohort')
                expected_labels = np.asarray([labels[index] for index in indices], dtype=np.float64)
                if not np.allclose(data['label'], expected_labels, rtol=0, atol=0):
                    raise ValueError(f'{path}: label differs from the frozen cohort')
                finite = {}
                for name in FINITE_FIELDS:
                    values = data[name]
                    finite[name] = bool(np.isfinite(values).all())
                    if not finite[name]:
                        raise ValueError(f'{path}: {name} contains non-finite values')
                    if name in EXPECTED_DIMS and values.shape != (len(indices), EXPECTED_DIMS[name]):
                        raise ValueError(f'{path}: {name} shape {values.shape} is not '
                                         f'({len(indices)}, {EXPECTED_DIMS[name]})')
                valid = data['geometry_valid'].astype(bool)
                counts = data['center_bond_count'].astype(np.int64)
                degenerate = assert_zero_z3_rows(data['z3'], counts, valid)
                unit[split] = {
                    'rows': int(len(indices)),
                    'dtypes': {name: str(data[name].dtype) for name in
                               ('z2', 'z3', 'graph_2d', 'graph_3d', 'label', 'center_bond_count')},
                    'finite': finite,
                    'geometry_valid': int(valid.sum()),
                    'center_bond_zero': int((counts == 0).sum()),
                    'zero_z3_rows': degenerate,
                    'sample_key_sha256': sha256_bytes(data['sample_key'].tobytes()),
                    'row_index_sha256': sha256_bytes(data['row_index'].astype('<i8').tobytes()),
                    'path': str(path),
                }
                print(json.dumps({'unit': f'{task}/fold{fold}', 'split': split,
                                  'rows': unit[split]['rows'],
                                  'finite': all(finite.values())}), flush=True)
            report['units'][f'{task}/fold{fold}'] = unit

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status': 'PASS', 'output': str(output)}))


if __name__ == '__main__':
    main()
