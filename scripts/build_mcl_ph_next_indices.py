#!/usr/bin/env python3
"""Create directly verifiable activity and pretraining row indices.

The source JSONL files remain immutable. The output contains integer source
rows; consumers must check each requested row's normalized identity against
the source when opening it. No file digest is used as a replacement for that
comparison.
"""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASKS = ('eat', 'eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc')
DOWNSTREAM = ROOT / 'data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1_paper5_outer/records.jsonl'
PI1M = ROOT / 'data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1/records.jsonl'
SPLITS = ROOT / 'data/splits/mips_paper5_outer'


def jsonl(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def role_lists(manifest, count):
    output = []
    for fold in manifest['folds']:
        roles = [None] * count
        for role in ('train', 'validation', 'test'):
            for index in fold[f'{role}_indices']:
                if type(index) is not int or index < 0 or index >= count or roles[index] is not None:
                    raise ValueError('invalid or duplicate split index')
                roles[index] = role
        if any(role is None for role in roles):
            raise ValueError('split leaves rows uncovered')
        output.append(roles)
    if len(output) != 5:
        raise ValueError('expected five folds')
    return output


def build(output):
    if output.exists():
        raise FileExistsError(output)
    raw = {}
    roles = {}
    for task in TASKS:
        with (ROOT / f'data/raw/smi_{task}.csv').open(newline='', encoding='utf-8') as stream:
            raw[task] = list(csv.reader(stream))[1:]
        manifest = json.loads((SPLITS / f'{task}.json').read_text())
        if manifest['task'] != task or manifest['protocol'] != 'paper5_outer' or manifest['sample_count'] != len(raw[task]):
            raise ValueError(f'{task}: activity split metadata mismatch')
        roles[task] = role_lists(manifest, len(raw[task]))
    downstream = []
    seen_task_rows = set()
    excluded = set()
    for source_index, row in enumerate(jsonl(DOWNSTREAM)):
        task, original_row = row['task'], int(row['original_row'])
        if task not in raw or not 0 <= original_row < len(raw[task]):
            raise ValueError('downstream task or row out of range')
        identity = (task, original_row)
        if identity in seen_task_rows:
            raise ValueError('duplicate downstream task/row')
        seen_task_rows.add(identity)
        if row['source_smiles'].strip() != raw[task][original_row][0].strip() or float(row['label']) != float(raw[task][original_row][1]):
            raise ValueError('downstream source SMILES or label differs from raw task table')
        excluded.add(row['normalized_smiles'])
        downstream.append({'source_index': source_index, 'task': task,
                           'original_row': original_row,
                           'normalized_smiles': row['normalized_smiles'],
                           'fold_roles': [roles[task][fold][original_row] for fold in range(5)]})
    if len(seen_task_rows) != sum(map(len, raw.values())):
        raise ValueError('downstream task rows incomplete')
    eligible = []
    overlap = 0
    seen_pi1m = set()
    for source_index, row in enumerate(jsonl(PI1M)):
        identity = row['normalized_smiles']
        if identity in seen_pi1m:
            raise ValueError('PI1M normalized identity repeats')
        seen_pi1m.add(identity)
        if identity in excluded:
            overlap += 1
        else:
            eligible.append(source_index)
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(np.asarray(eligible, dtype=np.int64))
    validation_count = round(len(shuffled) * 0.05)
    fixed = np.sort(shuffled[:1024])
    validation = np.sort(shuffled[:validation_count])
    train = np.sort(shuffled[validation_count:])
    if len(fixed) != 1024 or set(train.tolist()) & set(validation.tolist()):
        raise ValueError('invalid pretraining split')
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'downstream_rows.jsonl').open('x', encoding='utf-8') as stream:
        for row in downstream:
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    np.savez(output / 'pretrain_indices.npz', train_source_indices=train,
             validation_source_indices=validation,
             fixed_validation_source_indices=fixed)
    summary = {'schema': 'mcl-ph-next-direct-index-v1', 'seed': 42,
               'protocol': 'paper5_outer', 'downstream_rows': len(downstream),
               'downstream_unique_normalized_smiles': len(excluded),
               'pi1m_rows': len(seen_pi1m), 'excluded_overlap': overlap,
               'pretrain_train_count': len(train),
               'pretrain_validation_count': len(validation),
               'pretrain_fixed_validation_count': len(fixed),
               'rule': 'exclude all normalized SMILES appearing in any downstream task; '
                       'seed-42 permutation of remaining source rows; 5 percent validation'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))
