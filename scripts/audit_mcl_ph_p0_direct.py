#!/usr/bin/env python3
"""Read-only, row-by-row MCL-PH source and active-fold audit."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = ('eat', 'eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc')
DOWNSTREAM = ROOT / 'data/processed/glt_dual_v2/downstream'
COHORT = DOWNSTREAM / 'cohort_1545eda5a8f6a1_paper5_outer/records.jsonl'
UNION = DOWNSTREAM / 'union_paper5_outer/rows.jsonl'
PI1M = ROOT / 'data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1/records.jsonl'


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def active_roles(task, count):
    manifest = json.loads((ROOT / f'data/splits/mips_paper5_outer/{task}.json').read_text())
    if manifest['sample_count'] != count or manifest['task'] != task:
        raise ValueError(f'{task}: active split task/count mismatch')
    folds = []
    for fold in manifest['folds']:
        roles = [None] * count
        for role in ('train', 'validation', 'test'):
            for index in fold[f'{role}_indices']:
                if not 0 <= index < count or roles[index] is not None:
                    raise ValueError(f'{task}: invalid fold {fold["fold"]} index {index}')
                roles[index] = role
        if any(value is None for value in roles):
            raise ValueError(f'{task}: incomplete fold {fold["fold"]}')
        folds.append(roles)
    return folds


def audit():
    raw = {}
    roles = {}
    for task in TASKS:
        with (ROOT / f'data/raw/smi_{task}.csv').open(newline='', encoding='utf-8') as stream:
            raw[task] = list(csv.reader(stream))[1:]
        roles[task] = active_roles(task, len(raw[task]))
    counts = Counter()
    mismatches = defaultdict(Counter)
    examples = defaultdict(list)
    downstream_smiles = set()
    with COHORT.open(encoding='utf-8') as cohort, UNION.open(encoding='utf-8') as union:
        while True:
            a_line, b_line = cohort.readline(), union.readline()
            if not a_line and not b_line:
                break
            if not a_line or not b_line:
                raise ValueError('cohort and union row counts differ')
            a, b = json.loads(a_line), json.loads(b_line)
            task, index = a['task'], int(a['original_row'])
            if task not in raw or not 0 <= index < len(raw[task]):
                raise ValueError('unexpected task or row index')
            counts[task] += 1
            downstream_smiles.add(a['normalized_smiles'])
            def mark(kind, extra=None):
                mismatches[task][kind] += 1
                if len(examples[task]) < 3:
                    examples[task].append({'row': index, 'kind': kind, 'detail': extra})
            if a != b:
                mark('cohort_union_row')
            if a['source_smiles'].strip() != raw[task][index][0].strip():
                mark('raw_smiles')
            if float(a['label']) != float(raw[task][index][1]):
                mark('raw_label')
            if len(a['fold_membership']) != 5:
                mark('fold_count')
                continue
            for fold in range(5):
                observed = a['fold_membership'][fold]
                expected = roles[task][fold][index]
                if observed != {'fold': fold, 'role': expected}:
                    mark('fold_role', {'fold': fold, 'observed': observed, 'expected': expected})
    for task in TASKS:
        if counts[task] != len(raw[task]):
            mismatches[task]['row_count'] += 1
    pi1m_count = overlap = 0
    pi1m_unique = set()
    for row in rows(PI1M):
        pi1m_count += 1
        identity = row['normalized_smiles']
        pi1m_unique.add(identity)
        overlap += identity in downstream_smiles
    return {'status': 'PASS' if not any(mismatches.values()) else 'MISMATCH',
            'task_rows': dict(counts),
            'mismatches': {task: dict(mismatches[task]) for task in TASKS},
            'examples': dict(examples),
            'downstream_unique_normalized_smiles': len(downstream_smiles),
            'pi1m_rows': pi1m_count, 'pi1m_unique_normalized_smiles': len(pi1m_unique),
            'pi1m_rows_overlapping_downstream_identity': overlap,
            'pi1m_unique_overlapping_downstream_identity': len(pi1m_unique & downstream_smiles)}


if __name__ == '__main__':
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
