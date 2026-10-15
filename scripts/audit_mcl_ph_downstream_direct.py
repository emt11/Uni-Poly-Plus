#!/usr/bin/env python3
"""Independently read active task rows and compare static versus uncached geometry."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_mcl_ph_next_indices import TASKS
from src.dataset.glt_dual import build_dual_sample
from src.dataset.mcl_ph_direct import DirectMCLSource


def audit(args):
    if args.report.exists():
        raise FileExistsError(args.report)
    seen, counts, task_counts = set(), Counter(), {}
    for task in TASKS:
        with (ROOT / f'data/raw/smi_{task}.csv').open(newline='') as stream:
            raw = list(csv.reader(stream))[1:]
        split = json.loads((ROOT / f'data/splits/mips_paper5_outer/{task}.json').read_text())
        source = DirectMCLSource(args.source, task=task, selected_indices=None,
                                 split_manifest=split)
        try:
            if len(source) != len(raw):
                raise ValueError('task source row count differs')
            for index, row in enumerate(source.entries):
                if (row['source_smiles'].strip() != raw[index][0].strip()
                        or float(row['label']) != float(raw[index][1])):
                    raise ValueError(f'{task}/{index}: raw identity or label differs')
                key = source.samples[index][0]
                if key in seen:
                    continue
                record = source.record(index)
                expected = build_dual_sample(record.topology, record.trimer, row['source_smiles'])
                actual = build_dual_sample(record.topology, record.trimer, row['source_smiles'],
                                            static=record.static)
                if expected.geometry_valid != actual.geometry_valid:
                    raise ValueError('static versus uncached validity differs')
                for field in ('mips_x', 'bond_path_features', 'bond_path_mask', 'bond_z_a',
                              'bond_z_b', 'bond_distance', 'bond_type', 'bond_center',
                              'line_source', 'line_target', 'line_path', 'line_path_group',
                              'line_mask', 'line_angle', 'line_is_self'):
                    torch.testing.assert_close(getattr(actual, field), getattr(expected, field),
                                               atol=1e-5, rtol=1e-5, msg=f'{task}/{index}/{field}')
                counts[row['geometry_status']] += 1
                seen.add(key)
                if len(seen) % 100 == 0:
                    print(json.dumps({'unique_audited': len(seen), 'counts': dict(counts)}), flush=True)
            task_counts[task] = len(source)
        finally:
            source.close()
    if len(seen) != 3655 or sum(task_counts.values()) != 6265:
        raise ValueError('full direct downstream identity count differs')
    result = {'conclusion': 'PASS', 'unique_identities': len(seen),
              'task_rows': task_counts, 'geometry_status': dict(counts),
              'scope': 'all task labels and unique static interfaces; no model'}
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    audit(parser.parse_args())
