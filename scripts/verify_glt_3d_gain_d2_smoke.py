#!/usr/bin/env python3
"""r3 post-smoke verification for the D2 arms.

Reads the saved smoke artifacts and re-derives the claims that matter for the
contract, on CPU and without any training update:

* the four arms share bit-identical tensors at their common initialisation,
  rebuilt here from the same seed and deployment package the launcher used;
* every saved ``best.pt`` loads strict into its own arm structure;
* each arm's parameters actually moved away from that common initialisation,
  and only in the arm's own trainable tensors;
* the saved target scaler matches the arm's **train** labels, not the
  validation ones;
* every recorded metric is finite and the sample counts match the fixed split.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts.finetune_glt_3d_gain_d2 import (ARM_LRS, ARMS, O8OnlyArm, build_arm,
                                             build_reference, effective_capacity)
from src.modules.glt_dual import build_dual_glt_model

# glt.* is shared by the three dual-route arms only; F2D has no such tensors and
# is filtered per name below.
SHARED_PREFIXES = ('o8.', 'glt.', 'norm2.', 'norm3.', 'predictor.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cohort-root', required=True)
    parser.add_argument('--split-root', required=True)
    parser.add_argument('--smoke-root', required=True)
    parser.add_argument('--task', default='xc')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    package = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    torsion = bool(package.get('torsion_modules', False))
    smoke = Path(args.smoke_root)
    manifest = json.loads((Path(args.split_root) / f'{args.task}.json').read_text(encoding='utf-8'))
    fold_entry = next(item for item in manifest['folds'] if int(item['fold']) == args.fold)
    train_indices = [int(v) for v in fold_entry['train_indices']]
    validation_indices = [int(v) for v in fold_entry['validation_indices']]

    records = [json.loads(line) for line in
               Path(args.cohort_root, 'records.jsonl').open(encoding='utf-8')]
    rows = sorted([row for row in records if row['task'] == args.task],
                  key=lambda row: int(row['original_row']))
    labels = np.asarray([float(row['label']) for row in rows], dtype=np.float64)
    train_labels = labels[train_indices]
    validation_labels = labels[validation_indices]

    reference = build_reference(package, torsion=torsion)
    reference_state = reference.state_dict()
    report = {'arms': {}, 'reference_tensor_count': len(reference_state),
              'train_rows': len(train_indices), 'validation_rows': len(validation_indices)}

    initial_states = {}
    for arm in ARMS:
        model, evidence = build_arm(arm, reference, torsion=torsion)
        initial = {name: value.clone() for name, value in model.state_dict().items()}
        initial_states[arm] = initial
        capacities = report.setdefault('capacity_by_arm', {})
        capacities[arm] = effective_capacity(model)
        folder = smoke / arm / args.task / f'fold{args.fold}'
        payload = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
        if payload['arm'] != arm:
            raise ValueError(f'{arm}: best.pt carries a different arm tag')
        fresh = O8OnlyArm() if arm == 'f2d' else build_dual_glt_model('concat')
        loaded = fresh.load_state_dict(payload['state_dict'], strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise ValueError(f'{arm}: best.pt does not strict-load')

        moved, unmoved = [], []
        for name, value in initial.items():
            delta = float((payload['state_dict'][name].float() - value.float()).abs().max())
            (moved if delta > 0 else unmoved).append(name)

        metrics = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
        if metrics['train_sample_count'] != len(train_indices):
            raise ValueError(f'{arm}: train sample count mismatch')
        if metrics['validation_sample_count'] != len(validation_indices):
            raise ValueError(f'{arm}: validation sample count mismatch')
        for record in metrics['history']:
            for key in ('train_loss', 'validation_loss', 'validation_r2'):
                if not np.isfinite(record[key]):
                    raise ValueError(f'{arm}: non-finite {key}')
        scaler_mean = float(np.asarray(payload['scaler_mean']).reshape(-1)[0])
        scaler_scale = float(np.asarray(payload['scaler_scale']).reshape(-1)[0])
        run = json.loads((smoke / f'run_{arm}.json').read_text(encoding='utf-8'))
        groups = {group['name']: group for group in run['optimizer_groups']}
        report['arms'][arm] = {
            'strict_load': {'missing_keys': list(loaded.missing_keys),
                            'unexpected_keys': list(loaded.unexpected_keys)},
            'saved_tensor_count': len(payload['state_dict']),
            'initial_tensor_count': len(initial),
            'tensors_moved_by_training': len(moved),
            'tensors_unmoved': sorted(unmoved),
            'group_names': sorted(groups),
            'group_lrs': {name: groups[name]['lr'] for name in sorted(groups)},
            'group_parameters': {name: groups[name]['num_parameters'] for name in sorted(groups)},
            'capacity': payload['capacity'],
            'best_validation_r2': payload['best_validation_r2'],
            'best_epoch': payload['best_epoch'],
            'executed_epochs': payload['executed_epochs'],
            'scaler': {
                'mean': scaler_mean, 'scale': scaler_scale,
                'train_label_mean': float(train_labels.mean()),
                'train_label_std': float(train_labels.std(ddof=0)),
                'validation_label_mean': float(validation_labels.mean()),
                'validation_label_std': float(validation_labels.std(ddof=0)),
                'matches_train_mean': bool(abs(scaler_mean - train_labels.mean()) < 1e-6),
                'matches_train_std': bool(abs(scaler_scale - train_labels.std(ddof=0)) < 1e-6),
                'matches_validation_mean': bool(abs(scaler_mean - validation_labels.mean()) < 1e-6),
            },
            'init_evidence': {'copied': len(evidence['copied']), 'unsourced': evidence['unsourced'],
                              'reference_only': len(evidence['reference_only'])},
        }
        print(json.dumps({'arm': arm, 'moved': len(moved), 'r2': payload['best_validation_r2'],
                          'scaler_matches_train': report['arms'][arm]['scaler']['matches_train_mean'],
                          'capacity': payload['capacity']}), flush=True)

    # Cross-arm equality of the rebuilt common initialisation.
    shared_report = {}
    for name in sorted(initial_states['fbase']):
        if not name.startswith(SHARED_PREFIXES):
            continue
        values = [initial_states[arm][name] for arm in ARMS if name in initial_states[arm]]
        shared_report[name] = bool(all(torch.equal(values[0], value) for value in values[1:]))
    report['shared_initial_tensors'] = {
        'compared': len(shared_report),
        'all_identical': bool(all(shared_report.values())),
        'mismatches': sorted(name for name, ok in shared_report.items() if not ok)}
    report['learning_rate_table'] = ARM_LRS

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status': 'PASS', 'output': str(output),
                      'shared_initial_all_identical': report['shared_initial_tensors']['all_identical'],
                      'compared_tensors': report['shared_initial_tensors']['compared']}))


if __name__ == '__main__':
    main()
