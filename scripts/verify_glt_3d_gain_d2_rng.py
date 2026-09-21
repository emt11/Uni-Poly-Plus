#!/usr/bin/env python3
"""Bounded GPU check of the D2 random-stream contract (GLT-3D-GAIN-20260921-01/r4).

What it measures, per arm, in the production call path (``build_arm`` plus a
plain ``model(batch)`` in train mode):

* the ambient stream position after each batch's forward — the O8 and head
  dropout draws.  All four arms must land on the same position for the same
  batch index, and the position must advance from batch to batch.
* the private GLT stream position of the three dual arms — identical across
  those arms, and advancing batch to batch (so the 3D dropout is never replayed).

Budget: two small batches per arm, one forward and one backward each, eight
arm-level calls in total.  ``optimizer.step()`` is never called here, no epoch is
run, and no parameter update is produced; the loss values are recorded only as a
finiteness check and are not a performance comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn

from scripts.finetune_glt_3d_gain_d2 import (ARMS, ambient_rng_digest, build_arm,
                                            build_reference, optimizer_for_arm, resolve_fold)
from src.dataset.glt_dual import dual_glt_collate
from src.training.glt_dual_runtime import CleanLabeledDataset, open_source, require_tmux
from src.utils import set_global_seed

DUAL_ARMS = ('fbase', 'fnorm', 'fstable')
BATCHES = 2
BATCH_ROWS = 4


def digest_tensor(tensor):
    """Fingerprint of a prediction, for the before-any-update equality check."""
    values = tensor.detach().to(torch.float64).cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()[:16]


def exception_path_check(stream, device):
    """The common stream must survive an exception raised inside the 3D block."""
    before = ambient_rng_digest(device)
    entries = stream.entries
    try:
        with stream:
            raise RuntimeError('synthetic failure inside the isolated block')
    except RuntimeError:
        pass
    return {'common_stream_restored': ambient_rng_digest(device) == before,
            'digest_before': before, 'digest_after': ambient_rng_digest(device),
            'entries_advanced': stream.entries == entries + 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cohort-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--dual-static-root', required=True)
    parser.add_argument('--split-root', required=True)
    parser.add_argument('--task', default='xc')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require_tmux()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'{output} already exists; this check never overwrites')

    started = time.perf_counter()
    payload = {'status': 'INCOMPLETE'}
    try:
        run_verification(args, output, payload, started)
    except BaseException as error:
        # A failure must leave its evidence behind rather than losing the calls
        # it already made; the primary artifact path stays free for a rerun.
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix('.failed.json').write_text(json.dumps(
            {**payload, 'status': 'FAILED', 'error': f'{type(error).__name__}: {error}',
             'wall_seconds': float(time.perf_counter() - started)}, ensure_ascii=False, indent=2),
            encoding='utf-8')
        raise


def run_verification(args, output, payload, started):
    """The measured part; ``payload`` is filled as the evidence accumulates."""
    payload.update({'task': args.task, 'fold': int(args.fold), 'arms': list(ARMS),
                    'batch_rows': BATCH_ROWS, 'batches': BATCHES,
                    'fixture': 'xc/fold0 train rows[0:4] then rows[4:8] — identical for every arm',
                    'note': ('two batches per arm from one common seed; the losses are a '
                             'finiteness check, not a performance comparison'),
                    'optimizer_updates': 0})
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    package = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    torsion = bool(package.get('torsion_modules', False))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    payload['device'] = str(device)
    manifest = json.loads((Path(args.split_root) / f'{args.task}.json').read_text(encoding='utf-8'))

    source, frame = open_source(args.cohort_root, args.cache_root, task=args.task,
                               dual_static_root=args.dual_static_root)
    try:
        train_indices, validation_indices, split_evidence = resolve_fold(
            manifest, args.task, args.fold, cohort_rows=len(frame))
        dataset = CleanLabeledDataset(source, frame['label'].to_numpy(dtype=np.float64),
                                      torsion_mode=('on' if torsion else None))
        rows = [train_indices[index * BATCH_ROWS:(index + 1) * BATCH_ROWS]
                for index in range(BATCHES)]
        batches = [dual_glt_collate([dataset[index] for index in block]) for block in rows]

        reference = build_reference(package, torsion=torsion)
        criterion = nn.MSELoss()
        models, records, digests = {}, {}, {'common': {}, 'glt': {}}
        for arm in ARMS:
            model, _ = build_arm(arm, reference, torsion=torsion, device=device)
            models[arm] = model
            model.to(device)
            model.train()
            optimizer, _ = optimizer_for_arm(model, arm,
                                             weight_decay=config['finetune_weight_decay'])
            set_global_seed(config['seed'])
            entry = {'arm': arm, 'device': str(device), 'forward_calls': 0, 'backward_calls': 0,
                     'optimizer_steps': 0, 'glt_stream_entries_begin': None,
                     'glt_stream_entries_end': None, 'batches': [],
                     'ambient_digest_start': ambient_rng_digest(device)}
            stream = getattr(model, 'glt_stream', None)
            if stream is not None:
                entry['glt_stream_entries_begin'] = int(stream.entries)
            for index, batch in enumerate(batches):
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch)
                entry['forward_calls'] += 1
                loss = criterion(prediction, batch.y)
                loss.backward()
                entry['backward_calls'] += 1
                record = {'batch_index': index, 'rows': int(batch.y.numel()),
                          'loss': float(loss.detach().float()),
                          'prediction_digest': digest_tensor(prediction),
                          'finite_prediction': bool(torch.isfinite(prediction).all()),
                          'ambient_digest_after_forward': ambient_rng_digest(device),
                          'glt_stream_digest_after_forward': None if stream is None else stream.digest()}
                entry['batches'].append(record)
                digests['common'].setdefault(index, {})[arm] = record['ambient_digest_after_forward']
                if stream is not None:
                    digests['glt'].setdefault(index, {})[arm] = record['glt_stream_digest_after_forward']
            optimizer.zero_grad(set_to_none=True)   # no optimizer.step() anywhere in this script
            if stream is not None:
                entry['glt_stream_entries_end'] = int(stream.entries)
            records[arm] = entry
            payload.update({'records': records, 'digests': digests})
            print(json.dumps({'arm': arm, 'forward_calls': entry['forward_calls'],
                              'backward_calls': entry['backward_calls'],
                              'losses': [item['loss'] for item in entry['batches']],
                              'predictions': [item['prediction_digest']
                                              for item in entry['batches']],
                              'ambient': [item['ambient_digest_after_forward']
                                          for item in entry['batches']],
                              'glt': [item['glt_stream_digest_after_forward']
                                      for item in entry['batches']]}), flush=True)

        checks = {'arms_start_from_the_same_ambient_position':
                  len({records[arm]['ambient_digest_start'] for arm in ARMS}) == 1}
        for index in range(BATCHES):
            shared = {arm: digests['common'][index][arm] for arm in ARMS}
            checks[f'common_stream_shared_by_all_arms_batch{index}'] = len(set(shared.values())) == 1
            glt_shared = {arm: digests['glt'][index][arm] for arm in DUAL_ARMS}
            checks[f'glt_stream_shared_by_dual_arms_batch{index}'] = len(set(glt_shared.values())) == 1
        for index in range(BATCHES - 1):
            checks[f'common_stream_advances_batch{index}'] = (
                digests['common'][index]['fbase'] != digests['common'][index + 1]['fbase'])
            checks[f'glt_stream_advances_batch{index}'] = (
                digests['glt'][index]['fbase'] != digests['glt'][index + 1]['fbase'])
        checks['f2d_has_no_glt_stream'] = all(
            item['glt_stream_digest_after_forward'] is None for item in records['f2d']['batches'])
        exception = exception_path_check(models['fbase'].glt_stream, device)
        checks['exception_path_restores_common_stream'] = exception['common_stream_restored']
        checks['exception_path_advances_glt_stream'] = exception['entries_advanced']
        observations = {
            'dual_arms_predict_identically_before_any_update': len({
                tuple(records[arm]['batches'][index]['prediction_digest'] for index in range(BATCHES))
                for arm in DUAL_ARMS}) == 1,
            'f2d_prediction_differs_from_the_dual_arms': (
                records['f2d']['batches'][0]['prediction_digest']
                != records['fbase']['batches'][0]['prediction_digest']),
        }

        payload.update({
            'status': 'PASS' if all(checks.values()) else 'CHECKS_FAILED',
            'records': records, 'digests': digests, 'checks': checks, 'observations': observations,
            'all_checks_passed': bool(all(checks.values())),
            'split': split_evidence, 'wall_seconds': float(time.perf_counter() - started)})
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        if not payload['all_checks_passed']:
            failed = [name for name, passed in checks.items() if not passed]
            raise RuntimeError('random-stream checks failed: ' + ','.join(failed))
        print(json.dumps({'status': 'PASS', 'checks': len(checks), 'observations': observations,
                          'output': str(output)}))
    finally:
        source.close()


if __name__ == '__main__':
    main()
