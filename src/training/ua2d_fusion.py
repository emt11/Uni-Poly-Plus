"""UA-specific array-only policies; no fitting or model inference interfaces."""
import copy
from pathlib import Path
import numpy as np
from src.training.fce_kernel import (TASKS, GLOBAL_WEIGHTS, bind_branch_arrays,
                                     blend_arrays, select_global_weight, validate_split)

POLICY = dict(name='UA_2D_KRR_A1', grid=list(GLOBAL_WEIGHTS), fixed_weight=.75,
              objective='35 validation R2 arithmetic mean', tie='larger UA weight',
              alpha_selection='unchanged r10/r9', task_switching=False,
              units='original', clipping=False, rescaling=False)


def select_weight(validation):
    # Adapt only the low-level API's historical neural field name.
    result = select_global_weight([dict(task=r['task'], fold=r['fold'],
        y_true=r['y_true'], g05=r['ua'], krr=r['krr']) for r in validation])
    result['policy'] = copy.deepcopy(POLICY)
    return result


def check_unit(unit):
    validate_split(unit, unit['sample_count'])
    ua = unit['ua']
    if (ua['arm'], ua['task'], ua['fold']) != ('UA_2D', unit['task'], unit['fold']):
        raise ValueError('UA task/fold/arm mismatch')
    if ua['scaler'] != unit['scaler'] or ua['test_indices'] != unit['test_indices']:
        raise ValueError('UA scaler/split mismatch')
    if unit['alpha'] != unit['r10_alpha']:
        raise ValueError('r10 alpha changed')
    if unit['scaler'].get('fit') != 'train' or unit['scaler'].get('transform') != 'standard':
        raise ValueError('train-only scaler required')
    if ua['prediction_path'] != 'pure_2d' or ua['input_dim'] != 512:
        raise ValueError('UA prediction path mismatch')


def align(values, unit, role, identities, branch):
    check_unit(unit)
    key = 'y_pred' if branch == 'ua' or role == 'test' else 'y_krr'
    if branch not in ('ua', 'krr'):
        raise ValueError('invalid branch')
    if branch == 'ua':
        if int(values['best_epoch']) != unit['ua']['best_epoch']:
            raise ValueError('UA best epoch mismatch')
        if str(values['outer_test']) != ('RUN' if role == 'test' else 'NOT_RUN'):
            raise ValueError('UA split role mismatch')
    # Older validation kernel arrays omit alpha; their directory/metrics binding
    # is checked by the runner against the frozen r10 source, without reselecting.
    return bind_branch_arrays(values, unit, role, identities, (key,))


class FreezeGate:
    def __init__(self):
        self.payload = None
        self.path = None

    def freeze(self, payload, path):
        if self.payload is not None:
            raise ValueError('already frozen')
        if payload.get('policy') != POLICY or not payload.get('frozen') or payload.get('test_opened') is not False:
            raise ValueError('UA policy/freeze mismatch')
        expected = {(t, f) for t in TASKS for f in range(5)}
        units = payload['units']
        if len(units) != 35 or {(u['task'], u['fold']) for u in units} != expected:
            raise ValueError('all 35 units must freeze')
        for u in units:
            check_unit(u)
        scores = payload['scores']
        if len(scores) != 5 or {s['weight'] for s in scores} != set(GLOBAL_WEIGHTS):
            raise ValueError('175 validation scores required')
        for s in scores:
            if len(s['units']) != 35 or {(u['task'], u['fold']) for u in s['units']} != expected:
                raise ValueError('validation coverage mismatch')
            vals = np.asarray([u['r2'] for u in s['units']], dtype=float)
            if not np.isfinite(vals).all() or float(vals.mean()) != s['mean_validation_r2']:
                raise ValueError('validation objective mismatch')
        winner = max(scores, key=lambda s: (s['mean_validation_r2'], s['weight']))
        if payload['weights'] != dict(A_FIXED=.75, A_GLOBAL=winner['weight']):
            raise ValueError('weight/tie mismatch')
        # Exclusive creation: a durable global freeze precedes every TEST read.
        import json
        with Path(path).open('x') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            import os
            os.fsync(stream.fileno())
        self.payload = copy.deepcopy(payload)
        self.path = Path(path)

    def require_test(self):
        if self.payload is None or self.path is None or not self.path.is_file():
            raise ValueError('TEST before persisted global freeze')


def load_arrays(path, role, gate=None, on_read=None):
    if role == 'test':
        if gate is None:
            raise ValueError('TEST requires freeze')
        gate.require_test()
    elif role != 'validation':
        raise ValueError('invalid array role')
    if on_read:
        on_read(str(path), role)
    with np.load(path, allow_pickle=False) as values:
        return {k: values[k].copy() for k in values.files}


def policy_outputs(ua, krr, weights, gate):
    """Return shared objects for equal policies and exact existing endpoints."""
    gate.require_test()
    if weights != gate.payload['weights']:
        raise ValueError('unregistered TEST policy')
    cache = {0.: krr, 1.: ua}
    result = {}
    for name, weight in weights.items():
        if weight not in cache:
            cache[weight] = blend_arrays(ua, krr, weight)
        result[name] = cache[weight]
    return result
