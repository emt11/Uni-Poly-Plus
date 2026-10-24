"""Synthetic-only final suite. Never imports models or opens project predictions."""
import copy

import numpy as np
import pytest

from src.training import ua2d_p10k_r10fp as m


def unit(task='eea', fold=0):
    return dict(task=task, fold=fold, sample_count=6,
                train_indices=[0, 1], validation_indices=[2, 3], test_indices=[4, 5],
                units='original property units', alpha=.1,
                scaler=dict(mean=[10.], scale=[2.], fit='train', transform='standard'),
                identities={str(i): dict(sample_key=f'key-{i}') for i in range(6)},
                p10k=dict(config=dict(parent=str(m.PARENT))))


def array():
    return dict(indices=np.array([3, 2]), sample_keys=np.array(['key-3', 'key-2']),
                y_true=np.array([14., 12.]), y_pred=np.array([15., 11.]),
                scaler_mean=np.array([10.]), scaler_scale=np.array([2.]), alpha=.1)


def validation(same=False):
    return [dict(task=t, fold=f, y_true=np.array([10., 12., 15.]),
                 p10k_pred=np.array([10., 12., 15.]) if same else np.array([11., 13., 16.]),
                 krr_pred=np.array([10., 12., 15.]) if same else np.array([9., 11., 14.]))
            for t in m.TASKS for f in range(5)]


def payload():
    return dict(m.choose(validation()), plan=m.PLAN, revision=m.REVISION, schema=m.SCHEMA,
                frozen=True, units=[unit(t, f) for t in m.TASKS for f in range(5)],
                test_processed_by_this_cycle=False)


def test_original_row_and_key_reordering():
    b = m.bind(array(), unit(), 'validation', 'y_pred', krr=True)
    np.testing.assert_array_equal(b['y_true'], [12., 14.])
    np.testing.assert_array_equal(b['y_pred'], [11., 15.])


@pytest.mark.parametrize('change', ['duplicate_row', 'missing_row', 'extra_row', 'key_conflict',
                                   'duplicate_key', 'units', 'label_nan', 'scaler', 'alpha', 'float_row'])
def test_binding_rejects_conflicts(change):
    a, u = array(), unit()
    if change == 'duplicate_row': a['indices'] = np.array([2, 2])
    if change == 'missing_row': a['indices'] = np.array([2])
    if change == 'extra_row': a['indices'] = np.array([3, 4])
    if change == 'key_conflict': a['sample_keys'] = np.array(['key-2', 'key-3'])
    if change == 'duplicate_key': a['sample_keys'] = np.array(['key-3', 'key-3'])
    if change == 'units': u['units'] = 'standardized'
    if change == 'label_nan': a['y_true'][0] = np.nan
    if change == 'scaler': a['scaler_scale'] = np.array([3.])
    if change == 'alpha': a['alpha'] = 1.
    if change == 'float_row': a['indices'] = np.array([3., 2.])
    with pytest.raises(ValueError): m.bind(a, u, 'validation', 'y_pred', krr=True)


def test_label_mismatch_comparison_and_existing_metric_guard():
    a = m.bind(array(), unit(), 'validation', 'y_pred', krr=True)
    changed = array(); changed['y_true'][0] += 1
    b = m.bind(changed, unit(), 'validation', 'y_pred', krr=True)
    assert not np.array_equal(a['y_true'], b['y_true'])
    with pytest.raises(ValueError):
        m.check_metrics(m.metrics(a['y_true'], a['y_pred']), dict(r2=1., rmse=0.))


def test_raw_unit_blend_and_endpoints():
    a, b = np.array([12., 14.]), np.array([10., 18.])
    np.testing.assert_array_equal(m.blend(a, b, .75), [11.5, 15.])
    np.testing.assert_array_equal(m.blend(a, b, 1.), a)
    np.testing.assert_array_equal(m.blend(a, b, 0.), b)
    with pytest.raises(ValueError): m.blend(a, b, .6)


def test_fixed_grid_single_global_weight_and_exact_tie():
    result = m.choose(validation())
    assert result['weight'] == .5
    assert [s['weight'] for s in result['scores']] == list(m.WEIGHTS)
    assert sum(len(s['units']) for s in result['scores']) == 175
    assert m.choose(validation(same=True))['weight'] == 1.
    with pytest.raises(ValueError): m.choose(validation()[:-1])


def test_equal_unit_weight_not_sample_count_or_rounded_scores():
    data = validation()
    data[0]['y_true'] = np.tile(data[0]['y_true'], 20)
    data[0]['p10k_pred'] = np.tile(data[0]['p10k_pred'], 20)
    data[0]['krr_pred'] = np.tile(data[0]['krr_pred'], 20)
    scores = m.choose(data)['scores']
    for s in scores:
        assert s['mean_validation_r2'] == np.mean([r['r2'] for r in s['units']])
    p = payload()
    for s in p['scores']:
        v = .8 + (1e-13 if s['weight'] == .75 else 0.)
        for r in s['units']: r['r2'] = v
        s['mean_validation_r2'] = float(np.mean([r['r2'] for r in s['units']]))
    p['weight'] = .75
    gate = m.FrozenGate(); gate.freeze(p)
    assert gate.payload['weight'] == .75  # No epsilon / three-decimal tie.


def test_freeze_before_test_and_immutable_snapshot(tmp_path):
    gate = m.FrozenGate()
    with pytest.raises(ValueError): gate.require_test()
    with pytest.raises(ValueError): m.load_npz(tmp_path/'missing.npz', formal=True)
    p = payload(); gate.freeze(p); gate.require_test()
    p['weight'] = 1.
    assert gate.payload['weight'] == .5
    with pytest.raises(ValueError): gate.freeze(payload())
    m.write(tmp_path/'frozen.json', {'weight': .5})
    with pytest.raises(FileExistsError): m.write(tmp_path/'frozen.json', {'weight': 1.})


@pytest.mark.parametrize('change', ['parent', 'alpha', 'coverage', 'winner', 'processed', 'split'])
def test_frozen_identity_and_budget_coverage(change):
    p = payload()
    if change == 'parent': p['units'][0]['p10k']['config']['parent'] = 'other.pt'
    if change == 'alpha': p['units'][0]['alpha'] = 10.
    if change == 'coverage': p['units'].pop()
    if change == 'winner': p['weight'] = 1.
    if change == 'processed': p['test_processed_by_this_cycle'] = True
    if change == 'split': p['units'][0]['test_indices'] = [0, 5]
    with pytest.raises(ValueError): m.FrozenGate().freeze(p)


def test_zero_training_fit_inference_and_cpu_resource_guard():
    m.resource_guard(m.ENV)
    for k in ('CUDA_VISIBLE_DEVICES', 'WORLD_SIZE', 'OMP_NUM_THREADS'):
        env = dict(m.ENV); env[k] = '3' if k == 'CUDA_VISIBLE_DEVICES' else '2'
        with pytest.raises(ValueError): m.resource_guard(env)
    used = dict.fromkeys(m.LIMITS, 0); m.check_budget(used)
    for k in ('pt_starts','ft_starts','krr_fit','model_inference','gpu_tasks','workers','fp_generation'):
        bad = used.copy(); bad[k] = 1
        with pytest.raises(ValueError): m.check_budget(bad)


def test_failed_partial_accounting_is_not_reset_or_double_charged():
    used = dict.fromkeys(m.LIMITS, 0)
    used.update(prepare_starts=1, test_starts=3, test_graphs=30, outputs=2)
    reservation = dict(test_starts=32, test_graphs=5845, outputs=33)
    m.check_budget(used, reservation)
    conservative = {k: v + reservation.get(k, 0) for k, v in used.items()}
    m.check_budget(conservative)
    assert conservative['test_starts'] == 35 and conservative['outputs'] == 35
    with pytest.raises(ValueError): m.check_budget(conservative, dict(test_starts=1))
    with pytest.raises(ValueError): m.check_budget(used, dict(prepare_starts=1))
