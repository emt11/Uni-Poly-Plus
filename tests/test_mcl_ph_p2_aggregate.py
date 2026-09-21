"""Model-free regression for the P2 development aggregator (r10, section 11).

Pure CPU: no GPU, no model, no forward or backward pass.  Every unit is a
synthetic fixture written on disk; the checks run the aggregator itself, so what
is verified is the acceptance decision, the R^2/matched-delta arithmetic and the
pre-registered qualification and parent-selection rules -- never a real model.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.aggregate_mcl_ph_p2 import (ARMS, FOLDS, TASKS, aggregate,
                                         arm_table, matched_deltas,
                                         qualification, selection)

STEP = 5000
EPOCHS = 3
BEST_EPOCH = 2
STEPS_PER_EPOCH = 100
MCL_ARMS = ('m_cat', 'm_gate', 'm_xattn')

# A baseline pair plus one MCL scheme that clears the gate against both.
O8 = {'xc_fold0': 0.30, 'xc_fold1': 0.32, 'eps': 0.40, 'eat': 0.35}
GLT = {'xc_fold0': 0.28, 'xc_fold1': 0.30, 'eps': 0.38, 'eat': 0.33}
CAT_OK = {'xc_fold0': 0.33, 'xc_fold1': 0.35, 'eps': 0.42, 'eat': 0.37}


def _r2(scheme, arm, task, fold):
    key = f'{task}_fold{fold}' if task == 'xc' else task
    return scheme[arm][key]


def _shift(scheme, arm, **delta):
    return {key: value + delta.get(key, 0.0) for key, value in scheme[arm].items()}


def write_unit(root, arm, task, fold, value):
    directory = Path(root) / arm / task / f'fold{fold}'
    directory.mkdir(parents=True, exist_ok=True)
    history = [
        {'epoch': index + 1, 'training_steps': STEPS_PER_EPOCH,
         'train_loss': 1.0 - 0.05 * index,
         'validation_loss': 0.50 - 0.02 * index,
         'validation_r2': value + (0.0 if index == BEST_EPOCH - 1 else -0.04)}
        for index in range(EPOCHS)
    ]
    metrics = {
        'arm': arm, 'task': task, 'fold': fold, 'stage': 'development',
        'protocol': 'mcl_ph_development', 'outer_test': 'NOT_RUN',
        'pretrained_route': 'mcl_ph' if arm in MCL_ARMS else 'dual_glt',
        'requested_epochs': 30, 'executed_epochs': EPOCHS,
        'optimizer_updates': EPOCHS * STEPS_PER_EPOCH, 'history': history,
        'best_epoch': BEST_EPOCH, 'best_validation_r2': value,
        'pretrain_step': STEP, 'pretrain_package_sha256': '0' * 64,
        'optimizer_groups': [{'name': 'backbone', 'num_parameters': 1000},
                             {'name': 'head', 'num_parameters': 10}],
        'split': {'protocol': 'outer5_inner20', 'fold': fold, 'task': task,
                  'validation_is_test': False, 'outer_test': 'NOT_RUN',
                  'sets_disjoint': True, 'union_equals_full_cohort': True,
                  'train_rows': 276, 'validation_rows': 69, 'test_rows': 87},
    }
    (directory / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
    (directory / 'run.json').write_text(json.dumps({
        'arm': arm, 'task': task, 'fold': fold, 'stage': 'development',
        'protocol': 'mcl_ph_development', 'outer_test': 'NOT_RUN',
        'executed_epochs': EPOCHS, 'pretrain_step': STEP,
        'optimizer_groups': metrics['optimizer_groups'], 'history': history,
    }), encoding='utf-8')
    (directory / 'runtime.json').write_text(json.dumps({
        'arm': arm, 'task': task, 'fold': fold, 'stage': 'development',
        'status': 'PASS', 'exit_code': 0,
    }), encoding='utf-8')
    (directory / 'best.pt').write_bytes(b'synthetic-checkpoint')
    np.savez(directory / 'validation_predictions.npz',
             y_true=np.linspace(0, 1, 5), y_pred=np.linspace(0, 1, 5),
             sample_keys=np.asarray([f'{index:04d}' for index in range(5)]),
             best_epoch=np.asarray(BEST_EPOCH), outer_test=np.asarray('NOT_RUN'))


def write_root(root, scheme):
    for arm in ARMS:
        for task in TASKS:
            for fold in FOLDS:
                write_unit(root, arm, task, fold, _r2(scheme, arm, task, fold))
    return root


def full_scheme(**overrides):
    scheme = {'glt_ref': dict(GLT), 'o8_only': dict(O8),
              'm_cat': dict(CAT_OK),
              'm_gate': _shift({'m_gate': CAT_OK}, 'm_gate', xc_fold0=0.001,
                               xc_fold1=0.001, eps=0.001, eat=0.001),
              'm_xattn': _shift({'m_xattn': CAT_OK}, 'm_xattn', xc_fold0=-0.01,
                                xc_fold1=-0.01, eps=-0.01, eat=-0.01)}
    scheme.update(overrides)
    return scheme


def run(tmp_path, scheme, *, skip=None, mutate=None):
    root = write_root(tmp_path / 'downstream', scheme)
    if skip:
        for name in Path(root, *skip).glob('*'):
            name.unlink()
        Path(root, *skip).rmdir()
    if mutate:
        mutate(Path(root))
    return aggregate(root, expected_step=STEP)


# ---------------------------------------------------------------- completeness

def test_a_complete_thirty_unit_set_is_pass(tmp_path):
    payload = run(tmp_path, full_scheme())
    assert payload['status'] == 'PASS'
    assert payload['acceptance'] == 'PASS'
    assert (payload['units_expected'], payload['units_accepted'],
            payload['units_rejected']) == (30, 30, 0)
    assert payload['outer_test'] == 'NOT_RUN'
    assert payload['selection']['selection_status'] == 'SELECTED'


def test_b_a_missing_unit_is_incomplete(tmp_path):
    payload = run(tmp_path, full_scheme(), skip=('m_gate', 'eat', 'fold1'))
    assert payload['status'] == 'INCOMPLETE'
    assert payload['units_accepted'] == 29
    assert payload['units_rejected'] == 1
    assert payload['selection']['selection_status'] == 'INCOMPLETE'
    assert payload['selection']['selected_parent'] is None


def test_c_a_non_finite_metric_is_incomplete(tmp_path):
    def mutate(root):
        path = root / 'm_cat' / 'xc' / 'fold0' / 'metrics.json'
        record = json.loads(path.read_text())
        record['history'][BEST_EPOCH - 1]['validation_r2'] = float('nan')
        path.write_text(json.dumps(record))

    payload = run(tmp_path, full_scheme(), mutate=mutate)
    assert payload['status'] == 'INCOMPLETE'
    assert any('not finite' in problem for entry in payload['rejected']
               for problem in entry['problems'])
    assert payload['qualification'] is None


def test_d_a_wrong_outer_test_is_incomplete(tmp_path):
    def mutate(root):
        path = root / 'o8_only' / 'eps' / 'fold1' / 'metrics.json'
        record = json.loads(path.read_text())
        record['outer_test'] = 'RUN'
        record['split']['outer_test'] = 'RUN'
        path.write_text(json.dumps(record))

    payload = run(tmp_path, full_scheme(), mutate=mutate)
    assert payload['status'] == 'INCOMPLETE'
    assert any('outer_test' in problem for entry in payload['rejected']
               for problem in entry['problems'])


# ------------------------------------------------------------------ arithmetic

def test_e_task_mean_and_macro3_are_the_plain_means(tmp_path):
    payload = run(tmp_path, full_scheme())
    table = payload['r2']
    assert table['glt_ref']['xc']['mean'] == pytest.approx((GLT['xc_fold0'] + GLT['xc_fold1']) / 2)
    assert table['glt_ref']['macro3'] == pytest.approx(
        (table['glt_ref']['xc']['mean'] + GLT['eps'] + GLT['eat']) / 3)
    assert len(payload['accepted']) == 30
    assert payload['deltas']['m_cat']['vs_o8_only']['macro3'] == pytest.approx(
        table['m_cat']['macro3'] - table['o8_only']['macro3'])


# ------------------------------------------------------------------ selection

def test_f_qualified_cat_is_the_parent(tmp_path):
    payload = run(tmp_path, full_scheme())
    report = payload['qualification']
    assert report['m_cat']['qualified'] is True
    assert report['m_cat']['vs_o8_only']['qualified'] is True
    assert report['m_cat']['vs_glt_ref']['qualified'] is True
    assert payload['selection']['selected_parent'] == 'm_cat'
    assert payload['selection']['replacement']['m_gate']['eligible'] is False


def test_g_gate_can_replace_cat_only_on_the_stated_conditions(tmp_path):
    scheme = full_scheme()
    scheme['m_gate'] = _shift({'m_gate': CAT_OK}, 'm_gate', xc_fold0=0.005,
                              xc_fold1=0.005, eps=0.005, eat=0.005)
    payload = run(tmp_path, scheme)
    replacement = payload['selection']['replacement']['m_gate']
    assert replacement == {'macro3': True, 'xc_fold0_positive': True,
                           'xc_fold1_positive': True, 'eligible': True}
    assert payload['selection']['selected_parent'] == 'm_gate'


def test_h_xattn_without_xc_fold_consistency_cannot_replace(tmp_path):
    scheme = full_scheme()
    scheme['m_xattn'] = {'xc_fold0': CAT_OK['xc_fold0'] + 0.03,
                         'xc_fold1': CAT_OK['xc_fold1'] - 0.01,
                         'eps': CAT_OK['eps'] + 0.02,
                         'eat': CAT_OK['eat'] + 0.02}
    payload = run(tmp_path, scheme)
    assert payload['qualification']['m_xattn']['qualified'] is True
    replacement = payload['selection']['replacement']['m_xattn']
    assert replacement['macro3'] is True and replacement['xc_fold1_positive'] is False
    assert replacement['eligible'] is False
    assert payload['selection']['selected_parent'] == 'm_cat'


def test_i_gate_is_selected_when_cat_fails(tmp_path):
    scheme = full_scheme()
    scheme['m_cat'] = dict(O8)
    payload = run(tmp_path, scheme)
    assert payload['qualification']['m_cat']['qualified'] is False
    assert payload['qualification']['m_gate']['qualified'] is True
    assert payload['selection']['selected_parent'] == 'm_gate'


def test_j_no_qualified_arm_is_reported_and_stops(tmp_path):
    scheme = full_scheme()
    for arm in MCL_ARMS:
        scheme[arm] = dict(O8)
    payload = run(tmp_path, scheme)
    assert payload['selection']['selection_status'] == 'NO_QUALIFIED_ARM'
    assert payload['selection']['selected_parent'] is None
    assert 'STOP' in payload['selection']['reason']


def test_k_macro3_tie_prefers_gate(tmp_path):
    scheme = full_scheme()
    scheme['m_cat'] = dict(O8)                       # CAT fails the gate
    scheme['m_gate'] = _shift({'m_gate': CAT_OK}, 'm_gate', xc_fold0=0.03,
                              xc_fold1=0.03, eps=0.02, eat=0.02)
    scheme['m_xattn'] = dict(scheme['m_gate'])
    scheme['m_xattn']['eps'] += 0.0005               # inside the 0.002 tie band
    payload = run(tmp_path, scheme)
    assert payload['selection']['selected_parent'] == 'm_gate'
    assert payload['qualification']['m_xattn']['qualified'] is True


# --------------------------------------------------------------- pure helpers

def test_l_a_single_baseline_pass_is_not_qualified():
    """Section 9: clearing one baseline is NOT_QUALIFIED, not qualified."""
    table = {arm: {task: {'mean': 0.1, 'fold0': 0.1, 'fold1': 0.1} for task in TASKS}
             for arm in ARMS}
    table['glt_ref']['macro3'] = 0.10
    table['o8_only'] = {'xc': {'fold0': 0.36, 'fold1': 0.36, 'mean': 0.36},
                        'eps': {'fold0': 0.40, 'fold1': 0.40, 'mean': 0.40},
                        'eat': {'fold0': 0.40, 'fold1': 0.40, 'mean': 0.40},
                        'macro3': 0.3867}
    table['m_cat'] = {'xc': {'fold0': 0.32, 'fold1': 0.32, 'mean': 0.32},
                      'eps': {'fold0': 0.4, 'fold1': 0.4, 'mean': 0.4},
                      'eat': {'fold0': 0.4, 'fold1': 0.4, 'mean': 0.4},
                      'macro3': 0.3733}
    for arm in ('m_gate', 'm_xattn'):
        table[arm] = dict(table['m_cat'])
    report = qualification(matched_deltas(table))
    assert report['m_cat']['vs_glt_ref']['qualified'] is True
    assert report['m_cat']['vs_o8_only']['qualified'] is False
    assert report['m_cat']['qualified'] is False
    assert selection(report, table)['selection_status'] == 'NO_QUALIFIED_ARM'


def test_m_the_table_and_selection_helpers_are_pure():
    records = [{'arm': arm, 'task': task, 'fold': fold, 'best_validation_r2': 0.2,
                'best_epoch': 1, 'executed_epochs': 3} for arm in ARMS
               for task in TASKS for fold in FOLDS]
    table = arm_table(records)
    assert set(table) == set(ARMS)
    assert table['o8_only']['macro3'] == pytest.approx(0.2)
    deltas = matched_deltas(table)
    assert set(deltas) == set(MCL_ARMS)
    assert deltas['m_cat']['vs_o8_only']['macro3'] == 0.0
