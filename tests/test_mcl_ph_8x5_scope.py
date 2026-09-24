import pytest

import scripts.aggregate_mcl_ph_8x5 as full_aggregate
from scripts.run_mcl_ph_8x5 import unit_list
from scripts.run_mcl_ph_8x5_4gpu import distribute
from scripts.finetune_mcl_ph import (FOLDS, TASKS, unit_directory,
                                     validate_stage_scope)


def test_full8x5_scope_and_development_boundary(tmp_path):
    assert len(TASKS) == 8 and FOLDS == (0, 1, 2, 3, 4)
    for task in TASKS:
        for fold in FOLDS:
            validate_stage_scope('full8x5', task, fold, 30, 'trusted.json')
            assert unit_directory(tmp_path, 'm_cat', task, fold).name == f'fold{fold}'
    with pytest.raises(ValueError, match='development permits'):
        validate_stage_scope('development', 'egc', 2, 30, 'trusted.json')
    with pytest.raises(ValueError, match='trusted cohort index'):
        validate_stage_scope('full8x5', 'egc', 2, 30, None)
    assert len(unit_list(('m_cat',))) == 40
    assert len(unit_list(('glt_ref', 'o8_only', 'm_cat', 'm_gate', 'm_xattn'))) == 200
    with pytest.raises(ValueError, match='unique subset'):
        unit_list(('m_cat', 'm_cat'))


def test_full8x5_aggregate_requires_all_forty_units(monkeypatch, tmp_path):
    missing = {('egc', 4)}

    def check(_root, arm, task, fold, *, stage, expected_step):
        assert stage == 'full8x5' and expected_step == 5000
        if (task, fold) in missing:
            return None, ['missing artifacts']
        return {'arm': arm, 'task': task, 'fold': fold,
                'best_validation_r2': 0.25, 'pretrain_package_sha256': 'a' * 64}, []

    monkeypatch.setattr(full_aggregate, 'check_unit', check)
    result = full_aggregate.aggregate(tmp_path, arms=('m_cat',), expected_step=5000)
    assert (result['status'], result['units_expected'], result['units_accepted']) == \
           ('INCOMPLETE', 40, 39)
    assert result['validation_r2'] is None
    missing.clear()
    result = full_aggregate.aggregate(tmp_path, arms=('m_cat',), expected_step=5000)
    assert result['status'] == 'PASS' and result['units_accepted'] == 40
    assert result['validation_r2']['m_cat']['macro8'] == 0.25
    assert result['outer_test'] == 'NOT_RUN'


def test_four_gpu_assignment_covers_each_remaining_unit_once():
    remaining = unit_list(('m_cat', 'm_gate'))[7:]
    assigned = distribute(remaining)
    assert set(assigned) == {0, 1, 2, 3}
    assert sorted(row for rows in assigned.values() for row in rows) == sorted(remaining)
    assert max(map(len, assigned.values())) - min(map(len, assigned.values())) <= 1
