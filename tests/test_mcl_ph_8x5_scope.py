"""Scope and scheduling checks for the active official five-fold launcher."""
import pytest

from scripts.aggregate_mcl_ph_8x5 import aggregate
from scripts.finetune_mcl_ph import ARMS, FOLDS, PAPER_TASKS, validate_stage_scope
from scripts.run_mcl_ph_8x5_4gpu import distribute, unit_list


def test_official_scope_and_unit_count():
    assert FOLDS == (0, 1, 2, 3, 4)
    assert len(unit_list()) == 125
    assert set(unit_list()) == {(arm, task, fold)
                                for arm in ARMS for task in PAPER_TASKS for fold in FOLDS}
    validate_stage_scope('paper5_outer', 'eea', 0, 70, 'trusted-index')
    with pytest.raises(ValueError):
        validate_stage_scope('full8x5_outer', 'eea', 0, 70, 'trusted-index')
    with pytest.raises(ValueError):
        validate_stage_scope('paper5_outer', 'xc', 0, 70, 'trusted-index')
    with pytest.raises(ValueError):
        validate_stage_scope('paper5_outer', 'eea', 0, 70, None)


def test_four_gpu_assignment_covers_every_unit_once():
    assigned = distribute(unit_list())
    assert set(assigned) == {0, 1, 2, 3}
    assert sorted(row for rows in assigned.values() for row in rows) == sorted(unit_list())
    assert max(map(len, assigned.values())) - min(map(len, assigned.values())) <= 1


def test_paper_aggregate_rejects_missing_unit(monkeypatch, tmp_path):
    import scripts.aggregate_mcl_ph_8x5 as module
    missing = {('eea', 4)}

    def check(_root, arm, task, fold, *, stage, expected_step):
        assert stage == 'paper5_outer' and expected_step == 5000
        if (task, fold) in missing:
            return None, ['missing artifacts']
        return {'arm': arm, 'task': task, 'fold': fold,
                'finetune_strategy': 'periodic_tdl', 'best_validation_r2': .5,
                'test_r2': fold / 10, 'pretrain_package_sha256': 'a' * 64}, []

    monkeypatch.setattr(module, 'check_unit', check)
    result = aggregate(tmp_path, arms=('m_cat',), expected_step=5000)
    assert (result['status'], result['units_expected'], result['units_accepted']) == \
           ('INCOMPLETE', 25, 24)
    missing.clear()
    result = aggregate(tmp_path, arms=('m_cat',), expected_step=5000)
    assert result['status'] == 'PASS' and result['units_accepted'] == 25
    assert result['test_r2']['m_cat']['tasks']['eea']['mean'] == pytest.approx(.2)
