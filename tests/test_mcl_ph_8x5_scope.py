"""Scope and scheduling checks for the active eight-task launcher."""
import json

import pytest

from src.dataset.cache_lifecycle import json_hash
from src.training.glt_dual_runtime import sha256_file
from scripts.aggregate_mcl_ph_8x5 import aggregate
from scripts.finetune_mcl_ph import ARMS, FOLDS, PAPER_SPLIT_PROTOCOL, PAPER_TASKS, validate_stage_scope
from scripts.run_mcl_ph_8x5_4gpu import distribute, unit_list, verify_splits


def test_official_scope_and_unit_count():
    assert FOLDS == (0, 1, 2, 3, 4)
    assert len(unit_list()) == 200
    assert set(unit_list()) == {(arm, task, fold)
                                for arm in ARMS for task in PAPER_TASKS for fold in FOLDS}
    validate_stage_scope('paper5_outer', 'eea', 0, 70, 'trusted-index')
    validate_stage_scope('paper5_outer', 'egc', 0, 60, 'trusted-index')
    with pytest.raises(ValueError):
        validate_stage_scope('full8x5_outer', 'eea', 0, 70, 'trusted-index')
    with pytest.raises(ValueError):
        validate_stage_scope('paper5_outer', 'unknown', 0, 70, 'trusted-index')
    with pytest.raises(ValueError):
        validate_stage_scope('paper5_outer', 'eea', 0, 70, None)


def test_four_gpu_assignment_covers_every_unit_once():
    assigned = distribute(unit_list())
    assert set(assigned) == {0, 1, 2, 3}
    assert sorted(row for rows in assigned.values() for row in rows) == sorted(unit_list())
    assert max(map(len, assigned.values())) - min(map(len, assigned.values())) <= 1


def test_launcher_uses_sealed_external_union_and_source_splits(tmp_path):
    active, old = tmp_path / 'active', tmp_path / 'trusted_splits'
    cohort = tmp_path / 'downstream' / 'cohort'
    trusted_union = tmp_path / 'provenance' / 'union.json'
    for directory in (active, old, cohort, trusted_union.parent,
                      cohort.parent / 'union_paper5_outer'):
        directory.mkdir(parents=True)
    split_hashes = {}
    for task in PAPER_TASKS:
        row = {'sample_order_sha256': 'a' * 64, 'sample_count': 1}
        old_path = old / f'{task}.json'
        old_path.write_text(json.dumps(row))
        split_hashes[task] = sha256_file(old_path)
        (active / f'{task}.json').write_text(json.dumps({
            **row, 'protocol': PAPER_SPLIT_PROTOCOL, 'task': task,
            'splitter': {'random_state': 100},
            'inner_splitter': {'random_state': 42}}))
    union = {'split_manifest_sha256': split_hashes}
    trusted_union.write_text(json.dumps(union))
    (cohort.parent / 'union_paper5_outer' / 'manifest.json').write_text('{}')
    frozen = {'union_manifest_hash': json_hash(union),
              'split_manifest_sha256': split_hashes}
    (cohort / 'manifest.json').write_text(json.dumps(frozen))
    (cohort / '.frozen').write_text(json.dumps({'manifest_hash': json_hash(frozen)}))

    verify_splits(active, old, cohort, trusted_union)
    trusted_union.write_text('{}')
    with pytest.raises(ValueError, match='cohort/union'):
        verify_splits(active, old, cohort, trusted_union)
    trusted_union.write_text(json.dumps(union))
    (old / f'{PAPER_TASKS[0]}.json').write_text('{}')
    with pytest.raises(ValueError, match='source split SHA'):
        verify_splits(active, old, cohort, trusted_union)
    (cohort / '.frozen').write_text('{}')
    with pytest.raises(ValueError, match='manifest seal'):
        verify_splits(active, old, cohort, trusted_union)


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
           ('INCOMPLETE', 40, 39)
    missing.clear()
    result = aggregate(tmp_path, arms=('m_cat',), expected_step=5000)
    assert result['status'] == 'PASS' and result['units_accepted'] == 40
    assert result['test_r2']['m_cat']['tasks']['eea']['mean'] == pytest.approx(.2)
