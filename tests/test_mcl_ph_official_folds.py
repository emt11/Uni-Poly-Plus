"""Check the released Periodic-TDL rows and fold indices against our cohort."""
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd
import pytest
import numpy as np
from sklearn.model_selection import train_test_split

from scripts.aggregate_mcl_ph_8x5 import aggregate
from scripts.aggregate_mcl_ph import check_unit
from scripts.finetune_glt_3d_gain_d2 import resolve_fold
from scripts.finetune_mcl_ph import (PAPER_SPLIT_PROTOCOL, PAPER_TASKS,
                                     validate_stage_scope)


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / 'configs/mts/periodic_tdl_official5'
COHORT = ROOT / 'data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1/records.jsonl'


def test_official_rows_and_exact_outer_inner_folds():
    cohort = {task: [] for task in PAPER_TASKS}
    for line in COHORT.open(encoding='utf-8'):
        row = json.loads(line)
        if row['task'] in cohort:
            cohort[row['task']].append(row)
    for task in PAPER_TASKS:
        manifest = json.loads((SPLITS / f'{task}.json').read_text())
        source = manifest['official_source']
        assert source['commit'] == 'f3ba6dff6f0d065accdd235dfba160324714f30b'
        name = 'EPS' if task == 'eps' else task.capitalize()
        csv = SPLITS / 'source' / f'{name}_cleaned.csv'
        pkl = SPLITS / 'source' / f'{name}_folds.pkl'
        assert hashlib.sha256(csv.read_bytes()).hexdigest() == source['cleaned_csv_sha256']
        assert hashlib.sha256(pkl.read_bytes()).hexdigest() == source['folds_pkl_sha256']
        official = pd.read_csv(csv)
        local = pd.read_csv(ROOT / f'data/raw/smi_{task}.csv')
        assert official.smiles.equals(local.smiles)
        assert official.value.equals(local.iloc[:, 1])
        assert len(official) == len(cohort[task]) == manifest['sample_count']
        assert all(row['original_row'] == i and row['source_smiles'] == smiles
                   and row['label'] == value for i, (row, smiles, value)
                   in enumerate(zip(cohort[task], official.smiles, official.value)))
        # Only unpickle after checking the immutable released-file digest.
        released = pickle.loads(pkl.read_bytes())
        test_coverage = []
        for fold, source_fold in enumerate(released):
            entry = manifest['folds'][fold]
            train, validation = train_test_split(source_fold['train'], test_size=.2,
                                                 shuffle=True, random_state=42)
            assert entry['test_indices'] == source_fold['test']
            assert entry['train_indices'] == train
            assert entry['validation_indices'] == validation
            resolved_train, resolved_validation, evidence = resolve_fold(
                manifest, task, fold, cohort_rows=len(official),
                expected_protocol=PAPER_SPLIT_PROTOCOL)
            assert resolved_train == train and resolved_validation == validation
            assert evidence['test_rows'] == len(source_fold['test'])
            test_coverage.extend(source_fold['test'])
        assert sorted(test_coverage) == list(range(len(official)))


def test_paper_scope_rejects_unmatched_tasks():
    for task in PAPER_TASKS:
        validate_stage_scope('paper5_outer', task, 0, 70, 'trusted-index', 'periodic_tdl')
    for task in ('eat', 'egc', 'xc'):
        with pytest.raises(ValueError, match='verified only'):
            validate_stage_scope('paper5_outer', task, 0, 70, 'trusted-index', 'periodic_tdl')


def test_paper_aggregate_requires_only_five_matched_tasks(monkeypatch, tmp_path):
    import scripts.aggregate_mcl_ph_8x5 as module

    def check(_root, arm, task, fold, *, stage, expected_step):
        assert stage == 'paper5_outer' and expected_step == 5000
        return {'arm': arm, 'task': task, 'fold': fold,
                'finetune_strategy': 'periodic_tdl',
                'best_validation_r2': .5, 'test_r2': fold / 10,
                'pretrain_package_sha256': 'a' * 64}, []

    monkeypatch.setattr(module, 'check_unit', check)
    result = aggregate(tmp_path, arms=('m_cat',), expected_step=5000,
                       stage='paper5_outer')
    assert result['status'] == 'PASS' and result['units_expected'] == 25
    assert result['tasks'] == list(PAPER_TASKS)
    assert result['test_r2']['m_cat']['macro5'] == pytest.approx(.2)


def test_paper_unit_acceptance_requires_official_identity_and_test_r2(tmp_path):
    path = SPLITS / 'eea.json'
    manifest = json.loads(path.read_text())
    fold = manifest['folds'][0]
    folder = tmp_path / 'm_cat/eea/fold0'
    folder.mkdir(parents=True)
    history = [dict(stage='head' if i < 10 else 'joint', training_steps=1,
                    train_loss=.1, validation_loss=.1, validation_r2=.5,
                    validation_rmse=.1 if i == 0 else .2) for i in range(70)]
    truth = np.arange(len(fold['test_indices']), dtype=np.float64)
    prediction = truth + .1
    r2 = 1 - np.sum((truth - prediction)**2) / np.sum((truth - truth.mean())**2)
    common = dict(arm='m_cat', task='eea', fold=0, stage='paper5_outer',
                  protocol='mcl_ph_paper5_outer', finetune_strategy='periodic_tdl',
                  outer_test='RUN', pretrain_step=5000, pretrained_route='mcl_ph',
                  requested_epochs=70, executed_epochs=70, optimizer_updates=70,
                  history=history, best_epoch=1, best_validation_r2=.5,
                  best_validation_rmse=.1, selection_metric='validation_rmse',
                  test_r2=float(r2), test_sample_count=len(truth),
                  evaluation_split_path=str(path),
                  evaluation_split_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  cohort_split_sha256='0'*64,
                  official_source=manifest['official_source'],
                  optimizer_groups=[{'name':'backbone','num_parameters':1},
                                    {'name':'head','num_parameters':1}],
                  optimizer_groups_by_stage={'head':[{'name':'head'}],
                                             'joint':[{'name':'backbone'}, {'name':'head'}]},
                  split={'protocol':PAPER_SPLIT_PROTOCOL,'validation_is_test':False,
                         'outer_test':'RUN','sets_disjoint':True,
                         'union_equals_full_cohort':True,
                         'train_rows':235,'validation_rows':59,'test_rows':74})
    for name in ('metrics.json', 'run.json'):
        (folder / name).write_text(json.dumps(common))
    (folder / 'runtime.json').write_text(json.dumps(
        {'arm':'m_cat','task':'eea','fold':0,'status':'PASS','exit_code':0}))
    (folder / 'best.pt').write_bytes(b'fixture')
    np.savez(folder / 'validation_predictions.npz', y_true=[1.], y_pred=[1.],
             sample_keys=['one'], validation_indices=[fold['train_indices'][0]],
             best_epoch=1, outer_test='NOT_RUN', split_protocol=PAPER_SPLIT_PROTOCOL)
    np.savez(folder / 'test_predictions.npz', y_true=truth, y_pred=prediction,
             sample_keys=[str(i) for i in range(len(truth))],
             test_indices=fold['test_indices'], best_epoch=1,
             outer_test='RUN', split_protocol=PAPER_SPLIT_PROTOCOL)
    record, problems = check_unit(tmp_path, 'm_cat', 'eea', 0,
                                  stage='paper5_outer', expected_step=5000)
    assert problems == [] and record['test_r2'] == pytest.approx(r2)
    common['evaluation_split_sha256'] = '1'*64
    (folder / 'metrics.json').write_text(json.dumps(common))
    _, problems = check_unit(tmp_path, 'm_cat', 'eea', 0,
                             stage='paper5_outer', expected_step=5000)
    assert 'official evaluation split file does not match unit identity' in problems
