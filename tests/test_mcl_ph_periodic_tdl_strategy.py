"""Model-free checks for the paper-inspired fine-tuning schedule."""

import json
import numpy as np
import pytest
import torch
from torch import nn

from scripts import finetune_mcl_ph as runner
from scripts.aggregate_mcl_ph import check_unit


class TinyArm(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(1, 1), nn.Dropout(0.1))
        self.head = nn.Sequential(nn.Linear(1, 1), nn.Dropout(0.1))


def test_periodic_tdl_scope_and_task_hyperparameters():
    runner.validate_stage_scope('full8x5', 'egc', 4, 60, 'index', 'periodic_tdl')
    runner.validate_stage_scope('full8x5', 'xc', 4, 70, 'index', 'periodic_tdl')
    with pytest.raises(ValueError, match='requires 60 epochs'):
        runner.validate_stage_scope('full8x5', 'egc', 4, 70, 'index', 'periodic_tdl')
    arm = TinyArm()
    runner.configure_periodic_tdl_dropout(arm, 'xc')
    assert arm.encoder[1].p == 0.0
    assert arm.head[1].p == 0.3
    runner.periodic_tdl_trainability(arm, head_only=True)
    assert not arm.encoder[0].weight.requires_grad
    assert arm.head[0].weight.requires_grad
    optimizer, groups = runner.periodic_tdl_optimizer(arm, 'xc', head_only=True)
    assert [group['name'] for group in groups] == ['head']
    assert optimizer.param_groups[0]['lr'] == 3e-4
    runner.periodic_tdl_trainability(arm, head_only=False)
    _, groups = runner.periodic_tdl_optimizer(arm, 'xc', head_only=False)
    assert [(group['name'], group['lr']) for group in groups] == [
        ('backbone', 2e-4), ('head', 1e-3)]


def test_periodic_tdl_selects_rmse_across_both_stages(monkeypatch):
    arm = TinyArm()
    seen = []

    def fake_train(wrapped, loader, criterion, optimizer, scheduler, device, **kwargs):
        wrapped.train()
        stage = 'head' if len(optimizer.param_groups) == 1 else 'joint'
        seen.append(stage)
        assert arm.encoder.training is (stage == 'joint')
        assert arm.head.training
        optimizer.step()
        scheduler.step()
        return (0.1, None, None, None, None, None, {'training_steps': 1})

    def fake_evaluate(*args, **kwargs):
        epoch = len(seen)
        # R2 prefers joint training; RMSE prefers the first frozen-head epoch.
        error = 0.1 if epoch == 1 else 0.2
        return 0.1, (0.1 if epoch == 1 else 0.9), np.array([1.0]), np.array([1.0 + error])

    monkeypatch.setattr(runner, 'train_epoch', fake_train)
    monkeypatch.setattr(runner, 'evaluate', fake_evaluate)
    result = runner.run_periodic_tdl_epochs(
        arm, [None], [None], nn.MSELoss(), torch.device('cpu'),
        task='egc', scaler=runner.IdentityTargetScaler())
    assert seen == ['head'] * 10 + ['joint'] * 50
    assert result['best_epoch'] == 1
    assert result['best_rmse'] == pytest.approx(0.1)
    assert result['best_r2'] == 0.1
    assert result['optimizer_updates'] == 60
    assert result['history'][9]['learning_rates'][0] == pytest.approx(3e-5)
    assert result['history'][19]['learning_rates'][0] == pytest.approx(1e-4)


def test_periodic_tdl_acceptance_uses_rmse_and_complete_five_fold_unit(tmp_path):
    folder = tmp_path / 'm_cat' / 'egc' / 'fold4'
    folder.mkdir(parents=True)
    history = [dict(epoch=i + 1, stage='head' if i < 10 else 'joint',
                    training_steps=1, train_loss=0.1, validation_loss=0.1,
                    validation_r2=0.9 if i else 0.2,
                    validation_rmse=0.1 if i == 0 else 0.2)
               for i in range(60)]
    common = dict(arm='m_cat', task='egc', fold=4, stage='full8x5',
                  protocol='mcl_ph_full8x5', finetune_strategy='periodic_tdl',
                  outer_test='NOT_RUN', pretrain_step=5000,
                  pretrained_route='mcl_ph', requested_epochs=60,
                  executed_epochs=60, optimizer_updates=60, history=history,
                  best_epoch=1, best_validation_r2=0.2,
                  best_validation_rmse=0.1, selection_metric='validation_rmse',
                  pretrain_package_sha256='a' * 64,
                  optimizer_groups=[{'name': 'backbone', 'num_parameters': 1},
                                    {'name': 'head', 'num_parameters': 1}],
                  optimizer_groups_by_stage={
                      'head': [{'name': 'head'}],
                      'joint': [{'name': 'backbone'}, {'name': 'head'}]},
                  split={'protocol': 'outer5_inner20', 'validation_is_test': False,
                         'outer_test': 'NOT_RUN', 'sets_disjoint': True,
                         'union_equals_full_cohort': True,
                         'train_rows': 10, 'validation_rows': 3})
    (folder / 'metrics.json').write_text(json.dumps(common))
    (folder / 'run.json').write_text(json.dumps(common))
    (folder / 'runtime.json').write_text(json.dumps({
        'arm': 'm_cat', 'task': 'egc', 'fold': 4, 'status': 'PASS', 'exit_code': 0}))
    (folder / 'best.pt').write_bytes(b'fixture')
    np.savez(folder / 'validation_predictions.npz', y_true=np.array([1.0]),
             y_pred=np.array([1.1]), sample_keys=np.array(['a']),
             best_epoch=np.array(1), outer_test=np.array('NOT_RUN'))
    _, problems = check_unit(tmp_path, 'm_cat', 'egc', 4,
                             stage='full8x5', expected_step=5000)
    assert problems == []
    common['best_validation_rmse'] = 0.2
    (folder / 'metrics.json').write_text(json.dumps(common))
    _, problems = check_unit(tmp_path, 'm_cat', 'egc', 4,
                             stage='full8x5', expected_step=5000)
    assert 'the selected epoch is not the minimum validation RMSE' in problems
    common.update(best_validation_rmse=0.1, stage='full8x5_outer',
                  protocol='mcl_ph_full8x5_outer', outer_test='RUN', test_r2=0.99,
                  test_sample_count=2)
    common['split'].update(outer_test='RUN', test_rows=2)
    (folder / 'metrics.json').write_text(json.dumps(common))
    (folder / 'run.json').write_text(json.dumps(common))
    np.savez(folder / 'validation_predictions.npz', y_true=np.array([1.0]),
             y_pred=np.array([1.1]), sample_keys=np.array(['a']),
             validation_indices=np.array([10]), best_epoch=np.array(1),
             split_protocol=np.array('outer5_inner20'),
             outer_test=np.array('NOT_RUN'))
    np.savez(folder / 'test_predictions.npz', y_true=np.array([0., 1.]),
             y_pred=np.array([0.05, .95]), sample_keys=np.array(['b', 'c']),
             test_indices=np.array([11, 12]), best_epoch=np.array(1),
             split_protocol=np.array('outer5_inner20'), outer_test=np.array('RUN'))
    record, problems = check_unit(tmp_path, 'm_cat', 'egc', 4,
                                  stage='full8x5_outer', expected_step=5000)
    assert problems == [] and record['test_r2'] == .99
    common['test_r2'] = .5
    (folder / 'metrics.json').write_text(json.dumps(common))
    _, problems = check_unit(tmp_path, 'm_cat', 'egc', 4,
                             stage='full8x5_outer', expected_step=5000)
    assert 'outer-test R2 disagrees with predictions' in problems
