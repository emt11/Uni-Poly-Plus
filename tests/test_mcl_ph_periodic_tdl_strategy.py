"""Model-free checks for the active Periodic-TDL fine-tuning schedule."""

import numpy as np
import pytest
import torch
from torch import nn

from scripts import finetune_mcl_ph as runner
from src.utils import scale_targets, set_global_seed


class TinyArm(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(1, 1), nn.Dropout(0.1))
        self.head = nn.Sequential(nn.Linear(1, 1), nn.Dropout(0.1))


def test_xc_keeps_original_target_scale_and_records_real_joint_lr():
    class Targets:
        raw_targets = np.array([20.0, 40.0, 80.0])
        def set_target_override(self, values):
            self.targets = values

    dataset = Targets()
    scaler = scale_targets(dataset, 'xc', train_indices=[0, 1], transform_mode='identity')
    assert scaler.transform_mode == 'identity'
    assert dataset.targets.tolist() == [20.0, 40.0, 80.0]
    assert scaler.inverse_transform([20.0, 40.0]).reshape(-1).tolist() == [20.0, 40.0]
    arm = TinyArm()
    _, groups = runner.periodic_tdl_optimizer(arm, 'xc', head_only=False)
    assert [(group['name'], group['lr']) for group in groups] == [
        ('backbone', 2e-4), ('head', 1e-3)]


def test_reference_predictor_seed_and_initial_hash_are_reproducible():
    class Reference(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.predictor = nn.Linear(4, 1)

    set_global_seed(43)
    first = runner.head_initial_sha256(Reference())
    set_global_seed(43)
    assert runner.head_initial_sha256(Reference()) == first
    set_global_seed(44)
    assert runner.head_initial_sha256(Reference()) != first


def test_periodic_tdl_scope_and_task_hyperparameters():
    for task in runner.TASKS:
        runner.validate_stage_scope(
            'paper5_outer', task, 4, 60 if task == 'egc' else 70,
            'index', 'periodic_tdl')
    with pytest.raises(ValueError, match='only paper5_outer'):
        runner.validate_stage_scope('full8x5', 'eea', 4, 70, 'index', 'periodic_tdl')
    with pytest.raises(ValueError, match='requires 60'):
        runner.validate_stage_scope('paper5_outer', 'egc', 4, 70, 'index', 'periodic_tdl')
    arm = TinyArm()
    runner.configure_periodic_tdl_dropout(arm, 'eea')
    assert arm.encoder[1].p == 0.0
    assert arm.head[1].p == 0.1
    runner.periodic_tdl_trainability(arm, head_only=True)
    assert not arm.encoder[0].weight.requires_grad
    assert arm.head[0].weight.requires_grad
    optimizer, groups = runner.periodic_tdl_optimizer(arm, 'eea', head_only=True)
    assert [group['name'] for group in groups] == ['head']
    assert optimizer.param_groups[0]['lr'] == 3e-4
    runner.periodic_tdl_trainability(arm, head_only=False)
    _, groups = runner.periodic_tdl_optimizer(arm, 'eea', head_only=False)
    assert [(group['name'], group['lr']) for group in groups] == [
        ('backbone', 1e-4), ('head', 1e-3)]


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
        error = 0.1 if epoch == 1 else 0.2
        return 0.1, (0.1 if epoch == 1 else 0.9), np.array([1.0]), np.array([1.0 + error])

    monkeypatch.setattr(runner, 'train_epoch', fake_train)
    monkeypatch.setattr(runner, 'evaluate', fake_evaluate)
    result = runner.run_periodic_tdl_epochs(
        arm, [None], [None], nn.MSELoss(), torch.device('cpu'),
        task='eea', scaler=None)
    assert seen == ['head'] * 10 + ['joint'] * 60
    assert result['best_epoch'] == 1
    assert result['best_rmse'] == pytest.approx(0.1)
    assert result['best_r2'] == 0.1
    assert result['optimizer_updates'] == 70
    assert result['history'][9]['learning_rates'][0] == pytest.approx(3e-5)
    assert result['history'][19]['learning_rates'][0] == pytest.approx(1e-4)
