"""Development must save validation evidence without opening outer-test rows."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from scripts import finetune_mcl_ph as runner


def test_mcl_o8_extraction_and_scale_preserve_loaded_tensors():
    from src.modules.mcl_ph import MCLPHEncoder, deployment_package
    encoder = MCLPHEncoder('xattn')
    package = deployment_package(encoder, 2)
    o8, copied = runner.build_mcl_o8_only(package, expected_step=2)
    assert copied == ['o8.' + name for name in encoder.o8.state_dict()]
    for name, tensor in encoder.o8.state_dict().items():
        assert torch.equal(tensor, o8.o8.state_dict()[name])
    arm = runner.build_mcl_arm('xattn', package, expected_step=2)
    arm.encoder.fusion.enable_learnable_scale()
    expected = package['state_dict']
    actual = arm.encoder.state_dict()
    assert set(actual) - set(expected) == {'fusion.residual_scale'}
    for name, tensor in expected.items():
        assert torch.equal(tensor, actual[name])
    torch.testing.assert_close(actual['fusion.residual_scale'], torch.tensor(.1))


def test_development_scope_and_smoke_budget():
    runner.validate_stage_scope('development', 'ei', 0, 70, 'index')
    runner.validate_stage_scope('smoke', 'ei', 0, 2, 'index')
    for stage, task, fold, epochs in [
        ('development', 'eea', 0, 70), ('development', 'nc', 2, 70),
        ('smoke', 'ei', 1, 2), ('smoke', 'ei', 0, 3),
    ]:
        with pytest.raises(ValueError):
            runner.validate_stage_scope(stage, task, fold, epochs, 'index')


@pytest.mark.parametrize('direct,task', [(False, 'ei'), (True, 'ei'), (True, 'xc')])
def test_development_run_never_opens_test(monkeypatch, tmp_path, direct, task):
    class Arm(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(1, 1)
            self.model = nn.Linear(1, 1)

    class Source:
        samples = [(bytes([i]) * 32, '*CC*') for i in range(3)]
        def __len__(self):
            return len(self.samples)
        def close(self):
            pass

    model = Arm()
    monkeypatch.setattr(runner, 'GLTReferenceArm', lambda **kwargs: model)
    monkeypatch.setattr(runner, 'load_dual_deployment', lambda *args: None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(runner, 'resolve_fold', lambda *args, **kwargs:
                        ([0, 1], [2], {'protocol': 'paper5_outer'}))
    opened = []
    def source(*args, selected_indices, **kwargs):
        opened.append(selected_indices)
        assert selected_indices == [0, 1, 2], 'outer-test rows were opened'
        return Source(), pd.DataFrame({'original_row': [0, 1, 2],
                                       'label': [1., 2., 3.]})
    monkeypatch.setattr(runner, 'open_source', source)
    groups = {'head': [{'name': 'head', 'lr': 3e-4}],
              'joint': [{'name': 'backbone', 'lr': 1e-4},
                        {'name': 'head', 'lr': 1e-3}]}
    monkeypatch.setattr(runner, 'run_periodic_tdl_epochs', lambda *args, **kwargs: {
        'optimizer_groups_by_stage': groups, 'state': model.state_dict(),
        'history': [{}] * 70, 'predictions': (np.array([3.]), np.array([2.5])),
        'best_epoch': 12, 'best_rmse': .5, 'best_r2': .2, 'optimizer_updates': 0})
    monkeypatch.setattr(runner, 'evaluate', lambda *args, **kwargs:
                        pytest.fail('development invoked outer-test evaluation'))
    manifest = {'sample_count': 5, 'sample_order_sha256': 'fixture',
                'protocol': 'paper5_outer'}
    (tmp_path / f'{task}.json').write_text(json.dumps(manifest))
    (tmp_path / 'config.json').write_text('{}')
    torch.save({}, tmp_path / 'package.pt')
    if direct:
        (tmp_path / 'direct_source.json').write_text('{}')
        monkeypatch.setattr(runner, 'sha256_file', lambda *args:
                            pytest.fail('direct execution used a file digest'))
        monkeypatch.setattr(runner, 'head_initial_sha256', lambda *args:
                            pytest.fail('direct execution used a tensor digest'))
    output = tmp_path / 'output'
    output.mkdir()
    args = SimpleNamespace(
        stage='development', arm='glt_ref', task=task, fold=0, epochs=70,
        cohort_split_root=tmp_path, cohort_union_manifest=tmp_path / 'union.json',
        checkpoint=tmp_path / 'package.pt', expected_pretrain_step=5000,
        split_root=tmp_path, cohort_root=tmp_path, cache_root=tmp_path,
        dual_static_root=tmp_path, cohort_index=tmp_path / 'index.json',
        config=tmp_path / 'config.json', finetune_strategy='periodic_tdl')
    if task == 'xc':
        args.experiment_plan = runner.R5_PLAN
        args.target_transform = 'standard'
    result = runner.run_unit(args, output, runner.time.perf_counter(), None,
                             {'seed': 42}, manifest)
    assert opened == [[0, 1, 2]]
    assert result['outer_test'] == result['split']['outer_test'] == 'NOT_RUN'
    assert not (output / 'test_predictions.npz').exists()
    assert (output / 'best.pt').exists()
    if task == 'xc':
        assert result['target_transform'] == 'standard'
        assert result['scaler_fit_split'] == 'train'
    if direct:
        assert (output / 'head_initial.pt').exists()
        assert result['source_identity_contract'] == 'direct'
    with np.load(output / 'validation_predictions.npz') as predictions:
        assert predictions['outer_test'].item() == 'NOT_RUN'
    if direct:
        # Only after development is complete may the selected checkpoint read
        # test rows. Promotion must never invoke the epoch/optimizer loop.
        manifest['folds'] = [{'fold': 0, 'test_indices': [3, 4]}]
        def promotion_source(*args, selected_indices, **kwargs):
            opened.append(selected_indices)
            instance = Source()
            instance.samples = [(bytes([i]) * 32, '*CC*') for i in selected_indices]
            return instance, pd.DataFrame({'original_row': selected_indices,
                'label': [float(i + 1) for i in selected_indices]})
        monkeypatch.setattr(runner, 'open_source', promotion_source)
        monkeypatch.setattr(runner, 'run_periodic_tdl_epochs', lambda *args, **kwargs:
                            pytest.fail('promotion retrained the development checkpoint'))
        monkeypatch.setattr(runner, 'evaluate', lambda *args, **kwargs:
                            (0., .5, np.array([4., 5.]), np.array([4., 4.5])))
        args.stage = 'paper5_outer'
        args.reuse_development_unit = output
        args.frozen_selection = tmp_path / 'frozen_selection.json'
        promoted = tmp_path / 'promoted'
        promoted.mkdir()
        result = runner.run_unit(args, promoted, runner.time.perf_counter(), None,
                                 {'seed': 42}, manifest)
        assert opened == [[0, 1, 2], [0, 1, 2], [3, 4]]
        assert result['promotion_optimizer_updates'] == 0
        assert result['outer_test'] == 'RUN'
        assert not (output / 'test_predictions.npz').exists()
        assert (promoted / 'test_predictions.npz').exists()
        if task == 'xc':
            assert result['target_transform'] == 'standard'
            state = torch.load(promoted / 'best.pt', weights_only=False)
            assert state['scaler_mean'] == [1.5] and state['scaler_scale'] == [.5]
