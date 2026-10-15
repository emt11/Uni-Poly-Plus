"""Development must save validation evidence without opening outer-test rows."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from scripts import finetune_mcl_ph as runner


@pytest.mark.parametrize('direct,task', [(True, 'xc')])
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
        args.experiment_plan = runner.R6_PLAN
        args.target_transform = 'standard'
    args.candidate_config = tmp_path / 'candidate.json'
    args.candidate_config.write_text(Path('results/mcl_ph_r6_20261004/candidate.json').read_text())
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


def test_lr_groups_schedule_and_legacy_default():
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    class Arm(nn.Module):
        def __init__(self):
            super().__init__(); self.head=nn.Linear(1,1);self.backbone=nn.Linear(1,1)
    model=Arm()
    runner.periodic_tdl_trainability(model,head_only=True)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    opt,e=runner.periodic_tdl_optimizer(model,'xc',head_only=True,backbone_lr=1e-4)
    assert [g['lr'] for g in opt.param_groups]==[3e-4]
    runner.periodic_tdl_trainability(model,head_only=False)
    opt,e=runner.periodic_tdl_optimizer(model,'xc',head_only=False,backbone_lr=1e-4)
    assert [g['lr'] for g in opt.param_groups]==[1e-4,1e-3]
    scheduler=CosineAnnealingWarmRestarts(opt,T_0=10,T_mult=1,eta_min=1e-6)
    scheduler.step(5)
    np.testing.assert_allclose([g['lr'] for g in opt.param_groups],[5.05e-5,5.005e-4])
    scheduler.step(10)
    assert [g['lr'] for g in opt.param_groups]==[1e-4,1e-3]
    opt,_=runner.periodic_tdl_optimizer(model,'xc',head_only=False)
    assert opt.param_groups[0]['lr']==2e-4


def test_r6_scope():
    for f in (0,1):
        runner.validate_stage_scope('development','xc',f,70,'index',experiment_plan=runner.R6_PLAN,arm='glt_ref',target_transform='standard')
    for task,fold,arm,transform in [('xc',2,'glt_ref','standard'),('nc',0,'m_xattn','standard'),('xc',0,'glt_ref','identity')]:
        with pytest.raises(ValueError):
            runner.validate_stage_scope('development',task,fold,70,'index',experiment_plan=runner.R6_PLAN,arm=arm,target_transform=transform)

@pytest.mark.parametrize('missing_freeze',[False,True])
def test_r6_frozen_contract_rejects_missing_or_wrong_lr(tmp_path,monkeypatch,missing_freeze):
    import sys
    from scripts.run_mcl_ph_r6 import unit_command
    selection=tmp_path/'selection.json'
    selection.write_text(json.dumps(dict(status='FROZEN',allowed_units=['glt_ref/xc/fold0'],unit_contracts={'glt_ref/xc/fold0':{'backbone_joint_lr':2e-4}})))
    cmd,_,_,_=unit_command('paper5_outer','xc_standard',0,selection=None if missing_freeze else selection,reuse=tmp_path/'old')
    cmd[cmd.index('--output')+1]=str(tmp_path/'output')
    monkeypatch.setattr(sys,'argv',cmd[1:])
    with pytest.raises(ValueError,match='frozen'):
        runner.main()
    assert not (tmp_path/'output').exists()
