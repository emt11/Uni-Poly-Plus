from types import SimpleNamespace
import numpy as np
import pytest
from scripts import finetune_mcl_ph as ft


def test_r5_scope_keeps_legacy_limits_and_rejects_other_units():
    ft.validate_stage_scope('development', 'xc', 0, 70, 'index',
        experiment_plan=ft.R5_PLAN, arm='glt_ref', target_transform='standard')
    for fold in (2, 3, 4):
        ft.validate_stage_scope('development', 'nc', fold, 70, 'index',
            experiment_plan=ft.R5_PLAN, arm='m_xattn', target_transform='standard')
    ft.validate_stage_scope('smoke', 'xc', 0, 2, 'index',
        experiment_plan=ft.R5_PLAN, arm='glt_ref', target_transform='standard')
    for stage, task, fold, arm, transform in [
        ('development', 'xc', 2, 'glt_ref', 'standard'),
        ('development', 'nc', 1, 'm_xattn', 'standard'),
        ('development', 'xc', 0, 'glt_ref', 'identity'),
        ('development', 'nc', 2, 'glt_ref', 'standard'),
        ('smoke', 'nc', 0, 'm_xattn', 'standard')]:
        with pytest.raises(ValueError):
            ft.validate_stage_scope(stage, task, fold, 2 if stage == 'smoke' else 70, 'index',
                experiment_plan=ft.R5_PLAN, arm=arm, target_transform=transform)
    with pytest.raises(ValueError):
        ft.validate_stage_scope('development', 'nc', 2, 70, 'index')


def test_explicit_target_transform_is_not_silent_global_change():
    assert ft.resolve_target_transform(SimpleNamespace(task='xc')) == 'identity'
    with pytest.raises(ValueError):
        ft.resolve_target_transform(SimpleNamespace(task='xc', target_transform='standard'))
    assert ft.resolve_target_transform(SimpleNamespace(task='xc', target_transform='standard',
        experiment_plan=ft.R5_PLAN)) == 'standard'


def test_standard_scaler_fits_train_and_roundtrips_raw_validation():
    from src.utils import scale_targets
    class Dataset:
        raw_targets = np.array([10., 20., 1000.])
        def set_target_override(self, values): self.targets = values
    data = Dataset()
    scaler = scale_targets(data, 'xc', train_indices=[0, 1], transform_mode='standard')
    np.testing.assert_array_equal(scaler.scaler.mean_, [15.])
    np.testing.assert_array_equal(scaler.scaler.scale_, [5.])
    np.testing.assert_allclose(scaler.inverse_transform(data.targets).reshape(-1), data.raw_targets)


def test_task_gates_use_new_nc_folds_and_reject_regression():
    from scripts.run_mcl_ph_r5 import task_gate
    assert task_gate('xc',[.97,.98],[1.,1.])['passed']
    assert not task_gate('xc',[.94,1.001],[1.,1.])['passed']
    assert task_gate('nc',[.97,.98,1.],[1.,1.,1.])['passed']
    assert not task_gate('nc',[.9,.9,1.021],[1.,1.,1.])['passed']
    with pytest.raises(AssertionError):task_gate('nc',[.9,.9],[1.,1.])
    with pytest.raises(AssertionError):task_gate('xc',[np.nan,.9],[1.,1.])


@pytest.mark.parametrize('missing_freeze',[False,True])
def test_r5_cli_requires_frozen_exact_transform_contract(tmp_path,monkeypatch,missing_freeze):
    import json,sys
    from scripts.run_mcl_ph_r5 import unit_command
    selection=tmp_path/'selection.json'
    selection.write_text(json.dumps(dict(status='FROZEN',allowed_units=['glt_ref/xc/fold0'],
        unit_contracts={'glt_ref/xc/fold0':{'target_transform':'identity'}})))
    cmd,folder,_,_=unit_command('paper5_outer','xc_standard',0,
        selection=None if missing_freeze else selection,reuse=tmp_path/'old')
    cmd[cmd.index('--output')+1]=str(tmp_path/'output')
    folder=tmp_path/'output/glt_ref/xc/fold0'
    monkeypatch.setattr(sys,'argv',cmd[1:])
    with pytest.raises(ValueError,match='frozen'):
        ft.main()
    assert not folder.exists()


def test_command_uses_explicit_candidate_scope_and_fixed_recipe():
    from scripts.run_mcl_ph_r5 import unit_command
    cmd,_,arm,task=unit_command('development','nc_fixed_m_xattn',4)
    assert (arm,task)==('m_xattn','nc')
    assert cmd[cmd.index('--target-transform')+1]=='standard'
    assert cmd[cmd.index('--experiment-plan')+1]==ft.R5_PLAN
    assert cmd[cmd.index('--epochs')+1]=='70'
    assert cmd[cmd.index('--expected-pretrain-step')+1]=='5000'
