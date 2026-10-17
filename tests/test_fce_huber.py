"""r13 CPU fixtures only; no real training, GPU probing or old test replay."""
import copy
import numpy as np
import pytest
import torch
from torch import nn
from src.training.fce_param import (HUBER_FT_LOSS_POLICY,finetune_criterion,FixedTwiceHuberLoss,recipe_for)
from scripts import run_fce_huber_unit as unit
from scripts import run_fce_huber as controller


def test_small_residual_values_and_gradients_match_mse():
    x=torch.tensor([-.99,-.5,0.,.5,.99,1.],dtype=torch.float64,requires_grad=True)
    y=torch.zeros_like(x);criterion=finetune_criterion(HUBER_FT_LOSS_POLICY)
    loss=criterion(x,y);grad=torch.autograd.grad(loss,x)[0]
    other=x.detach().clone().requires_grad_();mse=finetune_criterion()(other,y)
    torch.testing.assert_close(loss,mse);torch.testing.assert_close(grad,torch.autograd.grad(mse,other)[0])
    assert isinstance(finetune_criterion(),nn.MSELoss)


def test_large_residual_saturation_fixed_delta_and_shape_no_broadcast():
    x=torch.tensor([-4.,-2.,2.,4.],dtype=torch.float64,requires_grad=True);y=torch.zeros_like(x)
    criterion=finetune_criterion(HUBER_FT_LOSS_POLICY);loss=criterion(x,y)
    torch.testing.assert_close(loss,(2*x.abs()-1).mean())
    torch.testing.assert_close(torch.autograd.grad(loss,x)[0],2*x.sign()/len(x))
    assert criterion.huber.delta==1. and criterion.huber.reduction=='mean'
    with pytest.raises(ValueError):finetune_criterion(dict(HUBER_FT_LOSS_POLICY,delta=2.))
    for a,b in ((torch.ones(3,1),torch.ones(3)),(torch.ones(2,2),torch.ones(2,2))):
        with pytest.raises(ValueError):criterion(a,b)
    with pytest.raises(FloatingPointError):criterion(torch.tensor([float('nan')]),torch.zeros(1))


@pytest.mark.parametrize('task',unit.TASKS)
def test_seven_tasks_public_loss_and_unchanged_r0_both_stages(task,monkeypatch):
    cfg=unit.config_for(task);captured={}
    def capture(model,train,validation,criterion,device,**kwargs):
        captured.update(kwargs,criterion=criterion)
        # The unchanged common epoch loop supplies this one object to head and joint.
        assert isinstance(criterion,FixedTwiceHuberLoss)
        return 'same-public-epoch-loop'
    monkeypatch.setattr(unit,'run_periodic_tdl_epochs',capture)
    assert unit.train_candidate(None,None,None,'cpu',cfg=cfg,task=task,scaler=None,progress=None,step_observer=None)=='same-public-epoch-loop'
    assert captured['recipe']==recipe_for('R0',task) and captured['task']==task
    assert cfg['seed']=='42+fold' and cfg['loss_policy']==HUBER_FT_LOSS_POLICY
    bad=copy.deepcopy(cfg);bad['recipe']['joint_backbone_lr']*=.3
    with pytest.raises(ValueError):unit.train_candidate(None,None,None,'cpu',cfg=bad,task=task,scaler=None,progress=None,step_observer=None)


def test_original_unit_rmse_best_strict_tie_not_huber_or_fusion():
    rows=[dict(epoch=1,validation_rmse=2.,validation_huber2_loss=.1,fused_rmse=.01),dict(epoch=2,validation_rmse=1.,validation_huber2_loss=99.,fused_rmse=99.),dict(epoch=3,validation_rmse=1.,validation_huber2_loss=.001,fused_rmse=.001)]
    assert unit.selected_history(rows)['epoch']==2
    assert unit.POLICY['prediction_weight']==.75
    assert unit.POLICY['loss']['best_selection'].startswith('original-unit neural validation RMSE')


def bindings():
    result=[]
    for task in unit.TASKS:
        cfg=unit.config_for(task)
        for fold in range(5):
            scaler=dict(mean=[2.],scale=[1.],fit='train',transform='standard')
            result.append(dict(task=task,fold=fold,seed=42+fold,config=cfg,parent=str(unit.PARENT),head_initial=str(unit.head_path(task,fold)),loss_policy=copy.deepcopy(HUBER_FT_LOSS_POLICY),
                best_epoch=4,scaler=scaler,expected_updates=100,reference=dict(seed=42+fold,config=dict(seed='42+fold',recipe=cfg['recipe'])),fixed_krr=dict(scaler=scaler)))
    return result


def test_checkpoint_and_complete_freeze_bind_loss_seed_recipe():
    all_units=bindings();frozen=dict(frozen=True,policy=copy.deepcopy(unit.POLICY),units=all_units);unit.validate_frozen(frozen)
    b=all_units[0];meta={k:b[k] for k in ('config','task','fold','seed','parent','head_initial','best_epoch','scaler','loss_policy')};meta['optimizer_updates']=100
    unit.validate_checkpoint(meta,b)
    for key,value in [('loss_policy',None),('seed','42+fold'),('best_epoch',8),('optimizer_updates',101)]:
        bad=dict(meta);bad[key]=value
        with pytest.raises(ValueError):unit.validate_checkpoint(bad,b)
    for mutate in (lambda p:p.update(frozen=False),lambda p:p['units'].pop(),lambda p:p['units'][0].update(seed='42+fold'),lambda p:p['units'][0].update(loss_policy=None)):
        bad=copy.deepcopy(frozen);mutate(bad)
        with pytest.raises(ValueError):unit.validate_frozen(bad)


def test_budget_complete_matrix_no_krr_pt_or_extra_runs():
    controller.check_budget(dict(controller.LIMITS));controller.require_stage('CPU',dict(controller.LIMITS))
    for key in controller.LIMITS:
        bad=dict(controller.LIMITS);bad[key]+=1
        with pytest.raises(ValueError):controller.check_budget(bad)
    for key in ('ft_completed','ft_epochs','ft_attempted_updates','ft_completed_updates','validation_calls','formal_completed','formal_outputs'):
        bad=dict(controller.LIMITS);bad[key]-=1
        with pytest.raises(ValueError):controller.require_stage('CPU',bad)


def test_existing_gpu_contexts_and_pending_reservations_count(monkeypatch,tmp_path):
    monkeypatch.setattr(controller,'OUT',tmp_path)
    def query(argv,**kwargs):
        if '--query-gpu=index,uuid' in argv:return '2, gpu-two\n3, gpu-three\n'
        return 'gpu-two, 100\ngpu-two, 101\n'
    monkeypatch.setattr(controller.subprocess,'check_output',query)
    running={str(i):dict(gpu=2,task='eea',fold=i,formal=False) for i in range(2)}
    count,observed=controller.gpu_slots(2,running)
    assert count==0 and observed['conservative_occupied']==4
    count,_=controller.gpu_slots(3,{})
    assert count==4


def test_existing_output_root_is_not_replayed(tmp_path,monkeypatch):
    monkeypatch.setattr(controller,'require_tmux',lambda:None)
    existing=tmp_path/'owned';existing.mkdir();monkeypatch.setattr(controller,'OUT',existing);monkeypatch.setattr(controller,'LOG',tmp_path/'newlogs')
    with pytest.raises(FileExistsError):controller.setup()
    assert not (tmp_path/'newlogs').exists()
