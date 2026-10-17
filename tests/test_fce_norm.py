"""r14 final-only CPU fixtures and direct saved frozen tensor comparison."""
import copy
import numpy as np
import pytest
import torch
from torch import nn
from src.training import fce_param as common
from scripts import run_fce_norm_unit as unit
from scripts import run_fce_norm as controller


class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.model=nn.Module()
        self.model.norm_named_but_linear=nn.Linear(3,3)
        self.model.actual=nn.Sequential(nn.LayerNorm(3),nn.Linear(3,3))
        self.model.predictor=nn.Sequential(nn.Linear(3,3),nn.LayerNorm(3),nn.Linear(3,1))
    def forward(self,x):
        return self.model.predictor(self.model.actual(self.model.norm_named_but_linear(x)))


def test_identity_stage_groups_lr_wd_and_gradient_through_frozen_ops():
    m=Toy();policy=common.NORM_ADAPTATION_POLICY;partition=common.norm_parameter_partition(m)
    assert partition['encoder_layernorm']['parameter_names']==['model.actual.0.weight','model.actual.0.bias']
    for head_only,key in [(True,'head'),(False,'joint')]:
        common.periodic_tdl_trainability(m,head_only=head_only,adaptation_policy=policy)
        assert {n for n,p in m.named_parameters() if p.requires_grad}==set(partition[key]['parameter_names'])
        opt,evidence=common.periodic_tdl_optimizer(m,'xc',head_only=head_only,recipe=common.recipe_for('R0','xc'),adaptation_policy=policy)
        members=[id(p) for g in opt.param_groups for p in g['params']]
        assert len(members)==len(set(members)) and set(members)=={id(p) for p in m.parameters() if p.requires_grad}
        assert [(g['name'],g['lr']) for g in opt.param_groups]==([('head',3e-4)] if head_only else [('backbone',2e-4),('head',1e-3)])
        assert all(g['weight_decay']==common.recipe_for('R0','xc')['weight_decay'] for g in opt.param_groups)
    x=torch.tensor([[1.,-2.,3.],[2.,4.,-1.]])
    common.finetune_criterion()(m(x),torch.ones(2,1)).backward()
    assert m.model.actual[0].weight.grad is not None and torch.isfinite(m.model.actual[0].weight.grad).all()
    assert m.model.actual[0].weight.grad.abs().sum()>0
    assert all(p.grad is None for n,p in m.named_parameters() if n in partition['frozen_encoder']['parameter_names'])
    common.periodic_tdl_trainability(m,head_only=False)
    assert all(p.requires_grad for p in m.parameters())
    assert isinstance(common.finetune_criterion(),nn.MSELoss)


@pytest.mark.parametrize('override',[dict(ft_candidate='FT_PARTIAL'),dict(profile='uniform_u1'),dict(backbone_lr=1e-4),dict(smoke=True),dict(recipe=common.recipe_for('B_LR','ei'))])
def test_adaptation_override_rejected(override):
    with pytest.raises(ValueError):common.validate_adaptation(common.NORM_ADAPTATION_POLICY,**override)


@pytest.mark.parametrize('task',unit.TASKS)
def test_new_entry_uses_public_mse_r0_policy(task,monkeypatch):
    cfg=unit.config_for(task);seen={}
    def capture(model,train,validation,criterion,device,**kw):
        seen.update(kw);assert isinstance(criterion,nn.MSELoss);return 'public'
    monkeypatch.setattr(unit,'run_periodic_tdl_epochs',capture)
    assert unit.train_candidate(None,None,None,'cpu',cfg=cfg,task=task,scaler=None,progress=None,step_observer=None)=='public'
    assert seen['recipe']==common.recipe_for('R0',task) and seen['adaptation_policy']==common.NORM_ADAPTATION_POLICY
    bad=copy.deepcopy(cfg);bad['adaptation_policy']=None
    with pytest.raises(ValueError):unit.validate_config(bad,task)


def test_joint_cache_invalidated_before_training_and_rmse_best(monkeypatch):
    from types import SimpleNamespace
    import src.modules.glt_adaptation as adaptation
    m=Toy();m.feature_cache_context={};events=[]
    original=SimpleNamespace(dataset=[],batch_size=2,collate_fn=None)
    cached=SimpleNamespace(cached=True)
    class Cache:
        def __init__(self,*args):pass
        def build(self,*args):events.append('build')
        def loader(self,loader):return cached
        def invalidate(self):events.append('invalidate')
    monkeypatch.setattr(adaptation,'HeadFeatureCache',Cache)
    def train(wrapped,loader,criterion,opt,scheduler,device,**kw):
        if opt.param_groups[0]['name']=='backbone':
            assert events[-1]=='invalidate' and loader is original
            assert m.model.actual[0].weight.requires_grad
        else:assert loader is cached
        return (0.,None,dict(training_steps=1))
    counter=[0]
    def evaluate(*args,**kwargs):
        counter[0]+=1
        # First epoch remains best, irrespective of training loss or stage.
        return 0.,.1,np.array([0.,1.]),np.array([0.,1.])+counter[0]
    monkeypatch.setattr(common,'train_epoch',train);monkeypatch.setattr(common,'evaluate',evaluate)
    result=common.run_periodic_tdl_epochs(m,original,original,common.finetune_criterion(),'cpu',task='egc',scaler=None,recipe=common.recipe_for('R0','egc'),adaptation_policy=common.NORM_ADAPTATION_POLICY)
    assert events==['build','invalidate'] and result['best_epoch']==1 and len(result['history'])==60
    rows=[dict(epoch=1,validation_rmse=1.,validation_mse_loss=99.),dict(epoch=2,validation_rmse=1.,validation_mse_loss=0.)]
    assert unit.selected_history(rows)['epoch']==1


def test_budget_and_overwrite_rejection(monkeypatch,tmp_path):
    controller.check_budget(dict(controller.LIMITS));controller.require_stage('CPU',dict(controller.LIMITS))
    for key in controller.LIMITS:
        bad=dict(controller.LIMITS);bad[key]+=1
        with pytest.raises(ValueError):controller.check_budget(bad)
    for key in ('ft_completed','ft_epochs','ft_attempted_updates','ft_completed_updates','validation_calls','formal_completed','formal_outputs'):
        bad=dict(controller.LIMITS);bad[key]-=1
        with pytest.raises(ValueError):controller.require_stage('CPU',bad)
    monkeypatch.setattr(controller,'require_tmux',lambda:None)
    existing=tmp_path/'owned';existing.mkdir();monkeypatch.setattr(controller,'OUT',existing);monkeypatch.setattr(controller,'LOG',tmp_path/'newlogs')
    with pytest.raises(FileExistsError):controller.setup()
    assert not (tmp_path/'newlogs').exists()


def test_saved_35_bests_frozen_encoder_exact_parent_and_policy():
    """Actual tensor values, no model construction/forward/data/hash or repeat test."""
    frozen=unit.validate_frozen(unit.read(unit.OUT/'selection_before_test.json'))
    parent=torch.load(unit.PARENT,map_location='cpu',weights_only=False)['state_dict']
    records=[];errors=[]
    for u in frozen['units']:
        best=torch.load(u['best'],map_location='cpu',weights_only=False);unit.validate_checkpoint(best,u)
        part=u['parameter_policy'];unit.validate_parameter_policy(part,best['optimizer_groups'])
        excluded=set(part['encoder_layernorm']['parameter_names'])|set(part['head']['parameter_names'])
        keys=[k for k in parent if 'model.'+k not in excluded]
        frozen_names=set(part['frozen_encoder']['parameter_names'])
        assert frozen_names<=set('model.'+k for k in keys)
        assert sum(best['state_dict'][n].numel() for n in frozen_names)==39324407
        bad=[];elements=0
        for name in keys:
            a=parent[name];b=best['state_dict']['model.'+name];elements+=a.numel()
            if a.shape!=b.shape or a.dtype!=b.dtype or not torch.equal(a,b):bad.append(name)
        record=dict(task=u['task'],fold=u['fold'],tensor_count=len(keys),element_count=elements,
            frozen_parameter_elements=39324407,exact_equal=not bad,mismatched_names=bad)
        records.append(record);errors.extend((u['task'],u['fold'],n) for n in bad)
        for stage in ('head','joint'):
            recipe=u['config']['recipe']
            for g in best['optimizer_groups'][stage]:
                lr=recipe['head_lr'] if stage=='head' else recipe['joint_backbone_lr' if g['name']=='backbone' else 'joint_head_lr']
                assert g['lr']==lr and g['weight_decay']==recipe['weight_decay']
        del best
    path=unit.OUT/'final_review/frozen_encoder_comparison.json'
    assert not path.exists()
    unit.write(path,dict(status='PASS' if not errors else 'FAIL',parent=str(unit.PARENT),units=len(records),
        comparison='direct torch.equal saved tensors only; no hash or forward',records=records,total_compared_tensors=sum(r['tensor_count'] for r in records),total_compared_elements=sum(r['element_count'] for r in records)))
    assert len(records)==35 and not errors,errors
    damaged=copy.deepcopy(frozen);damaged['units'].pop()
    with pytest.raises(ValueError):unit.validate_frozen(damaged)
    damaged=copy.deepcopy(frozen);damaged['units'][0]['adaptation_policy']=None
    with pytest.raises(ValueError):unit.validate_frozen(damaged)
    damaged=copy.deepcopy(frozen);damaged['units'][0]['config']['loss_policy']=None
    with pytest.raises(ValueError):unit.validate_frozen(damaged)
