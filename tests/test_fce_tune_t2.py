"""One final CPU-only check of T2 group boundaries and fixed comparisons."""
import copy
import math
import pytest
import torch
from torch import nn
from src.training.fce_param import (T2_ARMS,TASKS,t2_recipe_for,recipe_for,validate_recipe,
    periodic_tdl_optimizer,periodic_tdl_trainability,configure_periodic_tdl_dropout,scheduler_for)
from scripts.run_fce_tune_t2_unit import config_for,validate_config
from scripts.select_fce_tune_t2 import fixed_formal_units
from scripts.run_fce_tune_t2 import check_budget,LIMITS

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.glt=nn.Sequential(nn.Linear(3,3),nn.LayerNorm(3),nn.Dropout(.7))
        self.o8=nn.Linear(3,3)
        self.token_fusion=nn.Linear(3,3)
        self.norm2=nn.LayerNorm(3);self.norm3=nn.LayerNorm(3)
        self.predictor=nn.Sequential(nn.Linear(3,4),nn.GELU(),nn.Dropout(.9),nn.Linear(4,1))

class Toy(nn.Module):
    def __init__(self):super().__init__();self.model=Encoder()

@pytest.mark.parametrize('task',TASKS)
def test_actual_groups_and_original_regularization(task):
    m=Toy();base=recipe_for('R0',task)
    glt_ids={id(p) for p in m.model.glt.parameters()};head_ids={id(p) for p in m.model.predictor.parameters()}
    for arm in T2_ARMS:
        r=t2_recipe_for(arm,task);configure_periodic_tdl_dropout(m,task,recipe=r)
        assert m.model.predictor[2].p==base['head_dropout']
        assert m.model.glt[2].p==0
        for head_only in (True,False):
            periodic_tdl_trainability(m,head_only=head_only)
            opt,e=periodic_tdl_optimizer(m,task,head_only=head_only,recipe=r)
            assert [g['name'] for g in opt.param_groups]==(['head'] if head_only else ['glt','backbone_other','head'])
            ids=[id(p) for g in opt.param_groups for p in g['params']]
            assert len(ids)==len(set(ids))
            assert set(ids)=={id(p) for p in m.parameters() if p.requires_grad}
            for g in opt.param_groups:
                assert g['weight_decay']==base['weight_decay']
                members={id(p) for p in g['params']}
                if g['name']=='glt':assert members==glt_ids
                elif g['name']=='head':assert members==head_ids
                else:assert members=={id(p) for p in m.parameters()}-glt_ids-head_ids
                expected=(3e-4 if head_only else 1e-3 if g['name']=='head' else base['joint_backbone_lr']*(.3 if g['name']=='glt' and arm=='T2_G_LR' else 1))
                assert g['lr']==expected
        difference={k for k in r if r[k]!=t2_recipe_for('T2_R0',task)[k]}
        assert difference==(set() if arm=='T2_R0' else {'arm','joint_glt_lr'})

@pytest.mark.parametrize('task',['eea','egc','xc'])
def test_r0_group_split_preserves_adamw_update_and_schedule(task):
    old=Toy().double();new=copy.deepcopy(old)
    original,_=periodic_tdl_optimizer(old,task,head_only=False,recipe=recipe_for('R0',task))
    grouped,_=periodic_tdl_optimizer(new,task,head_only=False,recipe=t2_recipe_for('T2_R0',task))
    for n,(a,b) in enumerate(zip(old.parameters(),new.parameters())):
        grad=torch.full_like(a,.01*(n+1));a.grad=grad.clone();b.grad=grad.clone()
    original.step();grouped.step()
    for a,b in zip(old.parameters(),new.parameters()):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    a=scheduler_for(original,'joint',60,recipe_for('R0',task));b=scheduler_for(grouped,'joint',60,t2_recipe_for('T2_R0',task))
    for _ in range(12):
        a.step();b.step()
        assert [g['lr'] for g in grouped.param_groups]==[original.param_groups[0]['lr']]*2+[original.param_groups[1]['lr']]
    candidate,_=periodic_tdl_optimizer(Toy(),task,head_only=False,recipe=t2_recipe_for('T2_G_LR',task))
    schedule=scheduler_for(candidate,'joint',60,t2_recipe_for('T2_G_LR',task))
    for _ in range(5):schedule.step()
    for g,initial in zip(candidate.param_groups,[t2_recipe_for('T2_G_LR',task)['joint_glt_lr'],recipe_for('R0',task)['joint_backbone_lr'],1e-3]):
        assert math.isclose(g['lr'],1e-6+(initial-1e-6)*.5,rel_tol=1e-12)

def test_reject_mismatched_recipe_and_group_boundary():
    for arm in T2_ARMS:
        cfg=config_for(arm,'xc');assert validate_config(cfg,arm,'xc')==cfg
        for key in ('joint_glt_lr','head_dropout','weight_decay','joint_epochs'):
            bad=copy.deepcopy(cfg);bad['recipe'][key]+=1
            with pytest.raises(ValueError):validate_recipe(bad['recipe'],'xc')
            with pytest.raises(ValueError):validate_config(bad,arm,'xc')
        with pytest.raises(ValueError):periodic_tdl_optimizer(Toy(),'xc',head_only=False,profile='uniform_u1',recipe=cfg['recipe'])
    m=Toy();m.model.glt=m.model.predictor
    with pytest.raises(ValueError):periodic_tdl_optimizer(m,'eea',head_only=False,recipe=t2_recipe_for('T2_R0','eea'))

def test_fixed_two_arm_freeze_and_budget():
    units=[dict(arm=a,task=t,fold=f) for a in T2_ARMS for t in TASKS for f in range(5)]
    assert fixed_formal_units(units)==[[u['arm'],u['task'],u['fold']] for u in units]
    with pytest.raises(ValueError):fixed_formal_units(units[:-1])
    with pytest.raises(ValueError):fixed_formal_units(units[:-1]+[units[0]])
    bad=copy.deepcopy(units);bad[0]['arm']='T2_TUNED'
    with pytest.raises(ValueError):fixed_formal_units(bad)
    check_budget(LIMITS)
    assert LIMITS['ft_starts']==70 and LIMITS['ft_epochs']==4800 and LIMITS['ft_attempted_updates']==102900 and LIMITS['cpu_test_starts']==1
    for key in LIMITS:
        bad=dict(LIMITS);bad[key]+=1
        with pytest.raises(RuntimeError):check_budget(bad)
