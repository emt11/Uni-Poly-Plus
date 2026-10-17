"""Only the new T1 wiring, local selection and hard limits; CPU toy objects."""
import copy
import pytest
from torch import nn
from src.training.fce_param import (T1_ARMS,TASKS,t1_recipe_for,recipe_for,validate_recipe,
    periodic_tdl_optimizer,periodic_tdl_trainability,configure_periodic_tdl_dropout)
from scripts.run_fce_tune_unit import config_for,validate_config
from scripts.select_fce_tune import choose,select_folds,formal_units
from scripts.run_fce_tune import check_budget,LIMITS

class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=nn.Sequential(nn.Linear(2,3),nn.Dropout(.7))
        self.head=nn.Sequential(nn.Linear(3,1),nn.Dropout(.5))

@pytest.mark.parametrize('task',TASKS)
def test_t1_actual_regularization_and_groups(task):
    m=Toy();base=recipe_for('R0',task)
    for arm in T1_ARMS:
        r=t1_recipe_for(arm,task);configure_periodic_tdl_dropout(m,task,recipe=r)
        assert m.backbone[1].p==0
        assert m.head[1].p==base['head_dropout']+(.1 if arm in ('T1_D','T1_DW') else 0)
        for head_only in (True,False):
            periodic_tdl_trainability(m,head_only=head_only)
            opt,e=periodic_tdl_optimizer(m,task,head_only=head_only,recipe=r)
            ids=[id(p) for g in opt.param_groups for p in g['params']]
            assert len(ids)==len(set(ids))
            assert set(ids)=={id(p) for p in m.parameters() if p.requires_grad}
            for g in e:
                assert g['weight_decay']==base['weight_decay']+(.02 if g['name']=='backbone' and arm in ('T1_W','T1_DW') else 0)
                assert g['lr']==(3e-4 if head_only else base['joint_head_lr'] if g['name']=='head' else base['joint_backbone_lr'])
            if arm=='T1_R0':
                _,old=periodic_tdl_optimizer(m,task,head_only=head_only,recipe=base)
                assert e==old

def test_strict_recipe_and_config():
    for arm in T1_ARMS:
        cfg=config_for(arm,'xc');assert validate_config(cfg,arm,'xc')==cfg
        for key in ('head_dropout','head_weight_decay','joint_backbone_weight_decay','joint_epochs'):
            bad=copy.deepcopy(cfg);bad['recipe'][key]+=1
            with pytest.raises(ValueError):validate_recipe(bad['recipe'],'xc')
            with pytest.raises(ValueError):validate_config(bad,arm,'xc')
        with pytest.raises(ValueError):periodic_tdl_optimizer(Toy(),'xc',head_only=False,profile='uniform_u1',recipe=cfg['recipe'])
        with pytest.raises(ValueError):configure_periodic_tdl_dropout(Toy(),'xc',profile='uniform_u1',recipe=cfg['recipe'])

def test_local_rmse_selection_ties_fallback_and_deduplication():
    rows=[dict(arm=a,task=t,fold=f,rmse=1 if a=='T1_R0' else 2) for a in T1_ARMS for t in TASKS for f in range(5)]
    s=select_folds(rows);assert all(x['arm']=='T1_R0' for x in s);assert len(formal_units(s))==35
    # Other folds may prefer D; fold0 remains R0 and no global mean enters selection.
    for row in rows:
        if row['fold']>0 and row['arm']=='T1_D':row['rmse']=.1
    s=select_folds(rows);assert all(x['arm']=='T1_R0' for x in s if x['fold']==0)
    assert len(formal_units(s))==63
    assert choose(dict.fromkeys(T1_ARMS,1))=='T1_R0'
    assert choose(dict(T1_R0=2,T1_D=1,T1_W=1,T1_DW=1))=='T1_D'
    assert choose(dict(T1_R0=1,T1_D=1-1e-12,T1_W=2,T1_DW=2))=='T1_D'
    with pytest.raises(ValueError):choose(dict(T1_R0=float('nan'),T1_D=1,T1_W=2,T1_DW=3))
    with pytest.raises(ValueError):select_folds(rows[:-1])
    with pytest.raises(ValueError):formal_units(s[:-1])

def test_hard_budget_boundaries():
    check_budget(LIMITS)
    for key in LIMITS:
        c=dict(LIMITS);c[key]+=1
        with pytest.raises(RuntimeError):check_budget(c)
