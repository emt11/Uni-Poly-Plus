"""Final bounded CPU checks of recipe wiring and validation selection."""
import copy
import math
import pytest
import torch
from torch import nn
import scripts.finetune_mcl_ph as original
from src.training.fce_param import (ARMS,recipe_for,validate_recipe,periodic_tdl_trainability,
    periodic_tdl_optimizer,scheduler_for,configure_periodic_tdl_dropout)
from scripts.run_fce_param_unit import config_for,validate_config
from scripts.select_fce_param import choose


class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=nn.Sequential(nn.Linear(2,3),nn.Dropout(.7))
        self.head=nn.Sequential(nn.Linear(3,1),nn.Dropout(.1))


@pytest.mark.parametrize('task',['eea','egc','xc'])
def test_recipe_groups_defaults_and_trainability(task):
    model=Toy()
    for head_only in (True,False):
        periodic_tdl_trainability(model,head_only=head_only)
        old,e=original.periodic_tdl_optimizer(model,task,head_only=head_only)
        opt,actual=periodic_tdl_optimizer(model,task,head_only=head_only,recipe=recipe_for('R0',task))
        assert e==actual
        assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad}
        assert sum(len(g['params']) for g in opt.param_groups)==len({id(p) for g in opt.param_groups for p in g['params']})
        for arm in ARMS[1:]:
            opt,groups=periodic_tdl_optimizer(model,task,head_only=head_only,recipe=recipe_for(arm,task))
            expected=copy.deepcopy(e)
            if not head_only:
                for g in expected:
                    if g['name']=='head' and arm=='H_LR':g['lr']=3e-4
                    if g['name']=='backbone' and arm=='B_LR':g['lr']*=.3
            assert groups==expected
    configure_periodic_tdl_dropout(model,task)
    assert model.backbone[1].p==0
    assert model.head[1].p==original.PERIODIC_TDL_HEAD_DROPOUT[task]


@pytest.mark.parametrize('task',['eea','egc','xc'])
def test_scheduler_actual_epoch_convention(task):
    for arm in ARMS:
        recipe=recipe_for(arm,task);model=Toy()
        opt,_=periodic_tdl_optimizer(model,task,head_only=False,recipe=recipe)
        sched=scheduler_for(opt,'joint',recipe['joint_epochs'],recipe)
        used=[]
        for epoch in range(recipe['joint_epochs']):
            used.append([g['lr'] for g in opt.param_groups])
            # Scheduler is after epoch, as in the actual loop; no synthetic training.
            opt.step();sched.step()
        assert used[0]==[recipe['joint_backbone_lr'],recipe['joint_head_lr']]
        if arm=='S_COS':
            assert all(a[1]>b[1] for a,b in zip(used,used[1:]))
            assert used[-1][1]>1e-6
            assert all(math.isclose(g['lr'],1e-6,rel_tol=1e-12) for g in opt.param_groups)
        else:
            assert used[0]==used[10]
        head,_=periodic_tdl_optimizer(model,task,head_only=True,recipe=recipe)
        hs=scheduler_for(head,'head',10,recipe)
        for _ in range(10):head.step();hs.step()
        assert math.isclose(head.param_groups[0]['lr'],3e-5,rel_tol=1e-12)


def test_strict_recipe_and_metadata_binding():
    cfg=config_for('B_LR','xc');assert validate_config(cfg,'B_LR','xc')==cfg
    for key in ('joint_head_lr','weight_decay','joint_epochs','head_dropout','joint_backbone_lr'):
        bad=copy.deepcopy(cfg);bad['recipe'][key]+=1
        with pytest.raises(ValueError):validate_config(bad,'B_LR','xc')
        with pytest.raises(ValueError):validate_recipe(bad['recipe'],'xc')
    for key in ('parent','source','split','runtime_revision','feature_cache'):
        bad=copy.deepcopy(cfg);bad[key]='wrong'
        with pytest.raises(ValueError):validate_config(bad,'B_LR','xc')
    with pytest.raises(ValueError):periodic_tdl_optimizer(Toy(),'xc',head_only=False,profile='uniform_u1',recipe=cfg['recipe'])


def test_validation_selection_includes_negative_delta_and_exact_ties():
    assert choose(dict(R0=.9,H_LR=.8,B_LR=.8,S_COS=.8))=='H_LR'
    assert choose(dict(R0=.9,H_LR=.8,B_LR=.800000000001,S_COS=.8))=='B_LR'
    with pytest.raises(ValueError):choose(dict(H_LR=.8,B_LR=.8,S_COS=.8))
    with pytest.raises(ValueError):choose(dict(R0=.9,H_LR=float('nan'),B_LR=.8,S_COS=.8))
