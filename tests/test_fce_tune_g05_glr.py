"""One final CPU-only check of G05 parent/GLR recipe and bounded wiring."""
import copy
import pytest
from torch import nn
import scripts.run_fce_tune_g05_glr_unit as unit
from scripts.run_fce_tune_t3b_unit import pt_config_for,bind_deployment
from src.training.fce_param import t2_recipe_for,periodic_tdl_optimizer
from scripts.select_fce_tune_g05_glr import fixed_formal_units,validate_frozen,REFERENCE
from scripts.run_fce_tune_g05_glr import LIMITS,check_budget,require_stage


def parent_toy():
    cfg=pt_config_for('T3B_G05')
    keys=('fusion_mode','fusion_revision','use_md200','atom_ce_input','align_enabled','local_enabled','third_task','opt_mode','geometry_revision','geometry_mode','angle_value_enabled','readout','torsion_modules','torsion_mode','geometry_target','element_mask','input_schema')
    return bind_deployment(dict(step=5000,**{k:cfg[k] for k in keys}),cfg)


def test_g05_parent_is_fixed_and_wrong_pt_weights_rejected():
    package=parent_toy();unit.validate_parent(package,'G05_GLR')
    assert str(unit.parent('G05_GLR')).endswith('/t3b/pretrain/T3B_G05/deploy_05000.pt')
    for key,value in [('step',2500),('loss_weights',[1,1,0]),('arm','T3_REF'),('t3b_revision','wrong')]:
        bad=copy.deepcopy(package);bad[key]=value
        with pytest.raises(ValueError):unit.validate_parent(bad,'G05_GLR')
    bad=copy.deepcopy(package);bad['training_config']['loss_weights']=[1,1,0]
    with pytest.raises(ValueError):unit.validate_parent(bad,'G05_GLR')


class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.model=nn.Module()
        self.model.glt=nn.Sequential(nn.Linear(2,3),nn.LayerNorm(3))
        self.model.o8=nn.Linear(2,3);self.model.token_fusion=nn.Linear(3,3)
        self.model.norm2=nn.LayerNorm(3);self.model.norm3=nn.LayerNorm(3)
        self.model.predictor=nn.Linear(3,1)

@pytest.mark.parametrize('task',unit.TASKS)
def test_new_entry_consumes_existing_recipe_and_group_members(task,monkeypatch):
    cfg=unit.config_for('G05_GLR',task)
    assert cfg['arm']=='G05_GLR' and cfg['recipe']==t2_recipe_for('T2_G_LR',task)
    assert cfg['training_config']['arm']=='T3B_G05' and cfg['training_config']['loss_weights']==[1,.5,0]
    captured={}
    def capture(*args,**kwargs):captured.update(kwargs);return 'captured'
    monkeypatch.setattr(unit,'run_periodic_tdl_epochs',capture)
    m=Toy()
    assert unit.train_candidate(m,None,None,None,'cpu',cfg=cfg,task=task,scaler=None,progress=None,step_observer=None)=='captured'
    assert captured['recipe']==t2_recipe_for('T2_G_LR',task) and captured['task']==task
    optimizer,_=periodic_tdl_optimizer(m,task,head_only=False,recipe=captured['recipe'])
    groups={g['name']:{id(p) for p in g['params']} for g in optimizer.param_groups}
    glt={id(p) for p in m.model.glt.parameters()};head={id(p) for p in m.model.predictor.parameters()}
    assert groups['glt']==glt and groups['head']==head
    assert groups['backbone_other']=={id(p) for p in m.parameters()}-glt-head
    all_ids=[id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(all_ids)==len(set(all_ids))==len(list(m.parameters()))
    bad=copy.deepcopy(cfg);bad['recipe']=t2_recipe_for('T2_R0',task)
    with pytest.raises(ValueError):unit.train_candidate(m,None,None,None,'cpu',cfg=bad,task=task,scaler=None,progress=None,step_observer=None)


def test_fixed_35_candidate_and_35_reused_reference_freeze():
    units=[dict(arm='G05_GLR',task=t,fold=f,parent=str(unit.parent('G05_GLR')),config=unit.config_for('G05_GLR',t)) for t in unit.TASKS for f in range(5)]
    refs=[dict(arm='T3B_G05',task=t,fold=f) for t in unit.TASKS for f in range(5)]
    frozen=dict(frozen=True,validation_units=35,units=units,reference_units=refs,reference_root=str(REFERENCE),formal_units=fixed_formal_units(units))
    validate_frozen(frozen);assert len(frozen['formal_units'])==35
    for key,value in [('frozen',False),('validation_units',34),('reference_units',refs[:-1]),('units',units[:-1]),('formal_units',[]),('reference_root','old_T2')]:
        bad=copy.deepcopy(frozen);bad[key]=value
        with pytest.raises(ValueError):validate_frozen(bad)
    with pytest.raises(ValueError):fixed_formal_units(units[:-1]+[units[0]])


def test_zero_pt_and_ft_budget_stage_boundaries():
    check_budget(LIMITS);require_stage('CPU',LIMITS)
    assert LIMITS['pt_starts']==0 and LIMITS['pt_validation_starts']==0 and LIMITS['pt_endpoint_starts']==0
    assert LIMITS['ft_starts']==35 and LIMITS['ft_epochs']==2400 and LIMITS['ft_attempted_updates']==51450 and LIMITS['formal_starts']==35 and LIMITS['cpu_test_starts']==1
    for key in LIMITS:
        bad=dict(LIMITS);bad[key]+=1
        with pytest.raises(RuntimeError):check_budget(bad)
    for key in ('ft_completed','ft_epochs','ft_attempted_updates','ft_completed_updates'):
        bad=dict(LIMITS);bad[key]-=1
        with pytest.raises(ValueError):require_stage('formal',bad)
    bad=dict(LIMITS);bad['formal_completed']=34
    with pytest.raises(ValueError):require_stage('CPU',bad)
