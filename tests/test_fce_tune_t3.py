"""Single final CPU check: T3 wiring and budget without science execution."""
import copy
import math
import pytest
import torch
from scripts.run_fce_tune_t3_unit import (ARMS,TASKS,COMMON,pt_config_for,validate_pt_config,
    validate_pt_runtime,export_initial_state,bind_deployment,validate_parent,config_for)
from scripts.evaluate_glt_pt_endpoint import t3_validation_world,process_validation_positions
from scripts.run_fce_tune_t3 import LIMITS,check_budget,require_stage,pt_command
from scripts.select_fce_tune_t3 import fixed_formal_units
from src.training.glt_dual_runtime import scheduled_lr


def test_only_pt_schedule_scientific_change_and_lr_index():
    ref=pt_config_for('T3_REF');cos=pt_config_for('T3_COS5K')
    assert {k for k in ref if ref[k]!=cos[k]}=={'arm','initial_state','schedule_total_steps'}
    assert ref['common_initialization']==cos['common_initialization']==str(COMMON)
    for cfg in (ref,cos):
        assert (cfg['expected_world_size'],cfg['microbatch'],cfg['accumulation'],cfg['global_batch'])==(2,84,6,1008)
        assert cfg['max_optimizer_steps']==5000 and cfg['warmup_steps']==2000
        assert cfg['lr']==2e-4 and cfg['end_lr']==1e-9
        opts={k:cfg[k] for k in ('lr','warmup_steps','schedule_total_steps','end_lr')}
        assert math.isclose(scheduled_lr(0,**opts),1e-7)
        assert math.isclose(scheduled_lr(1999,**opts),2e-4)
        assert scheduled_lr(4999,**opts)>cfg['end_lr']
    opts=lambda cfg:{k:cfg[k] for k in ('lr','warmup_steps','schedule_total_steps','end_lr')}
    assert math.isclose(scheduled_lr(4999,**opts(ref)),1.866113e-4,rel_tol=1e-6)
    assert math.isclose(scheduled_lr(4999,**opts(cos)),1.054830862e-9,rel_tol=1e-6)
    for task in TASKS:
        assert config_for(ARMS[0],task)['recipe']==config_for(ARMS[1],task)['recipe']
        assert config_for(ARMS[0],task)['recipe']['arm']=='R0'


def test_step0_toy_export_config_binding_and_all_weights_preserved(tmp_path):
    original=dict(step=0,opt_mode='GLT_FUSION_R1',model={'encoder':torch.arange(6,dtype=torch.float32).reshape(2,3),'decoder':torch.tensor([.25,-.5])})
    for arm in ARMS:
        cfg=pt_config_for(arm);package=export_initial_state(original,cfg)
        file=tmp_path/(arm+'.pt');torch.save(package,file);restored=torch.load(file,weights_only=False)
        assert restored['config']==cfg and restored['arm']==arm and restored['step']==0
        for key,value in original['model'].items():
            assert torch.equal(restored['model'][key],value)
            assert package['model'][key].data_ptr()!=value.data_ptr()
    bad=copy.deepcopy(original);bad['step']=5000
    with pytest.raises(ValueError):export_initial_state(bad,pt_config_for(ARMS[0]))
    bad=copy.deepcopy(original);bad['model']['encoder'][0,0]=float('nan')
    with pytest.raises(ValueError):export_initial_state(bad,pt_config_for(ARMS[0]))

@pytest.mark.parametrize('arm',ARMS)
def test_two_rank_validation_wiring_and_forbidden_configs(arm):
    cfg=pt_config_for(arm);validate_pt_runtime(cfg,2,6)
    assert t3_validation_world(cfg)==2
    for world,accum in ((4,3),(2,3),(1,12)):
        with pytest.raises(ValueError):validate_pt_runtime(cfg,world,accum)
    for key,value in [('schedule_total_steps',6000),('expected_world_size',4),('atom_mask_ratio',.15),('reference_revision','old_S5')]:
        bad=copy.deepcopy(cfg);bad[key]=value
        with pytest.raises(ValueError):validate_pt_config(bad)
        with pytest.raises(ValueError):t3_validation_world(bad)
    seen=[]
    for rank in (0,1):
        indices=[p for start in range(0,1024,168) for p in process_validation_positions(start,rank,2)]
        assert len(indices)==512;seen+=indices
    assert sorted(seen)==list(range(1024))
    command=pt_command(arm)
    assert '--nproc_per_node=2' in command and '--max_restarts=0' in command
    assert command[command.index('--prep-workers')+1]=='8'
    assert '--lmdb-readahead' in command

@pytest.mark.parametrize('arm',ARMS)
def test_parent_arm_pt_config_and_step_binding(arm):
    cfg=pt_config_for(arm)
    keys=('fusion_mode','fusion_revision','use_md200','atom_ce_input','align_enabled','local_enabled','third_task','opt_mode','geometry_revision','geometry_mode','angle_value_enabled','readout','torsion_modules','torsion_mode','geometry_target','element_mask','input_schema')
    package=bind_deployment(dict(step=5000,**{k:cfg[k] for k in keys}),cfg)
    assert validate_parent(package,arm)==cfg
    other=next(a for a in ARMS if a!=arm)
    with pytest.raises(ValueError):validate_parent(package,other)
    for key,value in [('step',2500),('schedule_total_steps',123),('torsion_modules',True),('common_initialization','wrong')]:
        bad=copy.deepcopy(package);bad[key]=value
        with pytest.raises(ValueError):validate_parent(bad,arm)
    bad=copy.deepcopy(package);bad['training_config']['warmup_steps']=500
    with pytest.raises(ValueError):validate_parent(bad,arm)


def test_fixed_matrix_stage_and_hard_budget_boundaries():
    units=[dict(arm=a,task=t,fold=f) for a in ARMS for t in TASKS for f in range(5)]
    assert len(fixed_formal_units(units))==70
    for bad in (units[:-1],units[:-1]+[units[0]]):
        with pytest.raises(ValueError):fixed_formal_units(bad)
    check_budget(LIMITS)
    for stage in ('FT','freeze','formal','CPU'):require_stage(stage,LIMITS)
    for key in LIMITS:
        bad=dict(LIMITS);bad[key]+=1
        with pytest.raises(RuntimeError):check_budget(bad)
    for stage,key in [('FT','pt_completed_updates'),('FT','pt_validation_presentations'),('formal','ft_completed'),('CPU','formal_completed')]:
        bad=dict(LIMITS);bad[key]-=1
        with pytest.raises(ValueError):require_stage(stage,bad)
