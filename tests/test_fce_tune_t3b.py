"""One final CPU check for the geometry-weight intervention and reused reference."""
import copy
import json
import pytest
import torch
from scripts.run_fce_tune_t3b_unit import (
    ARMS,TASKS,pt_config_for,validate_pt_config,validate_pt_runtime,
    export_initial_state,bind_deployment,validate_parent,config_for,
)
from scripts.run_fce_tune_t3_unit import pt_config_for as reference_config
from scripts.run_fce_tune_t3b import LIMITS,check_budget,require_stage,pt_command
from scripts.evaluate_glt_pt_endpoint import t3b_validation_world
from scripts.select_fce_tune_t3b import fixed_formal_units
from scripts.finalize_fce_tune_t3 import execution_record,exit_index


def test_reused_reference_has_only_one_scientific_change():
    candidate=pt_config_for('T3B_G05');reference=reference_config('T3_REF')
    identity={'arm','initial_state','revision','cycle_root','t3_revision','t3b_revision'}
    difference={k for k in set(candidate)|set(reference) if candidate.get(k)!=reference.get(k)}
    assert difference-identity=={'loss_weights'}
    assert candidate['loss_weights']==[1.,.5,0.] and reference['loss_weights']==[1.,1.,0.]
    assert candidate['schedule_total_steps']==reference['schedule_total_steps']==20000
    assert candidate['warmup_steps']==2000 and candidate['max_optimizer_steps']==5000
    assert (candidate['expected_world_size'],candidate['microbatch'],candidate['accumulation'],candidate['global_batch'])==(2,84,6,1008)
    for task in TASKS:
        assert config_for('T3B_G05',task)['recipe']['arm']=='R0'


@pytest.mark.parametrize('field,value',[('loss_weights',[1.,1.,0.]),('schedule_total_steps',5000),('atom_mask_ratio',.15),('t3_revision','old')])
def test_reject_compound_or_wrong_revision(field,value):
    config=pt_config_for('T3B_G05');config[field]=value
    with pytest.raises(ValueError):validate_pt_config(config)
    with pytest.raises(ValueError):t3b_validation_world(config)


def test_two_rank_runtime_and_no_restarts():
    config=pt_config_for('T3B_G05')
    assert t3b_validation_world(config)==2
    validate_pt_runtime(config,2,6)
    for world,accum in ((4,3),(2,3),(1,12)):
        with pytest.raises(ValueError):validate_pt_runtime(config,world,accum)
    command=pt_command('T3B_G05')
    assert '--nproc_per_node=2' in command and '--max_restarts=0' in command
    assert command[command.index('--prep-workers')+1]=='8'


def test_complete_step0_toy_export_keeps_decoder_and_encoder(tmp_path):
    config=pt_config_for('T3B_G05')
    original={'step':0,'opt_mode':'GLT_FUSION_R1','model':{'encoder':torch.arange(6.),'decoder':torch.tensor([3.,-2.])}}
    exported=export_initial_state(original,config)
    for key,tensor in original['model'].items():
        assert torch.equal(exported['model'][key],tensor)
        assert exported['model'][key].data_ptr()!=tensor.data_ptr()
    assert exported['config']==config
    invalid=copy.deepcopy(original);invalid['step']=5000
    with pytest.raises(ValueError):export_initial_state(invalid,config)
    invalid=copy.deepcopy(original);invalid['model']['decoder'][0]=float('nan')
    with pytest.raises(ValueError):export_initial_state(invalid,config)


def test_parent_rejects_wrong_geometry_weight_or_source():
    config=pt_config_for('T3B_G05')
    structure=('fusion_mode','fusion_revision','use_md200','atom_ce_input','align_enabled','local_enabled','third_task','opt_mode','geometry_revision','geometry_mode','angle_value_enabled','readout','torsion_modules','torsion_mode','geometry_target','element_mask','input_schema')
    parent=bind_deployment({'step':5000,**{k:config[k] for k in structure}},config)
    assert validate_parent(parent,'T3B_G05')==config
    for field,value in [('loss_weights',[1.,1.,0.]),('step',2500),('common_initialization','wrong')]:
        bad=copy.deepcopy(parent);bad[field]=value
        with pytest.raises(ValueError):validate_parent(bad,'T3B_G05')
    bad=copy.deepcopy(parent);bad['training_config']['loss_weights']=[1.,1.,0.]
    with pytest.raises(ValueError):validate_parent(bad,'T3B_G05')


def test_only_candidate_is_new_formal_and_budget_is_not_doubled():
    units=[dict(arm='T3B_G05',task=t,fold=f) for t in TASKS for f in range(5)]
    assert len(fixed_formal_units(units))==35
    for invalid in (units[:-1],units[:-1]+[units[0]],units+[dict(arm='T3_REF',task='eea',fold=0)]):
        with pytest.raises(ValueError):fixed_formal_units(invalid)
    check_budget(LIMITS)
    assert LIMITS['ft_epochs']==2400 and LIMITS['ft_attempted_updates']==51450
    assert LIMITS['pt_attempted_updates']==5000 and LIMITS['formal_starts']==35
    for stage in ('FT','freeze','formal','CPU'):require_stage(stage,LIMITS)
    for stage,field in [('FT','pt_completed'),('FT','pt_validation_presentations'),('formal','ft_completed'),('CPU','formal_completed')]:
        bad=dict(LIMITS);bad[field]-=1
        with pytest.raises(ValueError):require_stage(stage,bad)
    bad=dict(LIMITS);bad['pt_starts']=2
    with pytest.raises(RuntimeError):check_budget(bad)


def test_delivery_keeps_raw_conflict_unknown_times_and_failed_helper(tmp_path):
    ft=tmp_path/'ft_candidate';ft.mkdir()
    raw={'status':'RUNNING','start':'2026-10-15T15:00:00+00:00','gpu':3}
    (ft/'execution.json').write_text(json.dumps(raw));(ft/'exit.txt').write_text('0')
    normalized=execution_record(ft)
    assert normalized['exit_code']==0 and normalized['raw_status']=='RUNNING'
    assert normalized['start'] is None and normalized['end'] is None
    assert normalized['metadata_conflicts'] and normalized['raw_execution']==raw
    assert json.loads((ft/'execution.json').read_text())==raw
    failed=tmp_path/'finalize';failed.mkdir()
    (failed/'execution.json').write_text(json.dumps({'start':'2026-10-15T16:00:00+00:00','end':'2026-10-15T16:01:00+00:00','exit_code':1}))
    (failed/'exit.txt').write_text('1')
    index=exit_index(tmp_path)
    assert index['exit_counts']=={'0':1,'1':1,'None':0}
    assert index['complete_interval_ft_formal_observed_peaks']['3']==0
    pending=tmp_path/'pending';pending.mkdir();(pending/'execution.json').write_text(json.dumps(raw))
    assert execution_record(pending)['exit_code'] is None
