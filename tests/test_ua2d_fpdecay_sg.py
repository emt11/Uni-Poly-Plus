"""Single final r2-SG correctness suite; no scientific training or baseline replay."""
import copy
from dataclasses import asdict
import time
import numpy as np
import pytest
import torch
from src.dataset.ua2d_fpdecay_sg import resolve_slot,micro_start,PreparedSingleMicrobatches
from src.training.glt_dual_runtime import OrderedSampleStream
from src.training import ua2d_fpdecay_sg as t

torch.set_num_threads(2)


def test_single_gpu_window_normalization_and_gradient():
    p=torch.tensor(2.,requires_grad=True);counts=[1,3,2,5,4,6]
    denominator={'a':torch.tensor(float(sum(counts))),'empty':torch.tensor(0.)};weights={'a':.3,'empty':.1}
    for count in counts:
        result=dict(sums={'a':p.square()*count,'empty':p*0})
        t.sg_objective(result,weights,denominator).backward()
    # Accumulating six microbatches gives a weighted window mean exactly once.
    assert p.grad.item()==pytest.approx(.3*4.)
    assert p.grad.item()!=pytest.approx(.3*4.*2)
    assert p.grad.item()!=pytest.approx(.3*4./6)
    with pytest.raises(ValueError):t.sg_objective(dict(sums={'bad':p}),weights,denominator)


def test_single_gpu_positions_and_logical_slot_mapping():
    assert PreparedSingleMicrobatches.__len__(None)==30000
    for u in (0,1,3999,4000,4999):
        positions=[pos for micro in range(6) for pos in range(micro_start(u*6+micro),micro_start(u*6+micro)+168)]
        assert positions==list(range(u*1008,(u+1)*1008))
    with pytest.raises(ValueError):micro_start(30000)
    keys=np.array([bytes([k])*32 for k in range(5)],dtype='V32');rows=np.array([500,101,7,400,88]);eligible=np.array([4,2,0])
    assert resolve_slot(eligible,keys,rows,0)==(4,bytes([4])*32,88)
    a,b=OrderedSampleStream(3,42),OrderedSampleStream(3,42)
    for pos in (0,167,168,1007,1008,4031999,4032000,5039999):
        assert resolve_slot(eligible,keys,rows,a.index_at(pos))==resolve_slot(eligible,keys,rows,b.index_at(pos))
    with pytest.raises(ValueError):resolve_slot(np.array([500]),keys,rows,0)


def test_fp_cosine_boundary_and_no_align_change():
    from src.modules.ua2d_galformer_pretrain import PretrainConfig,UA2DGalformerPretrainer
    for step in (1,1000,3999,4000):assert t.fp_decay_multiplier('FP_DECAY',step)==1.
    assert t.fp_decay_multiplier('FP_DECAY',4001)<1.
    assert t.fp_decay_multiplier('FP_DECAY',4500)==pytest.approx(.5)
    assert t.fp_decay_multiplier('FP_DECAY',5000)==0.
    assert all(t.fp_decay_multiplier('BASE',step)==1. for step in (1,4000,4001,4500,5000))
    model=UA2DGalformerPretrainer(PretrainConfig(fp_enabled=True))
    for u in (0,499,999,3999,4000,4499,4999):
        model.completed_updates=u;original=model.weights
        base=t.pt_weights(model,'BASE',u);decay=t.pt_weights(model,'FP_DECAY',u)
        assert base==original
        assert decay['fp']==pytest.approx(original['fp']*t.fp_decay_multiplier('FP_DECAY',u+1))
        assert {k:v for k,v in decay.items() if k!='fp'}=={k:v for k,v in original.items() if k!='fp'}
    assert t.pt_learning_rate(0)==pytest.approx(2e-4/500)
    assert t.pt_learning_rate(499)==pytest.approx(2e-4)
    assert t.pt_learning_rate(4999)==pytest.approx(1e-6)
    with pytest.raises(ValueError):t.fp_decay_multiplier('FP_DECAY',0)
    with pytest.raises(ValueError):t.fp_decay_multiplier('FP_DECAY',5001)


def test_common_initialization_and_fp_seed_are_identical():
    from src.modules.ua2d_galformer_pretrain import UA2DGalformerPretrainer,PretrainConfig
    cfg=PretrainConfig(fp_enabled=True)
    torch.manual_seed(42);first=UA2DGalformerPretrainer(cfg);fp={k:v.clone() for k,v in first.fp_head.state_dict().items()}
    t.load_pt_initialization(first,t.INITIALIZATION)
    for k,v in fp.items():assert torch.equal(v,first.fp_head.state_dict()[k])
    torch.manual_seed(42);second=UA2DGalformerPretrainer(cfg);t.load_pt_initialization(second,t.INITIALIZATION)
    for k,v in first.state_dict().items():assert torch.equal(v,second.state_dict()[k])
    common=torch.load(t.INITIALIZATION,map_location='cpu',weights_only=False)
    for k,v in common['state_dict'].items():assert torch.equal(v,first.state_dict()[k])
    heads=torch.load(t.INITIALIZATION.parent/'task_heads/V1.pt',map_location='cpu',weights_only=False)
    for k,v in heads['state_dict'].items():assert torch.equal(v,first.state_dict()[k])


def package(arm):
    return dict(schema=t.PT_SCHEMA,plan=t.PLAN,revision=t.REVISION,arm=arm,step=5000,completed_updates=5000,
        attempted_updates=5000,schedule_position=5000,next_global_position=5040000,next_micro_index=30000,execution=dict(world_size=1))


@pytest.mark.parametrize('arm',t.ARMS)
def test_resume_and_downstream_parent_identity(arm,tmp_path):
    from src.modules.ua2d_galformer_pretrain import PretrainConfig
    source=torch.nn.Linear(2,1);opt=torch.optim.AdamW(source.parameters(),lr=.0002)
    saved=dict(package(arm),model=copy.deepcopy(source.state_dict()),optimizer=opt.state_dict())
    path=tmp_path/'resume.pt';torch.save(saved,path)
    target=torch.nn.Linear(2,1);other=torch.optim.AdamW(target.parameters(),lr=.5)
    evidence=t.restore_pt_evidence(target,other,torch.load(path,weights_only=False),arm)
    assert evidence['step']==5000 and evidence['next_global_position']==5040000
    assert other.param_groups[0]['lr']==.0002
    for k,v in source.state_dict().items():assert torch.equal(v,target.state_dict()[k])
    for field in ('step','completed_updates','schedule_position','next_global_position','next_micro_index'):
        bad=copy.deepcopy(saved);bad[field]+=1
        with pytest.raises(ValueError):t.validate_resume_metadata(bad,arm)
    other_arm=next(a for a in t.ARMS if a!=arm)
    with pytest.raises(ValueError):t.validate_resume_metadata(saved,other_arm)
    parent=dict(package(arm),config=asdict(PretrainConfig(fp_enabled=True)),eligible_rows=910756,
        execution=dict(plan=t.PLAN,world_size=1,microbatch=168,accumulation=6,global_batch=1008))
    assert t.validate_parent(parent,arm) is parent
    with pytest.raises(ValueError):t.validate_parent(parent,other_arm)
    with pytest.raises(ValueError):t.validate_parent(dict(parent,execution=dict(parent['execution'],world_size=2)),arm)
    cfg=t.config(arm,'eea',0);model=t.MeanModel()
    best=dict(schema=t.SCHEMA,architecture=t.ARCHITECTURE,config=cfg,state_dict=model.state_dict())
    rebuilt=t.restore(best,cfg)
    for k,v in model.state_dict().items():assert torch.equal(v,rebuilt.state_dict()[k])
    with pytest.raises(ValueError):t.restore(dict(best,config=dict(cfg,parent_step=10000)),cfg)


def test_profile_exactly_reused_and_epoch_failure_accounting():
    from src.training import ua2d_ptscale as reference
    assert t.PROFILE==reference.PROFILE and t.ARCHITECTURE==reference.ARCHITECTURE
    assert t.PROFILE is not reference.PROFILE and t.ARCHITECTURE is not reference.ARCHITECTURE
    state=dict(epochs_started=0,validation_started=0,attempted_updates=0,completed_updates=0,completed_epochs=0)
    t.begin_epoch_reservation(state,11);state.update(attempted_updates=2,completed_updates=2)
    charged=t.conservative_ft_counts(state)
    assert charged['charged']['attempted_updates']==11 and charged['known_actual_lower_bound']['attempted_updates']==2
    assert charged['charged']['validation_started']==1
    with pytest.raises(ValueError):t.close_epoch_reservation(state)
    state.update(attempted_updates=11,completed_updates=11,validation_started=1,completed_epochs=1);t.close_epoch_reservation(state)
    assert not t.conservative_ft_counts(state)['open_epoch']


def test_two_serial_pt_budget_counts_no_preparation_or_double_charge(tmp_path,monkeypatch):
    from scripts import run_ua2d_fpdecay_sg as script
    monkeypatch.setattr(script,'OUT',tmp_path)
    c=script.Controller.__new__(script.Controller);c.start=time.monotonic();c.started_unix=1.;c.cached_state=None;c.cached_budget=None;c.evidence_only=False;c.jobs=[]
    for arm in t.ARMS:
        log=tmp_path/arm;log.mkdir()
        c.jobs.append(dict(name=arm,kind='pt',arm=arm,exit=0,log=str(log),reservation=dict(pt_updates=5000,pt_presentations=5040000)))
        t.write(tmp_path/'pretrain'/arm/'consumption.json',dict(status='COMPLETE',attempted_updates=5000,completed_updates=5000,
            consumed_presentations=5040000,sample_presentations=5040000,prefetched_not_consumed=0))
    b=c.refresh()
    assert b['used']['pt_starts']==2 and b['used']['pt_updates']==10000 and b['used']['pt_presentations']==10080000
    assert b['used']['prepare_starts']==b['used']['initialization_starts']==0
    assert not b['budget_violations'] and c.refresh()['time']==b['time']
    c.jobs[-1]['exit']=None
    t.write(tmp_path/'pretrain'/'FP_DECAY'/'consumption.json',dict(attempted_updates=10,completed_updates=9,
        consumed_presentations=9*1008,sample_presentations=11*1008,prefetched_not_consumed=2*1008))
    b=c.refresh()
    assert b['used']['pt_updates']==5010 and b['used']['pt_presentations']==5010*1008
    assert b['known_actual_lower_bound']['pt_presentations']==5009*1008
    assert b['active_remaining_reservation']['pt_presentations']==4990*1008
    assert b['jobs'][-1]['progress']['read_prefetched_presentations']==11*1008
    assert not b['budget_violations']


def test_fixed_candidate_freeze_all70_no_validation_gate(tmp_path,monkeypatch):
    monkeypatch.setattr(t,'OUT',tmp_path);monkeypatch.setattr(t,'SPLIT',tmp_path/'splits')
    for task in t.TASKS:t.write(tmp_path/'splits'/f'{task}.json',dict(folds=[dict(test_indices=[f]) for f in range(5)]))
    validation={a:dict(macro=.9 if a=='BASE' else -.1,units={f'{a}/{task}/{f}':dict(config=t.config(a,task,f)) for task in t.TASKS for f in range(5)}) for a in t.ARMS}
    frozen=t.freeze_final(validation)
    assert frozen['FINAL']=='FP_DECAY' and len(frozen['units'])==70 and frozen['models']==list(t.ARMS)
    with pytest.raises(FileExistsError):t.freeze_final(validation)
    incomplete=dict(validation);incomplete.pop('BASE')
    with pytest.raises(ValueError):t.nominate(incomplete)
    assert t.outcome(.010,.010)['performance_pass'] is True
    assert t.outcome(.0099999,.020)['performance_pass'] is False
    assert t.outcome(.020,.0099999)['performance_pass'] is False
    assert t.outcome(None,None)['performance_status']=='未完成／不可比较'
