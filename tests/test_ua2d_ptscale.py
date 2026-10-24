"""One final, isolated PTSCALE CPU suite. No dataset construction or experiment replay."""
import copy
import time
from pathlib import Path
import numpy as np
import pytest
import torch
from src.dataset.ua2d_ptscale import resolve_slot,PreparedScaleMicrobatches
from src.training.glt_dual_runtime import OrderedSampleStream
from src.training import ua2d_ptscale as t

torch.set_num_threads(2)


def test_eligible_is_logical_slot_not_original_source_row():
    keys=np.array([bytes([i])*32 for i in range(4)],dtype='V32')
    rows=np.array([101,9,800,22]);eligible=np.array([3,0,2])
    assert resolve_slot(eligible,keys,rows,0)==(3,bytes([3])*32,22)
    assert resolve_slot(eligible,keys,rows,1)==(0,bytes([0])*32,101)
    with pytest.raises(ValueError):resolve_slot(np.array([101]),keys,rows,0)
    assert PreparedScaleMicrobatches.__len__(None)==20000*6


def test_rank_partition_and_stream_continuity_at_checkpoint_boundaries():
    stream=OrderedSampleStream(17,42);rebuilt=OrderedSampleStream(17,42)
    for boundary in (5000,10000,20000):
        positions=[]
        for micro in range(boundary*6-1,boundary*6+1):
            for rank in (0,1):
                start=(micro*2+rank)*84
                positions.extend(range(start,start+84))
                for pos in (start,start+83):assert stream.index_at(pos)==rebuilt.index_at(pos)
        assert positions==list(range((boundary*6-1)*168,(boundary*6+1)*168))
        assert (boundary*6*2)*84==boundary*1008
    absolute=5000*1008
    assert rebuilt.index_at(absolute)==OrderedSampleStream(17,42).index_at(absolute)
    assert [OrderedSampleStream(17,42).index_at(absolute+i) for i in range(17)]!=[OrderedSampleStream(17,42).index_at(i) for i in range(17)]


def test_cyclic_lr_and_global_warmup_are_distinct():
    for cycle in range(4):
        base=5000*cycle
        assert t.pt_learning_rate(base)==pytest.approx(2e-4/500)
        assert t.pt_learning_rate(base+499)==pytest.approx(2e-4)
        assert t.pt_learning_rate(base+4999)==pytest.approx(1e-6)
        assert t.global_weight_warmup(base)==(0. if cycle==0 else 1.)
    assert t.global_weight_warmup(500)==.5
    from src.modules.ua2d_galformer_pretrain import UA2DGalformerPretrainer,PretrainConfig
    model=UA2DGalformerPretrainer(PretrainConfig(fp_enabled=True))
    for u in (0,500,1000,4999,5000,10000,19999):
        model.completed_updates=u
        assert model.weights['contrast']==pytest.approx(.1*t.global_weight_warmup(u))
        assert model.weights['fp']==pytest.approx(.1*t.global_weight_warmup(u))
    with pytest.raises(ValueError):t.pt_learning_rate(20000)


def resume(step):
    return dict(schema=t.PT_SCHEMA,plan=t.PLAN,step=step,completed_updates=step,attempted_updates=step,
        schedule_position=step,schedule_cycle_position=step%5000,global_weight_warmup_position=step,
        next_global_position=step*1008,next_micro_index=step*6)


@pytest.mark.parametrize('step',[5000,10000,20000])
def test_true_step_resume_provenance(step,tmp_path):
    package=resume(step);p=tmp_path/'resume.pt';torch.save(package,p)
    assert t.validate_resume_metadata(torch.load(p,weights_only=False))==step
    for field in ('step','schedule_position','schedule_cycle_position','global_weight_warmup_position','next_global_position','next_micro_index'):
        bad=dict(package);bad[field]+=1
        with pytest.raises(ValueError):t.validate_resume_metadata(bad)
    from src.modules.ua2d_galformer_pretrain import PretrainConfig
    from dataclasses import asdict
    parent=dict(schema=t.PT_SCHEMA,step=step,plan=t.PLAN,revision=t.REVISION,
        config=asdict(PretrainConfig(fp_enabled=True)),execution=dict(plan=t.PLAN),eligible_rows=910756)
    assert t.validate_parent(parent,step) is parent
    with pytest.raises(ValueError):t.validate_parent(parent,10000 if step!=10000 else 5000)
    with pytest.raises(ValueError):t.validate_parent(dict(parent,eligible_rows=911391),step)


def test_complete_ft_strict_restore_has_real_parent():
    cfg=t.config('P10K','eea',0);model=t.MeanModel()
    p=dict(schema=t.SCHEMA,architecture=t.ARCHITECTURE,config=cfg,state_dict=model.state_dict())
    restored=t.restore(p,cfg)
    for k,v in model.state_dict().items():assert torch.equal(v,restored.state_dict()[k])
    wrong=dict(cfg,parent_step=5000)
    with pytest.raises(ValueError):t.restore(dict(p,config=wrong),wrong)
    missing=copy.copy(p);missing['state_dict']=dict(p['state_dict']);missing['state_dict'].pop(next(iter(missing['state_dict'])))
    with pytest.raises(ValueError):t.restore(missing,cfg)


def test_epoch_failure_charges_full_open_reservation():
    state=dict(epochs_started=0,validation_started=0,attempted_updates=0,completed_updates=0,completed_epochs=0)
    t.begin_epoch_reservation(state,11)
    state['attempted_updates']=2;state['completed_updates']=2
    charged=t.conservative_ft_counts(state)
    assert charged['known_actual_lower_bound']['attempted_updates']==2
    assert charged['charged']['attempted_updates']==11 and charged['charged']['validation_started']==1
    with pytest.raises(ValueError):t.close_epoch_reservation(state)
    state.update(attempted_updates=11,completed_updates=11,validation_started=1,completed_epochs=1)
    t.close_epoch_reservation(state)
    assert not t.conservative_ft_counts(state)['open_epoch']


def test_one_pt_trajectory_budget_read_prefetch_separate(tmp_path,monkeypatch):
    from scripts import run_ua2d_ptscale as script
    monkeypatch.setattr(script,'OUT',tmp_path)
    c=script.Controller.__new__(script.Controller)
    c.jobs=[];c.start=time.monotonic();c.started_unix=1.;c.cached_state=None;c.cached_budget=None;c.evidence_only=False
    for name,kind in (('prepare','prepare'),('pt','pt')):
        log=tmp_path/name;log.mkdir();c.jobs.append(dict(name=name,kind=kind,exit=0,log=str(log),reservation=dict(pt_updates=20000,pt_presentations=20160000) if kind=='pt' else {}))
    for rank in range(2):t.write(tmp_path/'pretrain'/f'rank{rank}_consumption.json',dict(attempted_updates=20000,completed_updates=20000,consumed_presentations=10080000,sample_presentations=10080000,prefetched_not_consumed=0))
    b=c.refresh()
    assert b['used']['pt_starts']==1 and b['used']['pt_updates']==20000 and b['used']['pt_presentations']==20160000
    assert b['used']['initialization_starts']==b['used']['ft_starts']==0
    assert not b['budget_violations']
    assert c.refresh()['time']==b['time']
    r=dict(attempted_updates=5,completed_updates=4,consumed_presentations=4*504,sample_presentations=5*504+84,prefetched_not_consumed=504+84)
    for rank in range(2):t.write(tmp_path/'pretrain'/f'rank{rank}_consumption.json',r)
    c.jobs[-1]['exit']=None;b=c.refresh()
    assert b['used']['pt_presentations']==5*1008 and b['known_actual_lower_bound']['pt_presentations']==4*1008
    assert b['active_remaining_reservation']['pt_presentations']==20160000-5*1008
    assert b['jobs'][-1]['progress']['read_prefetched_presentations']==2*(5*504+84)
    assert not b['budget_violations']


def test_freeze_all105_and_final_excludes_p5k(tmp_path,monkeypatch):
    monkeypatch.setattr(t,'OUT',tmp_path);monkeypatch.setattr(t,'SPLIT',tmp_path/'splits')
    for task in t.TASKS:t.write(tmp_path/'splits'/f'{task}.json',dict(folds=[dict(test_indices=[f]) for f in range(5)]))
    validation={a:dict(macro=.99 if a=='P5K' else .8,units={f'{a}/{task}/{f}':dict(config=t.config(a,task,f)) for task in t.TASKS for f in range(5)}) for a in t.ARMS}
    frozen=t.freeze_final(validation)
    assert frozen['FINAL']=='P10K' and len(frozen['units'])==105 and frozen['models']==list(t.ARMS)
    with pytest.raises(FileExistsError):t.freeze_final(validation)
    assert t.outcome(.011,.009)['performance_pass'] is False
    assert t.outcome(.010,.010)['performance_pass'] is True
    assert t.outcome(None,None)['performance_status']=='未完成／不可比较'


def test_optimizer_and_model_evidence_restore_preserves_true_step(tmp_path):
    model=torch.nn.Linear(2,1);opt=torch.optim.AdamW(model.parameters(),lr=.0002)
    package=dict(resume(10000),model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()))
    path=tmp_path/'evidence.pt';torch.save(package,path)
    restored=torch.nn.Linear(2,1);other=torch.optim.AdamW(restored.parameters(),lr=.5)
    evidence=t.restore_pt_evidence(restored,other,torch.load(path,weights_only=False))
    assert restored.completed_updates==10000 and evidence['next_global_position']==10080000
    assert other.param_groups[0]['lr']==.0002
    for k,v in model.state_dict().items():assert torch.equal(v,restored.state_dict()[k])
