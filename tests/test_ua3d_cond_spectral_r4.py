"""Sole final CPU suite; never run before complete formal TEST."""
import json
import numpy as np
import torch
from src.modules import ua3d_cond_spectral_r4 as m
from src.training import ua3d_cond_spectral_r4 as t

def test_formal_completion_and_budget():
    budget=t.read(t.OUT/'budget.json')
    assert budget['test_complete']==210 and budget['test_rows']==35250
    assert budget['pt_completed']==0 and budget['available_source_pt_completed']==10000 and budget['ft_complete']==210 and budget['ft_completed_updates']==336000
    frozen=t.read(t.OUT/'selection/final_frozen.json')
    assert frozen['frozen'] and len(frozen['units'])==210
    for arm in m.ARMS:
        met=t.aggregate(arm,True)
        assert len(met['units'])==35
        for task in m.TASKS:
            for fold in range(5):
                path=t.unit_path(arm,task,fold,True)
                with np.load(path/'test_predictions.npz',allow_pickle=False) as p:
                    rows=p['indices'];keys=p['sample_keys']
                    assert len(rows)==len(set(rows.tolist()))==len(keys)
                    assert rows.tolist()==sorted(t.read(t.SPLIT/f'{task}.json')['folds'][fold]['test_indices'])
                    assert np.isfinite(p['y_true']).all() and np.isfinite(p['y_pred']).all()

def fixture_surface():
    # Orthonormal eigenbasis for a small discrete heat operator, one repeated root.
    torch.manual_seed(9);Q,_=torch.linalg.qr(torch.randn(6,6));lam=torch.tensor([0.,.2,.2,.7,1.2,2.])
    F=torch.randn(6,16);w=torch.rand(6);C=Q.T@F;a=Q.T@w
    return Q,F,w,dict(lam=lam[None],C=C[None],a=a[None],A=torch.tensor([6.]),Ac=w.sum()[None],stat=torch.randn(1,16),mode_mask=torch.ones(1,6,dtype=torch.bool),mask=torch.ones(1))

def test_compact_heat_and_invariance():
    Q,F,w,s=fixture_surface();model=m.Surface('S_COND_GATE').eval();z=[torch.randn(1,256) for _ in range(3)];avail=torch.ones(1,3,dtype=torch.bool)
    with torch.no_grad():
        context=torch.cat((*[n(v) for n,v in zip(model.norms,z)],avail.float()),-1)
        times=.0625+(64.-.0625)*torch.sigmoid(model.beta+model.condition(context))
        B=torch.exp(-s['lam'][:,:,None]*times[:,None,:])*model.W(s['C'])
        explicit=torch.stack([Q@torch.diag(torch.exp(-s['lam'][0]*times[0,c]))@Q.T@(F@model.W.weight[c]) for c in range(64)],-1)
        assert torch.allclose((Q@B[0]),explicit,atol=2e-5,rtol=2e-5)
        assert torch.allclose((s['a'][:,:,None]*B).sum(1)/s['Ac'][:,None],(explicit*w[:,None]).sum(0)[None]/w.sum(),atol=2e-5,rtol=2e-5)
        out=model(z,avail,s)
        rot=torch.eye(6);angle=.7;rot[1:3,1:3]=torch.tensor([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        rot[4,4]=-1
        other=dict(s,C=(rot.T@s['C'][0])[None],a=(rot.T@s['a'][0])[None])
        assert torch.allclose(out,model(z,avail,other),atol=2e-5,rtol=2e-5)

def test_time_gradient_and_missing_surface():
    _,_,_,s=fixture_surface();model=m.Surface('S_COND_GATE').train();z=[torch.randn(1,256,requires_grad=True) for _ in range(3)];avail=torch.tensor([[True,False,True]])
    # Output layer starts zero by contract; nonzero weights here test reachable
    # conditioning -> BASE gradient after training has begun.
    with torch.no_grad():model.condition[-1].weight.normal_(0,.01)
    loss=model(z,avail,s).square().sum();loss.backward()
    assert model.time_tensor.grad is not None and torch.isfinite(model.time_tensor.grad).all()
    assert z[0].grad is not None and z[0].grad.abs().sum()>0
    assert z[1].grad.abs().sum()==0
    s['mask'].zero_();s['C'].fill_(float('nan'));s['a'].fill_(float('nan'));s['stat'].fill_(float('nan'))
    assert torch.equal(model(z,avail,s),torch.zeros(1,256))

def test_production_metrics():
    result=t.metrics(np.array([1.,2.,3.]),np.array([1.,2.,3.]))
    assert result['r2']==1 and result['rmse']==0

def test_memory_parser_pss_vs_shared_rss_and_environment():
    fields=t.parse_kb_fields('Rss: 200000000 kB\nPss: 100 kB\nSwapPss: 20 kB\nPss_Anon: 70 kB\nPss_File: 30 kB\nPss_Shmem: 0 kB\n')
    assert fields['Rss']>160*1024**3
    assert fields['Pss']+fields['SwapPss']==120*1024
    env=dict(mem_available=64*1024**3,**{'memory.current':220*1024**3,'memory.max':'max'})
    assert t.memory_decision(fields['Pss']+fields['SwapPss'],env) is None
    assert t.memory_decision(161*1024**3,env).startswith('MEMORY_STOP')
    assert t.memory_decision(None,env,False).startswith('TELEMETRY_ERROR')
    assert t.memory_decision(1,dict(env,mem_available=31*1024**3)).startswith('ENVIRONMENT_STOP')

def test_breach_evidence_persisted_before_stop(tmp_path):
    sample=dict(full_pss=True,coverage_complete=True,time_unix=1.,pids=[dict(pid=9,start_ticks=42,fields=dict(Pss=161*1024**3,SwapPss=0))],decision='MEMORY_STOP: fixture',pss_swap_bytes=161*1024**3)
    t.persist_memory_sample(tmp_path,sample)
    assert json.loads((tmp_path/'memory_breach.json').read_text())==sample
    assert json.loads((tmp_path/'memory_latest.json').read_text())==sample
    assert json.loads((tmp_path/'memory.jsonl').read_text().splitlines()[0])==sample
    t.persist_memory_sample(tmp_path,dict(sample,time_unix=2.))
    assert json.loads((tmp_path/'memory_breach.json').read_text())==sample

def test_bounded_esrch_transition_and_terminal_states():
    import errno
    entry=dict(pid=7,start_ticks=40,state='R');now=[0.];reads=[]
    def tick():return now[0]
    def sleep(seconds):now[0]+=seconds
    for state in ('Z','X','x'):
        now[0]=0.;identities=iter([entry,entry,dict(entry,state=state)])
        def identity(pid):return next(identities)
        def rollup(pid):reads.append(pid);raise ProcessLookupError(errno.ESRCH,'transition')
        result=t.observe_pid_memory(entry,rollup,identity,lambda:False,tick,sleep)
        assert result['status']=='EXITED' and result['reason']=='terminal_state_'+state
        assert len(result['attempts'])==1 and result['attempts'][0]['errno']==errno.ESRCH
        assert now[0]<=.250

def test_pid_reuse_event_and_persistent_unreadable():
    import errno,pytest
    entry=dict(pid=7,start_ticks=40,state='R');now=[0.];calls=[]
    def clock():return now[0]
    def sleep(seconds):now[0]+=seconds
    def unreadable(pid):calls.append(pid);raise ProcessLookupError(errno.ESRCH,'still live')
    with pytest.raises(t.PSSReadError) as error:t.observe_pid_memory(entry,unreadable,lambda pid:entry,lambda:False,clock,sleep)
    assert len(calls)==3 and now[0]<=.250
    assert error.value.evidence['before']==entry and error.value.evidence['after']==entry
    assert all(x['errno']==errno.ESRCH for x in error.value.evidence['attempts'])
    assert t.observe_pid_memory(entry,unreadable,lambda pid:dict(entry,start_ticks=41),lambda:False)['reason']=='PID_reused'
    assert t.observe_pid_memory(entry,unreadable,lambda pid:entry,lambda:True)['reason']=='pidfd_exit_event'
    with pytest.raises(t.PSSReadError):t.observe_pid_memory(entry,lambda pid:'Rss: 0 kB',lambda pid:entry,lambda:False)

def test_non_direct_descendants_never_reaped():
    import ast
    from pathlib import Path
    tree=ast.parse(Path(t.__file__).read_text())
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('waitpid','waitid') for n in ast.walk(tree))
    assert t.identity_status(dict(start_ticks=1),dict(start_ticks=2,state='R'))=='PID_reused'

def test_input_whitelist_schema_and_no_interrupted_best():
    import pytest
    assert t.allowed_input_path(t.INPUT_PT,'pt','eps',0)==t.INPUT_PT
    assert t.allowed_input_path(t.head_path('eps',0),'head','eps',0)==t.head_path('eps',0)
    with pytest.raises(ValueError):t.allowed_input_path(t.P3/'finetune/C_FRESH/eps/fold0/best.pt','pt','eps',0)
    with pytest.raises(ValueError):t.allowed_input_path(t.PARENT/'initialization/common_step0.pt','pt','eps',0)
    with pytest.raises(ValueError):m.initialize_ft(dict(schema='ua3d-cond-spectral-best-r3'),{},'C_FRESH',0)
    assert m.FT_SCHEMA=='ua3d-cond-spectral-best-r4' and m.INPUT_PT_SCHEMA=='ua3d-cond-spectral-pretrain-r3'

def test_stage_clock_and_three_generation_budget():
    clock=t.read(t.OUT/'authorization/clock.json');parent=t.read(t.P3/'authorization/clock.json')
    assert clock['deadline_unix']==parent['deadline_unix']==1794048655.5459058
    assert clock['t0_unix']==parent['t0_unix']
    assert clock['repair_started_unix']==1793838400.0
    assert clock['repair_prepare_deadline_unix']==min(clock['repair_started_unix']+7200,clock['deadline_unix'])
    assert clock['parent_prepare_deadline_unix']==parent['prepare_deadline_unix']
    assert clock['parent_pt_deadline_unix']==parent['pt_deadline_unix']
    # Finished parent stage deadlines cannot block a downstream current-stage check.
    t.check_clock(False)
    budget=t.read(t.OUT/'budget.json');c=t.read(t.OUT/'cumulative_budget.json')
    assert budget['pt_updates']==0 and budget['available_source_pt_completed']==10000
    assert budget['ft_updates']==budget['ft_completed_updates']==336000
    assert budget['model_reads']['current_campaign_PT_alias']==210 and budget['model_reads']['random_head_alias']==210
    assert budget['model_reads']['historical_BASE_tensor_reads']==0
    assert c['total_attempts']==347155 and c['logged_completed']==347154 and c['actual_completed_interval']==[347154,347155]
    assert c['ft_starts']==211 and c['ft_attempts']==336297
    assert c['parent_r2_prefetch_final']=='UNKNOWN' and c['parent_r3_prefetch_not_consumed']==0

def test_pss_sum_uses_final_live_set_not_exited_history(monkeypatch,tmp_path):
    env=dict(mem_available=64*1024**3,**{'memory.current':200*1024**3,'memory.max':'max'})
    a=dict(pid=7,start_ticks=40,state='R',role='worker',rss_bytes=1)
    b=dict(pid=8,start_ticks=41,state='R',role='descendant',rss_bytes=1)
    monitor=t.MemoryMonitor.__new__(t.MemoryMonitor)
    monitor.out=tmp_path;monitor.active={'7:40':a,'8:41':b};monitor.retired=set();monitor.last_pss=0.;monitor.last_light=0.;monitor.total_seconds=0.;monitor.samples=0;monitor.last_identity_error=None
    snapshots=iter([{'7:40':a,'8:41':b},{'7:40':a}])
    def discover():
        monitor.active=next(snapshots);return dict(monitor.active)
    def observe(entry,**kwargs):
        fields={k:0 for k in t.PSS_FIELDS};fields['Pss']=100 if entry['pid']==7 else 200
        return dict(status='MEASURED',fields=fields,identity=entry,attempts=[])
    monitor.discover=discover
    monkeypatch.setattr(t,'memory_environment',lambda:env);monkeypatch.setattr(t,'observe_pid_memory',observe)
    sample=monitor.sample(force=True)
    assert sample['pss_swap_bytes']==100 and len(sample['pids'])==1
    assert sample['coverage']['rate']==1 and len(sample['exited_measurements'])==1
