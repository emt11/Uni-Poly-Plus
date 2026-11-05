"""Sole final CPU suite; never run before complete formal TEST."""
import json
import numpy as np
import torch
from src.modules import ua3d_cond_spectral_r5 as m
from src.training import ua3d_cond_spectral_r5 as t

def test_formal_completion_and_budget():
    budget=t.read(t.OUT/'budget.json')
    assert budget['test_complete']==210 and budget['test_rows']==35250
    assert budget['pt_completed']==0 and budget['available_source_pt_completed']==10000 and budget['ft_complete']==209 and budget['canonical_ft_complete']==210 and budget['ft_completed_updates']==335230
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
    sample=dict(full_pss=True,coverage_complete=True,coverage_result='STOP',fatal_stop='MEMORY_STOP: fixture',time_unix=1.,pids=[dict(pid=9,start_ticks=42,fields=dict(Pss=161*1024**3,SwapPss=0))],decision='MEMORY_STOP: fixture',pss_swap_bytes=161*1024**3)
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
    assert m.FT_SCHEMA=='ua3d-cond-spectral-best-r5' and m.INPUT_PT_SCHEMA=='ua3d-cond-spectral-pretrain-r3'

def test_pending_deadline_immutable_edf_and_real_service():
    now=[10.];events=[]
    queue=t.PendingQueue(clock=lambda:now[0],unix=lambda:1000.+now[0],emit=lambda kind,**data:events.append((kind,data)))
    entry=dict(pid=1,start_ticks=2,state='R');queue.add('1:2',entry)
    first=queue.entries['1:2']['deadline_monotonic'];now[0]=10.2;queue.add('1:2',entry)
    assert queue.entries['1:2']['deadline_monotonic']==first==15.
    queue.add('3:4',dict(pid=3,start_ticks=4,state='S'))
    assert queue.snapshot()==['1:2','3:4']
    queue.started('1:2');queue.resolve('1:2','MEASURED',1)
    assert '1:2' not in queue.entries and not queue.expired()
    now[0]=16.;assert queue.expired()[0]['key']=='3:4'
    assert any(kind=='pending_service_started' for kind,_ in events)

def test_pending_is_not_passing_and_does_not_persist_stop(tmp_path):
    sample=dict(coverage_result='PENDING',fatal_stop=None,full_pss=True,cohort_complete=True,pss_total_all_active=None,active_coverage=dict(measured=7,active=8,rate=.875),pending=[dict(key='8:80',first_seen_monotonic=1.,deadline_monotonic=6.,read_attempts=0)],dispatch_allowed=False)
    t.persist_memory_sample(tmp_path,sample)
    assert not (tmp_path/'telemetry_error.json').exists() and not (tmp_path/'memory_breach.json').exists()
    assert not (tmp_path/'memory_last_full.json').exists() and not (tmp_path/'first_failure.json').exists()
    assert json.loads((tmp_path/'memory_latest.json').read_text())['pss_total_all_active'] is None
    stopped=dict(sample,coverage_result='STOP',fatal_stop='TELEMETRY_ERROR_PENDING_DEADLINE: fixture')
    t.persist_memory_sample(tmp_path,stopped)
    assert (tmp_path/'telemetry_error.json').exists()

def test_actual_read_failure_cannot_be_requeued_for_five_seconds():
    now=[1.];q=t.PendingQueue(clock=lambda:now[0],unix=lambda:now[0]);q.add('1:1',dict(pid=1,start_ticks=1,state='R'))
    deadline=q.entries['1:1']['deadline_monotonic'];q.started('1:1');q.read_failed('1:1',dict(attempts=[dict(errno=3)]))
    assert q.entries['1:1']['phase']=='READ_FAILED' and q.entries['1:1']['read_attempts']==1
    now[0]=2.;q.add('1:1',dict(pid=1,start_ticks=1,state='R'))
    assert q.entries['1:1']['phase']=='READ_FAILED' and q.entries['1:1']['deadline_monotonic']==deadline

def test_fixed_cohort_tail_pending_then_priority_service(monkeypatch,tmp_path):
    env=dict(mem_available=64*1024**3,**{'memory.current':200*1024**3,'memory.max':'max'})
    a=dict(pid=7,start_ticks=40,state='R',role='worker',rss_bytes=1)
    b=dict(pid=8,start_ticks=41,state='R',role='descendant',rss_bytes=1)
    mon=t.MemoryMonitor.__new__(t.MemoryMonitor);mon.out=tmp_path;mon.active={'7:40':a};mon.retired=set();mon.last_pss=0.;mon.last_light=0.;mon.total_seconds=0.;mon.samples=0;mon.window=None;mon.pass_number=0;mon.next_pending_wake=None;mon.refresh_needed=False;mon.refresh_deadline=None
    mon.pending_queue=t.PendingQueue(emit=lambda *args,**kwargs:None)
    states=[0]
    def discover():
        states[0]+=1
        if states[0]==1:mon.active={'7:40':a}
        else:
            mon.active={'7:40':a,'8:41':b}
            if states[0]==2:mon.pending_queue.add('8:41',b)
        return dict(mon.active)
    seen=[]
    def observe(entry,**kwargs):
        seen.append(entry['pid']);fields={k:0 for k in t.PSS_FIELDS};fields['Pss']=entry['pid']
        return dict(status='MEASURED',fields=fields,identity=entry,attempts=[dict(errno=None)])
    mon.discover=discover;mon.exit_event=lambda key:False
    monkeypatch.setattr(t,'memory_environment',lambda:env);monkeypatch.setattr(t,'observe_pid_memory',observe)
    first=mon.sample(force=True)
    assert first['cohort_complete'] and len(first['cohort_targets'])==1 and seen==[7]
    assert first['coverage_result']=='PENDING' and first['active_coverage']['rate']==.5
    assert first['pss_total_all_active'] is None and not first['dispatch_allowed']
    deadline=mon.pending['8:41']['deadline_monotonic']
    # Force immediate normal queue service; it must not wait for light/PSS throttle.
    second=mon.sample(force=True)
    assert seen==[7,8] and second['coverage_result']=='COMPLETE' and second['dispatch_allowed']
    assert second['pss_total_all_active']==15 and not mon.pending
    assert second['pending_service_requested'] and deadline>second['actual_service_monotonic']

def test_canonical_alias_configs_all_formal_outputs_and_budget():
    canonical=t.read(t.OUT/'registry/canonical.json');entries=canonical['units']
    assert len(entries)==210 and sum(e['kind']=='parent_complete_alias' for e in entries.values())==1
    alias=entries['C_FRESH/eps/0'];assert alias['estimator_config']['revision']=='r4' and alias['runner_config']['revision']=='r5'
    assert alias['best_schema']=='ua3d-cond-spectral-best-r4'
    frozen=t.read(t.OUT/'selection/final_frozen.json');unit=t.frozen_unit(frozen,'C_FRESH','eps',0)
    assert unit['best_epoch']==32 and unit['best'].startswith(str(t.P4))
    formal=t.unit_path('C_FRESH','eps',0,True);assert formal.is_relative_to(t.OUT) and (formal/'test_predictions.npz').exists()
    budget=t.read(t.OUT/'budget.json');c=t.read(t.OUT/'cumulative_budget.json')
    assert budget['ft_starts']==budget['ft_complete']==209 and budget['parent_complete_ft_aliases']==1
    assert budget['ft_updates']==budget['ft_completed_updates']==335230
    assert budget['model_reads']['current_campaign_PT_alias']==209 and budget['model_reads']['random_head_alias']==209
    assert budget['model_reads']['parent_complete_BEST']==1 and budget['model_reads']['current_r5_BEST']==209
    assert c['total_attempts']==347518 and c['logged_completed']==347517 and c['actual_completed_interval']==[347517,347518]
    assert c['ft_starts']==212 and c['ft_attempts']==336660
    assert c['parent_complete_alias_updates_not_recharged']==770 and c['parent_r2_prefetch_final']=='UNKNOWN'
    clock=t.read(t.OUT/'authorization/clock.json');parent=t.read(t.P4/'authorization/clock.json')
    assert clock['deadline_unix']==parent['deadline_unix']==1794048655.5459058
    assert clock['repair_started_unix']==1793840882.0
    assert clock['repair_prepare_deadline_unix']==min(1793840882.+7200,clock['deadline_unix'])
    assert clock['parent_r4_clock']==parent

def test_controller_cannot_admit_on_pending(monkeypatch):
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('condspec_r5_final_fixture',t.ROOT/'scripts/run_ua3d_cond_spectral_r5.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    ctl=runner.Controller.__new__(runner.Controller);ctl.live=[];ctl.boundary_due=False
    outcomes=iter([dict(coverage_result='PENDING',dispatch_allowed=False),dict(coverage_result='COMPLETE',dispatch_allowed=True)])
    class Memory:
        def sample(self,force=False):return next(outcomes)
    ctl.memory=Memory();ctl.reap=lambda:None;waited=[];ctl.wait_event=lambda coverage_wait=False:waited.append(coverage_wait)
    monkeypatch.setattr(t,'check_clock',lambda prepare=False:None)
    assert ctl.ensure_complete()['coverage_result']=='COMPLETE' and waited==[True]
