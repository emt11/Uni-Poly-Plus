"""Only final CPU suite, after all210 formal units. No GPU forward fixtures."""
import json,time
from pathlib import Path
import numpy as np
import torch
import pytest
from src.training import ua3d_base_stabilize_r1 as t
from src.training.ua3d_cond_spectral_r6 import capture_mm_evidence,observe_pid_memory,PendingQueue,PSSReadError

def test_complete_accounting_and_phase_order():
    b=t.read(t.OUT/'budget.json');assert b['pt_complete']==2 and b['pt_updates']==20000 and b['pt_microbatches']==120000 and b['pt_presentations']==20160000
    assert b['ft_complete']==140 and b['ft_updates']==224000 and b['ft_epochs']==9800 and b['train_rows']==5260080
    assert b['validation_rows']==1319920 and b['swa_validation']==140 and b['swa_validation_rows']==18856
    assert b['total_updates']==244000 and b['test_complete']==210 and b['test_rows']==35250 and b['cpu_suites']==1
    assert b['pre_cpu']==b['diagnostics']==b['extra_forwards']==b['historical_tensor_reads']==0 and b['ft_pairs_passed']==35
    f=t.read(t.OUT/'selection/final_frozen.json');j=t.read(t.OUT/'jobs.json');cpu=next(x for x in j if x['kind']=='cpu')
    assert max(x['ended_unix'] for x in j if x['kind'] in ('ft','compare_ft'))<f['frozen_unix']<min(x['started_unix'] for x in j if x['kind']=='test')
    assert max(x['ended_unix'] for x in j if x['kind']=='test')<cpu['started_unix']
    assert f['FINAL_ESTIMATOR']=='BASE_SWA_LAST4_CYCLES' and f['PRIMARY_RUN']=='S42_A' and len(f['units'])==210

def test_new_factory_schema_and_forbidden_sources():
    factory=t.read(t.OUT/'initialization/source_factory.json');assert factory['native_seed']==42 and factory['modal_seed']==20261025 and factory['line_head_isolated_seed']==20261019 and not factory['modal_second_reset']
    initial=t.load_tensor(t.INITIAL);t.m.validate_pt(initial,0);assert not any(k.startswith(('proj2.','proj3.','fp_head.')) for k in initial['state_dict'])
    for task in t.TASKS:
        for fold in range(5):
            h=t.load_tensor(t.head_path(task,fold));assert h['schema']==t.m.HEAD_SCHEMA and h['seed']==20261025+100*t.TASKS.index(task)+fold
    with pytest.raises(ValueError):t.load_tensor(t.B/'pretrain/F_PD_N1_F0/deploy.pt')
    with pytest.raises(ValueError):t.m.validate_pt(dict(schema='ua3d-base-opt-pretrain-r6'),10000)

def test_actual_schema_conversion_dropout_and_optimizer():
    p=t.load_tensor(t.pt_path('PT_R0'));h=t.load_tensor(t.head_path('eps',0));model=t.m.initialize_ft(p,h,'eps',0)
    assert not model.encoder3d.mask_token.requires_grad
    for name,prob in [('o8',0.),('encoder3d',0.),('fp_encoder',.2),('head',.5)]:assert all(x.p==prob for x in getattr(model,name).modules() if isinstance(x,torch.nn.Dropout))
    opt,groups=t.m.optimizer_ft(model,'head');assert len(opt.param_groups)==1 and groups[0]['lr']==3e-4
    assert not any(p.requires_grad for n,p in model.named_parameters() if n.startswith(('o8.','encoder3d.','norm2.','norm3.')))
    opt,groups=t.m.optimizer_ft(model,'joint');assert [g['lr'] for g in opt.param_groups]==[1e-4,1e-3] and not model.encoder3d.mask_token.requires_grad

def test_swa_fixed_float64_average_and_buffers():
    model=torch.nn.Linear(2,1);model.register_buffer('constant',torch.tensor([3.]));avg=t.m.SWA()
    for epoch in (40,50,60,70):
        with torch.no_grad():
            for p in model.parameters():p.fill_(float(epoch))
        avg.collect(model,epoch)
    state=avg.state();assert state['weight'].dtype==torch.float32 and torch.equal(state['weight'],torch.full_like(state['weight'],55.)) and state['constant'].item()==3
    bad=t.m.SWA()
    with pytest.raises(ValueError):bad.collect(model,30)
    bad.collect(model,40);model.constant.fill_(4.)
    with pytest.raises(ValueError):bad.collect(model,50)
    with pytest.raises(ValueError):t.m.SWA().collect(torch.nn.BatchNorm1d(2),40)

def test_reproducibility_and_exact_not_approximate():
    assert t.read(t.OUT/'comparisons/PT.json')['equal']
    for task in t.TASKS:
        for fold in range(5):assert t.read(t.OUT/'comparisons/FT'/task/f'fold{fold}.json')['equal']
    assert not t.m.exact_compare(torch.tensor([1.]),torch.tensor([1.+1e-6]))['equal']
    assert not t.m.exact_compare(torch.tensor([1.]),torch.tensor([1.],dtype=torch.float64))['equal']

def test_all_formal_original_rows_and_finite_predictions():
    rows=0
    for g in t.CANONICAL:
        for e in ('BEST','SWA'):
            agg=t.aggregate(g,e,True);assert len(agg['units'])==35
            for task in t.TASKS:
                for fold in range(5):
                    p=t.unit_path(g,task,fold,e)
                    with np.load(p/'test_predictions.npz',allow_pickle=False) as a:
                        assert np.isfinite(a['y_true']).all() and np.isfinite(a['y_pred']).all();indices=a['indices'].tolist();rows+=len(indices)
                        assert indices==sorted(t.read(t.SPLIT/f'{task}.json')['folds'][fold]['test_indices']) and len(indices)==len(set(indices))==len(a['sample_keys'])
    assert rows==35250 and not (t.OUT/'formal/S42_B').exists()

def test_loader_clock_metric_and_pending_contract():
    clock=t.read(t.OUT/'authorization/clock.json');assert clock['deadline_unix']==clock['t0_unix']+48*3600 and clock['prepare_deadline_unix']==clock['t0_unix']+4*3600
    assert t.metrics(np.array([1.,2.,3.]),np.array([1.,2.,3.]))['r2']==1.
    now=[1.];q=PendingQueue(clock=lambda:now[0],unix=lambda:now[0]);entry=dict(pid=7,start_ticks=8);q.add('7:8',entry);now[0]=2.;q.add('7:8',entry);assert q.entries['7:8']['deadline_monotonic']==6.
    now[0]=6.;assert q.expired()
    for group in t.GROUPS:assert t.config(group,'eps',0)['workers']==0 and t.config(group,'eps',0)['seed']==t.m.SEEDS[group]

def test_combined_no_mm_proof_and_permission_rejection():
    import errno
    parts=['0']*50;parts[0]='D';parts[1]='6';parts[17]='1';parts[19]='40';entry=dict(pid=7,start_ticks=40,ppid=6,state='D',raw_stat='7 (pt_data_worker) '+' '.join(parts),no_mm_eligible=True,mm_ancestry=[dict(pid=6,start_ticks=3,role='pt')])
    files=dict(status='Pid:\t7\nTgid:\t7\nPPid:\t6\nState:\tD\nThreads:\t1\nKthread:\t0\n',maps='',smaps='',statm='0 0 0 0 0 0 0\n');clock=[0.]
    proof=lambda row:capture_mm_evidence(row,lambda pid:entry,lambda pid,name:files[name],lambda pid:['7'],lambda:clock[0])
    def rollup(pid):raise ProcessLookupError(errno.ESRCH,'actual fixture ESRCH')
    r=observe_pid_memory(entry,rollup,lambda pid:entry,lambda:False,lambda:clock[0],lambda seconds:clock.__setitem__(0,clock[0]+seconds),proof)
    assert r['status']=='NO_USER_MM_OBSERVED' and r['Pss'] is None and len(r['mm_rounds'])==2
    files['status']+='VmRSS:\t0 kB\n';assert not proof(entry)['valid']
    def denied(pid):raise PermissionError(errno.EACCES,'not ESRCH')
    with pytest.raises(PSSReadError):observe_pid_memory(entry,denied,lambda pid:entry,lambda:False)

def test_exit_wait_not_exit_and_immutable_deadline(tmp_path):
    mon=t.MemoryMonitor.__new__(t.MemoryMonitor);entry=dict(pid=7,start_ticks=40,first_seen_unix=1.)
    mon.active={'7:40':entry};mon.exit_wait={};mon.mm_deadlines={};mon.exit_wait_late=[];mon.refresh_deadline=None;mon.pending_queue=PendingQueue();mon.event=lambda *args,**kwargs:None
    result=dict(status='NO_USER_MM_OBSERVED',Pss=None,SwapPss=None,attempts=[{},{}]);mon.enter_exit_wait('7:40',result);deadline=mon.exit_wait['7:40']['deadline_monotonic'];mon.enter_exit_wait('7:40',result)
    assert mon.exit_wait['7:40']['deadline_monotonic']==deadline and '7:40' in mon.active and mon.has_pending()
    mon.exit_wait['7:40']['deadline_monotonic']=time.monotonic()-1
    assert mon.pending_timeout()['reason']=='TELEMETRY_ERROR_EXIT_DRAIN_TIMEOUT'


def test_actual_numerical_policies_and_immutable_sources():
    expected=dict(deterministic=True,warn_only=False,cudnn_benchmark=False,cudnn_deterministic=True,matmul_tf32=False,cudnn_tf32=False,matmul_precision='highest',fill_uninitialized_memory=True,cpu_threads=2,interop_threads=1)
    for p in (t.OUT/'finetune').glob('*/*/fold*/environment.json'):
        v=t.read(p)['policy'];assert all(v[k]==x for k,x in expected.items()) and v['env']['CUBLAS_WORKSPACE_CONFIG']==':4096:8' and v['env']['PYTHONHASHSEED']=='20261028'
    for run in ('PT_R0','PT_R1'):
        v=t.read(t.OUT/'pretrain'/run/'config.json')['setup']['actual'];assert all(v[k]==x for k,x in expected.items())
    locked=t.read(t.OUT/'authorization/locked_sources.json')
    assert all((t.ROOT/p).read_bytes()==(t.OUT/'source_snapshot'/p).read_bytes() for p in locked['files'])
