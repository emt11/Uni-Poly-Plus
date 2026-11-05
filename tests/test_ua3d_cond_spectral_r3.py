"""Sole final CPU suite; never run before complete formal TEST."""
import json
import numpy as np
import torch
from src.modules import ua3d_cond_spectral_r3 as m
from src.training import ua3d_cond_spectral_r3 as t

def test_formal_completion_and_budget():
    budget=t.read(t.OUT/'budget.json')
    assert budget['test_complete']==210 and budget['test_rows']==35250
    assert budget['pt_completed']==10000 and budget['ft_complete']==210 and budget['ft_completed_updates']==336000
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

def test_pid_exit_race_and_start_tick_registry():
    import pytest
    entry=dict(pid=123,start_ticks=10,state='R')
    def vanished(pid):raise FileNotFoundError('exit during /proc read')
    assert t.observe_pid_memory(entry,vanished,lambda pid:None)['status']=='EXITED'
    assert t.observe_pid_memory(entry,vanished,lambda pid:dict(pid=123,start_ticks=11,state='R'))['status']=='EXITED'
    with pytest.raises(RuntimeError,match='living PID'):
        t.observe_pid_memory(entry,vanished,lambda pid:entry)
    with pytest.raises(RuntimeError,match='required PSS'):
        t.observe_pid_memory(entry,lambda pid:'Rss: 0 kB',lambda pid:entry)


def test_template_lru_storage_and_owned_lmdb_lifecycle(tmp_path):
    from src.dataset import ua3d_cond_spectral_r3 as d
    import lmdb,pickle
    cache=d.TemplateLRU(entries=2,max_bytes=32)
    shared=torch.ones(4)
    cache.put('a',dict(x=shared,y=shared.view(2,2)));assert cache.bytes==16
    cache.put('b',dict(x=shared));assert cache.bytes==16
    cache.put('c',dict(x=torch.ones(4)));assert len(cache.data)==2 and cache.bytes==32
    assert cache.get('a') is None
    assert not cache.put('oversize',dict(x=torch.ones(100)))
    root=tmp_path/'fixture.lmdb';env=lmdb.open(str(root),map_size=1024*1024)
    with env.begin(write=True) as txn:txn.put(b'key',pickle.dumps(np.arange(4,dtype=np.float32)))
    env.close()
    env=lmdb.open(str(root),readonly=True,lock=False,readahead=False,max_readers=128)
    with env.begin(write=False,buffers=False) as txn:payload=txn.get(b'key')
    assert isinstance(payload,bytes)
    tensor=torch.from_numpy(pickle.loads(payload));env.close()
    # Closed LMDB cannot invalidate an owned payload or its decoded tensor.
    assert torch.equal(tensor,torch.arange(4,dtype=torch.float32))
    env=lmdb.open(str(root),readonly=True,lock=False,readahead=False,max_readers=128);env.close()

def test_inherited_clock_parent_unknown_tail_and_prefix():
    c=t.read(t.OUT/'authorization/clock.json');parent=t.read(t.PARENT/'authorization/clock.json');pt=t.read(t.PARENT/'pretrain/BASE_FRESH/clock.json')
    for key in ('t0_unix','prepare_deadline_unix','deadline_unix','engineering_target_unix'):assert c[key]==parent[key]
    assert c['pt_deadline_unix']==pt['deadline_unix']
    cumulative=t.read(t.OUT/'cumulative_budget.json')
    assert cumulative['parent_actual_completed_interval']==[857,858]
    assert cumulative['total_attempts']==346858
    assert cumulative['actual_completed_interval']==[346857,346858]
    assert cumulative['parent_final_prefetch']=='UNKNOWN'
    consume=t.read(t.OUT/'pretrain/BASE_FRESH/consumption.json')
    assert consume['prefix_checked']==5148 and consume['completed_updates']==10000
    assert all(x['lru_entries']<=128 and x['lru_storage_bytes']<=256*1024**2 and x['reopens']>0 for x in consume['worker_io'].values())
