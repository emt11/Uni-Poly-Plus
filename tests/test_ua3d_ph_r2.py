"""Only the two explicitly registered CPU suites. No real model or data in preflight."""
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.dataset import ua3d_ph_r2 as d
from src.modules import ua3d_ph_r2 as m

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ua3d_ph_20261028/r1/repairs/gpu2_r2'


def synthetic_side(n=3, edges=None, available=True):
    edges=np.asarray(edges if edges is not None else [(0,1),(1,2)],np.int64).reshape(-1,2)
    q=torch.as_tensor(d.random_walk_pe(n,edges))
    return dict(rwpe=q,edges=edges,center=torch.arange(n),graph_available=True,
                events=[d.graph_events(q[:,k].numpy(),edges) for k in (1,3,7,15)],available=available,
                physical_atoms=n,bond_atoms=torch.tensor([[0,1],[1,2]]),
                incidence=torch.tensor([[0,0,1,1],[0,1,1,2]]),canonical_graph=torch.zeros(n,dtype=torch.long),
                sp_edges=[torch.tensor([[0,1,1,2],[1,0,2,1]])]*3,
                sp_distances=[torch.ones(4)]*3,router=torch.zeros((31,5)),router_raw=np.zeros((31,5)))


def test_preflight(monkeypatch):
    assert os.environ.get('PH_R2_SUITE')=='preflight' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    folder=OUT/'preflight_cpu'; folder.mkdir(exist_ok=False)
    state=dict(status='RUNNING',preflight_cpu_suites=1,preflight_adapter_constructs=0,preflight_forwards=0,
               preflight_backwards=0,preflight_graph_reductions=0,preflight_vr_reductions=0,
               full_base_constructs=0,real_data_rows=0,optimizer_updates=0,groups_completed=[])
    limits=json.loads((ROOT/'configs/ua3d_ph_r2_gpu2.json').read_text())['limits']
    def record():
        p=folder/'consumption.json'; temp=p.with_suffix('.tmp'); temp.write_text(json.dumps(state,indent=2)+'\n'); temp.replace(p)
        # Also provide the explicitly named independent counters.
        q=OUT/'preflight_cpu_consumption.continuation01.json'; q.write_text(json.dumps(state,indent=2)+'\n')
    def charge(key):
        state[key]+=1; record(); assert state[key]<=limits[key]
    def finish_group(name):
        state['groups_completed'].append(name); record()
    old_events=d.graph_events; old_vr=d.vr_betti
    def events(*args,**kwargs):
        charge('preflight_graph_reductions'); return old_events(*args,**kwargs)
    def vr(*args,**kwargs):
        charge('preflight_vr_reductions'); return old_vr(*args,**kwargs)
    monkeypatch.setattr(d,'graph_events',events); monkeypatch.setattr(d,'vr_betti',vr)
    for cls in (m.GraphPHAdapter,m.SpatialRouter):
        original_init=cls.__init__; original_forward=cls.forward
        def init(self,*args,_init=original_init,**kwargs):
            charge('preflight_adapter_constructs'); _init(self,*args,**kwargs)
        def forward(self,*args,_forward=original_forward,**kwargs):
            charge('preflight_forwards'); return _forward(self,*args,**kwargs)
        monkeypatch.setattr(cls,'__init__',init); monkeypatch.setattr(cls,'forward',forward)
    record()
    try:
        import gudhi
        def reference(n,e,f):
            charge('preflight_vr_reductions')
            st=gudhi.SimplexTree()
            for i in range(n):
                st.insert([i],filtration=float(f[i]))
            for a,b in d.undirected(e,n):
                st.insert([int(a),int(b)],filtration=float(max(f[a],f[b])))
            st.compute_persistence(homology_coeff_field=2,persistence_dim_max=True,min_persistence=-1)
            return [sorted((float(a),float(b)) for a,b in st.persistence_intervals_in_dimension(k)) for k in (0,1)]
        def encoded_bars(n,e,f):
            result=d.graph_events(f,e); bars=[]
            for es in result:
                values=d.event_tuples(torch.tensor(f,dtype=torch.float64),es).tolist()
                bars.append(sorted((v[0],math.inf if v[3] else v[1]) for event,v in zip(es,values) for _ in range(event['count'])))
            return result,bars
        cases=[(0,[],[]),(1,[],[.25]),(3,[],[.125,.25,.5]),
               (2,[(0,1)],[.125,.75]),(3,[(0,1),(1,2)],[.125,.25,.5]),
               (3,[(0,1),(1,2),(2,0)],[.125,.25,.5]),
               (4,[(0,1),(1,2),(2,3),(3,0)],[.125,.25,.5,.75]),
               (4,[(0,1),(2,3)],[.25,.25,.5,.5]),
               (3,[(0,1),(1,2),(0,2)],[.5,.5,.5]),
               (4,[(0,1),(1,2),(2,0),(2,3)],[.25,.25,.25,.75])]
        for n,e,f in cases:
            _,bars=encoded_bars(n,e,np.asarray(f)); assert bars==reference(n,e,f)
        finish_group('01_graph_H0_H1_independent_GUDHI')
        for n,e,f in cases[:3]:
            result,bars=encoded_bars(n,e,np.asarray(f)); assert sum(x['count'] for x in result[0])==n
            assert sum(x['count'] for x in result[0] if x['censored'])==n
        finish_group('02_empty_singleton_isolated_disconnected')
        graph=d.graph_events(np.array([.25,.25,.25]),[(0,1),(1,2),(0,2)])
        assert sum(e['count'] for e in graph[1])==1
        spatial=d.vr_betti([[0,0,0],[1,0,0],[.5,.5,0]])
        assert np.count_nonzero(spatial['curves'][:,1])==0
        finish_group('03_graph_cycle_vs_filled_VR_triangle')
        f=np.asarray([.25,.25,.5,.5]); e=np.asarray([(0,1),(1,2),(2,3),(3,0),(0,2)])
        p=np.asarray([2,0,3,1]); inverse=np.argsort(p); shuffled=inverse[e][::-1]
        aa=d.graph_events(f,e); bb=d.graph_events(f[p],np.concatenate((shuffled,shuffled[:,::-1])))
        for es,fs in zip(aa,bb):
            # A nonlinear encoder makes premature tuple summation detectable.
            charge('preflight_forwards')
            a=d.distribute(d.event_tuples(torch.tensor(f),es).square(),es,4)
            b=d.distribute(d.event_tuples(torch.tensor(f[p]),fs).square(),fs,4)
            assert torch.equal(a,b[inverse])
        finish_group('04_tie_blocks_duplicates_and_permutation')
        g=d.lifted_graph(np.array([[0,1,-1],[1,0,-1]]),np.array([[0,1,0],[1,0,0]]),np.array([[1,1,0],[1,1,0]],bool),np.array([1,1]),2)
        assert g['nodes']==6 and g['cross_ru']==2 and g['edges'].tolist()==[[0,3],[2,5]] and g['center'].tolist()==[2,3]
        finish_group('05_three_copy_lift_and_center')
        adapter=m.GraphPHAdapter(); side=synthetic_side()
        # Learnable output is made active solely on this synthetic object; no optimizer update.
        with torch.no_grad():
            adapter.out[-1].weight.fill_(.01)
        delta,info=adapter(side=side,arm='P_LEARN'); charge('preflight_backwards'); delta.square().sum().backward()
        assert info['learnable'] and all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.filter.parameters())
        assert sum(float(p.grad.abs().sum()) for p in adapter.filter.parameters())>0
        finish_group('06_learned_filtration_gradient')
        ftorch=torch.tensor([.125,.25,.5],requires_grad=True); es=d.graph_events(ftorch.detach().numpy(),[(0,1),(1,2)])
        assert not es[1]
        charge('preflight_forwards'); h1=d.distribute(adapter.h1(d.event_tuples(ftorch,es[1])),es[1],3)
        assert torch.count_nonzero(h1)==0
        tuples=d.event_tuples(ftorch,es[0]); charge('preflight_forwards')
        exact=d.distribute(adapter.h0(tuples),es[0],3)
        assert exact.shape==(3,32)
        # Direct reference sums event encodings after the nonlinearity.
        expected=torch.zeros_like(exact)
        for col,event in enumerate(es[0]):
            expected[event['nodes']]+=adapter.h0(tuples[col])*event['count']/len(event['nodes'])
        assert torch.allclose(exact,expected,atol=1e-7,rtol=1e-6)
        finish_group('07_encode_events_before_distribution')
        from src.dataset.ua3m_paperfuse import ModalBatch
        from torch_geometric.data import Data
        batch=ModalBatch(Data(x=torch.zeros(2,1)),None,torch.empty(0,dtype=torch.long),torch.zeros(1,2048),torch.tensor([[True,False,True]]),['mock'],torch.zeros(1))
        carrier=d.PHBatch(batch,[side]); moved=carrier.to('cpu')
        assert moved.sides[0]['center'].device.type=='cpu' and torch.equal(moved.base.availability,batch.availability)
        # Each side keeps its own node/edge/bond tables. Two graphs never share indices.
        carrier2=d.PHBatch(batch,[side,synthetic_side()]); assert carrier2.sides[0]['bond_atoms'].max()==2 and carrier2.sides[1]['center'].min()==0
        finish_group('08_separate_local_offsets_and_ModalBatch_to')
        for arm in m.P_ARMS:
            module=m.GraphPHAdapter(); assert sum(p.numel() for p in module.parameters())==42820
            value,_=module(side=side,arm=arm); assert torch.count_nonzero(value)==0
        for arm in m.M_ARMS:
            module=m.SpatialRouter(); assert sum(p.numel() for p in module.parameters())==36643
            value,details=module(states=torch.ones(2,512),side=side,arm=arm)
            assert torch.count_nonzero(value)==0 and torch.allclose(details['alpha'].sum(),torch.tensor(1.))
        class Mock:
            ph_arm='P_FIXED'
            graph_ph=m.GraphPHAdapter()
            norm2=torch.nn.Identity()
            A2=torch.nn.Linear(512,256)
        mock=Mock(); original=[torch.ones(3,256),torch.ones(3,256),torch.ones(3,256)]
        mock_batch=SimpleNamespace(sides=[side],two_d=SimpleNamespace(canonical_graph_index=torch.arange(3)),availability=torch.ones(3,3,dtype=torch.bool))
        charge('preflight_forwards'); z=m.PHFuse.modified_modalities(mock,mock_batch,original,torch.ones(3,512),None)
        assert all(torch.equal(x,torch.ones_like(x)) for x in original) and z[0] is not original[0]
        finish_group('09_parameter_counts_zero_residual_ALIGN_isolation')
        singleton=d.vr_betti([[0,0,0]]); assert np.array_equal(singleton['curves'][:,0],np.ones(31))
        pair=d.vr_betti([[0,0,0],[4.5,0,0]])
        assert pair['curves'][-1,0]==.5 and pair['curves'][-2,0]==1.
        far=d.vr_betti([[0,0,0],[5.,0,0]])
        assert far['curves'][-1,0]==1. and np.count_nonzero(np.isinf(far['intervals'][:,2]))==2
        finish_group('10_VR_singleton_endpoint_right_censoring')
        values=np.zeros((3,31,5)); values[1]=2; values[2]=1000
        stats=d.fit_router(values,[0,1]); assert np.asarray(stats['mean']).mean()==1 and np.asarray(stats['scale']).mean()==1
        missing=SimpleNamespace(ph_arm='M_PH',last_ph={})
        avail=torch.tensor([[True,False,True]])
        charge('preflight_forwards'); z=m.PHFuse.modified_modalities(missing,SimpleNamespace(three_d=None,availability=avail),[torch.ones(1,256),torch.zeros(1,256),torch.ones(1,256)],None,None)
        assert torch.count_nonzero(z[1])==0
        zero_side=dict(side,sp_edges=[torch.empty((2,0),dtype=torch.long)]*3,sp_distances=[torch.empty(0)]*3)
        value,detail=module(states=torch.ones(2,512),side=zero_side,arm='M_PH')
        assert torch.isfinite(value).all() and torch.allclose(detail['alpha'].sum(),torch.tensor(1.))
        finish_group('11_missing_geometry_no_edges_train_only_router')
        class Sentinel:
            topology=SimpleNamespace(); trimer=SimpleNamespace(); static={}; normalized_smiles='mock'; final_geometry_status='PASS'
            @property
            def label(self):
                raise AssertionError('target read')
            @property
            def target(self):
                raise AssertionError('target read')
        selected=d.structure_fields(Sentinel()); assert not hasattr(selected,'target')
        from src.training import ua3d_ph_r2 as t
        from scripts import run_ua3d_ph_r2 as c
        assert all(t.gpu_for(task,f)==2 for task in t.TASKS for f in range(5))
        assert t.read(OUT/'execution_manifest.json')['status']=='STOPPED'
        assert c.CAPS==dict(pt=2,diagnostic=1,ft=4,test=4)
        r=c.reservation('diagnostic','P_LEARN'); assert r['p_learn_reductions']==2112
        assert c.SPEC['limits']['pt_starts']==c.SPEC['limits']['pt_updates']==0
        finish_group('12_structure_target_sentinel_GPU_budget_parent_protection')
        assert len(state['groups_completed'])==12
        state['status']='COMPLETE'; record()
    except BaseException:
        import traceback
        state.update(status='FAILED',error=traceback.format_exc()); record(); raise


def test_final():
    assert os.environ.get('PH_R2_SUITE')=='final' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    from src.training import ua3d_ph_r2 as t
    from scripts import run_ua3d_ph_r2 as c
    folder=OUT/'final_cpu'; folder.mkdir(exist_ok=False)
    state=dict(status='RUNNING',cpu_suites=1)
    t.write(folder/'consumption.json',state)
    try:
        budget=t.read(c.BUDGET); jobs=budget['jobs']; frozen=t.read(OUT/'selection/final_frozen.json')
        assert not budget['budget_violations'] and budget['deadline_unix']==1793372574.008285
        assert len(frozen['units'])==245 and len(frozen['reference_aliases'])==35
        assert frozen['FINAL']==t.select_final(frozen['selection']['validation'])
        assert sum(j['kind']=='diagnostic' and j['exit']==0 for j in jobs)==14
        assert sum(j['kind']=='ft' and j['exit']==0 for j in jobs)==245
        assert sum(j['kind']=='test' and j['exit']==0 for j in jobs)==245
        assert sum(j['kind']=='preflight' and j['exit']==0 for j in jobs)==1
        assert sum(j['kind']=='cpu' for j in jobs)==1
        assert t.read(OUT/'preflight_cpu/consumption.json')['status']=='COMPLETE'
        assert t.read(OUT/'diagnostics/comparison.json')['numerical_diagnostic_pass'] is True
        assert t.stability_gate()['count']==35
        t.check_prepared(); coverage=t.read(OUT/'inputs/coverage.json'); assert coverage['geometry_available']==5860 and coverage['geometry_missing']==15
        for task in t.TASKS:
            for fold in range(5):
                stats=t.read(OUT/'inputs/scalers'/task/f'fold{fold}.json')
                assert stats['train_indices']==t.read(t.SPLIT/f'{task}.json')['folds'][fold]['train_indices']
        intervals=[]
        for j in jobs:
            if j['kind'] not in c.GPU_KINDS:
                continue
            assert j['env']['CUDA_VISIBLE_DEVICES']=='2' and j['env']['WORLD_SIZE']=='1' and j['physical_gpus']==[2]
            proc=t.read(Path(j['log'])/'process.json'); end=t.read(Path(j['log'])/'end.json')
            intervals.append((proc['started_unix'],1,j['kind'])); intervals.append((end['ended_unix'],-1,j['kind']))
            u=t.read(j['consumption']); assert u['status']=='COMPLETE'
            env=t.read(Path(j['consumption']).parent/'environment.json')
            assert env['numerical']['deterministic'] is True and env['numerical']['warn_only'] is False
            assert env['physical']['actual_uuid']==t.UUID
            if j['kind']=='ft':
                assert end['ended_unix']<=frozen['time_unix']
            if j['kind']=='test':
                assert proc['started_unix']>=frozen['time_unix']
        active={}
        for when,change,kind in sorted(intervals):
            active[kind]=active.get(kind,0)+change
            assert active[kind]<=c.CAPS[kind] and sum(n>0 for n in active.values())<=1
        for arm in m.ARMS:
            aggregate=t.aggregate(arm,True); assert len(aggregate['tasks'])==7
        assert t.REFERENCE==.804675462276949 and c.SPEC['performance_threshold']==.814675462276949
        assert t.SOTA['egb'][2]==.923 and t.SOTA['egc'][2]==.911
        assert t.read(OUT/'execution_manifest.json')['status']=='STOPPED'
        state['status']='COMPLETE'; t.write(folder/'consumption.json',state)
    except BaseException:
        import traceback
        state.update(status='FAILED',error=traceback.format_exc()); t.write(folder/'consumption.json',state); raise
