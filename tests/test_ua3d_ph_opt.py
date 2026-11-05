"""One preflight synthetic equivalence suite, then one post-TEST final suite."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.dataset import ua3d_ph_opt as d
from src.modules import ua3d_ph_opt as m

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ua3d_ph_opt_20261031/r1'


def synthetic(n,edges,permutation=None):
    edges=np.asarray(edges,np.int64).reshape(-1,2)
    q=d.old.random_walk_pe(n,edges)
    if permutation is not None:
        inverse=np.argsort(permutation); q=q[permutation]; edges=inverse[edges]
    return dict(rwpe=torch.tensor(q),edges=edges,center=torch.arange(n),graph_available=True,
                events=[d.old.graph_events(q[:,k],edges) for k in (1,3,7,15)],nodes=n)


def test_preflight(monkeypatch):
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.environ['PH_OPT_SUITE']=='preflight'
    folder=OUT/'preflight_cpu'; folder.mkdir(exist_ok=False)
    state=dict(status='RUNNING',preflight_cpu_suites=1,preflight_adapter_constructs=0,preflight_forwards=0,
               preflight_backwards=0,preflight_graph_reductions=0,preflight_vr_reductions=0,
               full_base_constructs=0,real_data_rows=0,optimizer_updates=0,groups_completed=[])
    limits=json.loads((ROOT/'configs/ua3d_ph_opt_r1_gpu2.json').read_text())['limits']
    def record():
        p=folder/'consumption.json'; q=p.with_suffix('.tmp'); q.write_text(json.dumps(state,indent=2)+'\n'); q.replace(p)
    def charge(k):
        state[k]+=1; record(); assert state[k]<=limits[k]
    def group(n):
        state['groups_completed'].append(n); record()
    original_events=d.old.graph_events
    def events(*args,**kw):
        charge('preflight_graph_reductions'); return original_events(*args,**kw)
    monkeypatch.setattr(d.old,'graph_events',events)
    original_init=m.old.GraphPHAdapter.__init__
    def init(self):
        charge('preflight_adapter_constructs'); original_init(self)
    monkeypatch.setattr(m.old.GraphPHAdapter,'__init__',init)
    for cls,name in ((m.old.GraphPHAdapter,'forward'),(m.GraphPHAdapter,'forward_batch')):
        original=getattr(cls,name)
        def call(self,*args,_old=original,**kw):
            charge('preflight_forwards'); return _old(self,*args,**kw)
        monkeypatch.setattr(cls,name,call)
    def forbidden(*args,**kw):
        raise AssertionError('full BASE, VR or optimizer not permitted in preflight')
    monkeypatch.setattr(m.base.BaseOptFuse,'__init__',forbidden)
    monkeypatch.setattr(d.old,'vr_betti',forbidden)
    record()
    try:
        # Construction must consume exactly the same RNG and retain every state key.
        torch.manual_seed(9182); before=torch.get_rng_state().clone(); a=m.old.GraphPHAdapter(); after_old=torch.get_rng_state().clone()
        torch.set_rng_state(before); b=m.GraphPHAdapter(); after_new=torch.get_rng_state().clone()
        assert torch.equal(after_old,after_new) and set(a.state_dict())==set(b.state_dict())
        assert sum(p.numel() for p in b.parameters())==42820
        assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
        group('parameter_names_shapes_initialization_RNG')
        cases=[[(3,[(0,1),(1,2)],None),(3,[(0,1),(1,2),(2,0)],None)],
               [(1,[],None),(4,[(0,1),(2,3)],None)],
               [(0,[],None),(4,[(0,1),(1,2),(2,3),(3,0)],np.array([2,0,3,1]))],
               [(4,[(0,1),(1,0),(1,2),(2,3),(3,0)],None),(3,[],None)]]
        for case_id,specs in enumerate(cases):
            sides=[synthetic(*spec) for spec in specs]; packed_sides=[d.pack_side(side) for side in sides]
            for arm in m.P_ARMS:
                old=m.old.GraphPHAdapter(); new=m.GraphPHAdapter()
                with torch.no_grad():
                    old.out[-1].weight.fill_(.01); old.out[-1].bias.fill_(.02)
                new.load_state_dict(old.state_dict(),strict=True)
                rng=torch.get_rng_state().clone()
                previous=[old(side=side,arm=arm) for side in sides]
                plan=d.batch_plan(packed_sides,arm); delta,info=new.forward_batch(plan,arm)
                reference=torch.cat([x[0] for x in previous]); assert torch.equal(rng,torch.get_rng_state())
                assert torch.allclose(delta,reference,atol=1e-6,rtol=1e-5), (case_id,arm,'delta')
                for key in ('pe','filtration','r0','r1'):
                    target=torch.cat([x[1][key] for x in previous])
                    assert torch.allclose(info[key],target,atol=1e-6,rtol=1e-5), (case_id,arm,key)
                charge('preflight_backwards'); reference.square().sum().backward()
                charge('preflight_backwards'); delta.square().sum().backward()
                left=dict(old.named_parameters()); right=dict(new.named_parameters())
                for key,p in left.items():
                    q=right[key]
                    if p.grad is None or q.grad is None:
                        assert p.grad is None and q.grad is None, (case_id,arm,key,'gradient disconnected')
                    else:
                        assert torch.isfinite(q.grad).all()
                        assert torch.allclose(p.grad,q.grad,atol=1e-5,rtol=1e-4), (case_id,arm,key,'gradient')
                if arm=='P_LEARN':
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.filter.parameters())
                    assert sum(float(p.grad.abs().sum()) for p in new.filter.parameters())>0
            group('old_new_equivalence_case_'+str(case_id))
        # Strict event identities/assignment survive batching offsets and duplicate edges.
        side=synthetic(4,[(0,1),(1,2),(2,3),(3,0)])
        packed=d.pack_side(side); combined=d.batch_plan([packed,packed],'P_FIXED'); n=side['nodes']
        for dim in (0,1):
            p=combined['channel'+str(dim)]
            assert p['events']==2*packed['fixed_plans'][dim]['events']
            encoded=torch.arange(p['events']*32,dtype=torch.float32).reshape(-1,32)/100
            charge('preflight_forwards'); output=m.distribute(encoded,p,2*n)
            expected=torch.zeros_like(output)
            cols=0
            for graph in range(2):
                for k in range(4):
                    for event in side['events'][k][dim]:
                        for node in event['nodes']:
                            expected[k*(2*n)+graph*n+node]+=encoded[cols]*event['count']/len(event['nodes'])
                        cols+=1
            assert torch.allclose(output,expected,atol=1e-6,rtol=1e-5)
        group('event_identity_membership_offsets_nonlinear_order')
        perm=np.array([2,0,3,1]); inverse=np.argsort(perm)
        permuted=synthetic(4,[(0,1),(1,2),(2,3),(3,0)],perm)
        old=m.old.GraphPHAdapter(); new=m.GraphPHAdapter(); new.load_state_dict(old.state_dict(),strict=True)
        with torch.no_grad():
            old.out[-1].weight.fill_(.01); new.out[-1].weight.fill_(.01)
        a,_=new.forward_batch(d.batch_plan([d.pack_side(side)],'P_FIXED'),'P_FIXED')
        b,_=new.forward_batch(d.batch_plan([d.pack_side(permuted)],'P_FIXED'),'P_FIXED')
        assert torch.allclose(a,b[inverse],atol=1e-6,rtol=1e-5)
        group('storage_permutation')
        # Exactly zero residual, batched center order, no mutation of ALIGN's originals.
        new=m.GraphPHAdapter()
        for arm in m.P_ARMS:
            value,_=new.forward_batch(d.batch_plan([d.pack_side(side)],arm),arm)
            assert torch.count_nonzero(value)==0
        batch=SimpleNamespace(packed=d.batch_plan([d.pack_side(side)],'P_FIXED'),two_d=SimpleNamespace(canonical_graph_index=torch.zeros(4,dtype=torch.long)),availability=torch.tensor([[True,False,True]]))
        mock=SimpleNamespace(ph_arm='P_FIXED',graph_ph=new,norm2=torch.nn.Identity(),A2=torch.nn.Linear(512,256),last_ph={})
        original=[torch.ones(1,256),torch.zeros(1,256),torch.ones(1,256)]
        charge('preflight_forwards'); z=m.OptimizedPHFuse.modified_modalities(mock,batch,original,torch.ones(4,512),None)
        assert original[0].eq(1).all() and original[1].eq(0).all() and z[1] is original[1]
        group('zero_residual_center_order_original_ALIGN')
        from scripts import run_ua3d_ph_opt as c
        assert c.stage_state([],1)==dict(status='NOT_RUN',attempts=0,exit=None,started=None,ended=None)
        assert c.stage_state([dict(exit=None,reserved_unix=1.)],1)['status']=='RUNNING'
        assert c.stage_state([dict(exit=0,reserved_unix=1.)],1)['status']=='PASSED'
        assert c.stage_state([dict(exit=1,reserved_unix=1.)],1)['status']=='FAILED'
        from src.training import ua3d_ph_opt as t
        fake_recipe=dict(fake='synthetic'); monkeypatch.setattr(t,'recipe',lambda:fake_recipe)
        cfg=dict(plan='UA3D-PH-20261028-01',revision='r2',resource_revision='gpu2-pt2-fttest4-ph-r2',arm='M_EQ',fold=0,seed=42,
                 batch=24,epochs=70,head_epochs=10,precision='FP32',workers=0,world_size=1,physical_gpu=2,
                 recipe=fake_recipe,sidecar=str(t.SOURCE/'inputs'),input_cache=str(t.PARENT/'inputs'),numerical_policy='DET')
        t.alias_guard('M_EQ',cfg,dict(status='COMPLETE',completed=dict(ft_epochs=70)))
        with pytest.raises(ValueError):
            t.alias_guard('P_FIXED',dict(cfg,arm='P_FIXED'),dict(status='COMPLETE',completed=dict(ft_epochs=70)))
        with pytest.raises(ValueError):
            t.alias_guard('M_EQ',dict(cfg,batch=12),dict(status='COMPLETE',completed=dict(ft_epochs=70)))
        assert t.SPEC['limits']['ft_starts']==200 and t.SPEC['limits']['pt_starts']==0
        group('fresh_manifest_states_and_alias_schema_guard')
        state['status']='COMPLETE'; record()
    except BaseException:
        import traceback
        state.update(status='FAILED',error=traceback.format_exc()); record(); raise


def test_final():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.environ['PH_OPT_SUITE']=='final'
    from src.training import ua3d_ph_opt as t
    from scripts import run_ua3d_ph_opt as c
    folder=OUT/'final_cpu'; folder.mkdir(exist_ok=False)
    state=dict(status='RUNNING',cpu_suites=1); t.write(folder/'consumption.json',state)
    try:
        budget=t.read(c.BUDGET); js=budget['jobs']; matrix=t.read(OUT/'registry/matrix.json'); frozen=t.read(OUT/'selection/final_frozen.json')
        assert not budget['budget_violations'] and matrix['alias_count']==45 and len(matrix['new'])==200 and len(matrix['excluded_old_P'])==36
        assert sum(j['kind']=='ft' and j['exit']==0 for j in js)==200
        assert sum(j['kind']=='test' and j['exit']==0 for j in js)==245
        assert sum(j['kind']=='diagnostic' and j['exit']==0 for j in js)==28
        assert sum(j['kind']=='preflight' and j['exit']==0 for j in js)==1
        assert sum(j['kind']=='packing' and j['exit']==0 for j in js)==1
        assert sum(j['kind']=='cpu' for j in js)==1
        assert t.read(OUT/'gates/cost.json')['pass'] and t.read(OUT/'gates/pilot.json')['pass']
        assert len(frozen['units'])==245 and sum(u['role']=='inherited_readonly' for u in frozen['units'].values())==45
        assert frozen['FINAL']==t.select_final(frozen['selection']['validation'])
        assert t.read(OUT/'preflight_cpu/consumption.json')['status']=='COMPLETE'
        t.check_prepared()
        assert t.read(OUT/'packing/metadata.json')['new_graph_reductions']==0
        assert t.read(OUT/'packing/metadata.json')['new_vr_reductions']==0
        assert t.read(t.SOURCE/'controller_exit.continuation01.json')['exit']==1
        assert t.read(t.SOURCE/'execution_manifest.continuation01.json')['status']=='STOPPED'
        assert t.DEADLINE-t.T0==172800 and t.REFERENCE==.804675462276949
        intervals=[]
        for j in js:
            if j['kind'] not in c.GPU_KINDS:
                continue
            assert j['env']['CUDA_VISIBLE_DEVICES']=='2' and j['env']['WORLD_SIZE']=='1'
            process=t.read(Path(j['log'])/'process.json'); end=t.read(Path(j['log'])/'end.json')
            intervals.extend([(process['started_unix'],1,j['kind']),(end['ended_unix'],-1,j['kind'])])
            assert end['exit']==0 and end['ended_unix']<=t.DEADLINE-900
            if j['kind']=='ft':
                assert end['ended_unix']<=frozen['time_unix']
            if j['kind']=='test':
                assert process['started_unix']>=frozen['time_unix']
            env=t.read(Path(j['consumption']).parent/'environment.json')
            assert env['physical']['actual_uuid']==t.UUID and env['numerical']['deterministic'] and not env['numerical']['warn_only']
        active={}
        for when,change,kind in sorted(intervals):
            active[kind]=active.get(kind,0)+change
            assert active[kind]<=c.CAPS[kind] and sum(n>0 for n in active.values())<=1
        for arm in m.ARMS:
            a=t.aggregate(arm,True); assert len(a['tasks'])==7
        assert t.SOTA['egb'][2]==.923 and t.SOTA['egc'][2]==.911
        state['status']='COMPLETE'; t.write(folder/'consumption.json',state)
    except BaseException:
        import traceback
        state.update(status='FAILED',error=traceback.format_exc()); t.write(folder/'consumption.json',state); raise
