"""One synthetic preflight and exactly one artifact-only final CPU suite."""
import ast
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from src.dataset import ua3d_anchor_r2 as d
from src.modules import ua3d_anchor_r2 as m

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/ua3d_anchor_opt_20261102/r2'


class MockBase(nn.Module):
    """Small interface mock; no native encoder or pretrained checkpoint."""
    def __init__(self):
        super().__init__(); self.norm2=nn.LayerNorm(512); self.A2=nn.Linear(512,256)
        self.gate=nn.Linear(768,3); self.head=nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Dropout(.5),nn.Linear(256,1))
    def fuse(self,z,available):
        w=self.gate(torch.cat(z,-1)).masked_fill(~available,-torch.inf).softmax(-1)
        self.last_z=z
        return (torch.stack(z,1)*w[:,:,None]).sum(1),w
    def encode_modalities(self,*args,**kwargs):
        raise AssertionError('encoder call forbidden')


def synthetic():
    from src.dataset import ua3d_ph_r2 as old
    sides=[]; records=[]
    for i,edges in enumerate(([(0,1),(1,2)],[(0,1),(1,2),(2,0)])):
        q=torch.as_tensor(old.random_walk_pe(3,np.asarray(edges)))
        side=dict(rwpe=q,edges=np.asarray(edges),center=torch.arange(3),nodes=3,graph_available=True,
            events=[old.graph_events(q[:,k].numpy(),edges) for k in (1,3,7,15)],available=i==0,
            physical_atoms=3,bond_atoms=torch.tensor([[0,1],[1,2]]),incidence=torch.tensor([[0,0,1,1],[0,1,1,2]]),
            canonical_graph=torch.zeros(3,dtype=torch.long),sp_edges=[torch.tensor([[0,1,1,2],[1,0,2,1]])]*3,
            sp_distances=[torch.ones(4)]*3,router=torch.linspace(-1,1,155).reshape(31,5))
        sides.append(side)
        records.append(dict(z=[torch.randn(256),torch.randn(256) if i==0 else torch.zeros(256),torch.randn(256)],atoms=torch.randn(3,512),
            bonds=torch.randn(2,512) if i==0 else None,availability=torch.tensor([True,i==0,True]),
            y=torch.tensor(.6+i),y_base=torch.tensor(-.8+i),key='synthetic'+str(i),row=i))
    return records,sides


def test_preflight(monkeypatch):
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.environ['ANCHOR_CPU_SUITE']=='preflight'
    folder=OUT/'preflight_cpu'; folder.mkdir(exist_ok=False)
    value=dict(status='RUNNING',preflight_cpu_suites=1,preflight_adapter_constructs=0,preflight_forwards=0,preflight_backwards=0,
        preflight_graph_reductions=0,preflight_vr_reductions=0,real_data_rows=0,full_base_constructs=0,full_base_forwards=0,optimizer_updates=0,groups_completed=[])
    limits=json.loads((ROOT/'configs/ua3d_anchor_r2_gpu2.json').read_text())['limits']
    def record():
        p=folder/'consumption.json'; q=p.with_suffix('.tmp'); q.write_text(json.dumps(value,indent=2)+'\n'); q.replace(p)
    def charge(k):
        value[k]+=1; record(); assert value[k]<=limits[k]
    def group(n):
        value['groups_completed'].append(n); record()
    def forbidden(*args,**kwargs):
        raise AssertionError('preflight forbids full BASE, VR, real data and optimizer updates')
    monkeypatch.setattr(m.base.BaseOptFuse,'__init__',forbidden)
    old=d.spatial.graph_events
    def events(*args,**kwargs):
        charge('preflight_graph_reductions'); return old(*args,**kwargs)
    monkeypatch.setattr(d.spatial,'graph_events',events); monkeypatch.setattr(d.spatial,'vr_betti',forbidden)
    for cls in (m.ph.old.GraphPHAdapter,m.SpatialRouter):
        init=cls.__init__
        def construct(self,*args,_init=init,**kwargs):
            charge('preflight_adapter_constructs'); _init(self,*args,**kwargs)
        monkeypatch.setattr(cls,'__init__',construct)
    for cls,name in [(m.AnchorModel,'predict'),(m.AnchorModel,'H'),(m.AnchorModel,'U'),(m.ph.GraphPHAdapter,'forward_batch'),(m.SpatialRouter,'forward')]:
        old_=getattr(cls,name)
        def counted(self,*args,_call=old_,**kwargs):
            charge('preflight_forwards'); return _call(self,*args,**kwargs)
        monkeypatch.setattr(cls,name,counted)
    record()
    try:
        torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(1701)
        records,sides=synthetic(); models={}
        for arm in m.TRAIN_ARMS:
            frozen=MockBase(); model=m.AnchorModel(frozen,arm); model.train(); model.assert_frozen()
            assert all(not p.requires_grad for p in model.base.parameters()) and all(not x.training for x in model.base.modules())
            assert sum(p.numel() for p in model.adapter.parameters())==(36643 if arm=='G_GEOM' else 42820)
            batch=d.CachedBatch(records,arm,d.packed.batch_plan([d.packed.pack_side(s) for s in sides],arm) if arm!='G_GEOM' else None,sides if arm=='G_GEOM' else None).to('cpu')
            for _ in range(2):
                with torch.no_grad():
                    prediction=model.predict(batch)
                assert torch.equal(prediction,batch.y_base)
            expected_U=0 if arm=='G_GEOM' else 4
            assert model.cost['frozen_H_batches']==4 and model.cost['residual_U_batches']==expected_U and model.cost['full_encoder_batches']==0
            opt,sch=m.optimizer(model)
            assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in model.adapter.parameters()}
            assert opt.defaults['lr']==.001 and opt.defaults['weight_decay']==.0001 and opt.defaults['betas']==(.9,.999)
            source={k:v.detach().clone() for k,v in model.base.state_dict().items()}
            loss,parts=model.training_loss(batch); charge('preflight_backwards'); loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(parts['mse'])
            assert all(p.grad is None for p in model.base.parameters())
            output=model.adapter.up if arm=='G_GEOM' else model.adapter.out[-1]
            assert output.weight.grad is not None and output.weight.grad.abs().sum()>0
            # Move a synthetic output weight away from zero without an optimizer update.
            model.adapter.zero_grad(set_to_none=True)
            with torch.no_grad():
                # Distinct output channels avoid the LayerNorm-invariant constant direction.
                output.weight.copy_(torch.linspace(-.002,.002,output.weight.numel()).reshape_as(output.weight))
            loss,_=model.training_loss(batch); charge('preflight_backwards'); loss.backward()
            for name,p in model.adapter.named_parameters():
                if arm=='P_PE' and name.startswith('filter.'):
                    assert p.grad is None
                else:
                    assert p.grad is not None and torch.isfinite(p.grad).all(), (arm,name)
            if arm=='P_LEARN':
                assert sum(float(p.grad.abs().sum()) for p in model.adapter.filter.parameters())>0
            m.verify_unchanged(model,source)
            assert model.base.last_z[1][1].count_nonzero()==0
            assert torch.equal(batch.z[1][1],torch.zeros_like(batch.z[1][1]))
            models[arm]=(model,batch)
        group('all_arms_frozen_eval_exact_zero_anchor_gradient_optimizer_only_adapter')
        # Same-batch differences isolate original cached z and y from recomputation drift.
        for arm,(model,batch) in models.items():
            model.eval()
            with torch.no_grad():
                combined=model.predict(batch)
                singles=[]
                for i in range(2):
                    one=d.CachedBatch([records[i]],arm,d.packed.batch_plan([d.packed.pack_side(sides[i])],arm) if arm!='G_GEOM' else None,[sides[i]] if arm=='G_GEOM' else None)
                    singles.append(model.predict(one))
            assert torch.allclose(combined,torch.cat(singles),atol=1e-6,rtol=1e-5)
        group('batch_isolation_cached_prediction_offset_and_missing3D')
        frozen=MockBase(); before=torch.get_rng_state().clone(); model=m.AnchorModel(frozen,'P_PE'); assert torch.equal(before,torch.get_rng_state())
        model.base.head.train()
        with pytest.raises(ValueError):
            model.assert_frozen()
        model.train(); model.assert_frozen()
        group('isolated_initialization_RNG_and_eval_guard')
        from scripts import run_ua3d_anchor_r2 as c
        assert c.state_for([],1)==dict(status='NOT_RUN',attempts=0,exit=None,started=None,ended=None)
        assert c.state_for([dict(exit=0,reserved_unix=0.)],1)['status']=='PASSED'
        assert c.state_for([dict(exit=1,reserved_unix=0.)],1)['status']=='FAILED'
        from src.training import ua3d_anchor_r2 as t
        with monkeypatch.context() as local:
            local.setattr(t,'read',lambda p:dict(folds=[dict(train_indices=list(range(48)),validation_indices=list(range(24)),test_indices=list(range(24)))]))
            r=c.reservation('ft','P_LEARN','synthetic',0)
            assert r['ft_updates']==140 and r['frozen_H_batches']==420 and r['residual_U_batches']==420
        assert t.SPEC['limits']['frozen_H_batches']==427628 and t.SPEC['limits']['residual_U_batches']==284232
        assert t.SPEC['limits']['full_encoder_batches']==1280 and t.SPEC['limits']['pt_starts']==0
        for n in t.SOURCES:
            ast.parse((ROOT/n).read_text()) if n.endswith('.py') else json.loads((ROOT/n).read_text())
        group('schema_phase_counters_and_fixed_budget')
        value['status']='COMPLETE'; record()
    except BaseException:
        import traceback
        value.update(status='FAILED',error=traceback.format_exc()); record(); raise


def test_final():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.environ['ANCHOR_CPU_SUITE']=='final'
    from src.training import ua3d_anchor_r2 as t
    folder=OUT/'final_cpu'; folder.mkdir(exist_ok=False); value=dict(status='RUNNING',cpu_suites=1); t.write(folder/'consumption.json',value)
    try:
        b=t.read(OUT/'budget.json'); jobs=b['jobs']; frozen=t.read(OUT/'selection/final_frozen.json')
        assert not b['budget_violations'] and len(frozen['units'])==140
        assert sum(j['kind']=='test' and j['exit']==0 for j in jobs)==140
        assert sum(j['kind']=='ft' and j['exit']==0 for j in jobs)==105
        assert sum(j['kind']=='diagnostic' and j['exit']==0 for j in jobs)==12
        assert sum(j['kind']=='cache_trainval' and j['exit']==0 for j in jobs)==35
        assert sum(j['kind']=='cache_test' and j['exit']==0 for j in jobs)==35
        assert sum(j['kind']=='preflight' and j['exit']==0 for j in jobs)==1 and sum(j['kind']=='cpu' for j in jobs)==1
        assert frozen['FINAL']==t.select_final(frozen['validation'])
        assert t.read(OUT/'formal/anchor_gate.json')['pass'] is True
        assert t.read(OUT/'gates/pilot.json')['wrapper_elapsed_seconds']<=3600
        for key in ('frozen_H_batches','residual_U_batches','full_encoder_batches','encoder_U_batches','p_learn_reductions'):
            assert b['completed'][key]==t.SPEC['limits'][key],key
        assert b['completed']['adapter_test_forwards']==810 and b['completed']['anchor_output_batches']==270
        assert b['attempted']['pt_starts']==b['attempted']['pt_updates']==0
        assert b['attempted']['cache_bytes']<=64*1024**3
        intervals=[]
        for j in jobs:
            if not j['uses_gpu']:
                continue
            assert j['env']['CUDA_VISIBLE_DEVICES']=='2' and j['env']['WORLD_SIZE']=='1'
            p=t.read(Path(j['log'])/'process.json'); end=t.read(Path(j['log'])/'end.json')
            intervals.extend([(p['started_unix'],1,j['kind']),(end['ended_unix'],-1,j['kind'])])
            assert end['exit']==0
            if j['kind']=='cache_test':
                assert p['started_unix']>=frozen['time_unix']
            env=t.read(Path(j['consumption']).parent/'environment.json')
            assert env['numerical']['deterministic'] and not env['numerical']['warn_only']
        active={}; caps=t.SPEC['caps']
        for when,change,kind in sorted(intervals):
            active[kind]=active.get(kind,0)+change
            assert active[kind]<=caps[kind] and sum(v>0 for v in active.values())<=1
        for arm in m.ARMS:
            assert len(t.aggregate(arm,True)['tasks'])==7
        assert t.REFERENCE==.804675462276949 and t.SOTA['egb'][2]==.923 and t.SOTA['egc'][2]==.911
        value['status']='COMPLETE'; t.write(folder/'consumption.json',value)
    except BaseException:
        import traceback
        value.update(status='FAILED',error=traceback.format_exc()); t.write(folder/'consumption.json',value); raise
