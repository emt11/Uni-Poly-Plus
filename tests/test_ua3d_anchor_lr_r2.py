"""One synthetic preflight and exactly one artifact-only final CPU suite."""
import ast
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from src.dataset import ua3d_anchor_lr_r2 as d
from src.modules import ua3d_anchor_lr_r2 as m

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/ua3d_anchor_lr_20261102/r1/repairs/init_r2'


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


def test_preflight(monkeypatch,tmp_path):
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.environ['ANCHOR_CPU_SUITE']=='preflight'
    folder=OUT/'preflight_cpu'; folder.mkdir(exist_ok=False)
    value=dict(status='RUNNING',preflight_cpu_suites=1,preflight_adapter_constructs=0,preflight_forwards=0,preflight_backwards=0,
        preflight_graph_reductions=0,preflight_vr_reductions=0,real_data_rows=0,full_base_constructs=0,full_base_forwards=0,optimizer_updates=0,groups_completed=[])
    limits=json.loads((ROOT/'configs/ua3d_anchor_lr_r2_gpu2.json').read_text())['limits']
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
        # First: real production metadata initialization/refresh and expected failure boundaries.
        import copy
        from scripts import bootstrap_ua3d_anchor_lr_r2 as boot
        activation=json.loads((OUT/'authorization/activation.json').read_text())
        spec=json.loads((ROOT/'configs/ua3d_anchor_lr_r2_gpu2.json').read_text())
        parent_activation=json.loads((OUT/'authorization/parent_r1/authorization/activation.json').read_text())
        stages={name:dict(status='NOT_RUN',attempts=0,exit=None,started=None,ended=None) for name in ['implementation','preflight','cache_trainval','diagnostic','cost_gate','pilot','ft','freeze','cache_test','anchor_output','adapter_output','final_cpu','delivery']}
        manifest=dict(status='TEST_METADATA_ONLY',stages=stages)
        root=tmp_path/'valid';log=tmp_path/'logs'
        runtime,empty=boot.initialize_metadata(root,log,activation,spec,parent_activation,manifest,[],dict(synthetic=True),dict(plan=activation['plan'],revision='r2'))
        assert empty['jobs']==[] and all(x==0 for x in empty['reserved'].values())
        empty2=runtime.refresh();assert empty2['attempted']==empty['attempted']
        cache=boot.read(root/'cache_budget.json');disk=boot.read(root/'artifact_budget.json')
        assert cache['new_teacher_cache_bytes']==cache['new_teacher_cache_units']==cache['new_encoder_batches']==cache['reused_bytes']==0
        assert cache['expected_reused_bytes']==activation['cache_reuse']['expected_reused_bytes']>0
        assert disk['estimated_new_artifact_bytes']==activation['estimated_new_artifact_bytes']<=8*1024**3 and disk['actual_new_artifact_bytes']>=0
        assert all(v['status']=='NOT_RUN' and v['exit'] is None for v in boot.read(root/'execution_manifest.json')['stages'].values())
        for key in ['estimated_new_artifact_bytes','cache_reuse','preparation_deadline_unix','repair_started_unix']:
            bad=copy.deepcopy(activation);del bad[key]
            with pytest.raises(ValueError,match=key):boot.validate_contract(bad,spec,parent_activation)
        for key in ['expected_reused_bytes','new_teacher_cache_bytes','new_encoder_batches']:
            bad=copy.deepcopy(activation);del bad['cache_reuse'][key]
            with pytest.raises(ValueError,match=key):boot.validate_contract(bad,spec,parent_activation)
        for bad in [dict(activation,tN_unix=activation['tN_unix']+1),dict(activation,deadline_unix=activation['deadline_unix']+1),dict(activation,estimated_new_artifact_bytes=-1),dict(activation,estimated_new_artifact_bytes=float('nan'))]:
            with pytest.raises(ValueError):boot.validate_contract(bad,spec,parent_activation)
        bad=copy.deepcopy(activation);del bad['estimated_new_artifact_bytes']
        def first_fake_runner():
            boot.validate_contract(bad,spec,parent_activation)
            raise AssertionError('missing field must fail')
        assert boot.run_wait(root,'synthetic_configuration',first_fake_runner)==1
        first=boot.read(root/'first_failure.json');assert 'estimated_new_artifact_bytes' in first['error'] and first['phase']=='synthetic_configuration'
        assert boot.read(root/'execution_manifest.json')['status']=='STOPPED'
        assert boot.read(root/'preflight_cpu_exit.json')==dict(attempts=0,exit=None,status='NOT_RUN')
        assert boot.read(root/'cpu_exit.json')==dict(attempts=0,exit=None,status='NOT_RUN')
        assert boot.run_wait(root,'synthetic_secondary_cleanup_error',lambda:(_ for _ in ()).throw(RuntimeError('secondary')))==1
        assert boot.read(root/'first_failure.json')==first and boot.read(root/'budget.json')==empty2
        fake=tmp_path/'nonzero_wait';assert boot.run_wait(fake,'fake_controller_wait',lambda:7)==7
        assert boot.run_wait(fake,'fake_outer_wait',lambda:7)==7
        assert boot.read(fake/'first_failure.json')['exit']==7
        assert not list(tmp_path.glob('**/process.json'))
        group('production_metadata_init_empty_refresh_strict_fields_first_failure_and_nonzero_fake_wait')
        torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(1701)
        records,sides=synthetic(); models={}
        for arm in m.TRAIN_ARMS:
            backend=m.backend(arm)
            frozen=MockBase(); model=m.AnchorModel(frozen,arm); model.train(); model.assert_frozen()
            assert all(not p.requires_grad for p in model.base.parameters()) and all(not x.training for x in model.base.modules())
            assert sum(p.numel() for p in model.adapter.parameters())==(36643 if m.backend(arm)=='G_GEOM' else 42820)
            batch=d.CachedBatch(records,arm,d.packed.batch_plan([d.packed.pack_side(s) for s in sides],m.backend(arm)) if m.backend(arm)!='G_GEOM' else None,sides if m.backend(arm)=='G_GEOM' else None).to('cpu')
            for _ in range(2):
                with torch.no_grad():
                    prediction=model.predict(batch)
                assert torch.equal(prediction,batch.y_base)
            expected_U=0 if m.backend(arm)=='G_GEOM' else 4
            assert model.cost['frozen_H_batches']==4 and model.cost['residual_U_batches']==expected_U and model.cost['full_encoder_batches']==0
            opt,sch=m.optimizer(model)
            assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in model.adapter.parameters()}
            assert opt.defaults['lr']==m.RATES[arm] and opt.defaults['weight_decay']==.0001 and opt.defaults['betas']==(.9,.999)
            source={k:v.detach().clone() for k,v in model.base.state_dict().items()}
            loss,parts=model.training_loss(batch); charge('preflight_backwards'); loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(parts['mse'])
            assert all(p.grad is None for p in model.base.parameters())
            output=model.adapter.up if m.backend(arm)=='G_GEOM' else model.adapter.out[-1]
            assert output.weight.grad is not None and output.weight.grad.abs().sum()>0
            # Move a synthetic output weight away from zero without an optimizer update.
            model.adapter.zero_grad(set_to_none=True)
            with torch.no_grad():
                # Distinct output channels avoid the LayerNorm-invariant constant direction.
                output.weight.copy_(torch.linspace(-.002,.002,output.weight.numel()).reshape_as(output.weight))
            loss,_=model.training_loss(batch); charge('preflight_backwards'); loss.backward()
            for name,p in model.adapter.named_parameters():
                if m.backend(arm)=='P_PE' and name.startswith('filter.'):
                    assert p.grad is None
                else:
                    assert p.grad is not None and torch.isfinite(p.grad).all(), (arm,name)
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
                    one=d.CachedBatch([records[i]],arm,d.packed.batch_plan([d.packed.pack_side(sides[i])],m.backend(arm)) if m.backend(arm)!='G_GEOM' else None,[sides[i]] if m.backend(arm)=='G_GEOM' else None)
                    singles.append(model.predict(one))
            assert torch.allclose(combined,torch.cat(singles),atol=1e-6,rtol=1e-5)
        group('batch_isolation_cached_prediction_offset_and_missing3D')
        frozen=MockBase(); before=torch.get_rng_state().clone(); model=m.AnchorModel(frozen,'P_PE_1E4'); assert torch.equal(before,torch.get_rng_state())
        model.base.head.train()
        with pytest.raises(ValueError):
            model.assert_frozen()
        model.train(); model.assert_frozen()
        group('isolated_initialization_RNG_and_eval_guard')
        from scripts import run_ua3d_anchor_lr_r2 as c
        assert c.state_for([],1)==dict(status='NOT_RUN',attempts=0,exit=None,started=None,ended=None)
        assert c.state_for([dict(exit=0,reserved_unix=0.)],1)['status']=='PASSED'
        assert c.state_for([dict(exit=1,reserved_unix=0.)],1)['status']=='FAILED'
        from src.training import ua3d_anchor_lr_r2 as t
        with monkeypatch.context() as local:
            local.setattr(t,'read',lambda p:dict(folds=[dict(train_indices=list(range(48)),validation_indices=list(range(24)),test_indices=list(range(24)))]))
            r=c.reservation('ft','P_PE_1E4','synthetic',0)
            assert r['ft_updates']==140 and r['frozen_H_batches']==420 and r['residual_U_batches']==420
        assert t.SPEC['limits']['frozen_H_batches']==568464 and t.SPEC['limits']['residual_U_batches']==284232
        assert t.SPEC['limits']['full_encoder_batches']==0 and t.SPEC['limits']['pt_starts']==0
        for n in t.SOURCES:
            ast.parse((ROOT/n).read_text()) if n.endswith('.py') else json.loads((ROOT/n).read_text())
        assert t.improves(.9,1.) and not t.improves(1.,1.) and not t.improves(1.1,1.)
        validation={arm:dict(macro=1.,tasks={task:dict(mean=1.) for task in t.TASKS}) for arm in m.ARMS}
        assert t.select_final(validation)=='BASE_KEEP'
        for arm in m.TRAIN_ARMS:validation[arm]['macro']=1.01
        assert t.select_final(validation)=='P_PE_1E4'
        assert tuple(m.RATES.values())==(.0001,.00003,.0001,.00003)
        assert t.SPEC['performance_threshold']==.809 and t.SPEC['target_threshold']==.810
        assert t.SPEC['limits']['p_learn_reductions']==0 and t.SPEC['limits']['new_features']==0
        group('schema_phase_counters_lr_epoch0_earliest_tie_fixed_budget_and_global_final')
        value['status']='COMPLETE'; record()
    except BaseException:
        import traceback
        value.update(status='FAILED',error=traceback.format_exc()); record(); raise


def test_final():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.environ['ANCHOR_CPU_SUITE']=='final'
    from src.training import ua3d_anchor_lr_r2 as t
    folder=OUT/'final_cpu'; folder.mkdir(exist_ok=False); value=dict(status='RUNNING',cpu_suites=1); t.write(folder/'consumption.json',value)
    try:
        b=t.read(OUT/'budget.json'); jobs=b['jobs']; frozen=t.read(OUT/'selection/final_frozen.json')
        assert not b['budget_violations'] and len(frozen['units'])==175
        assert sum(j['kind']=='test' and j['exit']==0 for j in jobs)==175
        assert sum(j['kind']=='ft' and j['exit']==0 for j in jobs)==140
        assert sum(j['kind']=='diagnostic' and j['exit']==0 for j in jobs)==16
        assert sum(j['kind']=='cache_trainval' and j['exit']==0 for j in jobs)==35
        assert sum(j['kind']=='cache_test' and j['exit']==0 for j in jobs)==35
        assert sum(j['kind']=='preflight' and j['exit']==0 for j in jobs)==1 and sum(j['kind']=='cpu' for j in jobs)==1
        assert frozen['FINAL']==t.select_final(frozen['validation'])
        assert t.read(OUT/'formal/anchor_gate.json')['pass'] is True
        assert t.read(OUT/'gates/pilot.json')['wrapper_elapsed_seconds']<=5400
        for key in ('frozen_H_batches','residual_U_batches','full_encoder_batches','encoder_U_batches','p_learn_reductions'):
            assert b['completed'][key]==t.SPEC['limits'][key],key
        assert b['completed']['adapter_test_forwards']==1080 and b['completed']['anchor_output_batches']==270
        assert b['attempted']['pt_starts']==b['attempted']['pt_updates']==0
        assert b['attempted']['cache_bytes']==0 and b['attempted']['p_learn_reductions']==0
        for key in ['ft_epochs','ft_updates','validation_count','validation_forwards','validation_rows','epoch0_validation_aliases','epoch0_validation_rows','diagnostic_updates','test_rows','test_batches']:
            assert b['completed'][key]==t.SPEC['limits'][key],key
        assert max(j['ended_unix'] for j in jobs if j['kind']=='ft')<=frozen['time_unix']
        for arm in m.TRAIN_ARMS:
            for task in t.TASKS:
                for fold in range(5):
                    folder=t.job_path('ft',arm,task,fold);met=t.read(folder/'metrics.json');evidence=t.read(folder/'training_state_evidence.json');hist=t.read(folder/'history.json')
                    assert 0<=met['best_epoch']<=70 and len(hist)==70 and evidence['trained_epochs']==70 and evidence['scheduler_steps']==70
                    assert evidence['actual_initial_lr']==m.RATES[arm] and hist[0]['learning_rates']==[m.RATES[arm]]
                    zero=torch.load(folder/'epoch0.pt',map_location='cpu',weights_only=False);best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
                    assert zero['config']==best['config']==last['config']==t.config(arm,task,fold)
                    assert zero['best_epoch']==0 and last['epoch']==70 and best['best_epoch']==met['best_epoch']
                    alias=t.read(folder/'epoch0_validation_alias.json');assert alias['forward']==0
                    values=[alias['metrics']['rmse']]+[x['validation']['rmse'] for x in hist]
                    assert met['best_epoch']==min(range(71),key=lambda i:values[i])
                    names=['up.weight','up.bias'] if m.backend(arm)=='G_GEOM' else ['out.2.weight','out.2.bias']
                    assert all(torch.count_nonzero(zero['adapter_state'][n])==0 for n in names)
                    assert all(not p.is_floating_point() or torch.isfinite(p).all() for p in best['adapter_state'].values())
        for kind in ('cache_trainval','cache_test'):
            for j in [j for j in jobs if j['kind']==kind]:
                assert not j['uses_gpu'] and j['env']['CUDA_VISIBLE_DEVICES']==''
                if kind=='cache_test':assert j['started_unix']>=frozen['time_unix']
        from scripts import bootstrap_ua3d_anchor_lr_r2 as boot
        a=boot.read(OUT/'authorization/activation.json');pa=boot.read(OUT/'authorization/parent_r1/authorization/activation.json')
        boot.validate_contract(a,t.SPEC,pa)
        reuse=boot.read(OUT/'cache_budget.json');disk=boot.read(OUT/'artifact_budget.json')
        assert reuse['new_teacher_cache_bytes']==reuse['new_teacher_cache_units']==reuse['new_encoder_batches']==0
        assert reuse['reused_bytes']>0 and disk['estimated_new_artifact_bytes']==a['estimated_new_artifact_bytes']
        assert a['tN_unix']==pa['tN_unix'] and a['deadline_unix']==pa['deadline_unix']
        parent=Path(a['parent_failure_root'])
        for n in ['controller_exit.json','outer_launcher_exit.json','bootstrap_exit.json']:
            assert boot.read(parent/n)['exit']==1
        assert boot.read(parent/'execution_manifest.json')['status']=='STOPPED'
        assert boot.read(parent/'preflight_cpu_exit.json')['attempts']==boot.read(parent/'cpu_exit.json')['attempts']==0
        assert boot.read(parent/'budget.json')['jobs']==[]
        for old in ['src/dataset/ua3d_anchor_lr_r1.py','src/modules/ua3d_anchor_lr_r1.py','src/training/ua3d_anchor_lr_r1.py','scripts/prepare_ua3d_anchor_lr_r1.py','scripts/run_ua3d_anchor_lr_r1.py','configs/ua3d_anchor_lr_r1_gpu2.json','tests/test_ua3d_anchor_lr_r1.py']:
            assert (ROOT/old).read_bytes()==(OUT/'authorization/source_before'/old).read_bytes()
        assert not (OUT/'first_failure.json').exists()
        t.source_lock()
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
            for task in t.TASKS:
                for fold in range(5):
                    folder=t.job_path('test',arm,task,fold);unit=frozen['units'][f'{arm}/{task}/{fold}'];met=t.read(folder/'metrics.json')
                    with np.load(folder/'test_predictions.npz',allow_pickle=False) as p:
                        assert p['indices'].tolist()==unit['test_indices'] and p['sample_keys'].tolist()==unit['test_keys']
                        assert np.isfinite(p['y_true']).all() and np.isfinite(p['y_pred']).all()
                        assert t.metrics(p['y_true'],p['y_pred'])['r2']==met['r2']
        assert max(j['ended_unix'] for j in jobs if j['kind']=='test')<=min(j['started_unix'] for j in jobs if j['kind']=='cpu')
        assert t.REFERENCE==.804675462276949 and t.SOTA['egb'][2]==.923 and t.SOTA['egc'][2]==.911
        value['status']='COMPLETE'; t.write(folder/'consumption.json',value)
    except BaseException:
        import traceback
        value.update(status='FAILED',error=traceback.format_exc()); t.write(folder/'consumption.json',value); raise
