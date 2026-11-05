"""Exactly one preflight (synthetic only), and one read-only final CPU suite."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
COUNTS=dict(meshes=0,constructs=0,forwards=0,backwards=0,optimizer_updates=0,real_labels=0,encoder_forwards=0)

def preflight():
    import numpy as np
    import torch
    from src.dataset import ua3d_hmr_r2 as d
    from src.modules import ua3d_hmr_r2 as m
    from src.training import ua3d_hmr_r2 as t
    from scripts.bootstrap_ua3d_hmr_r2 import parse_jobs,state_for,write,read,fail
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    class Contract(unittest.TestCase):
        def test_runtime(self):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);write(root/'authorization/clock.json',dict(tN=time.time()))
                class Fake:
                    pid=os.getpid()
                    def poll(self):return 0
                    def wait(self):return 0
                c=t.Controller(root,root/'logs',runner=lambda *a,**kw:Fake());self.assertEqual(c.refresh()['counts'],{});c.dispatch('cpu');running=state_for(c.jobs,1);self.assertEqual(running['status'],'RUNNING');self.assertIsNone(c.jobs[0]['ended_unix']);c.poll();self.assertEqual(state_for(c.jobs,1)['status'],'PASSED');self.assertEqual(c.jobs[0]['exit'],0)
                with self.assertRaises(ValueError):c.dispatch('cpu')
                bad=dict(c.jobs[0]);del bad['started_unix']
                with self.assertRaises(ValueError):parse_jobs([bad])
                with patch('scripts.bootstrap_ua3d_hmr_r2.OUT',root):
                    fail('synthetic','first');fail('synthetic','second');self.assertEqual(read(root/'first_failure.json')['error'],'first')
        def test_surface(self):
            v=np.array([[1,1,1],[-1,-1,1],[-1,1,-1],[1,-1,-1]],dtype=float);f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
            COUNTS['meshes']+=1;L,mass,n,H,K=d.operators(v,f);lam,phi,audit=d.spectrum(L,mass);self.assertGreaterEqual(lam.min(),0);self.assertLessEqual(audit['mass_orthogonality'],1e-5);F=np.arange(len(v)*16).reshape(len(v),16)/30;w=np.array([1.,0.,1.,0.]);C,a=d.moments(lam,phi,mass,F,w)
            signs=np.where(np.arange(len(lam))%2,1,-1);C2,a2=d.moments(lam,phi*signs,mass,F,w);self.assertTrue(np.allclose(a[:,None]*C,a2[:,None]*C2))
            # Exact repeated-eigenspace orthogonal changes preserve the two readouts.
            q,_=np.linalg.qr(np.arange(1,17,dtype=float).reshape(4,4)+np.eye(4));p=np.eye(4)/np.sqrt(mass[:,None]);c,b=d.moments(np.zeros(4),p,mass,F,w);cc,bb=d.moments(np.zeros(4),p@q,mass,F,w);self.assertTrue(np.allclose(b@c,bb@cc));self.assertTrue(np.allclose((c*c).sum(0),(cc*cc).sum(0)))
            COUNTS['meshes']+=1
            with self.assertRaises(d.Unavailable):d.operators(v,f[:-1])
            COUNTS['meshes']+=1;s=d.surface(np.array([[0.,0.,0.]]),np.array([6]),np.array([0]),{'6':1.7});self.assertTrue(s['available']);self.assertEqual(s['C'].shape,(64,16));self.assertTrue(np.isfinite(d.fixed_descriptor(s)).all())
            x=d.sentinel('GRID_CAP');self.assertTrue(np.isfinite(d.fixed_descriptor(x)).all());self.assertEqual(x['A'],1);self.assertFalse(x['mode_mask'].any())
        def test_models(self):
            stats={k:([0.]*n if k.endswith('mean') else [1.]*n) for k,n in [('z_mean',768),('z_scale',768),('stat_mean',16),('stat_scale',16),('spec_mean',128),('spec_scale',128)]}
            torch.manual_seed(42);b=dict(z=torch.randn(5,768),availability=torch.ones(5,3),yb=torch.randn(5),y=torch.randn(5),lam=torch.linspace(0,5,64).repeat(5,1),C=torch.randn(5,64,16),a=torch.randn(5,64),A=torch.ones(5)*10,Ac=torch.ones(5)*5,stat=torch.randn(5,16),mode_mask=torch.ones(5,64),mask=torch.tensor([1.,0.,1.,0.,1.]))
            for k in ('lam','C','a','stat','mode_mask'):b[k][~b['mask'].bool()]=0
            for arm in m.TRAIN_ARMS:
                torch.manual_seed(42);model=m.Residual(arm,stats);COUNTS['constructs']+=1;model.eval();COUNTS['forwards']+=1;out=model(b);self.assertTrue(torch.equal(out,b['yb']));self.assertEqual(sum(p.numel() for n,p in model.named_parameters() if n=='filter_raw'),12)
                out.sum().backward();COUNTS['backwards']+=1;self.assertTrue(torch.count_nonzero(model.head[-1].weight.grad)>0)
                with torch.no_grad():model.head[-1].weight.fill_(.1)
                COUNTS['forwards']+=1;out=model(b);self.assertTrue(torch.equal(out[~b['mask'].bool()],b['yb'][~b['mask'].bool()]));self.assertTrue(torch.isfinite(out).all())
                model.zero_grad();out.sum().backward();COUNTS['backwards']+=1
                if arm=='S_HMR':self.assertTrue(torch.count_nonzero(model.filter_raw.grad)>0)
            self.assertEqual(t.choose({a:dict(macro=1.,tasks={task:dict(mean=1.) for task in t.TASKS}) for a in m.ARMS}),'BASE_KEEP')
        def test_train_stats(self):
            cache=dict(indices=torch.tensor([0,1,2]),z=torch.tensor([[1.]*768,[3.]*768,[999.]*768]));surface=dict(mask=torch.ones(3),stat=torch.tensor([[2.]*16,[4.]*16,[999.]*16]),lam=torch.zeros(3,64),C=torch.zeros(3,64,16),a=torch.zeros(3,64),A=torch.ones(3),Ac=torch.ones(3),mode_mask=torch.zeros(3,64))
            stats=m.fit_stats(cache,surface,[0,1]);self.assertEqual(stats['z_mean'],[2.]*768);self.assertEqual(stats['stat_mean'],[3.]*16);self.assertEqual(stats['spec_scale'],[1.]*128);self.assertEqual(stats['train_rows'],[0,1])
        def test_mapping(self):
            z=np.array([6,6,6,8]);ru=np.array([-1,-1,0,0]);center=np.array([2,-1,3,-1]);original=center.copy()
            self.assertEqual(d.validate_center(center,z,ru,2,np.array([6,1,8,1])).tolist(),[2,3]);self.assertTrue(np.array_equal(center,original))
            for bad,missing in [(np.array([2,-2,3,-1]),1),(np.array([2,2]),0),(np.array([4]),0),(np.array([0]),0),(center,1)]:
                with self.assertRaises(ValueError):d.validate_center(bad,z,ru,missing)
            with self.assertRaises(ValueError):d.validate_center(center,z,ru,2,np.array([6,6,8,1]))
            self.assertEqual(d.validate_center(np.array([-1,-1]),z,ru,2,np.array([1,1])).tolist(),[])
        def test_multicomponent(self):
            COUNTS['meshes']+=1
            s=d.surface(np.array([[0.,0.,0.],[10.,0.,0.]]),np.array([6,8]),np.array([0]),{'6':1.7,'8':2.5})
            self.assertEqual(s['audit']['components'],2);self.assertEqual(s['audit']['zero_modes'],2);self.assertLessEqual(int(s['mode_mask'].sum()),64);self.assertEqual(len(np.unique(s['component'])),2)
            self.assertLess(s['audit']['component_areas'][s['component'][s['owner']==0][0]],s['A']);self.assertGreater(s['Ac'],0);self.assertLess(s['Ac'],s['A']/2)
            self.assertTrue(np.array_equal(s['lam'][:2],np.zeros(2)));self.assertTrue(np.allclose(s['phi'].T@(s['mass'][:,None]*s['phi']),np.eye(s['phi'].shape[1]),atol=1e-5))
            # Arbitrary orthogonal mixing of the complete repeated-zero subspace.
            angle=.37;Q=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);F=np.arange(len(s['mass'])*16,dtype=float).reshape(-1,16)/1000;w=(s['owner']==0).astype(float);C,a=d.moments(s['lam'],s['phi'],s['mass'],F,w);phi=s['phi'].copy();phi[:,:2]=phi[:,:2]@Q;C2,a2=d.moments(s['lam'],phi,s['mass'],F,w)
            def compact(C,a):return dict(lam=s['lam'],C=np.pad(C,((0,64-len(a)),(0,0))),a=np.pad(a,(0,64-len(a))),A=s['A'],Ac=s['Ac'],mode_mask=s['mode_mask'])
            self.assertTrue(np.allclose(d.fixed_descriptor(compact(C,a)),d.fixed_descriptor(compact(C2,a2)),atol=1e-8,rtol=1e-10));self.assertEqual(d.RHO0,1.000001);self.assertEqual(d.TAU,.15);self.assertEqual(d.TAIL_EPS,1e-10)
        def test_entry(self):
            import subprocess
            evidence=[]
            for outcome in ('success','failure'):
                with tempfile.TemporaryDirectory() as temp:
                    root=Path(temp);(root/'SYNTHETIC_ONLY').write_text('synthetic fixture');env=dict(os.environ);env.pop('PYTHONPATH',None);cmd=[sys.executable,str(ROOT/'scripts/bootstrap_ua3d_hmr_r2.py'),'--synthetic-root',str(root),'--synthetic-outcome',outcome];r=subprocess.run(cmd,cwd='/',env=env,capture_output=True,text=True,timeout=120);expected=0 if outcome=='success' else 1;self.assertEqual(r.returncode,expected,r.stderr);self.assertEqual(read(root/'bootstrap_exit.json')['exit'],expected);self.assertTrue((root/'delivery_summary.md').exists())
                    if outcome=='success':self.assertTrue(read(root/'synthetic_success.json')['import_path_ok']);self.assertEqual(read(root/'synthetic_success.json')['real_jobs'],0)
                    else:self.assertIn('synthetic injected first error',read(root/'first_failure.json')['error']);self.assertEqual(read(root/'final_review/results.json')['status'],'STOPPED')
                    evidence.append(dict(outcome=outcome,argv=cmd,cwd='/',exit=r.returncode,current_process_import_ok=True,standard_library_report=True))
            t.write(t.OUT/'preflight/entry_evidence.json',evidence)
        def test_optimizer_epoch0(self):
            class Tiny(torch.nn.Module):
                def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1))
            model=Tiny();COUNTS['constructs']+=1;opt,sch=t.optimizer_for(model);self.assertEqual(opt.param_groups[0]['lr'],.0003);self.assertEqual(opt.param_groups[0]['weight_decay'],.0001);self.assertEqual(sch.T_max,70);self.assertFalse(t.better({'rmse':1.},{'rmse':1.}));self.assertTrue(t.better({'rmse':.9},{'rmse':1.}));self.assertEqual(COUNTS['optimizer_updates'],0)
        def test_cache(self):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);parent=root/'parent';base=root/'base';split=root/'splits';out=root/'out';best=base/'best.pt';best.parent.mkdir();best.write_bytes(b'synthetic')
                scaler=dict(mean=[0.],scale=[1.],fit='train',transform='standard');anchor=dict(best=str(best),source_config={},schema='synthetic',best_epoch=0,best_stage='synthetic',scaler=scaler,validation_predictions=str(base/'validation_predictions.npz'))
                source=parent/'trainval/eps/fold0';source.mkdir(parents=True);rows=[dict(row=i,key=f'k{i}',available=True) for i in range(4)];partitions={'train':[0,1],'validation':[2,3]}
                write(out/'registry/anchors.json',dict(rows={'eps':rows}));write(out/'authorization/clock.json',dict(deadline_unix=time.time()+86400,prepare_deadline_unix=time.time()+28800));write(split/'eps.json',dict(folds=[dict(train_indices=[0,1],validation_indices=[2,3],test_indices=[])]))
                write(source/'metadata.json',dict(schema='ua3d-anchor-feature-cache-r2',status='COMPLETE',task='eps',fold=0,formal=False,source_best=str(best),source_config={},source_schema='synthetic',source_best_epoch=0,source_best_stage='synthetic',scaler=scaler,partitions=partitions));write(source/'consumption.json',dict(status='COMPLETE',started_unix=time.time()+1));write(source/'validation_gate.json',dict(identity_equal=True,prediction_allclose=True,r2_delta=0,scaler=scaler))
                for part,ids in partitions.items():
                    p=dict(indices=np.array(ids),sample_keys=np.array([f'k{i}' for i in ids]),y_true=np.array(ids,dtype=float),y_pred=np.array(ids,dtype=float));np.savez(source/f'{part}_predictions.npz',**p)
                    if part=='validation':np.savez(base/'validation_predictions.npz',**p)
                    for i in ids:torch.save(dict(schema='ua3d-anchor-feature-cache-r2',task='eps',fold=0,row=i,key=f'k{i}',partition=part,source_best=str(best),z=[torch.ones(256) for _ in range(3)],availability=torch.ones(3,dtype=torch.bool),y=torch.tensor(float(i)),y_base=torch.tensor(float(i)),y_base_original=float(i)),source/f'row{i}.pt')
                with patch.multiple(t,OUT=out,PARENT=parent,BASE=base,SPLIT=split),patch.object(t,'anchor',lambda *a:anchor),patch.object(t,'check_clock',lambda *a:None):
                    t.pack_cache('eps',0,False);cache=t.load_cache('eps',0,False);self.assertEqual(cache['z'].shape,(4,768));self.assertEqual(cache['indices'].tolist(),[0,1,2,3]);self.assertEqual(read(out/'registry/cache_qualification/trainval/eps/fold0.json')['encoder_forwards'],0)
                    with self.assertRaises(ValueError):t.cache_path('eps',0,True)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contract));COUNTS['tests_run']=result.testsRun
    assert COUNTS['meshes']<=12 and COUNTS['constructs']<=24 and COUNTS['forwards']<=128 and COUNTS['backwards']<=32
    t.write(t.OUT/'preflight/consumption.json',dict(status='COMPLETE' if result.wasSuccessful() else 'FAILED',attempted=COUNTS,completed=COUNTS if result.wasSuccessful() else {},started_unix=STARTED,ended_unix=time.time()));return 0 if result.wasSuccessful() else 1

def final():
    import numpy as np
    from src.training import ua3d_hmr_r2 as t
    from scripts.bootstrap_ua3d_hmr_r2 import read,write,parse_jobs,state_for,SOURCES
    budget=read(t.OUT/'budget.json');jobs=budget['jobs'];parse_jobs(jobs);counts=budget['counts'];expected={'preflight':1,'prepare':1,'diagnostic':12,'ft':105,'test':140}
    for kind,n in expected.items():assert counts[kind]['completed']==n and counts[kind]['reserved']==n,(kind,counts[kind])
    assert counts['cpu']['reserved']==1 and counts['cpu']['started']==1 and counts['cpu']['completed']==0
    assert state_for([j for j in jobs if j['kind']=='cpu'],1)['status']=='RUNNING'
    totals={}
    for j in jobs:
        if j['kind'] not in ('diagnostic','ft','test'):continue
        u=read(j['consumption']);assert u['status']=='COMPLETE' and u['attempted']==u['completed'];target=totals.setdefault(j['kind'],{})
        for k,v in u['completed'].items():target[k]=target.get(k,0)+v
        proc=read(Path(j['log'])/'process.json');assert proc['CVD']==('2' if j['gpu'] else '') and proc['env']['WORLD_SIZE']=='1'
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):assert proc['env'][name]=='1'
        assert read(Path(j['log'])/'wait_exit.json')['exit']==0 and read(Path(j['log'])/'end.json')['exit']==0
    for k,v in dict(updates=240,train_rows=38400,train_forwards=240,backwards=240,eval_forwards=24,eval_rows=3840).items():assert totals['diagnostic'][k]==v,(k,totals['diagnostic'])
    for k,v in dict(updates=17850,epochs=7350,train_forwards=17850,backwards=17850,train_rows=3945060,validation=7350,validation_batches=9450,validation_rows=989940,epoch0_aliases=105,epoch0_alias_rows=14142).items():assert totals['ft'][k]==v,(k,totals['ft'])
    for k,v in dict(test_rows=23500,test_batches=180,candidate_forwards=135,anchor_batches=45).items():assert totals['test'][k]==v,(k,totals['test'])
    frozen=read(t.OUT/'selection/final_frozen.json');assert frozen['frozen'] and len(frozen['estimators'])==140;assert frozen['frozen_unix']<min(j['started_unix'] for j in jobs if j['kind']=='test');assert frozen['FINAL']==t.choose(frozen['validation'])
    for arm in t.m.TRAIN_ARMS:
        for task in t.TASKS:
            for fold in range(5):
                folder=t.job_folder('ft',arm,task,fold);h=read(folder/'history.json');assert len(h)==70 and [x['epoch'] for x in h]==list(range(1,71));values=[frozen['validation']['C_ANCHOR']['tasks'][task]['folds'][fold]]
                assert all((folder/f'{n}.pt').exists() for n in ('epoch0','best','last'))
                best=read(folder/'metrics.json');assert best['best_epoch']==frozen['estimators'][f'{arm}/{task}/{fold}']['epoch'];assert 0<=best['best_epoch']<=70
    result=t.scientific_results();assert result['formal_complete'];assert abs(result['formal']['C_ANCHOR']['macro']-t.REFERENCE)<=1e-6
    for arm in t.m.ARMS:
        for task in t.TASKS:
            for fold in range(5):
                folder=t.job_folder('test',arm,task,fold);met=read(folder/'metrics.json');e=read(t.SPLIT/f'{task}.json')['folds'][fold];mapping={r['row']:r for r in read(t.OUT/'registry/anchors.json')['rows'][task]}
                with np.load(folder/'test_predictions.npz',allow_pickle=False) as p:
                    assert p['indices'].tolist()==sorted(e['test_indices']);assert p['sample_keys'].tolist()==[mapping[r]['key'] for r in p['indices'].tolist()];computed=t.metrics(p['y_true'],p['y_pred']);assert computed==met['metrics']
    coverage=read(t.OUT/'surface/coverage.json');assert coverage['rows']==5875 and coverage['source_valid']==5860 and coverage['source_missing']==15 and coverage['overall']>=.99;assert all(v['rate']>=.95 for v in coverage['tasks'].values())
    mapping=read(t.OUT/'surface/mapping.json');assert len(mapping)==5875;status=read(t.OUT/'surface/status.json');assert len(status)==5875
    for s in status:
        assert s['available'] or s['reason'] in t.d.REASONS|{'SOURCE_MISSING_3D'}
    rotation=read(t.OUT/'surface/rotation.json');assert len(rotation)==4 and all(v['area']<=.05 and v['volume']<=.05 and v['descriptor_relative_L2']<=.1 for v in rotation)
    for s in (t.OUT/'surface/structures').iterdir():
        r=read(s/'status.json')
        if r['available']:assert r['audit']['residual']<=1e-5 and r['audit']['mass_orthogonality']<=1e-5 and (s/'audit.npz').exists()
    for stage in ('diagnostic','ft','test'):
        js=[j for j in jobs if j['kind']==stage and j['gpu']];events=sorted([(j['started_unix'],1) for j in js]+[(j['ended_unix'],-1) for j in js]);active=0;peak=0
        for _,delta in events:active+=delta;peak=max(peak,active)
        assert peak<=(1 if stage=='diagnostic' else 8)
        for a in js:
            for b in jobs:
                if b['gpu'] and b['kind']!=stage:assert a['ended_unix']<=b['started_unix'] or b['ended_unix']<=a['started_unix']
    assert t.size(t.OUT/'surface')<=48*1024**3 and t.size(t.OUT/'feature_cache')<=1024**3 and t.size(t.OUT)+t.size(t.LOG)<=64*1024**3
    for name in SOURCES:assert (ROOT/name).read_bytes()==(t.OUT/'authorization/source_locked'/name).read_bytes()
    cumulative=budget['cumulative'];assert cumulative['geometry_reserved']==7300 and cumulative['geometry_attempted']==4231 and cumulative['preflight']==2
    parent=read(t.OUT/'authorization/parent_budget.json');assert parent['surface_accounting']['reserved']==3650 and parent['surface_accounting']['attempted']==581
    assert read(t.OUT/'authorization/environment_reuse.json')['installations']==0
    assert len(read(t.OUT/'surface/priority_complete.json')['ids'])==120
    for p in (t.OUT/'surface/structures').glob('*/status.json'):
        v=read(p)
        if v['available']:assert v['audit']['zero_modes']==v['audit']['components']<64 and v['audit']['retained_modes']<=64 and v['audit']['vertices']<=16384
    clock=read(t.OUT/'authorization/clock.json');parent_clock=read(ROOT/'results/ua3d_hmr_20261103/r1/authorization/clock.json');assert all(clock[k]==parent_clock[k] for k in ('tN','prepare_deadline_unix','deadline_unix'));assert time.time()<clock['deadline_unix']-600;assert max(j['ended_unix'] for j in jobs if j['kind']=='diagnostic')<=clock['prepare_deadline_unix'];assert not (t.OUT/'first_failure.json').exists()
    write(t.OUT/'final_cpu/consumption.json',dict(status='COMPLETE',attempted=dict(suites=1),completed=dict(suites=1),started_unix=STARTED,ended_unix=time.time(),model_constructs=0,model_forwards=0,geometry=0,predictions=0));print('FINAL CPU PASS',flush=True);return 0
if __name__=='__main__':
    STARTED=time.time();p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--final',action='store_true');a=p.parse_args()
    if a.preflight==a.final:raise ValueError('select exactly one authorized suite')
    sys.exit(preflight() if a.preflight else final())
