"""One bounded synthetic preflight; final suite only reads saved artifacts."""
import argparse,errno,json,math,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from scripts import bootstrap_ua3d_hmr_readout_r1 as b
COUNTS=dict(suites=1,model_constructs=0,forwards=0,backwards=0,optimizer_updates=0,real_labels=0,geometry=0,GPU_calls=0)
class FakeClock:
    def __init__(self):self.value=100.
    def __call__(self):return self.value
    def advance(self,delta):self.value+=delta

class FakeProcess:
    def __init__(self,pid):self.pid=pid;self.returncode=None;self.waits=0
    def poll(self):return self.returncode
    def wait(self,timeout=None):
        if self.returncode is None:raise RuntimeError('fake wait while running')
        self.waits+=1;return self.returncode

class FakeProc:
    def __init__(self):self.jobs={};self.processes={};self.fail=None;self.close_on_fail=True;self.reuse=False;self.permission=False;self.wrong=None;self.signals=[]
    def stage(self,pid,name):
        if self.permission and name=='environ':raise PermissionError(errno.EACCES,'injected permission')
        if self.fail==name:
            if self.close_on_fail:self.processes[pid].returncode=0
            raise OSError(errno.ESRCH if name=='stat' else errno.ENOENT,'injected process disappeared')
    def stat(self,pid):self.stage(pid,'stat');return dict(pid=pid,start_ticks=11 if self.reuse else 10)
    def cmdline(self,pid):self.stage(pid,'cmdline');return 'synthetic-worker --kind ft'
    def environ(self,pid):self.stage(pid,'environ');return {'CUDA_VISIBLE_DEVICES':'wrong' if self.wrong=='CVD' else ''}
    def cwd(self,pid):self.stage(pid,'cwd');return self.jobs[pid]['cwd']
    def stdout(self,pid):self.stage(pid,'stdout');return 'wrong' if self.wrong=='stdout' else str(Path(self.jobs[pid]['log'])/'stdout.log')
    def namespace(self,pid):self.stage(pid,'namespace');return [pid]
    def send(self,p,j,sig):
        if p.poll() is not None:return False
        if self.stat(p.pid)!=j['process']:raise ValueError('identity changed before fake signal')
        self.signals.append((p.pid,sig));p.returncode=-sig;return True

def preflight():
    import torch
    from src.modules import ua3d_hmr_readout_r1 as m
    from src.dataset import ua3d_hmr_readout_r1 as d
    torch.set_num_threads(1);torch.set_num_interop_threads(1);started=time.time();entry_evidence=[]
    class Contract(unittest.TestCase):
        def test_models_and_rng(self):
            stats={k:([0.]*n if k.endswith('mean') else [1.]*n) for k,n in [('z_mean',768),('z_scale',768),('stat_mean',16),('stat_scale',16),('spec_mean',128),('spec_scale',128)]};models={};states={};rng={}
            for arm in m.TRAIN_ARMS:
                torch.manual_seed(42);models[arm]=m.Readout(arm,stats,0);COUNTS['model_constructs']+=1;states[arm]={k:v.clone() for k,v in models[arm].state_dict().items()};rng[arm]=torch.get_rng_state().clone()
                self.assertEqual(sum(p.numel() for p in models[arm].parameters() if p.requires_grad),251265 if arm=='N_BASE' else 280609);self.assertFalse(models[arm].filter_raw.requires_grad)
            for arm in ('N_ADD','N_BILIN'):
                self.assertTrue(torch.equal(rng['N_BASE'],rng[arm]));self.assertTrue(all(torch.equal(v,states[arm][k]) for k,v in states['N_BASE'].items()))
            self.assertTrue(all(torch.equal(v,states['N_BILIN'][k]) for k,v in states['N_ADD'].items()));self.assertTrue(torch.count_nonzero(models['N_ADD'].U.weight)>0);self.assertTrue(torch.count_nonzero(models['N_ADD'].V.weight)>0);self.assertEqual(torch.count_nonzero(models['N_ADD'].W.weight),0)
            torch.manual_seed(7);batch=dict(z=torch.randn(5,768),availability=torch.ones(5,3),yb=torch.randn(5),y=torch.randn(5),lam=torch.linspace(0,5,64).repeat(5,1),C=torch.randn(5,64,16),a=torch.randn(5,64),A=torch.ones(5)*10,Ac=torch.ones(5)*5,stat=torch.randn(5,16),mode_mask=torch.ones(5,64),mask=torch.tensor([1.,0.,1.,0.,1.]))
            for name in ('lam','C','a','A','Ac','stat','mode_mask'):batch[name][~batch['mask'].bool()]=float('nan')
            for arm,model in models.items():
                model.eval();COUNTS['forwards']+=1;p=model(batch);self.assertTrue(torch.equal(p,batch['yb']));x,z,s=model.inputs(batch)
                if arm=='N_BASE':self.assertEqual(torch.count_nonzero(s),0)
                else:
                    with torch.no_grad():model.W.weight.copy_(torch.linspace(-.1,.2,32).reshape(1,32))
                    COUNTS['forwards']+=1;actual=model.branch(z,s);u=model.LN_z(model.U(z));v=model.LN_s(model.V(s));expected=model.W((u+v)/math.sqrt(2) if arm=='N_ADD' else u*v).flatten();self.assertTrue(torch.equal(actual,expected));self.assertTrue(torch.isfinite(actual).all())
                with torch.no_grad():model.head[-1].weight.fill_(.05)
                COUNTS['forwards']+=1;p=model(batch);self.assertTrue(torch.equal(p[~batch['mask'].bool()],batch['yb'][~batch['mask'].bool()]));COUNTS['backwards']+=1;p.square().sum().backward();self.assertTrue(torch.isfinite(model.head[0].weight.grad).all());self.assertTrue(torch.count_nonzero(model.head[0].weight.grad)>0);self.assertIsNone(model.filter_raw.grad)
                if arm!='N_BASE':self.assertTrue(torch.count_nonzero(model.U.weight.grad)>0);self.assertTrue(torch.count_nonzero(model.V.weight.grad)>0)
                bad={k:v.clone() for k,v in batch.items()};bad['stat'][0,0]=float('nan');COUNTS['forwards']+=1
                with self.assertRaises(FloatingPointError):model(bad)
                self.assertTrue(torch.equal(model.filter_raw,states[arm]['filter_raw']))
        def fixture(self,temp,rc=None):
            root=Path(temp);log=root/'job';log.mkdir();j=dict(id='fixture',kind='ft',gpu=False,reserved_unix=99.,started_unix=None,ended_unix=None,exit=None,qualified=False,argv=['synthetic-worker'],cwd=str(root),log=str(log),consumption=str(log/'consumption.json'),ownership_tokens=['synthetic-worker']);b.write(log/'process.json',dict(pid=123,start_ticks=10,started_unix=100.));b.write(log/'end.json',dict(exit=rc or 0));b.write(log/'consumption.json',dict(status='COMPLETE',attempted={},completed={}));p=FakeProcess(123);p.returncode=rc;return j,p
        def test_lifecycle_races(self):
            for stage in ('stat','cmdline','environ','cwd','stdout','namespace'):
                with tempfile.TemporaryDirectory() as temp:
                    j,p=self.fixture(temp);proc=FakeProc();proc.jobs[123]=j;proc.processes[123]=p;proc.fail=stage;e=b.Lifecycle(lambda j:b.read(j['consumption']),proc=proc,clock=FakeClock(),runner=lambda *a,**k:p);e.spawn(j,{});e.poll_all();e.stop();self.assertEqual(p.waits,1);self.assertTrue(j['qualified']);self.assertFalse(e.active)
            with tempfile.TemporaryDirectory() as temp:
                j,p=self.fixture(temp,7);proc=FakeProc();proc.jobs[123]=j;proc.processes[123]=p;e=b.Lifecycle(lambda j:None,proc=proc,clock=FakeClock(),runner=lambda *a,**k:p)
                with self.assertRaises(RuntimeError):e.spawn(j,{})
                self.assertFalse(j['qualified']);self.assertEqual(j['exit'],7)
        def test_source_barriers_and_init(self):
            from unittest.mock import patch
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                with self.assertRaises(ValueError):d.cache_path('eea',0,True,root)
                self.assertEqual(d.cache_path('eea',0,False,root),b.P2/'feature_cache/trainval/eea/fold0.pt');self.assertEqual(b.parse_jobs([])['counts'],{});j=dict(id='cpu',kind='cpu',gpu=False,reserved_unix=1.,started_unix=2.,ended_unix=None,exit=None,qualified=False);self.assertEqual(b.state_for([j],1)['status'],'RUNNING')
                b.write(root/'registry/canonical.json',dict(entries={'N_BASE/eea/0':dict(arm='N_BASE',task='eea',fold=0,seed=42,source_kind='readout_new',folder=str(root/'finetune/N_BASE/eea/fold0'))}));self.assertEqual(d.canonical('N_BASE','eea',0,root)['source_kind'],'readout_new')
                b.write(root/'registry/canonical.json',dict(entries={'N_BASE/eea/0':dict(arm='N_BASE',task='eea',fold=0,seed=42,source_kind='wrong')}))
                with self.assertRaises(ValueError):d.canonical('N_BASE','eea',0,root)
        def test_entry(self):
            for outcome in ('success','failure'):
                with tempfile.TemporaryDirectory() as temp:
                    root=Path(temp);(root/'SYNTHETIC_ONLY').write_text('synthetic');env=dict(os.environ);env.pop('PYTHONPATH',None);cmd=[sys.executable,str(ROOT/'scripts/bootstrap_ua3d_hmr_readout_r1.py'),'--synthetic-root',str(root),'--synthetic-outcome',outcome];p=subprocess.run(cmd,cwd='/',env=env,capture_output=True,text=True,timeout=120);expected=0 if outcome=='success' else 1;self.assertEqual(p.returncode,expected,p.stderr);self.assertEqual(b.read(root/'bootstrap_exit.json')['exit'],expected);self.assertTrue((root/'delivery_summary.md').exists());entry_evidence.append(dict(argv=cmd,cwd='/',exit=p.returncode))
                    if expected:self.assertIn('injected synthetic first failure',b.read(root/'first_failure.json')['error'])
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contract));COUNTS['tests']=result.testsRun;COUNTS['synthetic_entries']=len(entry_evidence);assert COUNTS['model_constructs']<=24 and COUNTS['forwards']<=128 and COUNTS['backwards']<=32 and len(entry_evidence)<=8;b.write(b.OUT/'preflight/entry_evidence.json',entry_evidence);b.write(b.OUT/'preflight/consumption.json',dict(status='COMPLETE' if result.wasSuccessful() else 'FAILED',attempted=COUNTS,completed=COUNTS if result.wasSuccessful() else {},started_unix=started,ended_unix=time.time()));return 0 if result.wasSuccessful() else 1

def final_cpu():
    import numpy as np
    import torch
    from src.training import ua3d_hmr_readout_r1 as t
    from src.dataset import ua3d_hmr_readout_r1 as d
    started=time.time();budget=b.read(b.OUT/'budget.json');jobs=budget['jobs'];b.parse_jobs(jobs)
    for kind,n in [('preflight',1),('qualify',1),('diagnostic',12),('ft',105),('test',175)]:assert budget['counts'][kind]['reserved']==n and budget['counts'][kind]['completed']==n,(kind,budget['counts'][kind])
    assert budget['counts']['cpu']['reserved']==1 and budget['counts']['cpu']['completed']==0;assert b.state_for([j for j in jobs if j['kind']=='cpu'],1)['status']=='RUNNING'
    du=budget['usage']['diagnostic']['completed'];fu=budget['usage']['ft']['completed'];out=budget['usage']['test']['completed']
    for key,n in dict(updates=240,train_forwards=240,backwards=240,train_rows=38400,eval_forwards=24,eval_rows=3840).items():assert du[key]==n
    for key,n in dict(epochs=7350,updates=17850,train_forwards=17850,backwards=17850,train_rows=3945060,validation=7350,validation_batches=9450,validation_rows=989940,epoch0_aliases=105,epoch0_alias_rows=14142).items():assert fu[key]==n
    for key,n in dict(new_outputs=105,alias_outputs=70,new_rows=17625,alias_rows=11750,test_rows=29375,test_batches=225,candidate_forwards=135,alias_batches=90).items():assert out[key]==n
    family=budget['cumulative'];assert family['historical_hmr']['update_interval']==[18090,18230] and family['historical_hmr']['unknown_old_updates']==[0,140];assert family['family_total']['update_interval']==[36180,36320] and family['family_total']['FT_attempts']==212;assert family['family_total']['geometry_reserved']==7300 and family['family_total']['geometry_attempted']==4231
    frozen=b.read(b.OUT/'selection/final_frozen.json');assert frozen['frozen'] and len(frozen['estimators'])==175 and frozen['FINAL']==t.choose(frozen['validation']);assert frozen['frozen_unix']>max(j['ended_unix'] for j in jobs if j['kind']=='ft');assert frozen['frozen_unix']<min(j['started_unix'] for j in jobs if j['kind']=='test')
    for arm in b.ARMS:
        for task in b.TASKS:
            for fold in range(5):
                source=d.canonical(arm,task,fold);folder=t.job_folder('test',arm,task,fold);met=b.read(folder/'metrics.json');split=b.read(t.SPLIT/f'{task}.json')['folds'][fold];mapping={x['row']:x for x in b.read(b.OUT/'registry/anchors.json')['rows'][task]};entry=frozen['estimators'][f'{arm}/{task}/{fold}']
                if arm!='C_ANCHOR':
                    q=t.qualify_ft(arm,task,fold);payload=t.read_model(arm,task,fold);assert payload['epoch']==entry['epoch'];assert entry['path']==str(Path(source['folder'])/'best.pt')
                    if arm in b.TRAIN_ARMS:
                        for name in ('epoch0','last','best'):
                            state=t.read_model(arm,task,fold,name)['state_dict'];t.validate_state(state,arm)
                with np.load(folder/'test_predictions.npz',allow_pickle=False) as p:
                    assert p['indices'].tolist()==sorted(split['test_indices']);assert p['sample_keys'].tolist()==[mapping[i]['key'] for i in p['indices'].tolist()];assert t.metrics(p['y_true'],p['y_pred'])==met['metrics']
                    if arm in ('C_ANCHOR','C_FIXED'):
                        with np.load(d.old_output(arm,task,fold),allow_pickle=False) as old:assert all(np.array_equal(p[k],old[k]) for k in ('indices','sample_keys','y_true','y_pred'))
                assert b.read(b.OUT/'registry/test_qualification'/task/f'fold{fold}.json')['qualified_unix']>frozen['frozen_unix']
    for j in jobs:
        if j['kind']=='cpu':continue
        assert j['qualified'] and j['exit']==0;log=Path(j['log']);process=b.read(log/'process.json');assert b.read(log/'end.json')['exit']==0 and b.read(log/'wait_exit.json')['exit']==0;assert process['CVD']==('2' if j['gpu'] else '')
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE'):assert process['env'][key]=='1'
    for kind,cap in [('diagnostic',1),('ft',8),('test',8)]:
        entries=[j for j in jobs if j['kind']==kind and j['gpu']];events=sorted([(j['started_unix'],1) for j in entries]+[(j['ended_unix'],-1) for j in entries]);active=peak=0
        for _,delta in events:active+=delta;peak=max(active,peak)
        assert peak<=cap
        for j in entries:
            for other in jobs:
                if other['gpu'] and other['kind']!=kind:assert j['ended_unix']<=other['started_unix'] or other['ended_unix']<=j['started_unix']
    pilot=next(j for j in jobs if j['kind']=='ft' and (j['arm'],j['task'],j['fold'])==('N_BILIN','egc',1));assert pilot['ended_unix']-pilot['started_unix']<=1800;assert pilot['ended_unix']<=min(j['started_unix'] for j in jobs if j['kind']=='ft' and j is not pilot)
    sources=b.read(b.OUT/'registry/sources.json');assert sources['status']=='COMPLETE' and len(sources['units'])==35 and all(x['TEST_status']=='NOT_READ' for x in sources['units']);assert sources['new_geometry']==sources['new_spectrum']==sources['new_BASE_forward']==sources['new_fit']==sources['new_pack']==0
    for name in b.SOURCES:assert (ROOT/name).read_bytes()==(b.OUT/'authorization/source_locked'/name).read_bytes()
    for old in (b.P2,b.P3):
        for name in b.read(old/'authorization/source_lock.json')['files']:assert (ROOT/name).read_bytes()==(old/'authorization/source_locked'/name).read_bytes()
    assert b.read(b.P3/'cumulative_budget.json')['cumulative_update_interval']==[18090,18230]
    clock=b.read(b.OUT/'authorization/clock.json');assert max(j['ended_unix'] for j in jobs if j['kind']=='qualify')<=clock['prepare_deadline_unix'];assert time.time()<clock['deadline_unix']-600;assert not (b.OUT/'first_failure.json').exists();assert b.size(b.OUT)+b.size(b.LOG)<=8*1024**3
    result=t.scientific_results();assert result['formal_complete'] and result['performance_pass']==result['target_met'];assert abs(result['formal']['C_ANCHOR']['macro']-t.REFERENCE)<1e-6;assert abs(result['formal']['C_FIXED']['macro']-b.read(b.P3/'final_review/results.json')['formal']['S_FIXED']['macro'])<1e-12
    for name,value in result['comparisons'].items():
        a,other=name.split('-');assert value['macro']==result['formal'][a]['macro']-result['formal'][other]['macro']
    b.write(b.OUT/'final_cpu/audit.json',dict(status='PASSED',new_FT=105,new_outputs=105,alias_outputs=70,formal_rows=29375,total_outputs=175,model_constructs=0,forwards=0,backwards=0,geometry=0,predictions=0,time=time.time()));counts=dict(suites=1);b.write(b.OUT/'cpu/consumption.json',dict(status='COMPLETE',attempted=counts,completed=counts,started_unix=started,ended_unix=time.time()));print('FINAL CPU PASS',flush=True);return 0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--final',action='store_true');a=p.parse_args()
    if a.preflight==a.final:raise ValueError('exactly one authorized suite')
    sys.exit(preflight() if a.preflight else final_cpu())
