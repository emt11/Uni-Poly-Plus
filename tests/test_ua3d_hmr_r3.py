"""One pure lifecycle regression; final CPU reads existing artifacts only."""
import argparse
import errno
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from scripts import bootstrap_ua3d_hmr_r3 as b

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

def regression():
    started=time.time();evidence=[]
    class Contract(unittest.TestCase):
        def fixture(self,temp,pid=123,rc=None,missing=False):
            root=Path(temp);log=root/str(pid);log.mkdir();j=dict(id=str(pid),kind='ft',gpu=False,reserved_unix=99.,started_unix=None,ended_unix=None,exit=None,qualified=False,argv=['synthetic-worker'],cwd=str(root),log=str(log),consumption=str(log/'consumption.json'),ownership_tokens=['synthetic-worker'])
            b.write(log/'process.json',dict(pid=pid,start_ticks=10,started_unix=100.));b.write(log/'end.json',dict(exit=rc or 0));b.write(log/'consumption.json',dict(status='COMPLETE',attempted={'updates':70},completed={'updates':70}))
            if not missing:b.write(log/'artifact.json',dict(status='COMPLETE'))
            p=FakeProcess(pid);p.returncode=rc;return j,p
        def engine(self,j,p,proc=None,clock=None):
            proc=proc or FakeProc();clock=clock or FakeClock();proc.jobs[p.pid]=j;proc.processes[p.pid]=p
            def validate(job):
                u=b.read(Path(job['consumption']));a=b.read(Path(job['log'])/'artifact.json')
                if u['status']!='COMPLETE' or a['status']!='COMPLETE':raise ValueError('bad fixture artifact')
            engine=b.Lifecycle(validate,proc=proc,clock=clock,runner=lambda *a,**kw:p,pause=lambda delta:clock.advance(.2));return engine,proc,clock
        def test_disappearance_each_proc_stage(self):
            for stage in ('stat','cmdline','environ','cwd','stdout','namespace'):
                with self.subTest(stage=stage),tempfile.TemporaryDirectory() as temp:
                    j,p=self.fixture(temp);proc=FakeProc();proc.fail=stage;e,proc,c=self.engine(j,p,proc);e.spawn(j,{});self.assertFalse(e.active);self.assertTrue(j['qualified']);self.assertEqual(j['identity_source'],'exited_worker_record');self.assertEqual(p.waits,1);e.poll_all();e.stop();self.assertEqual(p.waits,1);self.assertEqual(len(e.events),1)
        def test_fast_exit_and_missing_artifacts(self):
            for rc,missing in ((0,False),(7,False),(0,True)):
                with tempfile.TemporaryDirectory() as temp:
                    j,p=self.fixture(temp,rc=rc,missing=missing);e,proc,c=self.engine(j,p)
                    if rc or missing:
                        with self.assertRaises((RuntimeError,FileNotFoundError)):e.spawn(j,{})
                        self.assertFalse(j['qualified'])
                    else:e.spawn(j,{});self.assertTrue(j['qualified'])
                    self.assertFalse(e.active);e.poll_all();self.assertEqual(p.waits,1);self.assertEqual(b.read(Path(j['log'])/'wait_exit.json')['exit'],rc)
        def test_exit_pending_close_and_timeout(self):
            for close in (True,False):
                with tempfile.TemporaryDirectory() as temp:
                    j,p=self.fixture(temp);proc=FakeProc();proc.fail='environ';proc.close_on_fail=False;e,proc,c=self.engine(j,p,proc);e.spawn(j,{});self.assertTrue(e.pending());self.assertIsNone(j['exit']);self.assertFalse(j['qualified'])
                    j2,p2=self.fixture(temp,pid=124)
                    with self.assertRaises(RuntimeError):e.spawn(j2,{})
                    if close:p.returncode=0;e.poll_all();self.assertTrue(j['qualified'])
                    else:
                        c.advance(5.1)
                        with self.assertRaises(TimeoutError):e.poll_all()
                        p.returncode=0;e.poll_all()
                    self.assertEqual(p.waits,1)
        def test_bad_json_is_not_disappearance(self):
            with tempfile.TemporaryDirectory() as temp:
                j,p=self.fixture(temp,rc=0);Path(j['consumption']).write_text('{invalid');e,proc,c=self.engine(j,p)
                with self.assertRaises(json.JSONDecodeError):e.spawn(j,{})
                self.assertEqual(p.waits,1);self.assertFalse(j['qualified'])
        def test_identity_permissions_and_stop(self):
            for mode in ('permission','CVD','stdout','reuse'):
                with tempfile.TemporaryDirectory() as temp:
                    j,p=self.fixture(temp);e,proc,c=self.engine(j,p);e.spawn(j,{})
                    if mode=='permission':proc.permission=True
                    elif mode=='reuse':proc.reuse=True
                    else:proc.wrong=mode
                    with self.assertRaises((PermissionError,ValueError)):e.poll_all()
                    if mode=='reuse':e.stop();self.assertEqual(proc.signals,[])
                    proc.permission=False;proc.reuse=False;proc.wrong=None;p.returncode=0;e.poll_all();self.assertEqual(p.waits,1)
            with tempfile.TemporaryDirectory() as temp:
                j,p=self.fixture(temp);e,proc,c=self.engine(j,p);e.spawn(j,{});e.stop();self.assertFalse(e.active);self.assertEqual(p.waits,1);self.assertEqual(proc.signals,[(p.pid,signal.SIGTERM)]);e.stop();self.assertEqual(p.waits,1)
        def test_multiple_endings_and_duplicate_poll(self):
            with tempfile.TemporaryDirectory() as temp:
                j,p=self.fixture(temp);e,proc,c=self.engine(j,p);e.spawn(j,{});j2,p2=self.fixture(temp,pid=124);proc.jobs[p2.pid]=j2;proc.processes[p2.pid]=p2;e.runner=lambda *a,**kw:p2;e.spawn(j2,{});p.returncode=p2.returncode=0;e.poll_all();e.poll_all();e.stop();self.assertEqual((p.waits,p2.waits),(1,1));self.assertEqual(b.parse_jobs([j,j2])['counts']['ft']['completed'],2)
        def test_resolver_payload_and_interval(self):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp)/'r3';parent=Path(temp)/'r2';entries={}
                for arm in b.ARMS:
                    for task in b.TASKS:
                        for fold in range(5):
                            alias=(arm,task,fold) in b.ALIASES;entries[f'{arm}/{task}/{fold}']=dict(arm=arm,task=task,fold=fold,seed=42+fold,source_kind='parent_complete_alias' if alias else 'r3_new_ft',folder=str((parent if alias else root)/'finetune'/arm/task/f'fold{fold}'),expected_best_epoch=b.ALIASES.get((arm,task,fold)))
                b.write(root/'registry/canonical.json',dict(entries=entries,parent_root=str(parent)));self.assertEqual(sum(b.canonical_entry(root,a,t,f)['source_kind']=='parent_complete_alias' for a in b.ARMS for t in b.TASKS for f in range(5)),2)
                config={'science':'r2','lr':.0003};entry=b.canonical_entry(root,'S_STAT','eea',0);payload=dict(schema='ua3d-hmr-residual-r2',config=config,arm='S_STAT',task='eea',fold=0,seed=42,epoch=0,state_dict={'synthetic':[1.]});self.assertIs(b.unwrap_science(payload,entry,config),payload)
                entry=b.canonical_entry(root,'S_STAT','eea',1);new=dict(payload,fold=1,seed=43);outer=dict(schema='ua3d-hmr-execution-r3',science_revision='r2',execution_revision='r3',source_kind='r3_new_ft',science_payload=new);self.assertEqual(b.unwrap_science(outer,entry,config)['epoch'],0)
                with self.assertRaises(ValueError):b.unwrap_science(outer,entry,{'lr':1})
                with self.assertRaises(ValueError):b.require_frozen(root)
                x=b.cumulative([{'kind':'ft'}]*103,17150);self.assertEqual(x['FT_attempts'],107);self.assertEqual(x['cumulative_update_interval'],[18090,18230]);self.assertEqual(x['old_unknown_updates']['sum'],[0,140]);self.assertEqual(x['geometry_attempted'],4231)
                with self.assertRaises(ValueError):b.cumulative([{'kind':'ft'}]*104,17150)
                self.assertEqual(b.parse_jobs([])['counts'],{});job=dict(id='cpu',kind='cpu',gpu=False,reserved_unix=1.,started_unix=2.,ended_unix=None,exit=None,qualified=False);self.assertEqual(b.state_for([job],1)['status'],'RUNNING')
        def test_real_bootstrap_entries(self):
            for outcome in ('success','failure'):
                with tempfile.TemporaryDirectory() as temp:
                    root=Path(temp);(root/'SYNTHETIC_ONLY').write_text('synthetic');env=dict(os.environ);env.pop('PYTHONPATH',None);cmd=[sys.executable,str(ROOT/'scripts/bootstrap_ua3d_hmr_r3.py'),'--synthetic-root',str(root),'--synthetic-outcome',outcome];p=subprocess.run(cmd,cwd='/',env=env,capture_output=True,text=True,timeout=120);expected=0 if outcome=='success' else 1;self.assertEqual(p.returncode,expected,p.stderr);self.assertEqual(b.read(root/'bootstrap_exit.json')['exit'],expected);self.assertTrue((root/'delivery_summary.md').exists())
                    if expected:self.assertIn('injected synthetic first failure',b.read(root/'first_failure.json')['error'])
                    evidence.append(dict(argv=cmd,cwd='/',exit=p.returncode,real_scientific_jobs=0))
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contract));counters=dict(suites=1,tests=result.testsRun,model_constructs=0,forwards=0,backwards=0,optimizer_updates=0,real_labels=0,geometry=0,GPU_calls=0,synthetic_entry_processes=len(evidence));b.write(b.OUT/'preflight/entry_evidence.json',evidence);b.write(b.OUT/'preflight/consumption.json',dict(status='COMPLETE' if result.wasSuccessful() else 'FAILED',attempted=counters,completed=counters if result.wasSuccessful() else {},started_unix=started,ended_unix=time.time()));return 0 if result.wasSuccessful() else 1

def final_cpu():
    import numpy as np
    from src.training import ua3d_hmr_r3 as t
    started=time.time();budget=b.read(b.OUT/'budget.json');jobs=budget['jobs'];b.parse_jobs(jobs)
    for kind,count in [('preflight',1),('ft',103),('test',140)]:
        assert budget['counts'][kind]['reserved']==count and budget['counts'][kind]['completed']==count,(kind,budget['counts'][kind])
    assert budget['counts']['cpu']['reserved']==1 and budget['counts']['cpu']['completed']==0
    assert b.state_for([j for j in jobs if j['kind']=='cpu'],1)['status']=='RUNNING'
    totals=budget['usage']['ft']['completed']
    for k,v in dict(epochs=7210,updates=17150,train_forwards=17150,backwards=17150,train_rows=3777200,validation=7210,validation_batches=9170,validation_rows=947940,epoch0_aliases=103,epoch0_alias_rows=13542).items():assert totals[k]==v,(k,totals[k],v)
    for k,v in dict(test_rows=23500,test_batches=180,candidate_forwards=135,anchor_batches=45).items():assert budget['usage']['test']['completed'][k]==v
    c=budget['cumulative'];assert c['FT_attempts']==107 and c['cumulative_update_interval']==[18090,18230] and c['old_unknown_updates']['sum']==[0,140] and c['geometry_reserved']==7300 and c['geometry_attempted']==4231 and c['preflight_total']==3
    resolved=[]
    for a in b.ARMS:
        for task in b.TASKS:
            for fold in range(5):resolved.append(t.qualify_ft(a,task,fold))
    assert len(resolved)==105 and sum(x['source_kind']=='parent_complete_alias' for x in resolved)==2 and sum(x['source_kind']=='r3_new_ft' for x in resolved)==103
    assert sum(x['consumption']['completed']['updates'] for x in resolved)==17850
    frozen=b.read(b.OUT/'selection/final_frozen.json');assert frozen['frozen'] and len(frozen['estimators'])==140 and frozen['science_revision']=='r2' and frozen['execution_revision']=='r3';assert frozen['FINAL']==t.choose(frozen['validation']);assert frozen['frozen_unix']>max(j['ended_unix'] for j in jobs if j['kind']=='ft');assert frozen['frozen_unix']<min(j['started_unix'] for j in jobs if j['kind']=='test')
    for arm in t.m.ARMS:
        for task in b.TASKS:
            for fold in range(5):
                key=f'{arm}/{task}/{fold}';folder=t.job_folder('test',arm,task,fold);met=b.read(folder/'metrics.json');split=b.read(t.SPLIT/f'{task}.json')['folds'][fold];mapping={r['row']:r for r in b.read(b.OUT/'registry/anchors.json')['rows'][task]}
                if arm!='C_ANCHOR':
                    entry=b.canonical_entry(b.OUT,arm,task,fold);saved=t.read_science(arm,task,fold);assert frozen['estimators'][key]['source_kind']==entry['source_kind'] and frozen['estimators'][key]['path']==str(Path(entry['folder'])/'best.pt') and frozen['estimators'][key]['epoch']==saved['epoch']
                with np.load(folder/'test_predictions.npz',allow_pickle=False) as p:
                    assert p['indices'].tolist()==sorted(split['test_indices']);assert p['sample_keys'].tolist()==[mapping[row]['key'] for row in p['indices'].tolist()];assert t.metrics(p['y_true'],p['y_pred'])==met['metrics']
    for j in jobs:
        if j['kind']=='cpu':continue
        assert j['qualified'] and j['exit']==0;log=Path(j['log']);process=b.read(log/'process.json');end=b.read(log/'end.json');wait=b.read(log/'wait_exit.json');assert wait['exit']==end['exit']==0;assert process['CVD']==('2' if j['gpu'] else '')
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE'):assert process['env'][key]=='1'
    for phase in ('ft','test'):
        gpu=[j for j in jobs if j['kind']==phase and j['gpu']];events=sorted([(j['started_unix'],1) for j in gpu]+[(j['ended_unix'],-1) for j in gpu]);active=0;peak=0
        for _,delta in events:active+=delta;peak=max(peak,active)
        assert peak<=8
        for j in gpu:
            for other in jobs:
                if other['gpu'] and other['kind']!=phase:assert j['ended_unix']<=other['started_unix'] or other['ended_unix']<=j['started_unix']
    for task in b.TASKS:
        for fold in range(5):assert b.read(b.OUT/'registry/cache_qualification/test'/task/f'fold{fold}.json')['completed_unix']>frozen['frozen_unix']
    qualification=b.read(b.OUT/'registry/parent_qualification.json');assert qualification['status']=='COMPLETE' and len(qualification['cache'])==35 and len(qualification['aliases'])==2
    for parent in (b.P1,b.P2):
        for name in ('controller','outer','bootstrap'):assert b.read(parent/f'{name}_exit.json')['exit']==1
    for name in b.SOURCES:assert (ROOT/name).read_bytes()==(b.OUT/'authorization/source_locked'/name).read_bytes()
    for name in b.read(b.P2/'authorization/source_lock.json')['files']:assert (ROOT/name).read_bytes()==(b.P2/'authorization/source_locked'/name).read_bytes()
    clock=b.read(b.OUT/'authorization/clock.json');old=b.read(b.P1/'authorization/clock.json');assert all(clock[k]==old[k] for k in ('tN','prepare_deadline_unix','deadline_unix'));assert time.time()<clock['deadline_unix']-600;assert b.read(b.OUT/'gates/initial.json')['observed_unix']<clock['prepare_deadline_unix'];assert not (b.OUT/'first_failure.json').exists();assert b.size(b.OUT)+b.size(b.LOG)<=8*1024**3
    result=t.scientific_results();assert result['formal_complete'];assert abs(result['formal']['C_ANCHOR']['macro']-t.REFERENCE)<=1e-6
    b.write(b.OUT/'final_cpu/audit.json',dict(status='PASSED',canonical=105,parent_alias=2,new_FT=103,formal_outputs=140,model_constructs=0,forwards=0,backwards=0,geometry=0,predictions=0,time=time.time()));counts=dict(suites=1);b.write(b.OUT/'cpu/consumption.json',dict(status='COMPLETE',attempted=counts,completed=counts,started_unix=started,ended_unix=time.time()));print('FINAL CPU PASS',flush=True);return 0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--final',action='store_true');a=p.parse_args()
    if a.preflight==a.final:raise ValueError('exactly one suite selector')
    sys.exit(regression() if a.preflight else final_cpu())
