#!/usr/bin/env python3
"""Single authorized anchored campaign; immutable identities and irreversible reservations."""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name,'2')
import argparse
from collections import deque
import difflib
import fcntl
import json
import math
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.training import ua3d_anchor_lr_r2 as t
from scripts.ph_local_worker_attestation import attest_worker
from scripts import bootstrap_ua3d_anchor_lr_r2 as boot

SPEC=t.SPEC; CAPS=SPEC['caps']; BUDGET=t.OUT/'budget.json'


state_for=boot.state_for


def reservation(kind,arm=None,task=None,fold=None):
    if kind=='preflight':
        return {k:v for k,v in SPEC['limits'].items() if k.startswith('preflight_')}
    if kind=='cpu':
        return dict(cpu_suites=1)
    e=t.read(t.SPLIT/f'{task}.json')['folds'][fold] if task is not None else None
    if kind in ('cache_trainval','cache_test'):
        formal=kind=='cache_test';ids=e['test_indices'] if formal else e['train_indices']+e['validation_indices']
        return {kind+'_units':1,kind+'_rows':len(ids)}
    if kind=='diagnostic':
        value=dict(diagnostic_starts=1,diagnostic_eval_forwards=2,diagnostic_eval_rows=48,diagnostic_train_forwards=20,
                   diagnostic_backwards=20,diagnostic_updates=20,diagnostic_train_rows=480,frozen_H_batches=44)
        if t.m.backend(arm)=='P_PE':
            value['residual_U_batches']=44
        return value
    if kind=='ft':
        train=70*math.ceil(len(e['train_indices'])/24); val=70*math.ceil(len(e['validation_indices'])/24)
        value=dict(ft_starts=1,ft_epochs=70,ft_updates=train,ft_train_forwards=train,ft_train_rows=70*len(e['train_indices']),
                   validation_count=70,validation_forwards=val,validation_rows=70*len(e['validation_indices']),frozen_H_batches=2*(train+val))
        if t.m.backend(arm)=='P_PE':
            value['residual_U_batches']=2*(train+val)
        value.update(epoch0_validation_aliases=1,epoch0_validation_rows=len(e['validation_indices']))
        return value
    if kind=='test':
        batches=math.ceil(len(e['test_indices'])/24); value=dict(test_starts=1,test_rows=len(e['test_indices']),test_batches=batches)
        if arm=='C_ANCHOR':
            value['anchor_output_batches']=batches
        else:
            value.update(adapter_test_forwards=batches,frozen_H_batches=2*batches)
            if t.m.backend(arm)=='P_PE':
                value['residual_U_batches']=2*batches
        return value
    raise ValueError('unregistered job kind')


def ledger(jobs):
    return boot.scientific_ledger(jobs,SPEC,t.contract(),t.ACTIVATION,t.read(t.OUT/'parent_accounting.json'),t.OUT)


def wrap(spec):
    j=t.read(spec); folder=Path(j['log']); env=os.environ.copy(); env.update(j['env']); identity=None
    def stop(sig,frame):
        if identity is not None and t.process_alive(identity):
            os.killpg(identity['pid'],signal.SIGTERM)
    signal.signal(signal.SIGINT,stop); signal.signal(signal.SIGTERM,stop)
    with (folder/'stdout.log').open('x') as out,(folder/'stderr.log').open('x') as err:
        child=subprocess.Popen(j['argv'],cwd=j['cwd'],env=env,stdout=out,stderr=err,start_new_session=True); identity=t.process_identity(child.pid)
        t.exclusive(folder/'process.json',dict(child=identity,wrapper=t.process_identity(os.getpid()),argv=j['argv'],cwd=j['cwd'],env=j['env'],started_unix=time.time()))
        rc=child.wait()
    t.exclusive(folder/'end.json',dict(exit=rc,ended_unix=time.time()))
    with (folder/'exit.txt').open('x') as f:
        f.write(str(rc)+'\n')
    return rc


class Controller:
    def __init__(self):
        t.require_tmux()
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
            raise ValueError('controller must hide GPU')
        self.lock=(t.OUT/'controller.lock').open('a'); fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        t.exclusive(t.OUT/'authorization/controller_start.json',dict(process=t.process_identity(os.getpid()),cwd=str(ROOT),argv=sys.argv,time=t.now()))
        self.jobs=[]; self.manifest=t.read(t.OUT/'execution_manifest.json')
        self.manifest.update(status='RUNNING',implementation_status='PASSED',controller_identity=t.process_identity(os.getpid()))
        self.manifest['stages']['implementation'].update(status='PASSED',exit=0,ended=time.time())
        sys.stdout=(t.LOG/'controller.stdout.log').open('x',buffering=1); sys.stderr=(t.LOG/'controller.stderr.log').open('x',buffering=1)
        def cutoff(sig,frame):
            raise TimeoutError('hard deadline or controlled termination')
        signal.signal(signal.SIGTERM,cutoff); signal.signal(signal.SIGINT,cutoff); signal.signal(signal.SIGALRM,cutoff)
        signal.setitimer(signal.ITIMER_REAL,max(.001,t.DEADLINE-900-time.time()))
        self.runtime,_=boot.initialize_metadata(t.OUT,t.LOG,t.ACTIVATION,SPEC,t.read(t.OUT/'authorization/parent_r1/authorization/activation.json'),self.manifest,self.jobs,t.read(t.OUT/'parent_accounting.json'),t.contract())
        self.capture(); self.refresh()
    def capture(self):
        lines=[]
        for n in t.SOURCES:
            lines+=difflib.unified_diff([], (ROOT/n).read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+n)
        lines+=difflib.unified_diff((t.OUT/'authorization/CURRENT_PLAN.md').read_text().splitlines(True),(ROOT/'CURRENT_PLAN.md').read_text().splitlines(True),fromfile='before/CURRENT_PLAN.md',tofile='after/CURRENT_PLAN.md')
        (t.OUT/'actual_changes.diff').write_text(''.join(lines))
    def status(self,phase,detail,status='RUNNING'):
        self.manifest.update(status=status,phase=phase,detail=detail,updated=t.now()); t.write(t.OUT/'execution_manifest.json',self.manifest)
        t.write(t.OUT/'status.json',dict(status=status,phase=phase,detail=detail,time=t.now(),independent_review='PENDING'))
        p=ROOT/'CURRENT_PLAN.md'; text=p.read_text(); a='<!-- UA3D_ANCHOR_LR_R2_STATUS_START -->'; b='<!-- UA3D_ANCHOR_LR_R2_STATUS_END -->'
        left,rest=text.split(a,1); old,right=rest.split(b,1); old='\n'.join(line for line in old.splitlines() if not line.startswith('- 实时执行状态：'))
        p.write_text(left+a+old+'\n- 实时执行状态：'+t.now()+'；'+status+'；'+phase+'；'+detail+'\n'+b+right)
    def refresh(self):
        for j in self.jobs:
            folder=Path(j['log'])
            if (folder/'process.json').exists():
                j['started_unix']=t.read(folder/'process.json')['started_unix']
            if j['exit'] is None and (folder/'exit.txt').exists():
                j['exit']=int((folder/'exit.txt').read_text()); j['ended_unix']=t.read(folder/'end.json')['ended_unix']
        self.runtime.jobs=self.jobs;self.runtime.manifest=self.manifest
        with (t.OUT/'budget.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX);budget=self.runtime.refresh()
        if (t.OUT/'gates/cost.json').exists():
            g=t.read(t.OUT/'gates/cost.json'); self.manifest['stages']['cost_gate']=dict(status='PASSED' if g['pass'] else 'FAILED',attempts=1,exit=0 if g['pass'] else 1,started=g.get('time_unix'),ended=g.get('time_unix'))
        if (t.OUT/'selection/final_frozen.json').exists():
            when=t.read(t.OUT/'selection/final_frozen.json')['time_unix']; self.manifest['stages']['freeze']=dict(status='PASSED',attempts=1,exit=0,started=when,ended=when)
        t.write(t.OUT/'execution_manifest.json',self.manifest)
        if budget['budget_violations']:
            raise RuntimeError('budget violation '+str(budget['budget_violations']))
        return budget
    def failures(self):
        j=next((j for j in self.jobs if j['exit'] not in (None,0)),None)
        if j is not None:
            raise RuntimeError(j['name']+' true exit='+str(j['exit'])+'; stop without retry')
    def guard(self,kind=None):
        t.guard(300 if kind=='cpu' else 900,preparation=not (t.OUT/'gates/cost.json').exists())
        if shutil.disk_usage(ROOT).free<100*1024**3:
            raise RuntimeError('disk reserve below100GiB')
    def inventory(self):
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
        row=next(x for x in raw.splitlines() if x.split(',')[0].strip()=='2'); _,uuid,free,total,util=[x.strip() for x in row.split(',')]
        saved=t.OUT/'registry/gpu.json'
        if saved.exists() and t.read(saved)['uuid']!=uuid:
            raise ValueError('GPU2 UUID changed')
        raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10); processes={}
        for line in raw.splitlines():
            cols=[x.strip() for x in line.split(',')]
            if cols[0]==uuid:
                processes[int(cols[1])]=int(cols[2]) if cols[2].isdigit() else 0
        return dict(uuid=uuid,free_mib=int(free),total_mib=int(total),utilization=int(util) if util.isdigit() else 100,host_processes=processes)
    def uses_gpu(self,kind,arm):
        return kind not in ('preflight','cpu','cache_trainval','cache_test') and not (kind=='test' and arm=='C_ANCHOR')
    def memory_requirement(self,kind,arm,task=None,fold=None):
        floor=SPEC['admission_mib'].get(kind,0)
        if kind not in ('ft','test') or arm=='C_ANCHOR' or not (t.OUT/'gates/cost.json').exists():
            return floor
        gate=t.read(t.OUT/'gates/cost.json'); peak=max(gate['profiles'][lab][arm]['peak_mib'] for lab in ('S','L'))
        inventory=t.read(t.OUT/'registry/workload.json'); e=t.read(t.SPLIT/f'{task}.json')['folds'][fold]
        ids=e['test_indices'] if kind=='test' else e['train_indices']+e['validation_indices']; records=[inventory['rows'][task][str(i)] for i in ids]
        columns=['nodes','bonds','spatial_edges'] if t.m.backend(arm)=='G_GEOM' else ['nodes','fixed_memberships']
        scale=max([1.]+[sum(sorted((r[c] for r in records),reverse=True)[:24])/max(inventory['diagnostic_inputs']['L']['work'][c],1) for c in columns])
        return max(floor,peak*scale+2048.)
    def available(self,kind,arm,task=None,fold=None):
        active=[j for j in self.jobs if j['exit'] is None]
        if any(j['kind']!=kind for j in active):
            raise ValueError('mutually exclusive stage overlap')
        cap=CAPS.get(kind,1)
        if len(active)>=cap:
            return False
        if not self.uses_gpu(kind,arm):
            return True
        inv=self.inventory(); attestations=[]; reserve=0.
        for j in active:
            if not Path(j['consumption']).exists() or not (Path(j['consumption']).parent/'environment.json').exists():
                return False
            result=attest_worker(j); attestations.append(result)
            if not result['attested']:
                if t.read(j['consumption'])['status'] in ('COMPLETE','FAILED'):
                    return False  # Await the wrapper's real exit marker during orderly teardown.
                raise ValueError('local worker attestation failed: '+str(result))
            reserve+=j['memory_reserve_mib']
        if inv['host_processes'] and not active:
            okay=False
        else:
            okay=inv['free_mib']>=self.memory_requirement(kind,arm,task,fold)+reserve+2048 and inv['utilization']<95
        t.write(t.OUT/'occupancy.json',dict(time=t.now(),inventory=inv,local_attestations=attestations,host_pid_mapping='UNKNOWN',
                host_pids_claimed_owned=[],unfulfilled_growth_mib=reserve,dispatch_allowed=okay,policy='no unverified PID deductions; whole-card admission only after every active local worker attests'))
        return okay
    def estimate(self,kind,arm,task,fold):
        gate=t.OUT/('gates/cost_after_pilot.json' if (t.OUT/'gates/cost_after_pilot.json').exists() else 'gates/cost.json')
        if gate.exists() and kind in ('ft','cache_test','test'):
            value=t.read(gate); unit=next((x for x in value['per_unit'] if x['kind']==kind and x['arm']==arm and x['task']==task and x['fold']==fold),None)
            if unit is None or unit['seconds'] is None:
                raise ValueError('no conservative per-unit ETA')
            return unit['seconds']
        if kind in ('preflight','cpu','diagnostic'):
            return 600.
        if kind in ('cache_trainval','cache_test'):
            return 15.
        raise ValueError('missing registered duration estimate')
    def launch(self,kind,arm=None,task=None,fold=None,repeat=None):
        self.guard(kind); self.refresh(); self.failures()
        if not self.available(kind,arm,task,fold):
            return False
        if any(j['kind']=='cpu' for j in self.jobs):
            raise ValueError('dispatch after final CPU')
        if kind=='cache_trainval' and sum(j['kind']=='preflight' and j['exit']==0 for j in self.jobs)!=1:
            raise ValueError('cache before preflight')
        if kind=='diagnostic':
            t.source_lock()
            if sum(j['kind']=='cache_trainval' and j['exit']==0 for j in self.jobs)!=35:
                raise ValueError('diagnostic before35 good teacher caches')
        pilot=kind=='ft' and (arm,task,fold)==('G_GEOM_1E4','egc',1)
        if kind=='ft':
            if not t.read(t.OUT/'gates/cost.json')['pass'] or not t.read(t.OUT/'diagnostics/comparison.json')['pass']:
                raise ValueError('FT before gates')
            if not pilot and not t.read(t.OUT/'gates/pilot.json')['pass']:
                raise ValueError('FT before pilot gate')
        if kind=='cache_test' and not (t.OUT/'selection/final_frozen.json').exists():
            raise ValueError('TEST teacher before freeze')
        if kind=='test':
            if sum(j['kind']=='cache_test' and j['exit']==0 for j in self.jobs)!=35:
                raise ValueError('output before35 TEST teachers')
            if arm!='C_ANCHOR' and not t.read(t.OUT/'formal/anchor_gate.json')['pass']:
                raise ValueError('adapter output before C_ANCHOR consistency')
        if kind=='cpu' and sum(j['kind']=='test' and j['exit']==0 for j in self.jobs)!=175:
            raise ValueError('final CPU before175TEST')
        identity=(kind,arm,task,fold,repeat)
        if any(tuple(j['identity'])==identity for j in self.jobs):
            raise ValueError('duplicate registered identity')
        folder=t.job_path(kind,arm,task,fold,repeat)
        if folder.exists():
            raise FileExistsError('job output already exists')
        estimated=self.estimate(kind,arm,task,fold)
        if time.time()+estimated+900>=t.DEADLINE:
            raise TimeoutError('unit conservative duration cannot fit remaining deadline')
        reserve=reservation(kind,arm,task,fold); budget=self.refresh()
        if any(budget['reserved'][k]+v>SPEC['limits'][k] for k,v in reserve.items()):
            raise RuntimeError('reservation exceeds fixed budget')
        name='ua3d-anchor-lr-r2-'+kind+('-'+arm if arm else '')+(f'-{task}-{fold}' if task is not None else '')+('-'+repeat if repeat else '')
        log=t.LOG/name; log.mkdir(exist_ok=False); gpu=self.uses_gpu(kind,arm)
        seconds=600 if kind in ('preflight','cpu','diagnostic') else 5400 if pilot else int(t.DEADLINE-900-time.time())
        argv=['timeout','--signal=TERM','--kill-after=30s',str(seconds)+'s',sys.executable,'-u']
        if kind in ('preflight','cpu'):
            argv+=['-m','pytest','-q','-p','no:cacheprovider','tests/test_ua3d_anchor_lr_r2.py','-k','test_preflight' if kind=='preflight' else 'test_final']
        elif kind in ('cache_trainval','cache_test'):
            argv+=['scripts/prepare_ua3d_anchor_lr_r2.py','--task',task,'--fold',str(fold)]
            if kind=='cache_test':
                argv+=['--formal']
        else:
            argv+=['scripts/run_ua3d_anchor_lr_r2.py',kind,'--arm',arm]
            argv+=['--repeat',repeat] if kind=='diagnostic' else ['--task',task,'--fold',str(fold)]
        env=dict(CUDA_VISIBLE_DEVICES='2' if gpu else '',CUDA_DEVICE_ORDER='PCI_BUS_ID',WORLD_SIZE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                 PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='20261028',UA3M_PHYSICAL_GPU_UUID=t.read(t.OUT/'registry/gpu.json')['uuid'] if gpu else '',
                 ANCHOR_CPU_SUITE='preflight' if kind=='preflight' else 'final' if kind=='cpu' else '')
        j=dict(**t.contract(),name=name,identity=list(identity),kind=kind,arm=arm,task=task,fold=fold,repeat=repeat,pilot=pilot,
            cwd=str(ROOT),argv=argv,env=env,log=str(log),uses_gpu=gpu,memory_reserve_mib=self.memory_requirement(kind,arm,task,fold) if gpu else 0,
            estimated_seconds=estimated,reservation=reserve,consumption=str(folder/'consumption.json'),exit=None,reserved_unix=time.time())
        self.jobs.append(j); self.refresh(); t.exclusive(log/'command.json',j)
        cmd=['env','CUDA_VISIBLE_DEVICES=','WORLD_SIZE=1','PYTHONDONTWRITEBYTECODE=1',sys.executable,'-u','scripts/run_ua3d_anchor_lr_r2.py','wrap','--spec',str(log/'command.json')]
        j['window']=subprocess.check_output(['tmux','new-window','-d','-P','-F','#{window_id}','-t','Uni-Poly:','-n',name,'-c',str(ROOT),shlex.join(cmd)],text=True).strip()
        self.status('pilot' if pilot else kind,'已原子预留并派发 '+name); return True
    def execute(self,requests):
        pending=deque(requests)
        while pending or any(j['exit'] is None for j in self.jobs):
            self.guard('cpu' if pending and pending[0][0]=='cpu' or any(j['kind']=='cpu' and j['exit'] is None for j in self.jobs) else None)
            self.refresh(); self.failures()
            while pending:
                if not self.launch(*pending[0]):
                    break
                pending.popleft(); self.refresh(); self.failures()
            t.write(t.OUT/'scheduler/pending.json',dict(time=t.now(),requests=list(pending),deadline=t.DEADLINE))
            if pending or any(j['exit'] is None for j in self.jobs):
                time.sleep(15)
        self.refresh(); self.failures()
    def run(self):
        # The stdlib bootstrap encloses construction/import as well as this entire run.
        self.guard();self.status('register','继承原tN；r1 STOPPED/exit1/科学0只读保留；科学缓存P独立登记。');t.register()
        t.exclusive(t.OUT/'registry/gpu.json',self.inventory())
        self.execute([('preflight',None,None,None,None)])
        self.execute([('cache_trainval',None,task,fold,None) for task in t.TASKS for fold in range(5)])
        self.execute([('diagnostic',arm,None,None,label+'_'+r) for label in ('S','L') for arm in t.m.TRAIN_ARMS for r in ('R0','R1')])
        self.status('cost_gate','16诊断完成；静态全矩阵成本门禁');t.compare_diagnostics();t.guard(preparation=True);t.cost_gate(self.jobs);self.refresh()
        self.execute([('ft','G_GEOM_1E4','egc',1,None)]);t.pilot_gate(self.jobs)
        self.execute([('ft',arm,task,fold,None) for task in t.TASKS for fold in range(5) for arm in t.m.TRAIN_ARMS if (arm,task,fold)!=('G_GEOM_1E4','egc',1)])
        self.status('freeze','140FT完成；按validation唯一全局冻结');t.exclusive(t.OUT/'selection/freeze_reservation.json',dict(time_unix=time.time(),freezes=1));self.refresh();t.freeze();self.refresh()
        self.execute([('cache_test',None,task,fold,None) for task in t.TASKS for fold in range(5)])
        self.execute([('test','C_ANCHOR',task,fold,None) for task in t.TASKS for fold in range(5)]);t.anchor_test_gate()
        self.execute([('test',arm,task,fold,None) for task in t.TASKS for fold in range(5) for arm in t.m.TRAIN_ARMS])
        signal.setitimer(signal.ITIMER_REAL,max(.001,t.DEADLINE-300-time.time()));self.execute([('cpu',None,None,None,None)])
        self.refresh();self.manifest.update(status='COMPLETE_PENDING_REVIEW',controller_exit=0)
        self.manifest['stages']['delivery']=dict(status='RUNNING',attempts=1,exit=None,started=time.time(),ended=None)
        t.write(t.OUT/'execution_manifest.json',self.manifest);boot.stage(t.OUT,'report')
        result=t.report(self.jobs,None);self.manifest.update(execution_complete=result['execution_complete'],performance_pass=result['performance_pass'])
        self.manifest['stages']['delivery'].update(status='PASSED',exit=0,ended=time.time())
        self.status('closeout','受控执行完成；目标状态见report；independent_review=PENDING','COMPLETE_PENDING_REVIEW')
        self.capture();self.refresh()
        for kind,name in [('preflight','preflight_cpu_exit.json'),('cpu','cpu_exit.json')]:
            js=[j for j in self.jobs if j['kind']==kind]
            t.exclusive(t.OUT/name,dict(attempts=len(js),exit=js[0]['exit'] if js else None,status='NOT_RUN' if not js else 'PASSED' if js[0]['exit']==0 else 'FAILED'))
        t.exclusive(t.OUT/'controller_return.json',dict(exit=0,time=t.now(),process=t.process_identity(os.getpid())))
        signal.setitimer(signal.ITIMER_REAL,0);return 0


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=('controller','wrap','diagnostic','ft','test')); p.add_argument('--spec')
    p.add_argument('--arm',choices=t.m.ARMS); p.add_argument('--task',choices=t.TASKS); p.add_argument('--fold',type=int,choices=range(5)); p.add_argument('--repeat',choices=('S_R0','S_R1','L_R0','L_R1'))
    args=p.parse_args()
    if args.action=='wrap':
        return wrap(args.spec)
    if args.action=='controller':
        return Controller().run()
    if args.action=='test' and args.arm=='C_ANCHOR':
        t.anchor_output(args.task,args.fold); return 0
    setup=t.numerical.configure('DET')
    if args.action=='diagnostic':
        t.diagnose(args.arm,args.repeat,setup)
    else:
        t.unit(args.arm,args.task,args.fold,args.action=='test',setup)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
