#!/usr/bin/env python3
"""Single authorized continuation; immutable prior failure evidence, no job retry."""
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
import statistics
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.training import ua3d_ph_r2 as t

SOURCES=('src/dataset/ua3d_ph_r2.py','src/modules/ua3d_ph_r2.py','src/training/ua3d_ph_r2.py',
         'scripts/prepare_ua3d_ph_r2.py','scripts/run_ua3d_ph_r2.py','configs/ua3d_ph_r2_gpu2.json','tests/test_ua3d_ph_r2.py')
SPEC=t.read(ROOT/'configs/ua3d_ph_r2_gpu2.json')
CAPS=SPEC['caps']; GPU_KINDS=('diagnostic','ft','test')
BUDGET=t.OUT/'budget.continuation01.json'
MANIFEST=t.OUT/'execution_manifest.continuation01.json'


def pid_aliases(identity):
    if not t.process_alive(identity):
        return set()
    text=(Path('/proc')/str(identity['pid'])/'status').read_text()
    line=next((x for x in text.splitlines() if x.startswith('NSpid:')),None)
    return {int(x) for x in line.split()[1:]} if line else {identity['pid']}


def descendants(identity):
    if not t.process_alive(identity):
        return []
    found=[identity]; pending=[identity['pid']]
    while pending:
        pid=pending.pop(); p=Path('/proc')/str(pid)/'task'/str(pid)/'children'
        try:
            children=[int(x) for x in p.read_text().split()]
        except FileNotFoundError:
            continue
        for child in children:
            try:
                obj=t.process_identity(child)
            except FileNotFoundError:
                continue
            found.append(obj); pending.append(child)
    return found


def reservation(kind,arm=None,task=None,fold=None):
    if kind=='preflight':
        return {k:v for k,v in SPEC['limits'].items() if k.startswith('preflight_')}
    if kind=='prepare':
        return {k:v for k,v in SPEC['limits'].items() if k.startswith('prepare')}
    if kind=='cpu':
        return dict(cpu_suites=1)
    if kind=='diagnostic':
        value=dict(diagnostic_starts=1,diagnostic_train_starts=1,diagnostic_eval_forwards=2,diagnostic_eval_rows=48,diagnostic_train_forwards=20,
                   diagnostic_backwards=20,diagnostic_updates=20,diagnostic_train_rows=480)
        if arm=='P_LEARN':
            value['p_learn_reductions']=4*528
        return value
    e=t.read(t.SPLIT/f'{task}.json')['folds'][fold]
    if kind=='ft':
        n=len(e['train_indices']); v=len(e['validation_indices']); batches=70*math.ceil(n/24)
        value=dict(ft_starts=1,ft_epochs=70,ft_updates=batches,ft_train_forwards=batches,ft_train_rows=70*n,
                   validation_rows=70*v,validation_count=70,validation_forwards=70*math.ceil(v/24))
        if arm=='P_LEARN':
            value['p_learn_reductions']=4*70*(n+v)
        return value
    if kind=='test':
        n=len(e['test_indices']); value=dict(test_starts=1,test_rows=n,test_forwards=math.ceil(n/24))
        if arm=='P_LEARN':
            value['p_learn_reductions']=4*n
        return value
    raise ValueError('unregistered stage')


def ledger(jobs):
    limits=SPEC['limits']; reserved=dict.fromkeys(limits,0); actual=dict(reserved); complete=dict(reserved); started=dict(reserved)
    start_key=dict(preflight='preflight_cpu_suites',prepare='prepare',diagnostic='diagnostic_starts',ft='ft_starts',test='test_starts',cpu='cpu_suites')
    intervals=[]; process_seconds=0.
    for j in jobs:
        for k,v in j['reservation'].items():
            reserved[k]+=v
        marker=Path(j['log'])/'process.json'; key=start_key[j['kind']]
        if marker.exists():
            started[key]+=1; actual[key]+=1
            marker_value=t.read(marker); end=t.read(Path(j['log'])/'end.json')['ended_unix'] if (Path(j['log'])/'end.json').exists() else time.time()
            process_seconds+=max(0,end-marker_value['started_unix'])
            if j['kind'] in GPU_KINDS:
                intervals.append((marker_value['started_unix'],end))
        p=Path(j['consumption'])
        if p.exists():
            u=t.read(p); attempts=u.get('attempted',u); done=u.get('completed',u if u.get('status')=='COMPLETE' else {})
            for k in j['reservation']:
                if k==key:
                    continue
                actual[k]+=int(attempts.get(k,0))
                complete[k]+=int(done.get(k,attempts.get(k,0) if k=='p_learn_reductions' and u.get('status')=='COMPLETE' else 0))
            if key in attempts and not marker.exists():
                actual[key]+=int(attempts[key])
        if j['exit']==0:
            complete[key]+=1
    for dst in (reserved,actual,complete,started):
        dst['total_train_starts']=dst['ft_starts']+dst['diagnostic_train_starts']
        dst['total_updates']=dst['ft_updates']+dst['diagnostic_updates']
        dst['total_train_rows']=dst['ft_train_rows']+dst['diagnostic_train_rows']
    freeze=t.OUT/'selection/freeze_reservation.json'; frozen=t.OUT/'selection/final_frozen.json'
    reserved['freezes']=actual['freezes']=int(freeze.exists()); complete['freezes']=int(frozen.exists())
    violation=[k for k in limits if reserved[k]>limits[k] or actual[k]>reserved[k] or complete[k]>actual[k]]
    busy=0.; end=0.
    for a,b in sorted(intervals):
        busy+=max(0,b-max(a,end)); end=max(end,b)
    if busy>48*3600:
        violation.append('physical_gpu_busy_hours')
    return dict(**t.contract(),attempt='continuation01',limits=limits,reserved=reserved,started=started,attempted=actual,completed=complete,
                jobs=jobs,budget_violations=violation,physical_gpu_busy_hours=busy/3600,process_wall_seconds=process_seconds,
                parent_accounting=t.read(t.OUT/'parent_accounting.json'),prior_failed_activation=t.read(t.OUT/'failure.json'),
                cumulative_scientific_known_actual=actual,accounting='irreversible pre-dispatch reservations; failed/partial no refunds; parent unknown CPU counts separately retained',
                t0_unix=t.T0,deadline_unix=t.DEADLINE,time=t.now())


def wrap(spec):
    j=t.read(spec); folder=Path(j['log']); env=os.environ.copy(); env.update(j['env']); identity=None
    def stop(sig,frame):
        if identity is not None and t.process_alive(identity):
            os.killpg(identity['pid'],signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    with (folder/'stdout.log').open('x') as out,(folder/'stderr.log').open('x') as err:
        child=subprocess.Popen(j['argv'],cwd=j['cwd'],env=env,stdout=out,stderr=err,start_new_session=True)
        identity=t.process_identity(child.pid)
        t.exclusive(folder/'process.json',dict(wrapper=t.process_identity(os.getpid()),child=identity,argv=j['argv'],env=j['env'],cwd=j['cwd'],started_unix=time.time()))
        rc=child.wait()
    t.exclusive(folder/'end.json',dict(exit=rc,ended_unix=time.time()))
    with (folder/'exit.txt').open('x') as stream:
        stream.write(str(rc)+'\n')
    return rc


class Controller:
    def __init__(self):
        t.require_tmux()
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
            raise ValueError('controller must hide GPU')
        self.lock=(t.OUT/'continuation01.lock').open('a'); fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        t.exclusive(t.OUT/'authorization/start_continuation_01/controller_start.json',dict(process=t.process_identity(os.getpid()),time=t.now(),argv=sys.argv,cwd=str(ROOT),**t.contract()))
        self.jobs=[]; self.m=t.read(MANIFEST); self.error=None
        self.m.update(status='RUNNING',scientific_controller_started=True,controller_identity=t.process_identity(os.getpid()),controller_started=t.now())
        t.write(MANIFEST,self.m)
        sys.stdout=(t.LOG/'controller.continuation01.stdout.log').open('x',buffering=1)
        sys.stderr=(t.LOG/'controller.continuation01.stderr.log').open('x',buffering=1)
        self.capture_diff(); self.refresh()
        def cutoff(sig,frame):
            raise TimeoutError('inherited cutoff or controlled termination')
        signal.signal(signal.SIGALRM,cutoff); signal.signal(signal.SIGTERM,cutoff); signal.signal(signal.SIGINT,cutoff)
        signal.setitimer(signal.ITIMER_REAL,max(.001,t.DEADLINE-900-time.time()))

    def capture_diff(self):
        lines=[]
        for name in SOURCES:
            lines.extend(difflib.unified_diff([], (ROOT/name).read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+name))
        before=t.OUT/'authorization/start_continuation_01/CURRENT_PLAN.md'
        lines.extend(difflib.unified_diff(before.read_text().splitlines(True),(ROOT/'CURRENT_PLAN.md').read_text().splitlines(True),fromfile='before/CURRENT_PLAN.md',tofile='after/CURRENT_PLAN.md'))
        (t.OUT/'actual_changes.continuation01.diff').write_text(''.join(lines))

    def status(self,phase,detail,status='RUNNING'):
        self.m.update(status=status,phase=phase,detail=detail,updated=t.now()); t.write(MANIFEST,self.m)
        t.write(t.OUT/'status.continuation01.json',dict(status=status,phase=phase,detail=detail,time=t.now(),independent_review='PENDING'))
        p=ROOT/'CURRENT_PLAN.md'; text=p.read_text(); a='<!-- UA3D_PH_R2_STATUS_START -->'; b='<!-- UA3D_PH_R2_STATUS_END -->'
        left,rest=text.split(a,1); old,right=rest.split(b,1)
        # Preserve activation failure and the explicit continuation authorization in this block.
        tag='\n- continuation01实时状态：'
        old=old.split(tag,1)[0]
        p.write_text(left+a+old+tag+f'{t.now()}；{status}；{phase}；{detail}\n'+b+right)

    def refresh(self):
        for j in self.jobs:
            if j['exit'] is not None:
                continue
            folder=Path(j['log']); exit_path=folder/'exit.txt'
            if exit_path.exists():
                j['exit']=int(exit_path.read_text()); j['ended']=t.now()
            elif (folder/'process.json').exists() and not t.process_alive(t.read(folder/'process.json')['child']):
                marker=t.read(folder/'process.json')
                if not t.process_alive(marker['wrapper']):
                    raise RuntimeError('job exited without true exit marker: '+j['name'])
        value=ledger(self.jobs)
        with (t.OUT/'budget.continuation01.lock').open('a') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX); t.write(BUDGET,value)
        if value['budget_violations']:
            raise RuntimeError('budget violation '+repr(value['budget_violations']))
        return value

    def failures(self):
        bad=next((j for j in self.jobs if j['exit'] not in (None,0)),None)
        if bad:
            raise RuntimeError(f'{bad["name"]} true exit={bad["exit"]}; no repair/retry')

    def guard(self,kind=None):
        t.guard_deadline(300 if kind=='cpu' else 900)
        if shutil.disk_usage(ROOT).free<SPEC['min_free_bytes']:
            raise RuntimeError('disk below100GiB; no deletion')

    def inventory(self):
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=10)
        row=next((line for line in rows.splitlines() if line.split(',')[0].strip()=='2'),None)
        if row is None:
            raise ValueError('physical GPU2 absent')
        _,uuid,free=[x.strip() for x in row.split(',')]
        if uuid!=t.UUID:
            raise ValueError('physical GPU2 UUID changed')
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
        processes={}
        for line in apps.splitlines():
            cols=[x.strip() for x in line.split(',')]
            if cols[0]==uuid:
                processes[int(cols[1])]=int(cols[2]) if cols[2].isdigit() else 0
        return dict(uuid=uuid,free_mib=int(free),processes=processes)

    def available(self,kind):
        active=[j for j in self.jobs if j['exit'] is None]
        if any(j['kind']!=kind for j in active):
            raise ValueError('stage overlap')
        if len(active)>=CAPS[kind]:
            return False
        inv=self.inventory(); known=set(); unmaterialized=0
        for j in active:
            p=Path(j['log'])/'process.json'; aliases=set()
            if p.exists():
                for obj in descendants(t.read(p)['child']):
                    aliases.update(pid_aliases(obj))
            known.update(aliases)
            observed=sum(v for p,v in inv['processes'].items() if p in aliases)
            unmaterialized+=max(0,SPEC['admission_free_mib'][kind]-observed)
        unknown=set(inv['processes'])-known
        okay=not unknown and inv['free_mib']>=SPEC['admission_free_mib'][kind]+unmaterialized
        t.write(t.OUT/'occupancy_wait.continuation01.json',dict(time=t.now(),inventory=inv,known_pids=sorted(known),unknown_pids=sorted(unknown),unmaterialized_reserved_mib=unmaterialized,dispatch_allowed=okay))
        return okay

    def launch(self,kind,arm=None,task=None,fold=None,repeat=None):
        self.guard(kind); self.refresh(); self.failures()
        if kind=='cpu':
            t.guard_deadline(900)
        active=[j for j in self.jobs if j['exit'] is None]
        if kind in GPU_KINDS:
            if not self.available(kind):
                return False
        elif active:
            raise ValueError('CPU/GPU or CPU/CPU stage overlap')
        if any(j['kind']=='cpu' for j in self.jobs):
            raise ValueError('dispatch after final CPU')
        if kind=='prepare' and not any(j['kind']=='preflight' and j['exit']==0 for j in self.jobs):
            raise ValueError('prepare before preflight')
        if kind=='diagnostic' and not any(j['kind']=='prepare' and j['exit']==0 for j in self.jobs):
            raise ValueError('diagnostic before prepare')
        if kind=='ft' and t.read(t.OUT/'diagnostics/comparison.json')['numerical_diagnostic_pass'] is not True:
            raise ValueError('FT before diagnostics')
        if kind=='test' and not (t.OUT/'selection/final_frozen.json').exists():
            raise ValueError('TEST before freeze')
        if kind=='cpu' and sum(j['kind']=='test' and j['exit']==0 for j in self.jobs)!=245:
            raise ValueError('final CPU before245TEST')
        if kind in ('ft','diagnostic') and (t.OUT/'selection/final_frozen.json').exists():
            raise ValueError('training after freeze')
        name='ua3d-ph-r2-'+kind+('---'+arm if arm else '')+(f'-{task}-{fold}' if task else '')+('-'+repeat if repeat else '')
        if any(j['name']==name for j in self.jobs):
            raise ValueError('duplicate job dispatch')
        folder=t.job_path(kind,arm,task,fold,repeat)
        if folder.exists():
            raise FileExistsError('job output exists '+str(folder))
        amounts=reservation(kind,arm,task,fold); budget=self.refresh()
        if any(budget['reserved'][k]+v>SPEC['limits'][k] for k,v in amounts.items()):
            raise RuntimeError('insufficient fixed budget')
        observed=[t.read(j['consumption'])['elapsed_seconds'] for j in self.jobs if j['kind']==kind and j.get('arm')==arm and j.get('task')==task and j['exit']==0 and kind in GPU_KINDS]
        if observed and t.DEADLINE-time.time()<=900+max(observed):
            raise TimeoutError('observed normal unit duration cannot fit CPU/delivery reserves')
        log=t.LOG/name; log.mkdir(exist_ok=False)
        remaining=max(1,int(t.DEADLINE-(300 if kind=='cpu' else 900)-time.time()))
        argv=['timeout','--signal=TERM','--kill-after=30s',str(600 if kind in ('preflight','cpu') else remaining)+'s',sys.executable,'-u']
        if kind in ('preflight','cpu'):
            argv+=['-m','pytest','-q','-p','no:cacheprovider','tests/test_ua3d_ph_r2.py','-k','test_preflight' if kind=='preflight' else 'test_final']
        elif kind=='prepare':
            argv+=['scripts/prepare_ua3d_ph_r2.py']
        else:
            argv+=['scripts/run_ua3d_ph_r2.py',kind,'--arm',arm]
            argv+=['--repeat',repeat] if kind=='diagnostic' else ['--task',task,'--fold',str(fold)]
        env=dict(CUDA_VISIBLE_DEVICES='2' if kind in GPU_KINDS else '',CUDA_DEVICE_ORDER='PCI_BUS_ID',WORLD_SIZE='1',
                 OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
                 UA3M_PHYSICAL_GPU_UUID=t.UUID if kind in GPU_KINDS else '',PH_R2_SUITE='preflight' if kind=='preflight' else 'final' if kind=='cpu' else '',
                 CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='20261028')
        j=dict(name=name,kind=kind,arm=arm,task=task,fold=fold,repeat=repeat,argv=argv,env=env,cwd=str(ROOT),log=str(log),consumption=str(folder/'consumption.json'),
               reservation=amounts,exit=None,reserved_unix=time.time(),**t.contract())
        self.jobs.append(j); self.refresh(); t.exclusive(log/'command.json',j)
        cmd=['env','CUDA_VISIBLE_DEVICES=','WORLD_SIZE=1','PYTHONDONTWRITEBYTECODE=1',sys.executable,'-u','scripts/run_ua3d_ph_r2.py','wrap','--spec',str(log/'command.json')]
        j['window']=subprocess.check_output(['tmux','new-window','-d','-P','-F','#{window_id}','-t','Uni-Poly:','-n',name,'-c',str(ROOT),shlex.join(cmd)],text=True).strip()
        self.status(kind,'已原子预留并派发 '+name); self.refresh(); return True

    def execute(self,requests):
        pending=deque(requests)
        while pending or any(j['exit'] is None for j in self.jobs):
            self.guard('cpu' if pending and pending[0][0]=='cpu' or any(j['kind']=='cpu' and j['exit'] is None for j in self.jobs) else None)
            self.refresh(); self.failures()
            while pending:
                if not self.launch(*pending[0]):
                    break
                pending.popleft(); self.refresh(); self.failures()
            durations={}
            for j in self.jobs:
                if j['exit']==0 and j['kind'] in GPU_KINDS:
                    u=t.read(j['consumption']); durations.setdefault(j['kind']+'/'+str(j['task']),[]).append(u['elapsed_seconds'])
            known=sum(float(statistics.mean(durations[k+'/'+str(task)])) for k,arm,task,fold,repeat in pending if k+'/'+str(task) in durations) if durations else 0.
            t.write(t.OUT/'scheduler/pending.continuation01.json',dict(time=t.now(),requests=list(pending),observed_durations=durations,
                eta_known_work_seconds=known,eta_limit='normal completed jobs only; unseen tasks/PH branch costs/waiting/test/CPU excluded; not a completion promise'))
            if pending or any(j['exit'] is None for j in self.jobs):
                time.sleep(3)
        self.refresh(); self.failures()

    def stop_owned(self):
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig in (signal.SIGINT,signal.SIGTERM):
            signal.signal(sig,signal.SIG_IGN)
        for j in self.jobs:
            p=Path(j['log'])/'process.json'
            if j['exit'] is None and p.exists():
                identity=t.read(p)['child']
                if t.process_alive(identity):
                    os.killpg(identity['pid'],signal.SIGTERM)
        until=time.time()+35
        while any(j['exit'] is None for j in self.jobs) and time.time()<until:
            self.refresh(); time.sleep(1)
        for j in self.jobs:
            p=Path(j['log'])/'process.json'
            if j['exit'] is None and p.exists():
                identity=t.read(p)['child']
                if t.process_alive(identity):
                    os.killpg(identity['pid'],signal.SIGKILL)
        self.refresh()

    def run(self):
        rc=1; error=None
        try:
            self.guard(); self.status('register','显式接续启动；旧失败文件只读保留。'); t.register()
            self.execute([('preflight',None,None,None,None)])
            self.execute([('prepare',None,None,None,None)])
            self.execute([('diagnostic',arm,None,None,repeat) for arm in t.m.ARMS for repeat in ('R0','R1')])
            t.compare_diagnostics(); t.stability_gate()
            self.execute([('ft',arm,task,fold,None) for task in t.TASKS for fold in range(5) for arm in t.m.ARMS])
            self.guard(); t.exclusive(t.OUT/'selection/freeze_reservation.json',dict(time=t.now(),freezes=1)); self.refresh(); t.freeze(); self.refresh()
            self.execute([('test',arm,task,fold,None) for task in t.TASKS for fold in range(5) for arm in t.m.ARMS])
            for arm in t.m.ARMS:
                t.aggregate(arm,True)
            signal.setitimer(signal.ITIMER_REAL,max(.001,t.DEADLINE-300-time.time()))
            self.execute([('cpu',None,None,None,None)]); rc=0
        except BaseException:
            error=traceback.format_exc()
            try:
                self.stop_owned()
            except BaseException:
                error+='\ncleanup error:\n'+traceback.format_exc()
            t.exclusive(t.OUT/'failure.continuation01.json',dict(time=t.now(),error=error,retry=False,resume=False,repair=False))
        finally:
            signal.setitimer(signal.ITIMER_REAL,max(.001,t.DEADLINE-time.time()))
        try:
            result=t.report(self.jobs,error)
            self.m.update(execution_complete=result['execution_complete'],performance_pass=result['performance_pass'],preflight_cpu_pass=result['preflight_cpu_pass'],final_cpu_pass=result['final_cpu_pass'])
        except BaseException:
            rc=1; error=(error or '')+'\nreport error:\n'+traceback.format_exc()
            t.exclusive(t.OUT/'report_failure.continuation01.json',dict(error=error))
        self.status('closeout','controller真实exit='+str(rc)+'；'+('固定执行结束，待审查' if rc==0 else '失败即停；未执行范围不可比较'), 'COMPLETE_PENDING_REVIEW' if rc==0 else 'STOPPED')
        self.capture_diff(); self.refresh()
        for kind,name in (('preflight','preflight_cpu_exit.continuation01.json'),('cpu','cpu_exit.continuation01.json')):
            js=[j for j in self.jobs if j['kind']==kind]
            t.exclusive(t.OUT/name,dict(attempts=len(js),exit=js[0]['exit'] if js else None,status='NOT_RUN' if not js else 'COMPLETE' if js[0]['exit']==0 else 'FAILED',log=js[0]['log'] if js else None))
        t.exclusive(t.OUT/'controller_exit.continuation01.json',dict(exit=rc,time=t.now(),process=t.process_identity(os.getpid()),error=error))
        signal.setitimer(signal.ITIMER_REAL,0)
        return rc


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=('controller','wrap','diagnostic','ft','test'))
    p.add_argument('--arm',choices=t.m.ARMS); p.add_argument('--task',choices=t.TASKS); p.add_argument('--fold',type=int,choices=range(5))
    p.add_argument('--repeat',choices=('R0','R1')); p.add_argument('--spec'); args=p.parse_args()
    if args.action=='wrap':
        return wrap(args.spec)
    if args.action=='controller':
        return Controller().run()
    setup=t.numerical.configure('DET')
    if args.action=='diagnostic':
        t.diagnostic(args.arm,args.repeat,setup)
    else:
        t.unit(args.arm,args.task,args.fold,args.action=='test',setup)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
