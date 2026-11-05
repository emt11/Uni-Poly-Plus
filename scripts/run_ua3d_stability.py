#!/usr/bin/env python3
"""One-shot DSH GPU2 strict stability fixed campaign; no retry/resume/repair."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(key,'2')
import argparse
from collections import deque
import datetime
import difflib
import fcntl
import json
import math
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training import ua3d_stability as t
from scripts.run_ua2d_a020_fp_fusion import atomic_exit,poll_exit,owned_group,pid_aliases
SOURCES=('src/modules/ua3d_stability.py','src/training/ua3d_stability.py','scripts/run_ua3d_stability.py','configs/ua3d_stability_r1.json','tests/test_ua3d_stability.py')
CAPS={'diagnostic':1,'ft':4,'test':4};MEMORY={'diagnostic':8192,'ft':8192,'test':4096}

class Deadline(RuntimeError):pass

def reservation(kind,task=None,fold=None):
    if kind=='register':return dict(registrations=1,added_seed_groups=2,same_seed_repeat_groups=1)
    if kind=='diagnostic':return dict(diagnostic_starts=1,diagnostic_train_starts=1,diagnostic_eval_forwards=2,diagnostic_eval_graphs=48,diagnostic_train_forwards=20,diagnostic_backwards=20,diagnostic_updates=20,diagnostic_train_graphs=480)
    if kind=='cpu':return dict(cpu_suites=1)
    e=t.read(t.SPLIT/f'{task}.json')['folds'][fold]
    if kind=='ft':return dict(ft_starts=1,ft_epochs=70,ft_updates=70*math.ceil(len(e['train_indices'])/24),train_graphs=70*len(e['train_indices']),validation_started=70,validation_graphs=70*len(e['validation_indices']))
    if kind=='test':return dict(test_starts=1,test_graphs=len(e['test_indices']))
    raise ValueError('unregistered phase')

def ledger(jobs):
    charged=dict.fromkeys(t.LIMITS,0);actual=dict(charged);completed=dict(charged)
    for j in jobs:
        for k,v in j['reservation'].items():charged[k]+=v
        start={'register':'registrations','diagnostic':'diagnostic_starts','ft':'ft_starts','test':'test_starts','cpu':'cpu_suites'}[j['kind']]
        if (Path(j['log'])/'process.json').exists():actual[start]+=1
        if j['exit']==0:completed[start]+=1
        p=Path(j['consumption']) if j.get('consumption') else None
        if p is not None and p.exists():
            u=t.read(p)
            for key in j['reservation']:
                if key==start:continue
                a,b={'ft_epochs':('epochs_started','completed_epochs'),'ft_updates':('attempted_updates','completed_updates')}.get(key,(key,'completed_'+key if key.startswith('diagnostic_') else key))
                actual[key]+=int(u.get(a,0));completed[key]+=int(u.get(b,u.get(key,0) if u.get('status')=='COMPLETE' else 0))
    for key,group in (('added_seed_groups',('S142','S242')),('same_seed_repeat_groups',('S42_B',))):
        for run in group:
            js=[j for j in jobs if j['kind']=='ft' and j['arm']==run]
            actual[key]+=int(any((Path(j['log'])/'process.json').exists() for j in js))
            completed[key]+=int(len(js)==35 and all(j['exit']==0 for j in js))
    for key,pattern,target in (('freezes','selection/freeze_reservation.json','selection/final_frozen.json'),('diagnostic_comparisons','diagnostics/comparison_reservation.json','diagnostics/comparison.json')):
        charged[key]=actual[key]=int((t.OUT/pattern).exists());completed[key]=int((t.OUT/target).exists())
    charged['repro_pairs']=actual['repro_pairs']=len(list((t.OUT/'reproducibility').glob('*/fold*.reservation.json')))
    completed['repro_pairs']=len(list((t.OUT/'reproducibility').glob('*/fold[0-4].json')))
    for dest in (charged,actual,completed):
        dest['total_train_starts']=dest['diagnostic_train_starts']+dest['ft_starts']
        dest['total_updates']=dest['diagnostic_updates']+dest['ft_updates']
        dest['total_train_graphs']=dest['diagnostic_train_graphs']+dest['train_graphs']
    bad=[k for k in charged if charged[k]>t.LIMITS[k] or actual[k]>charged[k] or completed[k]>actual[k]]
    return dict(limits=t.LIMITS,charged_or_reserved=charged,known_actual_lower_bound=actual,known_completed=completed,budget_violations=bad,jobs=jobs,
        accounting='full atomic pre-dispatch reservation; diagnostic/FT/comparison/seed accounting separate; failed/partial reservations never refunded')

def gpu_inventory():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True,timeout=10)
    row=[x.split(',') for x in rows.splitlines() if x.split(',')[0].strip()=='2']
    if len(row)!=1:raise ValueError('physical GPU2 inventory mismatch')
    _,uuid,free,total=[x.strip() for x in row[0]]
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
    processes={}
    for line in apps.splitlines():
        cols=[x.strip() for x in line.split(',')]
        if cols[0]==uuid:processes[int(cols[1])]=int(cols[2]) if cols[2].isdigit() else 0
    return dict(uuid=uuid,free_mib=int(free),total_mib=int(total),processes=processes,time=t.now())

def slot_available(kinds,kind):
    if kind not in CAPS:raise ValueError('GPU phase required')
    if any(x!=kind for x in kinds):raise ValueError('FT/TEST phase mixing')
    return len(kinds)<CAPS[kind]

def wrap(spec):
    j=t.read(spec);folder=Path(j['log']);env=os.environ.copy();env.update(j['env']);child=None;identity=None
    def stop(sig,frame):
        if identity is not None and t.process_alive(identity):os.killpg(identity['pid'],signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (folder/'stdout.log').open('x') as out,(folder/'stderr.log').open('x') as err:
        child=subprocess.Popen(j['argv'],cwd=j['cwd'],env=env,stdout=out,stderr=err,start_new_session=True)
        identity=t.process_identity(child.pid)
        t.write(folder/'process.json',dict(wrapper=t.process_identity(os.getpid()),timeout_identity=identity,command=j['argv'],env=j['env'],cwd=j['cwd'],started=t.now(),gpu_uuid=j['gpu_uuid']))
        rc=child.wait()
    atomic_exit(folder/'exit.txt',rc);t.write(folder/'end.json',dict(exit=rc,ended=t.now()))
    return rc

class Controller:
    def __init__(self):
        t.require_tmux()
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('controller must hide GPU')
        if os.path.lexists(t.OUT) or os.path.lexists(t.LOG):raise FileExistsError('root already exists; stop, no alternative root')
        t.check_space()
        self.started=t.SPEC['t0_unix'];self.deadline=self.started+86400;self.jobs=[]
        if time.time()>=self.deadline-900:raise Deadline('implementation consumed dispatch time')
        t.OUT.mkdir(parents=True,exist_ok=False);t.LOG.mkdir(parents=True,exist_ok=False)
        sys.stdout=(t.LOG/'controller.stdout.log').open('x',buffering=1);sys.stderr=(t.LOG/'controller.stderr.log').open('x',buffering=1)
        self.lock=(t.OUT/'controller.lock').open('x');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.manifest=dict(plan=t.PLAN,revision=t.REVISION,status='执行中',t0_unix=self.started,deadline_unix=self.deadline,
            started='2026-10-28T21:42:21+08:00',deadline='2026-10-29T21:42:21+08:00',t0_utc=t.SPEC['t0_utc'],deadline_utc=t.SPEC['deadline_utc'],execution_role='DSH',controller_started=t.now(),
            controller_identity=t.process_identity(os.getpid()),cwd=str(ROOT),argv=sys.argv,env={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','WORLD_SIZE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
            physical_gpus=[2],logical_device='cuda:0',world_size=1,diagnostic_max_concurrency=1,ft_max_concurrency=4,test_max_concurrency=4,phase_exclusive=True,
            cpu_reserve_seconds=600,delivery_reserve_seconds=300,automatic_retry=False,automatic_resume=False,independent_review='PENDING',based_on=t.SPEC['based_on'])
        t.write(t.OUT/'execution_manifest.json',self.manifest)
        before=(ROOT/'CURRENT_PLAN.md').read_text();(t.OUT/'authorization').mkdir()
        (t.OUT/'authorization/CURRENT_PLAN.before.md').write_text(before)
        prompt='用户2026-10-28明确授权UA3D-STABILITY-20261028-01/r1完整固定周期：GPU2诊断1路、FT4路、TEST4路，阶段互斥；4个GPU诊断/80更新；140FT、35对精确比较；S42_A/S142/S242共105TEST，S42_B不TEST；最后唯一CPU；24小时含实现等待交付；严格DET不降级；失败即停，无retry/resume/repair/扩scope，保持原BASE和待独立审查。'
        (t.OUT/'authorization/user_authorization.txt').write_text(prompt+'\n')
        t.exclusive(t.OUT/'authorization/authorization.json',dict(source='user explicitly authorized STABILITY/r1 complete fixed cycle and all six detailed constraints in conversation',execution_role='DSH',plan=t.PLAN,revision=t.REVISION,t0_unix=self.started,t0_policy='first related check after this explicit authorization, includes implementation and waiting',limits=t.LIMITS,physical_gpus=[2],todo_read=False,remote_sync=False,new_file_hash_contract=False))
        t.write(t.OUT/'authorization/protected_status.json',dict(status=subprocess.check_output(['git','status','--short'],text=True)))
        for name in SOURCES:
            p=t.OUT/'authorization/source_at_launch'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text((ROOT/name).read_text())
        self.capture_diff();self.refresh()
        def cutoff(sig,frame):raise Deadline('hard clock delivery reserve reached')
        signal.signal(signal.SIGALRM,cutoff);signal.setitimer(signal.ITIMER_REAL,max(.001,self.deadline-900-time.time()))

    def capture_diff(self):
        lines=[]
        for name in SOURCES:lines.extend(difflib.unified_diff([], (ROOT/name).read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+name))
        before=t.OUT/'authorization/CURRENT_PLAN.before.md'
        if before.exists():lines.extend(difflib.unified_diff(before.read_text().splitlines(True),(ROOT/'CURRENT_PLAN.md').read_text().splitlines(True),fromfile='before/CURRENT_PLAN.md',tofile='after/CURRENT_PLAN.md'))
        p=t.OUT/'actual_changes.diff';p.write_text(''.join(lines))

    def guard_clock(self,kind=None):
        reserve=300 if kind is None else 900
        if self.deadline-time.time()<=reserve:raise Deadline('insufficient remaining CPU/delivery reserve')

    def refresh(self):
        for j in self.jobs:
            if j['exit'] is None:
                rc=poll_exit(j)
                if rc is not None:j.update(exit=rc,ended=t.now())
        value=ledger(self.jobs);active=[j['kind'] for j in self.jobs if j['exit'] is None]
        value.update(time=t.now(),elapsed_seconds=time.time()-self.started,current_slots={k:active.count(k) for k in ('register','diagnostic','ft','test','cpu')},deadline_unix=self.deadline)
        t.write(t.OUT/'budget.json',value)
        if value['budget_violations']:raise RuntimeError('budget violation '+repr(value['budget_violations']))
        return value

    def failures(self):
        bad=[j for j in self.jobs if j['exit'] not in (None,0)]
        if bad:raise RuntimeError(f"{bad[0]['name']} true exit={bad[0]['exit']}; no retry/resume/repair")

    def available(self,kind):
        active=[j for j in self.jobs if j['exit'] is None]
        if not slot_available([j['kind'] for j in active],kind):return None
        inventory=gpu_inventory();known=set();unmaterialized=0
        for j in active:
            process=Path(j['log'])/'process.json';aliases=set()
            if process.exists():
                identity=t.read(process)['timeout_identity'];aliases=pid_aliases(identity)
                usage=Path(j['consumption'])
                if usage.exists() and t.read(usage)['process']!=identity:raise ValueError('child identity mismatch')
            known.update(aliases);observed=sum(v for p,v in inventory['processes'].items() if p in aliases)
            unmaterialized+=max(0,MEMORY[kind]-observed)
        unknown=set(inventory['processes'])-known
        enough=inventory['free_mib']>=MEMORY[kind]+unmaterialized
        t.write(t.OUT/'occupancy_wait.json',dict(inventory=inventory,unknown_pids=sorted(unknown),known_pids=sorted(known),dispatch_allowed=not unknown and enough,unmaterialized_reserved_mib=unmaterialized))
        return inventory if not unknown and enough else None

    def launch(self,kind,arm=None,task=None,fold=None):
        self.guard_clock(kind);t.check_space();self.refresh();self.failures()
        frozen=t.OUT/'selection/final_frozen.json'
        if any(j['kind']=='cpu' for j in self.jobs):raise RuntimeError('dispatch after sole CPU suite')
        if kind in ('diagnostic','ft') and frozen.exists():raise RuntimeError('diagnostic/FT after freeze')
        if kind=='ft' and t.read(t.OUT/'diagnostics/comparison.json')['numerical_diagnostic_pass'] is not True:raise RuntimeError('FT before DET diagnostic gate')
        if kind=='diagnostic' and any(j['kind']=='ft' for j in self.jobs):raise RuntimeError('late diagnostic forbidden')
        if kind in ('test','cpu') and not frozen.exists():raise RuntimeError('early TEST/CPU')
        if kind in CAPS:
            if kind=='diagnostic':
                if arm not in t.DIAGNOSTICS:raise ValueError('unregistered diagnostic')
            else:
                if arm not in t.RUNS:raise ValueError('unregistered run')
                t.config(arm,task,fold)
            if kind=='test':t.frozen_unit(t.read(frozen),arm,task,fold)
            inventory=self.available(kind)
            if inventory is None:return None
        else:
            if any(j['exit'] is None for j in self.jobs):raise RuntimeError('CPU/registration overlap')
            inventory=None
        if kind=='cpu' and len([j for j in self.jobs if j['kind']=='test' and j['exit']==0])!=105:raise RuntimeError('CPU before all TEST')
        name='stability_r1_'+kind+(f'_{arm}_{task}_{fold}' if arm else '')
        if any(j['name']==name for j in self.jobs):raise ValueError('duplicate attempt')
        if kind in CAPS:
            seen=[t.read(j['consumption'])['elapsed_seconds'] for j in self.jobs if j['kind']==kind and j['task']==task and j['exit']==0]
            if seen and self.deadline-time.time()<=900+max(seen):raise Deadline('observed normal unit duration cannot fit CPU/delivery reserves')
        reserve=reservation(kind,task,fold);charged=self.refresh()['charged_or_reserved']
        if any(charged[k]+v>t.LIMITS[k] for k,v in reserve.items()):raise RuntimeError('budget unavailable')
        output=t.OUT/'diagnostics'/arm if kind=='diagnostic' else t.unit_path(arm,task,fold,kind=='test') if kind in ('ft','test') else None
        if output is not None and output.exists():raise FileExistsError('writer/overwrite risk '+str(output))
        log=t.LOG/name;log.mkdir(exist_ok=False)
        if kind=='cpu':argv=['timeout','--signal=TERM','--kill-after=30s','600s',sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_ua3d_stability.py']
        else:
            argv=[sys.executable,'-u','scripts/run_ua3d_stability.py',kind]
            if arm:
                argv+=['--arm',arm]
                if kind!='diagnostic':argv+=['--task',task,'--fold',str(fold)]
        env=dict(CUDA_VISIBLE_DEVICES='2' if kind in CAPS else '',CUDA_DEVICE_ORDER='PCI_BUS_ID',WORLD_SIZE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',UA3M_PHYSICAL_GPU_UUID=inventory['uuid'] if inventory else '',UA3D_STABILITY_JOB_KIND=kind)
        if kind in ('ft','test') or (kind=='diagnostic' and arm.startswith('DET_')):env.update(t.m.DET_ENV)
        job=dict(name=name,kind=kind,arm=arm,task=task,fold=fold,argv=argv,env=env,cwd=str(ROOT),log=str(log),reservation=reserve,
            consumption=str(output/'consumption.json') if output else None,exit=None,reserved_at=t.now(),gpu_uuid=env['UA3M_PHYSICAL_GPU_UUID'],physical_gpus=[2] if kind in CAPS else [],world_size=1,dispatch_pending=True,attempt='r1-once')
        self.jobs.append(job);self.refresh();t.write(log/'command.json',job)
        cmd=['env','CUDA_VISIBLE_DEVICES=','WORLD_SIZE=1','PYTHONDONTWRITEBYTECODE=1',sys.executable,'-u','scripts/run_ua3d_stability.py','wrap','--spec',str(log/'command.json')]
        job['window']=subprocess.check_output(['tmux','new-window','-d','-P','-F','#{window_id}','-t','Uni-Poly:','-n',name,'-c',str(ROOT),shlex.join(cmd)],text=True).strip()
        subprocess.run(['tmux','set-option','-w','-t',job['window'],'remain-on-exit','on'],check=True)
        pane=int(subprocess.check_output(['tmux','display-message','-p','-t',job['window'],'#{pane_pid}'],text=True))
        job['wrapper_identity']=t.process_identity(pane);job['dispatch_pending']=False;self.refresh()
        t.update_plan('执行中',kind,f'已派发{len(self.jobs)}个固定作业；当前 {name}；真实exit见独立账本。')
        return job

    def execute(self,requests):
        pending=deque(requests)
        while pending or any(j['exit'] is None for j in self.jobs):
            self.guard_clock();t.check_space();self.refresh();self.failures()
            while pending:
                job=self.launch(*pending[0])
                if job is None:break
                pending.popleft();self.refresh();self.failures()
            durations={}
            for j in self.jobs:
                if j['exit']==0 and j['kind'] in CAPS:
                    usage=t.read(j['consumption']);durations.setdefault(j['kind']+'/'+str(j['task']),[]).append(usage['elapsed_seconds'])
            eta=sum(float(t.np.mean(durations[k+'/'+str(task)])) for k,arm,task,fold in pending if k+'/'+str(task) in durations)/4
            t.write(t.OUT/'scheduler/pending.json',dict(time=t.now(),requests=list(pending),observed_durations=durations,eta_known_pending_seconds=eta,
                eta_limitation='normal completed jobs only; unseen tasks, active jobs, waiting, registration/freeze/TEST/CPU overhead excluded; not a completion promise'))
            if pending or any(j['exit'] is None for j in self.jobs):time.sleep(3)
        self.refresh();self.failures()

    def stop_owned(self):
        t.ignore_termination();signal.setitimer(signal.ITIMER_REAL,0)
        for j in self.jobs:
            if j['exit'] is not None:continue
            group=owned_group(j)
            if group is not None:
                try:os.killpg(group,signal.SIGTERM)
                except ProcessLookupError:pass
        until=time.time()+35
        while any(j['exit'] is None for j in self.jobs) and time.time()<until:
            self.refresh();time.sleep(1)
        for j in self.jobs:
            if j['exit'] is None:
                group=owned_group(j)
                if group is not None:
                    try:os.killpg(group,signal.SIGKILL)
                    except ProcessLookupError:pass
        until=time.time()+10
        while any(j['exit'] is None for j in self.jobs) and time.time()<until:self.refresh();time.sleep(1)

    def report(self,status,error):return report_campaign(self,status,error)

    def run(self):
        rc=1;status='失败停止，待审查';error=None;result=None
        try:
            t.update_plan('执行中','登记','独立新根已创建；固定诊断和完整稳定性矩阵，无额外forward或提前CPU。')
            self.execute([('register',None,None,None)])
            for name in t.DIAGNOSTICS:self.execute([('diagnostic',name,None,None)])
            t.exclusive(t.OUT/'diagnostics/comparison_reservation.json',dict(time=t.now(),reserved=1,once=True));self.refresh();t.compare_diagnostics();self.refresh()
            for task in t.TASKS:
                for fold in range(5):
                    self.execute([('ft',run,task,fold) for run in t.RUNS])
                    t.check_space();self.guard_clock('ft')
                    t.exclusive(t.OUT/'reproducibility'/task/f'fold{fold}.reservation.json',dict(time=t.now(),reserved=1,once=True))
                    self.refresh();t.compare_pair(task,fold);self.refresh()
            self.guard_clock('test');t.exclusive(t.OUT/'selection/freeze_reservation.json',dict(time=t.now(),reserved=1,once=True));self.refresh();frozen=t.freeze();self.refresh()
            t.update_plan('执行中','唯一冻结','DET_BASE / PRIMARY_RUN=S42_A固定；35对精确复现通过；仅105个canonical BEST允许TEST。')
            self.execute([('test',run,task,fold) for task in t.TASKS for fold in range(5) for run in t.CANONICAL])
            for run in t.CANONICAL:t.aggregate(run,True)
            signal.setitimer(signal.ITIMER_REAL,max(.001,self.deadline-300-time.time()))
            self.execute([('cpu',None,None,None)])
            rc=0;status='完成，待审查'
        except BaseException:
            error=traceback.format_exc()
            if isinstance(sys.exc_info()[1],Deadline):status='截止停止，部分完成，待审查'
            try:self.stop_owned()
            except BaseException:error+='\nowned cleanup failure: '+traceback.format_exc()
            t.write(t.OUT/'failure.json',dict(error=error,time=t.now(),retry=False,resume=False,repair=False))
        finally:signal.setitimer(signal.ITIMER_REAL,max(.001,self.deadline-time.time()))
        try:result=self.report(status,error)
        except BaseException:
            error=(error or '')+'\nreport failure: '+traceback.format_exc();rc=1;status='失败停止，待审查'
            t.write(t.OUT/'final_review/report_failure.json',dict(error=error,time=t.now()))
        self.manifest.update(status=status,ended=t.now(),elapsed_seconds=time.time()-self.started,error=error,execution_detail=result.get('execution_detail') if result else '交付阻断')
        t.write(t.OUT/'execution_manifest.json',self.manifest)
        t.update_plan(status,'交付' if rc==0 else '异常停止',f'controller exit={rc}；证据 {t.OUT.relative_to(ROOT)}/final_review/results.json；'+('固定执行结束；稳定性/性能字段分别验收，仍待独立审查。' if rc==0 else '后续未执行范围见交付；不自动修复/重试或提前补CPU。'))
        (t.OUT/'closeout.md').write_text(f'状态：{status}；independent_review=PENDING。\ncontroller真实exit={rc}。\n完成层级：'+(result.get('execution_detail') if result else '交付阻断')+'。\n诊断、FT、精确比较、TEST与CPU消费及真实exit见budget.json和logs。\n'+('下一步：独立审查。\n' if rc==0 else '后续执行停止；下一步只读审查失败证据，修复/恢复需新的明确授权。\n')+(error or '阻断：无。')+'\n')
        self.capture_diff();t.exclusive(t.OUT/'controller_exit.json',dict(exit=rc,time=t.now(),process=t.process_identity(os.getpid())))
        atomic_exit(t.LOG/'controller.exit.txt',rc)
        return rc

def report_campaign(controller,status,error):
    budget=controller.refresh();validation={};formal={};costs={};sota={};registered=(t.OUT/'registry/recipes.json').exists()
    for run in t.ORDER:
        for dest,flag in ((validation,False),(formal,True)):
            if not registered or (flag and run=='S42_B'):continue
            try:dest[run]=t.aggregate(run,flag)
            except (FileNotFoundError,ValueError,KeyError):pass
        costs[run]=[]
        for task in t.TASKS:
            for fold in range(5):
                p=t.unit_path(run,task,fold)
                if (p/'consumption.json').exists():
                    met=t.read(p/'metrics.json') if (p/'metrics.json').exists() else {}
                    costs[run].append(dict(task=task,fold=fold,consumption=t.read(p/'consumption.json'),best_epoch=met.get('best_epoch'),best_stage=met.get('best_stage'),history=str(p/'history.json'),alias=run=='BASE_ORIGINAL'))
        if run in formal:
            sota[run]={task:dict(project=formal[run]['tasks'][task],base=formal['BASE_ORIGINAL']['tasks'][task],paper_mean=t.SOTA[task][2],paper_std={'eea':.013,'egb':.009,'egc':.008,'ei':.018,'eps':.033,'nc':.032,'xc':.102}[task],paper_model=t.SOTA[task][1],delta=formal[run]['tasks'][task]['mean']-t.SOTA[task][2],source='SOTA_RESULT.md') for task in t.TASKS}
    diagnostics={}
    for name in t.DIAGNOSTICS:
        p=t.OUT/'diagnostics'/name
        diagnostics[name]=dict(consumption=t.read(p/'consumption.json') if (p/'consumption.json').exists() else None,
            failure=t.read(p/'failure.json') if (p/'failure.json').exists() else None,source=str(p))
    diagnostic_file=t.OUT/'diagnostics/comparison.json';dc=t.read(diagnostic_file) if diagnostic_file.exists() else None
    det_failed=any(j['kind']=='diagnostic' and j['arm'].startswith('DET_') and j['exit'] not in (None,0) for j in controller.jobs)
    diagnostic_pass=dc['numerical_diagnostic_pass'] if dc else False if det_failed or any(diagnostics[n]['failure'] for n in ('DET_R0','DET_R1')) else None
    pairs={}
    for task in t.TASKS:
        for fold in range(5):
            p=t.OUT/'reproducibility'/task/f'fold{fold}.json'
            if p.exists():pairs[f'{task}/{fold}']=t.read(p)
    same_pass=False if any(not x['equal'] for x in pairs.values()) else True if len(pairs)==35 else None
    ft_complete=all(run in validation for run in t.RUNS);test_complete=all(run in formal for run in t.CANONICAL)
    cross=None;cross_pass=None
    if test_complete:cross,cross_pass=t.seed_statistics(formal)
    flags=[diagnostic_pass,same_pass,cross_pass]
    stability_pass=False if any(x is False for x in flags) else True if all(x is True for x in flags) else None
    primary=formal.get('S42_A');reference=formal.get('BASE_ORIGINAL');quality=None;performance=None;delta=None
    if primary and reference:quality,performance,delta=t.primary_assessment(formal)
    ready=False if stability_pass is False or (quality and quality['pass'] is False) else True if stability_pass is True and quality and quality['pass'] else None
    cpu_jobs=[j for j in controller.jobs if j['kind']=='cpu'];cpu=dict(attempts=len(cpu_jobs),exit=cpu_jobs[0]['exit'] if cpu_jobs else None,log=cpu_jobs[0]['log'] if cpu_jobs else None,status='NOT_RUN' if not cpu_jobs else 'COMPLETE' if cpu_jobs[0]['exit']==0 else 'FAILED')
    t.write(t.OUT/'cpu_exit.json',cpu)
    execution_complete=ft_complete and test_complete and diagnostic_pass is True and same_pass is True and cpu['exit']==0 and not error
    detail='完成固定执行，待独立审查' if execution_complete else '训练/TEST完整，最终验证失败，整体未验收' if ft_complete and test_complete and cpu['exit'] not in (None,0) else '部分完成，正式性能未完成/不可比较'
    if execution_complete and stability_pass is False:detail+='；跨seed稳定性验收未通过'
    if execution_complete and stability_pass is True and quality and not quality['pass']:detail+='；数值稳定但性能保留未通过'
    frozen=t.read(t.OUT/'selection/final_frozen.json') if (t.OUT/'selection/final_frozen.json').exists() else None
    result=dict(plan=t.PLAN,revision=t.REVISION,execution_role='DSH',execution_status=status,execution_detail=detail,FINAL_recipe='DET_BASE',PRIMARY_RUN='S42_A',
        numerical_diagnostic_pass=diagnostic_pass,same_seed_repro_pass=same_pass,cross_seed_stability_pass=cross_pass,stability_pass=stability_pass,
        quality_guard_pass=quality['pass'] if quality else None,stable_ready=ready,performance_pass=performance,
        primary_test_macro=primary['macro'] if primary else None,primary_delta_macro=delta,quality_guard=quality,cross_seed=cross,
        ft_complete=ft_complete,formal_test_complete=test_complete,final_cpu_pass=cpu['exit']==0,execution_complete=execution_complete,independent_review='PENDING',
        reference=t.REFERENCE,reference_id=t.REF_ID,reference_source=str(t.REF),validation=validation,formal=formal,
        partial={run:'未完成/不可比较' for run in t.CANONICAL if run not in formal},diagnostics=diagnostics,diagnostic_comparison=dc,reproducibility=pairs,costs=costs,sota=sota,
        sota_all7_pass=all(v['delta']>=0 for v in sota['S42_A'].values()) if 'S42_A' in sota else None,
        budget=budget,cpu=cpu,error=error,freeze_time=frozen['time'] if frozen else None,
        evaluation='shared development TEST; same seeds/folds and hardware/software only; no arbitrary-seed or cross-platform guarantees',
        statistics='per canonical seed: seven equal-weight task means, five overlapping outer folds per task, ddof=1; across seed: three macro values, not fifteen independent folds; S42_B excluded')
    t.write(t.OUT/'final_review/results.json',result)
    lines=['# UA3D STABILITY r1','',f'状态：{status}；{detail}；independent_review=PENDING。',
        '唯一配方DET_BASE，固定主运行S42_A；S42_B为独立同seed复跑，不TEST、不算第四个seed。原BASE保持F_PD_N1_F0。','',
        '| 验收项目 | 结果 |','|---|---|']
    for k in ('numerical_diagnostic_pass','same_seed_repro_pass','cross_seed_stability_pass','stability_pass','quality_guard_pass','stable_ready','performance_pass','ft_complete','formal_test_complete','final_cpu_pass','execution_complete'):
        lines.append(f'| {k} | {result[k] if result[k] is not None else "未完成/不可比较"} |')
    lines+=['',f'原BASE macro完整精度：{t.REFERENCE!r}；主运行macro：{result["primary_test_macro"]!r}；主运行Δmacro：{delta!r}。',
        '性能提升只由S42_A与原BASE比较；跨seed均值不是主比较，也不挑选最佳seed。','',
        '| 运行 | validation macro | TEST macro | TEST−原BASE |','|---|---:|---:|---:|']
    for run in t.ORDER:
        v=f"{validation[run]['macro']:.3f}" if run in validation else '未完成'
        f=f"{formal[run]['macro']:.3f}" if run in formal else '不执行' if run=='S42_B' else '未完成/不可比较'
        d=f"{formal[run]['macro']-t.REFERENCE:+.3f}" if run in formal else '—'
        lines.append(f'| {run} | {v} | {f} | {d} |')
    if cross:
        lines+=['',f'三个canonical macro跨seed统计：{cross["mean"]:.3f}±{cross["std"]:.3f}；极差={cross["range"]:.3f}，原始极差`{cross["range"]!r}`；阈值≤0.005。','',
            '| 任务 | S42_A五折均值 | S142五折均值 | S242五折均值 | 跨seed极差 | ≤0.020 |','|---|---:|---:|---:|---:|---|']
        for task,v in cross['tasks'].items():lines.append('| '+task+' | '+' | '.join(f'{v["seed_fold_means"][run]:.3f}' for run in t.CANONICAL)+f' | {v["range"]:.3f} | {v["pass_range"]} |')
        lines.append('跨seed极差与超界任务的完整精度见results.json cross_seed；不把S42_B或15个fold当独立seed。')
    for run,entries in sota.items():
        lines+=['',f'## {run} 七任务正式TEST与论文数值对照','', '| 任务 | 项目mean±std | 原BASEmean±std | 项目−BASE | 论文mean±std | 项目−论文 | 论文模型 |','|---|---:|---:|---:|---:|---:|---|']
        for task,v in entries.items():lines.append(f'| {t.SOTA[task][0]} | {v["project"]["mean"]:.3f}±{v["project"]["std"]:.3f} | {v["base"]["mean"]:.3f}±{v["base"]["std"]:.3f} | {v["project"]["mean"]-v["base"]["mean"]:+.3f} | {v["paper_mean"]:.3f}±{v["paper_std"]:.3f} | {v["delta"]:+.3f} | {v["paper_model"]} |')
    lines+=['','## 前置诊断与首个分叉','']
    for name,item in diagnostics.items():
        u=item['consumption'];lines.append(f'- {name}：'+(f'{u["status"]}；eval attempts={u["diagnostic_eval_forwards"]}；train/update attempts={u["diagnostic_train_forwards"]}/{u["diagnostic_updates"]}。' if u else '未执行。'))
    if dc:
        for policy,check in dc['checks'].items():lines.append(f'- {policy}：exact_equal={check["equal"]}；首个分叉={check["first_divergence"]!r}；max_abs_diff原始值={check["max_abs_diff"]!r}。')
    failed_pair=next((v for v in pairs.values() if not v['equal']),None)
    if failed_pair:lines.append('完整同seed首个分叉：`'+json.dumps(failed_pair['first_divergence'],ensure_ascii=False)+'`。')
    lines+=['','每个模型的70epoch正常validation预测、训练行序/batch边界、loss/MSE/ALIGN/entropy、实际LR/p/冻结模式和梯度裁剪记录见FT history与validation/；BEST/last及EPS边界快照与35对比较均保留。没有额外train-eval/预热/benchmark。',
        'DEFAULT仅代表本次默认入口的共同继承环境，不冒称还原历史全部开关；DET精确比较失败或不支持算子即停止，不降级warn-only，不更换CSR/依赖/科学配方。',
        '稳定性和性能保留是预登记工程条件，不是置信区间或任意seed保证；共享开发TEST不是独立盲测。',
        'SOTA来源SOTA_RESULT.md：逐任务最佳包络，不是一个可运行模型；Egc3380/论文4124、Xc432/论文430等数据/split/协议未证明等价，仅作数值对照。',
        'CPU：`'+json.dumps(cpu,ensure_ascii=False)+'`。','预算预留：`'+json.dumps(budget['charged_or_reserved'])+'`。','实际已知消费：`'+json.dumps(budget['known_actual_lower_bound'])+'`。',
        '真实exit、环境、冻结、诊断计费和精确比较见budget.json、logs、environment.json、diagnostics/、reproducibility/、selection/final_frozen.json。','阻断：'+(error or '无')+'。',
        '未执行范围保留为未完成/不可比较；若最终CPU揭示科学合同错误，受影响数值无效/不可比较，不抹去已执行TEST。',
        '无自动retry/resume/repair、补seed/单元、放宽容差或重复CPU；不替换BASE、不自动部署、归档已验收或启动下一周期。']
    (t.OUT/'report.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('controller','register','diagnostic','ft','test','wrap'));p.add_argument('--arm',choices=t.RUNS+t.DIAGNOSTICS);p.add_argument('--task',choices=t.TASKS);p.add_argument('--fold',type=int,choices=range(5));p.add_argument('--spec');args=p.parse_args()
    t.torch.set_num_threads(2)
    if args.action=='wrap':return wrap(args.spec)
    def stop(sig,frame):t.ignore_termination();raise RuntimeError('controlled termination '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.action=='controller':return Controller().run()
    if args.action=='register':
        t.require_tmux();t.exclusive(t.OUT/'registry/source_acceptance.json',t.register_sources());return 0
    if args.action=='diagnostic':
        if args.arm not in t.DIAGNOSTICS:p.error('registered diagnostic required')
        setup=t.m.configure('DET' if args.arm.startswith('DET_') else 'DEFAULT');t.diagnostic(args.arm,setup);return 0
    if args.arm is None or args.task is None or args.fold is None:p.error('fixed estimator required')
    setup=t.m.configure('DET')
    t.unit(args.arm,args.task,args.fold,args.action=='test',2,setup);return 0
if __name__=='__main__':raise SystemExit(main())
