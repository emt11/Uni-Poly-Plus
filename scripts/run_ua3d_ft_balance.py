#!/usr/bin/env python3
"""One-shot DSH GPU2 FT balance fixed 2x2 campaign; no retry/resume/repair."""
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
from src.training import ua3d_ft_balance as t
from scripts.run_ua2d_a020_fp_fusion import atomic_exit,poll_exit,owned_group,pid_aliases
SOURCES=('src/modules/ua3d_ft_balance.py','src/training/ua3d_ft_balance.py','scripts/run_ua3d_ft_balance.py','configs/ua3d_ft_balance_r1.json','tests/test_ua3d_ft_balance.py')
CAPS={'ft':4,'test':4};MEMORY={'ft':8192,'test':4096}

class Deadline(RuntimeError):pass

def reservation(kind,task=None,fold=None):
    if kind=='register':return dict(registrations=1)
    if kind=='cpu':return dict(cpu_suites=1)
    e=t.read(t.SPLIT/f'{task}.json')['folds'][fold]
    if kind=='ft':return dict(ft_starts=1,ft_epochs=70,ft_updates=70*math.ceil(len(e['train_indices'])/24),train_graphs=70*len(e['train_indices']),validation_started=70,validation_graphs=70*len(e['validation_indices']))
    if kind=='test':return dict(test_starts=1,test_graphs=len(e['test_indices']))
    raise ValueError('unregistered phase')

def ledger(jobs):
    charged=dict.fromkeys(t.LIMITS,0);actual=dict(charged);completed=dict(charged)
    for j in jobs:
        for k,v in j['reservation'].items():charged[k]+=v
        start={'register':'registrations','ft':'ft_starts','test':'test_starts','cpu':'cpu_suites'}[j['kind']]
        if (Path(j['log'])/'process.json').exists():actual[start]+=1
        if j['exit']==0:completed[start]+=1
        p=Path(j['consumption']) if j.get('consumption') else None
        if p is not None and p.exists():
            c=t.read(p)
            for key in j['reservation']:
                if key==start:continue
                a,b={'ft_epochs':('epochs_started','completed_epochs'),'ft_updates':('attempted_updates','completed_updates')}.get(key,(key,key))
                actual[key]+=int(c.get(a,0));completed[key]+=int(c.get(b,0))
    reserved=(t.OUT/'selection/freeze_reservation.json').exists();done=(t.OUT/'selection/final_frozen.json').exists()
    charged['freezes']=actual['freezes']=int(reserved);completed['freezes']=int(done)
    bad=[k for k in charged if charged[k]>t.LIMITS[k] or actual[k]>charged[k] or completed[k]>actual[k]]
    return dict(limits=t.LIMITS,charged_or_reserved=charged,known_actual_lower_bound=actual,known_completed=completed,budget_violations=bad,jobs=jobs,
        accounting='atomic full pre-dispatch reservation; no refunds for failed/partial units; actual values are recorded lower bounds')

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
        if t.OUT.exists() or t.LOG.exists():raise FileExistsError('root already exists; stop, no alternative root')
        self.started=t.SPEC['t0_unix'];self.deadline=self.started+86400;self.jobs=[]
        if time.time()>=self.deadline-900:raise Deadline('implementation consumed dispatch time')
        t.OUT.mkdir(parents=True,exist_ok=False);t.LOG.mkdir(parents=True,exist_ok=False)
        sys.stdout=(t.LOG/'controller.stdout.log').open('x',buffering=1);sys.stderr=(t.LOG/'controller.stderr.log').open('x',buffering=1)
        self.lock=(t.OUT/'controller.lock').open('x');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.manifest=dict(plan=t.PLAN,revision=t.REVISION,status='执行中',t0_unix=self.started,deadline_unix=self.deadline,
            started='2026-10-28T16:37:39+08:00',deadline='2026-10-29T16:37:39+08:00',t0_utc=t.SPEC['t0_utc'],deadline_utc=t.SPEC['deadline_utc'],execution_role='DSH',controller_started=t.now(),
            controller_identity=t.process_identity(os.getpid()),cwd=str(ROOT),argv=sys.argv,env={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','WORLD_SIZE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
            physical_gpus=[2],logical_device='cuda:0',world_size=1,ft_max_concurrency=4,test_max_concurrency=4,phase_exclusive=True,
            cpu_reserve_seconds=600,delivery_reserve_seconds=300,automatic_retry=False,automatic_resume=False,independent_review='PENDING',based_on=t.SPEC['based_on'])
        t.write(t.OUT/'execution_manifest.json',self.manifest)
        before=(ROOT/'CURRENT_PLAN.md').read_text();(t.OUT/'authorization').mkdir()
        (t.OUT/'authorization/CURRENT_PLAN.before.md').write_text(before)
        prompt='请DSH执行CURRENT_PLAN.md中的UA3D-FT-BALANCE-20261028-01/r1完整固定周期。仅物理GPU2，FT含validation最多4路、TEST最多4路，阶段互斥；24小时硬上限，t0从本次明确授权后的首次相关检查开始，包含实现与等待。固定C0_BASE、A_ALIGN_DECAY、B_MODAL_LR、AB_DECAY_LR四臂，各七任务五折70epochs，共140FT和140次正式TEST；只读复用原F预训练、原head/fusion及全部数据合同，PT/prepare/新增seed/重试均为0。按计划完成开发选择和一次冻结后再TEST，最终CPU仅最后一次。保护旧产物和dirty工作树，不读TODO.md、不远程同步、不加哈希合同。异常停止、保留真实exit与预算，不自动修复。交付保持待独立审查，不替换BASE。'
        (t.OUT/'authorization/user_prompt.txt').write_text(prompt+'\n')
        t.exclusive(t.OUT/'authorization/authorization.json',dict(source='user explicitly authorized complete FT-BALANCE/r1 fixed cycle on 2026-10-28',execution_role='DSH',plan=t.PLAN,revision=t.REVISION,t0_unix=self.started,t0_policy='first post-authorization related check, includes implementation and waiting',limits=t.LIMITS,physical_gpus=[2],todo_read=False,remote_sync=False,new_hash_contract=False))
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
        value.update(time=t.now(),elapsed_seconds=time.time()-self.started,current_slots={k:active.count(k) for k in ('register','ft','test','cpu')},deadline_unix=self.deadline)
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
        self.guard_clock(kind);self.refresh();self.failures()
        frozen=t.OUT/'selection/final_frozen.json'
        if any(j['kind']=='cpu' for j in self.jobs):raise RuntimeError('dispatch after sole CPU suite')
        if kind=='ft' and frozen.exists():raise RuntimeError('FT after freeze')
        if kind in ('test','cpu') and not frozen.exists():raise RuntimeError('early TEST/CPU')
        if kind in CAPS:
            if arm in t.ALIASES:raise ValueError('alias dispatch forbidden')
            t.config(arm,task,fold)
            if kind=='test':t.frozen_unit(t.read(frozen),arm,task,fold)
            inventory=self.available(kind)
            if inventory is None:return None
        else:
            if any(j['exit'] is None for j in self.jobs):raise RuntimeError('CPU/registration overlap')
            inventory=None
        if kind=='cpu' and len([j for j in self.jobs if j['kind']=='test' and j['exit']==0])!=140:raise RuntimeError('CPU before all TEST')
        name='ft_balance_r1_'+kind+(f'_{arm}_{task}_{fold}' if arm else '')
        if any(j['name']==name for j in self.jobs):raise ValueError('duplicate attempt')
        if kind in CAPS:
            seen=[t.read(j['consumption'])['elapsed_seconds'] for j in self.jobs if j['kind']==kind and j['task']==task and j['exit']==0]
            if seen and self.deadline-time.time()<=900+max(seen):raise Deadline('observed normal unit duration cannot fit CPU/delivery reserves')
        reserve=reservation(kind,task,fold);charged=self.refresh()['charged_or_reserved']
        if any(charged[k]+v>t.LIMITS[k] for k,v in reserve.items()):raise RuntimeError('budget unavailable')
        output=t.unit_path(arm,task,fold,kind=='test') if kind in CAPS else None
        if output is not None and output.exists():raise FileExistsError('writer/overwrite risk '+str(output))
        log=t.LOG/name;log.mkdir(exist_ok=False)
        if kind=='cpu':argv=['timeout','--signal=TERM','--kill-after=30s','600s',sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_ua3d_ft_balance.py']
        else:
            argv=[sys.executable,'-u','scripts/run_ua3d_ft_balance.py',kind]
            if arm:argv+=['--arm',arm,'--task',task,'--fold',str(fold)]
        env=dict(CUDA_VISIBLE_DEVICES='2' if kind in CAPS else '',CUDA_DEVICE_ORDER='PCI_BUS_ID',WORLD_SIZE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',UA3M_PHYSICAL_GPU_UUID=inventory['uuid'] if inventory else '')
        job=dict(name=name,kind=kind,arm=arm,task=task,fold=fold,argv=argv,env=env,cwd=str(ROOT),log=str(log),reservation=reserve,
            consumption=str(output/'consumption.json') if output else None,exit=None,reserved_at=t.now(),gpu_uuid=env['UA3M_PHYSICAL_GPU_UUID'],physical_gpus=[2] if kind in CAPS else [],world_size=1,dispatch_pending=True,attempt='r1-once')
        self.jobs.append(job);self.refresh();t.write(log/'command.json',job)
        cmd=['env','CUDA_VISIBLE_DEVICES=','WORLD_SIZE=1','PYTHONDONTWRITEBYTECODE=1',sys.executable,'-u','scripts/run_ua3d_ft_balance.py','wrap','--spec',str(log/'command.json')]
        job['window']=subprocess.check_output(['tmux','new-window','-d','-P','-F','#{window_id}','-t','Uni-Poly:','-n',name,'-c',str(ROOT),shlex.join(cmd)],text=True).strip()
        subprocess.run(['tmux','set-option','-w','-t',job['window'],'remain-on-exit','on'],check=True)
        pane=int(subprocess.check_output(['tmux','display-message','-p','-t',job['window'],'#{pane_pid}'],text=True))
        job['wrapper_identity']=t.process_identity(pane);job['dispatch_pending']=False;self.refresh()
        t.update_plan('执行中',kind,f'已派发{len(self.jobs)}个固定作业；当前 {name}；真实exit见独立账本。')
        return job

    def execute(self,requests):
        pending=deque(requests)
        while pending or any(j['exit'] is None for j in self.jobs):
            self.guard_clock();self.refresh();self.failures()
            while pending:
                job=self.launch(*pending[0])
                if job is None:break
                pending.popleft();self.refresh();self.failures()
            durations={}
            for j in self.jobs:
                if j['exit']==0 and j['kind'] in CAPS:
                    usage=t.read(j['consumption']);durations.setdefault(j['kind']+'/'+j['task'],[]).append(usage['elapsed_seconds'])
            eta=sum(float(t.np.mean(durations[k+'/'+task])) for k,arm,task,fold in pending if k+'/'+str(task) in durations)/4
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
            t.update_plan('执行中','登记','新根已独占创建；只读接受F来源，无prepare/PT/初态生成。')
            self.execute([('register',None,None,None)])
            self.execute([('ft',arm,task,fold) for task in t.TASKS for fold in range(5) for arm in t.ARMS])
            self.guard_clock('test');t.exclusive(t.OUT/'selection/freeze_reservation.json',dict(time=t.now(),reserved=1,once=True));self.refresh();frozen=t.freeze();self.refresh()
            t.update_plan('执行中','唯一冻结',f'FINAL={frozen["FINAL"]}；140个新BEST与70个历史alias冻结；开始140次固定TEST。')
            self.execute([('test',arm,task,fold) for task in t.TASKS for fold in range(5) for arm in t.ARMS])
            # Check all completed formal aggregates/identity before the final suite, without inference.
            for arm in t.ORDER:t.aggregate(arm,True)
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
        t.update_plan(status,'交付' if rc==0 else '异常停止',f'controller exit={rc}；证据 {t.OUT.relative_to(ROOT)}/final_review/results.json；'+('完成固定周期，待独立审查。' if rc==0 else '未完成/不可比较；不自动重试、恢复或修复。'))
        (t.OUT/'closeout.md').write_text(f'状态：{status}；independent_review=PENDING。\ncontroller真实exit={rc}。\n完成层级：{result.get("execution_detail") if result else "交付阻断"}。\n预算及逐作业exit见budget.json；CPU见cpu_exit.json；实际diff见actual_changes.diff。\n'+('下一步：独立审查。\n' if rc==0 else '下一步：只读审查事故及已完成范围；科学合同错误涉及的数值无效/不可比较，修复/恢复需新的明确授权。\n')+(error or '阻断：无。')+'\n')
        self.capture_diff();t.exclusive(t.OUT/'controller_exit.json',dict(exit=rc,time=t.now(),process=t.process_identity(os.getpid())))
        atomic_exit(t.LOG/'controller.exit.txt',rc)
        return rc

def report_campaign(controller,status,error):
    budget=controller.refresh();validation={};formal={};costs={};sota={}
    frozen=t.read(t.OUT/'selection/final_frozen.json') if (t.OUT/'selection/final_frozen.json').exists() else {}
    final=frozen.get('FINAL');registered=(t.OUT/'registry/recipes.json').exists()
    for arm in t.ORDER:
        for dest,flag in ((validation,False),(formal,True)):
            if not registered:continue
            try:dest[arm]=t.aggregate(arm,flag)
            except (FileNotFoundError,ValueError,KeyError):pass
        costs[arm]=[]
        for task in t.TASKS:
            for fold in range(5):
                p=t.unit_path(arm,task,fold)
                if (p/'consumption.json').exists():
                    met=t.read(p/'metrics.json') if (p/'metrics.json').exists() else {}
                    hist=t.read(p/'history.json') if (p/'history.json').exists() else []
                    best=hist[met['best_epoch']-1] if met.get('best_epoch') else {}
                    costs[arm].append(dict(task=task,fold=fold,consumption=t.read(p/'consumption.json'),best_epoch=met.get('best_epoch'),best_stage=met.get('best_stage'),history=str(p/'history.json'),
                        selected_training_components=best.get('training_components'),selected_align=best.get('align_coefficient',.08),selected_learning_rates=best.get('learning_rates'),alias=arm in t.ALIASES))
        if arm in formal:
            sota[arm]={task:dict(project=formal[arm]['tasks'][task],base=formal['BASE_ORIGINAL']['tasks'][task],paper_mean=t.SOTA[task][2],paper_std={'eea':.013,'egb':.009,'egc':.008,'ei':.018,'eps':.033,'nc':.032,'xc':.102}[task],paper_model=t.SOTA[task][1],delta=formal[arm]['tasks'][task]['mean']-t.SOTA[task][2],source='SOTA_RESULT.md') for task in t.TASKS}
    pairs={};interaction={}
    contrasts=[('A_ALIGN_DECAY','C0_BASE'),('B_MODAL_LR','C0_BASE'),('AB_DECAY_LR','C0_BASE'),('AB_DECAY_LR','B_MODAL_LR'),('AB_DECAY_LR','A_ALIGN_DECAY'),('B_MODAL_LR','A_ALIGN_DECAY')]
    contrasts += [(a,b) for a in t.ARMS for b in t.ALIASES]
    for name,values in (('validation',validation),('formal',formal)):
        pairs[name]={}
        for a,b in contrasts:
            if a not in values or b not in values:continue
            ds={task:(t.np.array(values[a]['tasks'][task]['folds'])-t.np.array(values[b]['tasks'][task]['folds'])).tolist() for task in t.TASKS}
            pairs[name][a+'_minus_'+b]=dict(a=a,b=b,delta_macro=values[a]['macro']-values[b]['macro'],task_fold_deltas=ds,
                task_mean_deltas={task:values[a]['tasks'][task]['mean']-values[b]['tasks'][task]['mean'] for task in t.TASKS},positive_folds={task:sum(x>0 for x in d) for task,d in ds.items()})
        if all(a in values for a in t.ARMS):
            c,a,b,ab=[values[k] for k in t.ARMS]
            ds={task:(t.np.array(ab['tasks'][task]['folds'])-t.np.array(a['tasks'][task]['folds'])-t.np.array(b['tasks'][task]['folds'])+t.np.array(c['tasks'][task]['folds'])).tolist() for task in t.TASKS}
            interaction[name]=dict(macro=ab['macro']-a['macro']-b['macro']+c['macro'],task_fold_values=ds,task_means={task:float(t.np.mean(v)) for task,v in ds.items()},interpretation='prespecified descriptive 2x2 interaction only; no significance or stable causal claim')
    cpu_jobs=[j for j in controller.jobs if j['kind']=='cpu'];cpu=dict(attempts=len(cpu_jobs),exit=cpu_jobs[0]['exit'] if cpu_jobs else None,log=cpu_jobs[0]['log'] if cpu_jobs else None,status='NOT_RUN' if not cpu_jobs else 'COMPLETE' if cpu_jobs[0]['exit']==0 else 'FAILED')
    t.write(t.OUT/'cpu_exit.json',cpu)
    key='BASE_ORIGINAL' if final=='BASE_KEEP' else final;selected=None
    if key in formal:
        delta=formal[key]['macro']-t.REFERENCE
        selected=dict(macro=formal[key]['macro'],delta_macro=delta,performance_pass=final!='BASE_KEEP' and delta>=.010,
            delta_vs_C0=formal[key]['macro']-formal['C0_BASE']['macro'] if 'C0_BASE' in formal else None,
            delta_vs_history=formal[key]['macro']-formal['BASE_REFT_HISTORY']['macro'] if 'BASE_REFT_HISTORY' in formal else None,
            matched_control_positive=(formal[key]['macro']-formal['C0_BASE']['macro']>0) if final in t.ARMS and 'C0_BASE' in formal else None)
    ft_complete=all(a in validation for a in t.ARMS);test_complete=all(a in formal for a in t.ARMS)
    execution_complete=ft_complete and test_complete and cpu['exit']==0 and not error
    if ft_complete and test_complete and cpu['exit'] not in (None,0):execution_detail='训练/TEST完整，最终验证失败，整体未验收'
    else:execution_detail='完成，待审查' if execution_complete else '未完成/不可比较：详见已完成量和异常'
    result=dict(plan=t.PLAN,revision=t.REVISION,execution_role='DSH',execution_status=status,execution_detail=execution_detail,ft_complete=ft_complete,formal_test_complete=test_complete,final_cpu_pass=cpu['exit']==0,execution_complete=execution_complete,
        independent_review='PENDING',FINAL=final,FINAL_p=.5,validation=validation,formal=formal,partial={k:'未完成/不可比较' for k in t.ORDER if k not in formal},
        reference=t.REFERENCE,reference_id=t.REF_ID,reference_source=str(t.REF),selected_comparison=selected,performance_pass=selected['performance_pass'] if selected else None,
        matched_control_positive=selected['matched_control_positive'] if selected else None,
        sota_all7_pass=all(v['delta']>=0 for v in sota[key].values()) if key in sota else None,
        test_highest_diagnostic=max(t.ARMS,key=lambda k:formal[k]['macro']) if test_complete else None,pairs=pairs,descriptive_interaction=interaction,sota=sota,costs=costs,budget=budget,cpu=cpu,error=error,
        evaluation='shared development validation/TEST, not independent blind TEST',statistics='seven tasks equally weighted, five overlapping outer folds, seed42+fold, sample std ddof=1; no independent-seed significance claim')
    t.write(t.OUT/'final_review/results.json',result)
    lines=['# UA3D FT BALANCE r1','',f'状态：{status}；{execution_detail}；independent_review=PENDING。',f'预冻结唯一FINAL：{final}；原BASE macro完整精度={t.REFERENCE!r}。',
        '共享开发评价。C0为新增匹配控制，不替换原BASE；历史alias零新增消费；BASE_KEEP表示保留原模型，不是新训练臂。','',
        '| 配方 | validation macro | TEST macro | TEST−原BASE | TEST−C0 | TEST−历史REFT |','|---|---:|---:|---:|---:|---:|']
    for arm in t.ORDER:
        v=f"{validation[arm]['macro']:.3f}" if arm in validation else '未完成'
        f=f"{formal[arm]['macro']:.3f}" if arm in formal else '未完成/不可比较'
        ds=[f"{formal[arm]['macro']-formal[b]['macro']:+.3f}" if arm in formal and b in formal else '—' for b in ('BASE_ORIGINAL','C0_BASE','BASE_REFT_HISTORY')]
        lines.append(f'| {arm} | {v} | {f} | '+' | '.join(ds)+' |')
    lines+=['','FINAL完整精度验收：`'+json.dumps(selected,ensure_ascii=False)+'`。',f'新TEST最高诊断臂：{result["test_highest_diagnostic"]}；不得因TEST改选FINAL。',
        '开发规则与淘汰原因见selection/validation_selection.json；全部四臂诊断无论FINAL是否BASE_KEEP均固定完成。']
    if selected and selected['performance_pass'] and selected['matched_control_positive'] is False:lines.append('原BASE数值门槛通过、同周期优化收益未确认。')
    for name,values in (('validation',validation),('TEST',formal)):
        for arm in t.ORDER:
            if arm not in values:continue
            lines+=['',f'## {arm} {name} 七任务五折','', '| 任务 | mean±std | Δ原BASE | ΔC0 | Δ历史REFT |','|---|---:|---:|---:|---:|']
            for task in t.TASKS:
                a=values[arm]['tasks'][task];ds=[f"{a['mean']-values[b]['tasks'][task]['mean']:+.3f}" if b in values else '—' for b in ('BASE_ORIGINAL','C0_BASE','BASE_REFT_HISTORY')]
                lines.append(f'| {task} | {a["mean"]:.3f}±{a["std"]:.3f} | '+' | '.join(ds)+' |')
    for name,items in pairs.items():
        lines+=['',f'## {name} 预登记配对与描述性交互','']
        for pair,item in items.items():
            lines += [f'### {pair}；Δmacro={item["delta_macro"]:+.3f}','', '| 任务 | Δmean | 正向折数 | fold0—4差值 |','|---|---:|---:|---|']
            for task,ds in item['task_fold_deltas'].items():lines.append(f'| {task} | {item["task_mean_deltas"][task]:+.3f} | {item["positive_folds"][task]}/5 | '+', '.join(f'{x:+.3f}' for x in ds)+' |')
        if name in interaction:
            item=interaction[name];lines+=['',f'描述性交互 I=AB−A−B+C0：macro={item["macro"]:+.3f}；原始值`{item["macro"]!r}`。','', '| 任务 | I均值 | fold0—4 I |','|---|---:|---|']
            for task,ds in item['task_fold_values'].items():lines.append(f'| {task} | {item["task_means"][task]:+.3f} | '+', '.join(f'{x:+.3f}' for x in ds)+' |')
    for arm,entries in sota.items():
        lines+=['',f'## {arm} 七任务论文数值对照','', '| 任务 | 项目mean±std | 原BASE | 论文mean±std | 项目−论文 | 论文模型 |','|---|---:|---:|---:|---:|---|']
        for task,v in entries.items():lines.append(f'| {t.SOTA[task][0]} | {v["project"]["mean"]:.3f}±{v["project"]["std"]:.3f} | {v["base"]["mean"]:.3f} | {v["paper_mean"]:.3f}±{v["paper_std"]:.3f} | {v["delta"]:+.3f} | {v["paper_model"]} |')
    lines+=['','BEST epoch/stage、实际λ/LR/参数分组、正常MSE/ALIGN/entropy/validation轨迹见各FT history.json和results.json costs；未额外train-eval，不用在线loss差值推断梯度冲突或因果。',
        '四臂只检验F endpoint、固定LR/WD/预算下的两项策略；五折训练集重叠，seed42+fold不是独立seed；不宣称统计显著、全局最优或稳定因果机制。',
        'SOTA来源SOTA_RESULT.md；论文macro是逐任务包络。Egc3380/论文4124、Xc432/论文430等数据/split/统计协议未证明等价，仅作数值对照。',
        '最终CPU失败时保留已完成训练/TEST数值并区分整体未验收；若失败揭示数据/模型科学合同错误，受影响数值无效/不可比较，需独立审查定位。',
        'CPU：`'+json.dumps(cpu,ensure_ascii=False)+'`。','实际已知消费：`'+json.dumps(budget['known_actual_lower_bound'])+'`。','预留：`'+json.dumps(budget['charged_or_reserved'])+'`。',
        '冻结/配置来源、预算及逐作业真实exit见selection/final_frozen.json、budget.json与logs。','阻断：'+(error or '无')+'。',
        '未执行：PT/prepare/FP或几何生成/KRR/融合、额外seed/候选/重试/恢复/修复、提前CPU/额外forward/benchmark。',
        '不自动替换BASE、部署、更新PROJECT_HISTORY为已验收或启动下一周期。']
    (t.OUT/'report.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('controller','register','ft','test','wrap'));p.add_argument('--arm',choices=t.ARMS);p.add_argument('--task',choices=t.TASKS);p.add_argument('--fold',type=int,choices=range(5));p.add_argument('--spec');args=p.parse_args()
    t.torch.set_num_threads(2)
    if args.action=='wrap':return wrap(args.spec)
    def stop(sig,frame):t.ignore_termination();raise RuntimeError('controlled termination '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.action=='controller':return Controller().run()
    if args.action=='register':
        t.require_tmux();t.exclusive(t.OUT/'registry/source_acceptance.json',t.register_sources());return 0
    if args.arm is None or args.task is None or args.fold is None:p.error('fixed estimator required')
    t.unit(args.arm,args.task,args.fold,args.action=='test',2);return 0
if __name__=='__main__':raise SystemExit(main())
