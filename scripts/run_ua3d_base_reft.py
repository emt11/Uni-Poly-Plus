#!/usr/bin/env python3
"""One-shot independent GPU2 exact BASE retraining; no retry/resume/repair."""
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
from src.training import ua3d_base_reft as t
from scripts.run_ua2d_a020_fp_fusion import atomic_exit,poll_exit,owned_group,pid_aliases
SOURCES=('src/training/ua3d_base_reft.py','scripts/run_ua3d_base_reft.py','configs/ua3d_base_reft_r1.json','tests/test_ua3d_base_reft.py')
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
            started='2026-10-28T15:22:37+08:00',deadline='2026-10-29T15:22:37+08:00',controller_started=t.now(),
            controller_identity=t.process_identity(os.getpid()),cwd=str(ROOT),argv=sys.argv,env={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','WORLD_SIZE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
            physical_gpus=[2],logical_device='cuda:0',world_size=1,ft_max_concurrency=4,test_max_concurrency=4,phase_exclusive=True,
            cpu_reserve_seconds=600,delivery_reserve_seconds=300,automatic_retry=False,automatic_resume=False,independent_review='PENDING',based_on=t.SPEC['based_on'])
        t.write(t.OUT/'execution_manifest.json',self.manifest)
        before=(ROOT/'CURRENT_PLAN.md').read_text();(t.OUT/'authorization').mkdir()
        (t.OUT/'authorization/CURRENT_PLAN.before.md').write_text(before)
        prior=before.split('</summary>\n\n',1)[1].rsplit('\n</details>\n',1)[0]
        (t.OUT/'authorization/CURRENT_PLAN.prior_cycle.md').write_text(prior)
        (t.OUT/'authorization/user_prompt.txt').write_text('请你重新微调和测试现有的 BASE ！仍然只使用 GPU2 ，最多 4 路 FT 和 TEST ！\n沿用用户此前无需主动监控的偏好。\n')
        t.exclusive(t.OUT/'authorization/authorization.json',dict(source='user explicitly requested existing BASE retraining and testing on GPU2 with FT/TEST cap4 on 2026-10-28',plan=t.PLAN,revision=t.REVISION,t0_unix=self.started,t0_policy='conservative first post-authorization read/execute tool time; includes implementation',limits=t.LIMITS,physical_gpus=[2],todo_read=False,remote_sync=False,new_hash_contract=False))
        t.write(t.OUT/'authorization/protected_status.json',dict(status=subprocess.check_output(['git','status','--short'],text=True)))
        for name in SOURCES:
            p=t.OUT/'authorization/source_at_launch'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text((ROOT/name).read_text())
        self.capture_diff();self.refresh()
        def cutoff(sig,frame):raise Deadline('hard clock delivery reserve reached')
        signal.signal(signal.SIGALRM,cutoff);signal.setitimer(signal.ITIMER_REAL,max(.001,self.deadline-300-time.time()))

    def capture_diff(self):
        lines=[]
        for name in SOURCES:lines.extend(difflib.unified_diff([], (ROOT/name).read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+name))
        before=t.OUT/'authorization/CURRENT_PLAN.prior_cycle.md'
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
            if arm=='BASE_ORIGINAL':raise ValueError('alias dispatch forbidden')
            t.config(arm,task,fold)
            if kind=='test':t.frozen_unit(t.read(frozen),arm,task,fold)
            inventory=self.available(kind)
            if inventory is None:return None
        else:
            if any(j['exit'] is None for j in self.jobs):raise RuntimeError('CPU/registration overlap')
            inventory=None
        if kind=='cpu' and len([j for j in self.jobs if j['kind']=='test' and j['exit']==0])!=35:raise RuntimeError('CPU before all TEST')
        name='base_reft_r1_'+kind+(f'_{arm}_{task}_{fold}' if arm else '')
        if any(j['name']==name for j in self.jobs):raise ValueError('duplicate attempt')
        if kind in CAPS:
            seen=[t.read(j['consumption'])['elapsed_seconds'] for j in self.jobs if j['kind']==kind and j['task']==task and j['exit']==0]
            if seen and self.deadline-time.time()<=900+max(seen):raise Deadline('observed normal unit duration cannot fit CPU/delivery reserves')
        reserve=reservation(kind,task,fold);charged=self.refresh()['charged_or_reserved']
        if any(charged[k]+v>t.LIMITS[k] for k,v in reserve.items()):raise RuntimeError('budget unavailable')
        output=t.unit_path(arm,task,fold,kind=='test') if kind in CAPS else None
        if output is not None and output.exists():raise FileExistsError('writer/overwrite risk '+str(output))
        log=t.LOG/name;log.mkdir(exist_ok=False)
        if kind=='cpu':argv=['timeout','--signal=TERM','--kill-after=30s','600s',sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_ua3d_base_reft.py']
        else:
            argv=[sys.executable,'-u','scripts/run_ua3d_base_reft.py',kind]
            if arm:argv+=['--arm',arm,'--task',task,'--fold',str(fold)]
        env=dict(CUDA_VISIBLE_DEVICES='2' if kind in CAPS else '',CUDA_DEVICE_ORDER='PCI_BUS_ID',WORLD_SIZE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',UA3M_PHYSICAL_GPU_UUID=inventory['uuid'] if inventory else '')
        job=dict(name=name,kind=kind,arm=arm,task=task,fold=fold,argv=argv,env=env,cwd=str(ROOT),log=str(log),reservation=reserve,
            consumption=str(output/'consumption.json') if output else None,exit=None,reserved_at=t.now(),gpu_uuid=env['UA3M_PHYSICAL_GPU_UUID'],physical_gpus=[2] if kind in CAPS else [],world_size=1,dispatch_pending=True,attempt='r1-once')
        self.jobs.append(job);self.refresh();t.write(log/'command.json',job)
        cmd=['env','CUDA_VISIBLE_DEVICES=','WORLD_SIZE=1','PYTHONDONTWRITEBYTECODE=1',sys.executable,'-u','scripts/run_ua3d_base_reft.py','wrap','--spec',str(log/'command.json')]
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

    def report(self,status,error):
        budget=self.refresh();validation={};formal={};costs={};paired={};sota={}
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
                        costs[arm].append(dict(task=task,fold=fold,consumption=t.read(p/'consumption.json'),best_epoch=met.get('best_epoch'),best_stage=met.get('best_stage'),history=str(p/'history.json'),
                            selected_training_components=hist[met['best_epoch']-1]['training_components'] if met.get('best_epoch') else None,alias=arm=='BASE_ORIGINAL'))
            if arm in formal:
                sota[arm]={task:dict(project=formal[arm]['tasks'][task],base=formal['BASE_ORIGINAL']['tasks'][task],paper_mean=t.SOTA[task][2],paper_std={'eea':.013,'egb':.009,'egc':.008,'ei':.018,'eps':.033,'nc':.032,'xc':.102}[task],paper_model=t.SOTA[task][1],delta=formal[arm]['tasks'][task]['mean']-t.SOTA[task][2]) for task in t.TASKS}
            if arm!='BASE_ORIGINAL':
                paired[arm]={}
                for name,values in (('validation',validation),('formal',formal)):
                    if arm not in values or 'BASE_ORIGINAL' not in values:continue
                    deltas={task:(t.np.array(values[arm]['tasks'][task]['folds'])-t.np.array(values['BASE_ORIGINAL']['tasks'][task]['folds'])).tolist() for task in t.TASKS}
                    paired[arm][name]=dict(delta_macro=values[arm]['macro']-values['BASE_ORIGINAL']['macro'],task_fold_deltas=deltas,
                        task_mean_deltas={task:values[arm]['tasks'][task]['mean']-values['BASE_ORIGINAL']['tasks'][task]['mean'] for task in t.TASKS},positive_folds={task:sum(v>0 for v in ds) for task,ds in deltas.items()})
        cpu_jobs=[j for j in self.jobs if j['kind']=='cpu'];cpu=dict(attempts=len(cpu_jobs),exit=cpu_jobs[0]['exit'] if cpu_jobs else None,log=cpu_jobs[0]['log'] if cpu_jobs else None,status='NOT_RUN' if not cpu_jobs else 'COMPLETE' if cpu_jobs[0]['exit']==0 else 'FAILED')
        t.write(t.OUT/'cpu_exit.json',cpu)
        selected=dict(macro=formal[final]['macro'],delta_macro=formal[final]['macro']-t.REFERENCE,performance_pass=formal[final]['macro']-t.REFERENCE>=.010) if final in formal else None
        result=dict(plan=t.PLAN,revision=t.REVISION,execution_status=status,independent_review='PENDING',FINAL=final,FINAL_p=frozen.get('FINAL_p'),validation=validation,formal=formal,
            partial={k:'未完成/不可比较' for k in t.ORDER if k not in formal},reference=t.REFERENCE,reference_id=t.REF_ID,reference_source=str(t.REF),selected_comparison=selected,
            performance_pass=selected['performance_pass'] if selected else None,sota_all7_pass=all(v['delta']>=0 for v in sota.get(final,{}).values()) if final in sota else None,
            test_highest_record=max(formal,key=lambda k:formal[k]['macro']) if len(formal)==2 else None,paired=paired,sota=sota,costs=costs,budget=budget,cpu=cpu,error=error,
            evaluation='shared development validation/TEST, not independent blind TEST',statistics='seven tasks equally weighted, five folds, seed42+fold, sample std ddof=1; overlapping training sets, not independent seed repetitions')
        t.write(t.OUT/'final_review/results.json',result)
        lines=['# UA3D BASE REFT r1','',f'状态：{status}；independent_review=PENDING。',f'固定重训FINAL：{final}；p={frozen.get("FINAL_p")}；当前BASE原始macro={t.REFERENCE!r}。',
            '共享开发评价；新FT/TEST未完整时为未完成/不可比较。控制仅复用，新增消费0。','',
            '| 配方 | p | validation macro | TEST macro | TEST−BASE | 角色 |','|---|---:|---:|---:|---:|---|']
        for arm,p in t.SPEC['recipes'].items():
            v=f"{validation[arm]['macro']:.3f}" if arm in validation else '未完成'
            f=f"{formal[arm]['macro']:.3f}" if arm in formal else '未完成/不可比较'
            delta=f"{formal[arm]['macro']-t.REFERENCE:+.3f}" if arm in formal else '—'
            lines.append(f'| {arm} | {p:.1f} | {v} | {f} | {delta} | '+('固定重训FINAL' if arm==final else '只读控制' if arm=='BASE_ORIGINAL' else '固定BASE重训')+' |')
        lines+=['','BASE重训完整精度：`'+json.dumps(selected,ensure_ascii=False)+'`。',f'TEST数值最高记录：{result["test_highest_record"]}；本轮FINAL固定为BASE_REFT，不事后改选。']
        for name,values in (('validation',validation),('TEST',formal)):
            for arm in t.ORDER:
                if arm not in values:continue
                lines+=['',f'## {arm} {name} 五折配对','', '| 任务 | R² mean±std | 相对控制Δmean | 正向折数 | 逐折差值 |','|---|---:|---:|---:|---|']
                for task in t.TASKS:
                    a=values[arm]['tasks'][task];b=values['BASE_ORIGINAL']['tasks'][task];ds=t.np.array(a['folds'])-t.np.array(b['folds'])
                    lines.append(f'| {task} | {a["mean"]:.3f}±{a["std"]:.3f} | {a["mean"]-b["mean"]:+.3f} | {int((ds>0).sum())}/5 | '+', '.join(f'{v:+.3f}' for v in ds)+' |')
        for arm,entries in sota.items():
            lines+=['',f'## {arm} 七任务SOTA数值对照','', '| 任务 | 项目mean±std | BASE | 论文mean±std | 项目−论文 | 论文模型 |','|---|---:|---:|---:|---:|---|']
            for task,v in entries.items():lines.append(f'| {t.SOTA[task][0]} | {v["project"]["mean"]:.3f}±{v["project"]["std"]:.3f} | {v["base"]["mean"]:.3f} | {v["paper_mean"]:.3f}±{v["paper_std"]:.3f} | {v["delta"]:+.3f} | {v["paper_model"]} |')
        lines+=['','原始指标、逐折差值、正向折数与消费见final_review/results.json；每个p的BEST epoch/stage、正常在线MSE/ALIGN/entropy与validation轨迹见costs和各单元history.json，未额外train-eval。',
            'EPS/Nc/Xc的变化见BASE复跑逐任务配对表；不以在线dropout训练MSE与eval误差直接相减推断欠拟合，不把loss/gate解释为因果证据。',
            '固定F endpoint和全部BASE超参数，head p始终为0.5；五折训练集重叠，seed42+fold不等于五个独立seed，不以35个task/fold宣传显著性，本轮只检验同配方复跑，不归因于新的科学因素。',
            'SOTA来源SOTA_RESULT.md；论文macro是逐任务包络而非单一模型。Egc3380/论文4124、Xc432/论文430，数据/split/统计等价性未证明，仅作数值对照。',
            'CPU：`'+json.dumps(cpu,ensure_ascii=False)+'`。', '消费：`'+json.dumps(budget['known_actual_lower_bound'])+'`。','预留：`'+json.dumps(budget['charged_or_reserved'])+'`。',
            '完整消费、冻结来源、所有真实exit见budget.json、selection/final_frozen.json、execution_manifest.json及logs。','阻断：'+(error or '无')+'。',
            '未执行：PT/prepare/FP或几何生成、额外seed/候选/重试、提前CPU/额外forward/benchmark。BASE不替换，不部署、不归档为已验收或开始下一周期。']
        (t.OUT/'report.md').write_text('\n'.join(lines)+'\n')
        return result

    def run(self):
        rc=1;status='失败停止，待审查';error=None
        try:
            t.update_plan('执行中','登记','新根已独占创建；只读接受F来源，无prepare/PT/初态生成。')
            self.execute([('register',None,None,None)])
            self.execute([('ft',arm,task,fold) for arm in t.ORDER[1:] for task in t.TASKS for fold in range(5)])
            self.guard_clock('test');frozen=t.freeze()
            t.update_plan('执行中','唯一冻结',f'FINAL={frozen["FINAL"]}；70个估计器/alias冻结；开始35次固定TEST。')
            self.execute([('test',arm,task,fold) for arm in t.ORDER[1:] for task in t.TASKS for fold in range(5)])
            # Check all completed formal aggregates/identity before the final suite, without inference.
            for arm in t.ORDER:t.aggregate(arm,True)
            self.execute([('cpu',None,None,None)])
            rc=0;status='完成，待审查'
        except BaseException:
            error=traceback.format_exc()
            if isinstance(sys.exc_info()[1],Deadline):status='截止停止，部分完成，待审查'
            try:self.stop_owned()
            except BaseException:error+='\nowned cleanup failure: '+traceback.format_exc()
            t.write(t.OUT/'failure.json',dict(error=error,time=t.now(),retry=False,resume=False,repair=False))
        finally:signal.setitimer(signal.ITIMER_REAL,0)
        try:result=self.report(status,error)
        except BaseException:
            error=(error or '')+'\nreport failure: '+traceback.format_exc();rc=1;status='失败停止，待审查'
            t.write(t.OUT/'final_review/report_failure.json',dict(error=error,time=t.now()))
        self.manifest.update(status=status,ended=t.now(),elapsed_seconds=time.time()-self.started,error=error)
        t.write(t.OUT/'execution_manifest.json',self.manifest)
        t.update_plan(status,'交付' if rc==0 else '异常停止',f'controller exit={rc}；证据 {t.OUT.relative_to(ROOT)}/final_review/results.json；'+('完成固定周期，待独立审查。' if rc==0 else '未完成/不可比较；不自动重试、恢复或修复。'))
        (t.OUT/'closeout.md').write_text(f'状态：{status}；independent_review=PENDING。\ncontroller真实exit={rc}。\n预算及逐作业exit见budget.json；CPU见cpu_exit.json；实际diff见actual_changes.diff。\n'+('下一步：独立审查。\n' if rc==0 else '未完成/不可比较。下一步：只读审查事故；修复/恢复需新的明确授权。\n')+(error or '阻断：无。')+'\n')
        self.capture_diff();t.exclusive(t.OUT/'controller_exit.json',dict(exit=rc,time=t.now(),process=t.process_identity(os.getpid())))
        atomic_exit(t.LOG/'controller.exit.txt',rc)
        return rc

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('controller','register','ft','test','wrap'));p.add_argument('--arm',choices=t.ORDER[1:]);p.add_argument('--task',choices=t.TASKS);p.add_argument('--fold',type=int,choices=range(5));p.add_argument('--spec');args=p.parse_args()
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
