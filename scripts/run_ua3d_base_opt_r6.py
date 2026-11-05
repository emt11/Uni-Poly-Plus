#!/usr/bin/env python3
"""One-shot r6 A-G controller; one GPU2 job, no retry/resume, fixed 96h clock."""
import time
ENTRY_T0=time.time()
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(key,'2')
import argparse
import copy
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
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training import ua3d_base_opt_r6 as t
from scripts.run_ua2d_a020_fp_fusion import atomic_exit,poll_exit,owned_group
SOURCES=('src/modules/ua3d_base_opt_r6.py','src/training/ua3d_base_opt_r6.py',
    'scripts/pretrain_ua3d_base_opt_r6.py','scripts/run_ua3d_base_opt_r6.py',
    'configs/ua3d_base_opt_r6.json','tests/test_ua3d_base_opt_r6.py')
HARD_SECONDS=96*3600
CLOSEOUT_SECONDS=300
CPU_SECONDS=600

class Deadline(RuntimeError):pass

def gpu_inventory():
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True,timeout=10)
    values=[line.split(',')[1].strip() for line in lines.splitlines() if line.split(',')[0].strip()=='2']
    if len(values)!=1:raise RuntimeError('physical GPU2 inventory mismatch')
    uuid=values[0]
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=10)
    pids=[int(line.split(',')[1]) for line in apps.splitlines() if line.split(',')[0].strip()==uuid]
    return dict(physical_gpu=2,uuid=uuid,pids=pids,time=t.now())

def reservation(kind,task=None,fold=None):
    if kind=='initialize':return dict(initialize_starts=1)
    if kind=='pt':return dict(pt_starts=1,pt_updates=10000,pt_microbatches=60000,pt_presentations=10080000)
    if kind=='cpu':return dict(cpu_suites=1)
    entry=t.read(t.SPLIT/f'{task}.json')['folds'][fold]
    if kind=='ft':return dict(ft_starts=1,ft_epochs=70,ft_updates=70*math.ceil(len(entry['train_indices'])/24))
    if kind=='test':return dict(test_starts=1,test_graphs=len(entry['test_indices']))
    raise ValueError('unregistered kind')

def ledger(jobs):
    charged=dict.fromkeys(t.LIMITS,0);actual=dict.fromkeys(t.LIMITS,0);complete=dict.fromkeys(t.LIMITS,0)
    for job in jobs:
        for key,value in job['reservation'].items():charged[key]+=value
        process=Path(job['log'])/'process.json'
        launched=process.exists()
        start={'initialize':'initialize_starts','pt':'pt_starts','ft':'ft_starts','test':'test_starts','cpu':'cpu_suites'}[job['kind']]
        if launched:actual[start]+=1
        if job['exit']==0:complete[start]+=1
        path=Path(job['consumption']) if job.get('consumption') else None
        if path is not None and path.exists():
            record=t.read(path)
            for key in actual:
                if key.endswith('starts') or key=='cpu_suites':continue
                if job['kind']=='ft' and key=='ft_epochs':a=record.get('epochs_started',0);b=record.get('completed_epochs',0)
                elif job['kind']=='ft' and key=='ft_updates':a=record.get('attempted_updates',0);b=record.get('completed_updates',0)
                elif job['kind']=='pt' and key=='pt_updates':a=record.get('pt_updates',0);b=record.get('completed_updates',0)
                else:a=b=record.get(key,0)
                actual[key]+=int(a);complete[key]+=int(b)
    bad=[key for key in charged if charged[key]>t.LIMITS[key] or actual[key]>charged[key] or complete[key]>actual[key]]
    return dict(limits=t.LIMITS,charged_or_reserved=charged,known_actual_lower_bound=actual,known_completed=complete,
        budget_violations=bad,jobs=jobs,zero_cost=t.ZERO_COST,
        accounting='full reservation before dispatch; failed/partial attempts not refunded; starts confirmed by wrapper process record')

def wrap(spec):
    job=t.read(spec);folder=Path(job['log']);env=os.environ.copy();env.update(job['env']);identity=None;forwarded=False
    def stop(sig,frame):
        nonlocal forwarded
        if not forwarded and identity is not None and t.process_alive(identity):
            forwarded=True;os.killpg(identity['pid'],signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (folder/'stdout.log').open('x') as out,(folder/'stderr.log').open('x') as err:
        child=subprocess.Popen(job['argv'],cwd=job['cwd'],env=env,stdout=out,stderr=err,start_new_session=True)
        identity=t.process_identity(child.pid)
        t.write(folder/'process.json',dict(wrapper=t.process_identity(os.getpid()),timeout_identity=identity,
            command=job['argv'],cwd=job['cwd'],started=t.now(),gpu_uuid=job['gpu_uuid']))
        rc=child.wait()
    atomic_exit(folder/'exit.txt',rc);t.write(folder/'end.json',dict(exit=rc,ended=t.now(),job=job['name']))
    return rc

class Controller:
    def __init__(self):
        t.require_tmux()
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('controller must hide GPU')
        if t.OUT.exists() or t.LOG.exists():raise FileExistsError('r6 root exists; stop without adopting/reusing')
        self.started=ENTRY_T0;self.deadline=self.started+HARD_SECONDS;self.jobs=[];self.active=None
        t.OUT.mkdir(parents=True,exist_ok=False);t.LOG.mkdir(parents=True,exist_ok=False)
        sys.stdout=(t.LOG/'controller.stdout.log').open('x',buffering=1)
        sys.stderr=(t.LOG/'controller.stderr.log').open('x',buffering=1)
        self.lock=(t.OUT/'controller.lock').open('x');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def cutoff(sig,frame):raise Deadline('96h hard deadline closeout reserve reached; no extension')
        signal.signal(signal.SIGALRM,cutoff)
        signal.setitimer(signal.ITIMER_REAL,max(.001,self.deadline-CLOSEOUT_SECONDS-time.time()))
        started=datetime.datetime.fromtimestamp(self.started,datetime.timezone(datetime.timedelta(hours=8))).isoformat()
        self.manifest=dict(plan=t.PLAN,revision=t.REVISION,status='执行中',started=started,t0_unix=self.started,
            deadline_unix=self.deadline,deadline=datetime.datetime.fromtimestamp(self.deadline,datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            hard_seconds=HARD_SECONDS,closeout_reserve_seconds=CLOSEOUT_SECONDS,cpu_reserve_seconds=CPU_SECONDS,
            cwd=str(ROOT),command=sys.argv,env={key:os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES','WORLD_SIZE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
            controller_identity=t.process_identity(os.getpid()),tmux=os.environ.get('TMUX_PANE'),
            tmux_window=subprocess.check_output(['tmux','display-message','-p','#{window_id} #{window_name}'],text=True).strip(),
            outer_argv=['env','CUDA_VISIBLE_DEVICES=','WORLD_SIZE=1','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2',
                'PYTHONDONTWRITEBYTECODE=1','timeout','--signal=TERM','--kill-after=30s','345600s',sys.executable,'-u','scripts/run_ua3d_base_opt_r6.py','controller'],
            based_on=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),physical_gpu=2,
            implementation_started_unix=t.read(ROOT/'configs/ua3d_base_opt_r6.json')['implementation_started_unix'],
            limits=t.LIMITS,automatic_retry=False,automatic_resume=False,independent_review='PENDING')
        self.manifest['implementation_seconds_before_controller']=self.started-self.manifest['implementation_started_unix']
        t.write(t.OUT/'execution_manifest.json',self.manifest)
        t.exclusive(t.OUT/'authorization/user_authorization.json',dict(plan=t.PLAN,revision=t.REVISION,stages='A-G complete fixed cycle',
            source='user explicitly authorized CURRENT_PLAN section12 in this conversation on 2026-10-26',physical_gpu=2,
            hard_hours=96,limits=t.LIMITS,automatic_retry=False,automatic_resume=False))
        (t.OUT/'authorization/CURRENT_PLAN.before.md').write_text((ROOT/'CURRENT_PLAN.md').read_text())
        t.write(t.OUT/'authorization/protected_status.json',dict(status=subprocess.check_output(['git','status','--short'],text=True),todo_read=False,remote_sync=False))
        t.exclusive(t.OUT/'config.json',t.read(ROOT/'configs/ua3d_base_opt_r6.json'))
        self.capture_diff();self.refresh()

    def capture_diff(self):
        diff=[]
        for name in SOURCES:
            diff.extend(difflib.unified_diff([], (ROOT/name).read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+name))
        diff.extend(difflib.unified_diff((t.OUT/'authorization/CURRENT_PLAN.before.md').read_text().splitlines(True),
            (ROOT/'CURRENT_PLAN.md').read_text().splitlines(True),fromfile='before/CURRENT_PLAN.md',tofile='after/CURRENT_PLAN.md'))
        path=t.OUT/'final_review/actual_changes.diff';path.parent.mkdir(exist_ok=True);path.write_text(''.join(diff))

    def refresh(self):
        for job in self.jobs:
            if job['exit'] is None:
                rc=poll_exit(job)
                if rc is not None:job.update(exit=rc,ended=t.now())
        value=ledger(self.jobs);value.update(time=t.now(),elapsed_seconds=time.time()-self.started,deadline_unix=self.deadline)
        t.write(t.OUT/'budget.json',value)
        if value['budget_violations']:raise RuntimeError('budget violation '+repr(value['budget_violations']))
        return value

    def remaining(self):return self.deadline-time.time()
    def guard_clock(self,kind=None):
        reserve=CLOSEOUT_SECONDS+(CPU_SECONDS if kind in ('initialize','pt','ft','test','cpu') else 0)
        if self.remaining()<=reserve:raise Deadline('insufficient 96h budget after CPU/closeout reserve; stop without dispatch')

    def launch(self,kind,arm=None,task=None,fold=None):
        self.guard_clock(kind);self.refresh()
        if any(j['exit'] is None for j in self.jobs):raise RuntimeError('overlapping jobs forbidden')
        if any(j['exit']!=0 for j in self.jobs):raise RuntimeError('failed chain cannot continue')
        frozen_path=t.OUT/'selection/final_frozen.json'
        if kind in ('initialize','pt','ft') and frozen_path.exists():raise RuntimeError('training after freeze forbidden')
        if kind in ('test','cpu') and not frozen_path.exists():raise RuntimeError('early TEST/CPU forbidden')
        if kind=='cpu':
            frozen=t.read(frozen_path)
            done=[j for j in self.jobs if j['kind']=='test' and j['exit']==0]
            if len(done)!=len(frozen['units']) or any(j['kind']=='cpu' for j in self.jobs):raise RuntimeError('CPU ordering/once contract')
        if kind=='test':t.frozen_unit(t.read(frozen_path),arm,task,fold)
        if kind=='pt' and arm not in t.registry()['pt']:raise ValueError('unregistered PT')
        if kind=='ft':t.config(arm,task,fold)
        gpu=kind in ('pt','ft','test');uuid=''
        if gpu:
            while True:
                self.guard_clock(kind)
                inventory=gpu_inventory();t.write(t.OUT/'occupancy_wait.json',inventory)
                if not inventory['pids']:uuid=inventory['uuid'];break
                t.update_plan('执行中','等待 GPU2','GPU2忙，等待计入固定96小时；不抢占、不换卡。')
                time.sleep(10)
        name='base_opt_r6_'+kind+(f'_{arm}' if arm else '')+(f'_{task}_{fold}' if task else '')
        if any(j['name']==name for j in self.jobs):raise ValueError('duplicate launch')
        reserve=reservation(kind,task,fold);charged=ledger(self.jobs)['charged_or_reserved']
        if any(charged[key]+value>t.LIMITS[key] for key,value in reserve.items()):raise RuntimeError('budget unavailable')
        if kind=='pt':output=t.OUT/'pretrain'/arm;consumption=output/'consumption.json'
        elif kind in ('ft','test'):output=t.unit_path(arm,task,fold,kind=='test');consumption=output/'consumption.json'
        else:output=None;consumption=None
        if output is not None and output.exists():raise FileExistsError('writer/overwrite conflict '+str(output))
        log=t.LOG/name;log.mkdir(exist_ok=False)
        if kind=='cpu':command=['timeout','--signal=TERM','--kill-after=30s','600s',sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_ua3d_base_opt_r6.py']
        elif kind=='pt':command=[sys.executable,'-u','scripts/pretrain_ua3d_base_opt_r6.py','--arm',arm,'--physical-gpu','2']
        else:
            command=[sys.executable,'-u','scripts/run_ua3d_base_opt_r6.py',kind]
            if arm:command+=['--arm',arm,'--task',task,'--fold',str(fold),'--physical-gpu','2']
        command=['env','CUDA_VISIBLE_DEVICES=2' if gpu else 'CUDA_VISIBLE_DEVICES=',*command]
        env=dict(CUDA_VISIBLE_DEVICES='2' if gpu else '',CUDA_DEVICE_ORDER='PCI_BUS_ID',WORLD_SIZE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2',UA3M_PHYSICAL_GPU_UUID=uuid,PYTHONDONTWRITEBYTECODE='1')
        job=dict(name=name,kind=kind,arm=arm,task=task,fold=fold,physical_gpus=[2] if gpu else [],gpu_concurrency=1,world_size=1,physical_gpu=2 if gpu else None,
            gpu_uuid=uuid,argv=command,env=env,cwd=str(ROOT),log=str(log),reservation=reserve,
            consumption=str(consumption) if consumption else None,exit=None,reserved_at=t.now(),attempt='r6-once',dispatch_pending=True)
        self.jobs.append(job);self.refresh();t.write(log/'command.json',job)
        cmd=['env','CUDA_VISIBLE_DEVICES=',sys.executable,'-u','scripts/run_ua3d_base_opt_r6.py','wrap','--spec',str(log/'command.json')]
        job['window']=subprocess.check_output(['tmux','new-window','-d','-P','-F','#{window_id}',
            '-t','Uni-Poly:','-n',name,'-c',str(ROOT),shlex.join(cmd)],text=True).strip()
        subprocess.run(['tmux','set-option','-w','-t',job['window'],'remain-on-exit','on'],check=True)
        pane=int(subprocess.check_output(['tmux','display-message','-p','-t',job['window'],'#{pane_pid}'],text=True))
        job['wrapper_identity']=t.process_identity(pane);job['dispatch_pending']=False;self.active=job;self.refresh()
        t.update_plan('执行中',kind,f'当前 {name}；GPU2/cuda:0/world1 串行；硬截止 {self.manifest["deadline"]}。')
        return job

    def wait(self,job):
        while job['exit'] is None:
            self.guard_clock();self.refresh();time.sleep(5)
        self.active=None
        if job['exit']!=0:raise RuntimeError(f"{job['name']} true exit={job['exit']}; no retry/resume")

    def ft_recipe(self,slot,stage,pt_id='HIST_R4_BASE',pt=None,ft=None,parent_id=None,control=None):
        canonical,new=t.register_recipe(slot,stage,pt_id,pt,ft,parent_id,control)
        if new:
            for task in t.TASKS:
                for fold in range(5):self.wait(self.launch('ft',canonical,task,fold))
        else:t.aggregate(canonical)  # aliases reference completed estimators, no inference or retraining
        return canonical

    def descendant(self,slot,stage,parent_id,control,**change):
        recipe=t.get_recipe(parent_id);ft=copy.deepcopy(recipe['ft']);ft.update(change)
        return self.ft_recipe(slot,stage,recipe['pt_id'],recipe['pt'],ft,parent_id,control)

    def develop(self):
        order=[]
        for name,a,b in (('L_A8_B8',.08,.08),('L_A0_B8',0.,.08),('L_A8_B0',.08,0.),('L_A0_B0',0.,0.)):
            ft=dict(t.m.DEFAULT_FT,align=a,entropy=b)
            order.append(self.ft_recipe('A_'+name,'A',ft=ft,control=t.BASELINE))
        a=t.select_stage('A',order,t.BASELINE)
        order=[]
        for name,lr,sched in (('U_HI_RESTART',.001,'restart'),('U_LO_RESTART',.0003,'restart'),('U_HI_MONO',.001,'mono'),('U_LO_MONO',.0003,'mono')):
            order.append(self.descendant('B_'+name,'B',a,a,joint_lr=lr,scheduler=sched))
        b=t.select_stage('B',order,a);order=[]
        for name,p2,p3 in (('D2_0_D3_0',0.,0.),('D2_1_D3_0',.1,0.),('D2_0_D3_1',0.,.1),('D2_1_D3_1',.1,.1)):
            order.append(self.descendant('C_'+name,'C',b,b,p2=p2,p3=p3))
        c=t.select_stage('C',order,b);order=[]
        for name,pfp,phead in (('DF_2_H5',.2,.5),('DF_0_H5',0.,.5),('DF_2_H1',.2,.1),('DF_0_H1',0.,.1)):
            order.append(self.descendant('D_'+name,'D',c,c,pfp=pfp,phead=phead))
        d=t.select_stage('D',order,c);order=[];ft=t.get_recipe(d)['ft']
        for name in t.m.PT_ARMS:
            pt_id='E_'+name;pt=t.m.pt_config(name);t.register_pt(pt_id,pt)
            self.wait(self.launch('pt',pt_id))
            order.append(self.ft_recipe('E_'+name,'E',pt_id,pt,ft,d,'E_P_CURRENT_SHARED'))
        e=t.select_stage('E',order,'E_P_CURRENT_SHARED');order=[];selected=t.get_recipe(e)
        for name,pn,pfp in (('PD_N1_F2',.1,.2),('PD_N0_F2',0.,.2),('PD_N1_F0',.1,0.),('PD_N0_F0',0.,0.)):
            pt=copy.deepcopy(selected['pt']);pt.update(pnative=pn,pfp=pfp)
            if name=='PD_N1_F2':pt_id=selected['pt_id']
            else:
                pt_id='F_'+name;t.register_pt(pt_id,pt);self.wait(self.launch('pt',pt_id))
            order.append(self.ft_recipe('F_'+name,'F',pt_id,pt,ft,e,e))
        t.select_stage('F',order,e)
        g=t.select_stage('G_PARENT',list(t.registry()['recipes']),t.BASELINE);order=[]
        for name,fusion in (('G_GATE','gate'),('G_RESIDUAL','residual'),('G_MODALNORM','modalnorm')):
            order.append(self.descendant(name,'G',g,g,fusion=fusion))
        record=t.registry()
        record['selections']['G']=dict(validation_completed=True,diagnostic_order=order,
            validation={key:t.aggregate(key) for key in order},time=t.now(),
            selection='no separate G winner; FINAL uses global pool and R6_BASE_REFT guard exactly once')
        t.write(t.OUT/'registry/recipes.json',record);t.write(t.OUT/'selection/G.json',record['selections']['G'])

    def stop_owned(self):
        t.ignore_termination();signal.setitimer(signal.ITIMER_REAL,0)
        for job in self.jobs:
            if job['exit'] is not None:continue
            group=owned_group(job)
            if group is not None:
                os.killpg(group,signal.SIGTERM)
                for _ in range(30):
                    rc=poll_exit(job)
                    if rc is not None:job.update(exit=rc,ended=t.now());break
                    time.sleep(1)
                if job['exit'] is None and owned_group(job) is not None:os.killpg(group,signal.SIGKILL)
            for _ in range(10):
                rc=poll_exit(job)
                if rc is not None:job.update(exit=rc,ended=t.now());break
                time.sleep(1)

    def report(self,status,error=None):
        record=t.registry() if (t.OUT/'registry/recipes.json').exists() else dict(recipes={},slots=[],aliases={},selections={},pt={})
        formal={};validation={};partial={};costs={}
        for recipe_id in record['recipes']:
            for destination,is_formal in ((validation,False),(formal,True)):
                try:destination[recipe_id]=t.aggregate(recipe_id,is_formal)
                except (OSError,ValueError,KeyError):
                    if is_formal:partial[recipe_id]='未完成／不可比较'
            costs[recipe_id]=[]
            for task in t.TASKS:
                for fold in range(5):
                    folder=t.unit_path(recipe_id,task,fold)
                    if (folder/'consumption.json').exists():
                        costs[recipe_id].append(dict(task=task,fold=fold,consumption=t.read(folder/'consumption.json'),
                            parameters=t.read(folder/'parameters.json') if (folder/'parameters.json').exists() else None))
        frozen=t.read(t.OUT/'selection/final_frozen.json') if (t.OUT/'selection/final_frozen.json').exists() else {}
        chosen=frozen.get('FINAL');budget=self.refresh()
        cpu_jobs=[job for job in self.jobs if job['kind']=='cpu']
        cpu=dict(attempts=len(cpu_jobs),exit=cpu_jobs[0]['exit'] if cpu_jobs else None,log=cpu_jobs[0]['log'] if cpu_jobs else None,
            status='NOT_RUN' if not cpu_jobs else 'COMPLETE' if cpu_jobs[0]['exit']==0 else 'FAILED' if cpu_jobs[0]['exit'] is not None else 'RUNNING')
        t.write(t.OUT/'final_review/cpu_exit.json',cpu)
        reference=t.read(t.REFERENCE_FILE)['formal']['FULL']['BASE']
        comparisons={key:dict(delta_macro=value['macro']-t.REFERENCE,performance_pass=value['macro']-t.REFERENCE>=.010,
            task_deltas={task:value['tasks'][task]['mean']-reference['tasks'][task]['mean'] for task in t.TASKS}) for key,value in formal.items()}
        sota={}
        for key in record['recipes']:
            value=formal.get(key);sota[key]={}
            for task,(name,model,ref) in t.SOTA.items():
                item=value['tasks'][task] if value else None
                sota[key][task]=dict(task=name,project=item,reference=ref,reference_model=model,source='SOTA_RESULT.md',
                    delta=item['mean']-ref if item else None,pass_numeric=item['mean']>=ref if item else None)
        final=formal.get(chosen)
        selected=dict(macro=final['macro'],delta_macro=final['macro']-t.REFERENCE,
            performance_pass=final['macro']-t.REFERENCE>=.010,sota_all7_pass=all(x['pass_numeric'] for x in sota[chosen].values())) if final else None
        paired=controlled_comparisons(record,formal,check_identity=True)
        validation_pairs=controlled_comparisons(record,validation,check_identity=False)
        pt_costs={key:t.read(t.OUT/'pretrain'/key/'consumption.json') if (t.OUT/'pretrain'/key/'consumption.json').exists() else None for key in record['pt']}
        result=dict(plan=t.PLAN,revision=t.REVISION,status=status,independent_review='PENDING',FINAL=chosen,
            reference=t.REFERENCE,reference_id='UA3D-SPATIAL-20261025-01/r4/FULL/BASE',reference_source=str(t.REFERENCE_FILE),
            formal=formal,partial=partial,validation=validation,comparisons=comparisons,selected_comparison=selected,
            performance_pass=selected['performance_pass'] if selected else None,sota_all7_pass=selected['sota_all7_pass'] if selected else None,
            sota=sota,registry=record,controlled_comparisons=paired,validation_controlled_comparisons=validation_pairs,
            costs=costs,pt_costs=pt_costs,budget=budget,cpu=cpu,error=error,
            test_scope='shared development evaluation; not independent blind TEST',
            statistics='seven tasks equally weighted; five outer folds; seed42+fold; std(ddof=1)',
            protocol_limitations='Egc3380 vs paper4124; Xc432 vs paper430; data/split/statistics equivalence not established; paper is per-task envelope')
        t.write(t.OUT/'results.json',result);t.write(t.OUT/'final_review/results.json',result)
        lines=['# UA3D BASE OPT r6 A—G','',f'状态：{status}；independent_review=PENDING。',
            f'预选FINAL：{chosen or "未冻结"}；历史BASE完整精度：`{t.REFERENCE!r}`。',
            '评价使用共享开发validation/TEST，非独立盲测。所有诊断臂均预登记；TEST不改变FINAL。',
            f'完整正式TEST：{len(formal)}/{len(record["recipes"])}套；不完整单元均为未完成／不可比较。',
            '','| 配方 | 角色 | TEST macro R² | 相对 BASE Δmacro R² | +0.010 |','|---|---|---:|---:|---|']
        for key in record['recipes']:
            value=formal.get(key);comparison=comparisons.get(key)
            lines.append(f'| {key} | '+('预选FINAL' if key==chosen else '预登记诊断臂')+' | '+
                (f'{value["macro"]:.3f} | {comparison["delta_macro"]:+.3f} | {comparison["performance_pass"]}' if value else '未完成／不可比较 | — | 未判定')+' |')
        lines+=['','预选FINAL完整精度：`'+json.dumps(selected,ensure_ascii=False)+'`。',
            '数值门槛与七任务SOTA目标分别判断；诊断臂的数值通过不使其事后晋级。']
        for key in record['recipes']:
            lines+=['',f'## {key} 七任务论文数值对照','',
                '| 任务 | 项目 R² mean±std | BASE mean | 项目−BASE | 论文最佳 mean | 项目−论文 | 论文模型 |',
                '|---|---:|---:|---:|---:|---:|---|']
            for task,item in sota[key].items():
                value=item['project'];base=reference['tasks'][task]['mean']
                lines.append(f'| {item["task"]} | '+(f'{value["mean"]:.3f}±{value["std"]:.3f} | {base:.3f} | {value["mean"]-base:+.3f}' if value else f'未完成／不可比较 | {base:.3f} | —')+
                    f' | {item["reference"]:.3f} | '+(f'{item["delta"]:+.3f}' if value else '—')+f' | {item["reference_model"]} |')
        lines+=['','配对比较、每fold差值、2×2主效应和差中差的完整精度见results.json的controlled_comparisons。',
            'StageA/B/C/D/E/F的完整2×2只解释各自固定父条件；跨阶段为顺序开发，不能推断跨阶段交互。',
            'StageE双向臂与独立投影不构成完整factorial；StageF将2D/3D PT dropout绑定；StageG新增容量单列，不称完整factorial。',
            '无多seed支持，不宣称统计显著或全局最优；不能仅凭gate权重解释模态贡献。',
            '项目为paper5_outer、每任务五折、seed42+fold、任务等权macro、std(ddof=1)。论文逐任务最佳组成包络。',
            'Egc3380／论文4124、Xc432／论文430等数据、split和统计协议差异未消除，仅作数值对照。',
            'history记录正常训练/validation前向的dropout、训练模式、optimizer组、模态范数、gate熵/均值/饱和比例、时间和显存；无额外模型前向。',
            '完整精度原始预测和row/key/target对齐在formal；BEST/validation/scaler/config在finetune；PT逐项sum/count/loss及消费在pretrain。',
            '预算预留：`'+json.dumps(budget['charged_or_reserved'])+'`。',
            '实际已知消费下界：`'+json.dumps(budget['known_actual_lower_bound'])+'`。',
            '实际已知完成：`'+json.dumps(budget['known_completed'])+'`。',
            'CPU suite：`'+json.dumps(cpu,ensure_ascii=False)+'`。',
            f'硬截止：{self.manifest["deadline"]}；t0不刷新；全部真实exit见budget.json及logs作业exit.txt。',
            '异常：'+(error or '无')+'。',
            '未完成时不使用validation代替TEST；未执行单元不虚报exit。最终保持待独立审查，不自动替换BASE、部署或归档为已验收。']
        report='\n'.join(lines)+'\n'
        (t.OUT/'report.md').write_text(report);(t.OUT/'final_review/report.md').write_text(report)
        t.update_plan(status,'交付' if status=='完成，待审查' else '异常停止' if error else '正式TEST汇总',
            f'{len(validation)}套validation、{len(formal)}套正式TEST完成；FINAL={chosen}；CPU={cpu["exit"]}；'+
            (f'阻断：{error.splitlines()[-1]}' if error else '证据见r6 final_review/results.json与report.md。'))
        return result

    def run(self):
        error=None;rc=1;status='失败停止，待审查'
        try:
            inventory=gpu_inventory();t.write(t.OUT/'startup_gpu_inventory.json',inventory)
            self.manifest.update(gpu_uuid=inventory['uuid'],physical_gpus=[2],gpu_concurrency=1,world_size=1)
            t.write(t.OUT/'execution_manifest.json',self.manifest)
            t.update_plan('执行中','初始化登记',f'首次t0={self.manifest["started"]}；硬截止={self.manifest["deadline"]}；GPU2单路串行。')
            self.wait(self.launch('initialize'))
            self.develop();self.guard_clock('test');frozen=t.freeze()
            t.update_plan('执行中','唯一冻结完成',f'FINAL={frozen["FINAL"]}；{len(frozen["units"])}个估计器已冻结，开始固定正式TEST。')
            for recipe_id in t.registry()['recipes']:
                for task in t.TASKS:
                    for fold in range(5):self.wait(self.launch('test',recipe_id,task,fold))
            self.report('执行中')
            self.wait(self.launch('cpu'))
            status='完成，待审查';rc=0
        except BaseException:
            error=traceback.format_exc()
            if isinstance(sys.exc_info()[1],Deadline):status='截止停止，部分完成，待审查'
            try:self.stop_owned()
            except BaseException:error+='\nowned cleanup failure: '+traceback.format_exc()
            t.write(t.OUT/'failure.json',dict(error=error,time=t.now(),automatic_retry=False,automatic_resume=False))
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
        try:
            self.report(status,error)
        except BaseException:
            error=(error or '')+'\ncloseout report failure: '+traceback.format_exc();rc=1;status='失败停止，待审查'
            t.write(t.OUT/'final_review/report_failure.json',dict(error=error,time=t.now()))
            # Minimal closeout preserves the original exception and true exits without retrying report/experiment.
            t.update_plan(status,'交付写入阻断',error.splitlines()[-1])
        self.manifest.update(status=status,run_outcome='COMPLETE' if rc==0 else 'STOPPED',ended=t.now(),
            elapsed_seconds=time.time()-self.started,error=error)
        t.write(t.OUT/'execution_manifest.json',self.manifest)
        closeout=('状态：'+status+'；independent_review=PENDING。\n'+
            'controller真实exit：'+str(rc)+'。\n'+
            '完整周期：'+str(rc==0)+'；未完成时为未完成／不可比较。\n'+
            '未执行单元无exit，不推断性能；不补跑CPU、不重试、不修复或扩时。\n'+
            '预算和逐项exit：budget.json；证据：final_review/results.json、report.md及logs。\n'+
            '阻断：'+(error or '无')+'\n'+
            ('下一步：提交独立审查。\n' if rc==0 else '下一步：只读审查事故与部分产物；如需修复/恢复，先提出具体方案等待新的明确授权。\n'))
        (t.OUT/'final_review/closeout.md').write_text(closeout)
        self.capture_diff()
        t.write(t.OUT/'controller_exit.json',dict(exit=rc,time=t.now(),process=t.process_identity(os.getpid())))
        atomic_exit(t.LOG/'controller.exit.txt',rc)
        return rc


def controlled_comparisons(record,values,check_identity=False):
    pairs={}
    def pair(a,b):
        name=a+'_minus_'+b
        if name in pairs:return pairs[name]
        if a not in values or b not in values:return None
        if check_identity:
            for task in t.TASKS:
                for fold in range(5):
                    paths=[t.unit_path(key,task,fold,True) for key in (a,b)]
                    with t.np.load(paths[0]/'test_predictions.npz',allow_pickle=False) as x,t.np.load(paths[1]/'test_predictions.npz',allow_pickle=False) as y:
                        for key in ('indices','sample_keys','y_true','availability'):
                            if not t.np.array_equal(x[key],y[key]):raise ValueError('paired TEST row/key/target mismatch')
                    left,right=[t.read(path/'identity_alignment.json') for path in paths]
                    if left!=right:raise ValueError('paired raw target/transform contract mismatch')
        task_deltas={task:(t.finite(values[a]['tasks'][task]['folds'])-t.finite(values[b]['tasks'][task]['folds'])).tolist() for task in t.TASKS}
        item=dict(a=a,b=b,delta_macro=values[a]['macro']-values[b]['macro'],paired_task_fold_deltas=task_deltas,
            task_mean_deltas={task:values[a]['tasks'][task]['mean']-values[b]['tasks'][task]['mean'] for task in t.TASKS},
            prediction_identity_checked=check_identity)
        pairs[name]=item;return item
    for key,recipe in record['recipes'].items():
        control=record['aliases'].get(recipe['control'],recipe['control'])
        if control and key!=control:pair(key,control)
    effects={}
    for stage,factor1,factor2 in (('A','ALIGN removal','entropy removal'),('B','joint modal LR low','monotone scheduler'),
        ('C','FT 2D p=0.1','FT 3D p=0.1'),('D','FT FP p=0','head p=0.1'),
        ('E','paper-order direction','independent FP projection'),('F','PT native p=0','PT FP p=0')):
        slots=[slot for slot in record['slots'] if slot['stage']==stage]
        if len(slots)<4:continue
        keys=[slot['canonical'] for slot in slots[:4]]
        if not all(key in values for key in keys):continue
        # E is [current/shared,current/ind,paper/shared,paper/ind]. Other stages are [00,10,01,11].
        if stage=='E':keys=[keys[0],keys[2],keys[1],keys[3]]
        arrays=[{task:t.finite(values[key]['tasks'][task]['folds']) for task in t.TASKS} for key in keys]
        def effect(coeff):
            folds={task:sum(c*a[task] for c,a in zip(coeff,arrays)).tolist() for task in t.TASKS}
            return dict(delta_macro=float(t.np.mean([t.np.mean(v) for v in folds.values()])),task_fold_deltas=folds)
        effects[stage]=dict(cells_00_10_01_11=keys,factor1=factor1,factor2=factor2,
            main_factor1=effect([-.5,.5,-.5,.5]),main_factor2=effect([-.5,-.5,.5,.5]),
            difference_in_differences=effect([1.,-1.,-1.,1.]),interpretation='conditional on fixed stage parent; no across-stage interaction claim')
    # Prespecified extra contrasts, including current new PT vs the historical endpoint.
    if 'D' in record['selections']:pair('E_P_CURRENT_SHARED',record['selections']['D']['selected'])
    pair('E_P_SYM_SHARED','E_P_CURRENT_SHARED')
    return dict(pairs=pairs,factorials=effects)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('controller','initialize','ft','test','wrap'))
    p.add_argument('--arm');p.add_argument('--task',choices=t.TASKS);p.add_argument('--fold',type=int,choices=range(5))
    p.add_argument('--physical-gpu',type=int,choices=(2,),default=2);p.add_argument('--spec');args=p.parse_args()
    t.torch.set_num_threads(2)
    if args.action=='wrap':return wrap(args.spec)
    def stop(sig,frame):t.ignore_termination();raise RuntimeError('controlled termination '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.action=='controller':return Controller().run()
    if args.action=='initialize':t.initialize_cycle();return 0
    if args.arm is None or args.task is None or args.fold is None:p.error('fixed estimator identity required')
    t.unit(args.arm,args.task,args.fold,args.action=='test',args.physical_gpu);return 0

if __name__=='__main__':raise SystemExit(main())
