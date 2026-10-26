#!/usr/bin/env python3
"""Single-writer fixed UA3M campaign with explicitly allocated world1 workers."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(key,'2')
import argparse
import difflib
import fcntl
import json
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training import ua3m_paperfuse as t
from scripts.run_ua2d_a020_fp_fusion import atomic_exit,poll_exit,ExitProtocolError,owned_group,gpu_snapshot,pid_aliases
SOURCES=('src/modules/ua3m_paperfuse.py','src/dataset/ua3m_paperfuse.py','src/training/ua3m_paperfuse.py',
    'scripts/pretrain_ua3m_paperfuse.py','scripts/run_ua3m_paperfuse.py','configs/ua3m_paperfuse_r1.json','tests/test_ua3m_paperfuse.py')



def gpu_snapshots(allowed):
    if tuple(allowed)==(3,):
        value=gpu_snapshot()
        return {3:dict(uuid='legacy_gpu3',pids=value)} if value is not None else None
    try:
        devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True,timeout=10)
        mapping={}
        for line in devices.splitlines():
            index,uuid=[x.strip() for x in line.split(',')];index=int(index)
            if index in allowed:
                if index in mapping or uuid in [v['uuid'] for v in mapping.values()]:raise ValueError('ambiguous physical device')
                mapping[index]=dict(uuid=uuid,pids=set())
        if set(mapping)!=set(allowed):raise ValueError('authorized GPU inventory incomplete')
        by_uuid={v['uuid']:v for v in mapping.values()}
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=10)
        for line in rows.splitlines():
            uuid,pid=[x.strip() for x in line.split(',')]
            if uuid not in by_uuid:raise ValueError('compute PID cannot map to authorized card')
            by_uuid[uuid]['pids'].add(int(pid))
        return mapping
    except (OSError,ValueError,subprocess.SubprocessError):return None


def reservation(kind,task=None,fold=None):
    if kind=='prepare':return dict(prepare_starts=1,cache_rows=5875)
    if kind=='pt':return dict(pt_starts=1,pt_updates=10000,pt_microbatches=60000,pt_presentations=10080000)
    if kind=='cpu':return dict(cpu_suites=1)
    entry=t.read(t.SPLIT/f'{task}.json')['folds'][fold]
    if kind=='ft':return dict(ft_starts=1,epochs_started=70,attempted_updates=70*int(__import__('math').ceil(len(entry['train_indices'])/24)),train_graphs=70*len(entry['train_indices']),validation_started=70,validation_graphs=70*len(entry['validation_indices']))
    if kind=='test':
        rows=t.read(t.prepared_root()/'inputs/rows.json')[task];available={r['row']:r['available'] for r in rows}
        return dict(test_starts=1,test_graphs=len(entry['test_indices']),test_3d_graphs=sum(available[r] for r in entry['test_indices']))
    raise ValueError('extra phase forbidden')


def ledger(jobs,history=None):
    charged=dict.fromkeys(t.LIMITS,0);actual=dict.fromkeys(t.LIMITS,0)
    for job in jobs:
        for key,value in job['reservation'].items():charged[key]+=value
        start={'prepare':'prepare_starts','pt':'pt_starts','ft':'ft_starts','test':'test_starts','cpu':'cpu_suites'}[job['kind']];actual[start]+=1
        path=Path(job['consumption']) if job.get('consumption') else None
        if path is not None and path.exists():
            record=t.read(path)
            if job['kind']=='prepare':actual['cache_rows']+=record.get('records',0)
            else:
                for key in actual:
                    if not key.endswith('starts') and key!='cpu_suites':actual[key]+=int(record.get(key,0))
    current=dict(charged_or_reserved=dict(charged),known_actual_lower_bound=dict(actual))
    if history is not None:
        for key in charged:
            charged[key]+=history['charged_or_reserved'][key];actual[key]+=history['known_actual_lower_bound'][key]
    bad=[key for key in charged if charged[key]>t.LIMITS[key]]
    return dict(limits=t.LIMITS,charged_or_reserved=charged,known_actual_lower_bound=actual,budget_violations=bad,
        zero_cost=t.ZERO_COST,jobs=jobs,parent_history=history,new_execution=current,cumulative=dict(charged_or_reserved=charged,known_actual_lower_bound=actual),accounting='irreversible full unit charged at launch; partial and failed attempts never refunded; durable actual separately')


class Controller:
    def __init__(self):
        t.require_tmux();self.lock=(t.OUT/'controller.lock').open('x');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (t.OUT/'budget.json').exists():raise FileExistsError('campaign started; no retry/resume')
        auth=t.read(t.OUT/'authorization/user_authorization.json')
        if auth.get('plan')!=t.PLAN or auth.get('revision')!=t.REVISION:raise ValueError('authorization mismatch')
        self.history=None
        if t.EXECUTION_REVISION=='gpu0123-r1':
            if auth.get('replacement_pt_increment')!=1 or auth.get('new_prepare_increment')!=1 or auth.get('execution_revision')!='gpu0123-r1':raise ValueError('replacement PT authorization missing')
            self.history=t.read(t.OUT/'authorization/stopped_execution.json')['budget']
            expected=dict.fromkeys(t.DEFAULT_LIMITS,0)|dict(prepare_starts=2,cache_rows=11750,pt_starts=1,pt_updates=10000,pt_microbatches=60000,pt_presentations=10080000)
            if self.history['charged_or_reserved']!=expected:raise ValueError('stopped cumulative budget mismatch')
        elif t.EXECUTION_REVISION:
            if auth.get('execution_revision')!='prepare-r1':raise ValueError('repair authorization mismatch')
            self.history=t.read(t.OUT/'authorization/parent_evidence.json')['budget.json']['object']
            if self.history['charged_or_reserved']!=(dict.fromkeys(t.DEFAULT_LIMITS,0)|dict(prepare_starts=1,cache_rows=5875)) or self.history['known_actual_lower_bound']['cache_rows']!=0:raise ValueError('parent budget mismatch')
            if len(self.history['jobs'])!=1 or self.history['jobs'][0]['exit']!=1:raise ValueError('parent failure mismatch')
        self.jobs=[];self.stopping=False;self.manifest=t.read(t.OUT/'execution_manifest.json')
        self.manifest.update(status='执行中',controller_pid=os.getpid(),controller_identity=t.process_identity(os.getpid()),cwd=str(ROOT),command=sys.argv,limits=t.LIMITS)
        t.write(t.OUT/'execution_manifest.json',self.manifest);self.capture_diff();self.refresh()
    def budget(self):return ledger(self.jobs,self.history)
    def deadline(self):
        if time.time()>=t.DEADLINE:raise TimeoutError('48h wallclock exhausted')
    def capture_diff(self):
        changes=[]
        if t.EXECUTION_REVISION:
            for name in (*SOURCES,'CURRENT_PLAN.md'):
                before=t.OUT/'authorization/before'/name
                if before.exists():changes.extend(difflib.unified_diff(before.read_text().splitlines(True),(ROOT/name).read_text().splitlines(True),fromfile='before/'+name,tofile='after/'+name))
            (t.OUT/'final_review/repair_diff.patch').write_text(''.join(changes));changes=[]
        for name in SOURCES:changes.extend(difflib.unified_diff([], (ROOT/name).read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+name))
        changes.extend(difflib.unified_diff((t.OUT/'authorization/CURRENT_PLAN.before.md').read_text().splitlines(True),(ROOT/'CURRENT_PLAN.md').read_text().splitlines(True),fromfile='before/CURRENT_PLAN.md',tofile='after/CURRENT_PLAN.md'))
        (t.OUT/'final_review/actual_source_diff.patch').write_text(''.join(changes))
    def collect_exits(self):
        errors=[]
        for job in self.jobs:
            if job['exit'] is not None:continue
            try:
                rc=poll_exit(job)
                if rc is not None:job.update(exit=rc,ended=t.now())
            except (ExitProtocolError,OSError,ValueError,KeyError) as error:job['infrastructure_failure']=str(error);errors.append(str(error))
        return errors
    def refresh(self):
        errors=self.collect_exits();value=self.budget();value.update(time=t.now(),elapsed_seconds=time.time()-self.manifest['started_unix'],exit_protocol_errors=errors)
        t.write(t.OUT/'budget.json',value)
        if not self.stopping and (errors or value['budget_violations']):raise RuntimeError('exit/budget protocol failure: '+repr(errors or value['budget_violations']))
        return value
    def check_failures(self):
        bad=[j for j in self.jobs if j['exit'] not in (None,0) or j.get('infrastructure_failure')]
        if bad:raise RuntimeError('job failed; no automatic retry: '+bad[0]['name']+' exit='+str(bad[0]['exit']))
    def gpu_ready(self,kind):return self.available_gpu(kind) is not None
    def available_gpu(self,kind):
        active=[j for j in self.jobs if j['kind'] in ('pt','ft','test') and j['exit'] is None]
        if any(j['kind']!=kind for j in active):raise ValueError('GPU phases cannot overlap')
        global_limit=(2 if kind=='pt' else 16) if len(t.ALLOWED_GPUS)==4 else (1 if kind=='pt' else 4)
        if len(active)>=global_limit:return None
        snapshot=gpu_snapshots(t.ALLOWED_GPUS)
        if snapshot is None:
            t.write(t.OUT/'occupancy_wait.json',dict(time=t.now(),reason='query_failed',action='wait_no_dispatch'));return None
        candidates=[];observations={}
        for gpu in t.ALLOWED_GPUS:
            on_card=[j for j in active if j.get('physical_gpu',3)==gpu]
            owned=set();identity_error=False
            try:
                for job in on_card:
                    p=Path(job['consumption'])
                    if p.exists():
                        identity=t.read(p).get('process')
                        if identity:owned.update(pid_aliases(identity))
            except (OSError,KeyError,ValueError,TypeError):identity_error=True
            unknown=snapshot[gpu]['pids']-owned
            observations[gpu]=dict(uuid=snapshot[gpu]['uuid'],active=len(on_card),unknown=sorted(unknown),known=sorted(owned),identity_error=identity_error)
            if not unknown and not identity_error and len(on_card)<(1 if kind=='pt' else 4):candidates.append((len(on_card),gpu))
        t.write(t.OUT/'occupancy_wait.json',dict(time=t.now(),cards=observations,action='dispatch_available' if candidates else 'wait_no_dispatch'))
        return min(candidates)[1] if candidates else None
    def launch(self,kind,arm=None,task=None,fold=None):
        self.deadline()
        physical_gpu=self.available_gpu(kind) if kind in ('pt','ft','test') else None
        if kind in ('pt','ft','test') and physical_gpu is None:return None
        self.refresh();self.check_failures()
        name=f'ua3m_{kind}'+(f'_{arm}' if arm else '')+(f'_{task}_{fold}' if task else '')
        if any(j['name']==name for j in self.jobs):raise ValueError('duplicate launch')
        r=reservation(kind,task,fold);charged=self.budget()['charged_or_reserved']
        if any(charged[k]+v>t.LIMITS[k] for k,v in r.items()):raise RuntimeError('budget unavailable')
        if kind in ('ft','test') and t.unit_path(arm,task,fold,kind=='test').exists():raise FileExistsError('unit output conflict')
        if kind=='pt' and (t.OUT/'pretrain'/arm).exists():raise FileExistsError('PT output conflict')
        log=t.LOG/name;log.mkdir(exist_ok=False)
        seconds=min({'prepare':3600,'pt':21600,'ft':21600,'test':1800,'cpu':600}[kind],max(1,int(t.DEADLINE-time.time())))
        if kind=='cpu':command=['python','-m','pytest','-q','-p','no:cacheprovider','tests/test_ua3m_paperfuse.py']
        elif kind=='pt':command=['python','-u','scripts/pretrain_ua3m_paperfuse.py','--arm',arm]
        else:
            command=['python','-u','scripts/run_ua3m_paperfuse.py',kind]
            if arm:command+=['--arm',arm,'--task',task,'--fold',str(fold)]
        if kind!='cpu':command+=['--execution-root',str(t.OUT)]
        if physical_gpu is not None:command+=['--physical-gpu',str(physical_gpu)]
        argv=['timeout','--signal=TERM','--kill-after=10s',str(seconds)+'s',*command]
        env=dict(UA3M_EXECUTION_ROOT=str(t.OUT),CUDA_VISIBLE_DEVICES=str(physical_gpu) if physical_gpu is not None else '',WORLD_SIZE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        consumption=(t.OUT/'inputs/coverage.json') if kind=='prepare' else (t.OUT/'pretrain'/arm/'consumption.json') if kind=='pt' else t.unit_path(arm,task,fold,kind=='test')/'consumption.json' if kind in ('ft','test') else None
        gpu_uuid=t.read(t.OUT/'occupancy_wait.json')['cards'][str(physical_gpu)]['uuid'] if physical_gpu is not None else None
        env['UA3M_PHYSICAL_GPU_UUID']=gpu_uuid or ''
        job=dict(gpu_uuid=gpu_uuid,name=name,kind=kind,arm=arm,task=task,fold=fold,attempt=t.EXECUTION_REVISION or 'r1-once',execution_root=str(t.OUT),physical_gpu=physical_gpu,allowed_gpus=list(t.ALLOWED_GPUS),resource_revision=t.EXECUTION_REVISION,argv=argv,env=env,cwd=str(ROOT),log=str(log),reservation=r,consumption=str(consumption) if consumption else None,exit=None,started=t.now())
        self.jobs.append(job);self.refresh();t.write(log/'command.json',job)
        cmd=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','python','-u','scripts/run_ua3m_paperfuse.py','wrap','--spec',str(log/'command.json'),'--execution-root',str(t.OUT)]
        job['window']=subprocess.check_output(['tmux','new-window','-d','-P','-F','#{window_id}','-t','Uni-Poly:','-n',name,'-c',str(ROOT),shlex.join(cmd)],text=True).strip()
        subprocess.run(['tmux','set-option','-w','-t',job['window'],'remain-on-exit','on'],check=True)
        pid=int(subprocess.check_output(['tmux','display-message','-p','-t',job['window'],'#{pane_pid}'],text=True));job['wrapper_identity']=t.process_identity(pid)
        self.refresh();return job
    def wait(self,jobs):
        while True:
            self.deadline();self.refresh();self.check_failures()
            if all(j['exit']==0 for j in jobs):return
            time.sleep(5)
    def pt_stage(self):
        queue=list(t.ARMS);launched=[]
        while queue or any(j['exit'] is None for j in launched):
            self.deadline();self.refresh();self.check_failures()
            while queue:
                arm=queue[0];job=self.launch('pt',arm)
                if job is None:break
                queue.pop(0);launched.append(job)
                t.update_plan('执行中','PT '+arm,f'共同step0训练10000 updates；已派发{len(launched)}/2，物理GPU{job["physical_gpu"]}，资源修订{t.EXECUTION_REVISION}。')
            if queue or any(j['exit'] is None for j in launched):time.sleep(5)
    def matrix(self,formal=False):
        queue=[(a,task,f) for a in t.ARMS for task in t.TASKS for f in range(5)];kind='test' if formal else 'ft';last=None
        while queue or any(j['exit'] is None for j in self.jobs):
            self.deadline();self.refresh();self.check_failures()
            while queue:
                a,task,f=queue[0];job=self.launch(kind,a,task,f)
                if job is None:break
                queue.pop(0)
            complete=sum(j['kind']==kind and j['exit']==0 for j in self.jobs);active=sum(j['kind']==kind and j['exit'] is None for j in self.jobs)
            if (complete,active)!=last:
                t.update_plan('执行中',kind,f'{kind}完成{complete}/70，运行中{active}，排队{len(queue)}。');last=(complete,active)
            if queue or active:time.sleep(5)
    def stop_owned(self):
        self.stopping=True;t.ignore_termination();end=time.monotonic()+30;sent=set();errors=[]
        while True:
            errors.extend(self.collect_exits());alive=[]
            for job in self.jobs:
                try:
                    group=owned_group(job)
                    if group is not None:
                        alive.append(group)
                        if group not in sent:os.killpg(group,signal.SIGTERM);sent.add(group)
                except (OSError,ValueError,KeyError) as error:errors.append(str(error))
            if not alive or time.monotonic()>=end:break
            time.sleep(min(1,max(0,end-time.monotonic())))
        for job in self.jobs:
            try:
                group=owned_group(job)
                if group is not None:os.killpg(group,signal.SIGKILL)
            except (OSError,ValueError,KeyError) as error:errors.append(str(error))
        errors.extend(self.collect_exits());t.write(t.OUT/'cleanup.json',dict(time=t.now(),bounded_seconds=30,groups=sorted(sent),errors=errors));t.write(t.OUT/'budget.json',self.budget())
    def report(self,status,error=None):
        formal={'B_A020':t.read(t.BASE/'final_review/results.json')['formal']['D_A020']};partial={}
        for arm in t.ARMS:
            try:formal[arm]=t.aggregate(arm,True)
            except (OSError,ValueError,KeyError):partial[arm]='未完成／不可比较'
        frozen=t.read(t.OUT/'selection/final_frozen.json') if (t.OUT/'selection/final_frozen.json').exists() else {}
        final=frozen.get('FINAL');pairs=t.comparisons(formal,final);delta=formal[final]['macro']-t.BASE_MACRO if final in formal else None
        budget=self.refresh();cpu=t.read(t.OUT/'final_review/cpu_exit.json') if (t.OUT/'final_review/cpu_exit.json').exists() else None
        result=dict(plan=t.PLAN,revision=t.REVISION,status=status,independent_review='PENDING',formal=formal,partial=partial,FINAL=final,delta=delta,
            performance_pass=delta>=.010 if delta is not None else None,comparisons=pairs,budget=budget,error=error,cpu=cpu,model=t.m.CONFIG,
            execution_revision=t.EXECUTION_REVISION,execution_root=str(t.OUT),geometry_missing=t.read(t.prepared_root()/'inputs/geometry_missing.json') if (t.prepared_root()/'inputs/geometry_missing.json').exists() else None)
        lines=['# UA3M-PAPERFUSE r1 实际交付','',f'状态：{status}，独立复审PENDING；未归档。',
            f'FINAL={final}；未舍入delta={delta!r}；性能通过={result["performance_pass"]}。',
            '\n科学问题：原生UA2D/3D保留，FP作为输入，分别采用UNI标准InfoNCE或TMOL循环排正项损失和三模态直接融合；原mask任务保留，无FP预测或核模型。',
            '这是完整方案包比较，含FP参与、ALIGN、下游3D和优化多因素变化，不能独立归因FP/3D/融合或interaction。','',
            '| 方法 | 正式macro R² | ΔB_A020 |','|---|---:|---:|']
        for arm in ('B_A020',*t.ARMS):
            value=formal.get(arm);lines.append(f'| {arm} | '+(f'{value["macro"]:.3f} | {value["macro"]-t.BASE_MACRO:+.3f} |' if value else '未完成／不可比较 | — |'))
        sota={}
        for arm in ('B_A020',*t.ARMS):
            sota[arm]={};lines+=['',f'## {arm} 七任务','','| Task | 项目R² | 论文参考 | 项目减参考 | 参考模型 |','|---|---:|---:|---:|---|']
            for task,(name,model,ref) in t.SOTA.items():
                value=formal.get(arm,{}).get('tasks',{}).get(task);sota[arm][task]=dict(project=value,reference=ref,reference_model=model,delta=value['mean']-ref if value else None)
                lines.append(f'| {name} | '+(f'{value["mean"]:.3f}±{value["std"]:.3f} | {ref:.3f} | {value["mean"]-ref:+.3f} | {model} |' if value else f'未完成／不可比较 | {ref:.3f} | — | {model} |'))
        for name,x in pairs.items():
            lines+=['',f'## {name}','',f'Δmacro={x["delta"]!r}；改善fold={x.get("improved_folds")}；状态{x["status"]}。']
            if x['status']=='COMPLETE':
                lines+=['','| Task | 配对差值mean±std | 改善fold |','|---|---:|---:|']
                for task,v in x['tasks'].items():lines.append(f'| {task} | {v["mean"]:+.3f}±{v["std"]:.3f} | {v["improved_folds"]}/5 |')
        if t.EXECUTION_REVISION=='gpu0123-r1':lines+=['','用户明确停止原GPU3执行后按四卡调度重启；用户明确要求不复用成功prepare；本次重新物化和初始化。中断UNI3M已完成3634 updates，整次PT收费不退款；补充重启授权及完整父/新/累计账本见authorization和budget。候选两方案均从step0重新训练，不从中断权重resume。']
        if t.EXECUTION_REVISION=='prepare-r1':lines+=['','返修prepare-r1：父prepare真实exit1、实际缓存0行及部分初始化保留；新增prepare仅一次，父预留5875行永久结转，累计预留11750行，实际新目标5875行。父历史、新执行及累计消费见budget三栏。']
        costs={}
        for arm in t.ARMS:
            costs[arm]=dict(pt=t.read(t.OUT/'pretrain'/arm/'consumption.json') if (t.OUT/'pretrain'/arm/'consumption.json').exists() else None,ft=[],test=[])
            for task in t.TASKS:
                for fold in range(5):
                    for formal_flag,key in ((False,'ft'),(True,'test')):
                        folder=t.unit_path(arm,task,fold,formal_flag)
                        if (folder/'consumption.json').exists():costs[arm][key].append(dict(task=task,fold=fold,consumption=t.read(folder/'consumption.json'),parameters=t.read(folder/'parameters.json') if (folder/'parameters.json').exists() else None))
        result.update(sota=sota,costs=costs)
        lines+=['','## 成本、可用率及协议','',json.dumps(budget['known_actual_lower_bound'],ensure_ascii=False),
            '所有完整精度fold差值、参数量、PT/FT/TEST实际耗时及可用模态呈现见results.json成本表与单元consumption/history；所有计时来自真实运行，未做benchmark。',
            '下游原行数5875，3D有效5860，固定缺失15（Egb7/Egc8）；缺失3D不进入编码器，权重严格0，其他模态重归一化。PT无效635条采样前排除；缺失名单不是运行错误白名单。',
            '七任务五折等权mean±std(ddof=1)，seed42+fold，七任务等权macro。论文是逐任务最佳包络，数据/split/协议等价性未证明，仅数值对照；Egc3380/Xc432与论文4124/430不同，共享开发TEST非独立盲测。',
            '基线D_A020 legacy PT随机偏差及历史ALIGN×2保留为基线限制。本轮UNI不继承ALIGN×2，TMOL保留作者可负损失数学；不同RNG种子不证明统计独立性。本项目适配而非论文完整复现，不宣称SOTA。',
            f'异常/未完成原因：{error or ("无" if not partial else "见partial，未完成／不可比较")}。',
            f'最终CPU：{json.dumps(cpu,ensure_ascii=False) if cpu else "尚未尝试"}。',
            'tmux '+self.manifest['tmux_window']+'；各独立worker窗口/命令/PID/起止/真实exit见budget.jobs及'+str(t.LOG)+'。']
        t.write(t.OUT/'final_review/results.json',result);report='\n'.join(lines)+'\n';(t.OUT/'final_review/report.md').write_text(report);t.update_plan(status,'交付报告',report)
    def run(self):
        try:
            t.update_plan('执行中','一次prepare','重新物化5875行既有结构输入和初始化；不复用上次prepare产物，无模型forward、FP或构象生成。');self.wait([self.launch('prepare')])
            t.check_prepared()
            self.pt_stage();self.matrix(False);frozen=t.freeze()
            t.update_plan('执行中','一次冻结',f'FINAL={frozen["FINAL"]}，70 BEST和35基线引用已冻结，随后两个方案全部正式TEST。')
            self.matrix(True);self.report('执行中')
            t.update_plan('执行中','唯一最终CPU suite','全部授权实验和完整报告结束；仅尝试一次600秒CPU suite。')
            job=self.launch('cpu');self.wait([job]);t.write(t.OUT/'final_review/cpu_exit.json',dict(exit=job['exit'],attempts=1,command=job['argv'],env=job['env'],log=job['log'],time=t.now()))
            self.report('待审查');self.manifest.update(status='待审查',ended=t.now(),elapsed_seconds=time.time()-self.manifest['started_unix'],cpu_exit=0);t.write(t.OUT/'execution_manifest.json',self.manifest);self.capture_diff();self.refresh();return 0
        except BaseException:
            error=traceback.format_exc()
            try:self.stop_owned()
            except BaseException:error+='\ncleanup: '+traceback.format_exc()
            status='阻断' if '48h wallclock exhausted' in error else '需返修'
            t.write(t.OUT/'failure.json',dict(status=status,error=error,time=t.now(),automatic_retry=False,automatic_resume=False))
            cpu=[j for j in self.jobs if j['kind']=='cpu'];t.write(t.OUT/'final_review/cpu_exit.json',dict(exit=cpu[0]['exit'] if cpu else None,attempts=len(cpu),not_run_reason=None if cpu else 'authorized experiments/report not complete; no early suite'))
            self.manifest.update(status=status,error=error,ended=t.now());t.write(t.OUT/'execution_manifest.json',self.manifest);self.report(status,error);self.capture_diff();self.refresh();return 1


def wrap(spec):
    job=t.read(spec);folder=Path(job['log']);env=os.environ.copy();env.update(job['env']);identity=None;requested=[];forwarded=False
    def stop(sig,frame):
        nonlocal forwarded
        requested.append(sig)
        if not forwarded and identity is not None and t.process_alive(identity) and os.getpgid(identity['pid'])==identity['pid']:
            forwarded=True;os.killpg(identity['pid'],signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (folder/'stdout.log').open('x') as out,(folder/'stderr.log').open('x') as err:
        child=subprocess.Popen(job['argv'],cwd=job['cwd'],env=env,stdout=out,stderr=err,start_new_session=True);identity=t.process_identity(child.pid)
        t.write(folder/'process.json',dict(wrapper=t.process_identity(os.getpid()),timeout_identity=identity,command=job['argv'],cwd=job['cwd'],started=t.now()))
        if requested:stop(requested[0],None)
        rc=child.wait()
    atomic_exit(folder/'exit.txt',rc);t.write(folder/'end.json',dict(exit=rc,ended=t.now(),job=job['name'],attempt=job.get('attempt','r1-once')));return rc


def install_termination():
    def stop(sig,frame):t.ignore_termination();raise RuntimeError('controlled termination '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);return stop


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('controller','prepare','ft','test','wrap'));p.add_argument('--arm',choices=t.ARMS);p.add_argument('--task',choices=t.TASKS);p.add_argument('--fold',type=int,choices=range(5));p.add_argument('--spec');p.add_argument('--physical-gpu',type=int,choices=range(4),default=3);p.add_argument('--execution-root');args=p.parse_args();t.configure_execution(args.execution_root);t.torch.set_num_threads(2)
    if args.action=='wrap':return wrap(args.spec)
    install_termination()
    if args.action=='controller':return Controller().run()
    if args.action=='prepare':t.prepare();return 0
    if args.arm is None or args.task is None or args.fold is None:p.error('fixed matrix identity required')
    t.unit(args.arm,args.task,args.fold,args.action=='test',args.physical_gpu);return 0

if __name__=='__main__':raise SystemExit(main())
