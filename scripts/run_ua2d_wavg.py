#!/usr/bin/env python3
"""One isolated fixed WAVG controller; one FT per task/fold, no retry or replay."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'2')
import argparse
import copy
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
from src.training.ua2d_wavg import *
SOURCES=('src/training/ua2d_wavg.py','scripts/run_ua2d_wavg.py','tests/test_ua2d_wavg.py')


class Controller:
    def __init__(self):
        require_tmux();self.lock=(OUT/'controller.lock').open('x');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'budget.json').exists() or (OUT/'execution_manifest.json').exists():raise FileExistsError('cycle already started; no resume/retry')
        self.auth=read(OUT/'authorization/user_authorization.json')
        if self.auth['plan']!=PLAN or self.auth['limits']!=LIMITS:raise ValueError('authorization mismatch')
        self.jobs=[];self.started_unix=None;self.start=None;self.cached_state=None;self.cached_budget=None;self.evidence_only=False
        self.validation={};self.formal={};self.partial={};self.frozen=None
        self.window=subprocess.check_output(['tmux','display-message','-p','-t',os.environ['TMUX_PANE'],'#{window_id}'],text=True).strip()
        self.manifest=dict(plan=PLAN,revision=REVISION,executor='DSH',writer_pid=os.getpid(),controller_window=self.window,
            command=sys.argv,cwd=str(ROOT),authorization=str(OUT/'authorization/user_authorization.json'),limits=LIMITS,
            only_gpus=[2,3],concurrency_per_gpu=4,wallclock_limit_seconds=43200,wallclock_started_unix=None,
            automatic_retry=False,automatic_resume=False,independent_review='PENDING',log=str(LOG/'controller'))
        with (OUT/'execution_manifest.json').open('x') as f:json.dump(self.manifest,f,ensure_ascii=False,indent=2)
        transfer=read(REFERENCE_TRANSFER);opt=read(REFERENCE_OPT)
        if not transfer.get('complete_formal_TEST') or not opt.get('complete_formal_TEST'):raise ValueError('project reference TEST incomplete')
        self.references=dict(B_FP=read(REFERENCE_BFP)['test']['B_FP'],A_MIX246=opt['formal']['A_MIX246'],R_MEAN=transfer['formal']['R_MEAN'])
        for label,value in self.references.items():
            if set(value['tasks'])!=set(TASKS) or any(len(value['tasks'][t]['folds'])!=5 for t in TASKS):raise ValueError('incomplete reference: '+label)
            finite([value['macro'],*[x for t in TASKS for x in value['tasks'][t]['folds']]])
        write(OUT/'authorization/reference_results.json',dict(results=self.references,paths=dict(B_FP=str(REFERENCE_BFP),A_MIX246=str(REFERENCE_OPT),R_MEAN=str(REFERENCE_TRANSFER)),reevaluated=False))
        cached=cache_metadata(CACHE_ROOT,CACHE_SCHEMA)
        if read(CACHE_ROOT/'original_splits.json')!={t:read(SPLIT/f'{t}.json') for t in TASKS}:raise ValueError('shared cache original split differs')
        self.manifest['cache_reference']=dict(root=str(CACHE_ROOT),schema=CACHE_SCHEMA,original_plan=CACHE_PLAN,metadata=cached,readonly=True,copied=False)
        write(OUT/'execution_manifest.json',self.manifest);write(OUT/'cache_reference.json',self.manifest['cache_reference'])
        expected=sum(self.updates(t,f) for t in TASKS for f in range(5))
        if expected!=56000:raise ValueError('fixed original FT split/update budget differs')
        self.capture_diff();self.refresh()

    def capture_diff(self):
        patch=[]
        for rel in SOURCES:patch.extend(difflib.unified_diff([], (ROOT/rel).read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+rel))
        (OUT/'final_review/actual_source_diff.patch').write_text(''.join(patch))
        before=(OUT/'authorization/CURRENT_PLAN.before.md').read_text()
        (OUT/'final_review/current_plan_diff.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),(ROOT/'CURRENT_PLAN.md').read_text().splitlines(True),fromfile='a/CURRENT_PLAN.md',tofile='b/CURRENT_PLAN.md')))

    def updates(self,task,fold):return 70*math.ceil(len(read(SPLIT/f'{task}.json')['folds'][fold]['train_indices'])/24)
    def deadline(self):
        if self.start is not None and time.monotonic()-self.start>=43200:raise TimeoutError('new cycle 12h deadline exhausted')
    def begin_clock(self):
        if self.start is None:
            self.start=time.monotonic();self.started_unix=time.time();self.manifest.update(wallclock_started_unix=self.started_unix,wallclock_started=now())
            write(OUT/'execution_manifest.json',self.manifest)

    def refresh(self):
        used=dict.fromkeys(LIMITS,0);known=dict.fromkeys(LIMITS,0);remaining=dict.fromkeys(LIMITS,0);uncertain=[]
        for j in self.jobs:
            key={'ft':'ft_starts','test':'test_starts','cpu':'cpu_suites'}[j['kind']]
            used[key]+=1;known[key]+=1
            p=Path(j['log'])/'exit.txt'
            if j['exit'] is None and p.exists():j.update(exit=int(p.read_text().strip()),ended=now())
            if j['kind'] in ('ft','test'):
                c=j.get('progress',{});p=Path(j['consumption'])
                if p.exists() and (j['exit'] is None or c.get('status')!='COMPLETE'):c=read(p);j['progress']=c
                values=conservative_ft_counts(c)
                for k,v in values['charged'].items():used[k]+=v;known[k]+=values['known_actual_lower_bound'][k]
                if values['open_epoch']:uncertain.append(dict(job=j['name'],**values))
                if not c and j['exit'] is not None:
                    for k,v in j['reservation'].items():used[k]+=v
                    uncertain.append(dict(job=j['name'],reason='no durable consumption; full dispatch reservation charged'))
                if j['exit'] is None:
                    for k,v in j['reservation'].items():remaining[k]+=max(0,v-values['charged'].get(k,0))
        violations=[k for k in LIMITS if used[k]+remaining[k]>LIMITS[k]]
        payload=dict(plan=PLAN,revision=REVISION,limits=LIMITS,used=used,known_actual_lower_bound=known,
            active_remaining_reservation=remaining,uncertainty=uncertain,budget_violations=violations,jobs=self.jobs,
            training_validation_started=sum(j.get('progress',{}).get('training_validation_started',0) for j in self.jobs if j['kind']=='ft'),
            endpoint_validation_started=sum(j.get('progress',{}).get('endpoint_validation_started',0) for j in self.jobs if j['kind']=='ft'),
            accounting='35 shared trajectories counted once; train validation2450 plus endpoint70; open epoch charged in full; no retry/resume')
        encoded=json.dumps(payload,sort_keys=True,allow_nan=False)
        if encoded!=self.cached_state:
            payload.update(time=now(),wallclock_started_unix=self.started_unix,elapsed_seconds=time.monotonic()-self.start if self.start else 0.)
            write(OUT/'budget.json',payload);self.cached_budget=payload;self.cached_state=encoded
        if violations and not self.evidence_only:raise RuntimeError('budget exceeded: '+','.join(violations))
        return self.cached_budget

    def check_failures(self):
        failed=[j for j in self.jobs if j['exit'] not in (None,0)]
        if failed:raise RuntimeError('job failed; no retry/resume: '+failed[0]['name']+' exit='+str(failed[0]['exit']))

    def launch(self,name,kind,args=(),gpu=None,estimate=None,task=None,fold=None,allow_final_cpu=False):
        self.deadline();b=self.refresh()
        if not allow_final_cpu:self.check_failures()
        if b['budget_violations']:raise RuntimeError('no further dispatch after budget violation')
        if kind not in ('ft','test','cpu'):raise ValueError('PT/extra phases prohibited')
        if any(j['name']==name for j in self.jobs):raise ValueError('duplicate launch; no retry')
        key={'ft':'ft_starts','test':'test_starts','cpu':'cpu_suites'}[kind]
        if b['used'][key]>=LIMITS[key]:raise RuntimeError('start allowance exhausted')
        active=[j for j in self.jobs if j['exit'] is None]
        if any(j['kind']!=kind for j in active):raise ValueError('stage overlap prohibited')
        if kind in ('ft','test'):
            if gpu not in (2,3) or sum(j['gpu']==gpu for j in active)>=4:raise ValueError('GPU/concurrency violation')
            output=trajectory_path(task,fold) if kind=='ft' else unit_path(estimate,task,fold,True)
            if output.exists():raise FileExistsError('output conflict')
        if kind=='ft':
            cache_metadata(CACHE_ROOT,CACHE_SCHEMA);reservation=dict(epochs_started=70,validation_started=72,attempted_updates=self.updates(task,fold))
        elif kind=='test':
            frozen=read(OUT/'selection/final_frozen.json')
            if not frozen['frozen'] or frozen['models']!=list(KINDS) or estimate not in frozen['models']:raise ValueError('TEST outside fixed freeze')
            reservation=dict(test_graphs=len(frozen['units'][f'{estimate}/{task}/{fold}']['test_indices']))
        else:reservation={}
        for k,v in reservation.items():
            if b['used'][k]+b['active_remaining_reservation'][k]+v>LIMITS[k]:raise RuntimeError('insufficient full-unit reservation')
        if kind=='ft':self.begin_clock()
        log=LOG/name;log.mkdir(exist_ok=False)
        seconds={'ft':7200,'test':300,'cpu':300}[kind]
        seconds=min(seconds,max(1,int(43200-(time.monotonic()-self.start))))
        if kind=='cpu':command=[sys.executable,'-m','pytest','-q','tests/test_ua2d_wavg.py']
        else:command=[sys.executable,'-u','scripts/run_ua2d_wavg.py',kind,*args]
        env=dict(CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else '',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        actual=['timeout','--signal=TERM','--kill-after=10s',f'{seconds}s',*command]
        text=shlex.join(['env',*[f'{k}={v}' for k,v in env.items()],*actual])
        script=log/'run.sh';script.write_text('#!/bin/bash\ncd '+shlex.quote(str(ROOT))+' || exit 90\ndate -Iseconds > '+shlex.quote(str(log/'start.txt'))+'\n'+text+' > '+shlex.quote(str(log/'stdout.log'))+' 2> '+shlex.quote(str(log/'stderr.log'))+'\nua_wavg_exit=$?\nprintf "%s\\n" "$ua_wavg_exit" > '+shlex.quote(str(log/'exit.txt'))+'\ndate -Iseconds > '+shlex.quote(str(log/'end.txt'))+'\nexit "$ua_wavg_exit"\n')
        j=dict(name=name,kind=kind,estimate=estimate,task=task,fold=fold,gpu=gpu,argv=actual,env=env,command=text,cwd=str(ROOT),log=str(log),
            consumption=str((trajectory_path(task,fold) if kind=='ft' else unit_path(estimate,task,fold,True))/'consumption.json') if kind in ('ft','test') else None,
            exit=None,started=now(),reservation=reservation)
        self.jobs.append(j);self.refresh();write(log/'command.json',j)
        j['window']=subprocess.check_output(['tmux','new-window','-d','-P','-F','#{window_id}','-t','Uni-Poly:','-n',name,'bash',str(script)],text=True).strip()
        subprocess.run(['tmux','set-option','-w','-t',j['window'],'remain-on-exit','on'],check=True);self.refresh();return j

    def wait(self,jobs):
        while True:
            self.deadline();self.refresh()
            if any(j['exit'] not in (None,0) for j in jobs):raise RuntimeError('job failed; no automatic retry')
            if all(j['exit']==0 for j in jobs):return
            time.sleep(5)

    def matrix(self,formal=False):
        if any(j['exit'] is None for j in self.jobs):raise ValueError('previous stage not closed')
        queue=([(k,t,f) for k in KINDS for t in TASKS for f in range(5)] if formal else [(None,t,f) for t in TASKS for f in range(5)])
        active=[];kind='test' if formal else 'ft'
        while queue or active:
            self.deadline();self.refresh();self.check_failures();active=[j for j in active if j['exit'] is None]
            for gpu in (2,3):
                while queue and sum(j['gpu']==gpu for j in active)<4:
                    k,t,f=queue.pop(0);args=['--task',t,'--fold',str(f)]
                    if formal:args+=['--estimate',k]
                    name=f'wavg_test_{k}_{t}_{f}' if formal else f'wavg_ft_{t}_{f}'
                    active.append(self.launch(name,kind,args,gpu=gpu,estimate=k,task=t,fold=f))
            if active:time.sleep(5)

    def stop_owned(self):
        def exits():
            for j in self.jobs:
                p=Path(j['log'])/'exit.txt'
                if j['exit'] is None and p.exists():j.update(exit=int(p.read_text().strip()),ended=now())
        exits();active=[j for j in self.jobs if j['exit'] is None and j.get('window')]
        for j in active:
            r=subprocess.run(['tmux','list-panes','-t',j['window'],'-F','#{pane_pid}'],capture_output=True,text=True)
            if r.returncode:continue
            roots={int(x) for x in r.stdout.split()};parents={}
            for pid in roots:
                p=Path('/proc')/str(pid)
                if str(Path(j['log'])/'run.sh') not in (p/'cmdline').read_bytes().replace(b'\0',b' ').decode():raise ValueError('owned wrapper identity mismatch')
            for p in Path('/proc').iterdir():
                if not p.name.isdigit():continue
                try:parents[int(p.name)]=int((p/'stat').read_text().split(') ',1)[1].split()[1])
                except (OSError,ValueError,IndexError):pass
            descendants=set(roots)
            while True:
                more={p for p,pp in parents.items() if pp in descendants}-descendants
                if not more:break
                descendants|=more
            for pid in sorted(descendants-roots,reverse=True):
                try:os.kill(pid,signal.SIGTERM)
                except ProcessLookupError:pass
        deadline=time.monotonic()+20
        while any(j['exit'] is None for j in active) and time.monotonic()<deadline:time.sleep(1);exits()
        if any(j['exit'] is None for j in active):raise RuntimeError('owned processes/true exits not safely closed; no CPU overlap')


    def collect(self):
        self.partial={}
        for kind in KINDS:
            try:self.validation[kind]=aggregate(kind)
            except (OSError,ValueError,KeyError):
                completed={}
                for task in TASKS:
                    for fold in range(5):
                        p=unit_path(kind,task,fold);c=trajectory_path(task,fold)/'consumption.json'
                        if (p/'metrics.json').exists() and c.exists() and read(c).get('status')=='COMPLETE':completed[f'{task}/{fold}']=read(p/'metrics.json')
                if completed:self.partial[kind]=dict(status='INCOMPLETE',complete_units=len(completed),metrics=completed)
        write(OUT/'selection/validation.json',dict(complete=self.validation,partial=self.partial))

    def deliver(self,error,cpu_exit):
        self.collect();b=self.refresh();final=self.frozen['FINAL'] if self.frozen else None
        complete=set(self.formal)==set(KINDS)
        selected=self.formal.get(final);direct=selected['macro']-self.formal['BEST']['macro'] if complete else None
        historical=selected['macro']-self.references['B_FP']['macro'] if complete else None
        status='待审查' if not error and cpu_exit==0 and complete and set(self.validation)==set(KINDS) else '需返修'
        comparisons=registered_comparisons(self.formal,self.references,final)
        result=dict(plan=PLAN,revision=REVISION,status=status,error=error,independent_review='PENDING',validation=self.validation,
            partial_validation=self.partial,formal=self.formal,FINAL=final,complete_formal_TEST=complete,
            delta_FINAL_same_trajectory_BEST=direct,delta_FINAL_historical_B_FP=historical,**performance(direct,historical),
            comparisons=comparisons,references=self.references,cpu_exit=cpu_exit,budget=b,
            cache_reference=self.manifest['cache_reference'],aggregation='five equal folds R2, std(ddof1), seed42+fold; seven equal task means',
            interpretation='BEST/SWA4/EMA60 come from the same 35 FT trajectories, not three independent training groups',
            protocol='shared development TEST; paper protocol equivalence unproven; numeric comparison only; Egc3380/Xc432 vs paper4124/430')
        lines=['# UA2D-WAVG r1 实际交付','',f'状态：{status}；DSH执行，待Codex独立审查。',
            f'FINAL={final or "未冻结"}；完整105单元正式TEST={complete}；双参考性能门槛：{result["performance_status"]}；CPU exit={cpu_exit}。','',
            '35条FT轨迹各只训练一次，共享产生BEST／SWA4／EMA60；不称三组独立训练。训练validation最多2450次，SWA／EMA终点额外各35次，BEST复用保存的预测。',
            'SWA4固定全局epoch40/50/60/70，保留4份纯模型权重；EMA60固定global11—70、beta0.9、S0=z0=0、最终S/z，最终S／z及60次更新记录保留。',
            '平均器独立detached FP32，覆盖完整浮点state；未修改学生／梯度／optimizer或RNG，无teacher、额外backward或平均模型逐epoch评估。',
            'prepare／公共初始化／PT均0，只读引用TRANSFER的5875条原2-hop缓存，保留其原plan和schema，不复制或冒充新缓存。',
            f'历史B_FP macro={self.references["B_FP"]["macro"]:.3f}，A_MIX246={self.references["A_MIX246"]["macro"]:.3f}，R_MEAN={self.references["R_MEAN"]["macro"]:.3f}，均直接复用。','',
            '| 预登记正式对照 | Δmacro R² | 完整性 |','|---|---:|---|']
        for label,c in comparisons.items():
            delta=f'{c["delta"]:+.3f}' if c['delta'] is not None else '未完成／不可比较';lines.append(f'| {label} | {delta} | {c["status"]} |')
        sota={}
        for kind in KINDS:
            values=self.formal.get(kind);sota[kind]={};macro=f'{values["macro"]:.3f}' if values else '未完成／不可比较'
            lines+=['',f'## {kind} 七任务SOTA数值对照（macro={macro}）','','| Task | 正式R² | 论文参考R² | 项目减参考 | 参考模型 |','|---|---:|---:|---:|---|']
            for task,(name,model,ref) in SOTA.items():
                x=values['tasks'][task] if values else None;sota[kind][task]=dict(project=x,reference=ref,reference_model=model,delta=x['mean']-ref if x else None,status='COMPLETE' if x else '未完成／不可比较')
                score=f'{x["mean"]:.3f}±{x["std"]:.3f}' if x else '未完成／不可比较';delta=f'{x["mean"]-ref:+.3f}' if x else '—'
                lines.append(f'| {name} | {score} | {ref:.3f} | {delta} | {model} |')
        lines+=['','性能通过须FINAL对同轨迹BEST及历史B_FP的未舍入Δmacro R²均≥0.010；失败不搜索窗口／系数或增加seed。',
            '每任务五折等权R²，std(ddof=1)，seed42+fold，再七任务等权macro；原始值完整精度，展示3位小数。共享开发TEST不是独立盲测，论文协议等价性未证明，只作数值比较。',
            '35份完整history／最终resume／SWA源／EMA累积器及记录在trajectories/；三估计checkpoint／scaler／validation在estimates/，正式预测在formal/；来源冻结selection/final_frozen.json；所有预登记配对差值final_review/results.json；预算budget.json；完整源码diff与训练计时final_review/；真实命令／window／exit在logs/ua2d_wavg_20261022/r1/。',
            f'未完成／阻断原因：{error or ("无" if complete else "正式流程未完整结束")}。']
        result['sota']=sota;write(OUT/'final_review/results.json',result)
        timings={};sources={}
        for task in TASKS:
            for fold in range(5):
                p=trajectory_path(task,fold)
                if (p/'epoch_timing.json').exists():timings[f'{task}/{fold}']=read(p/'epoch_timing.json')
                if (p/'estimator_sources.json').exists():sources[f'{task}/{fold}']=read(p/'estimator_sources.json')
        write(OUT/'final_review/estimator_sources.json',sources)
        write(OUT/'final_review/training_timing.json',dict(trajectories=timings,jobs=[{k:j.get(k) for k in ('name','started','ended','exit','window','log')} for j in self.jobs],measurement='real training, averaging, checkpoint writes and authorized endpoints only; no benchmark or added CUDA sync'))
        report='\n'.join(lines)+'\n';(OUT/'final_review/report.md').write_text(report)
        update_plan(status,'执行交付',report);self.manifest.update(status=status,ended=now());write(OUT/'execution_manifest.json',self.manifest);self.capture_diff();return status

    def run(self):
        error=None;cpu_exit=None;safe=True
        try:
            update_plan('执行中','35条共享FT轨迹',f'controller {self.window}；原缓存只读引用，prepare／初始化／PT=0；首次FT启动计12小时。每轨迹70训练validation＋2终点validation，无平均模型逐epoch评估。')
            self.matrix();self.collect();self.frozen=freeze_final(self.validation)
            update_plan('执行中','唯一105单元正式TEST',f'FINAL={self.frozen["FINAL"]}仅按SWA4／EMA60完整validation提名；BEST直接对照。三估计105个checkpoint类型／来源／config／scaler／split／indices一次冻结，不依TEST改选。')
            self.matrix(True)
            for kind in KINDS:self.formal[kind]=aggregate(kind,True)
        except BaseException:
            error=traceback.format_exc();write(OUT/'failure.json',dict(error=error,time=now(),automatic_retry=False,automatic_resume=False));self.evidence_only=True
            try:self.stop_owned()
            except BaseException:error+='\n'+traceback.format_exc();safe=False
        finally:
            if self.frozen:
                for kind in KINDS:
                    try:self.formal[kind]=aggregate(kind,True)
                    except (OSError,ValueError,KeyError):pass
            if safe:
                try:
                    update_plan('执行中' if not error else '需返修','唯一最终CPU suite','训练与TEST结束或安全异常收尾后，只执行tests/test_ua2d_wavg.py一次，≤300秒，失败不重测。')
                    j=self.launch('wavg_final_cpu','cpu',allow_final_cpu=True);self.wait([j]);cpu_exit=j['exit']
                except BaseException:error=(error or '')+'\n'+traceback.format_exc();cpu_exit=next((j['exit'] for j in self.jobs if j['kind']=='cpu'),None)
            status=self.deliver(error,cpu_exit);write(LOG/'controller/exit.json',dict(exit=0 if status=='待审查' else 1,status=status,error=error,time=now()))
        return 0 if status=='待审查' else 1


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['controller','ft','test']);p.add_argument('--task',choices=TASKS);p.add_argument('--fold',type=int,choices=range(5));p.add_argument('--estimate',choices=KINDS);args=p.parse_args();torch.set_num_threads(2)
    def terminate(signum,frame):raise RuntimeError(f'controlled termination {signum}; no retry/resume')
    signal.signal(signal.SIGTERM,terminate)
    if args.action=='controller':return Controller().run()
    if args.task is None or args.fold is None:p.error('fixed trajectory task/fold required')
    if args.action=='ft':
        if args.estimate is not None:p.error('three estimates share one FT, never train an estimate separately')
        run_ft(args.task,args.fold)
    else:
        if args.estimate is None:p.error('fixed TEST estimator required')
        run_test(args.estimate,args.task,args.fold)
    return 0

if __name__=='__main__':raise SystemExit(main())
