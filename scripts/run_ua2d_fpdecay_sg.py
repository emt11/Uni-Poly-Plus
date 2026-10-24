#!/usr/bin/env python3
"""One isolated finite single-GPU FP decay controller; never resumes or imports old ledgers."""
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
from src.training.ua2d_fpdecay_sg import *
SOURCES=('src/dataset/ua2d_fpdecay_sg.py','src/training/ua2d_fpdecay_sg.py','scripts/pretrain_ua2d_fpdecay_sg.py','scripts/run_ua2d_fpdecay_sg.py','tests/test_ua2d_fpdecay_sg.py')


class Controller:
    def __init__(self):
        require_tmux();self.lock=(OUT/'controller.lock').open('x');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'budget.json').exists() or (OUT/'execution_manifest.json').exists():raise FileExistsError('cycle already started; no resume/retry')
        self.auth=read(OUT/'authorization/user_authorization.json')
        if self.auth['plan']!=PLAN or self.auth['revision']!=REVISION or self.auth['limits']!=LIMITS:raise ValueError('authorization mismatch')
        self.jobs=[];self.started_unix=None;self.start=None;self.cached_state=None;self.cached_budget=None;self.evidence_only=False
        self.validation={};self.formal={};self.partial={};self.frozen=None
        self.window=subprocess.check_output(['tmux','display-message','-p','-t',os.environ['TMUX_PANE'],'#{window_id}'],text=True).strip()
        self.manifest=dict(plan=PLAN,revision=REVISION,executor='DSH',writer_pid=os.getpid(),controller_window=self.window,
            command=sys.argv,cwd=str(ROOT),authorization=str(OUT/'authorization/user_authorization.json'),limits=LIMITS,
            only_gpus=[3],world_size=1,pt_concurrency=1,concurrency_per_gpu=4,preregistered_candidate='FP_DECAY',wallclock_limit_seconds=86400,wallclock_started_unix=None,
            automatic_retry=False,automatic_resume=False,independent_review='PENDING',log=str(LOG/'controller'))
        with (OUT/'execution_manifest.json').open('x') as f:json.dump(self.manifest,f,ensure_ascii=False,indent=2)
        self.baseline=read(REFERENCE_BFP)['test']['B_FP']
        if set(self.baseline['tasks'])!=set(TASKS) or any(len(self.baseline['tasks'][t]['folds'])!=5 for t in TASKS):raise ValueError('incomplete B_FP formal reference')
        finite([self.baseline['macro'],*[x for t in TASKS for x in self.baseline['tasks'][t]['folds']]])
        scale=read(REFERENCE_PTSCALE)
        if not scale.get('complete_formal_TEST') or scale.get('FINAL')!='P10K':raise ValueError('PTSCALE/P10K formal reference not complete')
        self.p10k=scale['formal']['P10K']
        if set(self.p10k['tasks'])!=set(TASKS) or any(len(self.p10k['tasks'][t]['folds'])!=5 for t in TASKS):raise ValueError('incomplete P10K reference')
        finite([self.p10k['macro'],*[x for t in TASKS for x in self.p10k['tasks'][t]['folds']]])
        write(OUT/'authorization/reference_results.json',dict(B_FP=dict(source=str(REFERENCE_BFP),result=self.baseline),
            P10K=dict(source=str(REFERENCE_PTSCALE),result=self.p10k),reevaluated=False,
            interpretation='Only FP_DECAY-BASE controlled; P10K/B_FP are cross-resource/budget numerical references'))
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
        if self.start is not None and time.monotonic()-self.start>=86400:raise TimeoutError('new cycle 24h deadline exhausted')
    def begin_clock(self):
        if self.start is None:
            self.start=time.monotonic();self.started_unix=time.time();self.manifest.update(wallclock_started_unix=self.started_unix,wallclock_started=now())
            write(OUT/'execution_manifest.json',self.manifest)

    def refresh(self):
        used=dict.fromkeys(LIMITS,0);known=dict.fromkeys(LIMITS,0);remaining=dict.fromkeys(LIMITS,0);uncertain=[]
        for j in self.jobs:
            key={'pt':'pt_starts','ft':'ft_starts','test':'test_starts','cpu':'cpu_suites'}[j['kind']]
            used[key]+=1;known[key]+=1
            p=Path(j['log'])/'exit.txt'
            if j['exit'] is None and p.exists():j.update(exit=int(p.read_text().strip()),ended=now())
            if j['kind']=='pt':
                p=OUT/'pretrain'/j['arm']/'consumption.json';c=read(p) if p.exists() else {}
                attempted=int(c.get('attempted_updates',0));consumed=int(c.get('consumed_presentations',0))
                charged=attempted if c or j['exit'] in (None,0) else 5000
                used['pt_updates']+=charged;used['pt_presentations']+=charged*1008
                known['pt_updates']+=attempted;known['pt_presentations']+=consumed
                j['progress']=dict(c,charged_presentations=charged*1008,known_consumed_presentations=consumed,
                    read_prefetched_presentations=c.get('sample_presentations',0))
                if charged*1008!=consumed:uncertain.append(dict(job=j['name'],charged_presentations=charged*1008,known_consumed=consumed,reason='full in-progress/failed update reserved; reads/prefetch separate'))
                if j['exit'] is None:
                    remaining['pt_updates']+=max(0,5000-charged);remaining['pt_presentations']+=max(0,5040000-charged*1008)
            elif j['kind'] in ('ft','test'):
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
            accounting='all failed/partial starts count; unclosed epoch charged in full, known actual separately; PT attempted update conservatively reserves1008 presentations; read/prefetch recorded separately; no retry/resume')
        encoded=json.dumps(payload,sort_keys=True,allow_nan=False)
        if encoded!=self.cached_state:
            payload.update(time=now(),wallclock_started_unix=self.started_unix,elapsed_seconds=time.monotonic()-self.start if self.start else 0.)
            write(OUT/'budget.json',payload);self.cached_budget=payload;self.cached_state=encoded
        if violations and not self.evidence_only:raise RuntimeError('budget exceeded: '+','.join(violations))
        return self.cached_budget

    def check_failures(self):
        failed=[j for j in self.jobs if j['exit'] not in (None,0)]
        if failed:raise RuntimeError('job failed; no retry/resume: '+failed[0]['name']+' exit='+str(failed[0]['exit']))

    def launch(self,name,kind,args=(),gpu=None,arm=None,task=None,fold=None,allow_final_cpu=False):
        self.deadline();b=self.refresh()
        if not allow_final_cpu:self.check_failures()
        if b['budget_violations']:raise RuntimeError('no further dispatch after budget violation')
        if kind not in ('pt','ft','test','cpu'):raise ValueError('extra phases prohibited')
        if any(j['name']==name for j in self.jobs):raise ValueError('duplicate launch; no retry')
        key={'pt':'pt_starts','ft':'ft_starts','test':'test_starts','cpu':'cpu_suites'}[kind]
        if b['used'][key]>=LIMITS[key]:raise RuntimeError('start allowance exhausted')
        active=[j for j in self.jobs if j['exit'] is None]
        if any(j['kind']!=kind for j in active):raise ValueError('stage overlap prohibited')
        if kind in ('ft','test'):
            if gpu!=3 or sum(j['gpu']==gpu for j in active)>=4:raise ValueError('GPU/concurrency violation')
            if unit_path(arm,task,fold,kind=='test').exists():raise FileExistsError('output conflict')
        if kind=='ft':
            cache_metadata(CACHE_ROOT);reservation=dict(epochs_started=70,validation_started=70,attempted_updates=self.updates(task,fold))
        elif kind=='pt':
            if active:raise ValueError('PT branches strictly serial')
            if arm not in ARMS:raise ValueError('unregistered PT arm')
            if (OUT/'pretrain'/arm).exists():raise FileExistsError('PT branch output conflict')
            cache_references();reservation=dict(pt_updates=5000,pt_presentations=5040000)
        elif kind=='test':
            frozen=read(OUT/'selection/final_frozen.json')
            if not frozen['frozen'] or frozen['models']!=list(ARMS) or arm not in frozen['models']:raise ValueError('TEST outside fixed freeze')
            reservation=dict(test_graphs=len(frozen['units'][f'{arm}/{task}/{fold}']['test_indices']))
        else:reservation={}
        for k,v in reservation.items():
            if b['used'][k]+b['active_remaining_reservation'][k]+v>LIMITS[k]:raise RuntimeError('insufficient full-unit reservation')
        if kind=='pt':self.begin_clock()
        log=LOG/name;log.mkdir(exist_ok=False)
        seconds={'pt':21600,'ft':7200,'test':300,'cpu':300}[kind]
        seconds=min(seconds,max(1,int(86400-(time.monotonic()-self.start)))) if self.start is not None else seconds
        if kind=='pt':command=[sys.executable,'-u','scripts/pretrain_ua2d_fpdecay_sg.py','--arm',arm]
        elif kind=='cpu':command=[sys.executable,'-m','pytest','-q','tests/test_ua2d_fpdecay_sg.py']
        else:command=[sys.executable,'-u','scripts/run_ua2d_fpdecay_sg.py',kind,*args]
        env=dict(CUDA_VISIBLE_DEVICES='3' if kind in ('pt','ft','test') else '',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        if kind=='pt':env['WORLD_SIZE']='1'
        actual=['timeout','--signal=TERM','--kill-after=10s',f'{seconds}s',*command]
        text=shlex.join(['env',*[f'{k}={v}' for k,v in env.items()],*actual])
        script=log/'run.sh';script.write_text('#!/bin/bash\ncd '+shlex.quote(str(ROOT))+' || exit 90\ndate -Iseconds > '+shlex.quote(str(log/'start.txt'))+'\n'+text+' > '+shlex.quote(str(log/'stdout.log'))+' 2> '+shlex.quote(str(log/'stderr.log'))+'\nua_fpdecay_sg_exit=$?\nprintf "%s\\n" "$ua_fpdecay_sg_exit" > '+shlex.quote(str(log/'exit.txt'))+'\ndate -Iseconds > '+shlex.quote(str(log/'end.txt'))+'\nexit "$ua_fpdecay_sg_exit"\n')
        j=dict(name=name,kind=kind,arm=arm,task=task,fold=fold,gpu=gpu,argv=actual,env=env,command=text,cwd=str(ROOT),log=str(log),
            consumption=str(unit_path(arm,task,fold,kind=='test')/'consumption.json') if kind in ('ft','test') else None,
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
        queue=[(a,t,f) for a in ARMS for t in TASKS for f in range(5)];active=[];kind='test' if formal else 'ft'
        while queue or active:
            self.deadline();self.refresh();self.check_failures();active=[j for j in active if j['exit'] is None]
            for gpu in (3,):
                while queue and sum(j['gpu']==gpu for j in active)<4:
                    a,t,f=queue.pop(0)
                    active.append(self.launch(f'fpdecay_sg_{kind}_{a}_{t}_{f}',kind,['--arm',a,'--task',t,'--fold',str(f)],gpu=gpu,arm=a,task=t,fold=f))
            if active:time.sleep(5)

    def stop_owned(self):
        def exits():
            for j in self.jobs:
                p=Path(j['log'])/'exit.txt'
                if j['exit'] is None and p.exists():j.update(exit=int(p.read_text().strip()),ended=now())
        exits();active=[j for j in self.jobs if j['exit'] is None and j.get('window')]
        owned=[]
        for j in self.jobs:
            if j['kind']=='pt':
                p=OUT/'pretrain'/j['arm']/'consumption.json'
                if p.exists():
                    r=read(p);owned.extend([r['process'],*r['workers']])
            elif j['kind'] in ('ft','test') and Path(j['consumption']).exists():
                r=read(j['consumption'])
                if r.get('process'):owned.append(r['process'])
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
        for identity in owned:
            if process_alive(identity):
                try:os.kill(identity['pid'],signal.SIGTERM)
                except ProcessLookupError:pass
        deadline=time.monotonic()+20
        while (any(j['exit'] is None for j in active) or any(process_alive(p) for p in owned)) and time.monotonic()<deadline:time.sleep(1);exits()
        if any(j['exit'] is None for j in active) or any(process_alive(p) for p in owned):raise RuntimeError('owned processes/true exits not safely closed; no CPU overlap')

    def close_pt_branch(self,arm):
        folder=OUT/'pretrain'/arm;c=read(folder/'consumption.json');m=read(folder/'metadata.json')
        if c.get('status')!='COMPLETE' or c.get('completed_updates')!=5000 or c.get('attempted_updates')!=5000 or c.get('consumed_presentations')!=5040000:raise ValueError('single-GPU PT incomplete')
        if any(process_alive(p) for p in [c['process'],*c['workers']]):raise RuntimeError('PT/worker still alive; phase overlap prohibited')
        if (m.get('status')!='COMPLETE' or m.get('step')!=5000 or m.get('plan')!=PLAN or m.get('revision')!=REVISION
            or m.get('arm')!=arm or m.get('world_size')!=1):raise ValueError('PT branch provenance mismatch')
        for name in ('deploy.pt','resume.pt','rng.pt','initial_training_rng.pt'):
            if not (folder/name).is_file():raise FileNotFoundError('PT source missing '+str(folder/name))
        write(OUT/'selection'/f'pt_complete_{arm}.json',dict(plan=PLAN,revision=REVISION,arm=arm,status='COMPLETE',true_exit=0,
            consumption=c,pt_worker_processes_exited=True,time=now()))

    def collect(self):
        self.partial={}
        for arm in ARMS:
            try:self.validation[arm]=aggregate(arm)
            except (OSError,ValueError,KeyError):
                completed={}
                for t in TASKS:
                    for f in range(5):
                        p=unit_path(arm,t,f)
                        if (p/'metrics.json').exists() and (p/'consumption.json').exists():
                            if read(p/'consumption.json').get('status')=='COMPLETE':completed[f'{t}/{f}']=read(p/'metrics.json')
                if completed:self.partial[arm]=dict(status='INCOMPLETE',complete_units=len(completed),metrics=completed)
        write(OUT/'selection/validation.json',dict(complete=self.validation,partial=self.partial))

    def deliver(self,error,cpu_exit):
        self.collect();b=self.refresh();final=self.frozen['FINAL'] if self.frozen else None
        complete=set(self.formal)==set(ARMS)
        selected=self.formal.get(final);delta=selected['macro']-self.baseline['macro'] if selected and complete else None
        status='待审查' if not error and cpu_exit==0 and complete and len(self.validation)==2 else '需返修'
        delta_base=selected['macro']-self.formal['BASE']['macro'] if selected and complete else None
        delta_p10k=selected['macro']-self.p10k['macro'] if selected and complete else None
        comparisons=registered_comparisons(self.formal,self.baseline,self.p10k)
        result=dict(plan=PLAN,revision=REVISION,status=status,error=error,independent_review='PENDING',validation=self.validation,partial_validation=self.partial,
            formal=self.formal,FINAL=final,complete_formal_TEST=complete,delta_FINAL_B_FP=delta,delta_FP_DECAY_BASE=delta_base,delta_FP_DECAY_P10K=delta_p10k,**outcome(delta_base,delta_p10k),
            comparisons=comparisons,references=dict(B_FP=self.baseline,P10K=self.p10k),cpu_exit=cpu_exit,budget=b,
            paper_envelope_macro=float(np.mean([v[2] for v in SOTA.values()])),
            interpretation='Only FP_DECAY-BASE isolates FP late coefficient with identical single-GPU training. P10K/B_FP cross-resource/budget references are not single-factor causal effects.',
            aggregation='five equal folds R2, std(ddof1), seed42+fold; seven equal task means',
            protocol='shared development TEST; paper data/split/protocol equivalence unproven; numerical comparison only; Egc3380/Xc432 vs paper4124/430')
        sota={};lines=['# UA2D-FPDECAY r2-SG 实际交付','',f'状态：{status}；DSH执行，待Codex独立审查。',
            f'FINAL={final or "未冻结"}；完整两支正式TEST={complete}；性能：{result["performance_status"]}；最终CPU exit={cpu_exit}。','',
            f'历史B_FP macro={self.baseline["macro"]:.3f}，PTSCALE/P10K={self.p10k["macro"]:.3f}，直接复用，不重测。',
            '主要受控比较为FP_DECAY−单卡BASE；晋级要求FP_DECAY相对BASE及P10K未舍入Δmacro均≥0.010。不能凭Nc/EPS单项改善晋级。',
            '只用物理GPU3/world1/micro168/acc6/global1008；两支各从相同公共step0独立训练5000步；两支结束且worker退出后70FT及一次性70正式TEST。',
            '固定候选FP_DECAY不依validation或TEST改选；BASE保持原FP权重；唯一科学变化为FP_DECAY的第4001—5000步原FP warmup权重乘固定余弦退火。',
            'prepare=0、公共初始化作业0；只读引用原PT缓存、PTSCALE logical映射及TRANSFER全部5875条FT缓存，保留原身份，不复制、重建或在线fallback。',
            '单卡按六个microbatch优化窗口目标计数归一化，不乘world2或再除acc6；每微批对比最多168。P10K及历史B_FP为跨资源／预算数值参考，不解释为单因素因果效应。','',
            '| 预登记正式对照 | Δmacro R² | 完整性 |','|---|---:|---|']
        for label,c in comparisons.items():
            d=f'{c["delta"]:+.3f}' if c['delta'] is not None else '未完成／不可比较';lines.append(f'| {label} | {d} | {c["status"]} |')
        for arm in ARMS:
            values=self.formal.get(arm);sota[arm]={}
            macro=f'{values["macro"]:.3f}' if values else '未完成／不可比较'
            lines+=['',f'## {arm} 七任务SOTA数值对照（macro={macro}）','','| Task | 正式R² | 参考R² | 项目减参考 | 参考模型 |','|---|---:|---:|---:|---|']
            for task,(name,model,ref) in SOTA.items():
                x=values['tasks'][task] if values else None;sota[arm][task]=dict(project=x,reference=ref,reference_model=model,delta=x['mean']-ref if x else None,status='COMPLETE' if x else '未完成／不可比较')
                score=f'{x["mean"]:.3f}±{x["std"]:.3f}' if x else '未完成／不可比较';d=f'{x["mean"]-ref:+.3f}' if x else '—'
                lines.append(f'| {name} | {score} | {ref:.3f} | {d} | {model} |')
        lines+=['','各任务五折等权R²，std(ddof=1)，seed42+fold，再七任务等权macro；所有计算使用未舍入值，展示3位小数。',
            '论文逐任务最佳包络macro约0.810；协议等价性未证明，仅作数值对照，不宣称公平超过论文。共享开发TEST不称独立盲测。',
            '配置、复用初始化来源、映射 COMPLETE、两支真实step5000 PT来源、history／best／scaler／预测位于原新周期根；冻结名单selection/final_frozen.json，完整精度指标及配对差值final_review/results.json；预算budget.json；源码diff及真实训练计时final_review/；全部命令／window／真实exit在logs/ua2d_fpdecay_20261022/sg_r2/。',
            f'未完成／异常原因：{error or ("无" if complete else "流程未完整结束")}。']
        result['sota']=sota;write(OUT/'final_review/results.json',result)
        times={}
        for arm in ARMS:
            for t in TASKS:
                for f in range(5):
                    p=unit_path(arm,t,f)/'epoch_timing.json'
                    if p.exists():times[f'{arm}/{t}/{f}']=read(p)
        pt_times={str(p.relative_to(OUT/'pretrain')):read(p) for p in (OUT/'pretrain').glob('*/*timing*.json')} if (OUT/'pretrain').exists() else {}
        write(OUT/'final_review/training_timing.json',dict(pt=pt_times,pt_window_logs=[str(p) for p in (OUT/'pretrain').glob('*/windows.jsonl')],epochs=times,jobs=[{k:j.get(k) for k in ('name','started','ended','exit','window','log')} for j in self.jobs],measurement='actual real training only; no benchmark or added CUDA synchronization'))
        report='\n'.join(lines)+'\n';(OUT/'final_review/report.md').write_text(report)
        update_plan(status,'执行交付',report);self.manifest.update(status=status,ended=now());write(OUT/'execution_manifest.json',self.manifest);self.capture_diff()
        return status

    def run(self):
        error=None;cpu_exit=None;safe=True
        try:
            references=cache_references();write(OUT/'authorization/cache_references.json',references)
            for arm in ARMS:
                update_plan('执行中',f'{arm} 单卡PT',f'controller {self.window}；仅GPU3/world1/micro168/acc6/global1008；两支串行各5000步，从相同step0初始化，无准备／新公共初始化作业。')
                j=self.launch(f'fpdecay_sg_pt_{arm}','pt',gpu=3,arm=arm);self.wait([j]);self.close_pt_branch(arm)
            update_plan('执行中','70个完整FT','两PT真实exit0且worker退出；两支各35FT，仅GPU3最多4路，严格复用PTSCALE ARCHITECTURE/PROFILE。')
            self.matrix();self.collect()
            self.frozen=freeze_final(self.validation)
            update_plan('执行中','唯一正式TEST',f'FINAL={self.frozen["FINAL"]}为预登记固定候选；两支70TEST名单及best／scaler／config／split／indices一次冻结，不依TEST改变候选。')
            self.matrix(True)
            for arm in ARMS:self.formal[arm]=aggregate(arm,True)
        except BaseException:
            error=traceback.format_exc();write(OUT/'failure.json',dict(error=error,time=now(),automatic_retry=False,automatic_resume=False));self.evidence_only=True
            try:self.stop_owned()
            except BaseException:error+='\n'+traceback.format_exc();safe=False
        finally:
            if self.frozen:
                for arm in ARMS:
                    try:self.formal[arm]=aggregate(arm,True)
                    except (OSError,ValueError,KeyError):pass
            if safe:
                try:
                    update_plan('执行中' if not error else '需返修','唯一最终CPU suite','实际训练／TEST已结束或安全停止后，仅本轮新专项tests/test_ua2d_fpdecay_sg.py一次，≤300秒，失败不重测。')
                    j=self.launch('fpdecay_sg_final_cpu','cpu',allow_final_cpu=True);self.wait([j]);cpu_exit=j['exit']
                except BaseException:error=(error or '')+'\n'+traceback.format_exc();cpu_exit=next((j['exit'] for j in self.jobs if j['kind']=='cpu'),None)
            status=self.deliver(error,cpu_exit);write(LOG/'controller/exit.json',dict(exit=0 if status=='待审查' else 1,status=status,error=error,time=now()))
        return 0 if status=='待审查' else 1


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['controller','ft','test']);p.add_argument('--arm',choices=ARMS);p.add_argument('--task',choices=TASKS);p.add_argument('--fold',type=int,choices=range(5));args=p.parse_args()
    torch.set_num_threads(2)
    def terminate(signum,frame):raise RuntimeError(f'controlled termination {signum}; no retry/resume')
    signal.signal(signal.SIGTERM,terminate)
    if args.action=='controller':return Controller().run()
    if args.arm is None or args.task is None or args.fold is None:p.error('complete fixed unit identity required')
    unit(args.arm,args.task,args.fold,args.action=='test');return 0

if __name__=='__main__':raise SystemExit(main())
