#!/usr/bin/env python3
"""One isolated finite PT scaling controller; never resumes or imports old ledgers."""
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
from src.training.ua2d_ptscale import *
SOURCES=('src/dataset/ua2d_ptscale.py','src/training/ua2d_ptscale.py','scripts/prepare_ua2d_ptscale.py','scripts/pretrain_ua2d_ptscale.py','scripts/run_ua2d_ptscale.py','tests/test_ua2d_ptscale.py')


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
            only_gpus=[2,3],concurrency_per_gpu=4,wallclock_limit_seconds=86400,wallclock_started_unix=None,
            automatic_retry=False,automatic_resume=False,independent_review='PENDING',log=str(LOG/'controller'))
        with (OUT/'execution_manifest.json').open('x') as f:json.dump(self.manifest,f,ensure_ascii=False,indent=2)
        self.baseline=read(REFERENCE_BFP)['test']['B_FP']
        if set(self.baseline['tasks'])!=set(TASKS) or any(len(self.baseline['tasks'][t]['folds'])!=5 for t in TASKS):raise ValueError('incomplete B_FP formal reference')
        finite([self.baseline['macro'],*[x for t in TASKS for x in self.baseline['tasks'][t]['folds']]])
        write(OUT/'authorization/reference_results.json',dict(B_FP=dict(source=str(REFERENCE_BFP),result=self.baseline),
            reevaluated=False,old_pool=911391,new_pool=910756,interpretation='pool differs; not a single duration contrast'))
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
            key={'prepare':'prepare_starts','pt':'pt_starts','ft':'ft_starts','test':'test_starts','cpu':'cpu_suites'}[j['kind']]
            used[key]+=1;known[key]+=1
            p=Path(j['log'])/'exit.txt'
            if j['exit'] is None and p.exists():j.update(exit=int(p.read_text().strip()),ended=now())
            if j['kind']=='prepare':
                p=OUT/'inputs/metadata.json'
                if p.exists():j['progress']=read(p)
            elif j['kind']=='pt':
                ranks=[]
                for rank in range(2):
                    p=OUT/'pretrain'/f'rank{rank}_consumption.json'
                    if p.exists():ranks.append(read(p))
                attempted=max([r['attempted_updates'] for r in ranks],default=0)
                completed=min([r['completed_updates'] for r in ranks],default=0) if len(ranks)==2 else 0
                consumed=sum(r['consumed_presentations'] for r in ranks)
                charged_updates=attempted
                if j['exit'] not in (None,0) and not ranks:charged_updates=j['reservation']['pt_updates']
                used['pt_updates']+=charged_updates;used['pt_presentations']+=charged_updates*1008
                known['pt_updates']+=attempted;known['pt_presentations']+=consumed
                j['progress']=dict(ranks=ranks,attempted_updates=attempted,completed_updates=completed,
                    consumed_presentations=consumed,charged_presentations=charged_updates*1008,
                    read_prefetched_presentations=sum(r['sample_presentations'] for r in ranks),
                    prefetched_not_consumed=sum(r['prefetched_not_consumed'] for r in ranks))
                if charged_updates*1008!=consumed:uncertain.append(dict(job=j['name'],charged_presentations=charged_updates*1008,known_consumed=consumed,reason='in-progress/failed update full presentation reservation, not added to read/prefetch'))
                if j['exit'] is None:
                    remaining['pt_updates']+=max(0,20000-charged_updates);remaining['pt_presentations']+=max(0,20160000-charged_updates*1008)
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
        if kind not in ('prepare','pt','ft','test','cpu'):raise ValueError('extra phases prohibited')
        if any(j['name']==name for j in self.jobs):raise ValueError('duplicate launch; no retry')
        key={'prepare':'prepare_starts','pt':'pt_starts','ft':'ft_starts','test':'test_starts','cpu':'cpu_suites'}[kind]
        if b['used'][key]>=LIMITS[key]:raise RuntimeError('start allowance exhausted')
        active=[j for j in self.jobs if j['exit'] is None]
        if any(j['kind']!=kind for j in active):raise ValueError('stage overlap prohibited')
        if kind in ('ft','test'):
            if gpu not in (2,3) or sum(j['gpu']==gpu for j in active)>=4:raise ValueError('GPU/concurrency violation')
            if unit_path(arm,task,fold,kind=='test').exists():raise FileExistsError('output conflict')
        if kind=='ft':
            cache_metadata(CACHE_ROOT);reservation=dict(epochs_started=70,validation_started=70,attempted_updates=self.updates(task,fold))
        elif kind=='pt':
            if (OUT/'pretrain').exists():raise FileExistsError('PT output conflict')
            m=read(OUT/'inputs/metadata.json')
            if m.get('status')!='COMPLETE' or m.get('plan')!=PLAN or m.get('records')!=910756:raise ValueError('PT mapping not COMPLETE')
            reservation=dict(pt_updates=20000,pt_presentations=20160000)
        elif kind=='test':
            frozen=read(OUT/'selection/final_frozen.json')
            if not frozen['frozen'] or frozen['models']!=list(ARMS) or arm not in frozen['models']:raise ValueError('TEST outside fixed freeze')
            reservation=dict(test_graphs=len(frozen['units'][f'{arm}/{task}/{fold}']['test_indices']))
        else:reservation={}
        for k,v in reservation.items():
            if b['used'][k]+b['active_remaining_reservation'][k]+v>LIMITS[k]:raise RuntimeError('insufficient full-unit reservation')
        if kind=='prepare':self.begin_clock()
        log=LOG/name;log.mkdir(exist_ok=False)
        seconds={'prepare':3600,'pt':43200,'ft':7200,'test':300,'cpu':300}[kind]
        seconds=min(seconds,max(1,int(86400-(time.monotonic()-self.start))))
        if kind=='prepare':command=[sys.executable,'-u','scripts/prepare_ua2d_ptscale.py']
        elif kind=='pt':command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','scripts/pretrain_ua2d_ptscale.py',
            '--config',str(OUT/'inputs/pt_config.json'),'--output',str(OUT/'pretrain'),'--initialization',str(INITIALIZATION),
            '--prepared-cache',str(PT_CACHE),'--mapping',str(OUT/'inputs')]
        elif kind=='cpu':command=[sys.executable,'-m','pytest','-q','tests/test_ua2d_ptscale.py']
        else:command=[sys.executable,'-u','scripts/run_ua2d_ptscale.py',kind,*args]
        env=dict(CUDA_VISIBLE_DEVICES='2,3' if kind=='pt' else str(gpu) if gpu is not None else '',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        if kind=='pt':env['UA2D_PTSCALE_CAMPAIGN']='r1'
        actual=['timeout','--signal=TERM','--kill-after=10s',f'{seconds}s',*command]
        text=shlex.join(['env',*[f'{k}={v}' for k,v in env.items()],*actual])
        script=log/'run.sh';script.write_text('#!/bin/bash\ncd '+shlex.quote(str(ROOT))+' || exit 90\ndate -Iseconds > '+shlex.quote(str(log/'start.txt'))+'\n'+text+' > '+shlex.quote(str(log/'stdout.log'))+' 2> '+shlex.quote(str(log/'stderr.log'))+'\nua_ptscale_exit=$?\nprintf "%s\\n" "$ua_ptscale_exit" > '+shlex.quote(str(log/'exit.txt'))+'\ndate -Iseconds > '+shlex.quote(str(log/'end.txt'))+'\nexit "$ua_ptscale_exit"\n')
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
            for gpu in (2,3):
                while queue and sum(j['gpu']==gpu for j in active)<4:
                    a,t,f=queue.pop(0)
                    active.append(self.launch(f'ptscale_{kind}_{a}_{t}_{f}',kind,['--arm',a,'--task',t,'--fold',str(f)],gpu=gpu,arm=a,task=t,fold=f))
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
                for rank in range(2):
                    p=OUT/'pretrain'/f'rank{rank}_consumption.json'
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

    def close_pt_stage(self):
        ranks=[read(OUT/'pretrain'/f'rank{r}_consumption.json') for r in range(2)]
        if any(r['status']!='COMPLETE' or r['completed_updates']!=20000 or r['attempted_updates']!=20000 or r['consumed_presentations']!=10080000 for r in ranks):raise ValueError('PT not fully complete')
        processes=[p for r in ranks for p in [r['process'],*r['workers']]]
        if any(process_alive(p) for p in processes):raise RuntimeError('PT rank/worker remains alive; no FT overlap')
        for step in STEPS.values():
            folder=OUT/'pretrain/checkpoints'/f'step{step}';m=read(folder/'metadata.json')
            if m.get('status')!='COMPLETE' or m.get('step')!=step or m.get('plan')!=PLAN:raise ValueError('PT snapshot provenance incomplete')
            for name in ('deploy.pt','resume.pt','rank0_rng.pt','rank1_rng.pt'):
                if not (folder/name).is_file():raise FileNotFoundError('PT source missing '+str(folder/name))
        write(OUT/'selection/pt_complete.json',dict(plan=PLAN,status='COMPLETE',true_exit=0,steps=STEPS,ranks=ranks,rank_worker_processes_exited=True,time=now()))

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
        status='待审查' if not error and cpu_exit==0 and complete and len(self.validation)==3 else '需返修'
        delta_p5k=selected['macro']-self.formal['P5K']['macro'] if selected and complete else None
        comparisons=registered_comparisons(self.formal,self.baseline,final)
        result=dict(plan=PLAN,revision=REVISION,status=status,error=error,independent_review='PENDING',validation=self.validation,partial_validation=self.partial,
            formal=self.formal,FINAL=final,complete_formal_TEST=complete,delta_FINAL_B_FP=delta,delta_FINAL_P5K=delta_p5k,**outcome(delta,delta_p5k),
            comparisons=comparisons,references=dict(B_FP=self.baseline),cpu_exit=cpu_exit,budget=b,
            paper_envelope_macro=float(np.mean([v[2] for v in SOTA.values()])),
            interpretation='P5K/P10K/P20K share one PT trajectory. New pool910756 vs historical911391. Extra cycles include exposure plus fixed cyclic optimization, not separate mechanisms.',
            aggregation='five equal folds R2, std(ddof1), seed42+fold; seven equal task means',
            protocol='shared development TEST; paper data/split/protocol equivalence unproven; numerical comparison only; Egc3380/Xc432 vs paper4124/430')
        sota={};lines=['# UA2D-PTSCALE r1 实际交付','',f'状态：{status}；DSH执行，待Codex独立审查。',
            f'FINAL={final or "未冻结"}；完整三检查点正式TEST={complete}；性能：{result["performance_status"]}；最终CPU exit={cpu_exit}。','',
            f'历史B_FP正式macro={self.baseline["macro"]:.3f}直接复用，不重测。FINAL−P5K={format(delta_p5k,"+.3f") if delta_p5k is not None else "未完成／不可比较"}，双参考均须未舍入≥0.010。',
            '一个step0→20000连续PT轨迹，真实5k/10k/20k三个检查点，不是独立预训练；一个logical slot映射准备，不重建缓存，新公共初始化0。105FT及一次性105正式TEST。',
            'P5K为同条件主要基线；历史B_FP原池911391与新池910756不同，其差值不单归因于时长；额外周期同时包含曝光及循环优化。未通过不追加PT或目标搜索。','',
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
            '配置、复用初始化来源、映射 COMPLETE、真实step三份PT来源、history／best／scaler／预测位于原新周期根；冻结名单selection/final_frozen.json，完整精度指标及配对差值final_review/results.json；预算budget.json；源码diff及真实训练计时final_review/；全部命令／window／真实exit在logs/ua2d_ptscale_20261022/r1/。',
            f'未完成／异常原因：{error or ("无" if complete else "流程未完整结束")}。']
        result['sota']=sota;write(OUT/'final_review/results.json',result)
        times={}
        for arm in ARMS:
            for t in TASKS:
                for f in range(5):
                    p=unit_path(arm,t,f)/'epoch_timing.json'
                    if p.exists():times[f'{arm}/{t}/{f}']=read(p)
        pt_times={p.name:read(p) for p in (OUT/'pretrain').glob('*timing*.json')} if (OUT/'pretrain').exists() else {}
        write(OUT/'final_review/training_timing.json',dict(pt=pt_times,pt_window_logs=[str(p) for p in (OUT/'pretrain').glob('*windows.jsonl')],epochs=times,jobs=[{k:j.get(k) for k in ('name','started','ended','exit','window','log')} for j in self.jobs],measurement='actual real training only; no benchmark or added CUDA synchronization'))
        report='\n'.join(lines)+'\n';(OUT/'final_review/report.md').write_text(report)
        update_plan(status,'执行交付',report);self.manifest.update(status=status,ended=now());write(OUT/'execution_manifest.json',self.manifest);self.capture_diff()
        return status

    def run(self):
        error=None;cpu_exit=None;safe=True
        try:
            update_plan('执行中','唯一logical映射准备',f'controller {self.window}；首次prepare计24小时，物化910756个原logical slots及key/source映射，LMDB/FT缓存只读。')
            j=self.launch('ptscale_prepare','prepare');self.wait([j])
            update_plan('执行中','连续20000步PT','唯一world2 GPU2/3，5000步循环LR但全局warmup／RNG／optimizer／采样连续，实际5k/10k/20k保存来源。')
            j=self.launch('ptscale_pt_20k','pt');self.wait([j]);self.close_pt_stage()
            update_plan('执行中','105个完整FT','PT真实exit0且rank/worker已退出；P5K/P10K/P20K全部35FT，原70epochs/BEST配方，无validation准入门槛。')
            self.matrix();self.collect()
            self.frozen=freeze_final(self.validation)
            update_plan('执行中','唯一正式TEST',f'FINAL={self.frozen["FINAL"]}仅按完整validation提名；三检查点105TEST名单及best／scaler／config／split／indices一次冻结，不依TEST改变候选。')
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
                    update_plan('执行中' if not error else '需返修','唯一最终CPU suite','实际训练／TEST已结束或安全停止后，仅本轮新专项tests/test_ua2d_ptscale.py一次，≤300秒，失败不重测。')
                    j=self.launch('ptscale_final_cpu','cpu',allow_final_cpu=True);self.wait([j]);cpu_exit=j['exit']
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
