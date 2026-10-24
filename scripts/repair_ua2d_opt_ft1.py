#!/usr/bin/env python3
"""Explicit r1-FT-REPAIR1: eight fresh attempts once, then remaining finite cycle."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'2')
import copy
import datetime
import difflib
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_ua2d_opt as base
from src.training import ua2d_opt as training
from src.training.ua2d_opt import (OUT,LOG,PLAN,TASKS,B_ARMS,ARMS,TIE_ORDER,PARENT,SPLIT,CACHE,
    read,write,now,config,logical_unit,unit_path,aggregate,repair_attempt,REPAIR_FAILED_UNITS,
    conservative_ft_counts,c_decision,effective_d_policy,choose,factorial,final_decision,
    freeze_final,test_outcome,cd_formal_comparisons,update_plan,require_tmux)
REPAIR=OUT/'repair_ft1'
REVISION='r1-FT-REPAIR1'
CLOCK=1792541198.175118
LIMITS=dict(prepare_starts=1,initialization_starts=1,pt_starts=5,pt_updates=25000,pt_presentations=25200000,
    ft_starts=288,epochs_started=20160,validation_started=20160,attempted_updates=482090,
    test_starts=210,test_graphs=35250,cpu_suites=2)
FAILED_JOBS={f'opt_ft_{u.replace("/","_")}' for u in REPAIR_FAILED_UNITS}
COUNT_KEYS=('epochs_started','validation_started','attempted_updates','test_graphs')
SOTA={'eea':('Eea','Periodic-TDL',.934),'egb':('Egb','Periodic-TDL (+Chem, +RC)',.923),
    'egc':('Egc','Periodic-TDL (+Chem, +RC)',.911),'ei':('Ei','Periodic-TDL',.839),
    'eps':('EPS','Periodic-TDL',.787),'nc':('Nc','Periodic-TDL',.856),'xc':('Xc','Periodic-TDL',.423)}


def process_inventory():
    active=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:
            args=(p/'cmdline').read_bytes().split(b'\0')
            if (p/'cwd').resolve()!=ROOT:continue
            if any(x.endswith((b'scripts/run_ua2d_opt.py',b'scripts/pretrain_ua2d_opt.py',b'scripts/repair_ua2d_opt_ft1.py',b'/final_review/c_io1/deploy_and_report.py',b'/final_review/cd_required/handoff.py')) for x in args):
                active.append(dict(pid=int(p.name),argv=[x.decode() for x in args if x]))
        except OSError:pass
    return active


def unit_updates(task,fold):
    n=len(read(SPLIT/f'{task}.json')['folds'][fold]['train_indices'])
    return 70*((n+23)//24)


def validate_start(budget,manifest):
    if (budget['plan']!=PLAN or manifest['wallclock_started_unix']!=CLOCK or time.time()>=CLOCK+259200
        or budget['used']['ft_starts']!=197 or budget['used']['cpu_suites']!=1
        or budget['used']['pt_starts']!=4 or budget['used']['pt_updates']!=20000
        or budget['used']['pt_presentations']!=20160000 or budget['used']['test_starts']!=0):raise ValueError('repair starting budget/clock differs')
    jobs=budget['jobs'];names=[j['name'] for j in jobs]
    if len(names)!=len(set(names)) or {j['name'] for j in jobs if j['exit']!=0}!=FAILED_JOBS:
        raise ValueError('only exactly eight declared historical failures may be claimed')
    if sum(j['kind']=='ft' and j['exit']==0 for j in jobs)!=189:raise ValueError('historical successful count differs')
    for j in jobs:
        actual=int((Path(j['log'])/'exit.txt').read_text().strip())
        if actual!=j['exit']:raise ValueError('historical actual exit mismatch: '+j['name'])
        if j['kind']=='ft':
            if j['arm'] not in ('A_TOP2','A_MIX246',*B_ARMS):raise ValueError('later FT already started')
            c=read(j['consumption'])
            if j['exit']==0:
                cfg=read(Path(j['consumption']).parent/'config.json')
                if (c['status']!='COMPLETE' or c['completed_epochs']!=70
                    or c['attempted_updates']!=unit_updates(cfg['task'],cfg['fold'])
                    or c['completed_updates']!=c['attempted_updates']):raise ValueError('historical successful unit incomplete')
        if j['kind']=='pt' and (j['arm'] not in B_ARMS or any(r['status']!='COMPLETE' or r['completed_updates']!=5000 for r in j['rank_progress'])):
            raise ValueError('historical PT incomplete')
    if (OUT/'selection/final_frozen.json').exists():raise ValueError('formal TEST already frozen')
    if process_inventory():raise ValueError('another campaign process exists; do not take over')
    if read(OUT/'inputs/cache_manifest.json')['status']!='COMPLETE':raise ValueError('cache not COMPLETE')
    if effective_d_policy()!='MIX246':raise ValueError('required D effective policy mismatch')
    for arm in ('C_SCHED_E','C_SCHED_C','D_INTEGRATE'):
        if (OUT/'pretrain'/arm).exists() or (OUT/'finetune'/arm).exists():raise ValueError('later phase output exists')
    if (OUT/'formal').exists() and any((OUT/'formal').iterdir()):raise ValueError('formal outputs exist')
    if sum(unit_updates(t,int(f)) for _,t,f in (u.split('/') for u in REPAIR_FAILED_UNITS))!=34090:
        raise ValueError('declared failed-unit reservation differs from original split')


class RepairController(base.Controller):
    def __init__(self):
        require_tmux();self.ft_repair1=True;self.handoff=True
        self.lock=(OUT/'controller.lock').open('r+');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        historical=read(REPAIR/'snapshot/budget.json');manifest=read(OUT/'execution_manifest.json')
        if read(OUT/'budget.json')!=historical or manifest!=read(REPAIR/'snapshot/execution_manifest.json'):
            raise ValueError('another writer/recovery changed the original scene')
        validate_start(historical,manifest)
        if (REPAIR/'attempt_mapping.json').exists():raise FileExistsError('attempt mapping already exists; no repeated recovery')
        auth=read(REPAIR/'authorization.json')
        if auth['limits']!=LIMITS or set(auth['failed_jobs'])!=FAILED_JOBS:raise ValueError('repair authorization differs')
        self.window=subprocess.check_output(['tmux','display-message','-p','-t',os.environ['TMUX_PANE'],'#{window_id}'],text=True).strip()
        claim=dict(revision=REVISION,plan=PLAN,new_pid=os.getpid(),window=self.window,time=now(),clock=CLOCK,
            original_jobs=len(historical['jobs']),authorized_failures=sorted(FAILED_JOBS),automatic_retry=False,original_clock_reset=False)
        with (REPAIR/'claim.json').open('x') as f:json.dump(claim,f,ensure_ascii=False,indent=2)
        with (REPAIR/'effective_policy.json').open('x') as f:
            json.dump(dict(plan=PLAN,execution_policy_revision=REVISION,time=now(),authorization=str(REPAIR/'authorization.json'),
                C=dict(required=True,validation_gate=False),D=dict(required=True,FT_POLICY='MIX246'),
                historical_policy=str(OUT/'selection/a_policy.json'),prior_effective_policy=str(OUT/'selection/cd_required.json')),f,ensure_ascii=False,indent=2)
        self.jobs=copy.deepcopy(historical['jobs']);self.old_count=len(self.jobs);self.old_used=historical['used']
        self.started_unix=CLOCK;self.start=time.monotonic()-(time.time()-CLOCK)
        self.log_root=LOG/'repair_ft1';self.controller_log=self.log_root/'controller'
        self.limits=dict(LIMITS);base.LIMITS.update(self.limits)
        self.cached_budget=None;self.cached_state=None;self.validation={};self.formal={};self.partial={};self.allow_final_cpu=False
        self.mapping=dict(plan=PLAN,revision=REVISION,units={})
        lookup={j['name']:j for j in self.jobs}
        for arm in ('A_TOP2','A_MIX246',*B_ARMS,'D_INTEGRATE'):
            for task in TASKS:
                for fold in range(5):
                    key=logical_unit(arm,task,fold);oldname=f'opt_ft_{arm}_{task}_{fold}';old=lookup.get(oldname)
                    replacement=key in REPAIR_FAILED_UNITS
                    path=REPAIR/'attempts'/arm/task/f'fold{fold}'/'attempt1' if replacement else OUT/'finetune'/arm/task/f'fold{fold}'
                    self.mapping['units'][key]=dict(path=str(path),replacement=replacement,
                        job_name=f'opt_repair_ft1_{arm}_{task}_{fold}' if replacement else oldname,
                        historical_job=oldname if old else None,status='COMPLETE' if old and old['exit']==0 else 'PLANNED')
        write(REPAIR/'attempt_mapping.json',self.mapping)
        self.old_failed_counts={k:sum(j['progress'].get(k,0) for j in self.jobs if j['name'] in FAILED_JOBS) for k in COUNT_KEYS}
        self.failed_reserved=dict(epochs_started=560,validation_started=560,attempted_updates=34090,test_graphs=0)
        manifest.update(writer_pid=os.getpid(),writer='DSH',controller_window=self.window,command=sys.argv,
            execution_policy_revision=REVISION,repair_root=str(REPAIR),limits=self.limits,log=str(self.controller_log),status='执行中',phase='FT有限返修',
            controller_handoff=str(REPAIR/'claim.json'),automatic_retry=False,automatic_resume=False)
        write(OUT/'execution_manifest.json',manifest)
        self.refresh();self.capture_repair_diff();self.collect_validation();self.write_progress('执行中','已取得原唯一锁，旧189成功复用；指定8个fresh attempt及13未启动B3_C待派发')

    def refresh(self):
        if not hasattr(self,'old_count'):raise RuntimeError('repair accounting not initialized')
        used=dict(self.old_used)
        for k in COUNT_KEYS:used[k]=used.get(k,0)-self.old_failed_counts[k]+self.failed_reserved[k]
        new_known=dict.fromkeys(self.limits,0);new_charged=dict.fromkeys(self.limits,0);reserved=dict.fromkeys(self.limits,0);uncertainty=[]
        for j in self.jobs[self.old_count:]:
            start_key={'ft':'ft_starts','test':'test_starts','pt':'pt_starts','cpu':'cpu_suites'}[j['kind']]
            used[start_key]+=1;new_known[start_key]+=1;new_charged[start_key]+=1
            exitfile=Path(j['log'])/'exit.txt'
            if j['exit'] is None and exitfile.exists():j.update(exit=int(exitfile.read_text().strip()),ended=now())
            if j['kind'] in ('ft','test'):
                p=Path(j['consumption'])
                c=j.get('progress',{})
                if p.exists() and (j['exit'] is None or c.get('status')!='COMPLETE'):c=read(p);j['progress']=c
                counts=conservative_ft_counts(c)
                for k in COUNT_KEYS:
                    used[k]+=counts['charged'][k];new_charged[k]+=counts['charged'][k];new_known[k]+=counts['known_actual_lower_bound'][k]
                if counts['open_epoch']:uncertainty.append(dict(job=j['name'],**counts))
                if not c and j['exit'] is not None:
                    uncertainty.append(dict(job=j['name'],reason='exited without durable initialization record; full dispatch reservation retained',known_actual_lower_bound=0))
                    if j['kind']=='ft':
                        for k,v in j['full_unit_reservation'].items():used[k]+=v;new_charged[k]+=v
                if j['exit'] is None and j['kind']=='ft':
                    for k,v in j['full_unit_reservation'].items():reserved[k]+=max(0,v-counts['charged'].get(k,0))
                if j['kind']=='ft' and j['exit']==0:
                    if c.get('status')!='COMPLETE' or c.get('completed_epochs')!=70 or c.get('attempted_updates')!=j['full_unit_reservation']['attempted_updates'] or c.get('completed_updates')!=c.get('attempted_updates') or c.get('epoch_reservation',{}).get('open'):
                        if not getattr(self,'evidence_only',False):raise ValueError('successful exit lacks complete FT evidence')
                        uncertainty.append(dict(job=j['name'],reason='exit0 without complete FT evidence; never reused'))
                        continue
                    entry=self.mapping['units'][j['logical_unit']]
                    if entry['status']!='COMPLETE':
                        m=read(Path(entry['path'])/'metrics.json')
                        if m['status']!='COMPLETE' or m['config']!=config(j['arm'],m['config']['task'],m['config']['fold']):
                            if not getattr(self,'evidence_only',False):raise ValueError('successful FT config identity mismatch')
                            uncertainty.append(dict(job=j['name'],reason='FT config identity mismatch; never reused'));continue
                        entry.update(status='COMPLETE',exit=0,ended=j['ended']);write(REPAIR/'attempt_mapping.json',self.mapping)
                elif j['kind']=='ft' and j['exit'] is not None:
                    entry=self.mapping['units'][j['logical_unit']]
                    if entry['status']!='FAILED':entry.update(status='FAILED',exit=j['exit']);write(REPAIR/'attempt_mapping.json',self.mapping)
            elif j['kind']=='pt':
                ranks=[]
                for rank in (0,1):
                    p=OUT/'pretrain'/j['arm']/f'rank{rank}_consumption.json'
                    if p.exists():ranks.append(read(p))
                j['rank_progress']=ranks
                u=max((x['attempted_updates'] for x in ranks),default=0);n=sum(x['sample_presentations'] for x in ranks)
                for k,v in (('pt_updates',u),('pt_presentations',n)):
                    used[k]+=v;new_known[k]+=v;new_charged[k]+=v
                if j['exit'] is None:reserved['pt_updates']+=max(0,5000-u);reserved['pt_presentations']+=max(0,5040000-n)
        violations=[k for k in self.limits if used[k]+reserved[k]>self.limits[k]]
        if new_charged['ft_starts']>91 or new_charged['epochs_started']>6370 or new_charged['validation_started']>6370 or new_charged['attempted_updates']>156450 or new_charged['cpu_suites']>1:
            violations.append('new_finite_repair_allowance')
        if any(new_charged[k]+reserved[k]>v for k,v in (('epochs_started',6370),('validation_started',6370),('attempted_updates',156450))):violations.append('new_dispatch_reservation')
        payload=dict(plan=PLAN,revision=REVISION,limits=self.limits,used=used,jobs=self.jobs,
            old_used_recorded=self.old_used,old_failed_known_lower_bound=self.old_failed_counts,old_failed_full_unit_reservation=self.failed_reserved,
            old_failed_uncertain={k:self.failed_reserved[k]-self.old_failed_counts[k] for k in COUNT_KEYS},
            new_known_lower_bound=new_known,new_conservatively_charged=new_charged,active_unit_remaining_reserved=reserved,
            open_epoch_uncertainty=uncertainty,budget_violations=violations,unknown=[],accounting='old failed full unit charge REPLACES known lower bounds, never added twice; open epoch charged in full; immutable historical exits; no further retry')
        # Timestamps alone never cause a full-ledger write. Polling unchanged
        # epochs/jobs does not rewrite budget or timing.
        encoded=json.dumps(payload,sort_keys=True,allow_nan=False)
        if encoded!=self.cached_state:
            payload.update(time=now(),elapsed_seconds=time.monotonic()-self.start)
            write(OUT/'budget.json',payload);write(REPAIR/'budget.json',payload)
            write(OUT/'timing.json',dict(started_unix=CLOCK,elapsed_seconds=payload['elapsed_seconds'],limit_seconds=259200,time=payload['time']))
            self.cached_state=encoded;self.cached_budget=payload
        if violations and not getattr(self,'evidence_only',False):raise RuntimeError('budget exhausted: '+','.join(violations))
        return self.cached_budget

    def stop_owned(self):
        # Artifact/accounting exceptions must not prevent scoped shutdown of
        # this writer's own in-flight windows. This path updates real exits
        # only; it never alters history or dispatches a replacement.
        original=self.refresh
        def exits_only():
            for j in self.jobs[self.old_count:]:
                p=Path(j['log'])/'exit.txt'
                if j['exit'] is None and p.exists():j.update(exit=int(p.read_text().strip()),ended=now())
            return self.cached_budget
        self.refresh=exits_only
        try:super().stop_owned()
        finally:self.refresh=original

    def reserve_job(self,job,args):
        if job['kind']=='ft':
            task=args[args.index('--task')+1];fold=int(args[args.index('--fold')+1]);key=logical_unit(job['arm'],task,fold)
            entry=self.mapping['units'][key]
            if entry['job_name']!=job['name'] or entry['status']!='PLANNED':raise ValueError('attempt already started or wrong dispatch')
            job.update(logical_unit=key,full_unit_reservation=dict(epochs_started=70,validation_started=70,attempted_updates=unit_updates(task,fold)))
            entry.update(status='DISPATCHED',started=job['started']);write(REPAIR/'attempt_mapping.json',self.mapping)

    def launch(self,name,kind,args,**kwargs):
        if kind in ('prepare','initialize'):raise ValueError('preparation/initialization replay prohibited')
        self.refresh()
        if self.cached_budget.get('budget_violations'):raise RuntimeError('no further dispatch after budget violation')
        if not (kind=='cpu' and getattr(self,'allow_final_cpu',False)):self.check_new_failures()
        if kind=='ft':
            task=args[args.index('--task')+1];fold=int(args[args.index('--fold')+1]);arm=kwargs['arm'];entry=repair_attempt(arm,task,fold)
            if entry['status']!='PLANNED' or Path(entry['path']).exists():raise ValueError('attempt output conflict; no retry')
            counts=dict(epochs_started=70,validation_started=70,attempted_updates=unit_updates(task,fold))
            for k,v in counts.items():
                if self.cached_budget['used'][k]+self.cached_budget['active_unit_remaining_reserved'][k]+v>self.limits[k]:raise RuntimeError('insufficient full FT dispatch reservation')
        if kind=='pt' and (name not in ('opt_pt_C_SCHED_E','opt_pt_C_SCHED_C') or any(j['kind']=='pt' for j in self.jobs[self.old_count:])):
            raise ValueError('only one original C PT may start')
        if kind=='cpu' and name!='opt_repair_ft1_final_cpu':raise ValueError('one named final repair suite only')
        return super().launch(name,kind,args,**kwargs)

    def check_new_failures(self):
        failed=[j for j in self.jobs[self.old_count:] if j['exit'] not in (None,0)]
        if failed:raise RuntimeError('new job failed; no further retry: '+failed[0]['name']+' exit='+str(failed[0]['exit']))

    def matrix(self,arms,formal=False):
        self.refresh();self.check_new_failures()
        if any(j['exit'] is None for j in self.jobs):raise RuntimeError('stage overlap at repair matrix entry')
        queue=[];active=[]
        for arm in arms:
            for task in TASKS:
                for fold in range(5):
                    entry=repair_attempt(arm,task,fold,require_complete=formal)
                    if not formal and entry['status']=='COMPLETE':continue
                    if not formal and entry['status']!='PLANNED':raise ValueError('failed/started attempt must never be requeued')
                    queue.append((arm,task,fold,entry))
        while queue or active:
            self.deadline();self.refresh();self.check_new_failures();active=[j for j in active if j['exit'] is None]
            for gpu in (2,3):
                while queue and sum(j['gpu']==gpu for j in active)<4:
                    arm,task,fold,entry=queue.pop(0);kind='test' if formal else 'ft'
                    name=f'opt_test_{arm}_{task}_{fold}' if formal else entry['job_name']
                    active.append(self.launch(name,kind,[kind,'--arm',arm,'--task',task,'--fold',str(fold)],gpu=gpu,arm=arm,consumption=unit_path(arm,task,fold,formal)/'consumption.json'))
            if active:time.sleep(5)

    def pt(self,arm):
        from dataclasses import asdict
        from src.modules.ua2d_galformer_pretrain import PretrainConfig
        path=OUT/'configs'/f'{arm}.json'
        if path.exists() or (OUT/'pretrain'/arm).exists():raise FileExistsError('C PT output/config conflict; no restart')
        write(path,asdict(PretrainConfig()))
        j=self.launch('opt_pt_'+arm,'pt',['--arm',arm,'--config',str(path),'--output',str(OUT/'pretrain'/arm),
            '--initialization',str(OUT/'initialization'/f'{arm[-1]}_step0.pt'),'--prepared-cache',str(CACHE)],arm=arm)
        self.wait([j])
        if len(j['rank_progress'])!=2 or any(x['status']!='COMPLETE' or x['completed_updates']!=5000 for x in j['rank_progress']):raise ValueError('C PT incomplete')
        meta=read(OUT/'pretrain'/arm/'config.json');eng=meta['engineering']
        if (eng.get('io_plan')!='UA2D-OPT-20261021-01/r1-C-IO1' or eng.get('read_policy')!='b3c_io1_readahead'
            or eng.get('io_scope')!=arm+'_PT_only' or eng.get('lmdb_readahead') is not True
            or meta.get('schedule')!=[1,1,2,2,3,3]):raise ValueError('C actual IO/schedule identity mismatch')
        write(REPAIR/'c_io1_actual_load.json',dict(engineering=eng,arm=arm,config=str(OUT/'pretrain'/arm/'config.json'),timing=read(OUT/'pretrain'/arm/'timing.json')))
        self.matrix([arm])

    def add_C_mapping(self,arm):
        if any(k.startswith('C_') for k in self.mapping['units']):raise ValueError('C already selected; no second target')
        for task in TASKS:
            for fold in range(5):
                self.mapping['units'][logical_unit(arm,task,fold)]=dict(path=str(OUT/'finetune'/arm/task/f'fold{fold}'),replacement=False,
                    job_name=f'opt_ft_{arm}_{task}_{fold}',historical_job=None,status='PLANNED')
        write(REPAIR/'attempt_mapping.json',self.mapping)

    def collect_validation(self):
        self.partial={}
        for arm in ARMS:
            entries={k:v for k,v in self.mapping['units'].items() if k.startswith(arm+'/')}
            if not entries:continue
            done={k:v for k,v in entries.items() if v['status']=='COMPLETE'}
            if len(done)==35:self.validation[arm]=aggregate(arm)
            elif done:
                self.partial[arm]=dict(status='INCOMPLETE',complete_units=len(done),required_units=35,
                    units={k:dict(metrics=read(Path(v['path'])/'metrics.json'),attempt=v) for k,v in done.items()})
        write(REPAIR/'validation.json',dict(complete=self.validation,partial=self.partial,time=now()))

    def write_progress(self,status,detail):
        write(REPAIR/'status.json',dict(plan=PLAN,revision=REVISION,status=status,time=now(),detail=detail,
            pid=os.getpid(),window=self.window,independent_review='PENDING'))
        update_plan(status,REVISION,detail+'；当前交付指向repair_ft1/，原失败交付及原始exit保留。DSH自检不替代Codex独立审查。')

    def capture_repair_diff(self):
        patches=[]
        for rel in ('src/training/ua2d_opt.py','scripts/run_ua2d_opt.py','tests/test_ua2d_opt.py','scripts/repair_ua2d_opt_ft1.py'):
            before=REPAIR/'before'/rel;old=before.read_text() if before.exists() else ''
            patches.extend(difflib.unified_diff(old.splitlines(True),(ROOT/rel).read_text().splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
        (REPAIR/'actual_diff.patch').write_text(''.join(patches))

    def deliver_repair(self,decision,conditional,error,cpu_exit):
        try:self.collect_validation()
        except BaseException:
            error=(error or '')+'\nvalidation证据汇总异常：'+traceback.format_exc()
            write(REPAIR/'validation.json',dict(complete=self.validation,partial=self.partial,error=error))
        budget=self.refresh();old=read(PARENT.parents[2]/'final_review/results.json')
        comparisons=cd_formal_comparisons(self.formal,old['test']['B_FP'],conditional)
        final=decision.get('FINAL') if decision else None;selected=self.formal.get(final);delta=selected['macro']-old['test']['B_FP']['macro'] if selected else None
        paired_final={}
        if selected:
            for t in TASKS:
                values=training.np.asarray(selected['tasks'][t]['folds'])-training.np.asarray(old['test']['B_FP']['tasks'][t]['folds'])
                paired_final[t]=dict(folds=values.tolist(),mean=float(values.mean()),std=float(values.std(ddof=1)))
        comparisons['FINAL_vs_B_FP']=dict(arm=final,reference='B_FP',delta=delta,paired_tasks=paired_final,**test_outcome(delta))
        complete=bool(decision and set(self.formal)==set(decision['models']) and sum(v['status']=='COMPLETE' for v in self.mapping['units'].values())==280)
        status='待审查' if error is None and cpu_exit==0 and complete else '需返修'
        sota={}
        for arm in dict.fromkeys([final,conditional.get('C',{}).get('arm'),'D_INTEGRATE']):
            if arm is None:continue
            values=self.formal.get(arm);sota[arm]={}
            for t,(label,model,ref) in SOTA.items():
                v=values['tasks'][t] if values else None
                sota[arm][t]=dict(project=v,reference=ref,reference_model=model,delta=v['mean']-ref if v else None,status='COMPLETE' if v else '未完成／不可比较')
        result=dict(plan=PLAN,revision=REVISION,status=status,independent_review='PENDING',error=error,
            validation=self.validation,partial_validation=self.partial,formal=self.formal,decision=decision,
            conditional=conditional,comparisons=comparisons,baseline_B_FP=old['test']['B_FP'],final_delta_B_FP=delta,**test_outcome(delta),
            cpu_exit=cpu_exit,complete_formal_TEST=complete,budget=budget,sota=sota,
            aggregation='five equal folds R2, std(ddof=1), seed42+fold, then seven equal task means',
            protocol='shared development folds; paper protocol equivalence not proven; numeric comparison only',
            sources=dict(attempt_mapping=str(REPAIR/'attempt_mapping.json'),historical_failure_report=str(REPAIR/'snapshot/final_review/results.json')))
        write(REPAIR/'results.json',result)
        configs={k:dict(path=v['path'],config=str(Path(v['path'])/'config.json'),best=str(Path(v['path'])/'best.pt'),scaler=str(Path(v['path'])/'scaler.json'),history=str(Path(v['path'])/'history.json'),status=v['status']) for k,v in self.mapping['units'].items()}
        write(REPAIR/'artifact_index.json',configs)
        write(REPAIR/'configurations.json',{k:read(Path(v['path'])/'config.json') for k,v in self.mapping['units'].items() if (Path(v['path'])/'config.json').exists()})
        timings={k:read(Path(v['path'])/'epoch_timing.json') for k,v in self.mapping['units'].items() if (Path(v['path'])/'epoch_timing.json').exists()}
        write(REPAIR/'training_timing.json',dict(FT_epochs=timings,PT={p.parent.name:read(p) for p in (OUT/'pretrain').glob('*/timing.json')},measurement='actual training only; data_seconds includes denominator communication, not pure IO; no benchmark or added CUDA sync'))
        if (OUT/'selection/final_frozen.json').exists():write(REPAIR/'final_frozen.json',read(OUT/'selection/final_frozen.json'))
        lines=['# r1-FT-REPAIR1 实际交付','',f'状态：{status}；DSH执行，待Codex独立审查。',
            f'FINAL：{final or "未冻结"}；完整正式TEST：{complete}；CPU exit={cpu_exit}；正式性能：{result["performance_status"]}。','',
            '旧197个FT job／189个成功结果／8个失败及其真实exit原样保留，8个指定失败仅各一次独立fresh attempt，从PT step5000及原公共head完整70epochs。',
            '原8失败按560epochs／560validations／34090updates替代已知下界保守计账，不重复相加；原3份RUNNING残留在snapshot/incidents.json标明已中断，原记录未改。',
            '完整配置、checkpoint、scaler、history、预测及实际成功attempt见attempt_mapping.json和artifact_index.json；命令、window和真实exit见budget.json及logs/ua2d_opt_20261021/r1/repair_ft1/。',
            '旧事故快照与报告见snapshot/；实际源码差异见actual_diff.patch；本次CPU只在收尾执行一次，不重测。',
            f'完整validation臂：{", ".join(self.validation) or "无"}；部分结果另列validation.json，不以空表抹去已完成结果。','',
            '| 正式比较 | Δmacro R² | 判读 |','|---|---:|---|']
        for name,c in comparisons.items():
            d=f'{c["delta"]:+.3f}' if c['delta'] is not None else '未完成／不可比较'
            lines.append(f'| {name} | {d} | {c["performance_status"]} |')
        lines+=['',f'B_FP既有正式TEST macro R²={old["test"]["B_FP"]["macro"]:.3f}；不重测，差值使用完整精度。']
        for arm,values in sota.items():
            lines+=['',f'## {arm} 七任务SOTA数值对照','','| Task | 正式R² | 参考R² | 项目减参考 | 参考模型 |','|---|---:|---:|---:|---|']
            for t,v in values.items():
                x=v['project'];score=f'{x["mean"]:.3f}±{x["std"]:.3f}' if x else '未完成／不可比较';d=f'{v["delta"]:+.3f}' if x else '—'
                lines.append(f'| {SOTA[t][0]} | {score} | {v["reference"]:.3f} | {d} | {v["reference_model"]} |')
        if not sota:
            lines+=['','## 正式TEST未完成：七任务均不可比较','','| Task | 正式R² | 参考R² | 项目减参考 | 参考模型 |','|---|---:|---:|---:|---|']
            for _,(label,model,ref) in SOTA.items():lines.append(f'| {label} | 未完成／不可比较 | {ref:.3f} | — | {model} |')
        lines+=['','口径：各任务五折等权R²，std(ddof=1)，seed42+fold；七任务等权macro，原始数值完整精度，展示3位小数。',
            '协议等价性未证明，仅作论文数值对照；共享TEST不称独立盲测。完整B四臂未正式评估时不宣称2×2因子TEST结论。',
            f'未完成／阻断原因：{error or ("无" if complete else "正式流程尚未完整完成")}。']
        report='\n'.join(lines)+'\n';(REPAIR/'report.md').write_text(report)
        self.capture_repair_diff();self.write_progress(status,report)
        manifest=read(OUT/'execution_manifest.json');manifest.update(status=status,phase='返修交付',ended=now());write(OUT/'execution_manifest.json',manifest)
        return status

    def run(self):
        error=None;decision=None;conditional={};cpu_exit=None;safe=True
        try:
            self.write_progress('执行中','B3_C：仅指定8个fresh attempts＋13个从未启动FT，复用其余189成功单元')
            self.matrix(['B3_C']);self.collect_validation()
            write(REPAIR/'b_factorial.json',dict(validation={a:self.validation[a] for a in B_ARMS},effects=factorial(self.validation)))
            c=c_decision(self.validation);self.add_C_mapping(c['arm'])
            conditional=dict(C=c,D=dict(trigger=True,required=True,FT_POLICY='MIX246'),PT_WINNER=None,execution_policy_revision=REVISION)
            write(OUT/'selection/conditional_decisions.json',conditional);write(REPAIR/'conditional_decisions.json',conditional)
            self.write_progress('执行中',f'唯一{c["arm"]}：无validation增量门槛，原step0、schedule112233、5000updates＋35FT；现有C-IO1预读计时')
            self.pt(c['arm']);self.collect_validation()
            ptwinner=choose({a:self.validation[a]['macro'] for a in (*B_ARMS,c['arm'])},TIE_ORDER)
            conditional['PT_WINNER']=ptwinner;write(OUT/'selection/conditional_decisions.json',conditional);write(REPAIR/'conditional_decisions.json',conditional)
            self.write_progress('执行中',f'D_INTEGRATE：parent={ptwinner} PT step5000＋原公共head，MIX246标量从0完整35FT')
            self.matrix(['D_INTEGRATE']);self.collect_validation()
            if len(self.mapping['units'])!=280 or any(v['status']!='COMPLETE' for v in self.mapping['units'].values()):raise ValueError('280 logical FT not complete; no freeze')
            baseline=read(PARENT.parents[2]/'final_review/results.json')['validation']['B_FP']['macro']
            decision=final_decision(self.validation,baseline,ptwinner);write(REPAIR/'development_decision.json',decision)
            freeze_final(self.validation,decision,conditional);write(REPAIR/'final_frozen.json',read(OUT/'selection/final_frozen.json'))
            self.write_progress('执行中',f'唯一正式TEST：FINAL={decision["FINAL"]}，一次预冻结{decision["models"]}，actual attempt／best／scaler／config／indices固定')
            self.matrix(decision['models'],True)
            for a in decision['models']:self.formal[a]=aggregate(a,True)
            if all(a in self.formal for a in B_ARMS):write(REPAIR/'formal_factorial.json',factorial(self.formal))
        except BaseException:
            error=traceback.format_exc();write(REPAIR/'failure.json',dict(time=now(),error=error,automatic_retry=False,automatic_resume=False))
            self.evidence_only=True
            try:self.stop_owned()
            except BaseException:error+='\n'+traceback.format_exc();safe=False
        finally:
            # Recover all fully finished formal arms on failure without rerunning
            # any predictions; incomplete tasks remain explicitly incomparable.
            if decision:
                for a in decision['models']:
                    try:self.formal[a]=aggregate(a,True)
                    except (OSError,ValueError,KeyError):pass
            if safe:
                try:
                    self.write_progress('执行中' if not error else '需返修','全部实际训练／TEST已结束或安全停止；新增唯一最终CPU suite，≤300秒，失败不重测')
                    # A failed new training job must not prevent the explicitly
                    # authorized final CPU evidence after safe shutdown.
                    self.allow_final_cpu=True
                    j=self.launch('opt_repair_ft1_final_cpu','cpu',[]);self.wait([j]);cpu_exit=j['exit']
                except BaseException:error=(error or '')+'\n'+traceback.format_exc();cpu_exit=next((j['exit'] for j in self.jobs[self.old_count:] if j['kind']=='cpu'),None)
            status=self.deliver_repair(decision,conditional,error,cpu_exit)
            write(self.controller_log/'exit.json',dict(exit=0 if status=='待审查' else 1,status=status,error=error,time=now()))
        return 0 if status=='待审查' else 1


def main():
    base.torch.set_num_threads(2)
    def terminate(signum,frame):raise RuntimeError(f'controlled termination {signum}; no further retry')
    signal.signal(signal.SIGTERM,terminate)
    try:return RepairController().run()
    except BaseException:
        error=traceback.format_exc();write(REPAIR/'startup_failure.json',dict(status='阻断',time=now(),error=error,automatic_takeover_retry=False))
        raise

if __name__=='__main__':raise SystemExit(main())
