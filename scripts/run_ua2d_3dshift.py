#!/usr/bin/env python3
"""One isolated finite 3DSHIFT controller; carries historical consumption without retry or automatic resume."""
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
from src.training.ua2d_3dshift import *
from src.training.ua2d_fpdecay_sg import process_alive
SOURCES=('src/modules/ua2d_3dshift.py','src/dataset/ua2d_3dshift.py','src/training/ua2d_3dshift.py','scripts/prepare_ua2d_3dshift.py','scripts/pretrain_ua2d_3dshift.py','scripts/run_ua2d_3dshift.py','tests/test_ua2d_3dshift.py','configs/ua2d_3dshift_r1.json','src/modules/ua2d_galformer_pretrain.py','configs/ua2d_galformer_mask_align.json','scripts/pretrain_ua2d_fpdecay_sg.py','scripts/pretrain_ua2d_ptscale.py')


class Controller:
    def __init__(self):
        require_tmux()
        authorization=OUT/'authorization/user_authorization.json'
        if not authorization.is_file() or read(authorization).get('experiment_authorized') is not True:
            raise PermissionError('implementation-only phase: separate explicit run authorization required')
        replacing=(OUT/'execution_manifest.json').exists()
        if replacing:
            permission=read(OUT/'authorization/explicit_startup_replacement.json')
            previous=read(OUT/'execution_manifest.json')
            if (permission.get('explicit_single_controller_startup_replacement') is not True
                or permission.get('previous_training_starts')!=0 or permission.get('previous_cpu_starts')!=0
                or int(Path(permission['previous_exit_path']).read_text().strip())!=1
                or process_alive(previous['writer_process'])
                or (OUT/'budget.json').exists() or (OUT/'pretrain').exists()):
                raise PermissionError('startup replacement requires explicit user exception, dead prior controller and zero training')
            with (OUT/'authorization/startup_replacement_claim.json').open('x') as f:
                json.dump(dict(time=now(),process=process_identity(os.getpid()),authorization=permission),f,ensure_ascii=False,indent=2)
        self.lock=(OUT/'controller.lock').open('r+' if replacing else 'x');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'budget.json').exists():raise FileExistsError('training ledger exists; no retry/resume')
        self.auth=read(OUT/'authorization/user_authorization.json')
        if self.auth['plan']!=PLAN or self.auth['revision']!=REVISION or self.auth['limits']!=LIMITS:raise ValueError('authorization mismatch')
        if (self.auth.get('only_gpus')!=[2] or self.auth.get('execution_arms')!=list(EXECUTION_ARMS)
            or self.auth.get('FINAL')!=FINAL or self.auth.get('objective_policy')!='uniform1_align_mean_v1'
            or self.auth.get('formal_TEST_authorized') is not True):raise ValueError('authorization scope/resource/policy conflict')
        self.carry=read(OUT/'authorization/carry_in.json')
        if self.carry.get('active') is not False or self.carry.get('exit') is not None:raise ValueError('historical incident identity conflict')
        self.jobs=[];self.started_unix=self.auth['wallclock_started_unix'];self.start=None;self.cached_state=None;self.cached_budget=None;self.evidence_only=False
        self.recorded_checkpoint={}
        if self.started_unix!=ORIGINAL_WALLCLOCK_STARTED or self.auth['deadline_unix']!=ORIGINAL_DEADLINE:raise ValueError('persistent wallclock origin mismatch')
        self.validation={};self.formal={};self.partial={};self.frozen=None
        self.window=subprocess.check_output(['tmux','display-message','-p','-t',os.environ['TMUX_PANE'],'#{window_id}'],text=True).strip()
        self.manifest=dict(plan=PLAN,revision=REVISION,executor='DSH',writer_pid=os.getpid(),writer_process=process_identity(os.getpid()),controller_window=self.window,
            command=sys.argv,cwd=str(ROOT),authorization=str(OUT/'authorization/user_authorization.json'),limits=LIMITS,
            only_gpus=[2],world_size=1,pt_concurrency=1,concurrency_per_gpu=4,preregistered_candidate=FINAL,execution_arms=list(EXECUTION_ARMS),formal_TEST_authorized=True,objective_policy='uniform1_align_mean_v1',wallclock_limit_seconds=129600,wallclock_started_unix=self.started_unix,deadline_unix=ORIGINAL_DEADLINE,
            automatic_retry=False,automatic_resume=False,independent_review='PENDING',log=str(LOG/'controller_pg5k_gpu3'))
        with (OUT/'execution_manifest.json').open('w' if replacing else 'x') as f:json.dump(self.manifest,f,ensure_ascii=False,indent=2)
        self.reference=reference_scores();self.references=reused_cache_evidence()
        self.baseline=self.reference['formal_blocks']['F_G'];self.p10k=self.reference['formal_blocks']['P10K']
        self.manifest['references']=self.references
        write(OUT/'authorization/reference_results.json',self.reference)
        expected=sum(self.updates(t,f) for t in TASKS for f in range(5))
        if expected!=56000:raise ValueError('fixed original FT split/update budget differs')
        self.capture_diff();self.refresh()

    @classmethod
    def adopt_parallel_fg(cls):
        permission=read(OUT/'authorization/pg5k_gpu3.json')
        old=permission['old_controller']
        if process_alive(old) or permission.get('adopt_live_pt') is not True:
            raise ValueError('old controller must be fully stopped before exclusive adoption')
        self=cls.__new__(cls)
        self.lock=(OUT/'controller.lock').open('r+');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (OUT/'coordination/pg5k_gpu3_handoff/claim.json').open('x') as f:
            json.dump(dict(process=process_identity(os.getpid()),time=now(),authorization=permission),f,ensure_ascii=False,indent=2)
        prior=read(OUT/'coordination/pg5k_gpu3_handoff/execution_manifest.json')
        ledger=read(OUT/'coordination/pg5k_gpu3_handoff/budget.json')
        self.auth=read(OUT/'authorization/user_authorization.json');self.carry=read(OUT/'authorization/carry_in.json')
        if self.auth['limits']!=LIMITS or permission['limits']!=LIMITS or any(ledger['used'][k]>LIMITS[k] for k in LIMITS):raise ValueError('adoption budget mismatch')
        self.jobs=ledger['jobs'];self.started_unix=prior['wallclock_started_unix'];self.start=None
        if self.started_unix!=ORIGINAL_WALLCLOCK_STARTED:raise ValueError('clock reset prohibited')
        if (len([j for j in self.jobs if j['kind']=='pt'])!=2
            or len([j for j in self.jobs if j['kind']=='ft' and j['arm']=='F_G10K' and j['exit']==0])!=35
            or len([j for j in self.jobs if j['kind']=='test' and j['arm']=='F_G10K' and j['exit']==0])!=35
            or any(j['kind'] in ('ft','test') and j['arm']!='F_G10K' for j in self.jobs)):
            raise ValueError('unexpected prior evaluation jobs; no replay/adoption')
        current=read(OUT/'pretrain/P_G/consumption.json')
        if current['status']!='RUNNING' or not process_alive(current['process']):raise ValueError('P_G is not a healthy existing process')
        self.window=subprocess.check_output(['tmux','display-message','-p','-t',os.environ['TMUX_PANE'],'#{window_id}'],text=True).strip()
        self.manifest=dict(prior,writer_pid=os.getpid(),writer_process=process_identity(os.getpid()),controller_window=self.window,
            command=sys.argv,limits=LIMITS,additional_evaluation_arms=['P_G5K'],only_gpus=[2,3],arm_ft_test_gpus=FT_GPU,adopted_existing_pt=current['process'],handoff_authorization=permission,
            log=str(LOG/'controller_pg5k_gpu3'),status='执行中')
        self.cached_state=None;self.cached_budget=None;self.evidence_only=False
        self.recorded_checkpoint={j['arm']:j.get('progress',{}).get('last_checkpoint_step',0) for j in self.jobs if j['kind']=='pt'}
        self.validation={'F_G10K':aggregate('F_G10K')};self.formal={'F_G10K':aggregate('F_G10K',True)};self.partial={};self.frozen=None
        self.reference=read(OUT/'authorization/reference_results.json');self.references=read(OUT/'authorization/cache_reuse.json')
        self.baseline=self.reference['formal_blocks']['F_G'];self.p10k=self.reference['formal_blocks']['P10K']
        write(OUT/'execution_manifest.json',self.manifest);self.capture_diff();self.refresh();return self

    def capture_diff(self):
        (OUT/'final_review').mkdir(parents=True,exist_ok=True)
        patch=[]
        for rel in SOURCES:patch.extend(difflib.unified_diff([], (ROOT/rel).read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+rel))
        (OUT/'final_review/actual_source_diff.patch').write_text(''.join(patch))
        before=(OUT/'authorization/before/CURRENT_PLAN.md').read_text()
        incremental=[]
        for rel in SOURCES:
            old=(OUT/'authorization/before'/rel).read_text().splitlines(True)
            incremental.extend(difflib.unified_diff(old,(ROOT/rel).read_text().splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
        (OUT/'final_review/recovery_incremental_diff.patch').write_text(''.join(incremental))
        (OUT/'final_review/current_plan_diff.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),(ROOT/'CURRENT_PLAN.md').read_text().splitlines(True),fromfile='a/CURRENT_PLAN.md',tofile='b/CURRENT_PLAN.md')))

    def updates(self,task,fold):return 70*math.ceil(len(read(SPLIT/f'{task}.json')['folds'][fold]['train_indices'])/24)
    def deadline(self):
        if time.time()>=ORIGINAL_DEADLINE:raise TimeoutError('original persistent 36h deadline exhausted')
    def begin_clock(self):
        if self.started_unix!=ORIGINAL_WALLCLOCK_STARTED:raise ValueError('wallclock reset prohibited')
        self.deadline()

    def refresh(self):
        carry=getattr(self,'carry',{})
        used=dict(carry.get('conservative_charge',dict.fromkeys(LIMITS,0)))
        known=dict(carry.get('observed_lower_bound',dict.fromkeys(LIMITS,0)))
        remaining=dict.fromkeys(LIMITS,0);uncertain=[dict(historical_incident=carry)] if carry else []
        for j in self.jobs:
            key={'prepare':'prepare_starts','pt':'pt_starts','ft':'ft_starts','test':'test_starts','cpu':'cpu_suites'}[j['kind']]
            used[key]+=1;known[key]+=1
            p=Path(j['log'])/'exit.txt'
            if j['exit'] is None and p.exists():j.update(exit=int(p.read_text().strip()),ended=now(),status='COMPLETED' if int(p.read_text().strip())==0 else 'ABNORMAL_STOP')
            if j['exit'] is None and j.get('window') and j.get('status')!='EXIT_UNKNOWN':
                pane=subprocess.run(['tmux','list-panes','-t',j['window'],'-F','#{pane_dead}'],capture_output=True,text=True)
                if pane.returncode or (pane.stdout.strip() and all(v=='1' for v in pane.stdout.split())):
                    if p.exists():
                        code=int(p.read_text().strip());j.update(exit=code,ended=now(),status='COMPLETED' if code==0 else 'ABNORMAL_STOP')
                    else:j.update(status='EXIT_UNKNOWN',exit_evidence='owned window exited/disappeared without exit file; never fabricated')
            if j['kind']=='prepare':
                p=SIDECAR/'metadata.json'
                if p.exists():j['progress']=read(p)
            elif j['kind']=='pt':
                p=OUT/'pretrain'/j['arm']/'consumption.json';c=read(p) if p.exists() else {}
                attempted=int(c.get('attempted_updates',0));consumed=int(c.get('consumed_presentations',0))
                charged=attempted if c or j['exit'] in (None,0) else STEPS[j['arm']]
                used['pt_updates']+=charged;used['pt_presentations']+=charged*1008
                known['pt_updates']+=int(c.get('completed_updates',0));known['pt_presentations']+=consumed
                j['progress']=dict(c,charged_presentations=charged*1008,known_consumed_presentations=consumed,
                    read_prefetched_presentations=c.get('sample_presentations',0))
                if c.get('last_checkpoint_step',0)>getattr(self,'recorded_checkpoint',{}).get(j['arm'],0):
                    self.recorded_checkpoint[j['arm']]=c['last_checkpoint_step']
                    event=c['checkpoint_retention']
                    update_plan('执行中',j['arm']+'周期checkpoint',f"真实step={self.recorded_checkpoint[j['arm']]}；保留{event['retained']}；已删除仅本次旧中途文件{event['removed']}；依据授权最近两份策略，非旧事故/正式产物；累计保存耗时{c.get('checkpoint_seconds',0.)}秒。")
                if charged*1008!=consumed:uncertain.append(dict(job=j['name'],charged_presentations=charged*1008,known_consumed=consumed,reason='full in-progress/failed update reserved; reads/prefetch separate'))
                if j['exit'] is None:
                    remaining['pt_updates']+=max(0,STEPS[j['arm']]-charged);remaining['pt_presentations']+=max(0,STEPS[j['arm']]*1008-charged*1008)
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
            active_remaining_reservation=remaining,historical_carry_in=carry,new_consumption={k:used[k]-carry.get('conservative_charge',{}).get(k,0) for k in LIMITS},uncertainty=uncertain,budget_violations=violations,jobs=self.jobs,
            accounting='all failed/partial starts count; unclosed epoch charged in full, known actual separately; PT attempted update conservatively reserves1008 presentations; read/prefetch recorded separately; no retry/resume')
        encoded=json.dumps(payload,sort_keys=True,allow_nan=False)
        if encoded!=self.cached_state:
            payload.update(time=now(),wallclock_started_unix=self.started_unix,elapsed_seconds=max(0.,time.time()-self.started_unix) if self.started_unix is not None else 0.,deadline_unix=ORIGINAL_DEADLINE)
            write(OUT/'budget.json',payload);self.cached_budget=payload;self.cached_state=encoded
        if violations and not self.evidence_only:raise RuntimeError('budget exceeded: '+','.join(violations))
        return self.cached_budget

    def check_failures(self):
        failed=[j for j in self.jobs if j['exit'] not in (None,0) or j.get('status')=='EXIT_UNKNOWN']
        if failed:raise RuntimeError('job failed; no retry/resume: '+failed[0]['name']+' exit='+str(failed[0]['exit']))

    def launch(self,name,kind,args=(),gpu=None,arm=None,task=None,fold=None,allow_final_cpu=False):
        self.deadline();b=self.refresh()
        if not allow_final_cpu:self.check_failures()
        if b['budget_violations']:raise RuntimeError('no further dispatch after budget violation')
        if kind not in ('pt','ft','test','cpu'):raise ValueError('current stage prohibits prepare/L/other arms')
        if kind=='pt' and arm not in EXECUTION_ARMS:raise ValueError('only original two PT arms are authorized')
        if kind in ('ft','test') and arm not in EVAL_ARMS:raise ValueError('unregistered evaluation arm')
        if any(j['name']==name for j in self.jobs):raise ValueError('duplicate launch; no retry')
        key={'prepare':'prepare_starts','pt':'pt_starts','ft':'ft_starts','test':'test_starts','cpu':'cpu_suites'}[kind]
        if b['used'][key]>=LIMITS[key]:raise RuntimeError('start allowance exhausted')
        active=[j for j in self.jobs if j['exit'] is None and j.get('status')!='EXIT_UNKNOWN']
        compatible=[j for j in active if not (kind in ('ft','test') and arm in ('F_G10K','P_G5K') and j['kind']=='pt' and j['arm']=='P_G')]
        if any(j['kind']!=kind for j in compatible):raise ValueError('unauthorized stage overlap')
        if kind in ('ft','test'):
            if gpu!=FT_GPU[arm] or sum(j['gpu']==gpu for j in active)>=4:raise ValueError('GPU/concurrency violation')
            if unit_path(arm,task,fold,kind=='test').exists():raise FileExistsError('output conflict')
        if kind=='ft':
            cache_references();reservation=dict(epochs_started=70,validation_started=70,attempted_updates=self.updates(task,fold))
        elif kind=='pt':
            if gpu!=2:raise ValueError('PT restricted to physical GPU2')
            if active:raise ValueError('PT branches strictly serial')
            if arm not in ARMS:raise ValueError('unregistered PT arm')
            if (OUT/'pretrain'/arm).exists():raise FileExistsError('PT branch output conflict')
            cache_references();reservation=dict(pt_updates=STEPS[arm],pt_presentations=STEPS[arm]*1008)
        elif kind=='test':
            frozen=read(frozen_path(arm))
            if not frozen['frozen'] or not set(frozen['models']).issubset(set(EVAL_ARMS)) or arm not in frozen['models']:raise ValueError('TEST outside fixed freeze')
            reservation=dict(test_graphs=len(frozen['units'][f'{arm}/{task}/{fold}']['test_indices']))
        else:reservation={}
        for k,v in reservation.items():
            if b['used'][k]+b['active_remaining_reservation'][k]+v>LIMITS[k]:raise RuntimeError('insufficient full-unit reservation')
        if kind=='prepare':self.begin_clock()
        log=LOG/name;log.mkdir(parents=True,exist_ok=False)
        seconds={'prepare':43200,'pt':21600,'ft':7200,'test':300,'cpu':300}[kind]
        seconds=min(seconds,max(1,int(ORIGINAL_DEADLINE-time.time())))
        if kind=='prepare':command=[sys.executable,'-u','scripts/prepare_ua2d_3dshift.py','--workers','8']
        elif kind=='pt':command=[sys.executable,'-u','scripts/pretrain_ua2d_3dshift.py','--arm',arm]
        elif kind=='cpu':command=[sys.executable,'-m','pytest','-q','tests/test_ua2d_3dshift.py']
        else:command=[sys.executable,'-u','scripts/run_ua2d_3dshift.py',kind,*args]
        env=dict(CUDA_VISIBLE_DEVICES=str(gpu) if kind in ('pt','ft','test') else '',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        if kind in ('pt','ft','test'):env['WORLD_SIZE']='1'
        actual=['timeout','--signal=TERM','--kill-after=10s',f'{seconds}s',*command]
        text=shlex.join(['env',*[f'{k}={v}' for k,v in env.items()],*actual])
        script=log/'run.sh';script.write_text('#!/bin/bash\ncd '+shlex.quote(str(ROOT))+' || exit 90\ndate -Iseconds > '+shlex.quote(str(log/'start.txt'))+'\n'+text+' > '+shlex.quote(str(log/'stdout.log'))+' 2> '+shlex.quote(str(log/'stderr.log'))+'\nua_3dshift_exit=$?\nprintf "%s\\n" "$ua_3dshift_exit" > '+shlex.quote(str(log/'exit.txt'))+'\ndate -Iseconds > '+shlex.quote(str(log/'end.txt'))+'\nexit "$ua_3dshift_exit"\n')
        j=dict(name=name,kind=kind,arm=arm,task=task,fold=fold,gpu=gpu,argv=actual,env=env,command=text,cwd=str(ROOT),log=str(log),
            consumption=str(unit_path(arm,task,fold,kind=='test')/'consumption.json') if kind in ('ft','test') else None,
            exit=None,status='RUNNING',started=now(),reservation=reservation)
        self.jobs.append(j);self.refresh();write(log/'command.json',j)
        j['window']=subprocess.check_output(['tmux','new-window','-d','-P','-F','#{window_id}','-t','Uni-Poly:','-n',name,'bash',str(script)],text=True).strip()
        subprocess.run(['tmux','set-option','-w','-t',j['window'],'remain-on-exit','on'],check=True);write(log/'command.json',j);self.refresh();return j

    def wait(self,jobs):
        while True:
            self.deadline();self.refresh()
            if any(j['exit'] not in (None,0) or j.get('status')=='EXIT_UNKNOWN' for j in jobs):raise RuntimeError('job failed or exit unknown; no automatic retry')
            if all(j['exit']==0 for j in jobs):return
            time.sleep(5)

    def matrix(self,formal=False,arms=None):
        arms=EXECUTION_ARMS if arms is None else tuple(arms)
        self.refresh();self.check_failures()
        if any(j['exit'] is None and not (arms in (('F_G10K',),('P_G5K',)) and j['kind']=='pt' and j['arm']=='P_G') for j in self.jobs):
            raise ValueError('previous stage not closed outside authorized PG-PT / FG-FT-TEST overlap')
        queue=[(a,t,f) for a in arms for t in TASKS for f in range(5)];active=[];kind='test' if formal else 'ft'
        while queue or active:
            self.deadline();self.refresh();self.check_failures();active=[j for j in active if j['exit'] is None]
            while queue and len(active)<4:
                a,t,f=queue.pop(0);gpu=FT_GPU[a]
                active.append(self.launch(f'3dshift_{kind}_{a}_{t}_{f}',kind,['--arm',a,'--task',t,'--fold',str(f)],gpu=gpu,arm=a,task=t,fold=f))
            if active:time.sleep(5)

    def stop_owned(self):
        def exits():
            for j in self.jobs:
                p=Path(j['log'])/'exit.txt'
                if j['exit'] is None and p.exists():j.update(exit=int(p.read_text().strip()),ended=now(),status='COMPLETED' if int(p.read_text().strip())==0 else 'ABNORMAL_STOP')
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
                try:command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
                except FileNotFoundError:continue
                if str(Path(j['log'])/'run.sh') not in command:raise ValueError('owned wrapper identity mismatch')
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
                try:owned.append(process_identity(pid))
                except FileNotFoundError:continue
                try:os.kill(pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for identity in owned:
            if process_alive(identity):
                try:os.kill(identity['pid'],signal.SIGTERM)
                except ProcessLookupError:pass
        deadline=time.monotonic()+20;kill_at=time.monotonic()+10
        while (any(j['exit'] is None for j in active) or any(process_alive(p) for p in owned)) and time.monotonic()<deadline:
            if time.monotonic()>=kill_at:
                for identity in owned:
                    if process_alive(identity):
                        try:os.kill(identity['pid'],signal.SIGKILL)
                        except ProcessLookupError:pass
            time.sleep(1);exits()
        if any(process_alive(p) for p in owned):raise RuntimeError('owned processes not safely closed; no CPU overlap')
        for j in active:
            if j['exit'] is None:j.update(status='EXIT_UNKNOWN',exit_evidence='missing; no surviving owned process; never fabricated',ended=now())

    def close_pt_branch(self,arm):
        folder=OUT/'pretrain'/arm;c=read(folder/'consumption.json');m=read(folder/'metadata.json')
        if c.get('status')!='COMPLETE' or c.get('completed_updates')!=STEPS[arm] or c.get('attempted_updates')!=STEPS[arm] or c.get('consumed_presentations')!=STEPS[arm]*1008:raise ValueError('single-GPU PT incomplete')
        if any(process_alive(p) for p in [c['process'],*c['workers']]):raise RuntimeError('PT/worker still alive; phase overlap prohibited')
        if (m.get('status')!='COMPLETE' or m.get('step')!=STEPS[arm] or m.get('plan')!=PLAN or m.get('revision')!=REVISION
            or m.get('arm')!=arm or m.get('world_size')!=1):raise ValueError('PT branch provenance mismatch')
        for name in ('deploy.pt','resume.pt','rng.pt','initial_training_rng.pt'):
            if not (folder/name).is_file():raise FileNotFoundError('PT source missing '+str(folder/name))
        write(OUT/'selection'/f'pt_complete_{arm}.json',dict(plan=PLAN,revision=REVISION,arm=arm,status='COMPLETE',true_exit=0,
            consumption=c,pt_worker_processes_exited=True,time=now()))

    def collect(self):
        self.partial={}
        for arm in EVAL_ARMS:
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
        self.collect();budget=self.refresh();complete=set(self.formal)==set(EVAL_ARMS)
        compared,interaction=comparisons(self.validation,self.formal)
        selected=self.formal.get(FINAL) if complete else None
        delta_fg=selected['macro']-self.reference['test']['F_G'] if selected else None
        delta_control=selected['macro']-self.formal['F_G10K']['macro'] if selected else None
        delta_p10k=selected['macro']-self.reference['test']['P10K'] if selected else None
        passed=(delta_control>=.010) if selected else None
        status='待审查' if not error and cpu_exit==0 and complete else '需返修'
        all_formal=dict(self.reference['formal_blocks'],**self.formal)
        payload=dict(plan=PLAN,revision=REVISION,status=status,error=error,independent_review='PENDING',
            FINAL=FINAL,validation=self.validation,partial_validation=self.partial,formal=self.formal,
            formal_with_references=all_formal,complete_formal_TEST=complete,cpu_exit=cpu_exit,
            performance_pass=passed,objective_policy='uniform1_align_mean_v1',pt_physical_gpu=2,arm_ft_test_gpus=FT_GPU,carry_in=self.carry,delta_P_G_F_G=delta_fg,delta_P_G_P10K=delta_p10k,delta_P_G_F_G10K=delta_control,
            comparisons=compared,interaction=interaction,references=self.reference,budget=budget,
            aggregation='five equal folds per task, std(ddof1), then seven equal task means; raw unrounded decision',
            protocol='shared development TEST, not blind; F_G=historical BASE5k, P10K/B_FP are historical numerical references; PT common uniform1 policy and GPU2; F_G10K FT/TEST GPU3 vs P_G GPU2 is a resource difference; P_G5K is a user-added shared-development TEST evaluation after FG results, FINAL unchanged; no P by L interaction claim; paper protocol equivalence unproven')
        lines=[f'# {PLAN} / {REVISION} 交付','',f'状态：{status}；固定FINAL={FINAL}；完整正式TEST={complete}；CPU exit={cpu_exit}。',
            '执行自检不构成Codex独立审查。所有差值用未舍入值，展示3位小数；失败不追加候选或重试。','',
            '| 预登记对照 | 正式Δmacro R² | 完整性 |','|---|---:|---|']
        for name,c in compared.items():
            result=c['test'];value=format(result['delta'],'+.3f') if result['delta'] is not None else '未完成／不可比较'
            lines.append(f'| {name} | {value} | {result["status"]} |')
        lines+=['','| 臂 | validation macro | TEST macro |','|---|---:|---:|']
        all_validation=dict(self.reference['validation_blocks'],**self.validation)
        for arm in ('F_G',*EVAL_ARMS):
            vm=format(all_validation[arm]['macro'],'.3f') if arm in all_validation else '未完成／不可比较'
            tm=format(all_formal[arm]['macro'],'.3f') if arm in all_formal else '未完成／不可比较'
            lines.append(f'| {arm} | {vm} | {tm} |')
        rows={}
        for arm in ('F_G',*EVAL_ARMS):
            result=all_formal.get(arm);rows[arm]={}
            lines+=['',f'## {arm} 七任务数值对照','','| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |','|---|---:|---:|---:|---|']
            for task,(name,model,reference) in SOTA.items():
                value=result['tasks'][task] if result else None
                score=f'{value["mean"]:.3f}±{value["std"]:.3f}' if value else '未完成／不可比较'
                difference=f'{value["mean"]-reference:+.3f}' if value else '—'
                lines.append(f'| {name} | {score} | {reference:.3f} | {difference} | {model} |')
                rows[arm][task]=dict(project=value,reference=reference,reference_model=model,
                    delta=value['mean']-reference if value else None,protocol='numerical comparison only; equivalence unproven')
        lines+=['','共享开发TEST不是独立盲测；F_G=历史BASE5k，历史参考均只作数值对照；匹配对照为本次F_G10K；主要比较P_G−F_G10K，缺少L组合，不能估计严格2×2交互。P为整体表征包；FG微调/TEST使用GPU3，P_G使用GPU2，资源差异须说明；论文协议未证明等价。',
            '各任务五折seed42+fold等权、std(ddof1)，再七任务等权macro。完整预测、scaler、history、checkpoint、配置、冻结、预算、计时及真实exit保留原根。',
            f'主要对照P_G−F_G10K Δmacro={delta_control:+.3f}；性能通过={passed}。' if delta_control is not None else '主要正式比较未完成／不可比较。',
            '异常／未完成原因：'+str(error or ('无' if complete else '未完成／不可比较'))]
        payload['sota']=rows
        write(OUT/'final_review/results.json',payload)
        times={str(p.relative_to(OUT)):read(p) for p in OUT.glob('finetune/*/*/fold*/epoch_timing.json')}
        pt={str(p.relative_to(OUT)):read(p) for p in OUT.glob('pretrain/*/final_timing.json')}
        write(OUT/'final_review/training_timing.json',dict(epochs=times,pt=pt,jobs=self.jobs,real_training_only=True))
        report='\n'.join(lines)+'\n';(OUT/'final_review/report.md').write_text(report)
        update_plan(status,'实际交付',report);self.manifest.update(status=status,ended=now(),jobs=self.jobs);write(OUT/'execution_manifest.json',self.manifest);self.capture_diff()
        return status

    def run(self):
        error=None;cpu_exit=None;safe=True
        try:
            self.begin_clock()
            # Existing P_G PT and completed F_G10K evaluations are adopted, never restarted.
            update_plan('执行中','GPU3 P_G5K 35FT / GPU2 P_G PT并行','用户追加授权：读取固定保留的真实5000步checkpoint；相同公共head/70epochs配方，新增35FT/35TEST，原PT不中断。FG已完成结果只读复用。')
            self.matrix(arms=('P_G5K',));self.collect()
            freeze_final(self.validation,arms=('P_G5K',))
            update_plan('执行中','GPU3 P_G5K 35正式TEST','P_G5K全部35FT正常完成，独立冻结35BEST/scaler/parent/config/split/indices；每单元一次，不改选原FINAL。')
            self.matrix(True,arms=('P_G5K',));self.formal['P_G5K']=aggregate('P_G5K',True)
            pt=next(j for j in self.jobs if j['kind']=='pt' and j['arm']=='P_G')
            self.wait([pt]);self.close_pt_branch('P_G')
            update_plan('执行中','P_G 35FT','GPU2 P_G原PT完整10000且exit0、workers退出；不重训F_G10K或P_G5K，不追加候选。')
            self.matrix(arms=('P_G',));self.collect()
            freeze_final(self.validation,arms=('P_G',))
            self.frozen=freeze_final(self.validation,arms=EVAL_ARMS)
            self.matrix(True,arms=('P_G',));self.formal['P_G']=aggregate('P_G',True)
        except BaseException:
            error=traceback.format_exc();write(OUT/'failure.json',dict(error=error,time=now(),automatic_retry=False,automatic_resume=False));self.evidence_only=True
            try:self.stop_owned()
            except BaseException:error+='\n'+traceback.format_exc();safe=False
        finally:
            if self.frozen:
                for arm in EVAL_ARMS:
                    try:self.formal[arm]=aggregate(arm,True)
                    except (OSError,ValueError,KeyError):pass
            if safe:
                try:
                    update_plan('执行中' if not error else '需返修','唯一最终CPU suite','原两臂PT及含追加P_G5K的105FT/105TEST结束或安全异常停止后，仅本轮新专项tests/test_ua2d_3dshift.py一次，≤300秒，失败不重测。')
                    j=self.launch('3dshift_final_cpu','cpu',allow_final_cpu=True);self.wait([j]);cpu_exit=j['exit']
                except BaseException:error=(error or '')+'\n'+traceback.format_exc();cpu_exit=next((j['exit'] for j in self.jobs if j['kind']=='cpu'),None)
            status=self.deliver(error,cpu_exit);write(LOG/'controller_pg5k_gpu3/exit.json',dict(exit=0 if status=='待审查' else 1,status=status,error=error,time=now()))
        return 0 if status=='待审查' else 1


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','ft','test']);p.add_argument('--arm',choices=EVAL_ARMS);p.add_argument('--task',choices=TASKS);p.add_argument('--fold',type=int,choices=range(5));args=p.parse_args()
    torch.set_num_threads(2)
    def terminate(signum,frame):raise RuntimeError(f'controlled termination {signum}; no retry/resume')
    signal.signal(signal.SIGTERM,terminate)
    if args.action=='run':return Controller.adopt_parallel_fg().run()
    if args.arm not in EVAL_ARMS:p.error('outside explicitly authorized evaluation arms')
    if args.arm is None or args.task is None or args.fold is None:p.error('complete fixed unit identity required')
    unit(args.arm,args.task,args.fold,args.action=='test');return 0

if __name__=='__main__':raise SystemExit(main())
