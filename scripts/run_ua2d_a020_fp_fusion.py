#!/usr/bin/env python3
"""Authorized one-shot A020 neural fusion controller, GPU3 only; no retry/resume."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(key,'2')
import argparse
import difflib
import fcntl
import json
import re
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training import ua2d_a020_fp_fusion as t

SOURCES=('src/modules/ua2d_a020_fp_fusion.py','src/dataset/ua2d_a020_fp_fusion.py',
         'src/training/ua2d_a020_fp_fusion.py','scripts/run_ua2d_a020_fp_fusion.py','tests/test_ua2d_a020_fp_fusion.py')
WALLCLOCK=48*3600


def gpu_snapshot():
    try:
        devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True,timeout=10)
        uuid=None
        for line in devices.splitlines():
            index,value=[x.strip() for x in line.split(',')]
            if int(index)==3:uuid=value
        if uuid is None:return None
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=10)
        pids=set()
        for line in rows.splitlines():
            value,pid=[x.strip() for x in line.split(',')]
            if value==uuid:pids.add(int(pid))
        return pids
    except (OSError,ValueError,subprocess.SubprocessError):return None


def pid_aliases(identity):
    if not t.process_alive(identity):return set()
    try:
        status=(Path('/proc')/str(identity['pid'])/'status').read_text()
        for line in status.splitlines():
            if line.startswith('NSpid:'):
                values={int(x) for x in line.split()[1:]}
                return values if identity['pid'] in values else set()
    except (OSError,ValueError):pass
    return set()


def reservation(kind,task=None,fold=None):
    if kind=='prepare':return dict(prepare_starts=1)
    if kind=='cpu':return dict(cpu_suites=1)
    entry=t.read(t.SPLIT/f'{task}.json')['folds'][fold]
    if kind=='ft':return dict(ft_starts=1,epochs_started=70,attempted_updates=70*int(__import__('math').ceil(len(entry['train_indices'])/24)),
        train_graphs=70*len(entry['train_indices']),validation_started=70,validation_graphs=70*len(entry['validation_indices']))
    if kind=='test':return dict(test_starts=1,test_graphs=len(entry['test_indices']))
    if kind=='late':return dict(late_units=1,late_graphs=len(entry['test_indices']))
    raise ValueError('unregistered kind')


def ledger(jobs):
    # A failed/partial unit retains its full launch reservation. Known actual is separate.
    charged=dict.fromkeys(t.LIMITS,0);known=dict.fromkeys(t.LIMITS,0)
    for j in jobs:
        for key,value in j['reservation'].items():charged[key]+=value
        start_key={'prepare':'prepare_starts','ft':'ft_starts','test':'test_starts','late':'late_units','cpu':'cpu_suites'}[j['kind']]
        known[start_key]+=1
        path=Path(j['consumption']) if j.get('consumption') else None
        if path is not None and path.exists():
            c=t.read(path)
            for key in ('epochs_started','attempted_updates','train_graphs','validation_started','validation_graphs','test_graphs'):
                known[key]+=int(c.get(key,0))
        if j['kind']=='late' and j.get('exit')==0:known['late_graphs']+=j['reservation']['late_graphs']
    violations=[key for key,value in charged.items() if value>t.LIMITS[key]]
    return dict(limits=t.LIMITS,charged_or_reserved=charged,known_actual=known,budget_violations=violations,
        zero_cost=t.ZERO_COST,formal_policy_charged_units=charged['test_starts']+charged['late_units'],
        formal_policy_charged_presentations=charged['test_graphs']+charged['late_graphs'],
        accounting='full-unit irreversible reservation at launch; failed/partial units not refunded; known actual durable epoch records separately',jobs=jobs)


CUMULATIVE_LIMITS=dict(t.LIMITS,ft_starts=177,epochs_started=12390,attempted_updates=292740,
    train_graphs=6877920,validation_started=12390,validation_graphs=1725640)
NEW_LIMITS=dict(t.LIMITS,prepare_starts=0,ft_starts=129,epochs_started=9030,attempted_updates=208880,
    train_graphs=4900700,validation_started=9030,validation_graphs=1229900)


def cumulative_ledger(jobs,parent):
    fresh=ledger(jobs)
    cumulative={k:parent['charged_or_reserved'][k]+fresh['charged_or_reserved'][k] for k in t.LIMITS}
    lower={k:parent['known_actual'][k]+fresh['known_actual'][k] for k in t.LIMITS}
    violations=[k for k in t.LIMITS if cumulative[k]>CUMULATIVE_LIMITS[k] or fresh['charged_or_reserved'][k]>NEW_LIMITS[k]]
    fresh.update(parent_original_limits=parent['limits'],limits=CUMULATIVE_LIMITS,new_limits=NEW_LIMITS,
        parent_charged=parent['charged_or_reserved'],parent_actual_lower_bound=parent['known_actual'],
        new_charged_or_reserved=fresh['charged_or_reserved'].copy(),new_known_actual=fresh['known_actual'].copy(),
        charged_or_reserved=cumulative,known_actual=lower,budget_violations=violations,
        history_jobs=parent['jobs'],actual_semantics='cumulative durable lower bound; historical unpersisted progress unknown; no refunds')
    return fresh


def register_attempt_map(root,parent):
    path=root/'attempt_map.json'
    if path.exists():raise FileExistsError('attempt map is registered once only')
    jobs={f"{j['arm']}/{j['task']}/{j['fold']}":j for j in parent['jobs'] if j['kind']=='ft'}
    if len(jobs)!=48 or {k for k,j in jobs.items() if j['exit']!=0}!=t.RETRAIN_KEYS:
        raise ValueError('parent48/two-exception evidence mismatch')
    units={}
    for arm in t.ARMS:
        for task in t.TASKS:
            for fold in range(5):
                key=f'{arm}/{task}/{fold}';relative=Path('finetune')/arm/task/f'fold{fold}'
                job=jobs.get(key)
                if job is not None and job['exit']==0:
                    folder=t.OUT/relative;met=t.read(folder/'metrics.json');state=t.read(folder/'consumption.json')
                    if met.get('status')!='COMPLETE' or state.get('status')!='COMPLETE' or met.get('config')!=t.config(arm,task,fold):
                        raise ValueError('completed parent identity mismatch '+key)
                    if state.get('completed_epochs')!=70:raise ValueError('parent not full70 '+key)
                    entry=dict(source='parent_complete',attempt='r1',ft_dir=str(folder))
                else:
                    folder=root/relative
                    if folder.exists():raise FileExistsError('new attempt output exists '+key)
                    if job is None and (t.OUT/relative).exists():raise FileExistsError('unstarted parent has unexpected output '+key)
                    entry=dict(source='repair',attempt='repair-r1-retrain' if key in t.RETRAIN_KEYS else 'repair-r1-first',ft_dir=str(folder),
                        failed_parent_attempt=str(t.OUT/relative) if key in t.RETRAIN_KEYS else None,
                        initialization='original PT/head/common; no BEST resume')
                units[key]=entry
    result=dict(plan=t.PLAN,revision=t.REVISION,execution_plan=t.REPAIR_PLAN,execution_revision=t.REPAIR_REVISION,
        time=t.now(),units=units,parent_complete=46,new_attempts=129,retrain_exceptions=sorted(t.RETRAIN_KEYS))
    t.validate_attempt_map(result,root,t.OUT);t.exclusive_json(path,result)
    return result


def atomic_exit(path,rc):
    path=Path(path)
    if type(rc) is not int:raise ValueError('exit must be the true integer returncode')
    if path.exists():raise FileExistsError('terminal exit already exists: '+str(path))
    temporary=path.with_name(path.name+f'.{os.getpid()}.publishing')
    with temporary.open('x') as f:f.write(str(rc)+'\n');f.flush();os.fsync(f.fileno())
    if path.exists():raise FileExistsError('terminal writer conflict: '+str(path))
    os.replace(temporary,path)


class ExitProtocolError(RuntimeError):pass


def exit_context(job,raw,reason):
    process=Path(job['log'])/'process.json'
    try:identities=t.read(process) if process.exists() else job.get('wrapper_identity')
    except (OSError,ValueError) as error:identities=dict(read_error=repr(error))
    return ExitProtocolError(json.dumps(dict(reason=reason,job=job['name'],attempt=job.get('attempt'),
        path=str(Path(job['log'])/'exit.txt'),raw=raw,process=identities,window=job.get('window')),ensure_ascii=False))


def read_terminal(job):
    path=Path(job['log'])/'exit.txt'
    try:raw=path.read_text()
    except FileNotFoundError:return None,None
    except (OSError,UnicodeError) as error:raise exit_context(job,None,'exit read failed: '+repr(error))
    if not raw.strip():return None,raw
    if re.fullmatch(r'[+-]?\d+',raw.strip()) is None:raise exit_context(job,raw,'invalid terminal content')
    return int(raw.strip()),raw


def wrapper_status(job):
    path=Path(job['log'])/'process.json'
    try:
        if path.exists():
            process=t.read(path)
            return t.process_alive(process['wrapper']),t.process_alive(process['timeout_identity'])
        if job.get('wrapper_identity'):return t.process_alive(job['wrapper_identity']),False
        if job.get('window'):
            pane=subprocess.check_output(['tmux','display-message','-p','-t',job['window'],'#{pane_dead}'],text=True).strip()
            if pane not in ('0','1'):raise ValueError('unknown pane state')
            return pane=='0',False
        return True,True  # window publication has not completed yet
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as error:
        raise exit_context(job,None,'wrapper identity lookup failed: '+repr(error))


def poll_exit(job,clock=time.monotonic):
    rc,raw=read_terminal(job)
    if rc is not None:return rc
    alive,child_alive=wrapper_status(job)
    if not alive:
        # Re-read across the exit publication/window-death boundary.
        rc,latest=read_terminal(job)
        if rc is not None:return rc
        raise exit_context(job,latest,'dead wrapper without a legal terminal exit')
    if raw is None and child_alive:
        job.pop('terminal_wait_started',None)
        return None
    stamp=clock();started=job.setdefault('terminal_wait_started',stamp)
    if stamp-started>=10:raise exit_context(job,raw,'terminal publication exceeded10seconds')
    return None


def install_termination_handler():
    def interrupted(signum,frame):
        t.ignore_termination_signals()
        raise RuntimeError('controlled termination '+str(signum)+'; no retry/resume')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    return interrupted


def owned_group(job):
    path=Path(job['log'])/'process.json'
    if not path.exists():
        wrapper=job.get('wrapper_identity')
        if wrapper and t.process_alive(wrapper) and os.getpgid(wrapper['pid'])==wrapper['pid']:return wrapper['pid']
        return None
    process=t.read(path);leader=process['timeout_identity'];group=leader['pid']
    identities=[leader]
    if job.get('consumption') and Path(job['consumption']).exists():
        worker=t.read(Path(job['consumption'])).get('process')
        if worker:identities.append(worker)
    for identity in identities:
        if t.process_alive(identity) and os.getpgid(identity['pid'])==group:return group
    wrapper=process['wrapper']
    if t.process_alive(wrapper) and os.getpgid(wrapper['pid'])==wrapper['pid']:return wrapper['pid']
    return None


class Controller:
    @property
    def out(self):return t.execution_root()
    @property
    def log(self):return t.execution_log()
    def account(self):return cumulative_ledger(self.jobs,self.parent_budget) if self.repair else ledger(self.jobs)
    def __init__(self,repair=False):
        t.require_tmux()
        self.repair=repair;self.parent_budget=None
        if repair:
            root=t.OUT/'repairs/controller_repair1'
            self.lock=(root/'controller.lock').open('x');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.parent_budget=t.read(root/'parent_evidence.json')['parent_budget']
            register_attempt_map(root,self.parent_budget)
            t.enable_repair(root)
        if not repair:
            self.lock=(self.out/'controller.lock').open('x');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (self.out/'budget.json').exists():raise FileExistsError('already started; no retry/resume')
        self.auth=t.read(self.out/'authorization/user_authorization.json')
        if self.auth.get('plan')!=(t.REPAIR_PLAN if repair else t.PLAN) or self.auth.get('revision')!=(t.REPAIR_REVISION if repair else t.REVISION):raise ValueError('authorization identity mismatch')
        self.manifest=t.read(self.out/'execution_manifest.json');self.jobs=[];self.stopping=False
        self.manifest.update(controller_pid=os.getpid(),controller_identity=t.process_identity(os.getpid()),command=sys.argv,cwd=str(ROOT))
        if repair:self.manifest.update(parent_budget=str(t.OUT/'budget.json'),new_limits=NEW_LIMITS,cumulative_limits=CUMULATIVE_LIMITS,attempt_map=str(self.out/'attempt_map.json'),prepare_added=0,initialization_added=0)
        t.write(self.out/'execution_manifest.json',self.manifest)
        self.capture_diff();self.refresh()
    def deadline(self):
        if time.time()-self.manifest['started_unix']>=WALLCLOCK:raise TimeoutError('48h total wallclock exhausted')
    def capture_diff(self):
        changes=[]
        if self.repair:
            for name in ('scripts/run_ua2d_a020_fp_fusion.py','src/training/ua2d_a020_fp_fusion.py','tests/test_ua2d_a020_fp_fusion.py','CURRENT_PLAN.md'):
                before=(self.out/'before'/name).read_text()
                changes.extend(difflib.unified_diff(before.splitlines(True),(ROOT/name).read_text().splitlines(True),fromfile='before/'+name,tofile='after/'+name))
            (self.out/'repair_diff.patch').write_text(''.join(changes))
        else:
            for name in SOURCES:changes.extend(difflib.unified_diff([], (ROOT/name).read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+name))
        (self.out/'final_review/actual_source_diff.patch').write_text(''.join(changes))
    def collect_exits(self):
        errors=[]
        for j in self.jobs:
            if j['exit'] is not None:continue
            try:
                rc=poll_exit(j)
                if rc is not None:j.update(exit=rc,ended=t.now())
            except (ExitProtocolError,OSError,ValueError,KeyError) as error:
                j['infrastructure_failure']=str(error);errors.append(str(error))
        return errors
    def refresh(self):
        errors=self.collect_exits()
        payload=self.account();payload.update(time=t.now(),elapsed_seconds=time.time()-self.manifest['started_unix'],exit_protocol_errors=errors)
        t.write(self.out/'budget.json',payload)
        if errors and not self.stopping:raise ExitProtocolError('\n'.join(errors))
        if payload['budget_violations'] and not self.stopping:raise RuntimeError('budget violation')
        return payload
    def check_failures(self):
        failed=[j for j in self.jobs if j['exit'] not in (None,0) or j.get('infrastructure_failure')]
        if failed:raise RuntimeError('worker failure; no retry: '+failed[0]['name']+' exit='+str(failed[0]['exit']))
    def gpu_ready(self):
        active=[j for j in self.jobs if j['kind'] in ('ft','test') and j['exit'] is None]
        if len(active)>=4:return False
        pids=gpu_snapshot()
        if pids is None:
            t.write(self.out/'occupancy_wait.json',dict(time=t.now(),reason='query_failed_or_unresolved',action='wait_no_dispatch'))
            return False
        owned=set()
        try:
            for j in active:
                path=Path(j['consumption'])
                if path.exists():
                    state=t.read(path)
                    if state.get('process'):owned.update(pid_aliases(state['process']))
        except (OSError,KeyError,ValueError,TypeError):return False
        unknown=pids-owned
        if unknown:
            t.write(self.out/'occupancy_wait.json',dict(time=t.now(),compute_pids=sorted(pids),known_aliases=sorted(owned),unknown=sorted(unknown),action='wait_no_dispatch'))
        return not unknown
    def launch(self,kind,arm=None,task=None,fold=None):
        self.deadline()
        if getattr(self,'repair',False):
            if kind=='prepare':raise ValueError('repair prepare allowance is zero')
            if kind=='ft' and t.attempt_record(arm,task,fold)['source']!='repair':raise ValueError('parent COMPLETE cannot be launched again')
        if kind in ('ft','test') and not self.gpu_ready():return None
        self.refresh();self.check_failures()
        name=f'a020r1_{kind}' if self.repair else f'a020_{kind}'
        name=name+(f'_{arm}_{task}_{fold}' if arm else '')
        if any(j['name']==name for j in self.jobs):raise ValueError('duplicate launch')
        expected=reservation(kind,task,fold);b=self.account()
        if any(b['charged_or_reserved'][key]+value>b['limits'][key] for key,value in expected.items()):raise RuntimeError('unit budget unavailable')
        if kind in ('ft','test') and t.unit_path(arm,task,fold,kind=='test').exists():raise FileExistsError('unit output conflict')
        folder=self.log/name;folder.mkdir(parents=True,exist_ok=False)
        seconds={'prepare':1800,'ft':21600,'test':1800,'cpu':600}[kind]
        seconds=min(seconds,max(1,int(WALLCLOCK-(time.time()-self.manifest['started_unix']))))
        if kind=='cpu':command=['python','-m','pytest','-q','-p','no:cacheprovider','tests/test_ua2d_a020_fp_fusion.py']
        else:
            command=['python','-u','scripts/run_ua2d_a020_fp_fusion.py',kind]
            if arm:command+=['--arm',arm,'--task',task,'--fold',str(fold)]
            if self.repair:command+=['--repair-root',str(self.out)]
        argv=['timeout','--signal=TERM','--kill-after=10s',f'{seconds}s',*command]
        env=dict(CUDA_VISIBLE_DEVICES='3' if kind in ('ft','test') else '',WORLD_SIZE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        job=dict(name=name,kind=kind,arm=arm,task=task,fold=fold,argv=argv,env=env,cwd=str(ROOT),log=str(folder),
            reservation=expected,exit=None,started=t.now(),consumption=str(t.unit_path(arm,task,fold,kind=='test')/'consumption.json') if arm else None)
        job['attempt']=t.attempt_record(arm,task,fold) if arm else 'repair-r1-cpu' if self.repair else 'r1'
        self.jobs.append(job);self.refresh();t.write(folder/'command.json',job)
        launch=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2',
            'python','-u',str(ROOT/'scripts/run_ua2d_a020_fp_fusion.py'),'wrap','--spec',str(folder/'command.json')]
        job['window']=subprocess.check_output(['tmux','new-window','-d','-P','-F','#{window_id}','-t','Uni-Poly:','-n',name,'-c',str(ROOT),shlex.join(launch)],text=True).strip()
        subprocess.run(['tmux','set-option','-w','-t',job['window'],'remain-on-exit','on'],check=True)
        pane_pid=int(subprocess.check_output(['tmux','display-message','-p','-t',job['window'],'#{pane_pid}'],text=True))
        job['wrapper_identity']=t.process_identity(pane_pid)
        self.refresh();return job
    def wait(self,jobs):
        while True:
            self.deadline();self.refresh();self.check_failures()
            if all(j['exit']==0 for j in jobs):return
            time.sleep(5)
    def matrix(self,formal=False):
        queue=[(a,task,fold) for a in t.ARMS for task in t.TASKS for fold in range(5)]
        if self.repair and not formal:queue=[(a,task,f) for a,task,f in queue if t.attempt_record(a,task,f)['source']=='repair']
        target=len(queue);kind='test' if formal else 'ft';last_progress=None
        while queue or any(j['exit'] is None for j in self.jobs):
            self.deadline();self.refresh();self.check_failures()
            while queue:
                a,task,fold=queue[0];job=self.launch(kind,a,task,fold)
                if job is None:break
                queue.pop(0)
            complete=sum(j['kind']==kind and j['exit']==0 for j in self.jobs)
            active=sum(j['kind']==kind and j['exit'] is None for j in self.jobs)
            progress=(complete,active)
            if progress!=last_progress:
                t.update_plan('执行中','正式TEST' if formal else '175 FT',f'{kind}已完成{complete}/{target}，运行中{active}，待派发{len(queue)}；仅GPU3，外部占用门控；失败不重试。')
                last_progress=progress
            if queue or active:time.sleep(5)
    def stop_owned(self):
        self.stopping=True;t.ignore_termination_signals()
        errors=[];sent=set();end=time.monotonic()+30
        while True:
            errors.extend(self.collect_exits())
            live=[]
            for job in self.jobs:
                try:
                    group=owned_group(job)
                    if group is not None:
                        live.append(group)
                        if group not in sent:
                            os.killpg(group,signal.SIGTERM);sent.add(group)
                except (OSError,ValueError,KeyError) as error:errors.append(job['name']+': '+repr(error))
            if not live or time.monotonic()>=end:break
            time.sleep(min(1,max(0,end-time.monotonic())))
        for job in self.jobs:
            try:
                group=owned_group(job)
                if group is not None:os.killpg(group,signal.SIGKILL)
            except (OSError,ValueError,KeyError) as error:errors.append(job['name']+': '+repr(error))
        errors.extend(self.collect_exits())
        t.write(self.out/'cleanup.json',dict(time=t.now(),errors=errors,signaled_groups=sorted(sent),bounded_seconds=30))
        # Accounting is independent of the ordinary refresh/terminal reader.
        t.write(self.out/'budget.json',self.account())
    def run(self):
        try:
            if not self.repair:t.update_plan('执行中','一次prepare','只读缓存、身份/语义连接、train-only scaler与父证据，公共适配器初始化；无模型forward/生成。')
            if not self.repair:self.wait([self.launch('prepare')])
            else:t.check_prepared()
            t.update_plan('执行中','175 FT','五个固定训练臂；GPU3最多4路，70轮，无阶段测试。')
            self.matrix()
            frozen=t.freeze();t.update_plan('执行中','已一次冻结',f'FINAL={frozen["FINAL"]}，全局w={frozen["w"]}；随后175正式TEST，不按TEST重选。')
            self.matrix(True)
            for task in t.TASKS:
                for fold in range(5):
                    self.deadline();name=f'a020_late_NEURAL_LATE_{task}_{fold}'
                    job=dict(name=name,kind='late',task=task,fold=fold,started=t.now(),exit=None,
                        reservation=reservation('late',task,fold),log=str(self.log/name))
                    Path(job['log']).mkdir(exist_ok=False);self.jobs.append(job);self.refresh()
                    t.write(Path(job['log'])/'command.json',dict(command='cached original-unit blend; no model forward',cwd=str(ROOT),pid=os.getpid(),task=task,fold=fold,w=frozen['w'],started=job['started']))
                    t.late_unit(task,fold);job.update(exit=0,ended=t.now());t.write(Path(job['log'])/'exit.json',job);self.refresh()
            self.report('执行中')
            t.update_plan('执行中','唯一最终CPU suite','全部FT/冻结/正式政策/报告已完成；一次600秒CPU，失败不重跑。')
            job=self.launch('cpu');self.wait([job])
            t.write(self.out/'final_review/cpu_exit.json',dict(exit=job['exit'],command=job['argv'],log=job['log'],time=t.now(),attempts=1))
            self.report('待审查')
            self.manifest.update(status='待审查',ended=t.now(),elapsed_seconds=time.time()-self.manifest['started_unix'],cpu_exit=0)
            t.write(self.out/'execution_manifest.json',self.manifest);self.capture_diff();self.refresh()
            return 0
        except BaseException:
            error=traceback.format_exc();phase='execution' if self.repair or (self.out/'inputs/prepared.json').exists() else 'prepare'
            try:self.stop_owned()
            except BaseException:error+='\nstop_owned error: '+traceback.format_exc()
            status='阻断' if phase=='prepare' or '48h total wallclock exhausted' in error else '需返修'
            t.write(self.out/'failure.json',dict(status=status,error=error,time=t.now(),automatic_retry=False,automatic_resume=False))
            cpu=[j for j in self.jobs if j['kind']=='cpu']
            t.write(self.out/'final_review/cpu_exit.json',dict(exit=cpu[0]['exit'] if cpu else None,attempts=len(cpu),not_run_reason=None if cpu else 'not all authorized stages completed; no early suite'))
            self.manifest.update(status=status,error=error,ended=t.now());t.write(self.out/'execution_manifest.json',self.manifest)
            self.report(status,error);self.capture_diff();self.refresh();return 1
    def report(self,status,error=None):
        formal={};partial={}
        old=t.read(t.BASE/'final_review/results.json')['formal']['D_A020'];formal['B_A020']=old
        for arm in t.ARMS:
            try:formal[arm]=t.aggregate(arm,True)
            except (OSError,ValueError,KeyError):partial[arm]='未完成／不可比较'
        try:
            scores={task:[t.read(t.unit_path('NEURAL_LATE',task,f,True)/'metrics.json')['r2'] for f in range(5)] for task in t.TASKS}
            formal['NEURAL_LATE']=t.summary(scores)
        except (OSError,ValueError,KeyError):partial['NEURAL_LATE']='未完成／不可比较'
        frozen=t.read(self.out/'selection/final_frozen.json') if (self.out/'selection/final_frozen.json').exists() else {}
        final=frozen.get('FINAL');w=frozen.get('w');pairs=t.contrasts(formal,final)
        delta=(formal[final]['macro']-old['macro']) if final in formal else None
        result=dict(plan=t.PLAN,revision=t.REVISION,status=status,independent_review='PENDING',formal=formal,partial=partial,
            FINAL=final,w=w,baseline_macro=t.BASE_MACRO,delta=delta,performance_pass=delta>=.010 if delta is not None else None,
            comparisons=pairs,budget=self.refresh(),error=error,complete_formal_TEST=set(formal)==set(t.ALL_POLICIES),
            protocol='five equal folds, std ddof1; seed42+fold; seven equal tasks; shared development TEST, not independent blind',
            historical_deviations='A020 weighted ALIGN x2; legacy59998 cross-branch reused seeds; old all-GPU occupancy not independently verified; retained')
        lines=['# A020 神经FP融合 r1 实际交付','',f'状态：{status}；独立复审PENDING，未归档。',
            f'FINAL={final}；全局w={w}；未舍入Δmacro={delta!r}；性能通过={result["performance_pass"]}。','',
            '科学问题：同一无FP预训练A020上，注意力、门控或纯神经输出融合是否提高正式TEST；ZERO/FP配对仅改变输入。',
            '参考：D_A020原TEST macro=0.785044436513689。受控预算：五臂×七任务×五折×70epoch，seed42+fold。',
            '本周期无PT、FP/构象生成、核模型拟合/加载/查询/残差监督；旧FINAL=D_A005不变。','',
            '| 政策 | macro R² | ΔB_A020 |','|---|---:|---:|']
        for arm in t.ALL_POLICIES:
            x=formal.get(arm);lines.append(f'| {arm} | '+(f'{x["macro"]:.3f} | {x["macro"]-old["macro"]:+.3f} |' if x else '未完成／不可比较 | — |'))
        sota={}
        for arm in t.ALL_POLICIES:
            lines+=['',f'## {arm} 七任务数值对照','','| Task | 项目R² | 参考R² | 项目减参考 | 参考模型 |','|---|---:|---:|---:|---|']
            sota[arm]={}
            for task,(name,model,ref) in t.SOTA.items():
                value=formal.get(arm,{}).get('tasks',{}).get(task)
                sota[arm][task]=dict(project=value,reference=ref,reference_model=model,delta=value['mean']-ref if value else None)
                lines.append(f'| {name} | '+(f'{value["mean"]:.3f}±{value["std"]:.3f} | {ref:.3f} | {value["mean"]-ref:+.3f} | {model} |' if value else f'未完成／不可比较 | {ref:.3f} | — | {model} |'))
        lines+=['','## 预登记配对对照','','| 对照 | Δmacro R² | 改善fold |','|---|---:|---:|']
        for name,x in pairs.items():
            lines.append(f'| {name} | '+(f'{x["delta"]:+.3f} | {x["improved_folds"]}/35 |' if x['delta'] is not None else '未完成／不可比较 | — |'))
            if x['status']=='COMPLETE':
                for task,z in x['tasks'].items():
                    lines.append(f'\n{name} / {task}: mean±std={z["mean"]:+.3f}±{z["std"]:.3f}，五折差值='+', '.join(f'{v:+.3f}' for v in z['folds'])+f'，改善{z["improved_folds"]}/5。')
        lines+=['','## 实际成本','',json.dumps(result['budget']['known_actual'],ensure_ascii=False),
            '\n完整精度结果、35个逐fold差值、参数量和实际秒数见results.json/各单元parameters.json、consumption.json及history.json；所有计时来自真实执行，不作benchmark。']
        costs={}
        for arm in t.ARMS:
            records=[]
            for task in t.TASKS:
                for fold in range(5):
                    folder=t.unit_path(arm,task,fold)
                    if (folder/'parameters.json').exists():
                        records.append(dict(task=task,fold=fold,parameters=t.read(folder/'parameters.json'),
                            ft=t.read(folder/'consumption.json'),test=t.read(t.unit_path(arm,task,fold,True)/'consumption.json') if (t.unit_path(arm,task,fold,True)/'consumption.json').exists() else None))
            costs[arm]=records
            if records:
                lines.append(f'\n{arm}: 参数量{records[0]["parameters"]["total"]}，已记录FT实际墙钟合计{sum(x["ft"].get("elapsed_seconds",0.) for x in records):.3f}秒，TEST合计{sum((x["test"] or {}).get("elapsed_seconds",0.) for x in records):.3f}秒。')
        lines+=['','## 解释边界','',
            'FP信息效应必须看UNI_FP−UNI_ZERO、TMOL_FP−TMOL_ZERO；胜基线但不胜ZERO不归因于FP。NEURAL_LATE若w=1则无FP贡献。不把fused−baseline当interaction；门控/注意力仅诊断。',
            '父PT保留ALIGN×2及legacy_offset_v1的59998个跨分支seed复用；本周期不追溯修复历史随机独立性。历史全卡并发未独立核实。',
            '各任务五折等权mean±std(ddof=1)，seed42+fold，再七任务等权macro。论文为逐任务最佳包络；数据/split/协议等价性未证明，仅数值对照。Egc3380/Xc432不同于论文4124/430；共享开发TEST非独立盲测，单seed/fold不足以证明跨seed稳定优势。',
            f'异常／未完成原因：{error or ("无" if not partial else "部分正式结果未完成，见partial")}。']
        cpu=t.read(self.out/'final_review/cpu_exit.json') if (self.out/'final_review/cpu_exit.json').exists() else None
        lines+=['',f'最终CPU：{json.dumps(cpu,ensure_ascii=False) if cpu else "尚未运行；仅在全部阶段及报告完成后一次运行"}。',
            '\n窗口Uni-Poly:a020-fp-fusion-repair1（返修）/a020-fp-fusion-r1（原始）；worker独立window见budget.jobs；日志'+str(self.log)+'；命令、PID、起止、真实exit均在各job目录。']
        if self.repair:
            result.update(execution_plan=t.REPAIR_PLAN,execution_revision=t.REPAIR_REVISION,attempt_map=t.ATTEMPT_MAP,
                parent_evidence=str(self.out/'parent_evidence.json'),parent_failure=str(t.OUT/'failure.json'))
            lines+=['','## 累计返修与历史成本','',
                '复用46个原COMPLETE，仅129个新attempt；其中UNI_FP/egc/fold1、fold2是唯一从原PT/head/common重新训练的例外，未从BEST续训、不比较旧新attempt择优。父两个失败及事故证据完整保留。',
                '科学config/checkpoint保持原r1；repair-r1只记录执行层。module/dataset与旧无FP六文件未修改，prepare/common/缓存全部只读复用，新增prepare/初始化为0。',
                '父actual为持久化下界，中止时未落盘进度未知。下方known_actual累计仍只是下界；新成功单元有完整计数，不能据此把父失败成本改成精确actual。',
                '| 消费项 | 父已收费 | 本次收费/预留 | 累计收费/预留 | 累计上限 |','|---|---:|---:|---:|---:|']
            budget=result['budget']
            for key in t.LIMITS:
                lines.append(f'| {key} | {budget["parent_charged"][key]} | {budget["new_charged_or_reserved"][key]} | {budget["charged_or_reserved"][key]} | {CUMULATIVE_LIMITS[key]} |')
            lines+=['','父历史事故与无FP父PT legacy/ALIGN×2限制均保留。绝对截止2026-10-26 20:42:23.561393+08:00，未因返修重置。独立复审PENDING，未归档。']
        result.update(sota=sota,costs=costs,cpu=cpu)
        t.write(self.out/'final_review/results.json',result);report='\n'.join(lines)+'\n';(self.out/'final_review/report.md').write_text(report)
        t.update_plan(status,'交付报告',report)


def wrap(spec):
    """CPU-only wrapper preserves the child's true timeout exit and process group."""
    job=t.read(spec);folder=Path(job['log']);env=os.environ.copy();env.update(job['env'])
    identity=None;requested=[];forwarded=False
    def forward_stop(signum,frame):
        nonlocal forwarded
        requested.append(signum)
        if not forwarded and identity is not None and t.process_alive(identity) and os.getpgid(identity['pid'])==identity['pid']:
            forwarded=True;os.killpg(identity['pid'],signal.SIGTERM)
    signal.signal(signal.SIGTERM,forward_stop);signal.signal(signal.SIGINT,forward_stop)
    with (folder/'stdout.log').open('x') as out,(folder/'stderr.log').open('x') as err:
        child=subprocess.Popen(job['argv'],cwd=job['cwd'],env=env,stdout=out,stderr=err,start_new_session=True)
        identity=t.process_identity(child.pid)
        t.write(folder/'process.json',dict(wrapper=t.process_identity(os.getpid()),timeout_identity=identity,
            command=job['argv'],cwd=job['cwd'],started=t.now()))
        if requested:forward_stop(requested[0],None)
        rc=child.wait()
    atomic_exit(folder/'exit.txt',rc);t.write(folder/'end.json',dict(exit=rc,ended=t.now(),job=job['name'],attempt=job.get('attempt')))
    return rc


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('controller','repair','prepare','ft','test','wrap'))
    p.add_argument('--arm',choices=t.ARMS);p.add_argument('--task',choices=t.TASKS);p.add_argument('--fold',type=int,choices=range(5));p.add_argument('--spec');p.add_argument('--repair-root')
    args=p.parse_args();t.torch.set_num_threads(2)
    if args.action=='wrap':return wrap(args.spec)
    install_termination_handler()
    if args.action=='repair':return Controller(repair=True).run()
    if args.repair_root:t.enable_repair(args.repair_root)
    if args.action=='controller':return Controller().run()
    if args.action=='prepare':t.prepare();return 0
    if args.arm is None or args.task is None or args.fold is None:p.error('complete fixed identity required')
    t.unit(args.arm,args.task,args.fold,args.action=='test');return 0

if __name__=='__main__':raise SystemExit(main())
