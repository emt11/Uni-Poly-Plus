#!/usr/bin/env python3
"""UA2D-OPT r1 single writer: A -> preparation/B -> conditional C/D -> TEST -> CPU."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(name,'2')
import argparse,fcntl,json,shlex,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training.ua2d_opt import *

class Controller:
    def __init__(self,handoff=None):
        require_tmux()
        self.handoff=Path(handoff).resolve() if handoff else None
        if self.handoff is not None and self.handoff!=(OUT/'controller_handoff_repair1/request.json').resolve():
            raise ValueError('only the explicitly authorized repair1 handoff is allowed')
        self.lock=(OUT/'controller.lock').open('r+' if self.handoff else 'x');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if read(OUT/'execution_manifest.json')['plan']!=PLAN:raise ValueError('authorization identity mismatch')
        self.jobs=[];self.start=None;self.started_unix=None
        self.window=subprocess.check_output(['tmux','display-message','-p','-t',os.environ['TMUX_PANE'],'#{window_id}'],text=True).strip()
        manifest=read(OUT/'execution_manifest.json')
        if self.handoff:
            request=read(self.handoff);budget=read(OUT/'budget.json')
            if (request.get('plan')!=PLAN or request.get('repair')!='repair1'
                or request.get('state')!='controller_stopped_workers_untouched'
                or request.get('expected_limits')!=LIMITS or budget['limits']!=LIMITS
                or request['wallclock_started_unix']!=manifest['wallclock_started_unix']
                or any(j['kind']!='ft' or j['arm'] not in ('A_TOP2','A_MIX246') for j in budget['jobs'])
                or len(budget['jobs'])!=request['jobs']
                or len({j['name'] for j in budget['jobs']})!=len(budget['jobs'])):
                raise ValueError('repair1 ledger/phase/authorization mismatch')
            old=Path('/proc')/str(request['old_pid'])
            if old.exists() and old.joinpath('stat').read_text().split(') ',1)[1].split()[0]!='Z':
                raise RuntimeError('old controller still exists; no concurrent writer')
            self.jobs=budget['jobs'];self.started_unix=manifest['wallclock_started_unix']
            self.start=time.monotonic()-(time.time()-self.started_unix)
            with self.handoff.with_name('claim.json').open('x') as f:
                json.dump(dict(new_pid=os.getpid(),new_window=self.window,time=now(),adopted_jobs=[j['name'] for j in self.jobs],
                    wallclock_started_unix=self.started_unix,reset_budget=False,worker_retry=False),f,ensure_ascii=False,indent=2)
        self.controller_log=LOG/('controller_repair1' if self.handoff else 'controller')
        manifest.update(writer_pid=os.getpid(),controller_window=self.window,command=sys.argv,limits=LIMITS,tmux_session='Uni-Poly',log=str(self.controller_log),concurrency_per_gpu=4,pt_world=2,wallclock_limit_seconds=259200,
            runtime_policy='formal_TEST_outcome_only_numeric_context_cache_v1')
        if self.handoff:manifest['controller_handoff']=str(self.handoff)
        write(OUT/'execution_manifest.json',manifest)
        self.capture_diff('implementation_repair1' if self.handoff else 'implementation_before_start')
        self.refresh()
    def capture_diff(self,name):
        folder=OUT/'final_review';folder.mkdir(exist_ok=True)
        paths=['src/dataset/ua2d_opt.py','src/modules/ua2d_opt.py','src/training/ua2d_opt.py','scripts/run_ua2d_opt.py','scripts/prepare_ua2d_opt.py','scripts/pretrain_ua2d_opt.py','tests/test_ua2d_opt.py']
        patches=[]
        for p in paths:
            if (ROOT/p).is_file():
                r=subprocess.run(['git','diff','--no-index','/dev/null',p],capture_output=True,text=True)
                if r.returncode not in (0,1):raise RuntimeError('cannot capture actual diff '+p)
                patches.append(r.stdout)
        (folder/f'{name}.patch').write_text('\n'.join(patches))
        (folder/'plan_diff.patch').write_text(subprocess.check_output(['git','diff','--','CURRENT_PLAN.md'],text=True))
    def begin_clock(self):
        if self.start is None:
            self.start=time.monotonic();self.started_unix=time.time()
            manifest=read(OUT/'execution_manifest.json');manifest['wallclock_started']=now();manifest['wallclock_started_unix']=self.started_unix
            write(OUT/'execution_manifest.json',manifest)
    def deadline(self):
        if self.start is not None and time.monotonic()-self.start>=259200:raise TimeoutError('72h campaign deadline exhausted')
    def refresh(self):
        used=dict.fromkeys(LIMITS,0);unknown=[]
        for job in self.jobs:
            key={'prepare':'prepare_starts','initialize':'initialization_starts','pt':'pt_starts','ft':'ft_starts','test':'test_starts','cpu':'cpu_suites'}[job['kind']];used[key]+=1
            exitfile=Path(job['log'])/'exit.txt'
            if job['exit'] is None and exitfile.exists():job.update(exit=int(exitfile.read_text().strip()),ended=now())
            if job['kind'] in ('ft','test'):
                path=Path(job['consumption'])
                if path.exists():
                    c=read(path);job['progress']=c
                    for k in ('epochs_started','validation_started','attempted_updates','test_graphs'):used[k]+=c.get(k,0)
                elif job['exit'] is not None:unknown.append(dict(job=job['name'],reason='worker exited before durable consumption record; start counts, updates unavailable'))
            elif job['kind']=='pt':
                ranks=[]
                for rank in (0,1):
                    p=OUT/'pretrain'/job['arm']/f'rank{rank}_consumption.json'
                    if p.exists():ranks.append(read(p))
                used['pt_updates']+=max([r['attempted_updates'] for r in ranks],default=0)
                used['pt_presentations']+=sum(r['sample_presentations'] for r in ranks)
                job['rank_progress']=ranks
                if len(ranks)<2 and job['exit'] is not None:unknown.append(dict(job=job['name'],reason='missing final rank counters; recorded values lower bounds, unused allowance not transferable'))
        if any(used[k]>LIMITS[k] for k in LIMITS):raise RuntimeError('global budget exceeded')
        elapsed=time.monotonic()-self.start if self.start is not None else 0.
        budget=dict(plan=PLAN,revision='r1',limits=LIMITS,used=used,jobs=self.jobs,time=now(),elapsed_seconds=elapsed,unknown=unknown,accounting='all starts incl failures; durable counters are lower bounds on abrupt exit; PT updates max across ranks, presentations sum across ranks; no retry/resume')
        write(OUT/'budget.json',budget);write(OUT/'timing.json',dict(started_unix=self.started_unix,elapsed_seconds=elapsed,limit_seconds=259200,time=now()))
        return budget
    def launch(self,name,kind,args,gpu=None,arm=None,consumption=None):
        self.deadline();budget=self.refresh()
        if any(j['name']==name for j in self.jobs):raise RuntimeError('dispatch twice forbidden')
        key={'prepare':'prepare_starts','initialize':'initialization_starts','pt':'pt_starts','ft':'ft_starts','test':'test_starts','cpu':'cpu_suites'}[kind]
        if budget['used'][key]>=LIMITS[key]:raise RuntimeError('start budget exhausted')
        active=[j for j in self.jobs if j['exit'] is None]
        if kind=='pt' and any(j['kind'] in ('pt','ft','test') for j in active):raise RuntimeError('PT/FT overlap')
        if kind in ('ft','test') and any(j['kind']=='pt' for j in active):raise RuntimeError('FT/PT overlap')
        if kind in ('ft','test') and (gpu not in (2,3) or sum(j['gpu']==gpu for j in active)>=4):raise RuntimeError('physical GPU/concurrency limit')
        if kind=='test':
            frozen=read(OUT/'selection/final_frozen.json')
            if not frozen['frozen'] or arm not in frozen['models']:raise RuntimeError('TEST forbidden before freeze or outside frozen list')
        if kind in ('ft','prepare'):self.begin_clock()
        log=getattr(self,'log_root',LOG)/name;log.mkdir(parents=True,exist_ok=False)
        seconds={'prepare':14400,'initialize':300,'pt':18000,'ft':1800,'test':300,'cpu':300}[kind]
        if kind=='ft' and getattr(self,'ft_repair1',False):seconds=7200
        if kind=='cpu':command=[sys.executable,'-m','pytest','-q','tests/test_ua2d_opt.py']
        elif kind=='prepare':command=[sys.executable,'-u','scripts/prepare_ua2d_opt.py']
        elif kind=='pt':command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2','scripts/pretrain_ua2d_opt.py',*args]
        else:command=[sys.executable,'-u',str(Path(__file__).resolve()),*args]
        if self.start is not None:seconds=min(seconds,max(1,int(259200-(time.monotonic()-self.start))))
        env=dict(CUDA_VISIBLE_DEVICES='2,3' if kind=='pt' else str(gpu) if gpu is not None else '',UA2D_OPT='r1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        if getattr(self,'ft_repair1',False):env.update(UA2D_OPT_FT_REPAIR1='1',UA2D_OPT_UNIT_JOB=name)
        actual=['timeout','--signal=TERM','--kill-after=10s',f'{seconds}s',*command]
        cmd=shlex.join(['env',*[f'{k}={v}' for k,v in env.items()],*actual])
        wrapper=f'''#!/bin/bash
cd {shlex.quote(str(ROOT))} || exit 90
date -Iseconds > {shlex.quote(str(log/'start.txt'))}
{cmd} > {shlex.quote(str(log/'stdout.log'))} 2> {shlex.quote(str(log/'stderr.log'))}
ua_opt_exit=$?
printf '%s\\n' "$ua_opt_exit" > {shlex.quote(str(log/'exit.txt'))}
date -Iseconds > {shlex.quote(str(log/'end.txt'))}
exit "$ua_opt_exit"
'''
        script=log/'run.sh';script.write_text(wrapper)
        job=dict(name=name,kind=kind,arm=arm,gpu=gpu,argv=actual,env=env,command=cmd,cwd=str(ROOT),log=str(log),exit=None,started=now(),consumption=str(consumption) if consumption else None)
        if getattr(self,'ft_repair1',False):self.reserve_job(job,args)
        self.jobs.append(job);self.refresh();write(log/'command.json',job)
        job['window']=subprocess.check_output(['tmux','new-window','-d','-P','-F','#{window_id}','-t','Uni-Poly:','-n',name,'bash',str(script)],text=True).strip()
        if getattr(self,'ft_repair1',False):subprocess.run(['tmux','set-option','-w','-t',job['window'],'remain-on-exit','on'],check=True)
        self.refresh();return job
    def wait(self,jobs):
        while True:
            self.deadline();self.refresh()
            bad=[j for j in jobs if j['exit'] not in (None,0)]
            if bad:raise RuntimeError('job failed; no retry: '+bad[0]['name']+f" exit={bad[0]['exit']}")
            if all(j['exit']==0 for j in jobs):return
            time.sleep(5)
    def matrix(self,arms,formal=False):
        self.refresh();kind='test' if formal else 'ft'
        requested=[(a,t,f) for a in arms for t in TASKS for f in range(5)]
        names={f'opt_{kind}_{a}_{t}_{f}' for a,t,f in requested}
        existing={j['name']:j for j in self.jobs if j['name'] in names}
        if existing and not self.handoff:raise RuntimeError('matrix replay forbidden without explicit controller handoff')
        if any(j['exit'] not in (None,0) for j in existing.values()):raise RuntimeError('failed unit cannot be retried')
        if any(j['exit'] is None and j['name'] not in names for j in self.jobs):raise RuntimeError('unrelated in-flight stage during matrix handoff')
        queue=[(a,t,f) for a,t,f in requested if f'opt_{kind}_{a}_{t}_{f}' not in existing]
        active=[j for j in existing.values() if j['exit'] is None]
        while queue or active:
            self.deadline();self.refresh()
            if any(j['exit'] not in (None,0) for j in active):raise RuntimeError('matrix unit failed; stop dispatch')
            active=[j for j in active if j['exit'] is None]
            for gpu in (2,3):
                while queue and sum(j['gpu']==gpu for j in active)<4:
                    a,t,f=queue.pop(0);kind='test' if formal else 'ft'
                    active.append(self.launch(f'opt_{kind}_{a}_{t}_{f}',kind,[kind,'--arm',a,'--task',t,'--fold',str(f)],gpu,arm=a,consumption=unit_path(a,t,f,formal)/'consumption.json'))
            if active:time.sleep(5)
    def pt(self,arm):
        from dataclasses import asdict
        from src.modules.ua2d_galformer_pretrain import PretrainConfig
        path=OUT/'configs'/f'{arm}.json';write(path,asdict(PretrainConfig()))
        job=self.launch('opt_pt_'+arm,'pt',['--arm',arm,'--config',str(path),'--output',str(OUT/'pretrain'/arm),'--initialization',str(OUT/'initialization'/f'{arm[-1]}_step0.pt'),'--prepared-cache',str(CACHE)],arm=arm)
        self.wait([job]);self.matrix([arm])
    def stop_owned(self):
        # Only descendants of this campaign's own exact tmux job panes.
        self.refresh()
        active=[j for j in self.jobs if j['exit'] is None and j.get('window')]
        for j in active:
            r=subprocess.run(['tmux','list-panes','-t',j['window'],'-F','#{pane_pid}'],capture_output=True,text=True)
            if r.returncode:continue
            roots={int(x) for x in r.stdout.split()};parents={}
            for p in Path('/proc').iterdir():
                if not p.name.isdigit():continue
                try:
                    fields=(p/'stat').read_text().split(') ',1)[1].split();parents[int(p.name)]=int(fields[1])
                except (OSError,ValueError,IndexError):pass
            descendants=set(roots)
            while True:
                new={p for p,parent in parents.items() if parent in descendants}-descendants
                if not new:break
                descendants|=new
            descendants-=roots
            # Require this window's wrapper still names the fixed job path.
            for root in roots:
                try:cmd=(Path('/proc')/str(root)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
                except OSError:continue
                if str(Path(j['log'])/'run.sh') not in cmd:raise RuntimeError('owned wrapper identity mismatch')
            for pid in sorted(descendants,reverse=True):
                try:os.kill(pid,signal.SIGTERM)
                except ProcessLookupError:pass
        deadline=time.monotonic()+20
        while any(j['exit'] is None for j in active) and time.monotonic()<deadline:
            time.sleep(1);self.refresh()
        # Missing exit evidence or surviving ranks blocks CPU overlap.
        if any(j['exit'] is None for j in active):raise RuntimeError('owned processes not safely closed; no CPU overlap')
    def run(self):
        validation={};formal={};decision=None;error=None;cpu_exit=None;safe=True
        try:
            baseline=read(PARENT.parents[2]/'final_review/results.json')['validation']['B_FP']['macro']
            required=70*sum(math.ceil(len(read(SPLIT/f'{t}.json')['folds'][f]['train_indices'])/24) for t in TASKS for f in range(5))
            if required!=56000:raise ValueError('fixed FT matrix updates differ from plan')
            update_plan('执行中','A','A_TOP2／A_MIX246固定70FT；controller '+self.window+'，仅GPU2/3，每卡最多4路，启动即计账。')
            self.matrix(['A_TOP2','A_MIX246'])
            for arm in ('A_TOP2','A_MIX246'):validation[arm]=aggregate(arm)
            a=a_decision(validation,baseline);write(OUT/'selection/a_policy.json',a)
            update_plan('执行中','统一输入准备',f"A完成70FT；FT_POLICY={a['FT_POLICY']}，启动唯一输入准备。")
            self.wait([self.launch('opt_prepare','prepare',[])])
            self.wait([self.launch('opt_initialize','initialize',['initialize'])])
            for arm in B_ARMS:
                update_plan('执行中','B',f'{arm}：唯一5000updates PT后35FT；无阶段TEST。')
                self.pt(arm);validation[arm]=aggregate(arm)
            write(OUT/'selection/b_factorial.json',dict(validation={a:validation[a] for a in B_ARMS},effects=factorial(validation)))
            c=c_decision(validation);write(OUT/'selection/conditional_decisions.json',dict(C=c,D=dict(trigger=a['FT_POLICY']!='NONE'),PT_WINNER=None))
            if c['trigger']:
                update_plan('执行中','C',f"触发{c['arm']}，同step0、唯一分层schedule。")
                self.pt(c['arm']);validation[c['arm']]=aggregate(c['arm'])
            ptwinner=choose({arm:validation[arm]['macro'] for arm in (*B_ARMS,*([c['arm']] if c['trigger'] else []))},TIE_ORDER)
            conditional=dict(C=c,D=dict(trigger=a['FT_POLICY']!='NONE',FT_POLICY=a['FT_POLICY']),PT_WINNER=ptwinner)
            write(OUT/'selection/conditional_decisions.json',conditional)
            if a['FT_POLICY']!='NONE':
                update_plan('执行中','D',f"PT_WINNER={ptwinner}；重新35FT，政策{a['FT_POLICY']}，不继承FT best。")
                self.matrix(['D_INTEGRATE']);validation['D_INTEGRATE']=aggregate('D_INTEGRATE')
            decision=final_decision(validation,baseline,ptwinner);write(OUT/'selection/development_decision.json',decision)
            freeze_final(validation,decision,conditional)
            update_plan('执行中','唯一正式TEST',f"FINAL={decision['FINAL']}，不以validation性能设准入门槛；全部名单／best／scaler／indices已冻结，正式模型：{decision['models']}。")
            self.matrix(decision['models'],True)
            for arm in decision['models']:formal[arm]=aggregate(arm,True)
            if all(arm in formal for arm in B_ARMS):write(OUT/'selection/formal_factorial.json',factorial(formal))
        except BaseException:
            error=traceback.format_exc();write(OUT/'failure.json',dict(error=error,time=now(),automatic_retry=False,automatic_resume=False))
            try:self.stop_owned()
            except BaseException:
                error+='\n'+traceback.format_exc();safe=False
        finally:
            if safe:
                try:
                    # Once, after all real runs or their safe exceptional closeout.
                    update_plan('执行中' if error is None else '需返修','唯一最终CPU suite','全部实际作业已结束／安全停止，运行一次针对性CPU suite；失败不重测。')
                    job=self.launch('opt_final_cpu','cpu',[]);self.wait([job]);cpu_exit=job['exit']
                except BaseException:
                    error=(error or '')+'\n'+traceback.format_exc()
                    cpu_exit=next((j['exit'] for j in self.jobs if j['kind']=='cpu'),None)
            budget=self.refresh()
            status=deliver(validation,formal,decision,error,cpu_exit,budget)
            self.capture_diff('actual_code_diff')
            write(self.controller_log/'exit.json',dict(exit=0 if status=='待审查' else 1,status=status,error=error,time=now()))
        return 0 if status=='待审查' else 1



class CDRequiredController(Controller):
    """One explicit B3_C FT handoff; never enters A/preparation/initialization."""
    def __init__(self,handoff):
        require_tmux();self.handoff=Path(handoff).resolve()
        if self.handoff!=(OUT/'controller_handoff_cd_required/request.json').resolve():raise ValueError('only the explicitly authorized CD-required handoff is allowed')
        self.lock=(OUT/'controller.lock').open('r+');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        request=read(self.handoff);manifest=read(OUT/'execution_manifest.json');current=read(OUT/'budget.json')
        self.jobs=validate_cd_handoff(request,current,manifest)
        old=Path('/proc')/str(request['old_pid'])
        if old.exists() and old.joinpath('stat').read_text().split(') ',1)[1].split()[0]!='Z':raise RuntimeError('old dispatcher still exists; no concurrent budget writer')
        self.window=subprocess.check_output(['tmux','display-message','-p','-t',os.environ['TMUX_PANE'],'#{window_id}'],text=True).strip()
        self.started_unix=manifest['wallclock_started_unix'];self.start=time.monotonic()-(time.time()-self.started_unix)
        self.controller_log=LOG/'controller_cd_required'
        claim_cd_handoff(self.handoff.with_name('claim.json'),dict(plan=PLAN,execution_policy_revision=EXECUTION_POLICY_REVISION,
            new_pid=os.getpid(),new_window=self.window,time=now(),adopted_job_identities=[job_identity(j) for j in self.jobs],
            wallclock_started_unix=self.started_unix,reset_budget=False,training_retry=False,entry_phase='B3_C_remaining_FT'))
        old_a=read(OUT/'selection/a_policy.json')
        effective=dict(plan=PLAN,execution_policy_revision=EXECUTION_POLICY_REVISION,time=now(),
            authorization=request['authorization'],authorization_path=str(self.handoff),
            C=dict(required=True,validation_gate=False),D=dict(required=True,FT_POLICY='MIX246'),
            D_FT_POLICY='MIX246',A_winner=old_a['winner'],historical_FT_POLICY=old_a['FT_POLICY'],historical_a_policy=str(OUT/'selection/a_policy.json'))
        with (OUT/'selection/cd_required.json').open('x') as f:json.dump(effective,f,ensure_ascii=False,indent=2,allow_nan=False)
        manifest.update(writer_pid=os.getpid(),controller_window=self.window,controller_handoff=str(self.handoff),command=sys.argv,
            execution_policy_revision=EXECUTION_POLICY_REVISION,log=str(self.controller_log),
            runtime_policy='CD_required_MIX246_single_pre_frozen_TEST_no_validation_gate')
        write(OUT/'execution_manifest.json',manifest)
        self.capture_diff('implementation_cd_required');self.refresh()

    def run(self):
        validation={};formal={};decision=None;error=None;cpu_exit=None;safe=True
        try:
            baseline=read(PARENT.parents[2]/'final_review/results.json')['validation']['B_FP']['macro']
            update_plan('执行中','B3_C剩余FT接续',f'{EXECUTION_POLICY_REVISION}显式交接生效，controller {self.window}；保留全部既有／在途单元，仅派发从未启动的B3_C FT；C固定一个、D固定MIX246，旧a_policy保留。')
            self.matrix(['B3_C'])
            for arm in ('A_TOP2','A_MIX246',*B_ARMS):validation[arm]=aggregate(arm)
            write(OUT/'selection/b_factorial.json',dict(validation={a:validation[a] for a in B_ARMS},effects=factorial(validation)))
            c=c_decision(validation)
            d=dict(trigger=True,required=True,FT_POLICY=effective_d_policy(),historical_a_policy=str(OUT/'selection/a_policy.json'),reason='user_requires_D_MIX246_independent_of_old_NONE')
            conditional=dict(C=c,D=d,PT_WINNER=None,execution_policy_revision=EXECUTION_POLICY_REVISION,effective_policy=str(OUT/'selection/cd_required.json'))
            write(OUT/'selection/conditional_decisions.json',conditional)
            update_plan('执行中','C',f"{EXECUTION_POLICY_REVISION}固定执行{c['arm']}，目标仅按完整validation提名；delta={c['delta']:.9f}不作为启动门槛，同step0和既定分层schedule。")
            self.pt(c['arm']);validation[c['arm']]=aggregate(c['arm'])
            ptwinner=choose({arm:validation[arm]['macro'] for arm in (*B_ARMS,c['arm'])},TIE_ORDER)
            conditional['PT_WINNER']=ptwinner;write(OUT/'selection/conditional_decisions.json',conditional)
            update_plan('执行中','D',f'固定D_INTEGRATE，parent={ptwinner} step5000＋公共属性head重新35FT，MIX246标量重置0，不读取旧NONE为跳过依据。')
            self.matrix(['D_INTEGRATE']);validation['D_INTEGRATE']=aggregate('D_INTEGRATE')
            decision=final_decision(validation,baseline,ptwinner);write(OUT/'selection/development_decision.json',decision)
            freeze_final(validation,decision,conditional)
            update_plan('执行中','唯一正式TEST',f"FINAL={decision['FINAL']}；C、D及直接reference已按{EXECUTION_POLICY_REVISION}纳入一次预冻结名单：{decision['models']}；不按TEST改选或追加。")
            self.matrix(decision['models'],True)
            for arm in decision['models']:formal[arm]=aggregate(arm,True)
            if all(arm in formal for arm in B_ARMS):write(OUT/'selection/formal_factorial.json',factorial(formal))
        except BaseException:
            error=traceback.format_exc();write(OUT/'failure.json',dict(error=error,time=now(),execution_policy_revision=EXECUTION_POLICY_REVISION,automatic_retry=False,automatic_resume=False))
            try:self.stop_owned()
            except BaseException:error+='\n'+traceback.format_exc();safe=False
        finally:
            if safe:
                try:
                    update_plan('执行中' if error is None else '需返修','唯一最终CPU suite','全部实际作业／唯一TEST已结束或安全异常收尾，仅执行原一次最终suite，≤300秒，失败不重测。')
                    j=self.launch('opt_final_cpu','cpu',[]);self.wait([j]);cpu_exit=j['exit']
                except BaseException:
                    error=(error or '')+'\n'+traceback.format_exc();cpu_exit=next((j['exit'] for j in self.jobs if j['kind']=='cpu'),None)
            current=self.refresh();status=deliver(validation,formal,decision,error,cpu_exit,current)
            self.capture_diff('actual_code_diff')
            write(self.controller_log/'exit.json',dict(exit=0 if status=='待审查' else 1,status=status,error=error,time=now(),execution_policy_revision=EXECUTION_POLICY_REVISION))
        return 0 if status=='待审查' else 1


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['controller','initialize','ft','test'])
    p.add_argument('--arm',choices=ARMS);p.add_argument('--task',choices=TASKS);p.add_argument('--fold',type=int,choices=range(5))
    p.add_argument('--handoff',type=Path,help='one explicitly authorized phase-specific controller handoff; never retries a worker')
    args=p.parse_args()
    torch.set_num_threads(2)
    def terminate(signum,frame):raise RuntimeError(f'controlled termination signal {signum}; no automatic retry')
    signal.signal(signal.SIGTERM,terminate)
    if args.action=='controller':
        cls=CDRequiredController if args.handoff is not None and args.handoff.resolve()==(OUT/'controller_handoff_cd_required/request.json').resolve() else Controller
        return cls(args.handoff).run()
    if args.handoff is not None:p.error('handoff is only allowed for the campaign controller')
    if args.action=='initialize':
        require_tmux()
        from src.modules.ua2d_opt import expanded_initialization
        expanded_initialization(OUT,COMMON,PT_HEADS,read(OUT/'inputs/context/vocab.json')['size']);return 0
    if args.arm is None or args.task is None or args.fold is None:p.error('unit identity required')
    unit(args.arm,args.task,args.fold,formal=args.action=='test');return 0

if __name__=='__main__':raise SystemExit(main())
