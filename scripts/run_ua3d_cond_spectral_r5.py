#!/usr/bin/env python3
"""Unique r5 FT-only controller; lifecycle events, strict budgets, no retry path."""
import os,sys,json,time,subprocess,traceback,statistics,signal,select
from pathlib import Path
from collections import deque
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training import ua3d_cond_spectral_r5 as t

def inventory():
    row=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid,memory.total,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(', ')
    if row[0]!=t.UUID:raise ValueError('physical GPU2 UUID mismatch')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    return dict(uuid=row[0],total_mib=float(row[1]),used_mib=float(row[2]),processes=[s for s in apps.splitlines() if s.startswith(t.UUID)])

def budget():
    ft=[t.read(p) for p in (t.OUT/'finetune').glob('*/*/fold*/consumption.json')]
    formal=[t.read(p) for p in (t.OUT/'formal').glob('*/*/fold*/consumption.json')]
    jobs=t.read(t.OUT/'jobs.json') if (t.OUT/'jobs.json').exists() else []
    ft_started=max(len(ft),sum(j['kind']=='ft' for j in jobs));test_started=max(len(formal),sum(j['kind']=='test' for j in jobs))
    value=dict(plan=t.PLAN,execution_revision='r5',science_recipe='r2',pt_starts=0,pt_updates=0,pt_completed=0,pt_microbatches=0,pt_presentations=0,available_source_pt_completed=10000,
        ft_starts=ft_started,ft_complete=sum(c['status']=='COMPLETE' for c in ft),ft_partial=ft_started-sum(c['status']=='COMPLETE' for c in ft),
        ft_epochs=sum(c.get('epochs_started',0) for c in ft),ft_completed_epochs=sum(c.get('completed_epochs',0) for c in ft),
        ft_updates=sum(c.get('attempted_updates',0) for c in ft),ft_completed_updates=sum(c.get('completed_updates',0) for c in ft),
        train_rows=sum(c.get('train_graphs',0) for c in ft),validation_rows=sum(c.get('validation_graphs',0) for c in ft),
        test_starts=test_started,test_complete=sum(c['status']=='COMPLETE' for c in formal),test_rows=sum(c.get('test_graphs',0) for c in formal),
        final_cpu=int((t.OUT/'final_cpu/started.json').exists()),pre_cpu=0,diagnostic=0,extra_model_forwards=0,new_initialization=0,stat_fit=0,geometry=0,surface=0,spectrum=0,
        model_reads={name:sum(c.get('model_reads',{}).get(name,0) for c in ft+formal) for name in ('historical_BASE_tensor_reads','current_campaign_PT_alias','random_head_alias','current_r5_BEST','parent_complete_BEST')})
    value['consumption_semantics']='latest persisted worker snapshots; active snapshots are lower bounds'
    unresolved=[]
    for job in jobs:
        if job['kind']!='ft' or job['exit'] is None:continue
        folder=t.unit_path(job['arm'],job['task'],job['fold']);path=folder/'consumption.json'
        if path.exists():
            state=t.read(path)
            if state['status']=='RUNNING':
                entry=t.read(t.SPLIT/f"{job['task']}.json")['folds'][job['fold']]
                import math
                upper=70*math.ceil(len(entry['train_indices'])/24)
                unresolved.append(dict(unit=[job['arm'],job['task'],job['fold']],known_attempted=state['attempted_updates'],attempt_upper=upper,reason='worker exited without terminal consumption snapshot'))
    value['unresolved_worker_tails']=unresolved;value['ft_attempted_interval']=[value['ft_updates'],value['ft_updates']+sum(x['attempt_upper']-x['known_attempted'] for x in unresolved)]
    alias_ready=int((t.OUT/'registry/canonical.json').exists())
    value['parent_complete_ft_aliases']=alias_ready;value['canonical_ft_complete']=alias_ready+value['ft_complete']
    value['total_updates']=value['ft_updates'];limits=t.read(ROOT/'configs/ua3d_cond_spectral_r5_gpu2.json')['limits'];value['limits']=limits
    for name in ('pt_starts','pt_updates','ft_starts','ft_epochs','ft_updates','total_updates','train_rows','validation_rows','test_rows','final_cpu','stat_fit'):
        if value[name]>limits[name]:raise RuntimeError('new-r5 budget exceeded: '+name)
    if value['test_starts']>210:raise RuntimeError('TEST start allowance exceeded')
    p2=t.read(t.PARENT/'shutdown_audit.json');p3=t.read(t.P3/'budget.json');p4=t.read(t.P4/'budget.json');p4cum=t.read(t.P4/'cumulative_budget.json')
    if p4cum['total_attempts']!=12288 or p4cum['actual_completed_interval']!=[12287,12288]:raise ValueError('retained r4 cumulative interval changed')
    cumulative=dict(parent_r2_snapshot=p2,parent_r3_budget=p3,parent_r4_budget=p4,parent_r4_cumulative=p4cum,current=value,
        pt_starts=2,pt_unit_quota=20000,pt_attempts=10858,pt_logged_completed=10857,pt_actual_completed_interval=[10857,10858],pt_consumed_microbatches=65148,pt_consumed_presentations=10944864,
        ft_starts=3+value['ft_starts'],ft_complete=1+value['ft_complete'],canonical_ft_complete=value['canonical_ft_complete'],ft_started_epochs=130+value['ft_epochs'],ft_completed_train_validation_epochs=128+value['ft_completed_epochs'],
        ft_attempts=1430+value['ft_updates'],ft_logged_completed=1430+value['ft_completed_updates'],
        total_attempts=12288+value['ft_updates'],logged_completed=12287+value['ft_completed_updates'],actual_completed_interval=[12287+value['ft_completed_updates'],12288+value['ft_updates']],
        train_rows=31720+value['train_rows'],validation_rows=7808+value['validation_rows'],
        parent_r2_prefetch_final='UNKNOWN',parent_r2_prefetch_last_snapshot=866538,parent_r2_not_consumed_lower=1674,parent_r3_prefetch_not_consumed=0,parent_unknown_tail_preserved=True,alias_is_not_new_PT=True,parent_complete_alias_updates_not_recharged=770,attempted_interval=[12288+value['ft_updates'],12288+value['ft_attempted_interval'][1]],additional_worker_tails=unresolved)
    cumulative['actual_completed_interval'][1]=max(cumulative['actual_completed_interval'][1],12288+value['ft_attempted_interval'][1])
    if cumulative['attempted_interval'][1]>347518 or cumulative['ft_attempts']>336660:raise RuntimeError('four-generation cumulative allowance exceeded')
    t.write(t.OUT/'budget.json',value);t.write(t.OUT/'cumulative_budget.json',cumulative);t.write(t.OUT/'consumption.json',value)
    return value

def first_failure(error,**detail):
    path=t.OUT/'first_failure.json'
    if not path.exists():t.exclusive(path,dict(error=error,time=t.now(),automatic_retry=False,**detail))

class Controller:
    def __init__(self):
        self.jobs=[];self.live=[];self.finished_buffer=[];self.boundary_due=False;self.cpu_exit=None;self.memory=t.MemoryMonitor();self.next_check=0.;self.priors={};self.prior_peaks={};self.observed_peaks={};self.ratios={};self.max_ratio=1.;self.serial_peak_ratio=1.;self.surface_extra=0.;self.parallel_samples=[]
        for task in t.TASKS:
            for fold in range(5):
                h=t.read(t.base_path(task,fold)/'history.json')
                self.priors[(task,fold)]=sum(x['seconds'] for x in h);self.prior_peaks[(task,fold)]=max(x['gpu_peak_bytes'] for x in h)
    def check(self,force=False):
        t.check_clock()
        if any(j['kind']=='prepare' for _,_,j in self.live):t.check_clock(True)
        if self.memory.has_pending() or self.boundary_due or force or time.monotonic()>=self.next_check:
            result=self.memory.sample(force=force or self.boundary_due)
            if result and result['coverage_result']=='COMPLETE':self.boundary_due=False
        # EDF pending work has priority over large budget/history reloads.
        if not self.memory.has_pending() and (force or time.monotonic()>=self.next_check):
            budget();self.next_check=time.monotonic()+5
    def ensure_complete(self):
        fresh=True
        while True:
            self.reap();t.check_clock()
            if any(j['kind']=='prepare' for _,_,j in self.live):t.check_clock(True)
            result=self.memory.sample(force=fresh or self.boundary_due);fresh=False
            if result and result.get('dispatch_allowed'):
                self.boundary_due=False;return result
            self.wait_event(coverage_wait=True)
    def launch(self,kind,arm=None,task=None,fold=None):
        self.ensure_complete()
        if kind not in ('prepare','ft','test','cpu'):raise ValueError('r5 has no PT/diagnostic entry')
        name='-'.join(str(x) for x in (kind,arm,task,fold) if x is not None);folder=t.LOG/name;folder.mkdir(parents=True,exist_ok=False)
        argv=[sys.executable,str(Path(__file__).resolve()),'worker',kind]
        if kind in ('ft','test'):argv += [arm,task,str(fold)]
        if kind=='prepare':argv=[sys.executable,str(ROOT/'scripts/prepare_ua3d_cond_spectral_r5.py')]
        if kind=='cpu':argv=[sys.executable,'-m','pytest','-q','tests/test_ua3d_cond_spectral_r5.py']
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='' if kind in ('prepare','cpu') else '2',CUDA_DEVICE_ORDER='PCI_BUS_ID',WORLD_SIZE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',UA3M_PHYSICAL_GPU_UUID=t.UUID)
        log=(folder/'stdout.log').open('x');p=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        identity=t.pid_identity(p.pid);job=dict(kind=kind,arm=arm,task=task,fold=fold,pid=p.pid,start_ticks=identity['start_ticks'] if identity else None,cwd=str(ROOT),argv=argv,started_unix=time.time(),exit=None,log=str(folder/'stdout.log'))
        self.jobs.append(job);self.live.append((p,log,job));t.write(t.OUT/'jobs.json',self.jobs)
        if identity:self.memory.register(identity,kind)
        elif p.poll() is None:raise RuntimeError('TELEMETRY_ERROR: launched living worker identity unavailable')
        self.reap();self.memory.sample(force=True)
        return job
    def reap(self):
        # Actual Popen exits are reconciled and persisted before resource sampling.
        finished=[];failures=[]
        for entry in list(self.live):
            p,log,job=entry;code=p.poll()
            if code is None:continue
            code=p.wait();log.close();job.update(exit=code,ended_unix=time.time());self.live.remove(entry)
            t.write(t.OUT/'jobs.json',self.jobs);self.memory.complete_direct(job,code);finished.append(job)
            if job['kind']=='cpu':self.cpu_exit=code
            if code:failures.append(job)
        if failures:
            first=failures[0];first_failure('real worker nonzero exit',job=first,all_failures=failures)
            raise RuntimeError(f"real worker failure {first['kind']} {first['arm']} {first['task']} {first['fold']} exit={first['exit']}")
        if finished:self.boundary_due=True;self.finished_buffer.extend(finished)
        return finished
    def poll(self):
        self.reap();self.check();done=self.finished_buffer;self.finished_buffer=[];return done
    def wait_event(self,coverage_wait=False):
        if not self.live and not self.memory.has_pending() and not coverage_wait:return
        until_resource=max(0.,self.next_check-time.monotonic())
        timeout=max(0.,min(5.,until_resource,t.read(t.OUT/'authorization/clock.json')['deadline_unix']-time.time()))
        if self.memory.has_pending() or coverage_wait:timeout=min(.250,self.memory.service_timeout())
        poll=select.poll();count=0
        for _,_,job in self.live:
            row=self.memory.active.get(f"{job['pid']}:{job['start_ticks']}")
            if row and row.get('pidfd') is not None:poll.register(row['pidfd'],select.POLLIN|select.POLLHUP|select.POLLERR);count+=1
        if count:poll.poll(int(timeout*1000))
        elif timeout:time.sleep(min(.250,timeout))
    def wait(self):
        while self.live or self.memory.has_pending() or self.boundary_due:
            self.poll()
            if self.live or self.memory.has_pending():self.wait_event()
    def wait_gpu(self):
        next_gpu=0.
        while True:
            self.reap();self.check()
            if time.monotonic()>=next_gpu:
                inv=inventory();next_gpu=time.monotonic()+5
                t.write(t.OUT/'gates/gpu_wait.json',dict(time_unix=time.time(),**inv,policy='unknown GPU2 occupation waits; no unrelated signals'))
                if not inv['processes']:self.ensure_complete();return inv
            if self.memory.has_pending():self.wait_event(coverage_wait=True)
            else:time.sleep(min(1.,max(.01,next_gpu-time.monotonic())))
    def estimate_peak(self,task,fold):
        return int(max(self.prior_peaks[(task,fold)]*max(1.,self.serial_peak_ratio)+self.surface_extra,self.observed_peaks.get((task,fold),0))+1024**3)
    def observe_ft(self,job):
        if job['kind']!='ft':return
        h=t.read(t.unit_path(job['arm'],job['task'],job['fold'])/'history.json');key=(job['task'],job['fold']);peak=max(x['gpu_peak_bytes'] for x in h)
        self.observed_peaks[key]=max(self.observed_peaks.get(key,0),peak)
        wall=job['ended_unix']-job['started_unix'];ratio=wall/self.priors[key]
        self.ratios[key]=max(self.ratios.get(key,1.),ratio);self.max_ratio=max(self.max_ratio,ratio)
        return h,peak
    def eta(self,pending,concurrency=1):
        # Per-unit wall ratios include contention/startup/validation/teardown.
        # Parallelism is only credited after observed normal queue throughput.
        efficiency=1.
        if self.parallel_samples:
            window=self.parallel_samples[-1]
            efficiency=max(1.,min(float(concurrency),window['observed_mean_concurrency']))
        work=sum(self.priors[(task,fold)]*max(self.max_ratio,self.ratios.get((task,fold),1.)) for arm,task,fold in pending)
        formal=sum(max(5.,seconds/70)*6 for seconds in self.priors.values())*max(1.,self.max_ratio)
        eta=1.5*(work/efficiency+formal)+7200
        record=dict(time_unix=time.time(),pending=len(pending),concurrency_cap=concurrency,effective_parallelism=efficiency,max_observed_wall_ratio=self.max_ratio,work_seconds=work,formal_seconds=formal,remaining_seconds=eta,projected_finish_unix=time.time()+eta,includes=['load','validation','save','scheduler','monitoring','TEST','final CPU','delivery'])
        t.write(t.OUT/'gates/remaining_eta.json',record)
        if record['projected_finish_unix']>=t.read(t.OUT/'authorization/clock.json')['deadline_unix']:raise TimeoutError('ETA_STOP: complete remaining campaign cannot fit original deadline')
    def admit(self,unit,available,live_units):
        a,task,fold=unit;peaks=[self.estimate_peak(tk,fd) for _,tk,fd in live_units]+[self.estimate_peak(task,fold)]
        inv=inventory();free=(inv['total_mib']-inv['used_mib'])*1024**2
        allowed=sum(peaks)<=.8*available and peaks[-1]<=.8*free
        t.write(t.OUT/'gates/memory_admission.json',dict(time_unix=time.time(),candidate=unit,live_units=live_units,per_unit_peaks_including_1GiB_context=peaks,baseline_available_bytes=available,current_free_bytes=free,limit_fraction=.8,admitted=allowed,prior_source='matched historical task/fold full-run peak plus actual r4 peaks/surface overhead'))
        return allowed
    def queue(self,kind,units,available,cap):
        pending=deque(units);completed=0;started=time.time();job_wall=0.;current_cap=cap;max_live=0
        while pending or self.live or self.memory.has_pending() or self.boundary_due:
            done=self.poll()
            for job in done:
                if job['kind']!=kind:continue
                completed+=1;job_wall+=job['ended_unix']-job['started_unix']
                if kind=='ft':self.observe_ft(job)
            if completed and max_live>=2:
                self.parallel_samples.append(dict(time_unix=time.time(),completed=completed,elapsed=time.time()-started,observed_mean_concurrency=min(float(max_live),job_wall/max(time.time()-started,1e-9))))
            if kind=='ft' and completed>=2 and max_live>=2:current_cap=4
            while pending and len(self.live)<current_cap:
                live_units=[(j['arm'],j['task'],j['fold']) for _,_,j in self.live if j['kind']==kind]
                if not self.admit(pending[0],available,live_units):
                    if not self.live:raise RuntimeError('single-unit memory admission cannot fit unchanged batch')
                    break
                a,task,fold=pending.popleft();self.launch(kind,a,task,fold);max_live=max(max_live,len(self.live))
            if kind=='ft' and done:
                remaining=list(pending)+[(j['arm'],j['task'],j['fold']) for _,_,j in self.live if j['kind']==kind]
                self.eta(remaining,current_cap)
            if self.live or self.memory.has_pending():self.wait_event()
    def stop(self):
        # Signal only registered current identities; never waitpid DataLoader children.
        for _,_,job in list(self.live):
            if job.get('start_ticks') is not None:self.memory.signal_identity(f"{job['pid']}:{job['start_ticks']}",signal.SIGTERM)
        for p,log,job in list(self.live):
            try:code=p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if job.get('start_ticks') is not None:self.memory.signal_identity(f"{job['pid']}:{job['start_ticks']}",signal.SIGKILL)
                code=p.wait()
            log.close();job.update(exit=code,ended_unix=time.time(),controller_stopped=True);self.memory.complete_direct(job,code)
        self.live=[];t.write(t.OUT/'jobs.json',self.jobs)
        for key,row in list(self.memory.active.items()):
            if row['role'] not in ('bootstrap','outer','controller'):self.memory.signal_identity(key,signal.SIGTERM)
        try:self.memory.sample(force=True)
        except Exception:pass
    def run(self):
        t.update_plan('RUNNING','prepare','r5明确授权；PT新增0；固定采样C/五秒pending与唯一完整alias。');t.check_clock(True)
        self.launch('prepare');self.wait();t.check_clock(True)
        all_units=[(a,task,fold) for a in t.ARMS for task in t.TASKS for fold in range(5)]
        alias=t.read(t.OUT/'registry/canonical.json');parent=alias['units']['C_FRESH/eps/0'];parent_history=t.read(Path(parent['root'])/'history.json')
        parent_peak=max(x['gpu_peak_bytes'] for x in parent_history)
        speed={'C_FRESH':alias['parent_speed_seconds_per_batch']};eps_peaks={'C_FRESH':parent_peak}
        self.serial_peak_ratio=max(1.,parent_peak/self.prior_peaks[('eps',0)])
        new_units=[u for u in all_units if not t.is_parent_alias(*u)]
        self.eta(new_units,1);t.update_plan('RUNNING','FT_first_five','父完整C_FRESH canonical alias已资格；五套新表面EPS/fold0串行完整70epoch。')
        for arm in t.ARMS[1:]:
            self.wait_gpu();job=self.launch('ft',arm,'eps',0);self.wait();h,peak=self.observe_ft(job)
            stable=[x['training_seconds']/x['batches'] for x in h if x['stage']=='joint' and x['stage_epoch']>2]
            speed[arm]=statistics.median(stable);eps_peaks[arm]=peak
            ratios={a:v/speed['C_FRESH'] for a,v in speed.items()}
            t.write(t.OUT/'gates/ft_speed.json',dict(seconds_per_batch=speed,ratios=ratios,maximum=2.,control_source=parent['root'],control_is_complete_parent_alias=True,normal_units=True,excluded_joint_epochs=2))
            if any(ratio>2. for ratio in ratios.values()):raise RuntimeError('FT surface speed exceeds2.0x registered parent control')
            self.surface_extra=max(self.surface_extra,peak-parent_peak)
            remaining=[u for u in new_units if not (u[1:]==('eps',0) and u[0] in speed)];self.eta(remaining,1)
        inv=self.wait_gpu();available=(inv['total_mib']-inv['used_mib'])*1024**2
        pending=[u for u in all_units if u[1:]!=('eps',0)]
        t.update_plan('RUNNING','FT','父完整alias及五新正常单元门禁通过；无pending完整采样后按1→2→4准入。')
        self.queue('ft',pending,available,2)
        self.ensure_complete();t.freeze();t.update_plan('RUNNING','formal_TEST','全部210套FT完成；一次冻结后统一210单元正式TEST。')
        self.parallel_samples=[];self.queue('test',all_units,available,4)
        self.ensure_complete();t.report_results();t.exclusive(t.OUT/'final_cpu/started.json',dict(attempt=1,started_unix=time.time()))
        t.update_plan('RUNNING','final_CPU','正式210单元结束，唯一最终CPU suite。');self.launch('cpu');self.wait()
        budget();t.update_plan('COMPLETE_PENDING_REVIEW','delivery','完整FT/TEST/唯一CPU完成；independent_review=PENDING，BASE不变。')

def closeout(controller,error=None):
    b=budget();status='STOPPED' if error else 'COMPLETE_PENDING_REVIEW'
    if error and 'EXPIRED' in error:status='EXPIRED'
    elif error and 'ETA_STOP' in error:status='ETA_STOP'
    value=dict(status=status,plan=t.PLAN,execution_revision='r5',science_recipe='r2',budget=b,cumulative=t.read(t.OUT/'cumulative_budget.json'),clock=t.read(t.OUT/'authorization/clock.json'),error=error,formal_status='COMPLETE' if b['test_complete']==210 else '未完成/不可比较',FINAL=None,macro=None,delta_BASE=None,delta_C_FRESH=None,independent_review='PENDING',final_cpu_exit=controller.cpu_exit,real_exits=dict(controller=1 if error else 0,outer=None,bootstrap=None),retries=0)
    if b['test_complete']==210 and (t.OUT/'final_review/results.json').exists():
        result=t.read(t.OUT/'final_review/results.json');value.update({k:result[k] for k in ('FINAL','macro','delta_BASE','delta_C_FRESH','target_met','performance_pass')})
    if error:first_failure(error);t.write(t.OUT/'failure_closeout.json',value);t.update_plan(status,'closeout',error.splitlines()[-1])
    t.write(t.OUT/'delivery.json',value);t.write(t.OUT/'shutdown_audit.json',value)
    lines=[f'# {t.PLAN}/r5',f'状态：{status}；independent_review=PENDING；正式评价{value["formal_status"]}。',f'新增PT=0，允许来源PT完成10000；新增FT启动{b["ft_starts"]}、完整新FT{b["ft_complete"]}/209、父完整alias{b["parent_complete_ft_aliases"]}、canonical完整{b["canonical_ft_complete"]}/210、部分/中断{b["ft_partial"]}；开始epochs {b["ft_epochs"]}、完整训练验证epochs {b["ft_completed_epochs"]}；尝试/记录完成更新{b["ft_updates"]}/{b["ft_completed_updates"]}。',f'TEST完整{b["test_complete"]}/210、{b["test_rows"]}/35250行；最终CPU {b["final_cpu"]}次、exit={controller.cpu_exit}；前置CPU/额外前向0。',f'父累计12288尝试/[12287,12288]完成区间及预取未知尾部保留；当前累计见cumulative_budget.json。',f'原总截止不变；repair_diff.md/source_aliases.json/memory.jsonl/lifecycle.jsonl/jobs.json/exits/记录修复、来源、生命周期和真实exit。',f'停止原因：{error}' if error else '完整七任务/论文对照见final_review/results.json及comparison.md；数值和审查分开，BASE不替换。']
    (t.OUT/'delivery_summary.md').write_text('\n\n'.join(lines)+'\n')
    with (ROOT/'PROJECT_HISTORY.md').open('a') as f:f.write('\n\n## UA3D-COND-SPECTRAL-20261104-01/r5（待审查）\n\n'+'\n\n'.join(lines[1:5])+f'\n\n证据：{t.OUT.relative_to(ROOT)}/delivery_summary.md、delivery.json、cumulative_budget.json。历史BASE=F_PD_N1_F0不变。\n')

def main():
    if len(sys.argv)>1 and sys.argv[1]=='worker':
        if sys.argv[2] not in ('ft','test'):raise ValueError('r5 worker whitelist excludes PT/diagnostics')
        t.unit(sys.argv[3],sys.argv[4],int(sys.argv[5]),sys.argv[2]=='test');return
    t.require_tmux()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('controller must hide GPU')
    controller=None;error=None
    try:
        controller=Controller();controller.run()
    except BaseException:
        error=traceback.format_exc();first_failure(error)
        if controller is not None:controller.stop()
    if controller is None:
        from types import SimpleNamespace
        closeout(SimpleNamespace(cpu_exit=None),error)
    else:closeout(controller,error);controller.memory.close()
    if error:print(error,flush=True);raise SystemExit(1)
if __name__=='__main__':main()
