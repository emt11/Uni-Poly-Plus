#!/usr/bin/env python3
"""Independent fixed-budget stabilization controller; never resumes or repairs."""
import os,sys,json,time,subprocess,traceback,statistics,signal,select
from pathlib import Path
from collections import deque
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training import ua3d_base_stabilize_r1 as t

def inventory():
    row=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid,memory.total,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(', ')
    if row[0]!=t.UUID:raise ValueError('physical GPU2 UUID mismatch')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    return dict(uuid=row[0],total_mib=float(row[1]),used_mib=float(row[2]),processes=[s for s in apps.splitlines() if s.startswith(t.UUID)])


def first_failure(error,**detail):
    path=t.OUT/'first_failure.json'
    if not path.exists():t.exclusive(path,dict(error=error,time=t.now(),automatic_retry=False,**detail))


class Controller:
    def __init__(self):
        self.phase='prepare';self.completed_ft=set();self.compared_started=set();self.ready_pairs=deque();self.jobs=[];self.live=[];self.finished_buffer=[];self.boundary_due=False;self.cpu_exit=None;self.memory=t.MemoryMonitor();self.next_check=0.;self.priors={};self.prior_peaks={};self.observed_peaks={};self.ratios={};self.max_ratio=1.;self.serial_peak_ratio=1.;self.base_extra=0.;self.parallel_samples=[]
        for task in t.TASKS:
            for fold in range(5):
                h=t.read(t.D/'finetune/DET_BASE/S42_A'/task/f'fold{fold}'/'history.json')
                self.priors[(task,fold)]=sum(x['seconds'] for x in h);self.prior_peaks[(task,fold)]=max(x['gpu_peak_bytes'] for x in h)
    def check(self,force=False):
        t.check_clock();t.check_space()
        if self.phase=='PT' and (t.OUT/'authorization/pt_clock.json').exists():
            clock=t.read(t.OUT/'authorization/pt_clock.json')
            if time.time()>=clock['deadline_unix']:raise TimeoutError('EXPIRED: combined PT stage18h')
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
    def launch(self,kind,arm=None,task=None,fold=None,estimator=None):
        self.ensure_complete();t.check_space()
        if kind not in ('prepare','pt','ft','test','compare_pt','compare_ft','cpu'):raise ValueError('unregistered work kind')
        name='-'.join(str(x) for x in (kind,arm,task,fold,estimator) if x is not None);folder=t.LOG/name;folder.mkdir(parents=True,exist_ok=False)
        argv=[sys.executable,str(Path(__file__).resolve()),'worker',kind]
        if kind=='prepare':argv=[sys.executable,str(ROOT/'scripts/prepare_ua3d_base_stabilize_r1.py')]
        elif kind=='pt':argv=[sys.executable,str(ROOT/'scripts/pretrain_ua3d_base_stabilize_r1.py'),arm]
        elif kind in ('ft','test'):argv += [arm,task,str(fold)]+([estimator] if estimator else [])
        elif kind=='compare_ft':argv += [task,str(fold)]
        elif kind=='cpu':argv=[sys.executable,'-m','pytest','-q','tests/test_ua3d_base_stabilize_r1.py']
        if kind=='pt':
            path=t.OUT/'authorization/pt_clock.json';now=time.time()
            clock=t.read(path) if path.exists() else dict(first_dispatch_unix=now,deadline_unix=min(now+18*3600,t.read(t.OUT/'authorization/clock.json')['deadline_unix']),units={})
            if arm in clock['units']:raise RuntimeError('PT already dispatched; no retries')
            clock['units'][arm]=dict(started_unix=now,deadline_unix=min(now+12*3600,clock['deadline_unix']));t.write(path,clock)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2' if kind in ('pt','ft','test') else '',CUDA_DEVICE_ORDER='PCI_BUS_ID',WORLD_SIZE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='20261028',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',UA3M_PHYSICAL_GPU_UUID=t.UUID)
        log=(folder/'stdout.log').open('x');p=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        ident=t.pid_identity(p.pid);job=dict(kind=kind,arm=arm,task=task,fold=fold,estimator=estimator,pid=p.pid,start_ticks=ident['start_ticks'] if ident else None,cwd=str(ROOT),argv=argv,started_unix=time.time(),exit=None,log=str(folder/'stdout.log'))
        self.jobs.append(job);self.live.append((p,log,job));t.write(t.OUT/'jobs.json',self.jobs)
        if kind=='cpu':t.write(t.OUT/'final_cpu/command.json',job)
        if ident:self.memory.register(ident,kind)
        elif p.poll() is None:raise RuntimeError('launched living worker identity unavailable')
        self.reap();self.memory.sample(force=True);return job
    def reap(self):
        # Actual Popen exits are reconciled and persisted before resource sampling.
        finished=[];failures=[]
        for entry in list(self.live):
            p,log,job=entry;code=p.poll()
            if code is None:continue
            code=p.wait();log.close();job.update(exit=code,ended_unix=time.time());self.live.remove(entry)
            t.write(t.OUT/'jobs.json',self.jobs);self.memory.complete_direct(job,code);finished.append(job)
            if job['kind']=='cpu':
                self.cpu_exit=code;t.write(t.OUT/'final_cpu/exit.json',dict(exit=code,pid=p.pid,ended_unix=job['ended_unix'],log=job['log']))
            if code:failures.append(job)
            elif job['kind']=='ft':
                key=(job['arm'],job['task'],job['fold']);self.completed_ft.add(key)
                pair=(job['task'],job['fold'])
                if all((g,*pair) in self.completed_ft for g in ('S42_A','S42_B')) and pair not in self.compared_started:self.ready_pairs.append(pair);self.compared_started.add(pair)
        if failures:
            first=failures[0];first_failure('real worker nonzero exit',job=first,all_failures=failures)
            raise RuntimeError(f"real worker failure {first['kind']} {first['arm']} {first['task']} {first['fold']} exit={first['exit']}")
        if finished:self.boundary_due=True;self.finished_buffer.extend(finished)
        return finished
    def poll(self):
        self.reap();self.check()
        if self.ready_pairs and not any(j['kind']=='compare_ft' for _,_,j in self.live):
            task,fold=self.ready_pairs.popleft();self.launch('compare_ft',task=task,fold=fold)
        done=self.finished_buffer;self.finished_buffer=[];return done
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
        return int(max(self.prior_peaks[(task,fold)]*max(1.,self.serial_peak_ratio)+self.base_extra,self.observed_peaks.get((task,fold),0))+1024**3)
    def observe_ft(self,job):
        if job['kind']!='ft':return
        h=t.read(t.unit_path(job['arm'],job['task'],job['fold'])/'history.json');key=(job['task'],job['fold']);peak=max(x['gpu_peak_bytes'] for x in h)
        self.observed_peaks[key]=max(self.observed_peaks.get(key,0),peak)
        wall=job['ended_unix']-job['started_unix'];ratio=wall/self.priors[key]
        self.ratios[key]=max(self.ratios.get(key,1.),ratio);self.max_ratio=max(self.max_ratio,ratio)
        return h,peak
    def eta(self,pending,concurrency=1):
        effective=1.
        if self.parallel_samples:effective=max(1.,min(float(concurrency),self.parallel_samples[-1]['observed_mean_concurrency']))
        normal_ft=sum(self.priors[(u[1],u[2])]*max(self.max_ratio,self.ratios.get((u[1],u[2]),1.)) for u in pending)
        swa_overhead=sum(max(1.,self.priors[(u[1],u[2])]/70)+1. for u in pending);ft=normal_ft+swa_overhead
        formal=sum(max(5.,p/70)*6 for p in self.priors.values())*max(1.,self.max_ratio)
        comparison=1800.;remaining_pt=0.
        for run in ('PT_R0','PT_R1'):
            p=t.OUT/'pretrain'/run/'consumption.json';c=t.read(p) if p.exists() else {};speed=t.OUT/'pretrain'/run/'speed.json'
            per=t.read(speed)['seconds_per_update'] if speed.exists() else 11424.682/10000
            remaining_pt+=(10000-c.get('completed_updates',0))*per
        downstream=ft/effective+formal+comparison;eta=1.5*(remaining_pt+downstream)+7200
        record=dict(time_unix=time.time(),remaining_seconds=eta,projected_finish_unix=time.time()+eta,remaining_PT_seconds=remaining_pt,downstream_serial_seconds=ft+formal+comparison,work_seconds=ft,formal_seconds=formal,effective_parallelism=effective,comparison_seconds=comparison,swa_overhead_seconds=swa_overhead,includes=['data','validation','SWA validation','I/O','monitoring','exit waits','TEST','final CPU','delivery'])
        t.write(t.OUT/'gates/remaining_eta.json',record)
        if record['projected_finish_unix']>=t.read(t.OUT/'authorization/clock.json')['deadline_unix']:raise TimeoutError('ETA_STOP: fixed full stabilization matrix cannot fit')
    def admit(self,unit,available,live_units):
        a,task,fold=unit[:3];peaks=[self.estimate_peak(tk,fd) for _,tk,fd in live_units]+[self.estimate_peak(task,fold)]
        inv=inventory();owned={j['pid'] for _,_,j in self.live if j['kind'] in ('pt','ft','test')}
        unknown=[row for row in inv['processes'] if int(row.split(',')[1].strip()) not in owned]
        self.unknown_gpu_wait=bool(unknown)
        if unknown:
            t.write(t.OUT/'gates/gpu_wait.json',dict(time_unix=time.time(),unknown=unknown,policy='wait without new dispatch; never signal unknown GPU2 owners'))
            return False
        free=(inv['total_mib']-inv['used_mib'])*1024**2
        allowed=sum(peaks)<=.8*available and peaks[-1]<=.8*free
        t.write(t.OUT/'gates/memory_admission.json',dict(time_unix=time.time(),candidate=unit,live_units=live_units,per_unit_peaks_including_1GiB_context=peaks,baseline_available_bytes=available,current_free_bytes=free,limit_fraction=.8,admitted=allowed,prior_source='matched historical DET task/fold normal peak plus actual new BASE peaks and context'))
        return allowed
    def queue(self,kind,units,available,cap):
        pending=deque(units);completed=0;started=time.time();job_wall=0.;current_cap=cap;max_live=0;next_eta=0.
        while pending or self.live or self.memory.has_pending() or self.boundary_due:
            done=self.poll()
            for job in done:
                if job['kind']!=kind:continue
                completed+=1;job_wall+=job['ended_unix']-job['started_unix']
                if kind=='ft':self.observe_ft(job)
            if completed and max_live>=2:
                self.parallel_samples.append(dict(time_unix=time.time(),completed=completed,elapsed=time.time()-started,observed_mean_concurrency=min(float(max_live),job_wall/max(time.time()-started,1e-9))))
            if kind=='ft' and completed>=2 and max_live>=2:current_cap=4
            while pending and sum(j['kind']==kind for _,_,j in self.live)<current_cap:
                live_units=[(j['arm'],j['task'],j['fold']) for _,_,j in self.live if j['kind']==kind]
                if not self.admit(pending[0],available,live_units):
                    if not self.live and not self.unknown_gpu_wait:raise RuntimeError('single-unit memory admission cannot fit unchanged batch')
                    break
                u=pending.popleft();a,task,fold=u[:3];self.launch(kind,a,task,fold,u[3] if len(u)==4 else None);max_live=max(max_live,sum(j['kind']==kind for _,_,j in self.live))
            if kind=='ft' and (done or time.monotonic()>=next_eta):
                remaining=list(pending)+[(j['arm'],j['task'],j['fold']) for _,_,j in self.live if j['kind']==kind]
                self.eta(remaining,current_cap);next_eta=time.monotonic()+5.
            if self.live or self.memory.has_pending():self.wait_event()
            elif pending:time.sleep(.250)
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
        t.update_plan('RUNNING','prepare','一次全新随机factory/step0/35头；旧模型张量禁止读取。');self.launch('prepare');self.wait();t.check_clock(True)
        priority=('eps','nc','xc','eea','egb','egc','ei');units=[(g,task,fold) for task in priority for fold in range(5) for g in t.GROUPS]
        self.eta(units,1);self.phase='PT';t.update_plan('RUNNING','PT','两个独立seed42完整10K，从同一新step0串行；DET不可降级。')
        for run in ('PT_R0','PT_R1'):
            inv=self.wait_gpu();available=(inv['total_mib']-inv['used_mib'])*1024**2
            prior=t.read(t.ROOT/'results/ua3d_cond_spectral_20261104/r2/repairs/memory_r3/pretrain/BASE_FRESH/timing.json')['gpu_peak_bytes']+1024**3
            t.write(t.OUT/'gates/PT_memory_admission.json',dict(run=run,prior_peak_with_context=prior,available_bytes=available,limit_fraction=.8,admitted=prior<=.8*available,source='normal existing nonDET PT memory prior; no tensor read'))
            if prior>.8*available:raise RuntimeError('MEMORY_ADMISSION_STOP: unchanged PT microbatch cannot fit80percent')
            self.launch('pt',run);self.wait();self.eta(units,1)
        self.phase='PT_compare';self.launch('compare_pt');self.wait()
        self.phase='FT';t.update_plan('RUNNING','FT_first_pair','先串行S42_A/B EPS/fold0；正常完整单元和直接产物比较。')
        for g in ('S42_A','S42_B'):
            self.wait_gpu();job=self.launch('ft',g,'eps',0);self.wait();self.observe_ft(job)
        if not t.read(t.OUT/'comparisons/FT/eps/fold0.json')['equal']:raise RuntimeError('first normal pair not equal')
        inv=self.wait_gpu();available=(inv['total_mib']-inv['used_mib'])*1024**2
        remaining=[u for u in units if u not in [('S42_A','eps',0),('S42_B','eps',0)]]
        t.update_plan('RUNNING','FT','首对严格通过；其余138套，优先EPS/Nc/Xc，2→4准入、及时比较A/B。');self.queue('ft',remaining,available,2)
        self.ensure_complete();budget();t.freeze();self.phase='TEST';t.update_plan('RUNNING','formal_TEST','固定SWA/primary S42_A一次冻结；BEST/SWA×三个canonical组。')
        formal=[(g,task,fold,e) for g in t.CANONICAL for e in ('BEST','SWA') for task in t.TASKS for fold in range(5)]
        self.parallel_samples=[];self.queue('test',formal,available,4);self.ensure_complete();budget();t.report_results(False)
        t.exclusive(t.OUT/'final_cpu/started.json',dict(attempt=1,started_unix=time.time()));budget();self.phase='CPU';t.update_plan('RUNNING','final_CPU','完整210正式TEST之后唯一CPU suite。');self.launch('cpu');self.wait();budget();t.report_results(True)
        t.update_plan('COMPLETE_PENDING_REVIEW','delivery','完整执行；稳定性/质量/性能分开，independent_review=PENDING，BASE不替换。')

def budget():
    jobs=t.read(t.OUT/'jobs.json') if (t.OUT/'jobs.json').exists() else []
    def values(pattern):return [t.read(p) for p in t.OUT.glob(pattern)]
    pt=values('pretrain/*/consumption.json');ft=values('finetune/*/*/fold*/consumption.json');tests=values('formal/*/*/fold*/*/consumption.json');prepare=t.read(t.OUT/'prepare_consumption.json') if (t.OUT/'prepare_consumption.json').exists() else {}
    limits=t.spec()['limits'];s=lambda items,k:sum(x.get(k,0) for x in items)
    count=lambda kind:sum(j['kind']==kind for j in jobs)
    v=dict(prepare=count('prepare'),initialization=prepare.get('initialization',0),heads=prepare.get('heads',0),heads_complete=len(list((t.OUT/'initialization/heads').glob('*/fold*.pt'))),pt_starts=count('pt'),pt_complete=sum(x['status']=='COMPLETE' for x in pt),pt_updates=s(pt,'attempted_updates'),pt_completed_updates=s(pt,'completed_updates'),pt_microbatches=s(pt,'pt_microbatches'),pt_presentations=s(pt,'pt_presentations'),pt_prefetched_not_consumed=s(pt,'prefetched_not_consumed'),ft_starts=count('ft'),ft_complete=sum(x['status']=='COMPLETE' for x in ft),ft_epochs=s(ft,'epochs_started'),ft_completed_epochs=s(ft,'completed_epochs'),ft_updates=s(ft,'attempted_updates'),ft_completed_updates=s(ft,'completed_updates'),train_rows=s(ft,'train_graphs'),validation_epochs=s(ft,'validation_started'),validation_rows=s(ft,'validation_graphs'),swa_validation=s(ft,'swa_validation_started'),swa_validation_rows=s(ft,'swa_validation_graphs'),pt_pairs=count('compare_pt'),ft_pairs=count('compare_ft'),ft_pairs_passed=sum(t.read(p).get('equal') is True for p in (t.OUT/'comparisons/FT').glob('*/fold*.json')),freeze=int((t.OUT/'selection/final_frozen.json').exists()),test_starts=count('test'),test_complete=sum(x['status']=='COMPLETE' for x in tests),test_rows=s(tests,'test_graphs'),cpu_suites=int((t.OUT/'final_cpu/started.json').exists()),pre_cpu=0,diagnostics=0,extra_forwards=0,historical_tensor_reads=0,retries=0)
    v['total_updates']=v['pt_updates']+v['ft_updates'];v['logged_completed_updates']=v['pt_completed_updates']+v['ft_completed_updates'];v['reservation']=limits;v['actual_completed_interval']=[v['logged_completed_updates'],v['total_updates']];v['unknown_tails']=[]
    for j in jobs:
        if j['kind'] not in ('pt','ft') or j['exit'] is None:continue
        p=(t.OUT/'pretrain'/j['arm'] if j['kind']=='pt' else t.unit_path(j['arm'],j['task'],j['fold']))/'consumption.json'
        state=t.read(p) if p.exists() else {}
        if state.get('status') in ('COMPLETE','FAILED'):continue
        upper=10000 if j['kind']=='pt' else 70*__import__('math').ceil(len(t.read(t.SPLIT/f"{j['task']}.json")['folds'][j['fold']]['train_indices'])/24)
        known=state.get('attempted_updates',0);v['unknown_tails'].append(dict(job=j,known=known,attempt_upper=upper));v['actual_completed_interval'][1]+=upper-known
    for k,limit in limits.items():
        if k in v and v[k]>limit:raise RuntimeError('BUDGET_STOP: '+k)
    t.write(t.OUT/'budget.json',v);return v

def closeout(controller,error=None):
    b=budget();status='STOPPED' if error else 'COMPLETE_PENDING_REVIEW'
    if error and 'EXPIRED' in error:status='EXPIRED'
    elif error and 'ETA_STOP' in error:status='ETA_STOP'
    value=dict(plan=t.PLAN,revision='r1',status=status,independent_review='PENDING',execution_complete=not error,formal_status='COMPLETE' if b['test_complete']==210 else '未完成/不可比较',macro=None,delta_BASE=None,stable_ready=None,FINAL_ESTIMATOR='BASE_SWA_LAST4_CYCLES' if b['freeze'] else None,PRIMARY_RUN='S42_A',budget=b,cpu_exit=controller.cpu_exit if controller else None,error=error,retries=0,baseline_unchanged=True,real_exits=dict(controller=1 if error else 0,outer=None,bootstrap=None))
    if (t.OUT/'final_review/results.json').exists():value.update({k:v for k,v in t.read(t.OUT/'final_review/results.json').items() if k in ('macro','delta_BASE','stable_ready','same_seed_repro_pass','ft_seed_stability_pass','quality_retained','target_met','performance_pass')})
    if error:first_failure(error);t.write(t.OUT/'failure_closeout.json',value);t.update_plan(status,'closeout',error.splitlines()[-1])
    t.write(t.OUT/'delivery.json',value);t.write(t.OUT/'shutdown_audit.json',value)
    lines=[f'# {t.PLAN}/r1',f'状态{status}；independent_review=PENDING；正式评价{value["formal_status"]}。',f'本周期prepare{b["prepare"]}、新step0尝试{b["initialization"]}、新头{b["heads_complete"]}/35；PT启动{b["pt_starts"]}、完整{b["pt_complete"]}/2、更新{b["pt_updates"]}/{b["pt_completed_updates"]}；FT完整{b["ft_complete"]}/140、更新{b["ft_updates"]}/{b["ft_completed_updates"]}。',f'PT比较{b["pt_pairs"]}对；FT比较通过{b["ft_pairs_passed"]}/35；一次冻结{b["freeze"]}；正式TEST{b["test_complete"]}/210、{b["test_rows"]}/35250行；唯一最终CPU{b["cpu_suites"]}次、exit={value["cpu_exit"]}。',f'独立周期尝试{b["total_updates"]}、日志完成{b["logged_completed_updates"]}、完成区间{b["actual_completed_interval"]}。旧条件谱354471尝试/[354470,354471]区间、旧STABILITY预算和时钟只读，不混算。',f'历史模型张量读取0；preCPU/诊断/额外forward0；不在线修复/重试/resume。源代码、工厂、时钟、数据角色、训练与比较、内存、正式预测、CPU及真实exit在本根及对应日志。',error if error else '完整结果及七任务/论文包络数值对照见final_review/results.json与report.md；stable_ready/质量/性能分开，BASE不自动替换。']
    (t.OUT/'delivery_summary.md').write_text('\n\n'.join(lines)+'\n')
    with (ROOT/'PROJECT_HISTORY.md').open('a') as f:f.write('\n\n## '+t.PLAN+'/r1（执行交付，待审查）\n\n'+'\n\n'.join(lines[1:6])+f'\n\n证据：{t.OUT.relative_to(ROOT)}/delivery_summary.md、delivery.json、budget.json、authorization/clock.json；旧周期不改写。\n')

def main():
    if len(sys.argv)>1 and sys.argv[1]=='worker':
        kind=sys.argv[2]
        if kind in ('ft','test'):
            t.m.configure('DET');t.unit(sys.argv[3],sys.argv[4],int(sys.argv[5]),sys.argv[6] if kind=='test' else None)
        elif kind=='compare_pt':t.compare_pt()
        elif kind=='compare_ft':t.compare_pair(sys.argv[3],int(sys.argv[4]))
        else:raise ValueError('worker whitelist')
        return
    t.require_tmux()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('controller must hide GPU')
    controller=None;error=None
    try:controller=Controller();controller.run()
    except BaseException:
        error=traceback.format_exc();first_failure(error)
        if controller is not None:controller.stop()
    closeout(controller,error)
    if controller:controller.memory.close()
    if error:print(error,flush=True);raise SystemExit(1)
if __name__=='__main__':main()
