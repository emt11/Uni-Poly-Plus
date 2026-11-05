#!/usr/bin/env python3
"""Unique r2 controller and worker dispatch; no retry/resume branch."""
import os,sys,json,time,subprocess,traceback,statistics,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training import ua3d_cond_spectral_r2 as t

def inventory():
    row=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid,memory.total,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(', ')
    if row[0]!=t.UUID:raise ValueError('physical GPU2 UUID mismatch')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    return dict(uuid=row[0],total_mib=float(row[1]),used_mib=float(row[2]),processes=[s for s in apps.splitlines() if s.startswith(t.UUID)])

def budget():
    pt=t.read(t.OUT/'pretrain/BASE_FRESH/consumption.json') if (t.OUT/'pretrain/BASE_FRESH/consumption.json').exists() else {}
    ft=[t.read(p) for p in (t.OUT/'finetune').glob('*/*/fold*/consumption.json')]
    formal=[t.read(p) for p in (t.OUT/'formal').glob('*/*/fold*/consumption.json')]
    value=dict(pt_starts=int(bool(pt)),pt_updates=pt.get('attempted_updates',0),pt_completed=pt.get('completed_updates',0),pt_microbatches=pt.get('pt_microbatches',0),pt_presentations=pt.get('pt_presentations',0),pt_prefetched_not_consumed=pt.get('prefetched_not_consumed',0),ft_starts=len(ft),ft_epochs=sum(c.get('epochs_started',0) for c in ft),ft_updates=sum(c.get('attempted_updates',0) for c in ft),ft_completed_updates=sum(c.get('completed_updates',0) for c in ft),ft_complete=sum(c['status']=='COMPLETE' for c in ft),test_starts=len(formal),test_complete=sum(c['status']=='COMPLETE' for c in formal),test_rows=sum(c.get('test_graphs',0) for c in formal),final_cpu=int((t.OUT/'final_cpu/started.json').exists()),pre_cpu=0,diagnostic=0,geometry=0,surface=0,spectrum=0,historical_tensor_reads=0)
    value['total_updates']=value['pt_updates']+value['ft_updates'];value['limits']=t.read(ROOT/'configs/ua3d_cond_spectral_r2_gpu2.json')['limits']
    for k in ('pt_updates','ft_starts','ft_epochs','ft_updates','total_updates','test_rows','final_cpu'):
        if value[k]>value['limits'][k]:raise RuntimeError('budget exceeded '+k)
    if value['test_starts']>210:raise RuntimeError('TEST starts budget')
    t.write(t.OUT/'budget.json',value);t.write(t.OUT/'cumulative_budget.json',dict(current=value,historical='read-only separate lineage; no inherited initialization or refunded consumption',old_readout='results/ua3d_hmr_readout_20261103/r1/cumulative_budget.json'))
    t.write(t.OUT/'consumption.json',value);return value

def prior_costs():
    costs={}
    for task in t.TASKS:
        for fold in range(5):
            h=t.read(t.base_path(task,fold)/'history.json')
            costs[(task,fold)]=sum(x['seconds'] for x in h)
    return costs

class Controller:
    def __init__(self):self.jobs=[];self.live=[];self.priors=prior_costs();self.ratio=1.;self.cpu_exit=None
    def check(self):
        t.check_clock();budget()
        if any(job['kind']=='prepare' and job['exit'] is None for job in self.jobs):t.check_clock(True)
        # Sum only this controller's descendant RSS; process exit races use poll/wait.
        pids={os.getpid()};rss=0;items=[]
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:
                s=(p/'status').read_text();fields={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in s.splitlines() if ':' in line};items.append((int(p.name),int(fields.get('PPid','0')),int(fields.get('VmRSS','0 kB').split()[0])*1024))
            except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        for _ in range(8):
            for pid,parent,_ in items:
                if parent in pids:pids.add(pid)
        rss=sum(mem for pid,_,mem in items if pid in pids)
        if rss>160*1024**3:raise RuntimeError('160GiB host memory cap')
        t.write(t.OUT/'telemetry.json',dict(time=time.time(),descendant_rss_bytes=rss,pids=sorted(pids)))
    def launch(self,kind,arm=None,task=None,fold=None):
        self.check();name='-'.join(str(x) for x in (kind,arm,task,fold) if x is not None);path=t.LOG/name;path.mkdir(parents=True,exist_ok=False)
        argv=[sys.executable,str(Path(__file__).resolve()),'worker',kind]
        if arm is not None:argv += [arm,task,str(fold)]
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='' if kind in ('prepare','cpu') else '2',CUDA_DEVICE_ORDER='PCI_BUS_ID',WORLD_SIZE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',UA3M_PHYSICAL_GPU_UUID=t.UUID)
        if kind=='prepare':argv=[sys.executable,str(ROOT/'scripts/prepare_ua3d_cond_spectral_r2.py')]
        if kind=='cpu':argv=[sys.executable,'-m','pytest','-q','tests/test_ua3d_cond_spectral_r2.py']
        file=(path/'stdout.log').open('x');p=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=file,stderr=subprocess.STDOUT,start_new_session=True)
        job=dict(kind=kind,arm=arm,task=task,fold=fold,pid=p.pid,cwd=str(ROOT),argv=argv,started_unix=time.time(),exit=None,log=str(path/'stdout.log'))
        self.jobs.append(job);self.live.append((p,file,job));t.write(t.OUT/'jobs.json',self.jobs);return p
    def poll(self):
        self.check()
        for entry in list(self.live):
            p,file,job=entry
            code=p.poll()
            if code is None:continue
            code=p.wait();file.close();job.update(exit=code,ended_unix=time.time());self.live.remove(entry);t.write(t.OUT/'jobs.json',self.jobs)
            if job['kind']=='cpu':self.cpu_exit=code
            if code:raise RuntimeError(f"real worker failure {job['kind']} {job['arm']} {job['task']} {job['fold']} exit={code}")
    def wait(self):
        while self.live:self.poll();time.sleep(5)
    def wait_gpu(self):
        while True:
            self.check();inv=inventory();t.write(t.OUT/'gates/gpu_wait.json',dict(time=time.time(),**inv,policy='unknown occupation waits; never terminate unrelated process'))
            if not inv['processes']:return inv
            time.sleep(30)
    def eta(self,pending,concurrency,extra=0):
        work=sum(self.priors[(task,fold)]*self.ratio for arm,task,fold in pending)
        # Include full formal evaluation conservatively using one epoch-equivalent
        # per TEST unit from observed full 70-epoch unit wall time.
        formal=sum(self.priors.values())*6*self.ratio/70/max(1,concurrency)
        eta=1.5*(work/max(1,concurrency)+formal+extra)+7200
        t.write(t.OUT/'gates/remaining_eta.json',dict(time=time.time(),pending=len(pending),concurrency=concurrency,ratio=self.ratio,work_seconds=work,formal_seconds=formal,remaining_seconds=eta,includes=['validation','TEST','final CPU','delivery']))
        if time.time()+eta>=t.read(t.OUT/'authorization/clock.json')['deadline_unix']:raise TimeoutError('conservative total ETA gate')
    def stop(self):
        for p,file,job in self.live:
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for p,file,job in self.live:
            try:code=p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);code=p.wait()
            file.close();job.update(exit=code,ended_unix=time.time(),controller_stopped=True)
        self.live=[];t.write(t.OUT/'jobs.json',self.jobs)
    def run(self):
        t.update_plan('RUNNING','prepare','新周期t0已登记；无前置测试。')
        self.launch('prepare');self.wait();t.check_clock(True)
        self.wait_gpu();self.eta([(a,task,fold) for a in t.ARMS for task in t.TASKS for fold in range(5)],4,extra=13446.2)
        t.update_plan('RUNNING','PT','全新BASE 10K；仅本轮step0初始化。')
        self.launch('pt');self.wait()
        t.update_plan('RUNNING','FT_first_six','六臂EPS/fold0串行完整70epoch，计入210套。')
        obs={};peaks=[]
        for arm in t.ARMS:
            self.wait_gpu();self.launch('ft',arm,'eps',0);self.wait()
            h=t.read(t.unit_path(arm,'eps',0)/'history.json');stable=[x['training_seconds']/x['batches'] for x in h if x['stage']=='joint' and x['stage_epoch']>2]
            obs[arm]=statistics.median(stable);peaks.append(max(x['gpu_peak_bytes'] for x in h))
            ratios={a:v/obs['C_FRESH'] for a,v in obs.items()};t.write(t.OUT/'gates/ft_speed.json',dict(seconds_per_batch=obs,ratios=ratios,maximum=2.,excluded_first_joint_epochs=2,normal_units=True))
            if any(v>2. for v in ratios.values()):raise RuntimeError('FT surface speed > 2.0x C_FRESH')
            cost=sum(x['seconds'] for x in h);self.ratio=max(self.ratio,cost/self.priors[('eps',0)])
            pending=[(a,task,fold) for a in t.ARMS for task in t.TASKS for fold in range(5) if not (task=='eps' and fold==0 and a in obs)]
            self.eta(pending,4)
        pending=[(a,task,fold) for a in t.ARMS for task in t.TASKS for fold in range(5) if (task,fold)!=('eps',0)]
        inv=self.wait_gpu();available=(inv['total_mib']-inv['used_mib'])*1024**2;peak=max(peaks);concurrency=1
        if peak*2<=.8*available:concurrency=2
        t.write(t.OUT/'gates/memory_admission.json',dict(peak=peak,available=available,concurrency=concurrency,limit_fraction=.8))
        waves=0
        while pending:
            self.eta(pending,concurrency);wave=pending[:concurrency];pending=pending[concurrency:]
            for a,task,fold in wave:self.launch('ft',a,task,fold)
            self.wait();waves+=1
            for a,task,fold in wave:
                h=t.read(t.unit_path(a,task,fold)/'history.json');peak=max(peak,max(x['gpu_peak_bytes'] for x in h));cost=sum(x['seconds'] for x in h)
                self.ratio=max(self.ratio,cost/self.priors[(task,fold)])
            # First 2-way wave supplies actual observation before 4-way admission.
            if concurrency==2 and waves>=1 and peak*4<=.8*available:concurrency=4
            if concurrency*peak>.8*available:raise RuntimeError('observed concurrency memory admission exceeds 80%')
            t.write(t.OUT/'gates/memory_admission.json',dict(peak=peak,available=available,concurrency=concurrency,completed_waves=waves,limit_fraction=.8))
        t.freeze();t.update_plan('RUNNING','formal_TEST','210 FT完成；一次全局冻结；开始210正式单元。')
        units=[(a,task,fold) for a in t.ARMS for task in t.TASKS for fold in range(5)]
        for start in range(0,len(units),concurrency):
            for a,task,fold in units[start:start+concurrency]:self.launch('test',a,task,fold)
            self.wait()
        t.report_results();t.exclusive(t.OUT/'final_cpu/started.json',dict(started_unix=time.time(),attempt=1))
        t.update_plan('RUNNING','final_CPU','完整正式TEST后唯一一次CPU suite。');self.launch('cpu');self.wait()
        budget();t.update_plan('COMPLETE_PENDING_REVIEW','delivery','全周期完成，independent_review=PENDING，不自动替换BASE。')

def closeout(controller,error=None):
    b=budget();status='STOPPED' if error else 'COMPLETE_PENDING_REVIEW'
    value=dict(status=status,error=error,budget=b,independent_review='PENDING',FINAL=None,macro=None,delta_BASE=None,delta_C_FRESH=None,real_exits=dict(controller=1 if error else 0,outer=None,bootstrap=None),final_cpu_exit=controller.cpu_exit,formal_status='未完成/不可比较' if b['test_complete']!=210 else 'COMPLETE',retries=0)
    if (t.OUT/'final_review/results.json').exists():
        r=t.read(t.OUT/'final_review/results.json');value.update({k:r[k] for k in ('FINAL','macro','delta_BASE','delta_C_FRESH','target_met','performance_pass')})
    if error:
        t.exclusive(t.OUT/'first_failure.json',dict(error=error,time=t.now(),automatic_retry=False));t.write(t.OUT/'failure_closeout.json',value);t.update_plan('STOPPED','closeout',error.splitlines()[-1])
    t.write(t.OUT/'delivery.json',value)
    lines=[f'# {t.PLAN}/r2',f'状态：{status}；independent_review=PENDING。',f"正式评价：{value['formal_status']}；完成 {b['test_complete']}/210 单元。",f"PT完成更新 {b['pt_completed']}；FT完成 {b['ft_complete']}/210；最终CPU次数 {b['final_cpu']}，exit={controller.cpu_exit}。",f"真实controller exit={1 if error else 0}；outer/bootstrap由bootstrap实际wait后补记。",f"错误：{error}" if error else '完整结果见final_review/results.json；不自动替换BASE。','实现文件见implementation_diff.md，逐作业argv/cwd/log/真实exit见jobs.json；旧HMR/READOUT保持原样。']
    (t.OUT/'delivery_summary.md').write_text('\n\n'.join(lines)+'\n')
    history=ROOT/'PROJECT_HISTORY.md'
    with history.open('a') as f:f.write('\n\n## UA3D-COND-SPECTRAL-20261104-01/r2（待审查）\n\n'+ '\n\n'.join(lines[1:4])+f'\n\n证据：{t.OUT.relative_to(ROOT)}/delivery.json、delivery_summary.md、budget.json、jobs.json。历史BASE=F_PD_N1_F0不变。\n')

def main():
    if len(sys.argv)>1 and sys.argv[1]=='worker':
        kind=sys.argv[2]
        if kind=='pt':t.pt_train()
        elif kind in ('ft','test'):t.unit(sys.argv[3],sys.argv[4],int(sys.argv[5]),kind=='test')
        else:raise ValueError('unknown worker')
        return
    t.require_tmux()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('controller must hide GPU')
    controller=Controller();error=None
    try:controller.run()
    except BaseException:
        error=traceback.format_exc();controller.stop()
    closeout(controller,error)
    if error:print(error,flush=True);raise SystemExit(1)
if __name__=='__main__':main()
