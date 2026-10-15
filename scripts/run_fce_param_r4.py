"""Exclusive FCE r4 continuation; retain original scientific runtime and budget."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_fce_param import launch,consumption,PYTHON,GPUS
from scripts.run_fce_param_unit import OUT,LOG,ARMS,TASKS,read,write,validate_config
from src.training.glt_dual_runtime import require_tmux
SOUT=OUT/'scheduler_r4';SLOG=LOG/'scheduler_r4'


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def process(pid):
    p=Path('/proc')/str(pid)
    try:
        cmd=[s.decode() for s in (p/'cmdline').read_bytes().split(b'\0') if s]
        status=(p/'status').read_text()
        ids=[int(s) for line in status.splitlines() if line.startswith('NSpid:') for s in line.split()[1:]]
        state=next(line.split()[1] for line in status.splitlines() if line.startswith('State:'))
        return dict(pid=pid,namespace_pids=ids,argv=cmd,cwd=str((p/'cwd').resolve()),state=state)
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None


def unit_identity(log,unit,gpu,*,proc_root=Path('/proc')):
    """Attest our local process; never infer a host/NVML PID mapping."""
    log=Path(log)
    try:
        command=read(log/'command.json');pid=int((log/'pid.txt').read_text())
        info=process(pid)
        if info is None or info['state'] in ('Z','X'):return None
        expected=[PYTHON,'scripts/run_fce_param_unit.py','--arm',unit[0],'--task',unit[1],'--fold',str(unit[2])]
        if log.name.startswith('test_'):expected.append('--formal')
        if command['argv']!=expected or info['argv']!=expected:return None
        if command['gpu']!=gpu or command['cwd']!=str(ROOT) or info['cwd']!=str(ROOT):return None
        if command['environment'].get('CUDA_VISIBLE_DEVICES')!=str(gpu):return None
        proc=proc_root/str(pid)
        env=dict(x.split(b'=',1) for x in (proc/'environ').read_bytes().split(b'\0') if b'=' in x)
        if env.get(b'CUDA_VISIBLE_DEVICES')!=str(gpu).encode():return None
        # Open output files tie the live process to this exact launch, not just
        # a possibly reused PID or a coincidentally identical command line.
        if not (proc/'fd/1').samefile(log/'stdout.log') or not (proc/'fd/2').samefile(log/'stderr.log'):return None
        stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
        return dict(info,start_ticks=int(stat[19]),gpu=gpu,log=str(log),proof='argv_cwd_cuda_environment_stdout_stderr')
    except (OSError,ValueError,KeyError,IndexError):return None


def ownership(gpu,card,running):
    """Local launch ownership and NVML attribution are separate facts.

    With an attested local workload, admission uses total physical GPU memory,
    which includes all tenants. Unmapped NVML PIDs remain explicitly unknown.
    A busy card without an attested workload is never acquired by this fallback.
    """
    units=[(folder,unit) for slot,(folder,unit) in running.items() if slot[0]==gpu]
    proofs=[unit_identity(folder,unit,gpu) for folder,unit in units]
    known={n for p in proofs if p for n in p['namespace_pids']}
    unmatched=[a['pid'] for a in card['apps'] if a['pid'] not in known]
    valid=all(p is not None for p in proofs)
    allow_unmapped=bool(units) and valid
    admitted=valid and (not unmatched or allow_unmapped)
    mode=('local_launch_attested_nvml_unmapped' if unmatched and admitted else
          'unverified_busy_card' if unmatched else 'namespace_pid_or_empty_card')
    return dict(admitted=admitted,allow_unmapped=allow_unmapped,known_namespace_pids=sorted(known),
                unmatched_nvml_pids=unmatched,nvml_mapping_verified=not bool(unmatched),mode=mode,local_processes=proofs)


def resources():
    import subprocess
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    cards={}
    for line in rows.splitlines():
        p=[x.strip() for x in line.split(',')];g=int(p[0])
        if g in GPUS:cards[g]=dict(uuid=p[1],total_mib=int(p[2]),free_mib=int(p[3]),utilization=int(p[4]),apps=[])
    for line in apps.splitlines():
        p=[x.strip() for x in line.split(',')]
        for c in cards.values():
            if c['uuid']==p[0]:c['apps'].append(dict(pid=int(p[1]),memory_mib=p[2]))
    mem={p[0].rstrip(':'):int(p[1]) for line in Path('/proc/meminfo').read_text().splitlines() if len(p:=line.split())>=2}
    return dict(cards=cards,host_available_mib=mem['MemAvailable']/1024,host_load_average=list(os.getloadavg()),cpu_count=os.cpu_count(),swap_free_mib=mem['SwapFree']/1024,time=now())


def allowed_slots(gpu,occupied,card,known_namespace_pids,target_level,host_available_mib,*,allow_unmapped=False):
    """Unknown NVML PIDs can only coexist with attested local running units."""
    if gpu not in GPUS or target_level not in (1,2,4,6,8) or len(occupied)>=8:return []
    if card['free_mib']<12*1024:return []
    if any(a['pid'] not in known_namespace_pids for a in card['apps']) and not (occupied and allow_unmapped):return []
    if host_available_mib<32*1024:return []
    return [s for s in range(target_level) if s not in occupied]


def may_upgrade(level,joint_seconds,attributed,headroom,high_load,bad_rates):
    return level in (4,6) and joint_seconds>=180 and attributed and headroom and not high_load and bad_rates==0


def lower_level(level):
    levels=(1,2,4,6,8);return levels[max(0,levels.index(level)-1)]


def frozen_candidate(freeze_fn):
    p=OUT/'selection_before_test.json'
    if p.exists():
        frozen=read(p)
        if frozen.get('validation_units')!=140 or not frozen.get('frozen') or frozen.get('candidate') not in ARMS[1:]:
            raise ValueError('invalid existing freeze')
        return frozen['candidate']
    return freeze_fn()


def unit_state(arm,task,fold,formal=False):
    stage='formal' if formal else 'finetune';name=('test_' if formal else 'ft_')+arm+'_'+task+'_'+str(fold)
    log=LOG/name;folder=OUT/stage/arm/task/f'fold{fold}'
    if not log.exists() and not folder.exists():return 'pending'
    if not (log/'exit.txt').exists():
        if (log/'pid.txt').exists():
            p=process(int((log/'pid.txt').read_text()))
            if p and p['argv'][1:2]==['scripts/run_fce_param_unit.py'] and p['cwd']==str(ROOT):return 'running'
        return 'inconsistent'
    rc=int((log/'exit.txt').read_text())
    if rc!=0:return 'failed'
    needed=['metrics.json','test_predictions.npz'] if formal else ['metrics.json','best.pt','scaler.json','progress.json','updates.json','validation_predictions.npz']
    if any(not (folder/f).is_file() for f in needed):return 'inconsistent'
    execution=read(log/'execution.json');m=read(folder/'metrics.json')
    if execution.get('exit_code')!=0 or m.get('status')!='PASS':return 'inconsistent'
    return 'success'


def planned_queue(arms,formal=False):
    pending=[];failed=[]
    for arm in arms:
        for task in TASKS:
            for fold in range(5):
                state=unit_state(arm,task,fold,formal)
                if state=='pending':pending.append((arm,task,fold))
                elif state=='failed':failed.append((arm,task,fold))
                elif state!='success':raise RuntimeError(f'unowned/inconsistent {arm}/{task}/{fold}: {state}')
    return pending,failed


class Scheduler:
    def __init__(self):
        self.target={g:2 for g in GPUS};self.samples=[];self.baseline={};self.bad_rates={g:0 for g in GPUS}
        self.joint_seconds={g:0.0 for g in GPUS};self.last_change={g:time.time() for g in GPUS}
        self.gpu_util_history={g:[] for g in GPUS};self.holds={g:None for g in GPUS}
        self.previous={};self.previous_time=None

    def snapshot(self,stage,running,rs):
        c=consumption();wall=time.time();self.samples.append((wall,c['ft_completed_updates']))
        self.samples=[x for x in self.samples if wall-x[0]<600]
        duration=wall-self.samples[0][0];delta=c['ft_completed_updates']-self.samples[0][1]
        rate=delta/duration if duration>=90 and delta>0 else None
        eta=None if rate is None else (205800-c['ft_completed_updates'])/rate*1.3
        states={};current={}
        for slot,(log,unit) in running.items():
            arm,task,fold=unit;folder=OUT/'finetune'/arm/task/f'fold{fold}';epoch=None;phase=None;u=None
            if (folder/'progress.json').exists():
                h=read(folder/'progress.json')['history'];epoch=h[-1]['epoch'];phase=h[-1]['stage']
            if (folder/'updates.json').exists():u=read(folder/'updates.json')['completed']
            if log.name.startswith('test_'):phase='formal_prediction';u=None
            pid=int((log/'pid.txt').read_text()) if (log/'pid.txt').exists() else None
            states[f'{slot[0]}:slot{slot[1]}']=dict(unit=unit,log=str(log),pid=pid,process_identity=None if pid is None else process(pid),epoch=epoch,phase=phase,updates=u)
            current[slot]=(task,phase,u)
        # Compare only joint intervals of the same task on the same GPU.
        if self.previous_time is not None and wall-self.previous_time>=10:
            elapsed=wall-self.previous_time
            for gpu in GPUS:
                slots=[s for s in current if s[0]==gpu]
                if not slots or any(s not in self.previous for s in slots):continue
                entries=[current[s] for s in slots]
                if any(e[1]!='joint' or e[2] is None or self.previous[s][:2]!=e[:2] for s,e in zip(slots,entries)):continue
                tasks={e[0] for e in entries}
                if len(tasks)!=1:continue
                task=entries[0][0];v=sum(current[s][2]-self.previous[s][2] for s in slots)/elapsed
                if v<=0:continue
                key=(gpu,task)
                if len(slots)==1:self.baseline.setdefault(key,[]).append(v)
                elif len(self.baseline.get(key,[]))>=3:
                    baseline=sum(self.baseline[key])/len(self.baseline[key])
                    self.bad_rates[gpu]=self.bad_rates[gpu]+1 if v<=baseline else 0
                    if self.bad_rates[gpu]>=3 and wall-self.last_change[gpu]>=180:
                        self.fallback(gpu,'three comparable joint intervals show no aggregate throughput improvement',rs)
        for gpu in GPUS:
            card=rs['cards'][gpu];slots=[k for k in current if k[0]==gpu]
            self.gpu_util_history[gpu].append((wall,card['utilization']))
            self.gpu_util_history[gpu]=[x for x in self.gpu_util_history[gpu] if wall-x[0]<=180]
            identity=ownership(gpu,card,running)
            card['ownership']=identity
            attributed=identity['admitted']
            progressing=self.previous_time is not None and len(slots)>=self.target[gpu] and all(
                k in self.previous and current[k][1]=='joint' and self.previous[k][1]=='joint' and
                current[k][2] is not None and self.previous[k][2] is not None and current[k][2]>self.previous[k][2] for k in slots)
            if progressing:self.joint_seconds[gpu]+=wall-self.previous_time
            headroom=card['free_mib']>=12*1024 and 1-card['free_mib']/card['total_mib']<.8 and rs['host_available_mib']>=32*1024
            high_load=len(self.gpu_util_history[gpu])>=4 and sum(x[1] for x in self.gpu_util_history[gpu])/len(self.gpu_util_history[gpu])>=85
            if not attributed:self.holds[gpu]='local launch not attested or busy card without owned workload'
            elif high_load:self.holds[gpu]='sustained high GPU load; maintain effective level'
            elif not headroom:self.holds[gpu]='resource boundary'
            else:self.holds[gpu]=None
            if may_upgrade(self.target[gpu],self.joint_seconds[gpu],attributed,headroom,high_load,self.bad_rates[gpu]):
                old=self.target[gpu];self.target[gpu]=6 if old==4 else 8
                self.last_change[gpu]=wall;self.joint_seconds[gpu]=0
                with (SLOG/'adjustments.jsonl').open('a') as f:f.write(json.dumps(dict(time=now(),gpu=gpu,old=old,new=self.target[gpu],reason='>=180 seconds of normal advancing joint at current level with headroom',resources=rs),ensure_ascii=False)+'\n')
        self.previous=current;self.previous_time=wall
        value=dict(time=now(),stage=stage,consumption=c,running=states,resources=rs,target_concurrency_per_gpu=self.target,holds=self.holds,joint_observed_seconds=self.joint_seconds,
            current_concurrency=len(running),recent_actual_updates_per_second=rate,
            conservative_ft_eta_seconds=eta,eta_basis='actual recent aggregate completed-update throughput; mixed-task estimate, no assumed division by 4',
            status='执行中',independent_review='NOT_PERFORMED')
        write(SOUT/'status.json',value)
        with (SLOG/'progress.jsonl').open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
        # Only write handoff after the old controller has exited.
        if (LOG/'controller/exit.txt').exists():
            write(OUT/'status.json',value)
            p=ROOT/'CURRENT_PLAN.md';text=p.read_text().split('\n<!-- FCE_R2_LIVE -->\n')[0]
            estimate='尚不可估计' if eta is None else f'{eta/3600:.2f}小时（不含资源等待／正式预测／测试）'
            p.write_text(text+'\n<!-- FCE_R2_LIVE -->\n'+f'\nFCE r4实时记录：{now()} UTC，阶段{stage}；FT {c["ft_completed"]}/140，epochs {c["ft_epochs"]}/9600，attempted updates {c["ft_attempted_updates"]}/205800，正式预测 {c["formal_completed"]}/70；当前并发{len(running)}，并发目标{self.target}、等待原因{self.holds}，按真实近期吞吐估计剩余FT {estimate}。槽位／进程归属／显存／主机内存／重叠时段见scheduler_r4/status.json、progress.jsonl。\n')

    def fallback(self,gpu,reason,rs):
        old=self.target[gpu];self.target[gpu]=lower_level(old)
        self.last_change[gpu]=time.time();self.joint_seconds[gpu]=0;self.bad_rates[gpu]=0
        with (SLOG/'fallbacks.jsonl').open('a') as f:f.write(json.dumps(dict(time=now(),gpu=gpu,old=old,new=self.target[gpu],reason=reason,resources=rs),ensure_ascii=False)+'\n')

    def drain(self):
        record=read(SOUT/'takeover.json');pid=record['old_controller_pid']
        p=process(pid)
        if p is not None and (p['argv'][1:2]!=['scripts/run_fce_param.py'] or p['state']!='T'):raise RuntimeError('old controller not exclusively paused')
        while True:
            active={}
            for folder in list(LOG.glob('ft_*'))+list(LOG.glob('test_*')):
                if (folder/'command.json').exists() and not (folder/'exit.txt').exists():
                    command=read(folder/'command.json');argv=command['argv']
                    unit=(argv[argv.index('--arm')+1],argv[argv.index('--task')+1],int(argv[argv.index('--fold')+1]))
                    gpu=command['gpu'];active[gpu,0]=(folder,unit)
            self.snapshot('接管：仅旧控制器暂停，等待在途自然完成',active,resources())
            if not active:break
            time.sleep(30)
        # Every independent unit launcher has persisted a true exit first.
        p=process(pid)
        if p is not None:
            if p['argv'][1:2]!=['scripts/run_fce_param.py'] or p['cwd']!=str(ROOT) or p['state']!='T':raise RuntimeError('controller identity changed during drain')
            os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
        while not (LOG/'controller/exit.txt').exists():time.sleep(1)
        if process(pid) is not None:raise RuntimeError('old controller still alive')
        record.update(drained=now(),old_controller_exit=int((LOG/'controller/exit.txt').read_text()),
            unit_signals=0,new_controller_pid=os.getpid(),status='exclusive takeover complete')
        write(SOUT/'takeover.json',record)
        p=ROOT/'CURRENT_PLAN.md';text=p.read_text();text=text.replace('# 当前执行：FCE-PARAM-20261014-01 / r4（r2科学产物接续）','# 当前执行：FCE-PARAM-20261014-01 / r4（r2科学产物接续）',1)
        text=text.split('\n<!-- FCE_R2_LIVE -->\n')[0]
        text+='\n\n## FCE r4 接续授权与安全接管\n\n用户明确授权按PLAN.MD r4仅接续第4节四臂周期、第6／7节。旧GLT及第5节预算0、不执行。仅调整调度，每卡目标4→6→8、最多8、全局最多16；batch24、FP32、workers0、OMP4、BLAS/MKL1、训练runtime／配置／初始化／种子均不变。沿用全部r2账本与目录，不重置140 FT／9600 epochs／205800 attempted updates／70预测。\n\n旧FCE controller PID '+str(pid)+'仅单独SIGSTOP停止派发，所有在途unit及启动器自然结束并落真实exit后，单独SIGTERM/SIGCONT使旧controller退出，真实exit '+str(record['old_controller_exit'])+'（计划接管退出，不是训练失败）。未向unit或进程组发信号。接管证据scheduler_r4/takeover.json，新controller Uni-Poly:fce_param_r4_controller；唯一文件锁scheduler_r4/controller.lock。\n\n资源边界：每次派发空闲显存≥12 GiB、主机可用内存≥32 GiB，陌生NVML进程拒绝派发；PID namespace映射不确定不增槽。每次升档后至少180秒正常joint；显存占用≥80%／可用RAM<32 GiB／可比joint吞吐恶化时降低一级，在途自然结束。最终一次CPU合并原recipe suite与调度模拟suite，总额度不新增。\n'
        p.write_text(text)

    def matrix(self,arms,formal=False):
        queue,failed=planned_queue(arms,formal);running={}
        while queue or running:
            for slot,(folder,unit) in list(running.items()):
                if (folder/'exit.txt').exists():
                    state=unit_state(*unit,formal)
                    if state!='success':
                        failed.append(unit);self.fallback(slot[0],'failed unit '+str(unit),resources())
                    del running[slot]
            rs=resources()
            for gpu in GPUS:
                card=rs['cards'][gpu]
                if time.time()-self.last_change[gpu]>=180 and (rs['host_available_mib']<32*1024 or 1-card['free_mib']/card['total_mib']>=.8):
                    self.fallback(gpu,'GPU >=80% used or host available RAM <32 GiB',rs)
                occupied={s[1] for s in running if s[0]==gpu}
                identity=ownership(gpu,card,running);card['ownership']=identity
                slots=allowed_slots(gpu,occupied,card,set(identity['known_namespace_pids']),self.target[gpu],
                                    rs['host_available_mib'],allow_unmapped=identity['allow_unmapped']) if identity['admitted'] else []
                # Launch one per card per poll, attribute it before adding slots.
                if slots and queue:
                    arm,task,fold=queue.pop(0)
                    if unit_state(arm,task,fold,formal)!='pending':raise RuntimeError('unit became non-pending before dispatch')
                    argv=[PYTHON,'scripts/run_fce_param_unit.py','--arm',arm,'--task',task,'--fold',str(fold)]
                    if formal:argv.append('--formal')
                    name=('test_' if formal else 'ft_')+arm+'_'+task+'_'+str(fold)
                    folder=launch(name,argv,gpu);running[gpu,slots[0]]=(folder,(arm,task,fold))
                    with (SLOG/'dispatch.jsonl').open('a') as f:f.write(json.dumps(dict(time=now(),gpu=gpu,slot=slots[0],unit=[arm,task,fold],formal=formal,log=str(folder)),ensure_ascii=False)+'\n')
            self.snapshot(('正式预测' if formal else 'FT')+':'+','.join(arms),running,rs)
            if queue or running:time.sleep(30)
        return failed


def campaign():
    require_tmux()
    with (SOUT/'controller.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lease.seek(0);lease.truncate();lease.write(str(os.getpid()));lease.flush()
        if (OUT/'completion.json').exists():
            write(SOUT/'already_complete.json',read(OUT/'completion.json'));return
        scheduler=Scheduler()
        repair=SOUT/'pid_repair1/takeover.json'
        if repair.exists():
            handoff=read(repair)
            if handoff.get('status')!='drained_and_old_controller_exited' or process(handoff['old_controller_pid']) is not None:
                raise RuntimeError('PID repair takeover is not complete')
        else:scheduler.drain()
        for arm in ARMS:
            for task in TASKS:validate_config(read(OUT/'initialization'/arm/(task+'.json')),arm,task)
        failed=[]
        for arm in ARMS:failed+=scheduler.matrix([arm])
        if failed:
            write(SOUT/'blocked.json',dict(failed_units=failed,reason='no full trustworthy continuation state; no extra retry budget'))
            raise RuntimeError('incomplete FT; no selection/TEST')
        from scripts.select_fce_param import freeze,summarize
        candidate=frozen_candidate(freeze)
        if scheduler.matrix(['R0',candidate],True):raise RuntimeError('incomplete formal matrix')
        test=LOG/'final_cpu'
        if not test.exists():
            suites=['tests/test_fce_param.py','tests/test_fce_param_scheduler.py']
            repair_test=LOG/'final_cpu_pidrepair1/exit.txt'
            if repair_test.exists():
                if int(repair_test.read_text())!=0:raise RuntimeError('PID repair final CPU test failed')
                suites=['tests/test_fce_param.py']  # scheduler suite already completed; no duplicate regression
            test=launch('final_cpu',['timeout','300s',PYTHON,'-m','pytest','-q',*suites])
        while not (test/'exit.txt').exists():scheduler.snapshot('最终CPU测试',{},resources());time.sleep(10)
        rc=int((test/'exit.txt').read_text());review=OUT/'final_review/delivery.md'
        delivery=review.read_text() if review.exists() else summarize()
        c=consumption();state='待审查' if rc==0 else '需返修'
        write(OUT/'completion.json',dict(status=state,time=now(),test_exit=rc,candidate=candidate,consumption=c,independent_review='NOT_PERFORMED',scheduler='r4'))
        p=ROOT/'CURRENT_PLAN.md';p.write_text(p.read_text()+f'\n\n## FCE r4 全周期最终记录\n\n状态：**{state}**。140 FT及70正式预测实际完成；最终CPU真实exit {rc}。独立审查未进行，未归档模型历史。累计预算：{json.dumps(c,ensure_ascii=False)}。\n\n'+delivery+'\n')


if __name__=='__main__':
    try:campaign()
    except BaseException as e:
        write(SOUT/'campaign_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=now(),consumption=consumption()))
        raise
