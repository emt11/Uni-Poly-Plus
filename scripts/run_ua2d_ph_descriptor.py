#!/usr/bin/env python3
"""One finite isolated PTSCALE/P10K-PH descriptor controller; no PT, retries or baseline execution."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'1')
import sys,argparse,fcntl,signal,shlex,subprocess,time,json,traceback,difflib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training.ua2d_ph_descriptor import *
from src.training.ua2d_fpdecay_sg import process_alive
SOURCES=('src/dataset/ua2d_ph_descriptor.py','src/modules/ua2d_ph_descriptor.py','src/training/ua2d_ph_descriptor.py',
 'scripts/run_ua2d_ph_descriptor.py','configs/ua2d_ph_descriptor_r1.json','tests/test_ua2d_ph_descriptor.py')


def can_dispatch(active,arm,task,fold):
    gpu=gpu_for(task,fold)
    return arm in ARMS and len(active)<TOTAL_LIMIT and sum(j.get('gpu')==gpu for j in active)<GPU_LIMITS[gpu]


def append_plan(detail):
    # The existing campaign is not stopped or modified. Append only during its observed idle window.
    campaigns=[ROOT/'results/ua2d_3dshift_20261022/r1/all1_pcontrol_gpu2',ROOT/'results/ua2d_fg10k_morgan_20261023/r1']
    for _ in range(200):
        all_idle=True
        for campaign in campaigns:
            p=campaign/'execution_manifest.json'
            try:i=read(p).get('writer_process');alive=i and process_alive(i)
            except OSError:alive=False
            if not alive:continue
            try:waiting=(Path('/proc')/str(i['pid'])/'wchan').read_text().strip()
            except OSError:continue
            if waiting not in ('hrtimer_nanosleep','do_wait','poll_schedule_timeout.constprop.0'):all_idle=False
        if all_idle:break
        time.sleep(.05)
    else:
        write(OUT/'pending_plan_append.json',dict(time=now(),detail=detail));return
    entry='\n### '+PLAN+' / '+REVISION+' 执行记录\n\n- '+now()+'：'+detail+'\n- 独立结果：'+str(OUT)+'；日志：'+str(LOG)+'；待Codex独立审查。\n'
    descriptor=os.open(str(ROOT/'CURRENT_PLAN.md'),os.O_WRONLY|os.O_APPEND)
    try:os.write(descriptor,entry.encode());os.fsync(descriptor)
    finally:os.close(descriptor)


class Controller:
    def __init__(self):
        require_tmux();auth=read(OUT/'authorization/user_authorization.json')
        if auth['plan']!=PLAN or auth['revision']!=REVISION or auth['limits']!=LIMITS or not auth['experiment_authorized']:raise ValueError('authorization conflict')
        self.lock=(OUT/'controller.lock').open('x');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'budget.json').exists() or (OUT/'execution_manifest.json').exists():raise FileExistsError('existing execution; no retry/resume')
        self.jobs=[];self.started_unix=None;self.cached=None;self.budget=None;self.evidence_only=False
        self.validation={};self.formal={};self.reference=None;self.frozen=None
        self.window=subprocess.check_output(['tmux','display-message','-p','-t',os.environ['TMUX_PANE'],'#{window_id}'],text=True).strip()
        self.manifest=dict(plan=PLAN,revision=REVISION,writer_process=process_identity(os.getpid()),executor='DSH',controller_window=self.window,
            cwd=str(ROOT),command=sys.argv,only_gpus=[2],resource_override='user latest GPU2 only; plan GPU01 logical parity retained',world_size=1,limits=LIMITS,gpu_limits=GPU_LIMITS,total_limit=TOTAL_LIMIT,
            automatic_retry=False,automatic_resume=False,wallclock_started_unix=None,independent_review='PENDING')
        write(OUT/'execution_manifest.json',self.manifest);self.capture_diff();self.refresh()

    def capture_diff(self):
        full=[]
        for rel in SOURCES:full.extend(difflib.unified_diff([],(ROOT/rel).read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+rel))
        (OUT/'final_review').mkdir(exist_ok=True);(OUT/'final_review/actual_source_diff.patch').write_text(''.join(full))
        (OUT/'final_review/implementation_incremental_diff.patch').write_text(''.join(full))

    def deadline(self):
        if self.started_unix is not None and time.time()>=self.started_unix+129600:raise TimeoutError('independent36h deadline exhausted')

    def state(self,status,phase,detail):
        write(OUT/'status.json',dict(plan=PLAN,revision=REVISION,time=now(),status=status,phase=phase,detail=detail,independent_review='PENDING'))

    def refresh(self):
        used=dict.fromkeys(LIMITS,0);known=dict.fromkeys(LIMITS,0);remaining=dict.fromkeys(LIMITS,0);uncertainty=[]
        for j in self.jobs:
            key={'prepare':'prepare_starts','ft':'ft_starts','test':'test_starts','cpu':'cpu_suites'}[j['kind']];used[key]+=1;known[key]+=1
            exit_path=Path(j['log'])/'exit.txt'
            if j['exit'] is None and exit_path.exists():
                code=int(exit_path.read_text().strip());j.update(exit=code,ended=now(),status='COMPLETE' if code==0 else 'FAILED')
            if j['exit'] is None and j.get('window'):
                result=subprocess.run(['tmux','list-panes','-t',j['window'],'-F','#{pane_dead}'],capture_output=True,text=True)
                if result.returncode or result.stdout.strip()=='1':
                    if exit_path.exists():j.update(exit=int(exit_path.read_text().strip()),ended=now())
                    else:j['status']='EXIT_UNKNOWN'
            if j['kind']=='prepare':
                p=INPUTS/'consumption.json';c=read(p) if p.exists() else {};j['progress']=c
                used['initialization_starts']+=int(c.get('initialization_starts',0));known['initialization_starts']+=int(c.get('initialization_starts',0))
                if j['exit'] is None:remaining['initialization_starts']+=1-int(c.get('initialization_starts',0))
            elif j['kind'] in ('ft','test'):
                p=Path(j['consumption']);c=read(p) if p.exists() else {};j['progress']=c;counts=conservative_ft_counts(c)
                for k,v in counts['charged'].items():used[k]+=v;known[k]+=counts['known_actual_lower_bound'][k]
                if counts['open_epoch']:uncertainty.append(dict(name=j['name'],**counts))
                if not c and j['exit'] is not None:
                    for k,v in j['reservation'].items():used[k]+=v
                    uncertainty.append(dict(name=j['name'],reason='no durable counter; full dispatch reservation charged'))
                if j['exit'] is None:
                    for k,v in j['reservation'].items():remaining[k]+=max(0,v-counts['charged'].get(k,0))
        active=[j for j in self.jobs if j['exit'] is None and j.get('status')!='EXIT_UNKNOWN' and j['kind'] in ('ft','test')]
        concurrency={str(g):sum(j['gpu']==g for j in active) for g in GPU_LIMITS}
        violations=[k for k in LIMITS if used[k]+remaining[k]>LIMITS[k]]
        if len(active)>TOTAL_LIMIT or any(n>GPU_LIMITS[int(g)] for g,n in concurrency.items()):violations.append('concurrency')
        payload=dict(plan=PLAN,revision=REVISION,limits=LIMITS,used=used,known_actual_lower_bound=known,active_remaining_reservation=remaining,
            active_per_gpu=concurrency,active_total=len(active),gpu_limits=GPU_LIMITS,total_limit=TOTAL_LIMIT,
            budget_violations=violations,uncertainty=uncertainty,jobs=self.jobs,wallclock_started_unix=self.started_unix,
            deadline_unix=self.started_unix+129600 if self.started_unix is not None else None)
        encoded=json.dumps(payload,sort_keys=True,allow_nan=False)
        if encoded!=self.cached:
            payload['time']=now();write(OUT/'budget.json',payload);self.budget=payload;self.cached=encoded
        if violations and not self.evidence_only:raise RuntimeError('budget/concurrency exceeded')
        return self.budget

    def check_failures(self):
        if any(j['exit'] not in (None,0) or j.get('status')=='EXIT_UNKNOWN' for j in self.jobs):raise RuntimeError('job failed/exit unknown; no retry/resume')

    def launch(self,kind,arm=None,task=None,fold=None,final_cpu=False):
        self.deadline();b=self.refresh()
        if not final_cpu:self.check_failures()
        if b['budget_violations']:raise RuntimeError('no dispatch after exhausted budget')
        if kind not in ('prepare','ft','test','cpu'):raise ValueError('no PT/extra stage')
        key={'prepare':'prepare_starts','ft':'ft_starts','test':'test_starts','cpu':'cpu_suites'}[kind]
        if b['used'][key]>=LIMITS[key]:raise RuntimeError('start allowance exhausted')
        name='phdesc_'+kind+('_'+arm+'_'+task+'_'+str(fold) if arm else '')
        if any(j['name']==name for j in self.jobs):raise ValueError('duplicate job; no retry')
        active=[j for j in self.jobs if j['exit'] is None and j.get('status')!='EXIT_UNKNOWN']
        if any(j['kind']!=kind for j in active):raise ValueError('mixed stages prohibited')
        reservation={}
        if kind in ('ft','test'):
            if not can_dispatch(active,arm,task,fold):raise ValueError('physical GPU2 total4 violation')
            if unit_path(arm,task,fold,kind=='test').exists():raise FileExistsError('training/formal output exists')
            if kind=='ft':reservation=dict(epochs_started=70,validation_started=70,attempted_updates=70*math.ceil(len(read(SPLIT/f'{task}.json')['folds'][fold]['train_indices'])/24))
            else:
                frozen=read(OUT/'selection/final_frozen.json')
                if len(frozen['units'])!=70 or not frozen['frozen']:raise ValueError('TEST requires all70 frozen')
                reservation=dict(test_graphs=len(frozen['units'][f'{arm}/{task}/{fold}']['test_indices']))
        elif kind=='prepare':
            if self.started_unix is not None:raise ValueError('prepare already started')
            self.started_unix=time.time();self.manifest['wallclock_started_unix']=self.started_unix
            self.manifest['deadline_unix']=self.started_unix+129600;write(OUT/'execution_manifest.json',self.manifest)
        for k,v in reservation.items():
            if b['used'][k]+b['active_remaining_reservation'][k]+v>LIMITS[k]:raise RuntimeError('insufficient complete unit reservation')
        seconds={'prepare':14400,'ft':7200,'test':300,'cpu':300}[kind]
        seconds=min(seconds,max(1,int(self.started_unix+129600-time.time())))
        log=LOG/name;log.mkdir(parents=True,exist_ok=False)
        command=[sys.executable,'-u','scripts/run_ua2d_ph_descriptor.py',kind]
        if arm:command+=['--arm',arm,'--task',task,'--fold',str(fold)]
        if kind=='cpu':command=[sys.executable,'-m','pytest','-q','tests/test_ua2d_ph_descriptor.py']
        env=dict(CUDA_VISIBLE_DEVICES=str(gpu_for(task,fold)) if kind in ('ft','test') else '',WORLD_SIZE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        argv=['env',*[f'{k}={v}' for k,v in env.items()],'timeout','--signal=TERM','--kill-after=10s',f'{seconds}s',*command]
        script=log/'run.sh';script.write_text('#!/bin/bash\ncd '+shlex.quote(str(ROOT))+' || exit 90\ndate -Iseconds > '+shlex.quote(str(log/'start.txt'))+'\n'+shlex.join(argv)+' > '+shlex.quote(str(log/'stdout.log'))+' 2> '+shlex.quote(str(log/'stderr.log'))+'\nphdesc_exit=$?\nprintf "%s\\n" "$phdesc_exit" > '+shlex.quote(str(log/'exit.txt'))+'\ndate -Iseconds > '+shlex.quote(str(log/'end.txt'))+'\nexit "$phdesc_exit"\n')
        j=dict(name=name,kind=kind,arm=arm,task=task,fold=fold,gpu=gpu_for(task,fold) if arm else None,started=now(),exit=None,status='RUNNING',
            command=shlex.join(argv),cwd=str(ROOT),log=str(log),reservation=reservation,
            consumption=str(unit_path(arm,task,fold,kind=='test')/'consumption.json') if arm else str(INPUTS/'consumption.json') if kind=='prepare' else None)
        self.jobs.append(j);self.refresh();write(log/'command.json',j)
        j['window']=subprocess.check_output(['tmux','new-window','-d','-P','-F','#{window_id}','-t','Uni-Poly:','-n',name,'bash',str(script)],text=True).strip()
        subprocess.run(['tmux','set-option','-w','-t',j['window'],'remain-on-exit','on'],check=True);write(log/'command.json',j);self.refresh();return j

    def wait(self,jobs):
        while True:
            self.deadline();self.refresh()
            if any(j['exit'] not in (None,0) or j.get('status')=='EXIT_UNKNOWN' for j in jobs):raise RuntimeError('job failed/exit unknown; no retry')
            if all(j['exit']==0 for j in jobs):return
            time.sleep(2)

    def matrix(self,formal=False):
        if any(j['exit'] is None for j in self.jobs):raise ValueError('previous stage has not exited')
        queues={a:[(t,f) for t in TASKS for f in range(5)] for a in ARMS};active=[];kind='test' if formal else 'ft'
        while any(queues.values()) or active:
            self.deadline();self.refresh();self.check_failures();active=[j for j in active if j['exit'] is None]
            for arm in ARMS:
                while queues[arm]:
                    task,fold=queues[arm][0]
                    if not can_dispatch(active,arm,task,fold):break
                    queues[arm].pop(0);active.append(self.launch(kind,arm,task,fold))
            if active:time.sleep(2)

    def wait_assigned_gpus_idle(self):
        self.state('执行中','等待GPU2既有作业正常释放','不终止或接管其他周期；本周期尚未启动GPU任务。')
        while True:
            self.deadline()
            p=subprocess.run(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            if not p.stdout.strip():return
            time.sleep(5)

    def stop_owned(self):
            def exits():
                for j in self.jobs:
                    p=Path(j['log'])/'exit.txt'
                    if j['exit'] is None and p.exists():j.update(exit=int(p.read_text().strip()),ended=now(),status='COMPLETED' if int(p.read_text().strip())==0 else 'ABNORMAL_STOP')
            exits();active=[j for j in self.jobs if j['exit'] is None and j.get('window')]
            owned=[]
            for j in self.jobs:
                if j['kind']=='prepare':
                    p=INPUTS/'consumption.json'
                    if p.exists():
                        r=read(p)
                        if r.get('process'):owned.append(r['process'])
                        owned.extend(r.get('workers',[]))
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

    def deliver(self,error,cpu_exit):
        complete=len(self.formal)==2;status='待审查' if not error and complete and cpu_exit==0 else '需返修'
        reference=self.reference or (read(OUT/'authorization/P10K_REF.json') if (OUT/'authorization/P10K_REF.json').exists() else None)
        contrasts={};specs={'CONCAT_PH_minus_P10K_REF':('CONCAT_PH','P10K_REF'),'GATE_PH_minus_P10K_REF':('GATE_PH','P10K_REF'),'GATE_PH_minus_CONCAT_PH':('GATE_PH','CONCAT_PH')}
        blocks=dict(self.formal)
        if reference:blocks['P10K_REF']=reference
        for name,(a,b) in specs.items():
            if a not in blocks or b not in blocks:contrasts[name]=dict(status='未完成／不可比较',delta=None);continue
            tasks={}
            for task in TASKS:
                delta=finite(blocks[a]['tasks'][task]['folds'])-finite(blocks[b]['tasks'][task]['folds'])
                tasks[task]=dict(folds=delta.tolist(),mean=float(delta.mean()),std=float(delta.std(ddof=1)),positive_folds=int((delta>0).sum()))
            contrasts[name]=dict(status='COMPLETE',delta=blocks[a]['macro']-blocks[b]['macro'],tasks=tasks)
        passes={a:(contrasts[a+'_minus_P10K_REF']['delta']>=.010 if contrasts[a+'_minus_P10K_REF']['delta'] is not None else None) for a in ARMS}
        partial={}
        for a in ARMS:
            if a not in self.validation:
                partial[a]={str(p.parent.relative_to(OUT)):read(p) for p in (OUT/'finetune'/a).glob('*/fold*/metrics.json') if read(p).get('status')=='COMPLETE'}
        payload=dict(plan=PLAN,revision=REVISION,status=status,error=error,cpu_exit=cpu_exit,partial_validation=partial,FINAL=FINAL,formal=self.formal,validation=self.validation,
            reference=reference,contrasts=contrasts,performance_pass=passes,budget=self.refresh(),complete_formal_TEST=complete,independent_review='PENDING',
            protocol='shared development TEST; no capacity control: entire fusion effect, no PH-only/gate interaction attribution; both fixed candidates; no reselection')
        lines=[f'# {PLAN} / {REVISION} 交付','',f'状态：{status}；固定主候选GATE_PH；CPU exit={cpu_exit}。','执行自检不替代Codex独立审查。','',
            '| 对照 | 正式Δmacro R² |','|---|---:|']
        for name,c in contrasts.items():lines.append('| '+name+' | '+(format(c['delta'],'+.3f') if c['delta'] is not None else '未完成／不可比较')+' |')
        rows={}
        for arm in ('P10K_REF',*ARMS):
            block=blocks.get(arm);rows[arm]={};lines+=['',f'## {arm} 七任务数值对照','','| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |','|---|---:|---:|---:|---|']
            for task,(label,model,score) in SOTA.items():
                value=block['tasks'][task] if block else None
                shown=f"{value['mean']:.3f}±{value['std']:.3f}" if value else '未完成／不可比较'
                diff=f"{value['mean']-score:+.3f}" if value else '—';lines.append(f'| {label} | {shown} | {score:.3f} | {diff} | {model} |')
                rows[arm][task]=dict(project=value,reference=score,reference_model=model,delta=value['mean']-score if value else None)
            if block:lines.append(f"macro R²={block['macro']:.3f}。")
        lines+=['','五折seed42+fold等权mean/std(ddof1)，再七任务等权macro；使用未舍入值判定0.010门槛。',
            '论文参考为逐任务包络，协议未证明等价，仅数值比较；共享开发TEST非独立盲测。新增模型容量也改变，不将差值单独归因于PH或门控机制。',
            '异常／未完成：'+str(error or '无')]
        payload['sota']=rows;write(OUT/'final_review/results.json',payload)
        write(OUT/'final_review/training_timing.json',dict(epochs={str(p.relative_to(OUT)):read(p) for p in OUT.glob('finetune/*/*/fold*/epoch_timing.json')},jobs=self.jobs,
            elapsed_seconds=time.time()-self.started_unix if self.started_unix else None,real_training_only=True))
        report='\n'.join(lines)+'\n';(OUT/'final_review/report.md').write_text(report)
        self.state(status,'实际交付',f'完整TEST={complete}；CPU exit={cpu_exit}；报告{OUT}/final_review/report.md')
        self.manifest.update(status=status,ended=now(),jobs=self.jobs);write(OUT/'execution_manifest.json',self.manifest);self.capture_diff()
        append_plan(report);return status

    def run(self):
        error=None;cpu_exit=None;safe=True
        try:
            self.state('执行中','正常prepare','一次CPU准备：离线计算固定412维PH、保存barcode/映射/共同初始化；CPU workers≤4，无新构象或PT。')
            j=self.launch('prepare');self.wait([j]);self.reference=read(OUT/'authorization/P10K_REF.json')
            self.wait_assigned_gpus_idle();self.state('执行中','70FT初始化','仅按各fold有效训练样本拟合共享PH feature scaler，原target scaler只读复用。');ensure_feature_scalers();self.state('执行中','70FT','仅GPU2，两组该卡全部作业合计≤4；同task/fold同卡，70FT完成后才冻结统一TEST。')
            self.matrix()
            self.validation={a:aggregate(a) for a in ARMS};write(OUT/'selection/validation.json',self.validation)
            self.frozen=freeze_final(self.validation);self.state('执行中','70正式TEST','70BEST一次冻结完成；无validation性能准入，每单元只评一次。')
            self.matrix(True);self.formal={a:aggregate(a,True) for a in ARMS}
        except BaseException:
            error=traceback.format_exc();write(OUT/'failure.json',dict(error=error,time=now(),automatic_retry=False));self.evidence_only=True
            try:self.stop_owned()
            except BaseException:error+='\n'+traceback.format_exc();safe=False
        finally:
            for a in ARMS:
                try:self.validation[a]=aggregate(a)
                except (OSError,ValueError,KeyError):pass
                if self.frozen:
                    try:self.formal[a]=aggregate(a,True)
                    except (OSError,ValueError,KeyError):pass
            if safe:
                try:
                    self.state('执行中' if not error else '需返修','唯一最终CPU suite','训练/正式TEST结束或安全异常收尾；只运行新增专项suite一次，不重测。')
                    j=self.launch('cpu',final_cpu=True);self.wait([j]);cpu_exit=j['exit']
                except BaseException:
                    error=(error or '')+'\n'+traceback.format_exc()
                    cpu_exit=next((j['exit'] for j in self.jobs if j['kind']=='cpu'),None)
            status=self.deliver(error,cpu_exit);return 0 if status=='待审查' else 1


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','prepare','ft','test']);p.add_argument('--arm',choices=ARMS);p.add_argument('--task',choices=TASKS);p.add_argument('--fold',type=int,choices=range(5));args=p.parse_args()
    torch.set_num_threads(1)
    def stop(signum,frame):raise RuntimeError('controlled termination '+str(signum))
    signal.signal(signal.SIGTERM,stop)
    if args.action=='run':return Controller().run()
    if args.action=='prepare':prepare();return 0
    if args.arm is None or args.task is None or args.fold is None:p.error('complete fixed unit identity required')
    unit(args.arm,args.task,args.fold,args.action=='test');return 0
if __name__=='__main__':raise SystemExit(main())
