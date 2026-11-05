#!/usr/bin/env python3
"""Single GPU2, sequential, one-shot Stage0 controller. No retry or resume."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(key,'2')
import argparse
import difflib
import fcntl
import json
import math
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training import ua3d_base_opt as t
from scripts.run_ua2d_a020_fp_fusion import atomic_exit,poll_exit,owned_group

SOURCES=('src/modules/ua3d_base_opt.py','src/training/ua3d_base_opt.py',
    'scripts/pretrain_ua3d_base_opt.py','scripts/run_ua3d_base_opt.py',
    'configs/ua3d_base_opt_r5.json','tests/test_ua3d_base_opt.py')


def gpu_inventory():
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True,timeout=10)
    values=[line.split(',')[1].strip() for line in lines.splitlines() if line.split(',')[0].strip()=='2']
    if len(values)!=1:raise RuntimeError('physical GPU2 inventory mismatch')
    uuid=values[0]
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=10)
    pids=[int(line.split(',')[1]) for line in apps.splitlines() if line.split(',')[0].strip()==uuid]
    return dict(physical_gpu=2,uuid=uuid,pids=pids,time=t.now())


def reservation(kind,task=None,fold=None):
    if kind=='initialize':return {}
    if kind=='pt':return dict(pt_starts=1,pt_updates=10000,pt_microbatches=60000,pt_presentations=10080000)
    if kind=='cpu':return dict(cpu_suites=1)
    entry=t.read(t.SPLIT/f'{task}.json')['folds'][fold]
    if kind=='ft':return dict(ft_starts=1,ft_epochs=70,ft_updates=70*math.ceil(len(entry['train_indices'])/24))
    if kind=='test':return dict(test_starts=1,test_graphs=len(entry['test_indices']))
    raise ValueError('unregistered kind')


def ledger(jobs):
    charged=dict.fromkeys(t.LIMITS,0);actual=dict.fromkeys(t.LIMITS,0)
    for job in jobs:
        for key,value in job['reservation'].items():charged[key]+=value
        start={'pt':'pt_starts','ft':'ft_starts','test':'test_starts','cpu':'cpu_suites'}.get(job['kind'])
        if start:actual[start]+=1
        path=Path(job['consumption']) if job.get('consumption') else None
        if path is not None and path.exists():
            record=t.read(path)
            for key in actual:
                if key.endswith('starts') or key=='cpu_suites':continue
                if job['kind']=='ft' and key=='ft_epochs':value=record.get('epochs_started',0)
                elif job['kind']=='ft' and key=='ft_updates':value=record.get('attempted_updates',0)
                else:value=record.get(key,0)
                actual[key]+=int(value)
    bad=[k for k in charged if charged[k]>t.LIMITS[k] or actual[k]>charged[k]]
    return dict(limits=t.LIMITS,charged_or_reserved=charged,known_actual_lower_bound=actual,
        budget_violations=bad,jobs=jobs,zero_cost=t.ZERO_COST,
        accounting='full reservation charged before dispatch; failed/partial attempts never refunded; durable actual separately')


def wrap(spec):
    job=t.read(spec);folder=Path(job['log']);env=os.environ.copy();env.update(job['env']);identity=None;forwarded=False
    def stop(sig,frame):
        nonlocal forwarded
        if not forwarded and identity is not None and t.process_alive(identity):
            forwarded=True;os.killpg(identity['pid'],signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (folder/'stdout.log').open('x') as out,(folder/'stderr.log').open('x') as err:
        child=subprocess.Popen(job['argv'],cwd=job['cwd'],env=env,stdout=out,stderr=err,start_new_session=True)
        identity=t.process_identity(child.pid)
        t.write(folder/'process.json',dict(wrapper=t.process_identity(os.getpid()),timeout_identity=identity,
            command=job['argv'],cwd=job['cwd'],started=t.now(),gpu_uuid=job['gpu_uuid']))
        rc=child.wait()
    atomic_exit(folder/'exit.txt',rc)
    t.write(folder/'end.json',dict(exit=rc,ended=t.now(),job=job['name']))
    return rc


class Controller:
    def __init__(self):
        t.require_tmux()
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('controller must hide GPUs')
        # Check BOTH roots before creating either. Existing roots are never adopted.
        if t.OUT.exists() or t.LOG.exists():raise FileExistsError('r5 root exists; stop without reuse')
        t.OUT.mkdir(parents=True,exist_ok=False);t.LOG.mkdir(parents=True,exist_ok=False)
        self.started=time.time();self.jobs=[];self.active=None
        sys.stdout=(t.LOG/'controller.stdout.log').open('x',buffering=1)
        sys.stderr=(t.LOG/'controller.stderr.log').open('x',buffering=1)
        self.lock=(t.OUT/'controller.lock').open('x');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.manifest=dict(plan=t.PLAN,revision=t.REVISION,status='执行中',started=t.now(),cwd=str(ROOT),
            command=sys.argv,controller_identity=t.process_identity(os.getpid()),tmux=os.environ.get('TMUX_PANE'),
            based_on=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),physical_gpu=2,
            limits=t.LIMITS,automatic_retry=False,automatic_resume=False,independent_review='PENDING')
        t.write(t.OUT/'execution_manifest.json',self.manifest)
        t.exclusive(t.OUT/'authorization/user_authorization.json',dict(plan=t.PLAN,revision=t.REVISION,stage='Stage0',
            source='user supplied fixed Stage0 execution prompt in this conversation',physical_gpu=2,
            stages1_to5_authorized=False,limits=t.LIMITS,automatic_retry=False,automatic_resume=False))
        (t.OUT/'authorization/CURRENT_PLAN.before.md').write_text((ROOT/'CURRENT_PLAN.md').read_text())
        t.write(t.OUT/'authorization/protected_status.json',dict(status=subprocess.check_output(
            ['git','status','--short'],text=True),todo_read=False,remote_sync=False))
        t.exclusive(t.OUT/'config.json',t.read(ROOT/'configs/ua3d_base_opt_r5.json'))
        self.capture_diff();self.refresh()

    def capture_diff(self):
        diff=[]
        for name in SOURCES:
            diff.extend(difflib.unified_diff([], (ROOT/name).read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+name))
        if (t.OUT/'authorization/CURRENT_PLAN.before.md').exists():
            diff.extend(difflib.unified_diff((t.OUT/'authorization/CURRENT_PLAN.before.md').read_text().splitlines(True),
                (ROOT/'CURRENT_PLAN.md').read_text().splitlines(True),fromfile='before/CURRENT_PLAN.md',tofile='after/CURRENT_PLAN.md'))
        path=t.OUT/'final_review/actual_changes.diff';path.parent.mkdir(exist_ok=True);path.write_text(''.join(diff))

    def refresh(self):
        for job in self.jobs:
            if job['exit'] is None:
                rc=poll_exit(job)
                if rc is not None:job.update(exit=rc,ended=t.now())
        value=ledger(self.jobs);value.update(time=t.now(),elapsed_seconds=time.time()-self.started)
        t.write(t.OUT/'budget.json',value)
        if value['budget_violations']:raise RuntimeError('budget violation '+repr(value['budget_violations']))
        return value

    def launch(self,kind,arm=None,task=None,fold=None):
        self.refresh()
        if any(j['exit'] is None for j in self.jobs):raise RuntimeError('overlapping jobs forbidden')
        if any(j['exit']!=0 for j in self.jobs):raise RuntimeError('failed chain cannot continue')
        gpu=kind in ('pt','ft','test');uuid=''
        if gpu:
            while True:
                inventory=gpu_inventory();t.write(t.OUT/'occupancy_wait.json',inventory)
                if not inventory['pids']:uuid=inventory['uuid'];break
                t.update_plan('执行中','等待 GPU2','GPU2 有其他进程；保持队列，不抢占或换卡。');time.sleep(10)
        name='base_opt_r5_'+kind+(f'_{arm}' if arm else '')+(f'_{task}_{fold}' if task else '')
        if any(j['name']==name for j in self.jobs):raise ValueError('duplicate launch')
        reserve=reservation(kind,task,fold);charged=ledger(self.jobs)['charged_or_reserved']
        if any(charged[k]+v>t.LIMITS[k] for k,v in reserve.items()):raise RuntimeError('budget unavailable')
        if kind=='pt':consumption=t.OUT/'pretrain'/arm/'consumption.json';output=consumption.parent
        elif kind in ('ft','test'):output=t.unit_path(arm,task,fold,kind=='test');consumption=output/'consumption.json'
        else:output=None;consumption=None
        if output is not None and output.exists():raise FileExistsError('writer/overwrite conflict '+str(output))
        log=t.LOG/name;log.mkdir(exist_ok=False)
        if kind=='cpu':command=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_ua3d_base_opt.py'];command=['timeout','--signal=TERM','--kill-after=30s','600s',*command]
        elif kind=='pt':command=[sys.executable,'-u','scripts/pretrain_ua3d_base_opt.py','--arm',arm,'--physical-gpu','2']
        else:
            command=[sys.executable,'-u','scripts/run_ua3d_base_opt.py',kind]
            if arm:command+=['--arm',arm,'--task',task,'--fold',str(fold),'--physical-gpu','2']
        command=['env','CUDA_VISIBLE_DEVICES=2' if gpu else 'CUDA_VISIBLE_DEVICES=',*command]
        env=dict(CUDA_VISIBLE_DEVICES='2' if gpu else '',WORLD_SIZE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2',UA3M_PHYSICAL_GPU_UUID=uuid,PYTHONDONTWRITEBYTECODE='1')
        job=dict(name=name,kind=kind,arm=arm,task=task,fold=fold,mode='FULL',physical_gpu=2 if gpu else None,
            gpu_uuid=uuid,argv=command,env=env,cwd=str(ROOT),log=str(log),reservation=reserve,
            consumption=str(consumption) if consumption else None,exit=None,started=t.now(),attempt='r5-stage0-once')
        self.jobs.append(job);self.refresh();t.write(log/'command.json',job)
        cmd=['env','CUDA_VISIBLE_DEVICES=',sys.executable,'-u','scripts/run_ua3d_base_opt.py','wrap','--spec',str(log/'command.json')]
        job['window']=subprocess.check_output(['tmux','new-window','-d','-P','-F','#{window_id}',
            '-t','Uni-Poly:','-n',name,'-c',str(ROOT),shlex.join(cmd)],text=True).strip()
        subprocess.run(['tmux','set-option','-w','-t',job['window'],'remain-on-exit','on'],check=True)
        pane=int(subprocess.check_output(['tmux','display-message','-p','-t',job['window'],'#{pane_pid}'],text=True))
        job['wrapper_identity']=t.process_identity(pane);self.active=job;self.refresh()
        t.update_plan('执行中',kind,f'当前 {name}；所有 GPU 任务物理 GPU2、cuda:0、world_size=1、单路串行。')
        return job

    def wait(self,job):
        while job['exit'] is None:self.refresh();time.sleep(5)
        self.active=None
        if job['exit']!=0:raise RuntimeError(f"{job['name']} true exit={job['exit']}; no retry/resume")

    def report(self,status,error=None):
        formal={};partial={};costs={};sota={}
        for arm in t.ARMS:
            try:formal[arm]=t.aggregate(arm,True)
            except (OSError,ValueError,KeyError):partial[arm]='未完成／不可比较'
            p=t.OUT/'pretrain'/arm/'consumption.json'
            costs[arm]=dict(pt=t.read(p) if p.exists() else None)
            p=t.OUT/'pretrain'/arm/'config.json'
            costs[arm]['pt_config']=t.read(p) if p.exists() else None
            costs[arm]['full']=[]
            for task in t.TASKS:
                for fold in range(5):
                    folder=t.unit_path(arm,task,fold)
                    if (folder/'consumption.json').exists():costs[arm]['full'].append(dict(task=task,fold=fold,
                        consumption=t.read(folder/'consumption.json'),parameters=t.read(folder/'parameters.json') if (folder/'parameters.json').exists() else None))
        path=t.OUT/'selection/final_frozen.json';frozen=t.read(path) if path.exists() else {}
        path=t.OUT/'selection/validation_selection.json';selection=t.read(path) if path.exists() else {}
        chosen=frozen.get('PT_DIRECTION_SELECTED');budget=self.refresh()
        comparisons={arm:dict(delta_macro=v['macro']-t.REFERENCE,performance_pass=v['macro']-t.REFERENCE>=.010) for arm,v in formal.items()}
        selected=comparisons.get(chosen)
        cpu_jobs=[j for j in self.jobs if j['kind']=='cpu']
        cpu=dict(attempts=len(cpu_jobs),exit=cpu_jobs[0]['exit'] if cpu_jobs else None,log=cpu_jobs[0]['log'] if cpu_jobs else None)
        t.write(t.OUT/'final_review/cpu_exit.json',cpu)
        pairs={}
        for a,b in (('P_REVERSE','P_BASE'),('P_SYM','P_BASE'),('P_SYM_IND','P_SYM')):
            if a in formal and b in formal:pairs[a+'_minus_'+b]=dict(delta_macro=formal[a]['macro']-formal[b]['macro'],
                task_fold_deltas={task:(t.finite(formal[a]['tasks'][task]['folds'])-t.finite(formal[b]['tasks'][task]['folds'])).tolist() for task in t.TASKS})
        result=dict(plan=t.PLAN,revision=t.REVISION,stage='Stage0',status=status,independent_review='PENDING',
            reference=t.REFERENCE,PT_DIRECTION_SELECTED=chosen,selected_comparison=selected,
            formal=formal,partial=partial,validation=selection.get('validation'),comparisons=comparisons,
            controlled_comparisons=pairs,costs=costs,budget=budget,cpu=cpu,error=error,
            test_scope='shared development evaluation',ft_policy=t.m.CONFIG['ft_policy'])
        lines=['# UA3D BASE OPT r5 Stage0','',f'状态：{status}；independent_review=PENDING。',
            f'PT_DIRECTION_SELECTED={chosen}；BASE reference 完整精度 {t.REFERENCE!r}。',
            '四臂使用本周期公共 step0。仅 PT 方向／每项系数／投影共享不同。下游统一原三方向 ALIGN、共享 Q2/Q3；PT-only QFP 不进入 FT。',
            '四臂不是完整 factorial，不推断 interaction；投影差异伴随 PT 参数容量变化。',
            '共享开发 TEST，非独立盲测。validation 全部完成后唯一冻结；TEST 不用于重选。','',
            '| Arm | TEST macro R² | 相对 BASE Δmacro R² | 达到 +0.010 |','|---|---:|---:|---|']
        for arm in t.ARMS:
            v=formal.get(arm);c=comparisons.get(arm)
            lines.append(f'| {arm} | '+(f'{v["macro"]:.3f} | {c["delta_macro"]:+.3f} | {c["performance_pass"]}' if v else '未完成／不可比较 | — | 未判定')+' |')
            sota[arm]={}
            lines+=['',f'## {arm} 七任务论文数值对照','', '| 任务 | 项目 R² mean±std | 论文最佳 | Δ | 论文模型 |','|---|---:|---:|---:|---|']
            for task,(name,model,ref) in t.SOTA.items():
                x=v['tasks'][task] if v else None
                sota[arm][task]=dict(project=x,reference=ref,reference_model=model,delta=x['mean']-ref if x else None,reference_source='SOTA_RESULT.md')
                lines.append(f'| {name} | '+(f'{x["mean"]:.3f}±{x["std"]:.3f} | {ref:.3f} | {x["mean"]-ref:+.3f} | {model}' if x else f'未完成／不可比较 | {ref:.3f} | — | {model}')+' |')
        result['sota']=sota
        lines+=['','主要门槛以 validation 唯一所选臂相对指定历史 BASE 的未舍入 Δmacro≥0.010 判断。',
            '本周期新 P_BASE 用于受控 PT 比较；历史 BASE 只作外部 reference，不自动替换。',
            '五折 mean±std(ddof=1)、seed42+fold、七任务等权；论文是逐任务最佳包络。数据／split／统计协议等价性未证实，仅数值对照。',
            '每臂 PT history.jsonl/sampling.jsonl/config.json/consumption.json 记录方向实际正对、损失、总 ALIGN 暴露、参数、时间与显存。',
            'BEST、validation、scaler、config 位于 finetune/FULL；原始 TEST 位于 formal/FULL；冻结和 row/sample-key 记录位于 selection。',
            '完整精度及命令、cwd、PID、UUID、起止、日志和真实 exit 见 results.json、budget.json 与 logs 对应作业。',
            '实际预算：'+json.dumps(budget['known_actual_lower_bound'],ensure_ascii=False),f'CPU suite：{cpu!r}。',
            f'异常：{error or "无"}。Stage1—5 未授权，未启动。']
        t.write(t.OUT/'results.json',result);t.write(t.OUT/'final_review/results.json',result)
        report='\n'.join(lines)+'\n';(t.OUT/'report.md').write_text(report);(t.OUT/'final_review/report.md').write_text(report)
        t.update_plan(status,'交付' if status=='待审查' else '停止' if error else '汇总',
            f'Stage0：{len(formal)}/4 正式 TEST 臂完成；所选 {chosen}；CPU {cpu}；'+(f'阻断：{error.splitlines()[-1]}' if error else '证据见本周期 report.md/results.json。'))

    def stop_owned(self):
        t.ignore_termination()
        for job in self.jobs:
            if job['exit'] is not None:continue
            group=owned_group(job)
            if group is not None:
                os.killpg(group,signal.SIGTERM)
                for _ in range(30):
                    if owned_group(job) is None:break
                    time.sleep(1)
                if owned_group(job) is not None:os.killpg(group,signal.SIGKILL)
            for _ in range(10):
                rc=poll_exit(job)
                if rc is not None:job.update(exit=rc,ended=t.now());break
                time.sleep(1)

    def run(self):
        try:
            self.wait(self.launch('initialize'))
            for arm in t.ARMS:self.wait(self.launch('pt',arm))
            for arm in t.ARMS:
                for task in t.TASKS:
                    for fold in range(5):self.wait(self.launch('ft',arm,task,fold))
            frozen=t.freeze()
            t.update_plan('执行中','一次冻结',f'140 FULL 已冻结，PT_DIRECTION_SELECTED={frozen["PT_DIRECTION_SELECTED"]}；开始固定140 TEST。')
            for arm in t.ARMS:
                for task in t.TASKS:
                    for fold in range(5):self.wait(self.launch('test',arm,task,fold))
            self.report('执行中')
            self.wait(self.launch('cpu'))
            self.report('待审查');rc=0
        except BaseException:
            error=traceback.format_exc()
            try:self.stop_owned()
            except BaseException:error+='\nstop: '+traceback.format_exc()
            t.write(t.OUT/'failure.json',dict(error=error,time=t.now(),automatic_retry=False,automatic_resume=False))
            self.report('待审查',error);rc=1
        self.capture_diff();self.manifest.update(status='待审查',run_outcome='COMPLETE' if rc==0 else 'STOPPED',ended=t.now())
        t.write(t.OUT/'execution_manifest.json',self.manifest)
        t.write(t.OUT/'controller_exit.json',dict(exit=rc,time=t.now(),process=t.process_identity(os.getpid())))
        return rc


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('controller','initialize','ft','test','wrap'))
    p.add_argument('--arm',choices=t.ARMS);p.add_argument('--task',choices=t.TASKS);p.add_argument('--fold',type=int,choices=range(5))
    p.add_argument('--physical-gpu',type=int,choices=(2,),default=2);p.add_argument('--spec');args=p.parse_args()
    t.torch.set_num_threads(2)
    if args.action=='wrap':return wrap(args.spec)
    def stop(sig,frame):t.ignore_termination();raise RuntimeError('controlled termination '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.action=='controller':return Controller().run()
    if args.action=='initialize':t.initialize_cycle();return 0
    if args.arm is None or args.task is None or args.fold is None:p.error('fixed estimator identity required')
    t.unit(args.arm,args.task,args.fold,args.action=='test',args.physical_gpu);return 0

if __name__=='__main__':raise SystemExit(main())
