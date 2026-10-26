#!/usr/bin/env python3
"""Single-writer UA3D campaign; fixed one-shot budget and global TEST freeze."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(key,'2')
import argparse
import difflib
import fcntl
import math
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training import ua3d_spatial as t
from scripts.run_ua2d_a020_fp_fusion import atomic_exit,poll_exit,ExitProtocolError,owned_group,pid_aliases
SOURCES=('src/modules/ua3d_spatial.py','src/dataset/ua3d_spatial.py','src/training/ua3d_spatial.py',
         'scripts/pretrain_ua3d_spatial.py','scripts/run_ua3d_spatial.py','configs/ua3d_spatial_r4.json','tests/test_ua3d_spatial.py')


def gpu_inventory():
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True,timeout=10)
    result={}
    for line in devices.splitlines():
        index,uuid=[v.strip() for v in line.split(',')];index=int(index)
        if index in t.ALLOWED_GPUS:result[index]=dict(uuid=uuid,pids=set())
    if set(result)!=set(t.ALLOWED_GPUS):raise RuntimeError('authorized card inventory incomplete')
    by_uuid={v['uuid']:v for v in result.values()}
    if len(by_uuid)!=len(result):raise RuntimeError('ambiguous card UUID')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=10)
    for line in apps.splitlines():
        uuid,pid=[v.strip() for v in line.split(',')]
        if uuid in by_uuid:by_uuid[uuid]['pids'].add(int(pid))
    return result


def reservation(kind,task=None,fold=None,mode='FULL'):
    if kind=='prepare':return dict(prepare_starts=1,cache_rows=5875)
    if kind=='pt':return dict(pt_starts=1,pt_updates=10000,pt_microbatches=60000,pt_presentations=10080000)
    if kind=='cpu':return dict(cpu_suites=1)
    entry=t.read(t.SPLIT/f'{task}.json')['folds'][fold]
    if kind=='ft':return dict(ft_starts=1,ft_epochs=70,ft_updates=70*math.ceil(len(entry['train_indices'])/24))
    if kind=='probe':return dict(probe_starts=1,probe_epochs=30,probe_updates=30*math.ceil(len(entry['train_indices'])/24),feature_rows=len(set(entry['train_indices'])|set(entry['validation_indices'])))
    if kind=='test':return dict(test_starts=1,test_graphs=len(entry['test_indices']))
    raise ValueError('unregistered phase')


def ledger(jobs):
    charged=dict.fromkeys(t.LIMITS,0);actual=dict.fromkeys(t.LIMITS,0)
    for job in jobs:
        for key,value in job['reservation'].items():charged[key]+=value
        starts={'prepare':'prepare_starts','pt':'pt_starts','ft':'ft_starts','probe':'probe_starts','test':'test_starts','cpu':'cpu_suites'}
        actual[starts[job['kind']]]+=1
        path=Path(job['consumption']) if job.get('consumption') else None
        if path is not None and path.exists():
            record=t.read(path)
            if job['kind']=='prepare':actual['cache_rows']+=record.get('records',0)
            for key in actual:
                if key.endswith('starts') or key in ('cpu_suites','cache_rows'):continue
                actual[key]+=int(record.get(key,0))
    bad=[k for k in charged if charged[k]>t.LIMITS[k] or actual[k]>charged[k]]
    return dict(limits=t.LIMITS,charged_or_reserved=charged,known_actual_lower_bound=actual,budget_violations=bad,
        jobs=jobs,zero_cost=t.ZERO_COST,accounting='full unit irreversibly charged at dispatch; failed or partial units never refunded; durable actual separately')


class Controller:
    def __init__(self):
        t.require_tmux()
        # Root creation is exclusive and occurs inside this first controller launch.
        t.OUT.mkdir(parents=True,exist_ok=False);t.LOG.mkdir(parents=True,exist_ok=False)
        self.started=time.time();self.deadline_unix=self.started+48*3600
        sys.stdout=(t.LOG/'controller.stdout.log').open('x',buffering=1);sys.stderr=(t.LOG/'controller.stderr.log').open('x',buffering=1)
        self.lock=(t.OUT/'controller.lock').open('x');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.jobs=[];self.stopping=False
        (t.OUT/'final_review').mkdir()
        auth=dict(plan=t.PLAN,revision=t.REVISION,source='user: 请你直接执行！; then user: 请使用 4 GPU并行！',
            allowed_gpus=list(t.ALLOWED_GPUS),limits=t.LIMITS,automatic_retry=False,automatic_resume=False)
        t.exclusive(t.OUT/'authorization/user_authorization.json',auth)
        self.manifest=dict(plan=t.PLAN,revision=t.REVISION,status='执行中',started=t.now(),started_unix=self.started,
            deadline_unix=self.deadline_unix,deadline=datetime_iso(self.deadline_unix),cwd=str(ROOT),command=sys.argv,
            controller_identity=t.process_identity(os.getpid()),tmux_window=os.environ['UA3D_CONTROLLER_WINDOW'],allowed_gpus=list(t.ALLOWED_GPUS),
            based_on='dev/cd36ae7280b159f881b483ad5814835ff7b74c4a plus protected dirty worktree',limits=t.LIMITS)
        t.write(t.OUT/'execution_manifest.json',self.manifest)
        (t.OUT/'authorization/CURRENT_PLAN.before.md').write_text((ROOT/'CURRENT_PLAN.md').read_text())
        self.capture_diff();self.refresh()

    def deadline(self):
        if time.time()>=self.deadline_unix:raise TimeoutError('48h wallclock exhausted')

    def capture_diff(self):
        changes=[]
        for name in SOURCES:changes.extend(difflib.unified_diff([], (ROOT/name).read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+name))
        changes.extend(difflib.unified_diff((t.OUT/'authorization/CURRENT_PLAN.before.md').read_text().splitlines(True),(ROOT/'CURRENT_PLAN.md').read_text().splitlines(True),fromfile='before/CURRENT_PLAN.md',tofile='after/CURRENT_PLAN.md'))
        (t.OUT/'final_review/actual_changes.diff').write_text(''.join(changes))

    def collect_exits(self):
        errors=[]
        for job in self.jobs:
            if job['exit'] is not None:continue
            try:
                rc=poll_exit(job)
                if rc is not None:job.update(exit=rc,ended=t.now())
            except (ExitProtocolError,OSError,ValueError,KeyError) as error:job['infrastructure_failure']=str(error);errors.append(str(error))
        return errors

    def refresh(self):
        errors=self.collect_exits();value=ledger(self.jobs)
        value.update(time=t.now(),elapsed_seconds=time.time()-self.started,deadline_unix=self.deadline_unix,exit_protocol_errors=errors)
        t.write(t.OUT/'budget.json',value)
        if not self.stopping and (errors or value['budget_violations']):raise RuntimeError('exit or budget protocol failure: '+repr(errors or value['budget_violations']))
        return value

    def check_failures(self):
        bad=[j for j in self.jobs if j['exit'] not in (None,0) or j.get('infrastructure_failure')]
        if bad:raise RuntimeError('job failed; no automatic retry: '+bad[0]['name']+' exit='+str(bad[0]['exit']))

    def available_gpu(self,kind):
        active=[j for j in self.jobs if j['kind'] in ('pt','ft','probe','test') and j['exit'] is None]
        if kind=='pt' and any(j['kind']!='pt' for j in active):raise ValueError('PT/FT overlap forbidden')
        if kind!='pt' and any(j['kind']=='pt' for j in active):raise ValueError('FT before all PT complete')
        if kind=='probe' and sum(j['kind']=='probe' for j in active)>=4:return None
        if kind=='test' and sum(j['mode']=='probe' for j in active)>=4:return None
        try:snapshot=gpu_inventory()
        except (OSError,ValueError,subprocess.SubprocessError,RuntimeError) as error:
            t.write(t.OUT/'occupancy_wait.json',dict(time=t.now(),error=str(error),action='wait_no_dispatch'));return None
        observations={};choices=[]
        for gpu in t.ALLOWED_GPUS:
            on_card=[j for j in active if j['physical_gpu']==gpu];owned=set()
            for job in on_card:
                path=Path(job['consumption'])
                if path.exists():
                    identity=t.read(path).get('process')
                    if identity:owned.update(pid_aliases(identity))
            unknown=snapshot[gpu]['pids']-owned
            limit=1 if kind=='pt' else 4
            observations[gpu]=dict(uuid=snapshot[gpu]['uuid'],active=len(on_card),known=sorted(owned),unknown=sorted(unknown))
            if not unknown and len(on_card)<limit:choices.append((len(on_card),gpu))
        t.write(t.OUT/'occupancy_wait.json',dict(time=t.now(),cards=observations,action='dispatch_available' if choices else 'wait_no_dispatch'))
        return min(choices)[1] if choices else None

    def launch(self,kind,arm=None,task=None,fold=None,mode='FULL'):
        self.deadline();self.refresh();self.check_failures()
        gpu=self.available_gpu(kind) if kind in ('pt','ft','probe','test') else None
        if kind in ('pt','ft','probe','test') and gpu is None:return None
        name='ua3d_r4_'+kind+(f'_{mode}_{arm}' if arm else '')+(f'_{task}_{fold}' if task else '')
        if any(j['name']==name for j in self.jobs):raise ValueError('duplicate campaign launch')
        r=reservation(kind,task,fold,mode);charged=ledger(self.jobs)['charged_or_reserved']
        if any(charged[k]+v>t.LIMITS[k] for k,v in r.items()):raise RuntimeError('reservation allowance unavailable')
        if kind=='pt' and (t.OUT/'pretrain'/arm).exists():raise FileExistsError('PT writer conflict')
        if kind in ('ft','probe','test') and t.unit_path(arm,task,fold,kind=='test',mode).exists():raise FileExistsError('estimator writer conflict')
        log=t.LOG/name;log.mkdir(exist_ok=False)
        if kind=='cpu':command=['python','-m','pytest','-q','-p','no:cacheprovider','tests/test_ua3d_spatial.py']
        elif kind=='pt':command=['python','-u','scripts/pretrain_ua3d_spatial.py','--arm',arm,'--physical-gpu',str(gpu)]
        else:
            command=['python','-u','scripts/run_ua3d_spatial.py',kind]
            if arm:command+=['--arm',arm,'--task',task,'--fold',str(fold),'--mode',mode,'--physical-gpu',str(gpu)]
        seconds=min(600 if kind=='cpu' else 48*3600,max(1,int(self.deadline_unix-time.time())))
        argv=['timeout','--signal=TERM','--kill-after=30s',str(seconds)+'s',*command]
        uuid=t.read(t.OUT/'occupancy_wait.json')['cards'][str(gpu)]['uuid'] if gpu is not None else ''
        env=dict(CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else '',WORLD_SIZE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',UA3M_PHYSICAL_GPU_UUID=uuid,UA3D_DEADLINE_UNIX=str(self.deadline_unix))
        consumption=t.OUT/'inputs/coverage.json' if kind=='prepare' else t.OUT/'pretrain'/arm/'consumption.json' if kind=='pt' else t.unit_path(arm,task,fold,kind=='test',mode)/'consumption.json' if kind in ('ft','probe','test') else None
        job=dict(name=name,kind=kind,arm=arm,task=task,fold=fold,mode=mode,physical_gpu=gpu,gpu_uuid=uuid,argv=argv,env=env,cwd=str(ROOT),
            log=str(log),reservation=r,consumption=str(consumption) if consumption else None,exit=None,started=t.now(),attempt='r4-once')
        self.jobs.append(job);self.refresh();t.write(log/'command.json',job)
        cmd=['env','CUDA_VISIBLE_DEVICES=','python','-u','scripts/run_ua3d_spatial.py','wrap','--spec',str(log/'command.json')]
        job['window']=subprocess.check_output(['tmux','new-window','-d','-P','-F','#{window_id}','-t','Uni-Poly:','-n',name,'-c',str(ROOT),shlex.join(cmd)],text=True).strip()
        subprocess.run(['tmux','set-option','-w','-t',job['window'],'remain-on-exit','on'],check=True)
        pid=int(subprocess.check_output(['tmux','display-message','-p','-t',job['window'],'#{pane_pid}'],text=True));job['wrapper_identity']=t.process_identity(pid)
        self.refresh();return job

    def wait(self,jobs):
        while True:
            self.deadline();self.refresh();self.check_failures()
            if all(j is not None and j['exit']==0 for j in jobs):return
            time.sleep(5)

    def stage(self,kind,matrix):
        queue=list(matrix);launched=[];last=None
        while queue or any(j['exit'] is None for j in launched):
            self.deadline();self.refresh();self.check_failures()
            while queue:
                arm,task,fold,mode=queue[0];job=self.launch(kind,arm,task,fold,mode)
                if job is None:break
                launched.append(job);queue.pop(0)
            complete=sum(j['exit']==0 for j in launched);active=sum(j['exit'] is None for j in launched)
            if (complete,active)!=last:
                t.update_plan('执行中',kind,f'{kind}完成{complete}/{len(matrix)}，运行中{active}，排队{len(queue)}；仅GPU0/1/2/3，原48小时截止不刷新。');last=(complete,active)
            if queue or active:time.sleep(5)

    def stop_owned(self):
        self.stopping=True;t.ignore_termination();end=time.monotonic()+30;sent=set();errors=[]
        while True:
            errors.extend(self.collect_exits());alive=[]
            for job in self.jobs:
                try:
                    group=owned_group(job)
                    if group is not None:
                        alive.append(group)
                        if group not in sent:os.killpg(group,signal.SIGTERM);sent.add(group)
                except (OSError,ValueError,KeyError) as error:errors.append(str(error))
            if not alive or time.monotonic()>=end:break
            time.sleep(1)
        for job in self.jobs:
            try:
                group=owned_group(job)
                if group is not None:os.killpg(group,signal.SIGKILL)
            except (OSError,ValueError,KeyError) as error:errors.append(str(error))
        self.collect_exits();t.write(t.OUT/'cleanup.json',dict(time=t.now(),groups=sorted(sent),errors=errors));self.refresh()

    def report(self,status,error=None):
        formal={mode:{} for mode in ('FULL','probe')};partial={}
        for mode in formal:
            for arm in t.ARMS:
                try:formal[mode][arm]=t.aggregate(arm,True,mode)
                except (OSError,ValueError,KeyError):partial[f'{mode}/{arm}']='未完成／不可比较'
        frozen=t.read(t.OUT/'selection/final_frozen.json') if (t.OUT/'selection/final_frozen.json').exists() else {}
        selection=t.read(t.OUT/'selection/validation_selection.json') if frozen else {}
        g=frozen.get('G');final=frozen.get('FINAL');pairs=t.comparisons(formal,g);budget=self.refresh()
        cpu=t.read(t.OUT/'final_review/cpu_exit.json') if (t.OUT/'final_review/cpu_exit.json').exists() else None
        result=dict(plan=t.PLAN,revision=t.REVISION,status=status,independent_review='PENDING',FINAL=final,G=g,
            validation=selection.get('validation'),formal=formal,partial=partial,comparisons=pairs,budget=budget,cpu=cpu,error=error,model=t.m.CONFIG)
        lines=['# UA3D spatial r4 实际交付','',f'状态：{status}；独立审查PENDING，未归档或替换部署。',f'FINAL={final}；G={g}。',
            '科学问题：同一有限Trimer的非键空间状态消息及距离／夹角矩能否改善FULL预测与冻结PT-3D性质可读性。',
            'BASE为本轮共同step0匹配基线；C_CONST保留真实空间邻接，不是完全无几何控制；四组非完整factorial，不宣称交互效应。',
            '所有正式TEST在全部280估计器统一冻结后一次执行。共享开发TEST非独立盲测。','']
        sota={};costs={}
        for mode in formal:
            lines+=['| 模式 | Arm | 正式macro R² |','|---|---|---:|']
            for arm in t.ARMS:
                value=formal[mode].get(arm);lines.append(f'| {mode} | {arm} | '+(f'{value["macro"]:.3f}' if value else '未完成／不可比较')+' |')
                lines+=['',f'### {mode}/{arm} 七任务','','| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |','|---|---:|---:|---:|---|']
                sota[f'{mode}/{arm}']={}
                for task,(name,model,ref) in t.SOTA.items():
                    v=value['tasks'][task] if value else None
                    sota[f'{mode}/{arm}'][task]=dict(project=v,reference=ref,reference_model=model,delta=v['mean']-ref if v else None)
                    lines.append(f'| {name} | '+(f'{v["mean"]:.3f}±{v["std"]:.3f} | {ref:.3f} | {v["mean"]-ref:+.3f} | {model} |' if v else f'未完成／不可比较 | {ref:.3f} | — | {model} |'))
        for mode,comps in pairs.items():
            lines+=['',f'### {mode} 配对差值','','| 对照 | Δmacro R² | 达到0.010 |','|---|---:|---|']
            for name,v in comps.items():lines.append(f'| {name} | '+(f'{v["delta"]:+.3f} | {v["performance_pass"]}' if v['delta'] is not None else '未完成／不可比较 | 未判定')+' |')
        for arm in t.ARMS:
            path=t.OUT/'pretrain'/arm/'consumption.json';costs[arm]=dict(pt=t.read(path) if path.exists() else None,units=[])
            for mode in ('FULL','probe'):
                for task in t.TASKS:
                    for fold in range(5):
                        for formal_flag in (False,True):
                            folder=t.unit_path(arm,task,fold,formal_flag,mode)
                            if (folder/'consumption.json').exists():costs[arm]['units'].append(dict(mode=mode,task=task,fold=fold,formal=formal_flag,consumption=t.read(folder/'consumption.json'),parameters=t.read(folder/'parameters.json') if (folder/'parameters.json').exists() else None))
        result.update(sota=sota,costs=costs)
        full_pass=pairs['FULL'].get('G_minus_BASE',{}).get('performance_pass');probe_pass=pairs['probe'].get('G_minus_BASE',{}).get('performance_pass')
        conclusion='未完成／不可比较'
        if full_pass is True and probe_pass is True:conclusion='本协议下3D表征与完整预测均改善'
        elif probe_pass is True and full_pass is False:conclusion='表征诊断改善、整体性能未通过'
        elif full_pass is True and probe_pass is False:conclusion='完整预测收益、独立表征改善未证实'
        elif full_pass is False and probe_pass is False:conclusion='FULL和冻结表征均未达到0.010门槛'
        result['conclusion']=conclusion
        lines+=['',conclusion,'','实际预算：'+__import__('json').dumps(budget['known_actual_lower_bound'],ensure_ascii=False),
            '原始精度、配对fold差值、退化任务／fold、参数与真实耗时见results.json及各单元history/consumption。未完成不得以validation或旧结果补齐。',
            '五折mean±std(ddof=1)，seed42+fold，七任务等权macro。论文是逐任务最佳包络；Egc3380/Xc432与论文4124/430等协议差异未消除，仅数值对照。EAT/Eib/Tg不纳入。',
            '本轮同时改变PT和joint FT，未独立拆分两阶段作用；冻结探针仅衡量固定head的可读性。空间角度是两个矩，不是完整三元组描述。',
            'PT模板没有RU offset，接缝统计记未提供；下游sidecar记录原RU offset跨接缝边。无虚构周期盒、长链、MD或新增构象。',
            f'异常／缺项：{error or ("无" if not partial else "实验未完成，见partial")}。',f'CPU：{cpu!r}。',
            'tmux窗口、命令、cwd、设备UUID、进程身份、日志及真实exit见budget.jobs；controller为'+self.manifest['tmux_window']+'。']
        t.write(t.OUT/'final_review/results.json',result);report='\n'.join(lines)+'\n';(t.OUT/'final_review/report.md').write_text(report)
        t.update_plan(status,'交付报告',report)

    def run(self):
        try:
            t.update_plan('执行中','prepare','一次物化5875行空间sidecar，保留旧2D/FP/3D缓存与初始化。');self.wait([self.launch('prepare')]);t.check_prepared()
            self.stage('pt',[(a,None,None,'FULL') for a in t.ARMS])
            matrix=[(a,task,f,'FULL') for a in t.ARMS for task in t.TASKS for f in range(5)]
            self.stage('ft',matrix)
            probes=[(a,task,f,'probe') for a in t.ARMS for task in t.TASKS for f in range(5)]
            self.stage('probe',probes);frozen=t.freeze()
            t.update_plan('执行中','一次冻结',f'280估计器冻结；FINAL={frozen["FINAL"]}，G={frozen["G"]}，开始280正式TEST。')
            self.stage('test',matrix);self.stage('test',probes);self.report('执行中')
            job=self.launch('cpu');self.wait([job])
            t.write(t.OUT/'final_review/cpu_exit.json',dict(exit=job['exit'],attempts=1,command=job['argv'],env=job['env'],log=job['log'],time=t.now()))
            self.report('待审查');self.capture_diff();self.manifest.update(status='待审查',ended=t.now());t.write(t.OUT/'execution_manifest.json',self.manifest);return 0
        except BaseException:
            error=traceback.format_exc()
            try:self.stop_owned()
            except BaseException:error+='\ncleanup: '+traceback.format_exc()
            status='阻断' if '48h wallclock exhausted' in error else '需返修'
            t.write(t.OUT/'failure.json',dict(status=status,error=error,time=t.now(),automatic_retry=False,automatic_resume=False))
            cpu=[j for j in self.jobs if j['kind']=='cpu']
            t.write(t.OUT/'final_review/cpu_exit.json',dict(exit=cpu[0]['exit'] if cpu else None,attempts=len(cpu),not_run_reason=None if cpu else 'authorized experiments not complete; no early CPU suite'))
            self.manifest.update(status=status,error=error,ended=t.now());t.write(t.OUT/'execution_manifest.json',self.manifest)
            self.report(status,error);self.capture_diff();self.refresh();return 1


def datetime_iso(timestamp):
    from datetime import datetime,timezone
    return datetime.fromtimestamp(timestamp,timezone.utc).isoformat()


def wrap(spec):
    job=t.read(spec);folder=Path(job['log']);env=os.environ.copy();env.update(job['env']);identity=None;requested=[];forwarded=False
    def stop(sig,frame):
        nonlocal forwarded
        requested.append(sig)
        if not forwarded and identity is not None and t.process_alive(identity) and os.getpgid(identity['pid'])==identity['pid']:
            forwarded=True;os.killpg(identity['pid'],signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (folder/'stdout.log').open('x') as out,(folder/'stderr.log').open('x') as err:
        child=subprocess.Popen(job['argv'],cwd=job['cwd'],env=env,stdout=out,stderr=err,start_new_session=True);identity=t.process_identity(child.pid)
        t.write(folder/'process.json',dict(wrapper=t.process_identity(os.getpid()),timeout_identity=identity,command=job['argv'],cwd=job['cwd'],started=t.now()))
        if requested:stop(requested[0],None)
        rc=child.wait()
    atomic_exit(folder/'exit.txt',rc);t.write(folder/'end.json',dict(exit=rc,ended=t.now(),job=job['name']));return rc


def install_termination():
    def stop(sig,frame):t.ignore_termination();raise RuntimeError('controlled termination '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('controller','prepare','ft','probe','test','wrap'))
    p.add_argument('--arm',choices=t.ARMS);p.add_argument('--task',choices=t.TASKS);p.add_argument('--fold',type=int,choices=range(5));p.add_argument('--mode',choices=('FULL','probe'),default='FULL')
    p.add_argument('--spec');p.add_argument('--physical-gpu',type=int,choices=t.ALLOWED_GPUS,default=3);args=p.parse_args();t.torch.set_num_threads(2)
    if args.action=='wrap':return wrap(args.spec)
    install_termination()
    if args.action=='controller':
        controller=Controller();rc=controller.run()
        t.write(t.OUT/'controller_exit.json',dict(exit=rc,time=t.now(),process=t.process_identity(os.getpid())))
        return rc
    if args.action=='prepare':t.prepare();return 0
    if args.arm is None or args.task is None or args.fold is None:p.error('fixed estimator identity required')
    if args.action=='probe' or args.mode=='probe':t.probe_unit(args.arm,args.task,args.fold,args.action=='test',args.physical_gpu)
    else:t.unit(args.arm,args.task,args.fold,args.action=='test',args.physical_gpu)
    return 0

if __name__=='__main__':raise SystemExit(main())
