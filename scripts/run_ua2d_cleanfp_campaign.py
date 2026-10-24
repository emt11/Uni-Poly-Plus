#!/usr/bin/env python3
"""Authorized, isolated six-arm controller. Never resumes a historical queue."""
import argparse
import fcntl
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.training.ua2d_readout_distill import *


class Controller:
    def __init__(self):
        OUT.mkdir(parents=True,exist_ok=False)
        self.lock=(OUT/'controller.lock').open('x')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.jobs=[];self.gpu_uuids={}
        validate_cache(CACHE)
        previous=read(OLD/'budget.json')
        freeze(OUT/'history_reference.json',dict(path=str(OLD),status=read(OLD/'status.json'),
            budget_path=str(OLD/'budget.json'),used=previous['used'],reused_only='clean static templates and random step0'))
        info=subprocess.check_output(['nvidia-smi','-i','2,3','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
        for line in info.strip().splitlines():
            index,uuid=[x.strip() for x in line.split(',')];self.gpu_uuids[uuid]=index
        if set(self.gpu_uuids.values())!={'2','3'}:raise ValueError('GPU identity mismatch')
        self.manifest_path=OUT/'execution_manifest.json'
        freeze(self.manifest_path,dict(plan=PLAN,revision=REVISION,time=now(),authorization='请你自己执行！',
            scope='PRETRAIN.MD A–M six-arm full cycle',arms=list(ARMS),limits=LIMITS,only_gpus=[2,3],
            branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
            commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            dirty_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
            cwd=str(ROOT),gpu_mapping=self.gpu_uuids,cache=str(CACHE),source=str(SOURCE),index=str(INDEX),
            downstream=str(DOWN),split=str(SPLIT),head_root=str(HEADS),automatic_retry=False,
            deadline_per_pt=18000,new_fp_generated=0,code_schema=SCHEMA,outer_test='NOT_RUN'))
        self.refresh()

    def pt(self,arm):
        self.wait_free([2,3])
        if read(OUT/'configs'/f'{arm}.json')!=objective_config(arm):raise ValueError('arm config changed')
        counts=self.refresh()
        if counts['pt_updates']+5000>LIMITS['pt_updates'] or counts['pt_presentations']+5040000>LIMITS['pt_presentations']:
            raise RuntimeError('not enough budget for complete PT')
        output=parent_path(arm).parent
        if output.exists():raise FileExistsError(output)
        argv=['timeout','--signal=TERM','--kill-after=30s','18000s',sys.executable,'-m','torch.distributed.run',
            '--standalone','--nproc_per_node=2',str(ROOT/'scripts/pretrain_ua2d_cleanfp.py'),
            '--config',OUT/'configs'/f'{arm}.json','--output',output,
            '--initialization',OUT/'initialization/common_step0.pt','--prepared-cache',CACHE]
        self.wait(self.launch(f'cleanfp_pt_{arm}_r1','pt',argv,output,'2,3'))
        for rank in range(2):
            rec=read(output/f'rank{rank}_consumption.json')
            if rec['status']!='COMPLETE' or rec['completed_updates']!=5000 or rec['sample_presentations']!=2520000:
                raise RuntimeError('incomplete PT arm')

    def block(self,arm,formal=False):
        pending=[(t,f) for t in TASKS for f in range(5)];active={};failed=[]
        self.wait_free([2,3])
        while pending or active:
            for slot,job in list(active.items()):
                if self.poll(job):
                    if job['exit']!=0:failed.append(job)
                    del active[slot]
            if not failed:
                for gpu in ('2','3'):
                    if any(k[0]==gpu for k in active) or self.occupied().get(gpu):continue
                    for slot in (0,1):
                        if not pending:break
                        task,fold=pending.pop(0);mode='formal' if formal else 'ft'
                        argv=[sys.executable,str(ROOT/'scripts/run_ua2d_cleanfp_campaign.py'),mode,
                              '--arm',arm,'--task',task,'--fold',str(fold)]
                        active[(gpu,slot)]=self.launch(f'cleanfp_{mode}_{arm}_{task}_{fold}_r1',mode,argv,
                            unit_path(arm,task,fold,formal),gpu)
            self.refresh()
            if failed and not active:raise RuntimeError('failed FT/formal; stopped further dispatch: '+repr(failed))
            if active or pending:time.sleep(15)
        aggregate(arm,formal)

    def refresh(self):
        counts={k:0 for k in LIMITS}
        for job in self.jobs:
            kind=job['kind'];output=Path(job['output'])
            if kind=='pt':
                counts['pt_starts']+=1;records=[]
                for r in range(2):
                    path=output/f'rank{r}_consumption.json'
                    records.append(read(path) if path.exists() else {})
                counts['pt_updates']+=max((r.get('attempted_updates',0) for r in records),default=0)
                counts['pt_presentations']+=sum(r.get('sample_presentations',0) for r in records)
            elif kind in ('ft','formal'):
                c=read(output/'consumption.json') if (output/'consumption.json').exists() else {}
                if kind=='ft':
                    counts['ft_starts']+=1;counts['ft_updates']+=c.get('attempted_updates',0)
                    counts['ft_epochs']+=c.get('epochs_started',0);counts['validation_units']+=c.get('validation_started',0)
                else:counts['test_starts']+=1;counts['test_graphs']+=c.get('test_graphs',0)
            elif kind=='cpu':counts['cpu_suites']+=1

        check_limits(counts)
        write(OUT/'budget.json',dict(plan=PLAN,revision=REVISION,time=now(),used=counts,limits=LIMITS,
            authorization='user: 请你自己执行！; PRETRAIN.MD A–M',
            accounting='starts reserved before dispatch; attempted updates and begun epochs/validation include partials; PT updates count synchronized optimizer windows, samples sum both ranks',jobs=self.jobs))
        return counts

    def occupied(self):
        info=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'],text=True)
        occupied={}
        for line in info.strip().splitlines():
            parts=[s.strip() for s in line.split(',')]
            if parts and parts[0] in self.gpu_uuids:occupied.setdefault(self.gpu_uuids[parts[0]],[]).append(parts)
        return occupied

    def wait_free(self,gpus):
        while True:
            busy=self.occupied()
            if not any(str(g) in busy for g in gpus):return
            write(OUT/'status.json',dict(status='执行中',phase='WAIT_GPU',gpus=gpus,occupied=busy,time=now()))
            self.refresh();time.sleep(30)

    def launch(self,name,kind,argv,output,gpu=''):
        counts=self.refresh()
        key={'pt':'pt_starts','ft':'ft_starts','formal':'test_starts','cpu':'cpu_suites'}.get(kind)
        if key and counts[key]>=LIMITS[key]:raise RuntimeError('start budget exhausted: '+key)
        if kind=='pt' and gpu!='2,3':raise ValueError('PT GPU violation')
        if kind in ('ft','formal') and gpu not in ('2','3'):raise ValueError('FT/formal GPU violation')
        log=LOG/name;log.mkdir(parents=True,exist_ok=False)
        shell='#!/bin/bash\ncd '+shlex.quote(str(ROOT))+' || exit 90\n'
        shell+='export CUDA_VISIBLE_DEVICES='+shlex.quote(gpu)+' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1\n'
        shell+='export CUBLAS_WORKSPACE_CONFIG=:4096:8\n'
        shell+='export UA2D_CLEANFP_CAMPAIGN=r1\n'
        command=shlex.join([str(a) for a in argv])
        (log/'command.txt').write_text(command+'\n');(log/'cwd.txt').write_text(str(ROOT)+'\n')
        shell+='date -Iseconds > '+shlex.quote(str(log/'start.txt'))+'\n'
        shell+=command+' > '+shlex.quote(str(log/'stdout.log'))+' 2> '+shlex.quote(str(log/'stderr.log'))+'\n'
        shell+='ua2d_exit=$?\nprintf \'%s\\n\' "$ua2d_exit" > '+shlex.quote(str(log/'exit.txt'))+'\n'
        shell+='date -Iseconds > '+shlex.quote(str(log/'end.txt'))+'\nexit "$ua2d_exit"\n'
        (log/'run.sh').write_text(shell)
        job=dict(name=name,kind=kind,argv=[str(a) for a in argv],cwd=str(ROOT),output=str(output),log=str(log),gpu=gpu,start=now(),exit=None)
        self.jobs.append(job);self.refresh()
        try:
            ident=subprocess.check_output(['tmux','new-window','-d','-P','-F','#{window_id}','-t','Uni-Poly','-n',name,'bash '+shlex.quote(str(log/'run.sh'))],text=True).strip()
            job['tmux_window']=ident;job['tmux_name']=name;self.refresh()
        except BaseException:
            job['launch_error']=traceback.format_exc();self.refresh();raise
        return job

    def poll(self,job):
        exit_path=Path(job['log'])/'exit.txt'
        if exit_path.exists():
            job['exit']=int(exit_path.read_text());job['end']=now();return True
        # A vanished window with no exit record is a failure, never auto retry.
        names=subprocess.check_output(['tmux','list-windows','-t','Uni-Poly','-F','#{window_id}'],text=True).splitlines()
        if job.get('tmux_window') not in names:raise RuntimeError('window vanished without exit: '+job['name'])
        return False

    def wait(self,job):
        while not self.poll(job):self.refresh();time.sleep(15)
        self.refresh()
        if job['exit']!=0:raise RuntimeError(job['name']+' failed with exit '+str(job['exit'])+'; see '+job['log'])


def register_current(status,evidence):
    """Append only if old controllers are stopped and no observed writer changed the file."""
    process=subprocess.check_output(['ps','-eo','state,args'],text=True)
    for line in process.splitlines():
        if any(x in line for x in ('python scripts/ua2d_sota_dispatch.py','python scripts/ua2d_sota_continue.py','python scripts/run_ua2d_sota_campaign.py')) and line.strip()[0]!='T':
            write(OUT/'current_plan_registration.json',dict(status='DEFERRED',reason='old writer may be active',time=now()));return
    path=ROOT/'CURRENT_PLAN.md';before=path.stat()
    with (OUT/'current_plan.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        text=path.read_text();after=path.stat()
        if (before.st_mtime_ns,before.st_size)!=(after.st_mtime_ns,after.st_size):
            write(OUT/'current_plan_registration.json',dict(status='DEFERRED',reason='concurrent edit observed',time=now()));return
        note=f'\n\n## {PLAN} / {REVISION} 执行登记（{now()}）\n\n- 状态：**{status}**；授权来源：用户明确要求“请你自己执行！”，PRETRAIN.MD六臂完整有限周期，仅GPU2/3。唯一科学执行对象：[PRETRAIN.MD](PRETRAIN.MD)。旧周期保持停止，不恢复。\n- 证据：`{evidence}`；结果根 `{OUT.relative_to(ROOT)}`；日志根 `{LOG.relative_to(ROOT)}`。预算与实际退出码见独立 budget.json 和各 tmux 日志。\n- 不覆盖旧周期、不读取 TODO、不执行远程同步／哈希检查。执行自检不构成独立审查；未完成正式结果不可与 SOTA 比较。\n'
        if (OUT/'final_review/sota_comparison.md').exists():
            note+='\n'+(OUT/'final_review/sota_comparison.md').read_text().replace('# 七任务 SOTA 数值对照','### 本计划七任务 SOTA 数值对照',1)
        with path.open('a') as f:f.write(note)
        write(OUT/'current_plan_registration.json',dict(status='APPENDED',time=now(),recorded_status=status))


def final_suite(c):
    if c.refresh()['cpu_suites']:return
    argv=['timeout','600s',sys.executable,'-m','pytest','-q',str(ROOT/'tests/test_ua2d_cleanfp_readout_distill.py')]
    cpu=c.launch('cleanfp_final_cpu_r1','cpu',argv,OUT/'final_review')
    try:c.wait(cpu)
    finally:
        write(OUT/'final_review/cpu_result.json',dict(exit=cpu.get('exit'),log=cpu['log'],
            time=now(),only_suite=True,passed=cpu.get('exit')==0))


def controller():
    c=None
    try:
        c=Controller();register_current('执行中',str(c.manifest_path))
        updates=0
        for task in TASKS:
            manifest=read(SPLIT/f'{task}.json')
            if manifest.get('validation_is_test') is not False:raise ValueError('invalid split')
            updates+=sum(70*math.ceil(len(f['train_indices'])/24) for f in manifest['folds'])
        if updates*6!=336000:raise ValueError('fixed FT budget changed')
        c.wait(c.launch('cleanfp_initialize_r1','initialize',
            [sys.executable,str(ROOT/'scripts/run_ua2d_cleanfp_campaign.py'),'initialize'],OUT/'initialization'))
        for arm in ARMS:
            write(OUT/'status.json',dict(status='执行中',phase='PT',arm=arm,time=now()))
            c.pt(arm)
            write(OUT/'status.json',dict(status='执行中',phase='FT',arm=arm,time=now()))
            c.block(arm)
        freeze_final()
        for arm in ARMS:
            write(OUT/'status.json',dict(status='执行中',phase='FORMAL',arm=arm,time=now()))
            c.block(arm,True)
        final_suite(c);c.refresh();reports('待审查')
        write(OUT/'status.json',dict(status='待审查',time=now(),independent_review=False))
        register_current('待审查',str(OUT/'final_review/results.json'))
    except BaseException as error:
        if c:
            try:final_suite(c)
            except BaseException:write(OUT/'final_review/cpu_failure.json',dict(error=traceback.format_exc()))
            c.refresh();reports('阻断',repr(error))
            write(OUT/'status.json',dict(status='阻断',error=repr(error),traceback=traceback.format_exc(),time=now(),automatic_retry=False))
            register_current('阻断',str(OUT/'status.json'))
        raise


def main():
    from src.training.glt_dual_runtime import require_tmux
    require_tmux()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('controller','initialize','ft','formal'))
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--task',choices=TASKS);parser.add_argument('--fold',type=int)
    args=parser.parse_args()
    if args.mode=='controller':controller()
    elif args.mode=='initialize':initialize()
    else:finetune(args.arm,args.task,args.fold,args.mode=='formal')


if __name__=='__main__':main()
