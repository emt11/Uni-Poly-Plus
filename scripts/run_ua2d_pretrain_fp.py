#!/usr/bin/env python3
"""Single-writer finite campaign with explicit, one-time failed-r2 recovery."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training.ua2d_pretrain_fp import *


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['controller','recover-r2','initialize','ft','formal','cache'])
    p.add_argument('--arm',choices=ARMS);p.add_argument('--task',choices=TASKS);p.add_argument('--fold',type=int)
    args=p.parse_args()
    from src.training.glt_dual_runtime import require_tmux
    require_tmux()
    if args.mode=='initialize':initialize()
    elif args.mode=='cache':prepare_cache()
    elif args.mode in ('ft','formal'):finetune(args.arm,args.task,args.fold,args.mode=='formal')
    else:controller(recover=args.mode=='recover-r2')


class Controller:
    def __init__(self,recover=False):
        self.recover=recover
        # Recovery is an explicit user-authorized operation, never automatic.
        if not recover:OUT.mkdir(parents=True,exist_ok=False)
        self.lock=(OUT/'controller.lock').open('r+' if recover else 'x')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.jobs=[];self.gpu_uuids={}
        self.recovery_root=OUT/'recovery/r3'
        if recover:
            previous=read(OUT/'budget.json')
            self.jobs=inherited_recovery_jobs(previous)
            if read(OUT/'status.json').get('status')!='阻断':raise ValueError('recovery requires a blocked cycle')
            for relative in ('initialization/common_step0.pt','initialization/task_heads/V1.pt',
                             'initialization/task_heads/V2.pt','configs/V1.json'):
                if not (OUT/relative).is_file():raise ValueError('missing required existing artifact: '+relative)
            for rank in range(2):
                rec=read(OUT/'pretrain/V1'/f'rank{rank}_consumption.json')
                if rec!=dict(attempted_updates=1,completed_updates=0,sample_presentations=84,status='FAILED',rank=rank):
                    raise ValueError('failed V1 consumption changed since authorization')
            self.recovery_root.mkdir(parents=True,exist_ok=False)
            freeze(self.recovery_root/'budget_before.json',previous)
            freeze(self.recovery_root/'status_before.json',read(OUT/'status.json'))
            freeze(self.recovery_root/'execution_manifest_before.json',read(OUT/'execution_manifest.json'))
            snapshots=self.recovery_root/'r2_final_review';snapshots.mkdir()
            for source in (OUT/'final_review').iterdir():
                if source.is_file() and source.suffix in ('.json','.md'):
                    (snapshots/source.name).write_bytes(source.read_bytes())
            freeze(self.recovery_root/'authorization.json',dict(plan=PLAN,revision=REVISION,time=now(),
                user_instruction='请你给我修复并运行！',
                scope='repair all implementation issues, one additional CPU suite, then complete original three-arm cycle',
                previous_limits=R2_LIMITS,limits=LIMITS,inherited_used=previous['used'],
                scientific_completed_pt_updates=15000,only_gpus=[2,3],automatic_retry=False))
            freeze(OUT/'parent_bindings.json',dict(plan=PLAN,revision=REVISION,
                parents={a:str(OUT/'pretrain'/('V1_attempt2' if a=='V1' else a)/'deploy.pt') for a in ARMS},
                failed_attempt=str(OUT/'pretrain/V1'),common=str(OUT/'initialization/common_step0.pt')))
        info=subprocess.check_output(['nvidia-smi','-i','2,3','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
        for line in info.strip().splitlines():
            i,u=[s.strip() for s in line.split(',')];self.gpu_uuids[u]=i
        if set(self.gpu_uuids.values())!={'2','3'}:raise RuntimeError('GPU mapping mismatch')
        changes=subprocess.check_output(['git','status','--short','--untracked-files=normal'],cwd=ROOT,text=True)
        self.manifest_path=OUT/('execution_manifest_r3.json' if recover else 'execution_manifest.json')
        freeze(self.manifest_path,dict(plan=PLAN,revision=REVISION,time=now(),cwd=str(ROOT),
            authorization='user explicitly authorized repairs, one V1 replacement attempt and one additional CPU suite; only GPU2,3; no automatic retries' if recover else 'user explicit implementation and finite cycle; only GPU2,3; no retries',
            branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
            commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty_status=changes,
            gpu_mapping=self.gpu_uuids,limits=LIMITS,source=str(SOURCE),index=str(INDEX),downstream=str(DOWN),
            split=str(SPLIT),head_root=str(HEADS),plan_file=str(ROOT/'PRETRAIN.MD'),outer_test='NOT_RUN',
            environment=dict(python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda),
            inherited_jobs=len(self.jobs),previous_manifest=str(OUT/'execution_manifest.json') if recover else None,
            historical_jobs='old SOTA jobs not resumed; failed r2 evidence retained; no remote operations, file digests, TODO reads, smoke or benchmark'))
        self.refresh()

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
        if (OUT/'fp_cache/metadata.json').exists():counts['fp_generated']=read(OUT/'fp_cache/metadata.json')['generated']
        check_limits(counts)
        write(OUT/'budget.json',dict(plan=PLAN,revision=REVISION,time=now(),used=counts,limits=LIMITS,
            authorization=str(self.recovery_root/'authorization.json') if self.recover else None,
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
        if os.environ.get('UA2D_PRETRAIN_FP_CAMPAIGN') in ('fast_r4','attnscale_r1'):
            shell+='export UA2D_PRETRAIN_FP_CAMPAIGN='+os.environ['UA2D_PRETRAIN_FP_CAMPAIGN']+'\n'
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

    def pt(self,arm):
        self.wait_free([2,3]);cfg=objective_config(arm);path=OUT/'configs'/f'{arm}.json'
        if path.exists():
            if read(path)!=cfg:raise ValueError('existing objective config differs: '+str(path))
        else:freeze(path,cfg)
        output=parent_path(arm).parent
        if output.exists():raise FileExistsError('PT attempt output already exists: '+str(output))
        counts=self.refresh()
        if counts['pt_updates']+5000>LIMITS['pt_updates'] or counts['pt_presentations']+5040000>LIMITS['pt_presentations']:
            raise RuntimeError('insufficient budget for a complete PT arm')
        argv=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(ROOT/'scripts/pretrain_ua2d_galformer.py'),
            '--config',path,'--output',output,'--initialization',OUT/'initialization/common_step0.pt',
            '--source-root',SOURCE,'--index-artifact',INDEX,'--max-steps','5000','--microbatch','84','--accumulation','6']
        if arm=='B_FP':argv+=['--fp-cache',OUT/'fp_cache']
        name='pretrain_fp_pt_'+arm+('_attempt2' if self.recover and arm=='V1' else '')
        self.wait(self.launch(name,'pt',argv,output,'2,3'))
        for r in range(2):
            rec=read(output/f'rank{r}_consumption.json')
            if rec['status']!='COMPLETE' or rec['completed_updates']!=5000 or rec['sample_presentations']!=2520000:
                raise RuntimeError('incomplete PT consumption')

    def block(self,arm,formal=False):
        pending=[(t,f) for t in TASKS for f in range(5)];active={};failed=[]
        # A full block begins only after both assigned cards are free of external work.
        self.wait_free([2,3])
        while pending or active:
            for slot,job in list(active.items()):
                if self.poll(job):
                    if job['exit']!=0:failed.append(job)
                    del active[slot]
            if not failed:
                for gpu in ('2','3'):
                    # Dispatch at most two units as one wave on an idle GPU.
                    # This avoids guessing GPU ownership from host/container PIDs.
                    if any(k[0]==gpu for k in active) or self.occupied().get(gpu):continue
                    for slot in (0,1):
                        if not pending:break
                        t,f=pending.pop(0);mode='formal' if formal else 'ft'
                        name=f'pretrain_fp_{mode}_{arm}_{t}_{f}'
                        argv=[sys.executable,str(ROOT/'scripts/run_ua2d_pretrain_fp.py'),mode,'--arm',arm,'--task',t,'--fold',str(f)]
                        active[(gpu,slot)]=self.launch(name,mode,argv,unit_path(arm,t,f,formal),gpu)
            self.refresh()
            if failed and not active:raise RuntimeError('FT/formal failed; no further dispatch: '+repr(failed))
            if active or pending:time.sleep(15)
        aggregate(arm,formal)


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
        note=f'\n\n## {PLAN} / {REVISION} 执行登记（{now()}）\n\n- 状态：**{status}**；授权来源：用户当前消息，完整有限周期，仅GPU2/3。唯一科学执行对象：[PRETRAIN.MD](PRETRAIN.MD)。旧周期保持停止，不恢复。\n- 证据：`{evidence}`；结果根 `{OUT.relative_to(ROOT)}`；日志根 `{LOG.relative_to(ROOT)}`。预算与实际退出码见独立 budget.json 和各 tmux 日志。\n- 不覆盖旧周期、不读取 TODO、不执行远程同步／哈希检查。执行自检不构成独立审查；未完成正式结果不可与 SOTA 比较。\n'
        if (OUT/'final_review/sota_comparison.md').exists():
            note+='\n'+(OUT/'final_review/sota_comparison.md').read_text().replace('# 七任务 SOTA 数值对照','### 本计划七任务 SOTA 数值对照',1)
        with path.open('a') as f:f.write(note)
        write(OUT/'current_plan_registration.json',dict(status='APPENDED',time=now(),recorded_status=status))


def controller(recover=False):
    c=None
    try:
        c=Controller(recover=recover);register_current('执行中',str(c.manifest_path))
        # Derive the fixed full-cycle update allowance from existing split indices.
        updates=0
        for t in TASKS:
            m=read(SPLIT/f'{t}.json')
            if m.get('validation_is_test') is not False:raise ValueError('invalid development split')
            for f in m['folds']:
                updates+=70*math.ceil(len(f['train_indices'])/24)
        if updates*3!=168000:raise ValueError('fixed split FT update budget changed')
        if recover:
            # The only added CPU suite is the final repair check, before expensive PT.
            # Reuse its evidence at closeout; do not launch a third suite.
            argv=['timeout','300s',sys.executable,'-m','pytest','-q',str(ROOT/'tests/test_ua2d_pretrain_fp.py')]
            cpu=c.launch('pretrain_fp_repair_cpu_r3','cpu',argv,c.recovery_root)
            c.wait(cpu)
            freeze(c.recovery_root/'test_gate.json',dict(passed=True,exit=cpu['exit'],log=cpu['log'],time=now(),
                suite='tests/test_ua2d_pretrain_fp.py',no_additional_suite_at_closeout=True))
        else:
            c.wait(c.launch('pretrain_fp_initialization','initialize',[sys.executable,str(ROOT/'scripts/run_ua2d_pretrain_fp.py'),'initialize'],OUT/'initialization'))
        for arm in ('V1','V2'):
            write(OUT/'status.json',dict(status='执行中',phase=arm,time=now()))
            c.pt(arm);c.block(arm)
        c.refresh();freeze_baseline()
        c.wait(c.launch('pretrain_fp_cache','cache',[sys.executable,str(ROOT/'scripts/run_ua2d_pretrain_fp.py'),'cache'],OUT/'fp_cache'))
        c.pt('B_FP');c.block('B_FP');c.refresh();freeze_final()
        for arm in ARMS:c.block(arm,True)
        c.refresh()
        if not recover:
            argv=['timeout','300s',sys.executable,'-m','pytest','-q',str(ROOT/'tests/test_ua2d_pretrain_fp.py')]
            cpu=c.launch('pretrain_fp_final_cpu','cpu',argv,OUT/'final_review');c.wait(cpu)
        elif not read(c.recovery_root/'test_gate.json')['passed']:
            raise RuntimeError('repair test evidence missing')
        c.refresh();reports('待审查')
        write(OUT/'status.json',dict(status='待审查',time=now(),independent_review=False))
        register_current('待审查',str(OUT/'final_review/results.json'))
    except BaseException as error:
        if c:
            c.refresh();reports('阻断',repr(error))
            write(OUT/'status.json',dict(status='阻断',error=repr(error),traceback=traceback.format_exc(),time=now(),automatic_retry=False))
            register_current('阻断',str(OUT/'status.json'))
        raise

if __name__=='__main__':main()
