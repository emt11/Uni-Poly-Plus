#!/usr/bin/env python3
"""Explicit fresh, fixed-budget restart after the user's r3 cancellation."""
import os
os.environ['UA2D_PRETRAIN_FP_CAMPAIGN']='fast_r4'
import copy
import fcntl
import math
from pathlib import Path
import subprocess
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.training.ua2d_pretrain_fp import *
from scripts.run_ua2d_pretrain_fp import Controller,register_current
OLD=ROOT/'results/ua2d_pretrain_fp_20261019/r1'


class FastController(Controller):
    def __init__(self):
        self.recover=False;self.gpu_uuids={};self.recovery_root=OUT/'authorization'
        self.old_lock=(OLD/'controller.lock').open('r+')
        fcntl.flock(self.old_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=read(OLD/'budget.json')
        if (read(OLD/'status.json').get('status')!='已取消' or previous.get('revision')!='r3'
                or previous['used']!=dict.fromkeys(LIMITS,0)|dict(pt_starts=2,pt_updates=1392,
                    pt_presentations=1402296,cpu_suites=2)
                or any(j.get('exit') is None for j in previous['jobs'])):
            raise ValueError('old execution not fully stopped and accounted')
        OUT.mkdir(parents=True,exist_ok=False)
        self.lock=(OUT/'controller.lock').open('x')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.jobs=copy.deepcopy(previous['jobs'])
        self.inherited_fp=previous['used']['fp_generated']
        freeze(OUT/'inherited_budget.json',previous)
        freeze(OUT/'inherited_status.json',read(OLD/'status.json'))
        freeze(self.recovery_root/'user_instruction.json',dict(plan=PLAN,revision=REVISION,time=now(),
            user_request='停止旧任务，工程加速后重新预训练；每版约5小时，保留5000updates，仅GPU2/3',
            limits=LIMITS,inherited=previous['used'],fresh_pt_arms=3,automatic_retry=False,
            scientific_changes=False,deadline_per_pt_seconds=18000,
            cache_preparation='one shared CPU stage separately timed; deterministic clean FP labels prepared early but never read by V1/V2',
            cpu_suite='one new focused equivalence/ordering/context suite; prior science tests reused'))
        # Reuse exactly the original common and task-specific step-zero weights.
        for name in ('common_step0.pt','task_heads/V1.pt','task_heads/V2.pt'):
            if not (OLD/'initialization'/name).is_file():raise FileNotFoundError(name)
        (OUT/'initialization').symlink_to(OLD/'initialization',target_is_directory=True)
        info=subprocess.check_output(['nvidia-smi','-i','2,3','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
        for line in info.strip().splitlines():
            index,uuid=[s.strip() for s in line.split(',')];self.gpu_uuids[uuid]=index
        if set(self.gpu_uuids.values())!={'2','3'}:raise ValueError('GPU mapping changed')
        self.manifest_path=OUT/'execution_manifest.json'
        freeze(self.manifest_path,dict(plan=PLAN,revision=REVISION,time=now(),cwd=str(ROOT),limits=LIMITS,
            branch=subprocess.check_output(['git','branch','--show-current'],text=True,cwd=ROOT).strip(),
            commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
            dirty_status=subprocess.check_output(['git','status','--short'],text=True,cwd=ROOT),
            only_gpus=[2,3],gpu_mapping=self.gpu_uuids,source=str(SOURCE),index=str(INDEX),
            split=str(SPLIT),downstream=str(DOWN),head_root=str(HEADS),plan_file=str(ROOT/'PRETRAIN.MD'),
            inherited=str(OUT/'inherited_budget.json'),automatic_retry=False,outer_test='NOT_RUN'))
        self.refresh()

    def refresh(self):
        # Reuse accounting for every historical and new dispatched job.
        counts=super().refresh()
        path=OUT/'prepared_cache/metadata.json'
        if path.exists():
            counts['fp_generated']=self.inherited_fp+read(path)['generated']
        check_limits(counts)
        value=read(OUT/'budget.json');value.update(used=counts,
            authorization=str(self.recovery_root/'user_instruction.json'),
            inherited_budget=str(OUT/'inherited_budget.json'),
            accounting='all starts and optimizer windows cumulative; PT presentations include prepared prefetch, not just forwarded samples; cache preprocessing separately timed')
        write(OUT/'budget.json',value)
        return counts

    def pt(self,arm):
        self.wait_free([2,3])
        config=OUT/'configs'/f'{arm}.json';freeze(config,objective_config(arm))
        output=parent_path(arm).parent
        if output.exists():raise FileExistsError(output)
        counts=self.refresh()
        if counts['pt_updates']+5000>LIMITS['pt_updates'] or counts['pt_presentations']+5040000>LIMITS['pt_presentations']:
            raise RuntimeError('insufficient complete-arm budget')
        argv=['timeout','--signal=TERM','--kill-after=30s','18000s',sys.executable,
            '-m','torch.distributed.run','--standalone','--nproc_per_node=2',ROOT/'scripts/pretrain_ua2d_fast.py',
            '--config',config,'--output',output,'--initialization',OUT/'initialization/common_step0.pt',
            '--prepared-cache',OUT/'prepared_cache']
        write(OUT/'status.json',dict(status='执行中',phase='PT_'+arm,time=now(),
            deadline_seconds=18000,timing=str(output/'timing.json')))
        self.wait(self.launch('ua2d_fast_pt_'+arm,'pt',argv,output,'2,3'))
        for rank in range(2):
            value=read(output/f'rank{rank}_consumption.json')
            if value['status']!='COMPLETE' or value['completed_updates']!=5000 or value['sample_presentations']!=2520000:
                raise ValueError('PT incomplete; truncated run is not a completed arm')


def main():
    from src.training.glt_dual_runtime import require_tmux
    require_tmux()
    c=None
    try:
        c=FastController();register_current('执行中',str(c.manifest_path))
        updates=0
        for task in TASKS:
            value=read(SPLIT/f'{task}.json')
            if value.get('validation_is_test') is not False:raise ValueError('invalid development split')
            updates+=sum(70*math.ceil(len(f['train_indices'])/24) for f in value['folds'])
        if updates*3!=168000:raise ValueError('FT budget drift')
        write(OUT/'status.json',dict(status='执行中',phase='FINAL_ENGINEERING_CPU_CHECK',time=now()))
        cpu=c.launch('ua2d_fast_final_cpu','cpu',['timeout','300s',sys.executable,'-m','pytest','-q',
            ROOT/'tests/test_ua2d_fast_pretrain.py'],OUT/'engineering_review')
        c.wait(cpu)
        freeze(OUT/'engineering_review/test_gate.json',dict(passed=True,exit=cpu['exit'],log=cpu['log'],time=now(),
            previous_scientific_suite=str(OLD/'recovery/r3/test_gate.json'),reused_at_closeout=True))
        write(OUT/'status.json',dict(status='执行中',phase='SHARED_CPU_TEMPLATE_CACHE',time=now()))
        cache=c.launch('ua2d_fast_prepare_cache','cache',[sys.executable,ROOT/'scripts/prepare_ua2d_fast_cache.py',
            '--output',OUT/'prepared_cache','--workers','16'],OUT/'prepared_cache')
        c.wait(cache)
        for arm in ('V1','V2'):
            c.pt(arm)
            write(OUT/'status.json',dict(status='执行中',phase='FT_'+arm,time=now()))
            c.block(arm)
        c.refresh();freeze_baseline()
        c.pt('B_FP')
        write(OUT/'status.json',dict(status='执行中',phase='FT_B_FP',time=now()))
        c.block('B_FP');c.refresh();freeze_final()
        for arm in ARMS:
            write(OUT/'status.json',dict(status='执行中',phase='FORMAL_'+arm,time=now()))
            c.block(arm,True)
        c.refresh();reports('待审查')
        write(OUT/'status.json',dict(status='待审查',time=now(),independent_review=False))
        register_current('待审查',str(OUT/'final_review/results.json'))
    except BaseException as error:
        if c:
            c.refresh();reports('阻断',repr(error))
            write(OUT/'status.json',dict(status='阻断',error=repr(error),traceback=traceback.format_exc(),
                time=now(),automatic_retry=False))
            register_current('阻断',str(OUT/'status.json'))
        raise


if __name__=='__main__':main()
