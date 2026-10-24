#!/usr/bin/env python3
"""One authorized ET_SINGLE controller; bounded preparation/PT/FT, no retries."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ.setdefault(name,'2')
import argparse
from dataclasses import asdict
import fcntl
import shlex
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training.ua2d_et_single import *

class Controller:
    def __init__(self):
        require_tmux();OUT.mkdir(parents=True,exist_ok=False);LOG.mkdir(parents=True,exist_ok=False)
        folder=LOG/'controller';folder.mkdir()
        sys.stdout=(folder/'stdout.log').open('a',buffering=1);sys.stderr=(folder/'stderr.log').open('a',buffering=1)
        self.lock=(OUT/'controller.lock').open('x');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.jobs=[];self.started=None
        self.window=subprocess.check_output(['tmux','display-message','-p','-t',os.environ['TMUX_PANE'],'#{window_id}'],text=True).strip()
        write(OUT/'execution_manifest.json',dict(plan=PLAN,revision=REVISION,authorization='用户明确指令：请你直接执行！',
            started=now(),role='Codex-executor',independent_review='PENDING',cwd=str(ROOT),
            command=[sys.executable,str(Path(__file__).resolve()),'controller'],tmux_session='Uni-Poly',tmux_window=self.window,
            commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            dirty_status=subprocess.check_output(['git','status','--short'],text=True),limits=LIMITS,
            time_limits=dict(preparation=7200,pt=18000,ft_unit=1800,ft_cumulative=63000,test_unit=300,total=86400,cpu=300),
            gpus=[2,3],ft_per_gpu=2,automatic_retry=False,automatic_resume=False,pid=os.getpid()))
        write(OUT/'configs/pt.json',asdict(PretrainConfig()))
        write(OUT/'configs/architecture.json',ARCHITECTURE)
        update_plan('执行中','初始化',f'执行授权已登记；controller {self.window}，PID {os.getpid()}。准备／PT／FT／TEST尚未派发。')
        self.refresh()

    def refresh(self,enforce=True):
        used=dict.fromkeys(LIMITS,0);ft_seconds=0.
        for j in self.jobs:
            kind=j['kind']
            key=dict(prepare='preparation_starts',pt='pt_starts',ft='ft_starts',test='test_starts',cpu='cpu_suites',filter='preparation_starts')[kind]
            used[key]+=1
            exit_path=Path(j['log'])/'exit.txt'
            if j['exit'] is None and exit_path.exists(): j['exit']=int(exit_path.read_text().strip());j['ended']=now()
            if kind in ('ft','test'):
                p=Path(j['consumption'])
                if p.exists():
                    c=read(p);j['progress']=c
                    for k in ('epochs_started','validation_started','attempted_updates','test_graphs'):used[k]+=c.get(k,0)
                if kind=='ft':
                    end=Path(j['log'])/'elapsed.txt'
                    ft_seconds+=float(end.read_text()) if end.exists() else time.monotonic()-j['monotonic_start']
            elif kind=='pt':
                rank_values=[]
                for rank in range(2):
                    p=Path(j.get('pt_output',str(OUT/'pretrain/ET_SINGLE')))/f'rank{rank}_consumption.json'
                    if p.exists():rank_values.append(read(p))
                j['rank_progress']=rank_values
                used['pt_updates']+=max([x['attempted_updates'] for x in rank_values],default=0)
                reported=sum(x['sample_presentations'] for x in rank_values)
                j['reported_presentations']=reported
                used['pt_presentations']+=max(reported,j.get('presentation_reservation',0))
            elif kind=='prepare':
                p=OUT/'atomic_fields/metadata.json'
                if p.exists():j['progress']=read(p)
        elapsed=0. if self.started is None else time.monotonic()-self.started
        write(OUT/'budget.json',dict(plan=PLAN,revision=REVISION,limits=LIMITS,used=used,jobs=self.jobs,
            elapsed_since_preparation_seconds=elapsed,ft_cumulative_seconds=ft_seconds,time=now(),
            accounting='all starts, failed/partial starts, attempted updates, prefetched presentations, wall time'))
        if enforce and (any(used[k]>LIMITS[k] for k in LIMITS) or ft_seconds>63000 or elapsed>86400):
            raise RuntimeError('global count/time budget exhausted; no recovery')
        return used

    def launch(self,name,kind,args=None,gpu=None,consumption=None):
        if any(j['name']==name for j in self.jobs): raise RuntimeError('no dispatch twice')
        used=self.refresh()
        if kind=='test':
            decision=read(OUT/'selection/development_decision.json')
            if not decision['passed'] or not (OUT/'selection/final_frozen.json').is_file():
                raise RuntimeError('TEST forbidden before validation gate/freeze')
        key=dict(prepare='preparation_starts',pt='pt_starts',ft='ft_starts',test='test_starts',cpu='cpu_suites',filter='preparation_starts')[kind]
        if used[key]>=LIMITS[key]: raise RuntimeError('start budget exhausted')
        if gpu is not None and (gpu not in (2,3) or sum(j['gpu']==gpu and j['exit'] is None for j in self.jobs)>=2):
            raise RuntimeError('GPU concurrency exceeded')
        log=LOG/name;log.mkdir(exist_ok=False)
        if kind=='prepare':
            if self.started is not None: raise RuntimeError('preparation cannot restart')
            self.started=time.monotonic()
            write(OUT/'wallclock_start.json',dict(time=now(),monotonic=self.started,deadline_seconds=86400))
            argv=[sys.executable,'-u',str(ROOT/'scripts/prepare_ua2d_et_single.py')];deadline=7200
        elif kind=='filter':
            argv=[sys.executable,'-u',str(ROOT/'scripts/filter_ua2d_et_single.py')];deadline=6771
        elif kind=='pt':
            argv=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                str(ROOT/'scripts/pretrain_ua2d_et_single.py'),'--config',str(OUT/'configs/pt.json'),
                '--output',str(PARENT.parent),'--initialization',str(COMMON),
                '--prepared-cache',str(CACHE),'--atomic-fields',str(OUT/'atomic_fields')];deadline=18000
        elif kind=='cpu': argv=[sys.executable,'-m','pytest','-q',str(ROOT/('tests/test_ua2d_et_single_repair1.py' if os.environ.get('UA2D_ET_SINGLE_REPAIR')=='repair1' else 'tests/test_ua2d_et_single.py'))];deadline=300
        else:
            argv=[sys.executable,'-u',str(Path(__file__).resolve()),*args];deadline=1800 if kind=='ft' else 300
        if kind!='cpu' and self.started is not None:
            deadline=min(deadline,max(0,int(86400-(time.monotonic()-self.started))))
            if deadline<=0: raise RuntimeError('total wallclock exhausted')
        env=dict(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            CUDA_VISIBLE_DEVICES='2,3' if kind=='pt' else str(gpu) if gpu is not None else '',UA2D_ET_SINGLE='r1',
            UA2D_ET_SINGLE_REPAIR=os.environ.get('UA2D_ET_SINGLE_REPAIR',''))
        command=shlex.join(['env',*[f'{k}={v}' for k,v in env.items()],
            'timeout','--signal=TERM','--kill-after=5s',str(deadline)+'s',*argv])
        # A distinct process group is recorded before waiting so every owned
        # rank/preparation worker can be terminated on failure without touching others.
        wrapper=(f'cd {shlex.quote(str(ROOT))}\net_et_started=$SECONDS\n'
            f'setsid {command} >{shlex.quote(str(log/"stdout.log"))} 2>{shlex.quote(str(log/"stderr.log"))} &\n'
            f'et_et_pid=$!\nprintf "%s\\n" "$et_et_pid" >{shlex.quote(str(log/"pid.txt"))}\n'
            f'wait "$et_et_pid"\net_et_exit=$?\nprintf "%s\\n" "$((SECONDS-et_et_started))" >{shlex.quote(str(log/"elapsed.txt"))}\n'
            f'printf "%s\\n" "$et_et_exit" >{shlex.quote(str(log/"exit.txt"))}\nexit "$et_et_exit"')
        job=dict(name=name,kind=kind,gpu=gpu,args=args or [],command=command,cwd=str(ROOT),log=str(log),
            started=now(),monotonic_start=time.monotonic(),exit=None,consumption=str(consumption) if consumption else None)
        if kind=='pt':job['pt_output']=str(PARENT.parent)
        self.jobs.append(job);self.refresh();write(log/'command.json',job)
        job['tmux_window']=subprocess.check_output(['tmux','new-window','-d','-P','-F','#{window_id}',
            '-t','Uni-Poly:','-n',name,'bash','-c',wrapper],text=True).strip()
        self.refresh();return job

    def wait(self,jobs):
        while True:
            self.refresh()
            for j in jobs:
                if j['kind']=='pt':
                    path=Path(j.get('pt_output',str(PARENT.parent)))
                    if any((path/f'rank{rank}_failure.json').exists() for rank in (0,1)):
                        raise RuntimeError('PT rank failure stops all ranks immediately; no retry')
            if any(j['exit'] not in (None,0) for j in jobs): raise RuntimeError('worker failed; no retry: '+str([(j['name'],j['exit']) for j in jobs]))
            if all(j['exit']==0 for j in jobs): return
            time.sleep(5)

    def matrix(self,formal=False):
        remaining=[(t,f) for t in TASKS for f in range(5)];active=[]
        while remaining or active:
            self.refresh()
            if any(j['exit'] not in (None,0) for j in active): raise RuntimeError('matrix failure stops dispatch')
            active=[j for j in active if j['exit'] is None]
            for gpu in (2,3):
                while remaining and sum(j['gpu']==gpu for j in active)<2:
                    task,fold=remaining.pop(0);kind='test' if formal else 'ft'
                    path=unit_path('ET_SINGLE',task,fold,formal)
                    active.append(self.launch(f'et_{kind}_{task}_{fold}',kind,
                        [kind,'--task',task,'--fold',str(fold)],gpu,path/'consumption.json'))
            if active: time.sleep(5)

    def stop_owned(self):
        self.refresh(enforce=False)
        for j in self.jobs:
            if j['exit'] is not None: continue
            p=Path(j['log'])/'pid.txt'
            if not p.exists(): raise RuntimeError('cannot confirm owned process group; preserve evidence')
            pid=int(p.read_text())
            try:
                cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
                if 'ua2d_et_single' not in cmd: raise RuntimeError('owned PID command mismatch')
                os.killpg(pid,signal.SIGTERM)
            except ProcessLookupError: pass
        deadline=time.monotonic()+20
        while any(j['exit'] is None for j in self.jobs) and time.monotonic()<deadline:
            time.sleep(2);self.refresh(enforce=False)
        for j in self.jobs:
            if j['exit'] is None:
                pid=int((Path(j['log'])/'pid.txt').read_text())
                try: os.killpg(pid,signal.SIGKILL)
                except ProcessLookupError: pass
        self.refresh(enforce=False)

    def run(self):
        error=None;cpu=None
        try:
            initialize_sources()
            expected=70*sum(math.ceil(len(read(SPLIT/f'{t}.json')['folds'][f]['train_indices'])/24) for t in TASKS for f in range(5))
            if expected!=56000: raise RuntimeError('fixed FT matrix budget mismatch')
            update_plan('执行中','原子字段准备',f'controller {self.window}；唯一准备作业≤7200秒，后续严格单候选。')
            job=self.launch('et_atomic_prepare','prepare');self.wait([job])
            update_plan('执行中','PT','原子字段准备完成；开始唯一双卡5000-update PT，硬期限18000秒。')
            job=self.launch('et_pretrain','pt');self.wait([job])
            update_plan('执行中','FT','PT完整完成；35FT固定矩阵，每卡最多两路。')
            self.matrix();decision=decide_and_freeze()
            if decision['passed']:
                update_plan('执行中','正式TEST',f"validation对V1增量{decision['delta']:+.3f}通过，35份best冻结。")
                self.matrix(formal=True)
            else: update_plan('执行中','负结果收尾',f"validation对V1增量{decision['delta']:+.3f}未达0.010，TEST=0。")
        except BaseException:
            error=traceback.format_exc();write(OUT/'failure.json',dict(time=now(),error=error,automatic_retry=False))
            self.stop_owned()
        # After affected workers are stopped, exactly one final CPU suite.
        try:
            update_plan('执行中' if error is None else '需返修','最终CPU','仅一次最终针对性suite≤300秒；无重测。')
            cpu=self.launch('et_cpu','cpu');self.wait([cpu])
        except BaseException:
            error=(error or '')+'\n'+traceback.format_exc()
        self.refresh(enforce=False)
        status='待审查' if error is None else '需返修'
        final_report(status,error)
        paths=['src/modules/ua2d_et_single.py','src/dataset/ua2d_et_single.py','src/training/ua2d_et_single.py',
            'scripts/prepare_ua2d_et_single.py','scripts/pretrain_ua2d_et_single.py','scripts/run_ua2d_et_single.py','tests/test_ua2d_et_single.py']
        patches=[subprocess.run(['git','diff','--no-index','/dev/null',p],text=True,capture_output=True).stdout for p in paths]
        (OUT/'final_review/code_diff.patch').write_text('\n'.join(patches))
        update_plan(status,'执行收尾',f'controller {self.window}；CPU exit={cpu["exit"] if cpu else "未启动"}。\n\n'+(OUT/'final_review/report.md').read_text())
        (OUT/'final_review/plan_diff.patch').write_text(subprocess.check_output(['git','diff','--','CURRENT_PLAN.md'],text=True))
        write(LOG/'controller/exit.json',dict(exit=0 if error is None else 1,time=now(),error=error))
        return 0 if error is None else 1

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['controller','ft','test'])
    parser.add_argument('--task',choices=TASKS);parser.add_argument('--fold',type=int,choices=range(5));args=parser.parse_args()
    torch.set_num_threads(2)
    def terminate(signum,frame): raise RuntimeError(f'controlled termination {signum}; no retry')
    signal.signal(signal.SIGTERM,terminate)
    if args.action=='controller': return Controller().run()
    unit('ET_SINGLE',args.task,args.fold,formal=args.action=='test');return 0

if __name__=='__main__': raise SystemExit(main())
