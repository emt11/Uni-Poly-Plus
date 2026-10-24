#!/usr/bin/env python3
"""Explicit UA-HOP-PH-FROZEN3D r1 entrypoint; one controller, no automatic retries."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse
import fcntl
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.ua_hop_ph_frozen3d import *


class Controller:
    def __init__(self):
        require_tmux()
        OUT.mkdir(parents=True, exist_ok=False)
        LOG.mkdir(parents=True, exist_ok=False)
        folder = LOG/'controller'; folder.mkdir()
        sys.stdout = (folder/'stdout.log').open('a', buffering=1)
        sys.stderr = (folder/'stderr.log').open('a', buffering=1)
        self.lock = (OUT/'controller.lock').open('x')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.jobs = []; self.failed = None
        self.window = subprocess.check_output(['tmux', 'display-message', '-p', '-t',
            os.environ['TMUX_PANE'], '#{window_id}'], text=True).strip()
        write(OUT/'execution_manifest.json', dict(plan=PLAN, revision='r1', started=now(), role='DSH',
            authorization='User explicitly authorized full execution and asked this assistant to act as DSH',
            cwd=str(ROOT), command=[sys.executable, str(Path(__file__).resolve()), 'controller'],
            commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            dirty_status=subprocess.check_output(['git', 'status', '--short'], text=True),
            tmux_session='Uni-Poly', tmux_window=self.window, log=str(folder), only_gpus=[2, 3],
            concurrency_per_gpu=2, parent=str(PARENT), architecture=ARCHITECTURE, limits=LIMITS,
            automatic_retry=False, automatic_resume=False, pid=os.getpid()))
        for arm in ARMS: write(OUT/'configs'/f'{arm}.json', dict(arm=arm, architecture=ARCHITECTURE, profile=PROFILE))
        required = 70*sum(math.ceil(len(read(SPLIT/f'{t}.json')['folds'][f]['train_indices'])/24)
                            for t in TASKS for f in range(5))
        self.planned_updates = required
        self.refresh()
        update_plan('执行中', 'FT初始化', f'本对话按最新用户指令担任DSH；controller {self.window}，PID {os.getpid()}。'
                    f'固定矩阵预计更新{required}，上限56000；所有训练／预测尚未派发。')

    def refresh(self):
        used = dict.fromkeys(LIMITS, 0)
        for job in self.jobs:
            if job['kind'] == 'ft': used['ft_starts'] += 1
            elif job['kind'] == 'test': used['test_starts'] += 1
            elif job['kind'] == 'cpu': used['cpu_suites'] += 1
            cpath = job.get('consumption')
            if cpath and Path(cpath).exists():
                c = read(cpath)
                for key in ('epochs_started', 'validation_started', 'attempted_updates', 'test_graphs'):
                    used[key] += c.get(key, 0)
                job['progress'] = c
            exit_path = Path(job['log'])/'exit.txt'
            if job['exit'] is None and exit_path.exists():
                job['exit'] = int(exit_path.read_text().strip()); job['ended'] = now()
        if any(used[k] > LIMITS[k] for k in LIMITS): raise RuntimeError('global budget exceeded')
        write(OUT/'budget.json', dict(plan=PLAN, revision='r1', limits=LIMITS, used=used,
            planned_updates=self.planned_updates, jobs=self.jobs, time=now(),
            accounting='all starts, started epochs/validations and optimizer attempts, including failures'))
        return used

    def launch(self, name, kind, args, gpu=None, consumption=None):
        if any(j['name'] == name for j in self.jobs): raise RuntimeError('no job may be dispatched twice')
        used = self.refresh()
        if kind == 'test':
            decision = read(OUT/'selection/development_decision.json')
            if not decision['passed'] or not (OUT/'selection/final_frozen.json').is_file():
                raise RuntimeError('TEST dispatch is forbidden before validation gate/freeze')
        limit = {'ft': 'ft_starts', 'test': 'test_starts', 'cpu': 'cpu_suites'}.get(kind)
        if limit and used[limit] >= LIMITS[limit]: raise RuntimeError('dispatch budget exhausted')
        if gpu is not None and (gpu not in (2, 3) or sum(j['gpu'] == gpu and j['exit'] is None for j in self.jobs) >= 2):
            raise RuntimeError('GPU concurrency exceeded')
        log = LOG/name; log.mkdir(exist_ok=False)
        if kind == 'cpu':
            argv = ['timeout', '--signal=TERM', '--kill-after=5s', '300s', sys.executable,
                    '-m', 'pytest', '-q', str(ROOT/'tests/test_ua_hop_ph_frozen3d.py')]
        else:
            argv = [sys.executable, '-u', str(Path(__file__).resolve()), *args]
        env = dict(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else '')
        command = shlex.join(['env', *[f'{k}={v}' for k, v in env.items()], *map(str, argv)])
        wrapper = (f'cd {shlex.quote(str(ROOT))}\n{command} >{shlex.quote(str(log/"stdout.log"))} '
                   f'2>{shlex.quote(str(log/"stderr.log"))}\n'
                   f'ua_hop_exit=$?\nprintf "%s\\n" "$ua_hop_exit" >{shlex.quote(str(log/"exit.txt"))}\nexit "$ua_hop_exit"')
        job = dict(name=name, kind=kind, gpu=gpu, args=list(map(str, args)), command=command,
                   cwd=str(ROOT), log=str(log), started=now(), exit=None, consumption=str(consumption) if consumption else None)
        self.jobs.append(job); self.refresh()  # reserve start before process creation
        write(log/'command.json', job)
        job['tmux_window'] = subprocess.check_output(['tmux', 'new-window', '-d', '-P', '-F', '#{window_id}',
            '-t', 'Uni-Poly:', '-n', name, 'bash', '-c', wrapper], text=True).strip()
        self.refresh(); return job

    def wait(self, jobs):
        while True:
            self.refresh()
            bad = [j for j in jobs if j['exit'] not in (None, 0)]
            if bad: raise RuntimeError('job failed without retry: '+bad[0]['name'])
            if all(j['exit'] == 0 for j in jobs): return
            time.sleep(5)

    def matrix(self, formal=False):
        active = []; remaining = [(a, t, f) for a in ARMS for t in TASKS for f in range(5)]
        while remaining or active:
            self.refresh()
            if any(j['exit'] not in (None, 0) for j in active):
                raise RuntimeError('matrix worker failed; stop dispatch')
            active = [j for j in active if j['exit'] is None]
            for gpu in (2, 3):
                while remaining and sum(j['gpu'] == gpu for j in active) < 2:
                    arm, task, fold = remaining.pop(0)
                    kind = 'test' if formal else 'ft'
                    name = f'f3d_{kind}_{task}_{fold}'
                    path = unit_path(arm, task, fold, formal)
                    job = self.launch(name, kind, [kind, '--arm', arm, '--task', task, '--fold', str(fold)],
                                      gpu=gpu, consumption=path/'consumption.json')
                    active.append(job)
            if active: time.sleep(5)

    def stop_owned(self):
        self.refresh()
        for job in self.jobs:
            if job['exit'] is None and job.get('progress', {}).get('pid'):
                pid = job['progress']['pid']
                try:
                    cmdline = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
                    if 'run_ua_hop_ph_frozen3d.py' not in cmdline: raise RuntimeError('worker PID identity mismatch')
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic()+30
        while any(j['exit'] is None for j in self.jobs) and time.monotonic() < deadline:
            time.sleep(2); self.refresh()
        if any(j['exit'] is None for j in self.jobs):
            raise RuntimeError('worker still active; preserve evidence, no CPU overlap')

    def run(self):
        error = None
        try:
            if self.planned_updates > LIMITS['attempted_updates']:
                raise RuntimeError(f'fixed matrix requires {self.planned_updates} updates, exceeding56000')
            initialize_sources()
            update_plan('执行中', 'FT', f'controller {self.window}，仅F3D_PH；只读sidecar已绑定，无输入准备作业。真实3D前向保留，encoder3d冻结。')
            self.matrix()
            decision = decide_and_freeze()
            if decision['passed']:
                update_plan('执行中', '正式TEST', f"validation增量{decision['delta']:.9f}通过0.010门槛，35份best已一次冻结；各正式TEST一次。")
                self.matrix(formal=True)
            else:
                update_plan('执行中', 'validation负结果收尾', f"validation增量{decision['delta']:.9f}未达0.010，正式TEST保持0；不追加训练。")
        except BaseException:
            error = traceback.format_exc()
            write(OUT/'failure.json', dict(time=now(), error=error, automatic_retry=False))
            self.stop_owned()
        finally:
            update_plan('执行中' if error is None else '需返修', '最终CPU',
                        '授权训练及可能的TEST已结束，仅执行一次冻结政策针对性CPU suite；不重复r1 suite。'+(f'停止原因：{error.splitlines()[-1]}' if error else ''))
            cpu = self.launch('cpu', 'cpu', [])
            try:
                self.wait([cpu])
            except BaseException:
                error = (error or '')+'\n'+traceback.format_exc()
            self.refresh()
            status = '待审查' if error is None else '需返修'
            final_report(status, error)
            paths = ['src/modules/ua_hop_ph_frozen3d.py', 'src/training/ua_hop_ph_frozen3d.py',
                     'scripts/run_ua_hop_ph_frozen3d.py', 'tests/test_ua_hop_ph_frozen3d.py']
            patches = []
            for p in paths:
                r = subprocess.run(['git', 'diff', '--no-index', '/dev/null', p], text=True, capture_output=True)
                patches.append(r.stdout)
            (OUT/'final_review/code_diff.patch').write_text('\n'.join(patches))
            report = (OUT/'final_review/report.md').read_text()
            update_plan(status, '执行收尾', f'controller {self.window}，唯一CPU exit={cpu["exit"]}。\n\n'+report)
            (OUT/'final_review/plan_diff.patch').write_text(subprocess.check_output(['git','diff','--','CURRENT_PLAN.md'],text=True))
            write(LOG/'controller/exit.json', dict(exit=0 if error is None else 1, time=now(), error=error))
        return 0 if error is None else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['controller', 'ft', 'test'])
    parser.add_argument('--arm', choices=ARMS); parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--fold', type=int, choices=range(5))
    args = parser.parse_args()
    torch.set_num_threads(2)
    def terminate(signum, frame): raise RuntimeError(f'controlled termination signal {signum}; no automatic retry')
    signal.signal(signal.SIGTERM, terminate)
    if args.action == 'controller': return Controller().run()
    unit(args.arm, args.task, args.fold, formal=args.action == 'test')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
