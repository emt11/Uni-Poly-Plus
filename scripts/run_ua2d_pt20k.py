"""r8: one restricted P2 resume, then automatically dispatch a fresh G2 matrix."""
import argparse
import datetime
import difflib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.ua2d_pt20k import (OUT, LOG, PI, CONFIG, ORIGINAL, ORIGINAL_LOG,
    TRAIN_OUTPUT, RESUME, GPUS, validate_config, require_complete, checkpoint_index, classify_pt)
from src.training.glt_dual_runtime import require_tmux
from src.training.ua2d_fp_gate import ENV, PYTHON, write, read, EDITED
from scripts.run_ua2d_fp_gate import observe_gpus, process_row


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def plan_status(status, detail):
    path = ROOT / 'CURRENT_PLAN.md'
    content = path.read_text()
    start = content.index('状态：')
    end = content.index('\n\n', start)
    content = content[:start] + f'状态：**{status}（r8 P2恢复＋G2）**。用户明确授权一次15k→20k恢复（累计PT最多2次）、新G2完整70FT及唯一最终CPU；旧r7科学配置身份保留。' + content[end:]
    content += '\n\n### r8 P2 实际执行记录\n\n' + detail + '\n'
    path.write_text(content)


def snapshot_sources():
    names = {'CURRENT_PLAN.md': 'CURRENT_PLAN.md', 'scripts/run_ua2d_pt20k.py': 'run_ua2d_pt20k.py',
        'src/training/ua2d_pt20k.py': 'ua2d_pt20k.py', 'scripts/pretrain_glt_dual.py': 'pretrain_glt_dual.py',
        'scripts/run_ua2d_fp_gate.py': 'controller.py', 'src/training/ua2d_fp_gate.py': 'training.py',
        'scripts/run_ua2d_fp_gate_unit.py': 'unit.py', 'tests/test_ua2d_fp_gate.py': 'test.py'}
    changes = []
    for name, previous in names.items():
        before = OUT / 'source_before' / name
        used = OUT / 'source_used' / name
        before.parent.mkdir(parents=True, exist_ok=True)
        used.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path('/tmp/ua2d_r8_before') / previous, before)
        shutil.copyfile(ROOT / name, used)
        changes.extend(difflib.unified_diff(before.read_text().splitlines(True), used.read_text().splitlines(True),
            fromfile='before/' + name, tofile='after/' + name))
    (OUT / 'final_review/implementation.diff').write_text(''.join(changes))


def write_observation(detail):
    write(OUT / 'concurrency.json', detail, True)
    with (OUT / 'runtime_observations.jsonl').open('a') as stream:
        stream.write(json.dumps(detail, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def execute():
    require_tmux()
    if OUT.exists() or LOG.exists():
        raise FileExistsError('r8 recovery roots already exist; no second recovery launch')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    (OUT / 'final_review').mkdir()
    argv = [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
        'scripts/pretrain_glt_dual.py', '--config', str(CONFIG), '--cohort-root', str(PI),
        '--cache-root', str(PI), '--output', str(TRAIN_OUTPUT), '--prep-workers', '8',
        '--lmdb-readahead', '--resume', str(RESUME), '--ua2d-recovery']
    environment = dict(ENV, CUDA_VISIBLE_DEVICES='2,3', TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    write(LOG / 'command.json', dict(argv=argv, cwd=str(ROOT), environment=environment,
        window='Uni-Poly:ua2d_p2_r8_recovery1', artifacts=str(OUT)))
    execution = dict(start=now(), end=None, exit=None, status='INITIALIZING', pid=os.getpid(),
        torchrun_pid=None, new_pt_starts=0, cumulative_pt_starts=1, ft_launched=False, execution_revision='r8')
    write(LOG / 'execution.json', execution)
    proc, rc, bound = None, 1, []
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    with (LOG / 'stdout.log').open('x') as output, (LOG / 'stderr.log').open('x') as errors:
        try:
            config = validate_config(read(CONFIG))
            old = read(ORIGINAL / 'pretrain/P_ALIGN/consumption.json')
            old_exit = read(ORIGINAL_LOG / 'execution.json')
            if old['attempted_updates'] != 18348 or old['completed_updates'] != 18347 or old_exit['exit'] != 1:
                raise ValueError('old r7 failed consumption/true exit changed')
            if not RESUME.is_file():
                raise FileNotFoundError(RESUME)
            snapshot_sources()
            chain = dict(execution_revision='r8', scientific_config=str(CONFIG), config=config,
                resume=str(RESUME), segment_start=15000, segment_end=20000, maximum_new_updates=5000,
                old_failure=str(ORIGINAL / 'failure.json'), old_known=old,
                old_inflight='unknown: whether old18348 updated before bookkeeping', checkpoints=checkpoint_index(),
                old_unassigned_pid='historical362769 ownership unverified; no ignore rule', final_cpu='DEFERRED_UNTIL_G2')
            write(OUT / 'final_review/recovery_chain.json', chain)
            write(OUT / 'final_review/checkpoint_index.json', checkpoint_index())
            # This is the actual resource wait in the dispatch loop, not a probe.
            while True:
                initial = observe_gpus()
                contexts = set().union(*(set(initial['nvidia_pids'][str(g)]) for g in GPUS))
                reservations = set().union(*(set(initial['proc_pids'][str(g)]) for g in GPUS))
                write(OUT / 'device_before.json', initial, True)
                if not contexts and not reservations:
                    break
                execution['status'] = 'WAITING_FOR_GPU'
                write(LOG / 'execution.json', execution, True)
                if stop[0]:
                    raise InterruptedError('controller stopped before its sole recovery launch')
                time.sleep(15)
            write(OUT / 'pretrain_launch_intent.json', dict(time=now(), pid=os.getpid(),
                old_starts=1, new_starts=1, cumulative_limit=2, resume=str(RESUME)))
            proc = subprocess.Popen(argv, cwd=ROOT, env=dict(os.environ, **environment),
                stdout=output, stderr=errors, start_new_session=True)
            launcher_birth = process_row(proc.pid)
            execution.update(torchrun_pid=proc.pid, worker_identity=launcher_birth,
                new_pt_starts=1, cumulative_pt_starts=2, status='RUNNING')
            write(LOG / 'execution.json', execution, True)
            plan_status('执行中', f'唯一恢复已启动；{RESUME}→逻辑20000，仅5000更新。窗口Uni-Poly:ua2d_p2_r8_recovery1；日志{LOG}/。')
            while proc.poll() is None:
                if stop[0]:
                    raise InterruptedError('controller stopped; G2 will not launch')
                raw = observe_gpus()
                current_launcher = process_row(proc.pid)
                if current_launcher is None:
                    if proc.poll() is not None:
                        break
                    raise RuntimeError('live torchrun birth identity missing')
                if current_launcher['start_ticks'] != launcher_birth['start_ticks']:
                    raise RuntimeError('torchrun PID reused')
                ranks = {rank: read(TRAIN_OUTPUT / f'rank_lifecycle_{rank}.json') for rank in range(2)
                         if (TRAIN_OUTPUT / f'rank_lifecycle_{rank}.json').exists()}
                runtime_path = TRAIN_OUTPUT / 'runtime.json'
                finished = runtime_path.exists() and read(runtime_path).get('status') == 'PASS'
                detail = classify_pt(raw, ranks, proc.pid, bound, allow_exited=finished)
                detail.update(time=now(), baseline=str(OUT / 'device_before.json'))
                write_observation(detail)  # preserve exactly the raw failure before raising
                if detail['errors']:
                    raise RuntimeError('current P2 rank identity ambiguity: ' + repr(detail['errors']))
                if detail['unique_gpu_processes'] > 2:
                    raise RuntimeError('P2 unique GPU training process concurrency exceeded (maximum2)')
                if detail['registered_nvml'] and not bound:
                    bound = detail['registered_nvml']
                    write(OUT / 'rank_cuda_binding.json', detail)
                if bound and (detail['unassigned_nvml'] or detail['external_proc']):
                    raise RuntimeError('P2 current unassigned CUDA process; preserve evidence and stop')
                if len(detail['ready_ranks']) == 2 and len(set().union(*(set(raw['nvidia_pids'][str(g)]) for g in GPUS))) == 2 and not bound:
                    raise RuntimeError('P2 restored CUDA birth group cannot be attributed')
                time.sleep(15)
            proc.wait()
            (LOG / 'pretrain_exit.txt').write_text(str(proc.returncode) + '\n')
            if proc.returncode != 0:
                raise RuntimeError('P2 recovery torchrun failed, exit=' + str(proc.returncode))
            complete = require_complete()
            write(OUT / 'completion.json', dict(status='已完成', pretrain=complete, execution_revision='r8',
                final_cpu='DEFERRED_UNTIL_G2', independent_review='NOT_PERFORMED'))
            write(OUT / 'final_review/consumption.json', complete['consumption'])
            write(OUT / 'final_review/true_exit.json', dict(torchrun_exit=proc.returncode,
                log=str(LOG / 'pretrain_exit.txt'), ranks=2))
            # Wait for context teardown and external resource release; never kill
            # unrelated jobs or treat PT completion as a lease over their GPUs.
            while True:
                raw = observe_gpus()
                write(OUT / 'device_after.json', raw, True)
                if not any(raw['nvidia_pids'][str(g)] or raw['proc_pids'][str(g)] for g in GPUS):
                    break
                if stop[0]:
                    raise InterruptedError('stopped after PT; no G2 launch')
                time.sleep(5)
            subprocess.run([PYTHON, 'scripts/run_ua2d_fp_gate.py', '--launch'], cwd=ROOT,
                env=dict(os.environ, **ENV, CUDA_VISIBLE_DEVICES=''), stdout=output, stderr=errors, check=True)
            execution.update(status='PRETRAIN_COMPLETE_G2_LAUNCHED', ft_launched=True)
            plan_status('执行中', 'P2恢复完成5000新更新，逻辑20000、累计known completed23347；PT已退出，自动接续g2_fp_gate_pt20k_r8完整70FT。最终CPU尚未运行。')
            (OUT / 'final_review/delivery.md').write_text('P2 r8恢复完成，5000新更新，逻辑20000；四组checkpoint路径索引关联旧5k/10k/15k与新20k。G2已自动启动，正式七任务与最终CPU尚未完成；旧18348在途unknown保留。独立审查未进行。\n')
            rc = 0
        except BaseException as error:
            trace = traceback.format_exc()
            traceback.print_exc(file=errors)
            write(OUT / 'state_before_stop.json', dict(execution=execution, time=now(),
                last_observation=read(OUT / 'concurrency.json') if (OUT / 'concurrency.json').exists() else None))
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            if proc is not None and not (LOG / 'pretrain_exit.txt').exists():
                (LOG / 'pretrain_exit.txt').write_text(str(proc.returncode) + '\n')
            execution.update(status='阻断' if execution['new_pt_starts'] == 0 else '需返修', error=repr(error))
            consumption = read(TRAIN_OUTPUT / 'consumption.json') if (TRAIN_OUTPUT / 'consumption.json').exists() else dict(new_starts=execution['new_pt_starts'], attempted=None, completed=None)
            write(OUT / 'failure.json', dict(time=now(), error=repr(error), traceback=trace,
                execution=execution, consumption=consumption, ft_launched=execution['ft_launched'], tests='NOT_RUN'))
            write(OUT / 'final_review/consumption.json', consumption)
            (OUT / 'final_review/delivery.md').write_text(f'r8停止：{error!r}。无自动重试；正式七任务未完成／不可比较，最终CPU未执行；旧现场只读。见failure.json、concurrency.json及真实退出。\n')
            plan_status(execution['status'], f'r8在本次恢复启动{execution["new_pt_starts"]}次后停止：{error!r}；{OUT}/failure.json；不自动第三次PT启动。')
        finally:
            execution.update(end=now(), exit=rc)
            write(LOG / 'execution.json', execution, True)
            (LOG / 'exit.txt').write_text(str(rc) + '\n')
    return rc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.launch:
        if OUT.exists() or LOG.exists():
            raise FileExistsError('sole recovery roots already exist')
        session = subprocess.run(['tmux', 'has-session', '-t', 'Uni-Poly'], capture_output=True)
        if session.returncode:
            subprocess.run(['tmux', 'new-session', '-d', '-s', 'Uni-Poly'], check=True)
        argv = ['env', *[k+'='+v for k,v in dict(ENV, CUDA_VISIBLE_DEVICES='').items()],
                PYTHON, str(Path(__file__).resolve()), '--execute']
        console = LOG.parent / (LOG.name + '_console.log')
        console.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['tmux','new-window','-d','-t','Uni-Poly','-n','ua2d_p2_r8_recovery1',
            'tmux set-option -w -t "$TMUX_PANE" remain-on-exit on\ncd '+shlex.quote(str(ROOT))+
            '\nexec '+shlex.join(argv)+' >>'+shlex.quote(str(console))+' 2>&1'], check=True)
        print('r8 sole P2 recovery chain launched: Uni-Poly:ua2d_p2_r8_recovery1; logs:', LOG)
    elif args.execute:
        sys.exit(execute())
    else:
        parser.error('--launch or --execute required')
