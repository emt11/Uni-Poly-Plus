"""Fresh ALIGN20k, then fixed-parent 70 GATE fine-tunes; all work in tmux."""
import argparse
import datetime
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
from src.training.ua2d_pt20k import (OUT, LOG, PI, BASE, CONFIG, INITIAL,
                                    make_config, require_complete, validate_config, GPUS)
from src.training.glt_dual_runtime import require_tmux, save_checkpoint
from src.training.ua2d_fp_gate import ENV, PYTHON, write, read
from scripts.run_ua2d_fp_gate import observe_gpus, process_row


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def plan_status(status, detail):
    path = ROOT / 'CURRENT_PLAN.md'
    content = path.read_text()
    start = content.index('状态：')
    end = content.index('\n\n', start)
    content = content[:start] + f'状态：**{status}（P2＋G2）**。用户已授权新预训练20,000 step、每5,000 step保存，使用GPU2/3，PT最多2路、FT最多4路总并发；实际依据见第26节与P2日志。' + content[end:]
    content += '\n\n### P2 实际运行记录\n\n' + detail + '\n'
    path.write_text(content)


def execute(continue_prepared=False):
    global LOG
    require_tmux()
    evidence = OUT
    if continue_prepared:
        previous = read(LOG / 'execution.json')
        if (previous.get('pt_starts') != 0 or previous.get('ft_launched') is not False
                or previous.get('status') != '阻断' or not INITIAL.is_file()
                or (OUT / 'pretrain').exists() or (OUT / 'pretrain_launch_intent.json').exists()):
            raise ValueError('prepared continuation is only allowed before any PT launch')
        validate_config(read(CONFIG))
        attempt = 1
        while (LOG / f'continuation{attempt}').exists():
            attempt += 1
        LOG = LOG / f'continuation{attempt}'
        evidence = OUT / f'controller_attempts/continuation{attempt}'
        evidence.mkdir(parents=True, exist_ok=False)
        LOG.mkdir(parents=True, exist_ok=False)
    else:
        if OUT.exists() or LOG.exists():
            raise FileExistsError('P2 requires new roots; use --continue-prepared only after a zero-start device block')
        OUT.mkdir(parents=True)
        LOG.mkdir(parents=True)
    argv = [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
            'scripts/pretrain_glt_dual.py', '--config', str(CONFIG), '--cohort-root', str(PI),
            '--cache-root', str(PI), '--output', str(OUT / 'pretrain/P_ALIGN'),
            '--prep-workers', '8', '--lmdb-readahead']
    environment = dict(ENV, CUDA_VISIBLE_DEVICES='2,3', NCCL_ASYNC_ERROR_HANDLING='1',
                       TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    write(LOG / 'command.json', dict(argv=argv, cwd=str(ROOT), environment=environment,
          window='Uni-Poly:ua2d_p2_align20k_r7' + ('_continue' if continue_prepared else ''), artifacts=str(OUT)))
    execution = dict(start=now(), end=None, exit=None, status='INITIALIZING',
                     pid=os.getpid(), torchrun_pid=None, pt_starts=0, ft_launched=False)
    write(LOG / 'execution.json', execution)
    proc = None
    rc = 1
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    with (LOG / 'stdout.log').open('x') as output, (LOG / 'stderr.log').open('x') as errors:
        try:
            # Device availability and occupancy are runtime dependencies, not a GPU probe.
            # Keep raw query output, including a failed/malformed device response.
            config = validate_config(make_config())
            if not continue_prepared:
                (OUT / 'initialization').mkdir()
                write(CONFIG, config)
            import torch
            original = torch.load(BASE / 'initialization/P_ALIGN_step0.pt', map_location='cpu', weights_only=False)
            if (original.get('step') != 0 or original.get('arm') != 'P_ALIGN'
                    or original.get('config') != read(BASE / 'initialization/P_ALIGN.json')):
                raise ValueError('original U1 step0/config mismatch')
            # Tensor state is preserved exactly; only the run binding changes.
            original['config'] = config
            if continue_prepared:
                prepared = torch.load(INITIAL, map_location='cpu', weights_only=False)
                if (prepared.get('config') != config or prepared.get('step') != 0
                        or set(prepared['model']) != set(original['model'])
                        or any(not torch.equal(v, prepared['model'][k]) for k, v in original['model'].items())):
                    raise ValueError('prepared step0 tensor/config changed')
                del prepared
            else:
                save_checkpoint(INITIAL, original)
            del original
            for name in ('scripts/run_ua2d_pt20k.py', 'src/training/ua2d_pt20k.py',
                         'scripts/pretrain_glt_dual.py', 'scripts/evaluate_glt_pt_endpoint.py',
                         'scripts/run_ua2d_fp_gate.py', 'src/training/ua2d_fp_gate.py',
                         'scripts/run_ua2d_fp_gate_unit.py', 'tests/test_ua2d_fp_gate.py', 'CURRENT_PLAN.md'):
                target = evidence / 'source_used' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
                snapshot_name = {'src/training/ua2d_fp_gate.py': 'training.py',
                    'scripts/run_ua2d_fp_gate.py': 'controller.py',
                    'scripts/run_ua2d_fp_gate_unit.py': 'unit.py',
                    'tests/test_ua2d_fp_gate.py': 'test.py'}.get(name, Path(name).name)
                prior = Path('/tmp/ua2d_r7_before') / snapshot_name
                before = evidence / 'source_before' / name
                before.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(prior, before)
            initial = observe_gpus()
            write(evidence / 'device_before.json', initial)
            if any(initial['nvidia_pids'][str(g)] for g in GPUS):
                raise RuntimeError('P2 needs GPU2/3 idle before its two-rank launch')
            # Exclusive intent prevents a second controller from spending this PT budget.
            write(OUT / 'pretrain_launch_intent.json', dict(time=now(), pid=os.getpid(), log=str(LOG)))
            proc = subprocess.Popen(argv, cwd=ROOT, env=dict(os.environ, **environment),
                                    stdout=output, stderr=errors, start_new_session=True)
            execution.update(torchrun_pid=proc.pid, pt_starts=1, status='RUNNING',
                             worker_identity=process_row(proc.pid))
            write(LOG / 'execution.json', execution, True)
            plan_status('执行中', f'新P2已启动：{LOG}/command.json；预训练上限20,000步，checkpoint每5,000步；完成后自动启动G2完整70FT。')
            while proc.poll() is None:
                if stop[0]:
                    raise InterruptedError('P2 controller stopped; do not start fine-tuning')
                observation = observe_gpus()
                # This phase dispatches no FT. NVML is the single PID view used for
                # actual GPU contexts; never union it with namespace-local /proc PIDs.
                counts = {g: len(set(observation['nvidia_pids'][str(g)])) for g in GPUS}
                write(evidence / 'concurrency.json', dict(raw=observation, counts=counts,
                      method='NVML context PIDs only; no concurrent FT during P2'), True)
                unique_workers = set().union(*(set(observation['nvidia_pids'][str(g)]) for g in GPUS))
                if len(unique_workers) > 2:
                    raise RuntimeError('P2 GPU process concurrency exceeded (maximum 2 ranks)')
                with (evidence / 'runtime_observations.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(time=now(), counts=counts)) + '\n')
                time.sleep(15)
            (LOG / 'pretrain_exit.txt').write_text(str(proc.returncode) + '\n')
            if proc.returncode != 0:
                raise RuntimeError('P2 torchrun failed, exit=' + str(proc.returncode))
            complete = require_complete()
            write(OUT / 'completion.json', dict(status='已完成', pretrain=complete,
                                               final_cpu='DEFERRED_UNTIL_G2', independent_review='NOT_PERFORMED'))
            # Do not start FT while old CUDA contexts are still tearing down.
            deadline = time.monotonic() + 120
            while True:
                observation = observe_gpus()
                write(evidence / 'device_after.json', observation, True)
                if not any(observation['nvidia_pids'][str(g)] for g in GPUS):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError('GPU2/3 still occupied after successful P2 exit; FT not launched')
                time.sleep(5)
            subprocess.run([PYTHON, 'scripts/run_ua2d_fp_gate.py', '--launch'], cwd=ROOT,
                           env=dict(os.environ, **ENV, CUDA_VISIBLE_DEVICES=''),
                           stdout=output, stderr=errors, check=True)
            execution.update(status='PRETRAIN_COMPLETE_G2_LAUNCHED', ft_launched=True)
            plan_status('执行中', 'P2已完成20,000步及四组checkpoint；已启动G2完整70FT，实际进度看g2_fp_gate_pt20k_r7的消费账本。')
            rc = 0
        except BaseException as error:
            traceback.print_exc(file=errors)
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            if proc is not None and not (LOG / 'pretrain_exit.txt').exists():
                (LOG / 'pretrain_exit.txt').write_text(str(proc.returncode) + '\n')
            execution.update(status='阻断' if execution['pt_starts'] == 0 else '需返修', error=repr(error))
            write(evidence / 'failure.json', dict(time=now(), error=repr(error), traceback=traceback.format_exc(),
                  execution=execution, consumption=(read(OUT / 'pretrain/P_ALIGN/consumption.json')
                  if (OUT / 'pretrain/P_ALIGN/consumption.json').exists() else dict(starts=execution['pt_starts'],
                      attempted_updates=None if execution['pt_starts'] else 0,
                      completed_updates=None if execution['pt_starts'] else 0)),
                  ft_launched=execution['ft_launched'], tests='NOT_RUN'))
            plan_status(execution['status'], f'P2未能完成：{error!r}。PT启动{execution["pt_starts"]}次；未启动G2、未运行最终CPU测试。证据：{evidence}/failure.json，日志{LOG}/。')
        finally:
            execution.update(end=now(), exit=rc)
            write(LOG / 'execution.json', execution, True)
            (LOG / 'exit.txt').write_text(str(rc) + '\n')
    return rc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--continue-prepared', action='store_true')
    args = parser.parse_args()
    if args.launch:
        if (OUT.exists() or LOG.exists()) and not args.continue_prepared:
            raise FileExistsError('P2 roots already exist')
        subprocess.run(['tmux', 'has-session', '-t', 'Uni-Poly'], check=True)
        argv = ['env', *[k + '=' + v for k, v in dict(ENV, CUDA_VISIBLE_DEVICES='').items()],
                PYTHON, str(Path(__file__).resolve()), '--execute']
        if args.continue_prepared:
            argv.append('--continue-prepared')
        command = shlex.join(argv)
        window = 'ua2d_p2_align20k_r7_continue' if args.continue_prepared else 'ua2d_p2_align20k_r7'
        console = LOG.parent / (LOG.name + '_console.log')
        subprocess.run(['tmux', 'new-window', '-d', '-t', 'Uni-Poly', '-n', window,
                        'tmux set-option -w -t "$TMUX_PANE" remain-on-exit on\ncd ' + shlex.quote(str(ROOT)) + '\nexec ' + command +
                        ' >>' + shlex.quote(str(console)) + ' 2>&1'], check=True)
        print('P2 chain launched in Uni-Poly:ua2d_p2_align20k_r7; logs:', LOG)
    elif args.execute:
        sys.exit(execute(args.continue_prepared))
    else:
        parser.error('--launch or --execute required')
