#!/usr/bin/env python3
"""N1: 70 raw-label FT units, complete freeze, 70 predictions, one final CPU test."""
import datetime
import os
import shlex
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_glt_ft_raw_targets_unit import OUT, LOG, REVIEW, ARMS, TASKS, read, write, ft_folder

GPUS = (2, 3)
ACTIVE = []
ELIGIBLE = ()


def launch(name, command, gpus=(), limit=None):
    if any(g not in GPUS for g in gpus) or len(gpus) > 1:
        raise ValueError('only one authorized physical GPU2/3 per unit')
    folder = LOG / name
    folder.mkdir(parents=True, exist_ok=False)
    cmd = shlex.join(command)
    if limit:
        cmd = f'timeout {limit}s ' + cmd
    visible = ','.join(map(str, gpus))
    (folder / 'command.txt').write_text('CUDA_VISIBLE_DEVICES=' + visible + ' ' + cmd + '\n')
    (folder / 'cwd.txt').write_text(str(ROOT) + '\n')
    write(folder / 'environment.json', dict(CUDA_VISIBLE_DEVICES=visible,
          logical_device='cuda:0' if gpus else 'CPU', OMP_NUM_THREADS=4,
          OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1))
    script = folder / 'command.sh'
    script.write_text(f'''#!/bin/bash
cd {shlex.quote(str(ROOT))} || exit 90
export CUDA_VISIBLE_DEVICES={shlex.quote(visible)}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Iseconds > {shlex.quote(str(folder / 'start.txt'))}
{cmd} > {shlex.quote(str(folder / 'stdout.log'))} 2> {shlex.quote(str(folder / 'stderr.log'))} &
process_pid=$!
printf '%s\\n' "$process_pid" > {shlex.quote(str(folder / 'pid.txt'))}
wait "$process_pid"
code=$?
printf '%s\\n' "$code" > {shlex.quote(str(folder / 'exit.txt'))}
date -Iseconds > {shlex.quote(str(folder / 'end.txt'))}
exit "$code"
''')
    subprocess.run(['tmux', 'new-window', '-d', '-t', 'Uni-Poly', '-n',
                    'glt_opt_r7_N1_' + name, 'bash ' + shlex.quote(str(script))], check=True)
    ACTIVE.append(folder)
    return folder / 'exit.txt'


def consumption(enforce=True):
    epochs = head = joint = attempted = completed = ft_done = formal = 0
    for arm in ARMS:
        for task in TASKS:
            for fold in range(5):
                folder = ft_folder(arm, task, fold)
                if (folder / 'progress.json').exists():
                    history = read(folder / 'progress.json')['history']
                    epochs += len(history)
                    head += sum(h['stage'] == 'head' for h in history)
                    joint += sum(h['stage'] == 'joint' for h in history)
                if (folder / 'updates.json').exists():
                    updates = read(folder / 'updates.json')
                    attempted += updates['attempted']
                    completed += updates['completed']
                ft_done += int((folder / 'metrics.json').exists())
                formal += int((OUT / 'formal' / arm / task / f'fold{fold}' / 'metrics.json').exists())
    result = dict(pt_starts=0, pt_completed_updates=0, geometry_derivations=0,
                  conformer_generation=0, reference_reruns=0,
                  ft_starts=len(list(LOG.glob('ft_*/command.txt'))), ft_completed=ft_done,
                  ft_epochs=epochs, ft_head_epochs=head, ft_joint_epochs=joint,
                  ft_attempted_updates=attempted, ft_completed_updates=completed,
                  formal_starts=len(list(LOG.glob('test_*/command.txt'))), formal_completed=formal,
                  pytest_starts=int((LOG / 'final/command.txt').exists()))
    if enforce and (result['ft_starts'] > 70 or epochs > 4900 or attempted > 112000
                    or head > 700 or joint > 4200 or result['formal_starts'] > 70
                    or result['pytest_starts'] > 1):
        raise RuntimeError('hard budget exceeded')
    return result


def wait(path):
    while not path.exists():
        write(OUT / 'consumption.json', consumption())
        time.sleep(5)
    write(OUT / 'consumption.json', consumption())
    if int(path.read_text()) != 0:
        raise RuntimeError(str(path) + ' nonzero exit; no retry')


def matrix(formal, adopt=False):
    pending = []
    running = {}
    for arm in ARMS:
        for task in TASKS:
            for fold in range(5):
                folder = LOG / (('test_' if formal else 'ft_') + f'{arm}_{task}_{fold}')
                if adopt and folder.exists():
                    path = folder / 'exit.txt'
                    if path.exists():
                        wait(path)
                        if not (ft_folder(arm, task, fold) / 'metrics.json').exists():
                            raise ValueError('completed adopted FT has no result')
                    else:
                        gpu = int(read(folder / 'environment.json')['CUDA_VISIBLE_DEVICES'])
                        if gpu not in ELIGIBLE or gpu in running:
                            raise ValueError('adopted active FT GPU conflict')
                        running[gpu] = path
                        ACTIVE.append(folder)
                else:
                    pending.append((arm, task, fold))
    units = iter(pending)
    exhausted = False
    while running or not exhausted:
        for gpu, path in list(running.items()):
            if path.exists():
                wait(path)
                del running[gpu]
        for gpu in ELIGIBLE:
            if gpu in running or exhausted:
                continue
            try:
                arm, task, fold = next(units)
            except StopIteration:
                exhausted = True
                break
            command = ['python', 'scripts/run_glt_ft_raw_targets_unit.py', '--arm', arm,
                       '--task', task, '--fold', str(fold)]
            if formal:
                command.append('--formal')
            running[gpu] = launch(('test_' if formal else 'ft_') + f'{arm}_{task}_{fold}', command, [gpu])
        write(OUT / 'consumption.json', consumption())
        if running:
            time.sleep(5)


def finish_existing_ft_for_handoff():
    # Wait on recorded exits, not GPU PIDs: nvidia-smi reports host-namespace PIDs.
    for folder in sorted(LOG.glob('ft_*')):
        wait(folder / 'exit.txt')


def resources():
    global ELIGIBLE
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                         '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True)
    rows = [r.split(',') for r in inventory.strip().splitlines()]
    allowed = {int(r[0]): (r[1].strip(), float(r[2])) for r in rows if int(r[0]) in GPUS}
    ELIGIBLE = tuple(g for g in GPUS if g in allowed)
    write(OUT / 'resource_check.json', dict(authorized_gpus=GPUS, eligible_gpus=ELIGIBLE,
          unavailable_authorized_gpus=[g for g in GPUS if g not in allowed], inventory=inventory, compute_apps=apps))
    if not ELIGIBLE:
        raise RuntimeError('no authorized physical GPU available')
    busy = {r.split(',')[0].strip() for r in apps.strip().splitlines() if r.strip()}
    if any(allowed[g][0] in busy or allowed[g][1] < 12000 for g in ELIGIBLE):
        raise RuntimeError('authorized GPU busy or insufficient memory; no other process will be stopped')
    path = ROOT / 'CURRENT_PLAN.md'
    path.write_text(path.read_text() + '\n\nN1实际资源：授权物理GPU' + '/'.join(map(str, GPUS))
                    + '；本次可见且使用GPU' + '/'.join(map(str, ELIGIBLE))
                    + '，单卡单单元、内部cuda:0。不可见授权卡不调度，不使用授权集合外设备，训练条件和预算不变。\n')


def main():
    resource_fix = sys.argv[1:] == ['--continue-unstarted-after-resource-fix']
    adopt = sys.argv[1:] == ['--adopt-resource-change'] or resource_fix
    if sys.argv[1:] and not adopt:
        raise ValueError('no automatic resume/retry')
    from src.training.glt_dual_runtime import require_tmux
    require_tmux()
    if adopt:
        if (not OUT.is_dir() or ((OUT / 'failure.json').exists() and not resource_fix)
                or any((OUT / p).exists() for p in ('formal', 'evaluation_before_test.json'))
                or not (OUT / 'resource_handoff_authorization.json').exists()):
            raise ValueError('resource handoff limited to ongoing, nonfailed FT')
        contract = read(OUT / 'resource_handoff_authorization.json')
        if contract.get('authorized_gpus') != list(GPUS) or contract.get('retry') is not False:
            raise ValueError('resource handoff authorization mismatch')
        if resource_fix:
            failure = read(OUT / 'failure.json')
            if (failure.get('error') != "RuntimeError('authorized GPU busy or insufficient memory; no other process will be stopped')"
                    or not (OUT / 'resource_fix_scope.json').exists()):
                raise ValueError('continuation limited to recorded scheduler resource-check error')
    else:
        OUT.mkdir(parents=True, exist_ok=False)

    def interrupted(signum, frame):
        raise RuntimeError('controller interrupted by signal ' + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if adopt:
            finish_existing_ft_for_handoff()
        resources()
        if adopt:
            wait(LOG / 'initialization/exit.txt')
        else:
            wait(launch('initialization', ['python', 'scripts/run_glt_ft_raw_targets_unit.py', '--initialize']))
        matrix(False, adopt=adopt)
        count = consumption()
        if (count['ft_starts'], count['ft_completed'], count['ft_epochs'], count['ft_completed_updates']) != (70, 70, 4900, 112000):
            raise ValueError('FT matrix/budget incomplete')
        wait(launch('evaluation_freeze', ['python', 'scripts/select_glt_ft_raw_targets.py']))
        matrix(True)
        if consumption()['formal_completed'] != 70:
            raise ValueError('formal matrix incomplete')
        wait(launch('summary', ['python', 'scripts/select_glt_ft_raw_targets.py', '--summarize']))
        wait(launch('final', ['python', '-m', 'pytest', '-q', 'tests/test_glt_ft_raw_targets.py'], limit=300))
        write(OUT / 'execution.json', dict(status='待审查', engineering='PASS', independent_review='NOT_RUN',
              consumption=consumption(), time=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        wait(launch('handoff', ['python', 'scripts/select_glt_ft_raw_targets.py', '--handoff']))
        wait(launch('actual_diff', ['python', 'scripts/select_glt_ft_raw_targets.py', '--capture-diff']))
    except BaseException as error:
        for folder in ACTIVE:
            if not (folder / 'exit.txt').exists() and (folder / 'pid.txt').exists():
                pid = int((folder / 'pid.txt').read_text())
                try:
                    command = Path(f'/proc/{pid}/cmdline').read_bytes().decode()
                    if 'glt_ft_raw_targets' in command:
                        os.kill(pid, signal.SIGTERM)
                except (FileNotFoundError, ProcessLookupError):
                    pass
        state = '需返修' if (LOG / 'final/command.txt').exists() else '阻断'
        write(OUT / 'consumption.json', consumption(enforce=False))
        failure_path = OUT / ('recovery1/failure.json' if resource_fix else 'failure.json')
        write(failure_path, dict(status=state, error=repr(error), traceback=traceback.format_exc(),
              consumption=consumption(enforce=False), retry=False))
        REVIEW.mkdir(parents=True, exist_ok=True)
        from scripts.select_glt_ft_raw_targets import handoff, capture_diff
        if not (REVIEW / 'delivery.md').exists():
            handoff()
        if not (REVIEW / 'actual_changes.diff').exists():
            capture_diff()
        raise


if __name__ == '__main__':
    main()
