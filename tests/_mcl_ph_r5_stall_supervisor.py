#!/usr/bin/env python3
"""Stop this run's own process tree when its stage log stops advancing.

r4 established that a stall in the export epilogue leaves a process that answers
neither SIGINT nor an outside stack request (ptrace is restricted in this
environment), so the r5 order asks for a bounded external stop: 120 s without
progress must produce a stack, 180 s without progress must stop this process
tree and preserve the site.

The supervisor only ever signals the descendants of the root PID it is given, so
nothing else on the machine can be affected by it.  It kills nothing on a run
that keeps making progress, and it exits on its own as soon as the root process
is gone.

Usage (in the tmux window that owns the run):

    bash scripts/run_mcl_ph_pretrain_smoke.sh ... &
    LAUNCHER=$!
    python tests/_mcl_ph_r5_stall_supervisor.py --root-pid "$LAUNCHER" \
        --stages-dir results/.../cat --report results/.../stall_supervisor.json
    wait "$LAUNCHER"
"""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

EXIT_NO_INTERVENTION = 0
EXIT_STOPPED_THE_RUN = 9
EXIT_BAD_INVOCATION = 7


def alive(pid):
    """A zombie is not a live process: it cannot make progress and cannot be killed."""
    try:
        with open(f'/proc/{int(pid)}/stat', 'r', encoding='utf-8', errors='replace') as handle:
            stat = handle.read()
    except OSError:
        return False
    try:
        return stat.rsplit(')', 1)[1].split()[0] != 'Z'
    except IndexError:
        return False


def descendants(root):
    """Every live descendant of ``root``, parents before children."""
    children = {}
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        try:
            with open(f'/proc/{entry}/stat', 'r', encoding='utf-8', errors='replace') as handle:
                stat = handle.read()
        except OSError:
            continue
        try:
            ppid = int(stat.rsplit(')', 1)[1].split()[1])
        except (IndexError, ValueError):
            continue
        if stat.rsplit(')', 1)[1].split()[0] == 'Z':
            continue
        children.setdefault(ppid, []).append(int(entry))
    order, stack = [], [int(root)]
    while stack:
        for child in children.get(stack.pop(), []):
            order.append(child)
            stack.append(child)
    return order


def signature(stages_dir, log_path):
    """A change in any of these numbers means the run made observable progress."""
    total, newest = 0, 0.0
    for pattern in ('stages_rank*.log', 'stall_stack_rank*.txt'):
        for path in Path(stages_dir).glob(pattern):
            try:
                info = path.stat()
            except OSError:
                continue
            total += info.st_size
            newest = max(newest, info.st_mtime)
    if log_path is not None and Path(log_path).is_file():
        info = Path(log_path).stat()
        total += info.st_size
        newest = max(newest, info.st_mtime)
    return (total, round(newest, 3))


def stage_tail(stages_dir, lines=3):
    tail = {}
    for path in sorted(Path(stages_dir).glob('stages_rank*.log')):
        try:
            content = path.read_text(encoding='utf-8', errors='replace').splitlines()
        except OSError:
            continue
        tail[path.name] = content[-lines:]
    return tail


def stop_tree(root, grace_seconds):
    """SIGTERM the run's own tree (root included), then SIGKILL whatever ignored it."""
    order = descendants(root)
    killed = []
    for pid in reversed(order):
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except OSError:
            pass
    try:
        os.kill(int(root), signal.SIGTERM)
        killed.append(int(root))
    except OSError:
        pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not descendants(root) and not alive(root):
            break
        time.sleep(0.5)
    survivors = [pid for pid in descendants(root) if alive(pid)]
    for pid in reversed(survivors):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return {'signalled_terminate': killed, 'signalled_kill': survivors,
            'root_alive_after_kill': alive(root),
            'survivors_after_kill': [pid for pid in survivors if alive(pid)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-pid', type=int, required=True)
    parser.add_argument('--stages-dir', required=True)
    parser.add_argument('--log', default=None)
    parser.add_argument('--report', required=True)
    parser.add_argument('--poll-seconds', type=float, default=5.0)
    parser.add_argument('--stack-seconds', type=float, default=120.0)
    parser.add_argument('--stop-seconds', type=float, default=180.0)
    parser.add_argument('--grace-seconds', type=float, default=15.0)
    args = parser.parse_args()

    if not alive(args.root_pid):
        print(json.dumps({'status': 'ROOT_PID_MISSING', 'root_pid': args.root_pid}),
              flush=True)
        return EXIT_BAD_INVOCATION

    report = {'root_pid': args.root_pid, 'stages_dir': str(args.stages_dir),
              'log': args.log, 'stack_seconds': args.stack_seconds,
              'stop_seconds': args.stop_seconds, 'poll_seconds': args.poll_seconds,
              'intervened': False}
    last = signature(args.stages_dir, args.log)
    last_change = time.monotonic()
    stack_at = None
    while True:
        if not alive(args.root_pid):
            report['status'] = 'ROOT_EXITED'
            report['silent_seconds'] = round(time.monotonic() - last_change, 3)
            break
        current = signature(args.stages_dir, args.log)
        now = time.monotonic()
        if current != last:
            last, last_change = current, now
        silent = now - last_change
        if stack_at is None and silent >= args.stack_seconds:
            stack_at = round(silent, 3)
            report['stack_requested_after_silent_seconds'] = stack_at
            print(json.dumps({'event': 'STACK_REQUESTED', 'silent_seconds': stack_at,
                              'note': 'the in-process watchdog owns the dump; the stacks '
                                      'land in stall_stack_rank*.txt'}), flush=True)
        if silent >= args.stop_seconds:
            report['status'] = 'STOPPED_BY_SUPERVISOR'
            report['intervened'] = True
            report['silent_seconds_at_stop'] = round(silent, 3)
            report['stopped_at_monotonic'] = time.perf_counter()
            report['stage_tail'] = stage_tail(args.stages_dir)
            report['signature_at_stop'] = list(current)
            report['tree'] = stop_tree(args.root_pid, args.grace_seconds)
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True),
                                         encoding='utf-8')
            print(json.dumps({'event': 'STOPPED_BY_SUPERVISOR',
                              'silent_seconds': report['silent_seconds_at_stop'],
                              'tree': report['tree']}), flush=True)
            return EXIT_STOPPED_THE_RUN
        time.sleep(args.poll_seconds)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({'event': report['status'],
                      'silent_seconds_at_exit': report.get('silent_seconds')}), flush=True)
    return EXIT_NO_INTERVENTION


if __name__ == '__main__':
    sys.exit(main())
