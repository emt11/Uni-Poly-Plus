#!/usr/bin/env python3
"""Hand off a running legacy main build to parallel retry, then audit it.

Wait for the durable main_complete report before sending SIGINT to the exact
legacy parent. Its finally block closes the writer; no replacement writer is
opened until the parent exits and its shell records the exit code.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def process_identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return None
    if stat[0] == 'Z':
        return None
    return int(stat[19]), [part.decode() for part in command if part]


def read_progress(output):
    try:
        return json.loads((output / 'progress.json').read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        # The legacy report is rewritten in place; a partial read is retried.
        return {}


def wait_and_stop_main(pid, start_ticks, output, exit_path, report):
    interrupted = False
    while True:
        progress = read_progress(output)
        if exit_path.is_file():
            code = int(exit_path.read_text().strip())
            if code != 0 and not (interrupted and code == 130):
                raise RuntimeError(f'legacy build exited unexpectedly: {code}')
            if progress.get('phase') not in ('main_complete', 'finished'):
                raise RuntimeError('legacy build exited without a complete main report')
            return {'legacy_exit_code': code, 'interrupted_for_handoff': interrupted,
                    'main_report': progress}
        identity = process_identity(pid)
        if identity is not None and identity[0] != start_ticks:
            raise RuntimeError('legacy PID has been reused; refusing to signal it')
        if progress.get('phase') == 'main_complete' and not interrupted:
            if progress.get('done') != progress.get('target'):
                raise RuntimeError('main_complete report is incomplete')
            if identity is not None:
                command = identity[1]
                if ('scripts/build_mcl_ph_full_geometry.py' not in command or
                        '--output' not in command or
                        (ROOT / command[command.index('--output') + 1]).resolve() != output):
                    raise RuntimeError('legacy command differs; refusing to signal it')
                report('STOPPING_LEGACY_RETRY', main_report=progress)
                os.kill(pid, signal.SIGINT)
                interrupted = True
        # Once interrupted, wait for normal finally/ProcessPool shutdown.
        # No SIGKILL fallback and no second writer while this process is alive.
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-pid', type=int, required=True)
    parser.add_argument('--main-start-ticks', type=int, required=True)
    parser.add_argument('--main-exit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--log-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=56)
    parser.add_argument('--retry-workers', type=int, default=56)
    parser.add_argument('--poll-only-once', action='store_true',
                        help='Check configuration and process identity without signalling')
    args = parser.parse_args()
    output = args.output.resolve()
    log_dir = args.log_dir.resolve()
    if not 1 <= args.retry_workers <= 64 or not 1 <= args.workers <= 64:
        parser.error('workers must be in 1..64')
    metadata = json.loads((output / 'metadata.json').read_text())
    if (metadata['kind'] != 'pi1m' or metadata['main_workers'] != args.workers or
            metadata['limit'] != 0 or metadata['retry'] != 'TIMEOUT only; one worker after main'):
        raise ValueError('handoff requires the original full PI1M serial-retry contract')
    identity = process_identity(args.main_pid)
    if identity is None or identity[0] != args.main_start_ticks:
        raise RuntimeError('legacy main process is absent or differs from the recorded identity')
    if args.poll_only_once:
        print(json.dumps({'configuration': 'PASS', 'pid': args.main_pid,
                          'progress': read_progress(output)}), flush=True)
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    if (log_dir / 'exit.code').exists() or (log_dir / 'retry.stdout.log').exists():
        raise FileExistsError('handoff logs already contain an execution; use a new log directory')

    def report(phase, **detail):
        status = {'phase': phase, 'time_unix': time.time(),
                  'main_pid': args.main_pid, 'retry_workers': args.retry_workers, **detail}
        temporary = log_dir / 'status.json.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(log_dir / 'status.json')
        print(json.dumps(status), flush=True)

    def run_stage(name, command):
        report(f'{name.upper()}_RUNNING', command=command, cwd=str(ROOT))
        with (log_dir / f'{name}.stdout.log').open('x') as stdout, \
                (log_dir / f'{name}.stderr.log').open('x') as stderr:
            result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        (log_dir / f'{name}.exit.code').write_text(f'{result.returncode}\n')
        if result.returncode:
            raise RuntimeError(f'{name} exited with {result.returncode}; inspect preserved logs')

    try:
        report('WAIT_MAIN', main_report=read_progress(output))
        legacy = wait_and_stop_main(args.main_pid, args.main_start_ticks, output,
                                    args.main_exit.resolve(), report)
        (log_dir / 'legacy_handoff.json').write_text(json.dumps(legacy, indent=2) + '\n')
        run_stage('retry', [sys.executable, 'scripts/build_mcl_ph_full_geometry.py',
                           '--kind', 'pi1m', '--output', str(output),
                           '--workers', str(args.workers), '--retry-only',
                           '--retry-workers', str(args.retry_workers),
                           '--retry-progress-every', '10'])
        run_stage('audit', [sys.executable, 'scripts/audit_mcl_ph_full_geometry.py',
                           '--kind', 'pi1m', '--output', str(output),
                           '--progress-every', '10000'])
        report('FINISHED')
        (log_dir / 'exit.code').write_text('0\n')
    except Exception as error:
        report('FAILED', error_type=type(error).__name__, reason=str(error))
        (log_dir / 'exit.code').write_text('1\n')
        raise


if __name__ == '__main__':
    main()
