#!/usr/bin/env python3
"""Decide whether one finished arm may be called successful.

The launcher owns the completion marker, so it may only write it after the arm's
subprocess exited 0 *and* the products on disk agree with that exit code: a
runtime record whose training, export and cleanup all completed, the required
artifacts present, and the number of completed updates matching what was
requested.  A record that stops at ``TRAINING_COMPLETE`` (training and export
done, cleanup missing, no deploy package) is exactly the shape the r3 hang left
behind, and it must be rejected here rather than accepted on the exit code of a
launcher that had already lost the process.

Two contract levels, because the two runners differ and only one is in scope for
this route's record: ``--strict-cleanup`` (the MCL-PH runner) additionally
requires this route's own products and the ``cleanup``/``main_returned`` fields.
The r1 reference runner's ``runtime.json`` has no such fields and is not changed
by this round, so its arm gets the base level.

Model-free: it reads the arm directory the runner wrote and nothing else.
"""
import argparse
import json
import sys
from pathlib import Path


def verify(arm_dir, updates, strict_cleanup):
    problems = []
    arm_dir = Path(arm_dir)
    if not arm_dir.is_dir():
        return [f'arm directory is missing: {arm_dir}'], {}
    evidence = {}
    required = ['runtime.json']
    if strict_cleanup:
        required += ['run.json', 'steps.jsonl']
    for name in required:
        if not (arm_dir / name).is_file():
            problems.append(f'missing file: {name}')
    last = f'{int(updates):05d}'
    for name in (f'resume_{last}.pt', f'deploy_{last}.pt'):
        path = arm_dir / name
        if not path.is_file():
            problems.append(f'missing file: {name}')
        else:
            evidence[name] = path.stat().st_size
    runtime_path = arm_dir / 'runtime.json'
    if runtime_path.is_file():
        try:
            runtime = json.loads(runtime_path.read_text(encoding='utf-8'))
        except ValueError as error:
            problems.append(f'runtime.json is not JSON: {error}')
            runtime = {}
        status = runtime.get('status')
        evidence['status'] = status
        evidence['completed_steps'] = runtime.get('completed_steps')
        evidence['cleanup'] = runtime.get('cleanup')
        evidence['main_returned'] = runtime.get('main_returned')
        if status != 'PASS':
            problems.append(f'runtime.json status is {status!r}, not PASS')
        if runtime.get('completed_steps') != int(updates):
            problems.append(f"runtime.json completed_steps is "
                            f"{runtime.get('completed_steps')!r}, expected {int(updates)}")
        if strict_cleanup:
            if runtime.get('cleanup') != 'complete':
                problems.append(f"runtime.json cleanup is {runtime.get('cleanup')!r}, "
                                f"not 'complete' (training without a finished cleanup)")
            if runtime.get('main_returned') is not True:
                problems.append('runtime.json does not state that main() returned, so the '
                                'process exit code is the only evidence for this arm')
    return problems, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--arm-dir', required=True)
    parser.add_argument('--updates', type=int, required=True)
    parser.add_argument('--strict-cleanup', action='store_true')
    args = parser.parse_args()
    problems, evidence = verify(args.arm_dir, args.updates, args.strict_cleanup)
    print(json.dumps({'arm': args.label, 'arm_dir': str(args.arm_dir),
                      'strict_cleanup': bool(args.strict_cleanup),
                      'verdict': 'PASS' if not problems else 'REJECTED',
                      'evidence': evidence, 'problems': problems}, sort_keys=True))
    return 0 if not problems else 1


if __name__ == '__main__':
    sys.exit(main())
