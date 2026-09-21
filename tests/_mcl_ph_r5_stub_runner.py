#!/usr/bin/env python3
"""A stand-in runner for the launcher's completion contract, used by tests only.

It consumes no budget: no model, no data, no forward.  Its whole job is to leave
behind one of the record shapes the contract must tell apart, so the launcher's
"exit code 0 is not enough" rule can be exercised with a real subprocess instead
of argued about.  ``STUB_MODE`` picks the shape:

- ``pass``              training, export and cleanup all finished (the accepted shape)
- ``training_complete`` training and export finished, cleanup never did, and the
                        process still exits 0 (the r3 stall's record, minus the hang)
- ``fail``              no products, nonzero exit
- ``hang``              sleeps past any external timeout, writing nothing
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stop-after-step', type=int, default=2)
    args, _unknown = parser.parse_known_args()
    mode = os.environ.get('STUB_MODE', 'pass')
    output = Path(args.output)
    if mode == 'hang':
        time.sleep(3600)
        return 0
    if mode == 'fail':
        output.mkdir(parents=True, exist_ok=True)
        write(output / 'runtime.json', {'status': 'FAILED', 'stub': mode})
        return 3
    output.mkdir(parents=True, exist_ok=True)
    step = int(args.stop_after_step)
    write(output / 'run.json', {'status': 'stub', 'mode': mode})
    (output / 'steps.jsonl').write_text(json.dumps({'step': 1, 'stub': mode}) + '\n',
                                        encoding='utf-8')
    (output / f'resume_{step:05d}.pt').write_bytes(b'stub-resume')
    runtime = {'status': 'PASS', 'cleanup': 'complete', 'main_returned': True,
               'completed_steps': step, 'stub': mode}
    if mode == 'training_complete':
        runtime = {'status': 'TRAINING_COMPLETE', 'cleanup': 'pending',
                   'main_returned': False, 'completed_steps': step, 'stub': mode}
    else:
        (output / f'deploy_{step:05d}.pt').write_bytes(b'stub-deploy')
    write(output / 'runtime.json', runtime)
    return 0


if __name__ == '__main__':
    sys.exit(main())
