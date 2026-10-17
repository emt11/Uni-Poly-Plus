"""Single G1 controller: exclusive output roots, slots, freeze, reporting, one test."""
import argparse
import contextlib
import copy
import csv
import difflib
import json
import math
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
import numpy as np
import torch
from src.training.ua2d_fp_gate import *
from src.training.glt_dual_runtime import require_tmux, save_checkpoint


def launch_job(name, argv, gpu=''):
    folder = LOG / name
    folder.mkdir(parents=True, exist_ok=False)
    window = 'ua2d_g2_r7_' + name
    command = dict(argv=argv, cwd=str(ROOT), environment=dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu)),
                   window=window, artifacts=str(OUT))
    write(folder / 'command.json', command)
    line = shlex.join([PYTHON, str(Path(__file__).resolve()), '--job', str(folder)])
    (folder / 'command.sh').write_text('#!/bin/bash\ncd ' + shlex.quote(str(ROOT)) + '\nexec ' + line + '\n')
    subprocess.run(['tmux', 'new-window', '-d', '-t', 'Uni-Poly', '-n', window,
        'tmux set-option -w -t "$TMUX_PANE" remain-on-exit on\nexec bash ' + shlex.quote(str(folder / 'command.sh'))], check=True)
    return folder


def job_wrapper(folder):
    require_tmux()
    folder = Path(folder)
    c = read(folder / 'command.json')
    execution = dict(command=c, wrapper_pid=os.getpid(), pid=None, start=now(), end=None, exit=None)
    write(folder / 'execution.json', execution)
    rc = 1
    try:
        with (folder / 'stdout.log').open('x') as out, (folder / 'stderr.log').open('x') as err:
            if (folder / 'stop_requested.json').exists():
                raise InterruptedError('controller cancelled reserved unit before worker launch')
            proc = subprocess.Popen(c['argv'], cwd=c['cwd'], env=dict(os.environ, **c['environment']), stdout=out, stderr=err)
            execution['pid'] = proc.pid
            execution['worker_identity'] = process_row(proc.pid)
            write(folder / 'execution.json', execution, True)
            rc = proc.wait()
    except BaseException:
        execution['error'] = traceback.format_exc()
    finally:
        execution.update(end=now(), exit=rc)
        write(folder / 'execution.json', execution, True)
        (folder / 'exit.txt').write_text(str(rc) + '\n')
    return rc


def process_row(pid):
    item = Path('/proc') / str(pid)
    try:
        argv = [v.decode(errors='replace') for v in (item / 'cmdline').read_bytes().split(b'\0') if v]
        exe = os.readlink(item / 'exe')
        stat = (item / 'stat').read_text().rsplit(')', 1)[1].split()
        env = (item / 'environ').read_bytes().split(b'\0')
        visible = next((v.split(b'=', 1)[1].decode() for v in env if v.startswith(b'CUDA_VISIBLE_DEVICES=')), '')
        status = (item / 'status').read_text().splitlines()
        ns = next((line.split()[1:] for line in status if line.startswith('NSpid:')), [])
        return dict(pid=int(pid), ppid=int(stat[1]), start_ticks=int(stat[19]),
                    argv=argv, exe=exe, visible=visible, nspid=[int(v) for v in ns])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


_GPU_MAPPING_CACHE = None


def parse_gpu_csv(raw):
    rows = []
    for row in csv.reader(raw.splitlines(), skipinitialspace=True):
        if not row or not any(value.strip() for value in row):
            continue
        if len(row) != 2:
            raise RuntimeError('Malformed nvidia-smi CSV; raw=' + repr(raw))
        rows.append(tuple(value.strip() for value in row))
    return rows


def gpu_query(columns):
    argv = ['nvidia-smi', '-i', ','.join(map(str, GPUS)), '--query-' + columns, '--format=csv,noheader,nounits']
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode or result.stderr.strip():
        raise RuntimeError('nvidia-smi query failed: ' + repr(dict(argv=argv, exit=result.returncode,
                           stdout=result.stdout, stderr=result.stderr)))
    return result.stdout


def observe_gpus():
    global _GPU_MAPPING_CACHE
    started = now()
    if _GPU_MAPPING_CACHE is None:
        uuid_rows = gpu_query('gpu=index,uuid')
        pairs = parse_gpu_csv(uuid_rows)
        if any(not index.isdigit() or not uuid.startswith('GPU-') for index, uuid in pairs):
            raise RuntimeError('Invalid GPU mapping; raw=' + repr(uuid_rows))
        mapping = {uuid: int(index) for index, uuid in pairs}
        if set(mapping.values()) != set(GPUS):
            raise RuntimeError('GPU2/3 mapping absent; raw=' + repr(uuid_rows))
        _GPU_MAPPING_CACHE = (mapping, uuid_rows)
    mapping, uuid_rows = _GPU_MAPPING_CACHE
    raw = gpu_query('compute-apps=gpu_uuid,pid')
    nvidia, proc_pids, processes = {'2': [], '3': []}, {'2': [], '3': []}, []
    for uuid, pid in parse_gpu_csv(raw):
        if not pid.isdigit():
            raise RuntimeError('Invalid NVML PID; raw=' + repr(raw))
        gpu = mapping.get(uuid.strip())
        if gpu in GPUS:
            nvidia[str(gpu)].append(int(pid))
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():
            continue
        row = process_row(int(item.name))
        if row is None or not Path(row['exe']).name.startswith('python'):
            continue
        processes.append(row)
        # Inspect the program position, never arbitrary embedded text in argv.
        programs = row['argv'][1:3]
        experiment = any(any(word in Path(arg).name for word in ('train', 'finetune', '_unit.py')) for arg in programs)
        is_wrapper = any(Path(arg).name == 'run_ua2d_fp_gate.py' for arg in programs)
        if row['visible'] in ('2', '3') and experiment and not is_wrapper:
            proc_pids[row['visible']].append(row['pid'])
    return dict(start=started, end=now(), nvidia_raw=raw, gpu_mapping_raw=uuid_rows,
                nvidia_pids=nvidia, proc_pids=proc_pids, processes=processes,
                independent_nvml_pid_view=True)


def initialize():
    from src.training.ua2d_pt20k import require_complete
    pt = require_complete()
    (OUT / 'initialization').mkdir()
    write(OUT / 'initialization/pretraining.json', pt)
    r10 = read(R10 / 'selection_before_test.json')
    if not r10.get('frozen') or r10['weight'] != .75:
        raise ValueError('frozen r10 .75 required')
    ua_freeze = read(U1 / 'evaluation_before_test.json')
    cfg = read(U1 / 'initialization/UA_2D.json')
    from scripts.run_glt_align_2d_unit import validate_config
    validate_config(cfg, 'UA_2D')
    ua = {(u['task'], u['fold']): u for u in ua_freeze['units'] if u['arm'] == 'UA_2D'}
    sources = {(u['task'], u['fold']): u for u in r10['validation_sources']}
    units = r10['r9_freeze']['units']
    expected = {(t, f) for t in TASKS for f in range(5)}
    if len(units) != 35 or {(u['task'], u['fold']) for u in units} != expected or set(ua) != expected or set(sources) != expected:
        raise ValueError('35 source bindings required')
    old_policy = read(OLD_OUT / 'initialization/policy.json')
    if old_policy != {k: ('r2' if k == 'revision' else 'G1' if k == 'cycle' else v)
                      for k, v in POLICY.items() if k not in ('parent', 'parent_step')}:
        raise ValueError('old scientific gate policy changed')
    initial = torch.load(OLD_OUT / 'initialization/gate_initial.pt', map_location='cpu', weights_only=False)
    expected_initial = initial_state()
    if set(initial) != set(expected_initial) or any(not torch.equal(initial[k], expected_initial[k]) for k in initial):
        raise ValueError('inherited gate initialization mismatch')
    shutil.copyfile(OLD_OUT / 'initialization/gate_initial.pt', OUT / 'initialization/gate_initial.pt')
    write(OUT / 'initialization/policy.json', POLICY)
    write(OUT / 'initialization/r10_freeze.json', r10)
    for u in units:
        t, f = u['task'], u['fold']
        ref = ua[t, f]
        manifest = read(SPLIT / (t + '.json'))
        entry = manifest['folds'][f]
        validate_split(u, u['sample_count'])
        for role in ('train', 'validation', 'test'):
            values = entry[role + '_indices']
            if u[role + '_indices'] != (sorted(values) if role == 'test' else values):
                raise ValueError('frozen split changed')
        if ref['parent'] != str(OLD_PARENT) or ref['head_initial'] != str(head_path(t, f)) or ref['scaler'] != u['scaler'] or ref['test_indices'] != u['test_indices']:
            raise ValueError('UA parent/head/scaler/split mismatch')
        if ref['prediction_path'] != 'pure_2d' or ref['input_dim'] != 512 or ref['parent_align_revision'] != 'glt_align_2d_u1':
            raise ValueError('UA schema mismatch')
        for path in (ref['best'], ref['parent'], ref['head_initial']):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        val_source = sources[t, f]
        expected_cache = Path(u['model']).parent / 'validation' / f"alpha_{u['alpha']:g}" / 'predictions.npz'
        if val_source['alpha'] != u['alpha'] or val_source['model'] != u['model'] or val_source['path'] != str(expected_cache):
            raise ValueError('r10 selected alpha/source mismatch')
        if read(expected_cache.parent / 'metrics.json') != u['chosen_validation'] or u['chosen_validation']['alpha'] != u['alpha']:
            raise ValueError('selected alpha cache metadata mismatch')
        if read(Path(ref['best']).parent / 'scaler.json') != u['scaler']:
            raise ValueError('UA scaler file mismatch')
        binding = dict(unit=u, ua=ref, manifest=manifest, identities=r10['r9_freeze']['identities'][t],
                       validation_source=val_source, seed=42 + f, ua_config=cfg)
        if binding != read(OLD_OUT / 'initialization/bindings' / t / f'fold{f}.json'):
            raise ValueError('old binding content mismatch')
        binding.update(training_parent=str(PARENT), training_parent_step=20000)
        write(OUT / 'initialization/bindings' / t / f'fold{f}.json', binding)
    return units


class Controller:
    def __init__(self):
        self.budget = {k: 0 for k in LIMITS}
        self.slots = Slots()
        self.jobs = {}
        self.events = []
        self.fp = None
        self.units = {}
        self.phase = 'initialize'
        self.last_observation = None
        self.snapshot_sequence = 0

    def ledger(self):
        b = self.budget
        b['epochs'] = b['attempted'] = b['completed'] = b['ft_completed'] = b['formal_completed'] = 0
        b['ft_graphs'] = b['test_graphs'] = b['gate_arrays'] = 0
        unknown = []
        for key, job in self.jobs.items():
            p = unit_path(job['arm'], job['task'], job['fold'], job['formal'])
            if not job['formal']:
                if (p / 'updates.json').exists():
                    updates = read(p / 'updates.json')
                    b['attempted'] += updates['attempted']
                    b['completed'] += updates['completed']
                    if updates.get('inflight'):
                        unknown.append(dict(unit=key, kind='update_inflight', sequence=updates.get('sequence')))
                elif p.exists():
                    unknown.append(dict(unit=key, kind='update_record_not_yet_created'))
                if (p / 'progress.json').exists():
                    b['epochs'] += len(read(p / 'progress.json')['history'])
            if (p / 'input_consumption.json').exists():
                inputs = read(p / 'input_consumption.json')
                # Charge attempted materializations, including in-flight work.
                b['test_graphs' if job['formal'] else 'ft_graphs'] += inputs['attempted']
                if inputs['inflight'] is not None:
                    unknown.append(dict(unit=key, kind='input_inflight', identity=inputs['inflight']))
            elif p.exists():
                unknown.append(dict(unit=key, kind='input_record_not_yet_created'))
            if (p / 'metrics.json').exists():
                b['formal_completed' if job['formal'] else 'ft_completed'] += 1
            if job['formal']:
                b['gate_arrays'] += int((p / 'test_predictions.npz').exists()) + int((p / 'fused_predictions.npz').exists())
        if self.fp:
            b['fp_generated'] = self.fp.generated
        record = consumption_record(b, unknown)
        write(OUT / 'consumption.json', record, True)
        if self.fp:
            write(OUT / 'fp_consumption.json', dict(old_generated=self.fp.inherited,
                new_generated=self.fp.generated, cumulative_generated=self.fp.inherited + self.fp.generated,
                controller_hits=self.fp.hits, writer_pid=os.getpid(), cache=str(OUT / 'fingerprints.sqlite')), True)

    def refresh_bindings(self, snapshot=None):
        for key, active in self.slots.active.items():
            job = self.jobs.get(key)
            if job is None:
                continue
            path = Path(job['log']) / 'execution.json'
            if path.exists():
                e = read(path)
                active.update(pid=e.get('pid'), wrapper_pid=e.get('wrapper_pid'),
                              worker_identity=e.get('worker_identity'))
            if snapshot is not None and not active.get('wrapper_pid'):
                # Match the exact persisted job path, not a Python script name.
                for row in snapshot['processes']:
                    argv = row['argv']
                    if '--job' in argv and argv[argv.index('--job') + 1:] == [job['log']]:
                        active['wrapper_pid'] = row['pid']

    def register_cuda_requests(self):
        # Serialize only the first required model.to(cuda), not FT execution.
        # Workers cannot train until the one new NVML identity is bound.
        for key, active in list(self.slots.active.items()):
            if active.get('nvml_pid') is not None or key not in self.jobs:
                continue
            job = self.jobs[key]
            unit = unit_path(job['arm'], job['task'], job['fold'], job['formal'])
            request_path = unit / 'cuda_request.json'
            if not request_path.exists():
                continue
            self.refresh_bindings()
            request = read(request_path)
            if request['pid'] != active.get('pid') or request['gpu'] != active['gpu']:
                raise RuntimeError('CUDA request worker identity mismatch')
            before = observe_gpus()
            self.record_sample(before)
            write(unit / 'cuda_grant.json', dict(pid=request['pid'], gpu=active['gpu'],
                nvml_before=before['nvidia_pids'], time=now()))
            deadline = time.monotonic() + 60
            while not (unit / 'cuda_ready.json').exists():
                if (Path(job['log']) / 'exit.txt').exists():
                    raise RuntimeError('worker exited during required CUDA initialization: ' + key)
                if time.monotonic() >= deadline:
                    raise RuntimeError('required CUDA initialization timed out: ' + key)
                time.sleep(.1)
            ready = read(unit / 'cuda_ready.json')
            self.refresh_bindings()
            current = process_row(request['pid'])
            if (ready['pid'] != request['pid'] or ready['gpu'] != active['gpu'] or current is None
                    or current['start_ticks'] != ready['start_ticks']):
                raise RuntimeError('CUDA ready process birth mismatch')
            after = observe_gpus()
            evidence = dict(experiment=key, worker=ready, before=before['nvidia_pids'],
                after=after['nvidia_pids'], time=now(),
                method='one serialized required model.to(cuda); no other own CUDA births allowed')
            # Save both raw samples even when the identity is ambiguous.
            write(unit / 'cuda_birth_evidence.json', evidence)
            nvml_pid = cuda_birth_pid(before, after, active['gpu'])
            if any(v.get('nvml_pid') == nvml_pid for k, v in self.slots.active.items() if k != key):
                raise RuntimeError('NVML identity already assigned to another live experiment')
            active['nvml_pid'] = nvml_pid
            evidence['nvml_pid'] = nvml_pid
            self.jobs[key]['nvml_pid'] = nvml_pid
            write(OUT / 'jobs.json', self.jobs, True)
            self.record_sample(after)
            write(unit / 'cuda_mapped.json', evidence)
            print('CUDA PID bound', key, 'proc', request['pid'], 'NVML', nvml_pid, flush=True)

    def record_sample(self, raw):
        self.refresh_bindings(raw)  # reconcile births during NVML/proc sampling
        observed, detail = classify_observation(raw, self.slots.active)
        detail['counts'] = self.slots.occupancy(observed)
        detail['sequence'] = self.snapshot_sequence
        self.snapshot_sequence += 1
        self.last_observation = detail
        # Persist the exact failing observation before any check can throw.
        write(OUT / 'concurrency.json', detail, True)
        with (OUT / 'concurrency_observations.jsonl').open('a') as stream:
            stream.write(json.dumps(detail, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        if detail['errors']:
            raise RuntimeError('GPU process classification failed: ' + repr(detail['errors']))
        self.slots.record(observed)
        write(OUT / 'concurrency_peaks.json', dict(per_gpu_peak=self.slots.peak,
            total_peak=self.slots.total_peak, time=now()), True)
        return observed

    def sample(self):
        self.register_cuda_requests()
        return self.record_sample(observe_gpus())

    def prepare(self, task, fold, formal):
        if formal:
            require_freeze()
        binding = read(OUT / 'initialization/bindings' / task / f'fold{fold}.json')
        u = binding['unit']
        selected = u['test_indices'] if formal else sorted(u['train_indices'] + u['validation_indices'])
        for idx in selected:
            self.fp.ensure(binding['identities'][str(idx)]['normalized_smiles'])
        self.ledger()

    def schedule(self, formal=False):
        pending = [(a, t, f) for a in ARMS for t in TASKS for f in range(5)
                   if formal or not (unit_path(a, t, f) / 'metrics.json').exists()]
        while pending or self.slots.active:
            self.refresh_bindings()
            # Sample before releasing exited slots, so an NVML teardown lag
            # still belongs to the unit whose true exit has been recorded.
            observed = self.sample()
            released = False
            for key in list(self.slots.active):
                job = self.jobs[key]
                folder = Path(job['log'])
                if (folder / 'exit.txt').exists():
                    execution = read(folder / 'execution.json')
                    if int((folder / 'exit.txt').read_text()) != execution['exit']:
                        raise ValueError('exit record mismatch')
                    if execution.get('pid') and process_row(execution['pid']) is not None:
                        raise RuntimeError('worker still present after recorded exit')
                    record = self.slots.release(key, execution)
                    released = True
                    self.events.append(dict(event='release', experiment=key, **record, end=execution['end'], exit=execution['exit']))
                    write(OUT / 'dispatch.json', self.events, True)
                    if execution['exit'] != 0:
                        raise RuntimeError('unit failed without retry: ' + key + ' exit=' + str(execution['exit']))
                    if not (unit_path(job['arm'], job['task'], job['fold'], formal) / 'metrics.json').is_file():
                        raise ValueError('successful process missing metrics')
                    print('completed', key, flush=True)
            # Re-sample in the ordinary dispatch loop after any slot release;
            # births/exits and external occupancy may have changed meanwhile.
            if released:
                observed = self.sample()
            if pending:
                counts = self.slots.occupancy(observed)
                candidates = [g for g in GPUS if counts[g] < 4]
                if candidates and sum(counts.values()) < 4:
                    gpu = min(candidates, key=lambda g: (counts[g], g))
                    arm, task, fold = pending.pop(0)
                    key = ('test_' if formal else 'ft_') + arm + '_' + task + '_' + str(fold)
                    self.slots.reserve(key, gpu, observed)
                    self.jobs[key] = dict(arm=arm, task=task, fold=fold, formal=formal,
                        gpu=gpu, log=str(LOG / key), intent_at=now(), state='RESERVED')
                    self.events.append(dict(event='reserve', experiment=key, gpu=gpu, start=now(),
                        occupancy=self.slots.occupancy(observed)))
                    self.budget['formal_starts' if formal else 'ft_starts'] += 1
                    write(OUT / 'jobs.json', self.jobs, True)
                    write(OUT / 'dispatch.json', self.events, True)
                    self.ledger()  # durable reservation/intent before spawn
                    self.prepare(task, fold, formal)
                    argv = [PYTHON, 'scripts/run_ua2d_fp_gate_unit.py', '--arm', arm, '--task', task, '--fold', str(fold)]
                    if formal:
                        argv.append('--formal')
                    launch_job(key, argv, gpu)
                    self.jobs[key]['state'] = 'LAUNCHED'
                    write(OUT / 'jobs.json', self.jobs, True)
                    print('started', key, 'GPU', gpu, flush=True)
                    continue
            self.ledger()
            if pending or self.slots.active:
                time.sleep(5)

    def freeze(self):
        units = []
        for arm in ARMS:
            for t in TASKS:
                for f in range(5):
                    p = unit_path(arm, t, f)
                    m = read(p / 'metrics.json')
                    source = read(OUT / 'initialization/bindings' / t / f'fold{f}.json')
                    history = read(p / 'progress.json')['history']
                    best = min(history, key=lambda r: r['validation_rmse'])
                    if m['best_epoch'] != best['epoch'] or m['epochs'] != 70 or m['policy'] != POLICY or m['scaler'] != source['unit']['scaler']:
                        raise ValueError('validation-selected best mismatch')
                    u = {k: copy.deepcopy(source['unit'][k]) for k in ('task', 'fold', 'sample_count', 'train_indices', 'validation_indices', 'test_indices', 'scaler', 'alpha')}
                    u.update({k: m[k] for k in ('arm', 'seed', 'parent', 'head_initial', 'best_epoch', 'policy', 'optimizer_groups', 'expected_updates')})
                    u.update(best=str(p / 'best.pt'), validation=str(p / 'validation_predictions.npz'), source=source['unit'],
                        module_initial=str(module_initial_path(arm, t, f)), ua=source['ua'], validation_metrics={k: m[k] for k in ('r2', 'rmse')})
                    if not Path(u['best']).is_file():
                        raise FileNotFoundError(u['best'])
                    units.append(u)
        if len(units) != 70 or sum(u['expected_updates'] for u in units) != 112000:
            raise ValueError('full scientific FT matrix incomplete for the new P2 parent')
        # Paths are metadata; no TEST arrays or TEST graph source has been read.
        r10 = read(R10 / 'final_review/seven_task_results.json')
        refs = []
        a_fixed = ROOT / 'results/ua2d_fusion_20261017/a1_prediction'
        reuse = None
        if (a_fixed / 'completion.json').exists() and (a_fixed / 'selection_before_test.json').exists():
            prior = read(a_fixed / 'selection_before_test.json')
            if prior.get('weights', {}).get('A_FIXED') == .75 and prior.get('frozen'):
                report = read(a_fixed / 'final_review/seven_task_results.json')
                reuse = report
            else:
                raise ValueError('existing A_FIXED incompatible')
        for r in r10['folds']:
            t, f = r['task'], r['fold']
            binding = read(OUT / 'initialization/bindings' / t / f'fold{f}.json')
            if r['alpha'] != binding['unit']['alpha']:
                raise ValueError('formal r10 alpha mismatch')
            refs.append(dict(task=t, fold=f, ua=str(U1 / 'formal/UA_2D' / t / f'fold{f}/test_predictions.npz'),
                krr=r['paths']['krr'], r10=r['output'], alpha=r['alpha'], r10_metrics=r['metrics']['global'],
                ua_fixed75=None if reuse is None else next(v for v in reuse['folds'] if (v['task'], v['fold']) == (t, f))['paths']['A_FIXED']))
        if len(refs) != 35 or {(r['task'], r['fold']) for r in refs} != {(t, f) for t in TASKS for f in range(5)}:
            raise ValueError('reference TEST coverage mismatch')
        payload = dict(frozen=True, frozen_at=now(), policy=POLICY, units=units,
            references=refs, reference_policy='UA_FIXED75 = .75 old UA + .25 r10 KRR',
            test_opened=False, consumption_at_freeze=consumption_record(self.budget))
        validate_frozen(payload)
        write(OUT / 'selection_before_test.json', payload)
        with (OUT / 'selection_before_test.json').open('rb') as stream:
            os.fsync(stream.fileno())
        print('ALL 70 bests frozen before TEST', flush=True)

    def stop_own(self):
        write(OUT / 'state_before_stop.json', dict(active=copy.deepcopy(self.slots.active),
            observation=self.last_observation, jobs=self.jobs, time=now()))
        for key in self.slots.active:
            folder = Path(self.jobs[key]['log'])
            if folder.exists():
                write(folder / 'stop_requested.json', dict(reason='controller stopping', time=now()))
        deadline = time.monotonic() + 45
        signalled = set()
        while self.slots.active and time.monotonic() < deadline:
            self.refresh_bindings()
            for key in list(self.slots.active):
                folder = Path(self.jobs[key]['log'])
                active = self.slots.active[key]
                pid = active.get('pid')
                current = process_row(pid) if pid else None
                identity = active.get('worker_identity')
                if current and pid not in signalled and identity and current['start_ticks'] == identity['start_ticks']:
                    os.kill(pid, signal.SIGTERM)
                    signalled.add(pid)
                if (folder / 'exit.txt').exists():
                    e = read(folder / 'execution.json')
                    if e.get('pid') and process_row(e['pid']) is not None:
                        continue
                    record = self.slots.release(key, e)
                    self.events.append(dict(event='stopped_release', experiment=key, **record, end=e['end'], exit=e['exit']))
            if self.slots.active:
                time.sleep(1)
        # No unrelated PID is killed. A surviving own worker may be hard-stopped
        # only with the same saved process birth identity; in-flight stays unknown.
        for active in self.slots.active.values():
            pid, identity = active.get('pid'), active.get('worker_identity')
            current = process_row(pid) if pid else None
            if current and identity and current['start_ticks'] == identity['start_ticks']:
                os.kill(pid, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while self.slots.active and time.monotonic() < deadline:
            for key in list(self.slots.active):
                folder = Path(self.jobs[key]['log'])
                if (folder / 'exit.txt').exists():
                    e = read(folder / 'execution.json')
                    record = self.slots.release(key, e)
                    self.events.append(dict(event='stopped_release', experiment=key, **record, end=e['end'], exit=e['exit']))
            if self.slots.active:
                time.sleep(1)
        write(OUT / 'dispatch.json', self.events, True)
        write(OUT / 'state_after_stop.json', dict(active=self.slots.active, time=now()))
        self.ledger()


def paper_reference():
    paper = {}
    for line in (ROOT / 'SOTA_RESULT.md').read_text().splitlines():
        cells = [c.strip() for c in line.split('|')]
        if len(cells) == 5 and cells[1].lower() in TASKS and cells[3].startswith('**'):
            value = cells[3].strip('*').split(' ± ')
            paper[cells[1].lower()] = dict(model=cells[2], r2=float(value[0]), std=float(value[1]))
    if set(paper) != set(TASKS):
        raise ValueError('paper seven-task reference missing')
    return paper


def same_truth(*arrays):
    if any(not np.array_equal(arrays[0], a) for a in arrays[1:]):
        raise ValueError('branch raw truth mismatch')


def report(controller):
    import csv
    payload = require_freeze()
    frozen = {(u['arm'], u['task'], u['fold']): u for u in payload['units']}
    refs = {(u['task'], u['fold']): u for u in payload['references']}
    labels = ('UA', 'r10', 'UA_FIXED75', 'GATE_REAL_neural', 'GATE_REAL_fused', 'GATE_NULL_neural', 'GATE_NULL_fused')
    rows = []
    index = []
    for t in TASKS:
        for f in range(5):
            binding = read(OUT / 'initialization/bindings' / t / f'fold{f}.json')
            unit, identities = binding['unit'], binding['identities']
            ref = refs[t, f]
            ua = load_arrays(ref['ua'], unit, 'test', identities)
            krr = load_arrays(ref['krr'], unit, 'test', identities)
            same_truth(ua['y_true'], krr['y_true'])
            truth = ua['y_true']
            ua_metrics = metrics(truth, ua['y_pred'])
            prior = read(U1 / 'formal/UA_2D' / t / f'fold{f}/metrics.json')
            if any(not np.isclose(ua_metrics[k], prior[k], rtol=1e-12, atol=1e-12) for k in ua_metrics):
                raise ValueError('UA stored formal metrics mismatch')
            folder = OUT / 'reference/UA_FIXED75' / t / f'fold{f}'
            folder.mkdir(parents=True, exist_ok=False)
            if ref['ua_fixed75'] is None:
                controller.budget['reference_units'] += 1
                controller.ledger()
                fixed = fixed_blend(ua['y_pred'], krr['y_pred'])
                fixed_path = folder / 'predictions.npz'
                np.savez(fixed_path, indices=unit['test_indices'], sample_keys=[identities[str(i)]['sample_key'] for i in unit['test_indices']],
                    y_true=truth, y_pred=fixed, weight=.75, alpha=unit['alpha'], scaler_mean=unit['scaler']['mean'], scaler_scale=unit['scaler']['scale'])
                controller.budget['reference_arrays'] += 1
            else:
                fixed_path = Path(ref['ua_fixed75'])
                reused = load_arrays(fixed_path, unit, 'test', identities)
                same_truth(truth, reused['y_true'])
                fixed = reused['y_pred']
                write(folder / 'reused_prediction.json', dict(path=str(fixed_path), weight=.75))
            test = dict(UA=ua_metrics, r10=ref['r10_metrics'], UA_FIXED75=metrics(truth, fixed))
            paths = dict(UA=ref['ua'], r10=ref['r10'], KRR=ref['krr'], UA_FIXED75=str(fixed_path))
            ua_val_path = Path(binding['ua']['best']).parent / 'validation_predictions.npz'
            ua_val = load_arrays(ua_val_path, unit, 'validation', identities)
            kval = load_arrays(binding['validation_source']['path'], unit, 'validation', identities, ('y_krr', 'y_g05'))
            same_truth(ua_val['y_true'], kval['y_true'])
            vy = ua_val['y_true']
            validation = dict(UA=metrics(vy, ua_val['y_pred']),
                r10=metrics(vy, fixed_blend(kval['y_g05'], kval['y_krr'])),
                UA_FIXED75=metrics(vy, fixed_blend(ua_val['y_pred'], kval['y_krr'])))
            for arm in ARMS:
                own = frozen[arm, t, f]
                p = unit_path(arm, t, f, True)
                neural = load_arrays(p / 'test_predictions.npz', unit, 'test', identities)
                same_truth(truth, neural['y_true'])
                fm = read(p / 'metrics.json')
                nm = metrics(truth, neural['y_pred'])
                if any(not np.isclose(nm[k], fm[k], rtol=1e-12, atol=1e-12) for k in nm):
                    raise ValueError('formal neural metrics mismatch')
                fused = fixed_blend(neural['y_pred'], krr['y_pred'])
                output = p / 'fused_predictions.npz'
                if output.exists():
                    raise FileExistsError(output)
                np.savez(output, indices=unit['test_indices'], sample_keys=[identities[str(i)]['sample_key'] for i in unit['test_indices']],
                    y_true=truth, y_pred=fused, weight=.75, alpha=unit['alpha'], scaler_mean=unit['scaler']['mean'], scaler_scale=unit['scaler']['scale'])
                test[arm + '_neural'] = nm
                test[arm + '_fused'] = metrics(truth, fused)
                paths[arm + '_neural'], paths[arm + '_fused'] = str(p / 'test_predictions.npz'), str(output)
                nv = load_arrays(own['validation'], unit, 'validation', identities)
                same_truth(vy, nv['y_true'])
                validation[arm + '_neural'] = metrics(vy, nv['y_pred'])
                validation[arm + '_fused'] = metrics(vy, fixed_blend(nv['y_pred'], kval['y_krr']))
                if validation[arm + '_neural'] != own['validation_metrics']:
                    raise ValueError('frozen validation metric mismatch')
            differences = {candidate + '-minus-' + reference: {m: test[candidate][m] - test[reference][m] for m in ('r2', 'rmse')}
                for candidate in labels for reference in labels if candidate != reference}
            gap = {label: {m: test[label][m] - validation[label][m] for m in ('r2', 'rmse')} for label in labels}
            row = dict(task=t, fold=f, alpha=unit['alpha'], seed=42 + f, test=test, validation=validation,
                test_minus_validation=gap, differences=differences, paths=paths)
            rows.append(row)
            index.append(dict(task=t, fold=f, paths=paths, ua_best=binding['ua'], kernel_source=binding['validation_source'],
                gate_bests={a: frozen[a, t, f]['best'] for a in ARMS}))
            write(folder / 'metrics.json', dict(test=test['UA_FIXED75'], validation=validation['UA_FIXED75']))
            controller.ledger()
    paper = paper_reference()
    tasks = {}
    for t in TASKS:
        own = [r for r in rows if r['task'] == t]
        stats = {role: {label: {m: dict(mean=float(np.mean([r[role][label][m] for r in own])),
                    std=float(np.std([r[role][label][m] for r in own], ddof=0))) for m in ('r2', 'rmse')}
                    for label in labels} for role in ('test', 'validation', 'test_minus_validation')}
        deltas = {name: {m: float(np.mean([r['differences'][name][m] for r in own])) for m in ('r2', 'rmse')}
                  for name in own[0]['differences']}
        tasks[t] = dict(stats=stats, differences=deltas, paper=paper[t],
            minus_paper={label: stats['test'][label]['r2']['mean'] - paper[t]['r2'] for label in labels})
    macro = {role: {label: float(np.mean([tasks[t]['stats'][role][label]['r2']['mean'] for t in TASKS])) for label in labels}
             for role in ('test', 'validation', 'test_minus_validation')}
    contrasts = [('GATE_REAL_neural', 'GATE_NULL_neural'), ('GATE_REAL_neural', 'UA'),
        ('GATE_REAL_fused', 'GATE_NULL_fused'), ('GATE_REAL_fused', 'UA_FIXED75'), ('GATE_REAL_fused', 'r10')]
    effects = {}
    for candidate, reference in contrasts:
        name = candidate + '-minus-' + reference
        task_delta = {t: tasks[t]['differences'][name]['r2'] for t in TASKS}
        abs_sum = sum(abs(v) for v in task_delta.values())
        positive_sum = sum(max(v, 0) for v in task_delta.values())
        effects[name] = dict(macro_delta=macro['test'][candidate] - macro['test'][reference],
            degraded_tasks=[t for t, d in task_delta.items() if d < 0], task_r2_deltas=task_delta,
            task_macro_contributions={t: d / 7 for t, d in task_delta.items()},
            absolute_gain_shares={t: abs(d) / abs_sum if abs_sum else 0. for t, d in task_delta.items()},
            positive_gain_shares={t: max(d, 0) / positive_sum if positive_sum else 0. for t, d in task_delta.items()},
            improved_folds=sum(r['differences'][name]['r2'] > 0 for r in rows), degraded_folds=sum(r['differences'][name]['r2'] < 0 for r in rows))
    review = OUT / 'final_review'
    review.mkdir(exist_ok=False)
    result = dict(runtime_revision='r7_g2_pt20k', pretraining=read(OUT / 'initialization/pretraining.json'), old_failure=str(FAILED_OUT / 'failure.json'), consumption=consumption_record(controller.budget), policy=POLICY, tasks=tasks, macro_r2=macro, effects=effects, folds=rows,
        aggregation='five folds, seed42+fold, std ddof=0; seven equally weighted task means',
        protocol='shared validation/TEST development exposed; exploratory numerical comparison; paper split/data equivalence unverified',
        independent_review='NOT_PERFORMED')
    write(review / 'seven_task_results.json', result)
    write(review / 'prediction_index.json', index)
    write(review / 'parameter_groups.json', {a: {t: {str(f): read(unit_path(a, t, f) / 'parameter_groups.json') for f in range(5)} for t in TASKS} for a in ARMS})
    with (review / 'fold_differences.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['task', 'fold', 'comparison', 'delta_r2', 'delta_rmse'])
        for r in rows:
            for name, delta in r['differences'].items():
                writer.writerow([r['task'], r['fold'], name, delta['r2'], delta['rmse']])
    with (review / 'fold_metrics.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['task', 'fold', 'policy', 'test_r2', 'test_rmse', 'validation_r2', 'validation_rmse', 'test_minus_validation_r2', 'test_minus_validation_rmse'])
        for r in rows:
            for label in labels:
                writer.writerow([r['task'], r['fold'], label, *[r[role][label][m] for role in ('test', 'validation', 'test_minus_validation') for m in ('r2', 'rmse')]])
    lines = ['G2（P2 ALIGN20k parent）微调与正式预测完成；待最终CPU测试及独立审查。', '',
        '两臂固定0.75神经＋0.25既有KRR，不选新权重。五折seed42+fold、std ddof=0；全部原精度计算后显示3位小数。RMSE为原单位，不跨任务平均。', '',
        '|任务／政策|正式 R² mean±std|原单位 RMSE mean±std|validation R² mean±std|TEST−validation R²|',
        '|---|---:|---:|---:|---:|']
    for t in TASKS:
        for label in labels:
            s = tasks[t]['stats']
            fmt = lambda role, m: f"{s[role][label][m]['mean']:.3f}±{s[role][label][m]['std']:.3f}"
            lines.append(f"|{t.upper() if t in ('eps','nc','xc') else t.capitalize()}／{label}|{fmt('test','r2')}|{fmt('test','rmse')}|{fmt('validation','r2')}|{s['test_minus_validation'][label]['r2']['mean']:+.3f}|")
    lines += ['', '|政策|正式 macro R²|validation macro R²|TEST−validation|', '|---|---:|---:|---:|']
    for label in labels:
        lines.append(f"|{label}|{macro['test'][label]:.3f}|{macro['validation'][label]:.3f}|{macro['test_minus_validation'][label]:+.3f}|")
    lines += ['', '|比较|macro R²差|退化任务|改善／退化fold数|', '|---|---:|---|---:|']
    for name, e in effects.items():
        lines.append(f"|{name}|{e['macro_delta']:+.3f}|{','.join(e['degraded_tasks']) or '无'}|{e['improved_folds']}／{e['degraded_folds']}|")
    lines += ['', '|任务|论文最佳模型|论文R²|REAL神经−论文|REAL融合−论文|NULL神经−论文|NULL融合−论文|UA_FIXED75−论文|', '|---|---|---:|---:|---:|---:|---:|---:|']
    for t in TASKS:
        p = paper[t]
        values = '|'.join(f"{tasks[t]['minus_paper'][label]:+.3f}" for label in ('GATE_REAL_neural','GATE_REAL_fused','GATE_NULL_neural','GATE_NULL_fused','UA_FIXED75'))
        lines.append(f"|{t}|{p['model']}|{p['r2']:.3f}|{values}|")
    for candidate, reference in contrasts:
        name = candidate + '-minus-' + reference
        lines += ['', f'逐任务差与收益集中度：{name}', '', '|任务|R²差|RMSE差（负值较好）|macro贡献|正收益占比|', '|---|---:|---:|---:|---:|']
        for t in TASKS:
            d = tasks[t]['differences'][name]
            lines.append(f"|{t}|{d['r2']:+.3f}|{d['rmse']:+.3f}|{effects[name]['task_macro_contributions'][t]:+.3f}|{effects[name]['positive_gain_shares'][t]:.3f}|")
    lines += ['', '完整逐fold差、所有候选与UA／r10／UA_FIXED75的差及validation→TEST差见fold_differences.csv、fold_metrics.csv和seven_task_results.json；路径见prediction_index.json。',
        '共享validation／TEST已参与开发，仅探索性数值比较；论文数据／split／评估及统计协议等价性未核实（Egc与Xc样本量有历史差异），不宣称公平超过论文或独立盲测。',
        '门值不作因果贡献；REAL−NULL检验同门架构下的FP信息，不能据此宣称GATE优于固定残差。r10比较还混杂parent及训练配方差异。',
        '预训练单独见p2_align20k_r7：新20,000步、每5,000步保存，固定20k parent；PT两卡使ALIGN每次前向候选池由336变168，不能归因于训练长度一项。FT worker／teacher／新KRR fit或forward／新构象／额外seed全部0。没有A_GLOBAL搜索、额外候选或独立审查。']
    (review / 'delivery.md').write_text('\n'.join(lines) + '\n')
    return result


def update_plan(status, detail):
    p = ROOT / 'CURRENT_PLAN.md'
    s = p.read_text()
    a = s.index('状态：')
    b = s.index('\n\n', a)
    s = s[:a] + f'状态：**{status}（P2＋G2）**。授权来源：用户要求仅GPU2/3、PT最多2路、FT最多4路总并发，启动正常后停止监控；新PT20,000步、每5,000步保存。新parent固定20k，70个完整FT；旧43次启动及19个旧parent已完成单元仅保留消费与现场。自检不代替独立审查。' + s[b:]
    s += '\n\n## 27. r7 G2 实际交付记录\n\n' + detail + '\n'
    p.write_text(s)


def final_artifacts(controller):
    review = OUT / 'final_review'
    review.mkdir(exist_ok=True)
    changes = []
    for name in EDITED:
        previous = (OUT / 'source_before' / name).read_text().splitlines(True) if (OUT / 'source_before' / name).exists() else []
        current = (ROOT / name).read_text().splitlines(True)
        changes.extend(difflib.unified_diff(previous, current, fromfile='before/' + name, tofile='after/' + name))
    (review / 'g1_only_changes.diff').write_text(''.join(changes))
    checked = subprocess.run(['git', 'diff', '--check', '--', 'CURRENT_PLAN.md'], cwd=ROOT, capture_output=True, text=True)
    write(review / 'diff_format_check.json', dict(exit=checked.returncode, stdout=checked.stdout, stderr=checked.stderr))
    exits = []
    for folder in LOG.iterdir():
        if folder.is_dir() and (folder / 'execution.json').exists():
            e = read(folder / 'execution.json')
            exits.append(dict(log=str(folder), execution=e, exit_txt=int((folder / 'exit.txt').read_text()) if (folder / 'exit.txt').exists() else None))
    write(review / 'exit_index.json', dict(records=exits, controller_exit=str(LOG / 'controller/exit.txt'),
        note='controller own true exit written by outer wrapper after this function; not invented'))
    for name in EDITED:
        dest = OUT / 'source_final' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)


def execute():
    require_tmux()
    if OUT.exists() or LOG.exists():
        raise FileExistsError('G1 root already exists; refusing replay or overwrite')
    OUT.mkdir(parents=True, exist_ok=False)
    LOG.mkdir(parents=True, exist_ok=False)
    folder = LOG / 'controller'
    folder.mkdir()
    command = dict(argv=[PYTHON, 'scripts/run_ua2d_fp_gate.py', '--execute'], cwd=str(ROOT),
        environment=dict(ENV, CUDA_VISIBLE_DEVICES=''), pid=os.getpid(), window='ua2d_g2_r7_controller', artifacts=str(OUT))
    write(folder / 'command.json', command)
    execution = dict(command=command, start=now(), end=None, exit=None, pid=os.getpid())
    write(folder / 'execution.json', execution)
    controller = Controller()
    rc = 1
    with (folder / 'stdout.log').open('x') as stdout, (folder / 'stderr.log').open('x') as stderr, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            prior_files = {'CURRENT_PLAN.md': 'CURRENT_PLAN.md', 'src/training/ua2d_fp_gate.py': 'training.py',
                'scripts/run_ua2d_fp_gate.py': 'controller.py', 'scripts/run_ua2d_fp_gate_unit.py': 'unit.py',
                'tests/test_ua2d_fp_gate.py': 'test.py'}
            for name, snapshot in prior_files.items():
                before = OUT / 'source_before' / name
                before.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path('/tmp/ua2d_r7_before') / snapshot, before)
            before = OUT / 'source_before/src/modules/ua2d_fp_gate.py'
            before.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / 'src/modules/ua2d_fp_gate.py', before)
            for name in EDITED:
                dest = OUT / 'source_used' / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, dest)
            for name in ('scripts/pretrain_glt_dual.py', 'scripts/evaluate_glt_pt_endpoint.py', 'src/training/ua2d_pt20k.py', 'scripts/run_ua2d_pt20k.py'):
                before = OUT / 'source_before' / name
                before.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path('/tmp/ua2d_r7_before') / Path(name).name, before)
            source_units = initialize()
            controller.fp = FPWriter(OUT / 'fingerprints.sqlite', source=FAILED_OUT / 'fingerprints.sqlite')
            controller.phase = 'finetune'
            controller.schedule()
            controller.phase = 'freeze'
            controller.freeze()
            controller.phase = 'formal'
            controller.schedule(formal=True)
            controller.phase = 'array_report'
            result = report(controller)
            controller.ledger()
            b = controller.budget
            if (b['ft_completed'], b['epochs'], b['attempted'], b['completed'], b['formal_completed'], b['gate_arrays']) != (70, 4900, 112000, 112000, 70, 140):
                raise ValueError('science incomplete versus fixed full budget')
            controller.phase = 'final_cpu'
            b['final_cpu_starts'] += 1
            controller.ledger()
            test = launch_job('final_cpu', ['timeout', '300s', PYTHON, '-m', 'pytest', '-q', 'tests/test_ua2d_fp_gate.py'])
            while not (test / 'exit.txt').exists():
                time.sleep(2)
            test_exit = int((test / 'exit.txt').read_text())
            status = '待审查' if test_exit == 0 else '需返修'
            write(OUT / 'completion.json', dict(status=status, science='COMPLETE', start=execution['start'], end=now(),
                test_exit=test_exit, consumption=consumption_record(b), independent_review='NOT_PERFORMED'))
            detail = (OUT / 'final_review/delivery.md').read_text()
            detail += '\n实际消费：' + json.dumps(consumption_record(b), ensure_ascii=False) + '。\n'
            detail += f'唯一最终CPU测试 exit={test_exit}；日志{LOG}/final_cpu/。tmux Uni-Poly:ua2d_g2_r7_*；每卡峰值{controller.slots.peak}，总峰值{controller.slots.total_peak}。\n'
            detail += f'冻结{OUT}/selection_before_test.json；汇总、逐fold、来源索引、实际diff与退出索引在final_review/；controller最终真实exit见{LOG}/controller/exit.txt。所有旧产物保留，原A未接续；无删除、哈希校验、commit/push或追加候选。\n'
            update_plan(status, detail)
            rc = 0 if test_exit == 0 else 1
        except BaseException as error:
            traceback.print_exc()
            failure_traceback = traceback.format_exc()
            controller.stop_own()
            write(OUT / 'failure.json', dict(phase=controller.phase, error=repr(error), traceback=traceback.format_exc(),
                time=now(), consumption=consumption_record(controller.budget), active=controller.slots.active,
                pre_stop_snapshot=str(OUT / 'state_before_stop.json'), last_observation=controller.last_observation))
            write(OUT / 'completion.json', dict(status='需返修', science='STOPPED', phase=controller.phase, error=repr(error),
                test='NOT_RUN' if controller.budget['final_cpu_starts'] == 0 else 'SEE_LOG',
                consumption=consumption_record(controller.budget), independent_review='NOT_PERFORMED'))
            update_plan('需返修', f'执行在{controller.phase}停止：{error!r}。日志{LOG}/controller/，failure.json与消费账本在{OUT}/。没有自动重试或追加预算；未完成任务不可比较，不能报告性能或论文差。')
        finally:
            try:
                if controller.fp:
                    controller.fp.close()
                final_artifacts(controller)
            finally:
                execution.update(end=now(), exit=rc)
                write(folder / 'execution.json', execution, True)
                (folder / 'exit.txt').write_text(str(rc) + '\n')
    return rc


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--launch', action='store_true')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--job')
    args = p.parse_args()
    if args.launch:
        if OUT.exists() or LOG.exists():
            raise FileExistsError('G1 roots already exist')
        subprocess.run(['tmux', 'has-session', '-t', 'Uni-Poly'], check=True)
        line = shlex.join(['env', *[k + '=' + v for k, v in dict(ENV, CUDA_VISIBLE_DEVICES='').items()], PYTHON, str(Path(__file__).resolve()), '--execute'])
        subprocess.run(['tmux', 'new-window', '-d', '-t', 'Uni-Poly', '-n', 'ua2d_g2_r7_controller',
            'tmux set-option -w -t "$TMUX_PANE" remain-on-exit on\ncd ' + shlex.quote(str(ROOT)) + '\nexec ' + line], check=True)
        print('G1 launched in Uni-Poly:ua2d_g2_r7_controller; real status/logs:', LOG)
    elif args.execute:
        sys.exit(execute())
    elif args.job:
        sys.exit(job_wrapper(args.job))
    else:
        p.error('explicit --launch, --execute or --job required')
