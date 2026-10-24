"""UA2D-V1-ATTNSCALE: one finite arm with a 64**-0.5 QK attention scale.

The module is inert unless UA2D_PRETRAIN_FP_CAMPAIGN=attnscale_r1, which selects
the new result/log roots and the single-arm budget in src.training.ua2d_pretrain_fp.
It reuses the fast_r4 initialization, prepared cache, splits and head initials
read-only; only the attention scale differs from V1.
"""
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = 'attnscale_r1'
if os.environ.get('UA2D_PRETRAIN_FP_CAMPAIGN') not in (None, CAMPAIGN):
    raise RuntimeError('attnscale controller imported under a different campaign context')
os.environ['UA2D_PRETRAIN_FP_CAMPAIGN'] = CAMPAIGN
sys.path.insert(0, str(ROOT))

import torch

from src.modules.ua2d_galformer_pretrain import PretrainConfig, attention_record, package_attention
from src.training import ua2d_pretrain_fp as fp

ARM = 'V1_ATTN64'
SOURCE_INIT = ROOT / 'results/ua2d_pretrain_fp_20261019/fast_r4/initialization'
CACHE = ROOT / 'results/ua2d_pretrain_fp_20261019/fast_r4/prepared_cache'
OLD_RESULTS = ROOT / 'results/ua2d_pretrain_fp_20261019/fast_r4/final_review/results.json'
R10_RESULTS = ROOT / 'results/fce_tune_20261016/global_blend/final_review/seven_task_results.json'
REFERENCE = {
    'V1': dict(validation=0.8169937875891848, test=0.7872037608582197),
    'B_FP': dict(validation=0.8184636300392601, test=0.7912368083520195),
    'r10': dict(test=0.7961937366142565),
}
# Read from SOTA_RESULT.md on 2026-10-20 (best per task; do not modify that file).
SOTA = {
    'eea': (0.934, 'Periodic-TDL'),
    'egb': (0.923, 'Periodic-TDL (+Chem, +RC)'),
    'egc': (0.911, 'Periodic-TDL (+Chem, +RC)'),
    'ei': (0.839, 'Periodic-TDL'),
    'eps': (0.787, 'Periodic-TDL'),
    'nc': (0.856, 'Periodic-TDL'),
    'xc': (0.423, 'Periodic-TDL'),
}
LABEL = dict(eea='Eea', egb='Egb', egc='Egc', ei='Ei', eps='EPS', nc='Nc', xc='Xc')
THRESHOLD = 0.010


def f3(value):
    return format(float(value), '.3f')


def attention64():
    return attention_record(PretrainConfig(attention_scale_denominator=64))


class AttnScaleController:
    def __init__(self):
        if fp.OUT.exists():
            raise FileExistsError('result root already exists; verify ownership instead of reusing it: ' + str(fp.OUT))
        fp.OUT.mkdir(parents=True)
        self.lock = (fp.OUT / 'controller.lock').open('x')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.jobs = []
        info = subprocess.check_output(
            ['nvidia-smi', '-i', '2,3', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
        self.gpu_uuids = {}
        for line in info.strip().splitlines():
            index, uuid = [s.strip() for s in line.split(',')]
            self.gpu_uuids[uuid] = index
        if set(self.gpu_uuids.values()) != {'2', '3'}:
            raise RuntimeError('GPU mapping mismatch')
        changes = subprocess.check_output(['git', 'status', '--short', '--untracked-files=normal'], cwd=ROOT, text=True)
        self.authorization_path = fp.OUT / 'authorization.json'
        fp.freeze(self.authorization_path, dict(
            plan=fp.PLAN, revision=fp.REVISION, time=fp.now(),
            user_request='请实现并执行 V1 attention 缩放系数的单因素重训实验（UA2D-V1-ATTNSCALE-20261020-01 / r1）',
            scientific_question='将每头 64 维 QK 点积缩放由 512**-0.5 改为 64**-0.5 能否改善 V1 预训练迁移',
            controlled_change='only the QK dot-product scalar in SourceAttention512; bias, temperature, contrast weight and data are unchanged',
            reference=dict(old_V1=REFERENCE['V1'], V1_FP=REFERENCE['B_FP'], r10=REFERENCE['r10'],
                note='mixed recipes; numeric comparison only, never attributed to the scale alone'),
            budgets=fp.LIMITS, only_gpus=[2, 3], automatic_retry=False,
            seeds=dict(pretrain=42, finetune='42+fold'), arms=[ARM],
            initialization=dict(common=str(SOURCE_INIT / 'common_step0.pt'),
                task_head=str(SOURCE_INIT / 'task_heads/V1.pt'), plan='UA2D-PRETRAIN-FP-20261019-01', step=0),
            prepared_cache=dict(path=str(CACHE), mode='read-only reuse; rebuild=False'),
            outer_test='RUN exactly once after the 35 FT units are frozen'))
        self.manifest_path = fp.OUT / 'execution_manifest.json'
        fp.freeze(self.manifest_path, dict(
            plan=fp.PLAN, revision=fp.REVISION, time=fp.now(), cwd=str(ROOT),
            authorization='user request 2026-10-20: single-factor attention-scale retrain; only GPU2,3; no retries, no push',
            branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
            commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            dirty_status=changes, gpu_mapping=self.gpu_uuids, limits=fp.LIMITS,
            source=str(fp.SOURCE), index=str(fp.INDEX), downstream=str(fp.DOWN), split=str(fp.SPLIT),
            head_root=str(fp.HEADS), plan_file=str(ROOT / 'CURRENT_PLAN.md'), outer_test='NOT_RUN',
            initialization=dict(common=str(SOURCE_INIT / 'common_step0.pt'),
                task_heads=str(SOURCE_INIT / 'task_heads/V1.pt'), plan='UA2D-PRETRAIN-FP-20261019-01'),
            prepared_cache=str(CACHE), attention=attention64(),
            budget_arithmetic=dict(pt_starts=1, pt_updates=5000, pt_presentations=5040000,
                ft_units=35, ft_epochs=2450, ft_updates=56000, validations=2450,
                formal_units=35, test_graphs=5875, cpu_suites=1),
            environment=dict(python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda),
            historical_jobs='old fast_r4/SOTA jobs not resumed; no remote operations, file digests, TODO reads, smoke or benchmark'))
        self.refresh()

    def refresh(self):
        counts = {key: 0 for key in fp.LIMITS}
        for job in self.jobs:
            kind = job['kind']
            output = Path(job['output'])
            if kind == 'pt':
                counts['pt_starts'] += 1
                records = []
                for rank in range(2):
                    path = output / f'rank{rank}_consumption.json'
                    records.append(fp.read(path) if path.exists() else {})
                counts['pt_updates'] += max((r.get('attempted_updates', 0) for r in records), default=0)
                counts['pt_presentations'] += sum(r.get('sample_presentations', 0) for r in records)
            elif kind in ('ft', 'formal'):
                c = fp.read(output / 'consumption.json') if (output / 'consumption.json').exists() else {}
                if kind == 'ft':
                    counts['ft_starts'] += 1
                    counts['ft_updates'] += c.get('attempted_updates', 0)
                    counts['ft_epochs'] += c.get('epochs_started', 0)
                    counts['validation_units'] += c.get('validation_started', 0)
                else:
                    counts['test_starts'] += 1
                    counts['test_graphs'] += c.get('test_graphs', 0)
            elif kind == 'cpu':
                counts['cpu_suites'] += 1
        fp.check_limits(counts)
        fp.write(fp.OUT / 'budget.json', dict(
            plan=fp.PLAN, revision=fp.REVISION, time=fp.now(), used=counts, limits=fp.LIMITS,
            authorization=str(self.authorization_path),
            accounting='starts reserved before dispatch; attempted updates and begun epochs/validation include partials; PT updates count synchronized optimizer windows, samples sum both ranks',
            jobs=self.jobs))
        return counts

    def occupied(self):
        info = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader,nounits'], text=True)
        occupied = {}
        for line in info.strip().splitlines():
            parts = [s.strip() for s in line.split(',')]
            if parts and parts[0] in self.gpu_uuids:
                occupied.setdefault(self.gpu_uuids[parts[0]], []).append(parts)
        return occupied

    def wait_free(self, gpus):
        while True:
            busy = self.occupied()
            if not any(str(g) in busy for g in gpus):
                return
            fp.write(fp.OUT / 'status.json', dict(status='执行中', phase='WAIT_GPU', gpus=gpus, occupied=busy, time=fp.now()))
            self.refresh()
            import time
            time.sleep(30)

    def launch(self, name, kind, argv, output, gpu=''):
        counts = self.refresh()
        key = {'pt': 'pt_starts', 'ft': 'ft_starts', 'formal': 'test_starts', 'cpu': 'cpu_suites'}.get(kind)
        if key and counts[key] >= fp.LIMITS[key]:
            raise RuntimeError('start budget exhausted: ' + key)
        if kind == 'pt' and gpu != '2,3':
            raise ValueError('PT GPU violation')
        if kind in ('ft', 'formal') and gpu not in ('2', '3'):
            raise ValueError('FT/formal GPU violation')
        import shlex
        log = fp.LOG / name
        log.mkdir(parents=True, exist_ok=False)
        shell = '#!/bin/bash\ncd ' + shlex.quote(str(ROOT)) + ' || exit 90\n'
        shell += 'export CUDA_VISIBLE_DEVICES=' + shlex.quote(gpu) + ' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1\n'
        shell += 'export CUBLAS_WORKSPACE_CONFIG=:4096:8\n'
        shell += 'export UA2D_PRETRAIN_FP_CAMPAIGN=' + CAMPAIGN + '\n'
        command = shlex.join([str(a) for a in argv])
        (log / 'command.txt').write_text(command + '\n')
        (log / 'cwd.txt').write_text(str(ROOT) + '\n')
        shell += 'date -Iseconds > ' + shlex.quote(str(log / 'start.txt')) + '\n'
        shell += command + ' > ' + shlex.quote(str(log / 'stdout.log')) + ' 2> ' + shlex.quote(str(log / 'stderr.log')) + '\n'
        shell += 'ua2d_exit=$?\nprintf \'%s\\n\' "$ua2d_exit" > ' + shlex.quote(str(log / 'exit.txt')) + '\n'
        shell += 'date -Iseconds > ' + shlex.quote(str(log / 'end.txt')) + '\nexit "$ua2d_exit"\n'
        (log / 'run.sh').write_text(shell)
        job = dict(name=name, kind=kind, argv=[str(a) for a in argv], cwd=str(ROOT), output=str(output),
                   log=str(log), gpu=gpu, start=fp.now(), exit=None)
        self.jobs.append(job)
        self.refresh()
        try:
            ident = subprocess.check_output(
                ['tmux', 'new-window', '-d', '-P', '-F', '#{window_id}', '-t', 'Uni-Poly', '-n', name,
                 'bash ' + shlex.quote(str(log / 'run.sh'))], text=True).strip()
            job['tmux_window'] = ident
            job['tmux_name'] = name
            self.refresh()
        except BaseException:
            job['launch_error'] = traceback.format_exc()
            self.refresh()
            raise
        return job

    def poll(self, job):
        exit_path = Path(job['log']) / 'exit.txt'
        if exit_path.exists():
            job['exit'] = int(exit_path.read_text())
            job['end'] = fp.now()
            return True
        names = subprocess.check_output(['tmux', 'list-windows', '-t', 'Uni-Poly', '-F', '#{window_id}'], text=True).splitlines()
        if job.get('tmux_window') not in names:
            raise RuntimeError('window vanished without exit: ' + job['name'])
        return False

    def wait(self, job):
        import time
        while not self.poll(job):
            self.refresh()
            time.sleep(15)
        self.refresh()
        if job['exit'] != 0:
            raise RuntimeError(job['name'] + ' failed with exit ' + str(job['exit']) + '; see ' + job['log'])

    def prepare_initialization(self):
        folder = fp.OUT / 'initialization'
        if folder.exists():
            raise FileExistsError('initialization already exists: ' + str(folder))
        (folder / 'task_heads').mkdir(parents=True)
        for relative in ('common_step0.pt', 'task_heads/V1.pt'):
            source = SOURCE_INIT / relative
            if not source.is_file():
                raise FileNotFoundError('missing untrained initial state: ' + str(source))
            shutil.copyfile(source, folder / relative)
        fp.freeze(folder / 'source.json', dict(
            plan=fp.PLAN, revision=fp.REVISION, time=fp.now(),
            source=str(SOURCE_INIT), source_resolved=str(SOURCE_INIT.resolve()),
            copied=['common_step0.pt', 'task_heads/V1.pt'],
            upstream_plan='UA2D-PRETRAIN-FP-20261019-01', upstream_step=0, upstream_seed=42,
            note='unmodified copy of the untrained step0 package and V1 task-head initial state; no step5000 continuation'))
        common = torch.load(folder / 'common_step0.pt', map_location='cpu', weights_only=False)
        heads = torch.load(folder / 'task_heads/V1.pt', map_location='cpu', weights_only=False)
        if common.get('schema') != 'ua2d_galformer_objectives_v3' or common.get('step') != 0 or common.get('seed') != 42:
            raise ValueError('copied common step0 failed verification')
        if common.get('plan') not in fp.INITIALIZATION_PLANS:
            raise ValueError('copied common step0 plan is not an accepted ancestor')
        if heads.get('step') != 0 or heads.get('strategy') != 'mask_align':
            raise ValueError('copied V1 task head failed verification')
        fp.freeze(folder / 'verification.json', dict(
            common=dict(step=common['step'], seed=common['seed'], plan=common['plan'], schema=common['schema'],
                        tensors=len(common['state_dict'])),
            task_head=dict(step=heads['step'], strategy=heads['strategy'], schema=heads['schema'],
                           tensors=len(heads['state_dict'])),
            time=fp.now()))

    def freeze_config(self):
        value = fp.objective_config(ARM)
        path = fp.OUT / 'configs' / f'{ARM}.json'
        if path.exists():
            if fp.read(path) != value:
                raise ValueError('existing objective config differs: ' + str(path))
        else:
            fp.freeze(path, value)
        old = fp.read(ROOT / 'results/ua2d_pretrain_fp_20261019/fast_r4/configs/V1.json')
        expected = dict(old, attention_scale_denominator=64)
        if value != expected:
            raise ValueError('new objective config differs from old V1 beyond the attention-scale field')
        fp.freeze(fp.OUT / 'configs/derivation.json', dict(
            plan=fp.PLAN, revision=fp.REVISION, time=fp.now(),
            ancestor='results/ua2d_pretrain_fp_20261019/fast_r4/configs/V1.json',
            only_change='attention_scale_denominator: absent(512) -> 64', ancestor_config=old, candidate_config=value))

    def pt(self):
        self.wait_free([2, 3])
        value = fp.objective_config(ARM)
        path = fp.OUT / 'configs' / f'{ARM}.json'
        if not path.exists() or fp.read(path) != value:
            raise ValueError('PT objective config missing or changed')
        output = fp.parent_path(ARM).parent
        if output.exists():
            raise FileExistsError('PT attempt output already exists: ' + str(output))
        counts = self.refresh()
        if counts['pt_updates'] + 5000 > fp.LIMITS['pt_updates'] or counts['pt_presentations'] + 5040000 > fp.LIMITS['pt_presentations']:
            raise RuntimeError('insufficient budget for a complete PT arm')
        argv = ['timeout', '--signal=TERM', '--kill-after=30s', '18000s',
                sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                str(ROOT / 'scripts/pretrain_ua2d_fast.py'), '--config', str(path), '--output', str(output),
                '--initialization', str(fp.OUT / 'initialization/common_step0.pt'),
                '--prepared-cache', str(CACHE)]
        self.wait(self.launch('attnscale_pt_' + ARM, 'pt', argv, output, '2,3'))
        for rank in range(2):
            record = fp.read(output / f'rank{rank}_consumption.json')
            if record['status'] != 'COMPLETE' or record['completed_updates'] != 5000 or record['sample_presentations'] != 2520000:
                raise RuntimeError('incomplete PT consumption')
        package = torch.load(output / 'deploy.pt', map_location='cpu', weights_only=False)
        attention = package_attention(package)
        if package.get('step') != 5000 or attention['scale_denominator'] != 64 or attention['scale'] != 0.125:
            raise RuntimeError('PT deployment package attention record invalid')
        if package.get('config', {}).get('attention_scale_denominator') != 64 or package.get('attention') != attention64():
            raise RuntimeError('PT deployment package config/attention mismatch')
        fp.freeze(output / 'verification.json', dict(
            plan=fp.PLAN, revision=fp.REVISION, time=fp.now(), step=package['step'],
            attention=attention, execution_revision=package.get('execution', {}).get('revision'),
            execution_plan=package.get('execution', {}).get('plan'), config=package['config'],
            consumption=[fp.read(output / f'rank{rank}_consumption.json') for rank in range(2)]))

    def block(self, formal=False):
        import time
        pending = [(task, fold) for task in fp.TASKS for fold in range(5)]
        active = {}
        failed = []
        self.wait_free([2, 3])
        while pending or active:
            for slot, job in list(active.items()):
                if self.poll(job):
                    if job['exit'] != 0:
                        failed.append(job)
                    del active[slot]
            if not failed:
                for gpu in ('2', '3'):
                    if any(key[0] == gpu for key in active) or self.occupied().get(gpu):
                        continue
                    for slot in (0, 1):
                        if not pending:
                            break
                        task, fold = pending.pop(0)
                        mode = 'formal' if formal else 'ft'
                        name = f'attnscale_{mode}_{ARM}_{task}_{fold}'
                        argv = [sys.executable, str(ROOT / 'scripts/run_ua2d_pretrain_fp.py'), mode,
                                '--arm', ARM, '--task', task, '--fold', str(fold)]
                        active[(gpu, slot)] = self.launch(name, mode, argv, fp.unit_path(ARM, task, fold, formal), gpu)
            self.refresh()
            if failed and not active:
                raise RuntimeError('FT/formal failed; no further dispatch: ' + repr(failed))
            if active or pending:
                time.sleep(15)
        fp.aggregate(ARM, formal)

    def freeze_units(self):
        validation = fp.aggregate(ARM)
        attention = attention64()
        units = {}
        graphs = 0
        for task in fp.TASKS:
            manifest = fp.read(fp.SPLIT / f'{task}.json')
            for fold in range(5):
                folder = fp.unit_path(ARM, task, fold)
                metrics = fp.read(folder / 'metrics.json')
                consumption = fp.read(folder / 'consumption.json')
                if metrics['status'] != 'COMPLETE' or consumption['status'] != 'COMPLETE':
                    raise ValueError('incomplete FT unit: ' + str(folder))
                if consumption['completed_epochs'] != 70 or consumption['completed_updates'] != consumption['attempted_updates']:
                    raise ValueError('FT unit did not run the fixed 70 epochs: ' + str(folder))
                if metrics['attention'] != attention:
                    raise ValueError('FT unit attention record mismatch: ' + str(folder))
                scaler = fp.read(folder / 'scaler.json')
                if metrics['config'].get('objectives', {}).get('attention_scale_denominator') != 64:
                    raise ValueError('FT unit config attention mismatch: ' + str(folder))
                test_indices = sorted(manifest['folds'][fold]['test_indices'])
                graphs += len(test_indices)
                units[f'{ARM}/{task}/{fold}'] = dict(
                    task=task, fold=fold, seed=42 + fold, config=metrics['config'],
                    best=str(folder / 'best.pt'), best_epoch=metrics['best_epoch'], scaler=scaler,
                    attention=metrics['attention'], validation_r2=metrics['r2'], validation_metrics=metrics,
                    test_indices=test_indices)
        if len(units) != 35 or graphs != 5875:
            raise ValueError('frozen unit matrix does not match the authorized 35 units / 5875 graphs')
        fp.freeze(fp.OUT / 'selection/final_frozen.json', dict(
            plan=fp.PLAN, revision=fp.REVISION, frozen=True, time=fp.now(), arm=ARM,
            attention=attention, validation_macro=validation['macro'], validation=validation,
            units=units, test_graphs=graphs,
            rule='frozen once after all 35 FT units; formal TEST evaluates exactly these best checkpoints and scalers; no reselection',
            budget=fp.read(fp.OUT / 'budget.json')))

    def reference(self):
        old = fp.read(OLD_RESULTS)
        r10 = fp.read(R10_RESULTS)
        values = dict(
            V1=dict(validation=float(old['validation']['V1']['macro']), test=float(old['test']['V1']['macro'])),
            B_FP=dict(validation=float(old['validation']['B_FP']['macro']), test=float(old['test']['B_FP']['macro'])),
            r10=dict(test=float(r10['macro']['global']['r2'])))
        for arm, row in REFERENCE.items():
            for key, value in row.items():
                if abs(values[arm][key] - value) > 1e-12:
                    raise ValueError('recorded reference changed: ' + arm + '/' + key)
        values['paths'] = dict(old_V1=str(OLD_RESULTS), r10=str(R10_RESULTS),
            V1_FP='same results.json, validation/test B_FP')
        return values

    def report(self):
        validation = fp.aggregate(ARM)
        test = fp.aggregate(ARM, True)
        reference = self.reference()
        attention = attention64()
        rows = {}
        for task in fp.TASKS:
            units = [fp.read(fp.unit_path(ARM, task, fold) / 'metrics.json') for fold in range(5)]
            rows[task] = dict(
                validation=validation['tasks'][task], test=test['tasks'][task],
                paper_r2=SOTA[task][0], paper_model=SOTA[task][1],
                delta_test=test['tasks'][task]['mean'] - SOTA[task][0],
                protocol=dict(folds=5, seeds=[42 + fold for fold in range(5)],
                    best_epochs=[unit['best_epoch'] for unit in units],
                    selection='full-precision validation R2 per epoch; earlier epoch wins ties',
                    aggregation='task mean over 5 folds, macro equal weight over 7 tasks, std ddof=1'),
                paths=dict(pretrain=str(fp.parent_path(ARM)),
                    finetune=str(fp.OUT / 'finetune' / ARM / task),
                    formal=str(fp.OUT / 'formal' / ARM / task)))
        delta_test = dict(vs_V1=test['macro'] - reference['V1']['test'],
            vs_B_FP=test['macro'] - reference['B_FP']['test'],
            vs_r10=test['macro'] - reference['r10']['test'])
        delta_validation = dict(vs_V1=validation['macro'] - reference['V1']['validation'],
            vs_B_FP=validation['macro'] - reference['B_FP']['validation'])
        value = dict(
            plan=fp.PLAN, revision=fp.REVISION, arm=ARM, status='待审查', time=fp.now(),
            independent_review=False, attention=attention,
            scientific_question='将每头 64 维 QK 点积缩放由 512**-0.5 改为 64**-0.5 能否改善 V1 预训练迁移',
            controlled_change='only the QK dot-product scalar; structural/angle bias is added after scaling and is not scaled',
            validation=validation, test=test, reference=reference,
            delta_validation=delta_validation, delta_test=delta_test,
            threshold=dict(base='V1', metric='TEST macro R2', value=THRESHOLD,
                delta=delta_test['vs_V1'], met=bool(delta_test['vs_V1'] >= THRESHOLD),
                note='unrounded delta; the comparison to V1+FP and r10 includes recipe differences and is numeric only'),
            seven_task=rows,
            budget=fp.read(fp.OUT / 'budget.json'),
            cpu_suite=fp.read(fp.OUT / 'final_review/cpu_suite.json'),
            frozen_units=str(fp.OUT / 'selection/final_frozen.json'),
            evidence_limits=['DSH self-check is not an independent review',
                'shared TEST folds participated in development, so TEST is not an independent blind test',
                'deltas vs V1/V1+FP/r10 include recipe differences and cannot isolate the scale factor',
                'no push/pull, no file digests, no SOTA_RESULT.md edit'])
        fp.write(fp.OUT / 'final_review/results.json', value)
        lines = ['# UA2D-V1-ATTNSCALE 结果摘要', '',
            f'- 计划：{fp.PLAN} / {fp.REVISION}', f'- 候选：{ARM}（attention scale = 64**-0.5 = 0.125，仅缩放 QK 点积）',
            f'- 状态：待审查（DSH 自检，不构成独立审查）', '',
            '## 预算与实际消耗', '',
            f'- PT：1 次启动，5000 updates，5,040,000 样本呈现（2 卡各 2,520,000）',
            f'- FT：35 单元 × 70 epochs = 2450 epochs，56000 updates，2450 次训练内 validation',
            f'- 正式 TEST：35 单元，5875 图；CPU suite：1 次（≤300 秒）', '',
            '## 宏 R² 与差值（三位小数，差值由未舍入值计算）', '',
            '| 口径 | 本候选 | 旧 V1 | V1＋FP | r10 | 差 vs V1 | 差 vs V1＋FP | 差 vs r10 |',
            '| --- | --- | --- | --- | --- | --- | --- | --- |',
            f"| validation | {f3(validation['macro'])} | {f3(reference['V1']['validation'])} | {f3(reference['B_FP']['validation'])} | — | {f3(delta_validation['vs_V1'])} | {f3(delta_validation['vs_B_FP'])} | — |",
            f"| TEST | {f3(test['macro'])} | {f3(reference['V1']['test'])} | {f3(reference['B_FP']['test'])} | {f3(reference['r10']['test'])} | {f3(delta_test['vs_V1'])} | {f3(delta_test['vs_B_FP'])} | {f3(delta_test['vs_r10'])} |",
            '', f"TEST 宏 R² 相对旧 V1 的未舍入增量为 {delta_test['vs_V1']!r}，判定阈值 {THRESHOLD}，达到：{value['threshold']['met']}。", '',
            '## 七任务（TEST，五折均值，ddof=1）', '',
            '| 任务 | 本项目 R² | 标准差 | 论文最佳 | 模型 | 差（本项目−论文） |',
            '| --- | --- | --- | --- | --- | --- |']
        for task in fp.TASKS:
            row = rows[task]
            lines.append(f"| {LABEL[task]} | {f3(row['test']['mean'])} | {f3(row['test']['std'])} | "
                         f"{f3(row['paper_r2'])} | {row['paper_model']} | {f3(row['delta_test'])} |")
        lines += ['', '## 证据与限制', '',
            f"- 结果：`{fp.OUT.relative_to(fp.ROOT)}/final_review/results.json`",
            f"- 冻结：`{fp.OUT.relative_to(fp.ROOT)}/selection/final_frozen.json`",
            f"- 日志：`{fp.LOG.relative_to(fp.ROOT)}`",
            '- 与旧 V1／V1＋FP／r10 的差值包含配方差异，不能归因于缩放这一因素；相对旧 V1 的提升不等于超过当前项目基线。',
            '- 共享 TEST fold 参与过开发，不构成独立盲测；协议未证明一致，只作数值对照。']
        (fp.OUT / 'final_review/summary.md').write_text('\n'.join(lines) + '\n')
        sota = ['# 七任务 SOTA 数值对照', '',
            '| 任务 | 本项目 R² | 论文最佳 R² | 模型 | 差值（本项目−论文） | 口径 | 产物 |',
            '| --- | --- | --- | --- | --- | --- | --- |']
        for task in fp.TASKS:
            row = rows[task]
            sota.append(f"| {LABEL[task]} | {f3(row['test']['mean'])} | {f3(row['paper_r2'])} | {row['paper_model']} | "
                        f"{f3(row['delta_test'])} | 5 folds, seed 42–46, TEST | `{fp.OUT.relative_to(fp.ROOT)}/formal/{ARM}/{task}` |")
        sota += ['', '协议未证明与论文一致，只报告数值对照；不声称公平比较或超过论文结果。']
        (fp.OUT / 'final_review/sota_comparison.md').write_text('\n'.join(sota) + '\n')
        return value

    def register(self, status, evidence):
        process = subprocess.check_output(['ps', '-eo', 'state,args'], text=True)
        for line in process.splitlines():
            if any(x in line for x in ('python scripts/ua2d_sota_dispatch.py', 'python scripts/ua2d_sota_continue.py',
                                       'python scripts/run_ua2d_sota_campaign.py')) and line.strip()[0] != 'T':
                fp.write(fp.OUT / 'current_plan_registration.json',
                         dict(status='DEFERRED', reason='old writer may be active', time=fp.now()))
                return
        path = ROOT / 'CURRENT_PLAN.md'
        before = path.stat()
        with (fp.OUT / 'current_plan.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path.read_text()
            after = path.stat()
            if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                fp.write(fp.OUT / 'current_plan_registration.json',
                         dict(status='DEFERRED', reason='concurrent edit observed', time=fp.now()))
                return
            with path.open('a') as handle:
                handle.write(self.note(status, evidence))
            fp.write(fp.OUT / 'current_plan_registration.json',
                     dict(status='APPENDED', time=fp.now(), recorded_status=status))

    def note(self, status, evidence):
        line = '尚未产生正式结果'
        results_path = fp.OUT / 'final_review/results.json'
        if results_path.exists():
            value = fp.read(results_path)
            if 'validation' in value and 'test' in value:
                line = (f"validation 宏 R² {f3(value['validation']['macro'])}；TEST 宏 R² {f3(value['test']['macro'])}；"
                        f"TEST 相对旧 V1 差 {f3(value['delta_test']['vs_V1'])}、相对 V1＋FP {f3(value['delta_test']['vs_B_FP'])}、"
                        f"相对 r10 {f3(value['delta_test']['vs_r10'])}")
        parts = [f'\n\n## {fp.PLAN} / {fp.REVISION} 执行登记（{fp.now()}）\n',
            f'- 状态：**{status}**；授权来源：用户当前指令（计划 {fp.PLAN} / {fp.REVISION}）；执行自检不构成独立审查，状态“未独立核实”。',
            '- 科学问题：将 V1 每头 64 维 QK 点积缩放由 512**-0.5 改为 64**-0.5 能否改善预训练迁移；reference 为同批次 V1，受控变化仅该缩放。',
            '- 固定条件：strategy=mask_align、fusion=none、fp_enabled=false、数据/缓存/split/初态/优化器与旧 V1 相同；InfoNCE 温度 0.100、contrast_weight 0.100 保留。',
            f'- 数值对照（协议不完全一致，只作数值比较）：旧 V1 validation 0.817 / TEST 0.787；V1＋FP 0.818 / 0.791；r10 TEST 0.796。',
            f'- 预算：PT 1×5000 updates（5,040,000 呈现，GPU2/3 world2）；FT 35 单元×70 epochs＝2450 epochs／56000 updates／2450 validations；TEST 35 单元／5875 图；CPU suite 1 次≤300 s。',
            f'- 产物：`{fp.OUT.relative_to(ROOT)}`（configs、initialization、pretrain、finetune、formal、selection、final_review）；日志 `{fp.LOG.relative_to(ROOT)}`；预算与退出码见 budget.json 与各 tmux 日志。',
            f'- 本次结果：{line}。',
            '- 证据限制：与旧 V1／V1＋FP／r10 的差值包含配方差异，不能归因于缩放单一因素；共享 TEST 非独立盲测；未 push，未改 SOTA_RESULT.md，不读取 TODO。',
            f'- 证据：`{evidence}`。']
        sota_path = fp.OUT / 'final_review/sota_comparison.md'
        if sota_path.exists():
            parts.append('\n' + sota_path.read_text().replace('# 七任务 SOTA 数值对照', '### 七任务 SOTA 数值对照', 1).strip() + '\n')
        return '\n'.join(parts) + '\n'


def controller():
    c = None
    try:
        c = AttnScaleController()
        c.register('执行中', str(c.manifest_path))
        c.prepare_initialization()
        c.freeze_config()
        fp.write(fp.OUT / 'status.json', dict(status='执行中', phase='PT', time=fp.now()))
        c.pt()
        fp.write(fp.OUT / 'status.json', dict(status='执行中', phase='FT', time=fp.now()))
        c.block(False)
        c.freeze_units()
        fp.write(fp.OUT / 'status.json', dict(status='执行中', phase='TEST', time=fp.now()))
        c.block(True)
        fp.write(fp.OUT / 'status.json', dict(status='执行中', phase='CPU_SUITE', time=fp.now()))
        argv = ['timeout', '300s', sys.executable, '-m', 'pytest', '-q', str(ROOT / 'tests/test_ua2d_attnscale.py')]
        cpu = c.launch('attnscale_final_cpu', 'cpu', argv, fp.OUT / 'final_review')
        c.wait(cpu)
        fp.freeze(fp.OUT / 'final_review/cpu_suite.json', dict(
            suite='tests/test_ua2d_attnscale.py', exit=cpu['exit'], log=cpu['log'], time=fp.now(),
            timeout_seconds=300,
            scope='independent numeric attention-scale reference and bias placement; PT export -> downstream -> formal restore consistency; legacy 512 package restore'))
        c.refresh()
        c.report()
        fp.write(fp.OUT / 'status.json', dict(status='待审查', time=fp.now(), independent_review=False,
                                              results=str(fp.OUT / 'final_review/results.json')))
        c.register('待审查', str(fp.OUT / 'final_review/results.json'))
    except BaseException as error:
        if c:
            try:
                c.refresh()
            except BaseException:
                traceback.print_exc()
            fp.write(fp.OUT / 'status.json', dict(status='阻断', error=repr(error),
                traceback=traceback.format_exc(), time=fp.now(), automatic_retry=False))
            fp.write(fp.OUT / 'final_review/results.json', dict(
                plan=fp.PLAN, revision=fp.REVISION, arm=ARM, status='阻断', error=repr(error),
                budget=fp.read(fp.OUT / 'budget.json') if (fp.OUT / 'budget.json').exists() else None,
                time=fp.now()))
            c.register('阻断', str(fp.OUT / 'status.json'))
        raise
