#!/usr/bin/env python
"""One-shot CPU controller; each stage has its own actual subprocess exit."""
import argparse
import csv
import difflib
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training import ua2d_p10k_r10fp as m

FILES = ('src/training/ua2d_p10k_r10fp.py', 'scripts/run_ua2d_p10k_r10fp.py',
         'configs/ua2d_p10k_r10fp_r1.json', 'tests/test_ua2d_p10k_r10fp.py')
AUTHORIZATION = ('用户本会话明确指令：实施并完整执行 FP_PLAN.md 第13节 '
                 'UA2D-P10K-R10FP-20261023-01 / r1；一次prepare，validation选择唯一全局权重，'
                 '一次冻结，35正式TEST政策评估，唯一最终CPU测试。仅许可GPU3但本轮CPU-only，'
                 '新增PT/FT/KRR拟合/FP生成/模型推理/GPU/worker均0；不重试、不扩大范围，待Codex独立审查。')


def append_plan(text):
    with (m.ROOT / 'CURRENT_PLAN.md').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write('\n\n### ' + m.PLAN + ' / r1 执行记录\n\n' + text + '\n')
        f.flush(); os.fsync(f.fileno())


class Controller:
    def __init__(self):
        m.resource_guard()
        if not os.environ.get('TMUX'):
            raise RuntimeError('requires tmux Uni-Poly')
        self.window = subprocess.check_output(
            ['tmux', 'display-message', '-p', '-t', os.environ['TMUX_PANE'], '#S #{window_id}'], text=True).strip()
        if not self.window.startswith('Uni-Poly '):
            raise RuntimeError('requires Uni-Poly session')
        # No old controllers, lock files or result roots are changed.
        m.OUT.mkdir(parents=True, exist_ok=False)
        self.lock = (m.OUT / 'controller.lock').open('x')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.started = None; self.used = dict.fromkeys(m.LIMITS, 0)
        self.reserved = {}; self.jobs = []; self.error = None; self.child = None
        self.phase = 'deployment'; self.cpu_exit = None
        auth = m.OUT / 'authorization'; auth.mkdir()
        for rel in ('CURRENT_PLAN.md', 'FP_PLAN.md', 'AGENTS.md', 'SOTA_RESULT.md'):
            (auth / ('before_' + rel)).write_text((m.ROOT / rel).read_text())
        git = subprocess.check_output(['git', 'status', '--short'], cwd=m.ROOT, text=True)
        (auth / 'git_status.txt').write_text(git)
        branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=m.ROOT, text=True).strip()
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=m.ROOT, text=True).strip()
        m.write(auth / 'user_authorization.json', dict(plan=m.PLAN, revision=m.REVISION,
                time=m.now(), source='current user instruction in this conversation', text=AUTHORIZATION,
                branch=branch, commit=commit, limits=m.LIMITS, CPU_only=True,
                allowed_gpu=[3], actual_gpu_tasks=0, retry=False, resume=False))
        cfg = m.read(m.ROOT / FILES[2])
        if (cfg['plan'], cfg['revision'], cfg['CPU_only'], cfg['weight_grid']) != (m.PLAN, m.REVISION, True, list(m.WEIGHTS)):
            raise ValueError('deployment config identity mismatch')
        m.write(auth / 'configuration.json', cfg)
        self.manifest = dict(plan=m.PLAN, revision=m.REVISION, time=m.now(),
                process=dict(pid=os.getpid(), start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]),
                window=self.window, cwd=str(m.ROOT), command=sys.argv, env=m.ENV, jobs=self.jobs,
                authorized_gpu=[3], actual_gpu_tasks=0, workers=0, independent_review='PENDING')
        self.snapshot()
        append_plan('- ' + m.now() + '：' + AUTHORIZATION + '\n'
            '- 已确认dev/' + commit + '及dirty工作区；未发现本周期writer或冲突产物。保留旧周期及其他作业。\n'
            '- 新根：' + str(m.OUT) + '；日志：' + str(m.LOG) + '；独立预算：' + json.dumps(m.LIMITS) + '。\n'
            '- 状态：执行中；单CPU controller，CUDA_VISIBLE_DEVICES为空、WORLD_SIZE=1、线程1、workers0。'
            '从prepare首次启动起≤3600秒；prepare/selection/formal分别≤600/120/300秒，最终suite≤300秒。')
        self.refresh()

    def snapshot(self):
        dest = m.OUT / 'final_review'; dest.mkdir(exist_ok=True)
        lines = []
        for rel in FILES:
            lines.extend(difflib.unified_diff([], (m.ROOT / rel).read_text().splitlines(True),
                                             fromfile='/dev/null', tofile='b/' + rel))
        for name in ('actual_source_diff.patch', 'implementation_incremental_diff.patch'):
            (dest / name).write_text(''.join(lines))
        static = m.LOG / 'controller/static_checks.json'
        if static.exists():
            (dest / 'static_checks.json').write_text(static.read_text())

    def refresh(self):
        m.check_budget(self.used, self.reserved)
        m.write(m.OUT / 'budget.json', dict(plan=m.PLAN, revision=m.REVISION,
                limits=m.LIMITS, used=self.used, active_reservation=self.reserved,
                wallclock_started_unix=self.started,
                deadline_unix=None if self.started is None else self.started + 3600,
                jobs=self.jobs, uncertainty='Failed stage retains full unclosed reservation; no retry/resume.'), replace=True)
        self.manifest['jobs'] = self.jobs
        m.write(m.OUT / 'execution_manifest.json', self.manifest, replace=True)

    def state(self, status, phase):
        m.write(m.OUT / 'status.json', dict(plan=m.PLAN, revision=m.REVISION,
                time=m.now(), status=status, phase=phase, error=self.error,
                independent_review='PENDING'), replace=True)

    def launch(self, stage, args, seconds, start_counts, reserve=None):
        self.phase = stage
        if self.started is not None:
            seconds = min(seconds, self.started + 3600 - time.time())
        if seconds <= 0:
            raise TimeoutError('campaign deadline exhausted before ' + stage)
        for k, v in start_counts.items():
            self.used[k] += v
        self.reserved = {} if reserve is None else reserve.copy()
        job = dict(stage=stage, cwd=str(m.ROOT), command=args, environment=m.ENV,
                   window=self.window, start=m.now(), status='RUNNING', exit=None,
                   timeout_seconds=seconds, log=str(m.LOG / stage))
        self.jobs.append(job); self.refresh(); self.state('执行中', stage)
        log = m.LOG / stage; log.mkdir(parents=True, exist_ok=False)
        m.write(log / 'command.json', job)
        with (log / 'stdout.log').open('x') as stdout, (log / 'stderr.log').open('x') as stderr:
            try:
                self.child = subprocess.Popen(args, cwd=m.ROOT, env=os.environ.copy(),
                        stdout=stdout, stderr=stderr, start_new_session=True)
                job['pid'] = self.child.pid; self.refresh()
                try:
                    rc = self.child.wait(timeout=seconds)
                except subprocess.TimeoutExpired:
                    job['timed_out'] = True
                    os.killpg(self.child.pid, signal.SIGTERM)
                    try:
                        rc = self.child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.child.pid, signal.SIGKILL); rc = self.child.wait()
                job.update(exit=rc, ended=m.now(), status='COMPLETE' if rc == 0 and not job.get('timed_out') else 'FAILED')
                (log / 'exit.txt').write_text(str(rc) + '\n')
                m.write(log / 'exit.json', dict(subprocess_returncode=rc,
                        shell_equivalent_exit=128 - rc if rc < 0 else rc,
                        timed_out=job.get('timed_out', False), ended=job['ended']))
            finally:
                if self.child is not None and self.child.poll() is None:
                    os.killpg(self.child.pid, signal.SIGTERM)
                    try:
                        self.child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.child.pid, signal.SIGKILL); self.child.wait()
                    job.update(exit=self.child.returncode, ended=m.now(), status='FAILED')
                    (log / 'exit.txt').write_text(str(self.child.returncode) + '\n')
                self.child = None
                counter = m.OUT / f'{stage}_consumption.json'
                if counter.exists():
                    c = m.read(counter); job['consumption'] = c
                    for k, v in c['counts'].items():
                        if k not in self.reserved or v > self.reserved[k]:
                            raise ValueError('stage counter/reservation mismatch: ' + k)
                        self.used[k] += v; self.reserved[k] -= v
                if job.get('status') == 'COMPLETE':
                    if any(self.reserved.values()):
                        raise ValueError('normal exit with incomplete stage counters')
                    self.reserved = {}
                else:
                    job['unclosed_reservation_charged'] = self.reserved.copy()
                    for k, v in self.reserved.items():
                        self.used[k] += v
                    self.reserved = {}
                self.refresh()
        if job['status'] != 'COMPLETE':
            raise RuntimeError(stage + ' failed; true returncode=' + str(job['exit']) + ', timeout=' + str(job.get('timed_out', False)))
        return job['exit']

    def run(self):
        script = str(m.ROOT / FILES[1])
        def command(stage):
            return [sys.executable, '-u', script, stage]
        try:
            self.started = time.time()
            self.manifest['wallclock_started_unix'] = self.started
            self.launch('prepare', command('prepare'), 600, dict(prepare_starts=1),
                        dict(validation_units=35, validation_presentations=4714))
            self.launch('select', command('select'), 120, {}, dict(weight_candidates=5, validation_scores=175))
            self.launch('freeze', command('freeze'), 120, {}, dict(freezes=1))
            self.launch('formal', command('formal'), 300, {}, dict(test_starts=35, test_graphs=5875, outputs=35))
        except BaseException:
            self.error = traceback.format_exc()
            m.write(m.OUT / 'failure.json', dict(phase=self.phase, time=m.now(), error=self.error, retry=False))
        # The original one final suite is also used after safe abnormal closeout.
        if self.used['cpu_suites'] == 0 and time.time() < self.started + 3600:
            try:
                self.cpu_exit = self.launch('cpu', [sys.executable, '-m', 'pytest', '-q', FILES[3]],
                                            300, dict(cpu_suites=1))
            except BaseException:
                cpu_job = next((j for j in self.jobs if j['stage'] == 'cpu'), {})
                self.cpu_exit = cpu_job.get('exit')
                self.error = (self.error or '') + '\nCPU: ' + traceback.format_exc()
        status = '待审查' if self.error is None and self.cpu_exit == 0 else '需返修'
        self.state(status, 'delivery')
        self.deliver(status); self.refresh()
        return 0 if status == '待审查' else 1

    def deliver(self, status):
        complete = (m.OUT / 'formal/results.json').exists()
        rows = m.read(m.OUT / 'formal/results.json')['rows'] if complete else []
        if not complete:
            rows = [m.read(p) for p in sorted((m.OUT / 'formal').glob('*/fold*/metrics.json'))]
        result = dict(plan=m.PLAN, revision=m.REVISION, status=status, complete_test=complete,
                      cpu_exit=self.cpu_exit, rows=rows, error=self.error, independent_review='PENDING',
                      total_seconds=time.time() - self.started, selection=None,
                      protocol='shared development TEST, not independent blind evaluation; paper equivalence unverified; five folds equal, std ddof1, seven tasks equal')
        if (m.OUT / 'selection/validation_selection.json').exists():
            result['selection'] = m.read(m.OUT / 'selection/validation_selection.json')
        tasks = {}; sota = []
        paper = {}
        for line in (m.ROOT / 'SOTA_RESULT.md').read_text().splitlines():
            parts = [s.strip() for s in line.split('|')]
            if len(parts) == 5 and parts[1].lower() in m.TASKS and parts[3].startswith('**'):
                paper[parts[1].lower()] = dict(model=parts[2], r2=float(parts[3].replace('**', '').split('±')[0]))
        for task in m.TASKS:
            r = [x for x in rows if x['task'] == task]
            if len(r) != 5:
                sota.append(dict(task=task, status='未完成／不可比较', reference=paper.get(task)))
                continue
            a = m.np.asarray([x['metrics']['r2'] for x in r]); b = m.np.asarray([x['reference']['r2'] for x in r])
            tasks[task] = dict(r2_mean=float(a.mean()), r2_std=float(a.std(ddof=1)),
                    reference_mean=float(b.mean()), reference_std=float(b.std(ddof=1)),
                    delta_r2=float((a-b).mean()), positive_folds=int((a>b).sum()))
            sota.append(dict(task=task, project=tasks[task], reference=paper[task],
                             delta_paper=float(a.mean() - paper[task]['r2']), protocol='仅数值对照'))
        result.update(tasks=tasks, sota=sota)
        if complete and len(tasks) == 7:
            macro = float(m.np.mean([tasks[t]['r2_mean'] for t in m.TASKS]))
            ref = float(m.np.mean([tasks[t]['reference_mean'] for t in m.TASKS]))
            result.update(macro_r2=macro, reference_macro_r2=ref, delta_macro_r2=macro-ref,
                          performance_pass=bool(macro-ref >= .010))
        m.write(m.OUT / 'final_review/results.json', result)
        with (m.OUT / 'final_review/fold_differences.csv').open('x') as f:
            writer = csv.DictWriter(f, fieldnames=['task','fold','alpha','weight','policy_r2','p10k_r2','delta_r2','predictions'])
            writer.writeheader()
            for r in rows:
                writer.writerow({**{k:r[k] for k in ('task','fold','alpha','weight','delta_r2','predictions')},
                                 'policy_r2':r['metrics']['r2'], 'p10k_r2':r['reference']['r2']})
        text = '# ' + m.PLAN + ' / r1\n\n状态：' + status + '；交Codex独立审查，执行自检不构成独立审查。\n\n'
        text += '一次prepare → validation唯一全局权重 → 一次冻结 → 35正式政策评估 → 唯一最终CPU suite。\n\n'
        text += 'PT、FT、KRR拟合、FP生成、模型推理、GPU任务、worker全部为0。CPU exit=' + str(self.cpu_exit) + '。\n\n'
        if result['selection']:
            text += '全局P10K权重 w=' + str(result['selection']['weight']) + '，仅依据175个validation R²选择；端点按实际政策解释。\n\n'
        if complete:
            text += f"正式macro R²={result['macro_r2']:.3f}，P10K_REF={result['reference_macro_r2']:.3f}，Δ={result['delta_macro_r2']:+.3f}；性能门槛通过={result['performance_pass']}（未舍入Δ≥0.010）。\n\n"
        else:
            text += '正式结果未完成／不可比较；保留已有单元，不补跑。\n\n'
        text += '|任务|项目R²|P10K_REF|配对Δ|正向fold|论文参考|项目减论文|参考模型|\n|---|---:|---:|---:|---:|---:|---:|---|\n'
        for t in m.TASKS:
            if t not in tasks:
                text += f'|{t}|未完成／不可比较|||||||\n'; continue
            a = tasks[t]; p = paper[t]
            text += f"|{t}|{a['r2_mean']:.3f}±{a['r2_std']:.3f}|{a['reference_mean']:.3f}±{a['reference_std']:.3f}|{a['delta_r2']:+.3f}|{a['positive_folds']}/5|{p['r2']:.3f}|{a['r2_mean']-p['r2']:+.3f}|{p['model']}|\n"
        text += '\n五折seed42+fold等权mean/std(ddof1)，七任务等权macro。共享开发TEST非独立盲测；论文协议未证明等价，仅数值比较。'
        text += '历史r10采用不同神经分支且旧std为ddof0，不将跨周期差异作为严格消融。原始预测及指标保留完整精度。\n\n'
        text += '冻结：selection/final_frozen.json；35组预测：formal/{task}/fold{fold}/predictions.npz；所有配置、scaler和来源见冻结名单及inputs/metadata.json。\n\n'
        text += f'总墙钟：{result["total_seconds"]:.3f}秒；各阶段命令/PID/window/真实exit/stdout/stderr见execution_manifest.json及{m.LOG}。\n\n'
        text += '异常：' + (self.error or '无') + '\n'
        (m.OUT / 'final_review/report.md').write_text(text)
        m.write(m.OUT / 'final_review/budget_and_exits.json', dict(used=self.used, limits=m.LIMITS, jobs=self.jobs))
        append_plan('- ' + m.now() + '：' + text + '\n- 产物：' + str(m.OUT) + '；日志：' + str(m.LOG))
        plan_diff = ''.join(difflib.unified_diff(
            (m.OUT / 'authorization/before_CURRENT_PLAN.md').read_text().splitlines(True),
            (m.ROOT / 'CURRENT_PLAN.md').read_text().splitlines(True),
            fromfile='a/CURRENT_PLAN.md', tofile='b/CURRENT_PLAN.md'))
        for name in ('actual_source_diff.patch', 'implementation_incremental_diff.patch'):
            with (m.OUT / 'final_review' / name).open('a') as f:
                f.write(plan_diff)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['run','prepare','select','freeze','formal'])
    args = parser.parse_args()
    m.resource_guard()
    if args.stage == 'run':
        return Controller().run()
    # Stage entrypoints are not independent authorization / resume paths.
    parent = os.getppid(); manifest = m.read(m.OUT / 'execution_manifest.json')
    if manifest['process']['pid'] != parent:
        raise RuntimeError('stage requires the current unique controller parent')
    getattr(m, args.stage)()
    return 0


if __name__ == '__main__':
    sys.exit(main())
