"""One authorized repeat of the original 35 BASE BEST evaluations; no training."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/ua3d_base_opt_20261026/r6/repairs/concurrency_r1'
OUT = ROOT / 'results/ua3d_base_test_repeat_20261029/r2_gpu3'
LOG = ROOT / 'logs/ua3d_base_test_repeat_20261029/r2_gpu3'
ARM = 'F_PD_N1_F0'
TASKS = ('eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc')
UUID = 'GPU-665c4372-2ad2-31f1-7b34-8ec763ef6b38'

def read(path):
    return json.loads(path.read_text())

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)

def runtime():
    sys.path.insert(0, str(ROOT))
    os.environ['UA3D_BASE_OPT_RESOURCE_REVISION'] = 'gpu2-pt2-fttest4-c1'
    os.environ['UA3D_BASE_OPT_EXECUTION_ROOT'] = str(SOURCE)
    from src.training import ua3d_base_opt_r6 as t
    original = t.unit_path
    t.unit_path = lambda arm, task, fold, formal=False: (
        OUT / 'formal' / arm / task / f'fold{fold}' if formal
        else original(arm, task, fold, False))
    # Frozen checkpoint config retains its original GPU2 provenance. Only actual
    # execution placement changes; frozen estimator/config identity is preserved.
    t.ALLOWED_GPUS = (3,)
    def gpu_contract():
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '3' or os.environ.get('WORLD_SIZE') != '1':
            raise ValueError('GPU3/world1 env mismatch')
        if t.torch.cuda.device_count() != 1:
            raise ValueError('must expose exactly one CUDA device')
        actual = 'GPU-' + str(t.torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
        if actual != UUID:
            raise ValueError('GPU3 UUID mismatch')
        return dict(physical_gpus=[3], physical_gpu=3, logical_device='cuda:0',
                    gpu_uuid=actual, world_size=1, test_max_concurrency=4,
                    execution_revision='r2-gpu3', original_checkpoint_device_provenance=2)
    t.gpu_contract = gpu_contract
    return t

def inventory():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    row = next(x for x in rows.splitlines() if x.split(',')[0].strip() == '3')
    _, uuid, free = [x.strip() for x in row.split(',')]
    if uuid != UUID:
        raise ValueError('GPU3 UUID changed')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader,nounits'], text=True)
    pids = [int(x.split(',')[1]) for x in apps.splitlines() if x.split(',')[0].strip() == UUID]
    return dict(uuid=uuid, free_mib=int(free), pids=pids, time=time.time())

def controller():
    if not os.environ.get('TMUX'):
        raise RuntimeError('tmux required')
    OUT.mkdir(parents=True, exist_ok=False)
    LOG.mkdir(parents=True, exist_ok=False)
    rc = 1
    jobs = []
    active = []
    try:
        write(OUT / 'authorization.json', dict(
            user=['请你重新执行一遍 BASE 的 TEST ！', '使用 GPU 2 ，最多 4 路！', '请你切换到 GPU 3 执行！'],
            plan='UA3D-BASE-TEST-REPEAT-20261029-01/r2-gpu3', source=str(SOURCE),
            reference=0.804675462276949, physical_gpus=[3], concurrency=4,
            training=0, test_units=35, test_rows=5875, automatic_retry=False,
            shared_development_evaluation=True, independent_review='PENDING'))
        for name, command in [('git_status.txt', ['git', 'status', '--short']),
                              ('git_diff.patch', ['git', 'diff']),
                              ('git_head.txt', ['git', 'rev-parse', 'HEAD'])]:
            (OUT / name).write_text(subprocess.check_output(command, cwd=ROOT, text=True))
        frozen = read(SOURCE / 'selection/final_frozen.json')
        units = {f'{ARM}/{task}/{fold}': frozen['units'][f'{ARM}/{task}/{fold}']
                 for task in TASKS for fold in range(5)}
        for unit in units.values():
            if not Path(unit['best']).is_file():
                raise FileNotFoundError(unit['best'])
        write(OUT / 'frozen_source.json', dict(source_freeze=str(SOURCE / 'selection/final_frozen.json'),
              original_freeze_time=frozen['time'], units=units, time=time.time()))
        matrix = [(task, fold) for task in TASKS for fold in range(5)]
        for offset in range(0, len(matrix), 4):
            # Unknown host NVML PIDs cannot be mapped reliably in this container.
            # Admit a whole batch only after GPU3 is empty; never guess ownership.
            while True:
                inv = inventory()
                write(OUT / 'status.json', dict(status='WAIT_GPU', completed=offset, inventory=inv))
                if not inv['pids'] and inv['free_mib'] >= 32768:
                    break
                time.sleep(5)
            active = []
            for task, fold in matrix[offset:offset + 4]:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='3', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                           WORLD_SIZE='1', UA3M_PHYSICAL_GPU_UUID=UUID, UA3D_BASE_OPT_JOB_KIND='test',
                           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                           PYTHONDONTWRITEBYTECODE='1')
                argv = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker',
                        '--task', task, '--fold', str(fold)]
                log = LOG / f'{task}_fold{fold}.log'
                handle = log.open('x')
                job = dict(task=task, fold=fold, argv=argv, cwd=str(ROOT), log=str(log),
                           started=time.time(), exit=None,
                           env={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER',
                                'WORLD_SIZE','UA3M_PHYSICAL_GPU_UUID','UA3D_BASE_OPT_JOB_KIND')})
                jobs.append(job)
                write(OUT / 'budget.json', dict(reserved_units=len(jobs), max_units=35, max_rows=5875, jobs=jobs))
                process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                job['pid'] = process.pid
                active.append((process, handle, job))
            while any(p.poll() is None for p, _, _ in active):
                write(OUT / 'status.json', dict(status='RUNNING', completed=offset,
                      active=[j for p, _, j in active if p.poll() is None]))
                time.sleep(2)
            for process, handle, job in active:
                job.update(exit=process.returncode, ended=time.time())
                handle.close()
            write(OUT / 'budget.json', dict(reserved_units=len(jobs), max_units=35, max_rows=5875, jobs=jobs))
            if any(j['exit'] != 0 for _, _, j in active):
                raise RuntimeError('TEST worker failed; no retry')
        t = runtime()
        result = t.aggregate(ARM, True)
        original = read(SOURCE / 'results.json')['formal'][ARM]
        delta = result['macro'] - original['macro']
        comparisons = {}
        for task in TASKS:
            for fold in range(5):
                new_path = OUT / 'formal' / ARM / task / f'fold{fold}' / 'test_predictions.npz'
                old_path = SOURCE / 'formal' / ARM / task / f'fold{fold}' / 'test_predictions.npz'
                with t.np.load(new_path, allow_pickle=False) as new, t.np.load(old_path, allow_pickle=False) as old:
                    if set(new.files) != set(old.files):
                        raise ValueError('prediction schema mismatch')
                    comparisons[f'{task}/{fold}'] = {k:bool(t.np.array_equal(new[k], old[k])) for k in new.files}
        write(OUT / 'results.json', dict(result=result, reference_macro=original['macro'], delta_macro=delta,
              task_deltas={task:result['tasks'][task]['mean']-original['tasks'][task]['mean'] for task in TASKS},
              prediction_field_equality=comparisons, independent_review='PENDING'))
        rows = ['# BASE TEST 复测', '', f'macro R²={result["macro"]:.3f}；相对原BASE Δ={delta:+.3f}。', '',
                '| Task | R² mean ± std | Δ原BASE | 论文最佳 | Δ论文 |', '|---|---:|---:|---:|---:|']
        paper = dict(eea=.934, egb=.923, egc=.911, ei=.839, eps=.787, nc=.856, xc=.423)
        for task in TASKS:
            v = result['tasks'][task]
            rows.append(f'| {task} | {v["mean"]:.3f} ± {v["std"]:.3f} | {v["mean"]-original["tasks"][task]["mean"]:+.3f} | {paper[task]:.3f} | {v["mean"]-paper[task]:+.3f} |')
        rows += ['', '原35个BEST；原split、scaler、batch24、FP32、workers0；五折std ddof=1，七任务等权macro。',
                 '共享开发评价；论文包络不是单一模型，split/样本协议差异仅作数值对照。',
                 '本次训练0、额外选模0、CPU suite0；35次TEST/5875行；不替换BASE。independent_review=PENDING。',
                 '论文参考：SOTA_RESULT.md；Egb/Egc为Periodic-TDL (+Chem,+RC)，其余Periodic-TDL。',
                 '完整精度与逐字段预测比较见results.json；逐单元真实exit见budget.json。']
        (OUT / 'report.md').write_text('\n'.join(rows) + '\n')
        rc = 0
    except BaseException:
        write(OUT / 'failure.json', dict(error=traceback.format_exc(), automatic_retry=False))
        raise
    finally:
        for process, handle, job in active:
            if process.poll() is None:
                process.terminate()
                job['exit'] = process.wait()
            handle.close()
        write(OUT / 'controller_exit.json', dict(exit=rc, ended=time.time()))
        write(OUT / 'status.json', dict(status='COMPLETE' if rc == 0 else 'FAILED',
              completed=sum(j['exit'] == 0 for j in jobs), independent_review='PENDING'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('controller','worker'))
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--fold', type=int, choices=range(5))
    args = parser.parse_args()
    if args.action == 'controller':
        controller()
    else:
        runtime().unit(ARM, args.task, args.fold, True, 3)
