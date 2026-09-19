#!/usr/bin/env python3
"""Launch the 80 S5 formal shards (B_FP vs B_NONE) on four GPUs.

Shards are queued in matched pairs (same task/fold, B_FP then B_NONE) so the
two arms of each pair run at nearly the same time.  A shard whose output
directory already carries a COMPLETED outer-test marker and metrics is
skipped; a partially written shard directory refuses the launch so a failed
outer-test can only be resolved by human review, never by silently rerunning.
"""
import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ARMS = ('b_fp', 'b_none')
TASKS = ('eat', 'eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc')
FOLDS = (0, 1, 2, 3, 4)
CHECKPOINTS = {
    'b_fp': 'results/glt_pred_20260918/s3b_formal/b_fp/pretrain/deploy_05000.pt',
    'b_none': 'results/glt_pred_20260918/s3b_formal/b_none/pretrain/deploy_05000.pt',
}
CONFIGS = {
    'b_fp': 'configs/mts/glt_pred_s3b_b_fp.json',
    'b_none': 'configs/mts/glt_pred_s3b_b_none.json',
}


def shard_completed(folder):
    metrics = folder / 'metrics.json'
    marker = folder / 'outer_test_access.json'
    if not (metrics.is_file() and marker.is_file()):
        return False
    try:
        return json.loads(marker.read_text(encoding='utf-8'))['status'] == 'COMPLETED'
    except (json.JSONDecodeError, KeyError):
        return False


def shard_blocked(folder):
    return folder.exists() and any(folder.iterdir()) and not shard_completed(folder)


def run_shard(arm, task, fold, root, gpu, log_root):
    folder = root / arm / task / f'fold{fold}'
    if shard_completed(folder):
        return 0, 'skipped_completed'
    if shard_blocked(folder):
        raise RuntimeError(
            f'incomplete shard directory needs human review: {folder}')
    log = log_root / f'{arm}_{task}_fold{fold}.log'
    command = [sys.executable, '-u', 'scripts/finetune_glt_dual.py',
               '--config', CONFIGS[arm],
               '--checkpoint', CHECKPOINTS[arm],
               '--raw-root', 'data/raw',
               '--cohort-root', 'data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1',
               '--cache-root', 'data/processed/mips_trimer_scage_downstream',
               '--dual-static-root', 'data/processed/glt_dual_v2/downstream/dual_static_v1',
               '--split-root', 'data/splits/mips_outer5_inner20',
               '--formal-shard', '--adaptation', 'full',
               '--task', task, '--fold', str(fold),
               '--output', str(folder)]
    environment = dict(__import__('os').environ)
    environment.update(CUDA_VISIBLE_DEVICES=str(gpu),
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                       PYTHONPATH='.')
    started = time.perf_counter()
    with open(log, 'w') as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                   env=environment, cwd=Path(__file__).resolve().parent.parent)
    return completed.returncode, f'{time.perf_counter() - started:.0f}s'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='results/glt_pred_20260918/s5_formal')
    parser.add_argument('--log-root', default='logs/glt_pred_20260918/s5_formal')
    parser.add_argument('--gpus', default='0,1,2,3')
    args = parser.parse_args()
    root = Path(args.root)
    log_root = Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    gpus = [value.strip() for value in args.gpus.split(',') if value.strip()]
    if len(gpus) > 4:
        raise ValueError('S5 allows at most four concurrent shards')

    queue_items = queue.Queue()
    for fold in FOLDS:
        for task in TASKS:
            for arm in ARMS:
                queue_items.put((arm, task, fold))
    total = queue_items.qsize()
    failures = []
    lock = threading.Lock()

    def worker(gpu):
        while not queue_items.empty():
            try:
                arm, task, fold = queue_items.get_nowait()
            except queue.Empty:
                return
            with lock:
                if failures:
                    queue_items.task_done()
                    return
            try:
                code, note = run_shard(arm, task, fold, root, gpu, log_root)
            except Exception as exc:  # refused shard: stop everything
                with lock:
                    failures.append((arm, task, fold, str(exc)))
                queue_items.task_done()
                return
            print(json.dumps({'arm': arm, 'task': task, 'fold': fold,
                              'gpu': gpu, 'exit_code': code, 'note': note}),
                  flush=True)
            if code != 0:
                with lock:
                    failures.append((arm, task, fold, f'exit={code}'))
            queue_items.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        print(json.dumps({'status': 'FAILED', 'failures': failures,
                          'completed': total - len(failures) - queue_items.qsize()}),
              flush=True)
        raise SystemExit(1)
    print(json.dumps({'status': 'ALL_80_SHARDS_DONE'}), flush=True)


if __name__ == '__main__':
    main()
