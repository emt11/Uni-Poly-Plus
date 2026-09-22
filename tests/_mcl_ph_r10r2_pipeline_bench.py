#!/usr/bin/env python3
"""Online vs cached data-pipeline benchmark (r10R2, section 24).

The cache changes exactly one thing: the ``[31,5]`` topology trajectory.  This
script measures what that is worth on the declared four-rank configuration by
running the real per-rank stream, in the real worker-process layout, over the
same absolute positions both ways.

Each rank records, per microbatch, the time its consumer waited for the item
(the quantity the training loop reports as ``preparation_seconds``) and the time
the item took to produce.  The reported figure is the *slowest rank* per step,
because a synchronous update is limited by it.

A consumer that never waits cannot reveal a producer bottleneck, so the consumer
is paced to the measured forward/backward time of the real run (0.46 s per
optimizer update of three microbatches, section: the CAT development run); that
is stated in the report instead of being hidden.
"""
import argparse
import json
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MB = 84
ACCUMULATION = 3
CONSUMER_SECONDS = 0.46 / ACCUMULATION


def _rank_main(rank, arguments, destination):
    import torch

    from src.dataset.mcl_ph_trajectory_cache import MCLPHTrajectoryCache
    from src.dataset.mcl_ph_view import MCLPHMicrobatchStream, load_geometric_statistics
    from src.training.glt_dual_runtime import (IndexedFrozenDualSource,
                                               load_sample_index_artifact, open_source)

    statistics = load_geometric_statistics(ROOT / arguments['statistics'])
    split = load_sample_index_artifact(str(ROOT / arguments['sample_index_artifact']), 'train')
    base, _ = open_source(str(ROOT / arguments['cohort_root']),
                          str(ROOT / arguments['cache_root']),
                          dual_static_root=str(ROOT / arguments['dual_static_root']))
    source = IndexedFrozenDualSource(base, split['indices'])
    cache = (None if arguments['mode'] == 'online'
             else MCLPHTrajectoryCache(ROOT / arguments['cache'], verify_checksums=True))
    dataset = MCLPHMicrobatchStream(
        source, seed=42, world=arguments['ranks'], rank=rank, microbatch=MB,
        accumulation=ACCUMULATION, start_step=0, max_steps=arguments['steps'],
        sigma=0.03, ratio=0.3, statistics=statistics, trajectory_cache=cache)
    loader = torch.utils.data.DataLoader(dataset, batch_size=None,
                                         num_workers=arguments['workers'],
                                         prefetch_factor=4, persistent_workers=False)
    records = []
    try:
        iterator = iter(loader)
        for item in range(len(dataset)):
            step = item // ACCUMULATION
            offset = item % ACCUMULATION
            started = time.perf_counter()
            batch, labels = next(iterator)
            waited = time.perf_counter() - started
            records.append({'rank': rank, 'step': step + 1, 'offset': offset,
                            'wait_seconds': waited, 'samples': len(labels['sample_key'])})
            # Pace the consumer like the real loop instead of draining the queue.
            time.sleep(max(0.0, CONSUMER_SECONDS - waited))
    finally:
        base.close()
    Path(destination).write_text(json.dumps(records) + '\n', encoding='utf-8')


def _quantile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def run_mode(arguments):
    started = time.perf_counter()
    context = mp.get_context('fork')
    destinations = []
    processes = []
    for rank in range(arguments['ranks']):
        destination = Path(arguments['workdir']) / f"{arguments['mode']}_rank{rank}.json"
        destinations.append(destination)
        process = context.Process(target=_rank_main, args=(rank, arguments, str(destination)))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
        if process.exitcode:
            raise SystemExit(f"rank process exited with {process.exitcode}")
    elapsed = time.perf_counter() - started
    records = []
    for destination in destinations:
        records.extend(json.loads(destination.read_text(encoding='utf-8')))
    per_step = {}
    for row in records:
        per_step[row['step']] = max(per_step.get(row['step'], 0.0), row['wait_seconds'])
    waits = list(per_step.values())
    samples = sum(row['samples'] for row in records)
    return {'mode': arguments['mode'], 'ranks': arguments['ranks'],
            'workers_per_rank': arguments['workers'], 'steps': arguments['steps'],
            'positions': samples, 'wall_seconds': round(elapsed, 2),
            'samples_per_second': round(samples / elapsed, 2),
            'max_rank_wait_median': round(statistics.median(waits), 4),
            'max_rank_wait_p90': round(_quantile(waits, 0.9), 4),
            'max_rank_wait_p99': round(_quantile(waits, 0.99), 4),
            'max_rank_wait_max': round(max(waits), 4),
            'rank_median': {str(rank): round(statistics.median(
                [row['wait_seconds'] for row in records if row['rank'] == rank]), 4)
                for rank in range(arguments['ranks'])},
            'per_step': {str(step): round(value, 4) for step, value in sorted(per_step.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('online', 'cached'))
    parser.add_argument('--cache')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--ranks', type=int, default=4)
    parser.add_argument('--workdir', default='/tmp/p2r2_pipeline_bench')
    parser.add_argument('--cohort-root',
                        default='data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1')
    parser.add_argument('--cache-root', default='data/processed/mips_trimer_scage')
    parser.add_argument('--dual-static-root',
                        default='data/processed/glt_dual_v2/pi1m/dual_static_v1')
    parser.add_argument('--sample-index-artifact',
                        default='results/glt_pred_20260918/s3b_prep/pretrain_split_v1.json')
    parser.add_argument('--statistics', default='results/mcl_ph_20260921/p0/statistics.npz')
    parser.add_argument('--output')
    args = parser.parse_args()
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    if args.mode == 'cached' and not args.cache:
        raise SystemExit('--cache is required for the cached mode')
    report = run_mode({'mode': args.mode, 'cache': args.cache, 'steps': args.steps,
                       'workers': args.workers, 'ranks': args.ranks, 'workdir': args.workdir,
                       'cohort_root': args.cohort_root, 'cache_root': args.cache_root,
                       'dual_static_root': args.dual_static_root,
                       'sample_index_artifact': args.sample_index_artifact,
                       'statistics': args.statistics})
    report['consumer_seconds_per_microbatch'] = CONSUMER_SECONDS
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + '\n', encoding='utf-8')
    print(text)
    print('=== PIPELINE BENCH ' + args.mode.upper() + ' DONE')


if __name__ == '__main__':
    main()
