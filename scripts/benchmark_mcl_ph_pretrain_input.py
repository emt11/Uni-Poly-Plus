#!/usr/bin/env python3
"""Bounded CPU-only comparison of real pretraining inputs; zero model updates."""
import argparse
import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from src.dataset.mcl_ph_direct import DirectMCLSource
from src.dataset.mcl_ph_trajectory_cache import MCLPHTrajectoryCache
from src.dataset.mcl_ph_view import MCLPHMicrobatchStream, load_geometric_statistics
from src.training.glt_dual_runtime import (
    OrderedSampleStream, RankMicrobatchStream, load_sample_index_artifact,
    pretrain_loader, require_tmux)
from src.training.mcl_ph_direct_contract import assert_actual_equal


class SelectedSamples:
    def __init__(self, source):
        self.source = source

    def __getitem__(self, index):
        return self.source.base.samples[self.source.local[int(index)]]


class SelectedSource:
    """Keep the full sampling denominator while reading only benchmark rows."""
    def __init__(self, base, indices, train_count):
        self.base = base
        self.local = {int(index): slot for slot, index in enumerate(indices)}
        self.count = int(train_count)
        self.samples = SelectedSamples(self)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return self.base[self.local[int(index)]]

    def static_for(self, index):
        return self.base.static_for(self.local[int(index)])

    def target_for(self, index):
        return self.base.target_for(self.local[int(index)])


class SingleReadSource(SelectedSource):
    def pretrain_fields_for(self, index, *, include_target):
        return self.base.pretrain_fields_for(self.local[int(index)], include_target=include_target)


def values(value):
    if hasattr(value, 'to_dict'):
        return values(value.to_dict())
    if isinstance(value, dict):
        return {key: values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [values(item) for item in value]
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    return value


def measure(dataset, *, workers, expected=None, profile=None):
    torch.manual_seed(1971)
    saved = torch.get_rng_state().clone()
    started = time.perf_counter()
    iterator = (iter(pretrain_loader(dataset, workers=workers, seed=42, rank=0))
                if workers else iter(dataset))
    outputs, stamps = [], []
    # Dataset iteration without a loader does not enforce __len__ itself.
    for slot in range(len(dataset)):
        begin = time.perf_counter()
        if profile is not None and slot == 0:
            profile.enable()
        output = next(iterator)
        if profile is not None and slot == 0:
            profile.disable()
        stamps.append(time.perf_counter() - begin)
        output = values(output)
        if expected is not None:
            assert_actual_equal(output, expected[slot], f'microbatch[{slot}]')
        outputs.append(output)
    elapsed = time.perf_counter() - started
    assert torch.equal(torch.get_rng_state(), saved), 'input pipeline advanced parent RNG'
    if workers:
        iterator._shutdown_workers()
    return outputs, dict(workers=workers, microbatches=len(dataset), graphs=len(dataset)*84,
                         wall_seconds=elapsed, graphs_per_second=len(dataset)*84/elapsed,
                         microbatch_wait_seconds=stamps, parent_torch_rng='exact unchanged')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_tmux()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    index = load_sample_index_artifact('results/mcl_ph_next_20260929/p0/direct_indices_v1/summary.json', 'train')
    order = OrderedSampleStream(len(index['indices']), 42)
    # Twelve rank-0 microbatches spanning four updates at the next save boundary.
    sample_indices = sorted({order.index_at(1000*1008 + step*1008 + offset*4*84 + local)
                             for step in range(4) for offset in range(3) for local in range(84)})
    source_rows = sorted(int(index['indices'][item]) for item in sample_indices)
    base = DirectMCLSource('data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging',
                           task=None, selected_indices=source_rows, split_manifest=None)
    by_row = {int(row['source_index']): slot for slot, row in enumerate(base.entries)}
    legacy = SelectedSource(base, sample_indices, len(index['indices']))
    legacy.local = {item: by_row[int(index['indices'][item])] for item in sample_indices}
    fast = SingleReadSource(base, sample_indices, len(index['indices']))
    fast.local = legacy.local.copy()
    cache = MCLPHTrajectoryCache('data/processed/mcl_ph_r4_20261001/trajectory_train_5040000')
    stats = load_geometric_statistics('data/processed/mcl_ph_r4_20261001/statistics_train_full.npz')
    common = dict(seed=42, world=4, rank=0, microbatch=84, accumulation=3,
                  start_step=1000, max_steps=1004, sigma=.03, ratio=.30)
    results = {}
    try:
        for route in ('glt', 'mcl'):
            def dataset(source):
                if route == 'glt':
                    return RankMicrobatchStream(source, third_task='none', **common)
                return MCLPHMicrobatchStream(source, statistics=stats, trajectory_cache=cache, **common)
            profiler = cProfile.Profile()
            # Keep profiler overhead outside the timing comparison.
            profiler.enable()
            dataset(legacy)[0]
            profiler.disable()
            reference, original = measure(dataset(legacy), workers=0)
            stream = io.StringIO()
            pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(25)
            args.output.with_name(f'{route}_cpu_profile.txt').write_text(stream.getvalue())
            rows = [dict(variant='original_inline', **original)]
            print(json.dumps(dict(route=route, **rows[-1])), flush=True)
            for workers in (0, 4, 8):
                _, timing = measure(dataset(fast), workers=workers, expected=reference)
                rows.append(dict(variant='single_read_inline' if not workers else 'single_read_prefetch', **timing))
                print(json.dumps(dict(route=route, **rows[-1])), flush=True)
            results[route] = rows
        report = dict(conclusion='PASS', scope='CPU input preparation only; concurrent GLT formal run continues',
                      model_forward=0, backward=0, optimizer_updates=0, coordinate_generation=0,
                      positions='rank0, step1000..1003, same seed/key/absolute positions',
                      equality='all collated input and target values/dtypes/shapes exact',
                      results=results)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    finally:
        cache.close()
        base.close()


if __name__ == '__main__':
    main()
