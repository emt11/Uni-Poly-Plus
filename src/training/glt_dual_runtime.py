"""Isolated dual-route I/O and training utilities; importing starts no work."""
import json
import os
import random
import subprocess
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.dataset.glt_dual import FrozenDualLayerSource, build_dual_sample
from src.dataset.glt_dual_cache import load_dual_cohort
from src.dataset.glt_dual_static import CHUNK_CACHE_CAPACITY


TASKS = ('eat', 'eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc')


class OrderedSampleStream:
    """Recover sampling solely from absolute position; no hidden iterator RNG."""
    def __init__(self, size, seed):
        self.size, self.seed = size, seed
        self.epoch, self.permutation = -1, None

    def index_at(self, position):
        epoch, within = divmod(position, self.size)
        if epoch != self.epoch:
            self.epoch = epoch
            self.permutation = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + epoch))
        return int(self.permutation[within])


class RankMicrobatchStream(Dataset):
    """Per-rank microbatch stream for optional multi-worker prefetch.

    One item is one collated microbatch and items are produced in exact global
    position order.  Every per-sample random stream (BRICS motif mask,
    coordinate noise) is a pure function of ``(seed, sample_key, position)``,
    so preparing in worker processes cannot change the sample order, the
    mask/noise stream or the accumulation order.  The cost this hides is the
    graph/geometry construction, which dominates on CPU.
    """

    def __init__(self, source, *, seed, world, rank, microbatch, accumulation,
                 start_step, max_steps, sigma, ratio):
        from src.dataset.glt_dual_pretrain import (  # local: avoid import cycles
            prepare_pretrain_sample, pretrain_collate,
        )

        self._prepare = prepare_pretrain_sample
        self._collate = pretrain_collate
        self.source = source
        self.stream = OrderedSampleStream(len(source), int(seed))
        self.seed = int(seed)
        self.world = int(world)
        self.rank = int(rank)
        self.microbatch = int(microbatch)
        self.accumulation = int(accumulation)
        self.batch_size = self.microbatch * self.world * self.accumulation
        self.start_step = int(start_step)
        self.steps = max(0, int(max_steps) - self.start_step)
        self.sigma = float(sigma)
        self.ratio = float(ratio)

    def __len__(self):
        return self.steps * self.accumulation

    def __getitem__(self, item):
        step = self.start_step + item // self.accumulation
        offset = item % self.accumulation
        rows = []
        for local in range(self.microbatch):
            position = (step * self.batch_size
                        + offset * self.world * self.microbatch
                        + self.rank * self.microbatch + local)
            index = self.stream.index_at(position)
            key, _ = self.source.samples[index]
            rows.append(self._prepare(
                *self.source[index], seed=self.seed, key=key.hex(),
                position=position, sigma=self.sigma, ratio=self.ratio,
                static=self.source.static_for(index),
                target=self.source.target_for(index),
            ))
        return self._collate(rows)


def require_tmux():
    if not os.environ.get('TMUX'):
        raise RuntimeError('training must run in logged tmux session Uni-Poly')
    session = subprocess.check_output(['tmux', 'display-message', '-p', '#S'], text=True).strip()
    if session != 'Uni-Poly':
        raise RuntimeError('training requires tmux session Uni-Poly')


def open_source(cohort_root, cache_root, *, task=None, dual_static_root=None,
                pretrain_target_root=None,
                chunk_cache_capacity=CHUNK_CACHE_CAPACITY):
    full_cohort = load_dual_cohort(cohort_root, cache_root)
    static_cache = target_cache = None
    if dual_static_root is not None:
        from src.dataset.glt_dual_static import CHUNK_CACHE_CAPACITY, load_static_caches
        try:
            static_cache, target_cache = load_static_caches(
                dual_static_root, pretrain_target_root,
                parent_bundle_hash=full_cohort['manifest']['main_bundle_hash'],
                cohort_manifest_hash=full_cohort['manifest_hash'],
                chunk_cache_capacity=chunk_cache_capacity,
            )
        except Exception:
            if static_cache is not None:
                static_cache.close()
            if target_cache is not None:
                target_cache.close()
            raise
        expected_order = full_cohort['manifest'].get('ordered_sample_key_hash')
        for cache in (static_cache, target_cache):
            if cache is None:
                continue
            bound_order = cache.manifest.get('cohort_ordered_sample_key_hash')
            if bound_order is not None and bound_order != expected_order:
                raise ValueError('static cache cohort ordered-key binding mismatch')
    records = full_cohort["records"]
    cohort = full_cohort
    if task is not None:
        records = [row for row in records if row.get("task") == str(task)]
        if not records:
            if static_cache is not None:
                static_cache.close()
            if target_cache is not None:
                target_cache.close()
            raise ValueError(f"dual cohort has no rows for task={task}")
        cohort = {**cohort, "records": records}
    source = FrozenDualLayerSource(cache_root, cohort, static_cache=static_cache,
                                   target_cache=target_cache)
    if not len(source):
        source.close()
        raise ValueError('empty frozen source')
    if static_cache is not None:
        try:
            # A downstream artifact may be unique-by-structure while the cohort
            # retains repeated property rows.  Resolve every selected row by key;
            # missing keys are a hard cache contract failure.
            for key, _ in source.samples:
                static_cache.index_for_key(key)
                if target_cache is not None:
                    target_cache.index_for_key(key)
        except Exception:
            source.close()
            raise
    frame = pd.DataFrame(records)
    return source, frame


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_json(path, payload):
    with Path(path).open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def scheduled_lr(step, *, lr, warmup_steps, schedule_total_steps, end_lr):
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, schedule_total_steps - warmup_steps)
    return end_lr + (lr - end_lr) * .5 * (1 + np.cos(np.pi * min(1., progress)))


def move_labels(labels, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in labels.items()}


def _tensor_payload_bytes(value):
    """Count tensor payload bytes without charging Python/container overhead."""

    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_payload_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_payload_bytes(item) for item in value)
    return 0


class CleanLabeledDataset(Dataset):
    """Clean downstream samples with an optional bounded process-local cache.

    The cache contains only the label-free CPU ``Data`` object.  Every access
    returns a clone before attaching the current target, so scaler/label
    overrides and DataLoader collation cannot mutate a cached sample.
    """

    def __init__(self, source, targets, *, cache_capacity_bytes=0):
        self.source = source
        self.raw_targets = np.asarray(targets, dtype=np.float64)
        self.targets = self.raw_targets.copy()
        self.cache_capacity_bytes = max(0, int(cache_capacity_bytes))
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0
        self._cache_skipped_bytes = 0
        if self.raw_targets.reshape(-1).size != len(source):
            raise ValueError('downstream target count differs from frozen source')

    def set_target_override(self, targets):
        values = np.asarray(targets).reshape(-1)
        if values.size != len(self.source):
            raise ValueError('target override count differs from frozen source')
        self.targets = values

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        index = int(index)
        key = self.source.samples[index][0]
        if self.cache_capacity_bytes:
            entry = self._cache.pop(key, None)
            if entry is not None:
                cached, payload_bytes = entry
                self._cache[key] = entry
                self._cache_hits += 1
                data = cached.clone()
            else:
                self._cache_misses += 1
                built = build_dual_sample(*self.source[index], static=self.source.static_for(index))
                payload_bytes = _tensor_payload_bytes(built.to_dict())
                if payload_bytes > self.cache_capacity_bytes:
                    self._cache_skipped_bytes += payload_bytes
                    data = built.clone()
                else:
                    while (self._cache and
                           self._cache_bytes + payload_bytes > self.cache_capacity_bytes):
                        _, evicted = self._cache.popitem(last=False)
                        self._cache_bytes -= evicted[1]
                        self._cache_evictions += 1
                    cached = built.clone()
                    self._cache[key] = (cached, payload_bytes)
                    self._cache_bytes += payload_bytes
                    data = cached.clone()
        else:
            built = build_dual_sample(*self.source[index], static=self.source.static_for(index))
            data = built.clone()
        data.y = torch.tensor([float(self.targets[index])], dtype=torch.float32)
        return data

    def cache_stats(self):
        return {
            'enabled': bool(self.cache_capacity_bytes),
            'capacity_bytes': int(self.cache_capacity_bytes),
            'payload_bytes': int(self._cache_bytes),
            'entries': len(self._cache),
            'hits': int(self._cache_hits),
            'misses': int(self._cache_misses),
            'evictions': int(self._cache_evictions),
            'skipped_bytes': int(self._cache_skipped_bytes),
        }
