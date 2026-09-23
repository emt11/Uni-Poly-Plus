"""Isolated dual-route I/O and training utilities; importing starts no work."""
import json
import hashlib
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
from src.dataset.glt_dual_cache import load_dual_cohort, load_dual_cohort_task_rows
from src.dataset.glt_dual_static import CHUNK_CACHE_CAPACITY


TASKS = ('eat', 'eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sha256_int64(values):
    array = np.asarray(values, dtype='<i8')
    if array.ndim != 1:
        raise ValueError('sample index array must be one-dimensional')
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def ordered_text_hash(values):
    digest = hashlib.sha256()
    for value in values:
        raw = str(value).encode('utf-8')
        digest.update(len(raw).to_bytes(8, 'little'))
        digest.update(raw)
    return digest.hexdigest()


def load_sample_index_artifact(path, split):
    """Load and verify a read-only source-index subset contract."""

    if path is None:
        if split is not None:
            raise ValueError('sample index split requires --sample-index-artifact')
        return None
    artifact_path = Path(path).resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(f'sample index artifact is missing: {artifact_path}')
    payload = json.loads(artifact_path.read_text(encoding='utf-8'))
    if payload.get('schema_version') != 'glt-pred-pretrain-split-v1':
        raise ValueError('unsupported sample index artifact schema')
    split = str(split or 'train')
    names = {'train': 'train_source_indices',
             'validation': 'validation_source_indices',
             'fixed_validation': 'fixed_validation_source_indices'}
    if split not in names:
        raise ValueError(f'unsupported sample index split: {split}')
    npz_path = artifact_path.with_suffix('.npz')
    if not npz_path.is_file():
        raise FileNotFoundError(f'sample index npz is missing: {npz_path}')
    expected_npz = payload.get('npz_sha256')
    if expected_npz and sha256_file(npz_path) != expected_npz:
        raise ValueError('sample index npz hash mismatch')
    with np.load(npz_path, allow_pickle=False) as archive:
        if names[split] not in archive.files:
            raise ValueError(f'sample index npz is missing {names[split]}')
        indices = np.asarray(archive[names[split]], dtype=np.int64).copy()
    if indices.ndim != 1 or len(indices) != len(set(indices.tolist())):
        raise ValueError('sample index subset must be a unique one-dimensional array')
    expected_count = payload.get(f'{split}_count')
    if expected_count is not None and int(expected_count) != int(indices.size):
        raise ValueError(f'{split} source index count mismatch')
    expected_hash = payload.get(f'{split}_source_index_sha256')
    if expected_hash and sha256_int64(indices) != expected_hash:
        raise ValueError(f'{split} source index hash mismatch')
    return {
        'path': str(artifact_path),
        'sha256': sha256_file(artifact_path),
        'schema_version': payload['schema_version'],
        'split': split,
        'indices': indices,
        'payload': payload,
        'npz_path': str(npz_path),
    }


def apply_common_initialization(model, path, *, exclude=()):
    """Copy a frozen common-state initialization into one task variant.

    ``exclude`` names parameters this task variant replaces with a different
    output space; they keep their own fixed initialization and are never
    overwritten by the shared state.
    """

    artifact_path = Path(path).resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(f'common initialization artifact is missing: {artifact_path}')
    payload = torch.load(artifact_path, map_location='cpu', weights_only=False)
    state = payload.get('common_state_dict') if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not state:
        raise ValueError('common initialization artifact has no common_state_dict')
    skipped = set(exclude)
    current = model.state_dict()
    missing = [name for name, value in state.items()
               if name not in skipped
               and (name not in current
                    or tuple(current[name].shape) != tuple(value.shape))]
    if missing:
        raise ValueError('common initialization state is incompatible: ' + ','.join(missing[:5]))
    with torch.no_grad():
        for name, value in state.items():
            if name in skipped:
                continue
            current[name].copy_(value)
    return {
        'path': str(artifact_path),
        'sha256': sha256_file(artifact_path),
        'schema_version': payload.get('schema_version'),
        'common_state_sha256': payload.get('common_state_sha256'),
    }


class _IndexedSamples:
    """Light read-only Sequence mapping logical subset indices to source rows."""

    def __init__(self, base_samples, source_indices):
        self._base = base_samples
        self._indices = np.asarray(source_indices, dtype=np.int64)

    def __len__(self):
        return int(self._indices.size)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[int(i)] for i in range(*index.indices(len(self)))]
        return self._base[int(self._indices[int(index)])]


class IndexedFrozenDualSource(Dataset):
    """Read-only logical view over a frozen source without copying artifacts."""

    def __init__(self, base, source_indices):
        self.base = base
        self.source_indices = np.asarray(source_indices, dtype=np.int64).copy()
        if self.source_indices.ndim != 1:
            raise ValueError('subset source indices must be one-dimensional')
        if self.source_indices.size and (
                int(self.source_indices.min()) < 0
                or int(self.source_indices.max()) >= len(base)):
            raise IndexError('subset source index is outside the frozen source')
        if len(set(self.source_indices.tolist())) != int(self.source_indices.size):
            raise ValueError('subset source indices must be unique')
        self.samples = _IndexedSamples(base.samples, self.source_indices)

    def __len__(self):
        return len(self.source_indices)

    def __getitem__(self, index):
        return self.base[int(self.source_indices[int(index)])]

    def static_for(self, index):
        return self.base.static_for(int(self.source_indices[int(index)]))

    def target_for(self, index):
        return self.base.target_for(int(self.source_indices[int(index)]))

    @property
    def cohort(self):
        return self.base.cohort

    @property
    def bundle(self):
        return self.base.bundle

    @property
    def topology(self):
        return self.base.topology

    @property
    def trimer(self):
        return self.base.trimer

    @property
    def static_cache(self):
        return self.base.static_cache

    @property
    def target_cache(self):
        return self.base.target_cache

    def close(self):
        self.base.close()


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
                 start_step, max_steps, sigma, ratio, third_task='fp',
                 fgr_mu=0.0, fgr_sigma=1.0, fgr_max_pairs=32,
                 atom_target='element', env_vocab=None, torsion_mode=None):
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
        self.third_task = str(third_task)
        self.fgr_mu = float(fgr_mu)
        self.fgr_sigma = float(fgr_sigma)
        self.fgr_max_pairs = int(fgr_max_pairs)
        self.atom_target = str(atom_target)
        self.env_vocab = env_vocab
        self.torsion_mode = torsion_mode

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
                target=(self.source.target_for(index) if self.third_task == 'fp' else None),
                third_task=self.third_task, fgr_mu=self.fgr_mu,
                fgr_sigma=self.fgr_sigma, fgr_max_pairs=self.fgr_max_pairs,
                atom_target=self.atom_target, env_vocab=self.env_vocab,
                torsion_mode=self.torsion_mode,
            ))
        return self._collate(rows)


def require_tmux():
    if not os.environ.get('TMUX'):
        raise RuntimeError('training must run in logged tmux session Uni-Poly')
    session = subprocess.check_output(['tmux', 'display-message', '-p', '#S'], text=True).strip()
    if session != 'Uni-Poly':
        raise RuntimeError('training requires tmux session Uni-Poly')


def open_source(cohort_root, cache_root, *, task=None, dual_static_root=None,
                pretrain_target_root=None, selected_indices=None,
                expected_task_rows=None, expected_split_sha256=None,
                record_index_path=None,
                chunk_cache_capacity=CHUNK_CACHE_CAPACITY):
    if selected_indices is None:
        full_cohort = load_dual_cohort(cohort_root, cache_root)
    else:
        if task is None or expected_task_rows is None or expected_split_sha256 is None:
            raise ValueError(
                "selected cohort loading requires task, expected_task_rows and split SHA256"
            )
        full_cohort = load_dual_cohort_task_rows(
            cohort_root, cache_root, task=task, row_indices=selected_indices,
            expected_task_rows=expected_task_rows,
            expected_split_sha256=expected_split_sha256,
            record_index_path=record_index_path,
        )
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
    if task is not None and selected_indices is None:
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

    def __init__(self, source, targets, *, cache_capacity_bytes=0, torsion_mode=None):
        self.source = source
        self.raw_targets = np.asarray(targets, dtype=np.float64)
        self.targets = self.raw_targets.copy()
        self.cache_capacity_bytes = max(0, int(cache_capacity_bytes))
        self.torsion_mode = torsion_mode
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
                built = build_dual_sample(*self.source[index], static=self.source.static_for(index),
                                          torsion_mode=self.torsion_mode)
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
            built = build_dual_sample(*self.source[index], static=self.source.static_for(index),
                                      torsion_mode=self.torsion_mode)
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
