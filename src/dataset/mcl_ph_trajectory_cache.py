"""Offline noisy-topology trajectory cache for the MCL-PH route (r10R2).

The pre-training hot path of the MCL-PH route spends most of its CPU on the
``[31, 5]`` topology trajectory: a filtered Vietoris-Rips persistence plus a
shortest-path sweep at each of 31 radii.  None of that depends on the model, the
fusion module or the optimizer -- it is a pure function of the *noisy view* of
one presentation, and the noisy view is itself a pure function of
``(seed, key, position)`` once the shared motif-mask draw is taken into account.

This module stores exactly that function's output, keyed by absolute
presentation position, and nothing else.  The 2/3/4 A expert graphs, the
perturbation itself, the atom-mask targets, the local-geometry targets, the
non-bond targets and every model tensor stay online: caching them would change
the data path far more than it would save, and the three graphs are cheap next
to persistence.

Two consequences of the key choice are worth stating explicitly.

* The key is the *position*, never the molecule.  A molecule drawn at two
  positions is perturbed with different noise, so a ``sample_key -> trajectory``
  map would be wrong.
* The stored value is ``float32`` exactly as the online path casts it.  A
  ``float16`` cache would be a change of scientific definition, not an
  optimization, so the dtype is part of the cache identity and is verified on
  every read.

The reader fails closed: an identity mismatch, a missing shard, a truncated or
corrupt shard, or a run that needs more positions than the cache holds is an
error, never a fallback to the online computation.

Build state lives beside the shards rather than in the driver's memory.  Every
work segment writes a small marker under ``build_state/segments/`` once its rows
are flushed, so resuming, recovering from a failed driver and deciding whether a
shard is finished all read the filesystem instead of trusting a process that may
already be gone.  ``adopt_written_segments`` recovers an existing shard set whose
markers were never written; the manifest stays the reader's only contract.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .mcl_ph_view import (DESCRIPTOR_COLUMNS, ROUTER_RADII, build_mcl_ph_view,
                          five_descriptors, validate_geometry_carrier)

CACHE_SCHEMA = 'mcl-ph-noisy-trajectory-cache-v1'
MANIFEST_NAME = 'manifest.json'
SHARDS_DIRNAME = 'shards'
SHARD_PREFIX = 'trajectory'
DEFAULT_SHARD_SIZE = 100_000
DESCRIPTOR_DTYPE = np.float32
SHARD_ROW_FIELDS = ('shard', 'file', 'first_position', 'rows', 'sha256')

# Completion evidence of a build lives in the filesystem, one small JSON marker
# per *segment* (the unit of work), written by the process that wrote the rows.
# The driver's bookkeeping is derived from these markers rather than from the
# worker result stream, so a driver-side failure can never make a ten-hour build
# unreadable: the markers are still on disk and the next invocation adopts them.
BUILD_STATE_DIRNAME = 'build_state'
SEGMENTS_DIRNAME = 'segments'
MARKER_FIELDS = ('shard', 'first_position', 'rows', 'finished_at', 'pid')

# The fields that define *which* presentations the cache describes.  A run whose
# identity differs in any of them is refused.  ``total_positions`` is coverage
# and is checked separately, because a short smoke may legitimately read the
# prefix of a cache built for the full schedule.
IDENTITY_FIELDS = ('seed', 'noise_sigma', 'mask_ratio', 'global_batch',
                   'sample_index_split', 'sample_index_artifact_sha256',
                   'cohort_manifest_hash', 'dual_static_manifest_hash')


def cache_identity(*, seed, noise_sigma, mask_ratio, global_batch, sample_index_split,
                   sample_index_artifact_sha256, cohort_manifest_hash,
                   dual_static_manifest_hash):
    """The identity block a cache and a training run must agree on, field for field."""
    identity = {
        'seed': int(seed),
        'noise_sigma': float(noise_sigma),
        'mask_ratio': float(mask_ratio),
        'global_batch': int(global_batch),
        'sample_index_split': str(sample_index_split),
        'sample_index_artifact_sha256': str(sample_index_artifact_sha256),
        'cohort_manifest_hash': str(cohort_manifest_hash),
        'dual_static_manifest_hash': str(dual_static_manifest_hash),
    }
    for name in IDENTITY_FIELDS:
        value = identity[name]
        if value is None or (isinstance(value, str) and not value):
            raise ValueError(f'trajectory cache identity field {name} is empty')
    return identity


def online_provenance():
    """The ``trajectory_cache`` block of a run that computes its topology online."""
    return {'mode': 'online'}


def shard_bounds(shard_size, total_positions):
    """``(first_position, rows)`` of every shard, the last one possibly short."""
    shard_size = int(shard_size)
    total = int(total_positions)
    if shard_size <= 0 or total <= 0:
        raise ValueError('shard size and position count must be positive')
    bounds = []
    first = 0
    while first < total:
        rows = min(shard_size, total - first)
        bounds.append((first, rows))
        first += rows
    return bounds


def shard_file_name(first_position, rows):
    return f'{SHARD_PREFIX}_{first_position:07d}_{first_position + rows - 1:07d}.npy'


def shard_paths(root, shard_size, total_positions):
    """Every shard path of a cache, in position order."""
    root = Path(root)
    return [root / SHARDS_DIRNAME / shard_file_name(first, rows)
            for first, rows in shard_bounds(shard_size, total_positions)]


def position_chunks(shard_size, total_positions, chunk_size):
    """``(shard, first_position, rows)`` build tasks; no task crosses a shard."""
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError('chunk size must be positive')
    tasks = []
    for shard, (first, rows) in enumerate(shard_bounds(shard_size, total_positions)):
        offset = 0
        while offset < rows:
            take = min(chunk_size, rows - offset)
            tasks.append((shard, first + offset, take))
            offset += take
    return tasks


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def create_shard_files(root, shard_size, total_positions):
    """Create every shard file with its final ``.npy`` header, if it is absent.

    Creating the header in the driver rather than in a worker removes the only
    race in the layout: two workers that both saw a missing file would otherwise
    both truncate it, and the second one would win.
    """
    root = Path(root)
    (root / SHARDS_DIRNAME).mkdir(parents=True, exist_ok=True)
    created = []
    for path, (_, rows) in zip(shard_paths(root, shard_size, total_positions),
                               shard_bounds(shard_size, total_positions)):
        if path.is_file():
            continue
        handle = np.lib.format.open_memmap(path, mode='w+', dtype=DESCRIPTOR_DTYPE,
                                           shape=(rows, len(ROUTER_RADII), DESCRIPTOR_COLUMNS))
        handle.flush()
        del handle
        created.append(str(path))
    return created


def write_manifest(root, manifest):
    """Atomic manifest write: a reader never sees a half-written file."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / (MANIFEST_NAME + '.tmp')
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n',
                         encoding='utf-8')
    os.replace(temporary, root / MANIFEST_NAME)


def describe_shard(root, shard, first_position, rows):
    path = Path(root) / SHARDS_DIRNAME / shard_file_name(first_position, rows)
    return {'shard': int(shard), 'file': path.name, 'first_position': int(first_position),
            'rows': int(rows), 'sha256': sha256_file(path)}


def segment_key(shard, first_position):
    """The identity of one segment: its shard and its first position."""
    return int(shard), int(first_position)


def segment_marker_path(root, shard, first_position):
    name = f'segment_{int(shard):05d}_{int(first_position):09d}.json'
    return Path(root) / BUILD_STATE_DIRNAME / SEGMENTS_DIRNAME / name


def write_segment_marker(root, shard, first_position, rows):
    """Record, atomically, that one segment's rows are written.

    The marker is written by the process that wrote the rows, after the mapping
    was flushed, so its presence is evidence about the shard file -- not about
    the driver, whose own bookkeeping may be lost.
    """
    path = segment_marker_path(root, shard, first_position)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'shard': int(shard), 'first_position': int(first_position), 'rows': int(rows),
               'finished_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
               'pid': os.getpid()}
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)
    return payload


def read_segment_markers(root):
    """Every written segment marker of a cache, keyed by ``(shard, first_position)``."""
    directory = Path(root) / BUILD_STATE_DIRNAME / SEGMENTS_DIRNAME
    if not directory.is_dir():
        return {}
    markers = {}
    for path in sorted(directory.glob('*.json')):
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise ValueError(f'segment marker {path.name} is unreadable: {exc}') from exc
        missing = [name for name in MARKER_FIELDS if name not in payload]
        if missing:
            raise ValueError(f'segment marker {path.name} is missing: ' + ','.join(missing))
        key = segment_key(payload['shard'], payload['first_position'])
        if int(payload['rows']) <= 0:
            raise ValueError(f'segment marker {path.name} declares {payload["rows"]} rows')
        if key in markers:
            raise ValueError(f'segment marker {path.name} repeats {key}')
        markers[key] = payload
    return markers


def adopt_written_segments(root, shard_size, total_positions, *, chunk_size, log=None):
    """Mark the segments of an existing shard set whose rows are already written.

    A shard file is created as a sparse zero-filled ``.npy`` and every row is
    written exactly once by the segment that owns it, so a row that is still all
    zeros was never written -- unless the online path legitimately stores the
    declared zero trajectory of an invalid geometry.  Adoption therefore accepts
    a segment only when *all* of its rows are non-zero and reports the rest as
    missing, which can only turn a written row into a rebuild, never the other
    way round.  It exists to recover a build whose driver lost its bookkeeping.
    """
    root = Path(root)
    bounds = shard_bounds(shard_size, total_positions)
    handles, adopted, missing, zero_rows, segments = {}, 0, [], 0, 0
    for shard, first_position, rows in position_chunks(shard_size, total_positions, chunk_size):
        segments += 1
        index, offset = shard, int(first_position) - bounds[shard][0]
        if shard not in handles:
            path = root / SHARDS_DIRNAME / shard_file_name(*bounds[shard])
            if not path.is_file():
                raise FileNotFoundError(f'trajectory cache shard is missing: {path}')
            handles[shard] = np.load(path, mmap_mode='r', allow_pickle=False)
        block = handles[shard][offset:offset + int(rows)]
        zeros = int(np.count_nonzero(~np.asarray(block).any(axis=(1, 2))))
        if zeros:
            zero_rows += zeros
            if len(missing) < 16:
                missing.append({'shard': int(index), 'first_position': int(first_position),
                                'rows': int(rows), 'zero_rows': zeros})
            continue
        write_segment_marker(root, shard, first_position, rows)
        adopted += 1
    for handle in handles.values():
        del handle
    summary = {'segments': segments, 'adopted': adopted, 'missing_count': segments - adopted,
               'missing': missing, 'zero_rows': zero_rows}
    if log is not None:
        log(f'=== ADOPTED {adopted}/{segments} written segment(s); '
            f'{summary["missing_count"]} still to build, {zero_rows} zero row(s)')
    return summary


class MCLPHTrajectoryCache:
    """Read-only ``position -> [31,5] float32`` view over a built shard set.

    Shards are opened lazily with ``mmap_mode='r'`` and re-opened after a fork,
    so each DataLoader worker owns its own mapping and the 3 GB is never pickled
    or materialized as one array.  The first open of a shard verifies its content
    hash against the manifest, which turns silent corruption into a hard error at
    the moment the corrupted range is first needed.
    """

    def __init__(self, root, *, expected=None, required_positions=None,
                 verify_checksums=True):
        self.root = Path(root)
        manifest_path = self.root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f'trajectory cache manifest is missing: {manifest_path}')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('schema') != CACHE_SCHEMA:
            raise ValueError(f"trajectory cache schema is {manifest.get('schema')!r}, "
                             f'not {CACHE_SCHEMA!r}')
        if str(manifest.get('dtype')) != np.dtype(DESCRIPTOR_DTYPE).name:
            raise ValueError(f"trajectory cache dtype is {manifest.get('dtype')!r}, "
                             f'not {np.dtype(DESCRIPTOR_DTYPE).name!r}')
        shape = tuple(int(value) for value in manifest.get('shape_per_item', ()))
        declared = (len(ROUTER_RADII), DESCRIPTOR_COLUMNS)
        if shape != declared:
            raise ValueError(f'trajectory cache shape {shape} is not {declared}')
        self.manifest = manifest
        self.total_positions = int(manifest['total_positions'])
        self.shard_size = int(manifest['shard_size'])
        self.shape = shape
        self.verify_checksums = bool(verify_checksums)
        self._bounds = shard_bounds(self.shard_size, self.total_positions)
        entries = manifest.get('shards')
        if not isinstance(entries, list) or len(entries) > len(self._bounds):
            raise ValueError('trajectory cache manifest shard list is malformed')
        self._entries = {}
        for entry in entries:
            missing = [name for name in SHARD_ROW_FIELDS if name not in entry]
            if missing:
                raise ValueError('trajectory cache shard row is missing: ' + ','.join(missing))
            index = int(entry['shard'])
            if index in self._entries or not 0 <= index < len(self._bounds):
                raise ValueError(f'trajectory cache manifest repeats shard {index}')
            if int(entry['first_position']) != self._bounds[index][0] \
                    or int(entry['rows']) != self._bounds[index][1]:
                raise ValueError(f'trajectory cache shard {index} disagrees with the layout')
            self._entries[index] = entry
        self.completed_shards = len(self._entries)
        self.complete = self.completed_shards == len(self._bounds)
        self._handles = {}
        self._pid = os.getpid()
        if expected is not None:
            self.require_identity(expected)
        if required_positions is not None:
            self.require_positions(required_positions)

    # -- identity and coverage ------------------------------------------------

    def require_identity(self, expected):
        """Refuse a cache that does not describe the same presentations as this run."""
        problems = []
        for name in IDENTITY_FIELDS:
            if name not in expected:
                raise ValueError(f'the expected cache identity does not declare {name}')
            actual = self.manifest.get(name, '<missing>')
            if actual != expected[name]:
                problems.append(f'{name}: cache {actual!r} != run {expected[name]!r}')
        if problems:
            raise ValueError('trajectory cache identity mismatch: ' + '; '.join(problems))
        return True

    def require_positions(self, count):
        """Refuse a run whose positions are not covered by *completed* shards."""
        count = int(count)
        if count > self.total_positions:
            raise ValueError(f'the run needs {count} positions but the trajectory cache '
                             f'holds {self.total_positions}')
        needed = {int(value) // self.shard_size for value in range(count)}
        missing = sorted(needed - set(self._entries))
        if missing:
            raise ValueError(f'the trajectory cache is incomplete over the requested range: '
                             f'shards {missing} are not built')
        return True

    def provenance(self, *, required_positions=None, path=None):
        """The record written into ``run.json`` / ``runtime.json``."""
        block = {'mode': 'cached', 'path': str(path if path is not None else self.root),
                 'schema': self.manifest['schema'], 'dtype': self.manifest['dtype'],
                 'shape_per_item': list(self.shape),
                 'total_positions': self.total_positions,
                 'shard_size': self.shard_size,
                 'shards': len(self._bounds), 'completed_shards': self.completed_shards,
                 'complete': bool(self.complete),
                 'manifest_identity': {name: self.manifest.get(name) for name in IDENTITY_FIELDS}}
        if required_positions is not None:
            block['required_positions'] = int(required_positions)
        return block

    # -- reading --------------------------------------------------------------

    def __len__(self):
        return self.total_positions

    def shard_path(self, shard):
        first, rows = self._bounds[int(shard)]
        return self.root / SHARDS_DIRNAME / shard_file_name(first, rows)

    def _handle(self, shard):
        if os.getpid() != self._pid:
            # A forked worker must not share the parent's mapping bookkeeping.
            self._handles = {}
            self._pid = os.getpid()
        handle = self._handles.get(shard)
        if handle is not None:
            return handle
        entry = self._entries.get(shard)
        if entry is None:
            raise ValueError(f'trajectory cache shard {shard} is not built')
        path = self.root / SHARDS_DIRNAME / str(entry['file'])
        if not path.is_file():
            raise FileNotFoundError(f'trajectory cache shard is missing: {path}')
        handle = np.load(path, mmap_mode='r', allow_pickle=False)
        expected = (int(entry['rows']),) + self.shape
        if handle.shape != expected:
            raise ValueError(f'trajectory cache shard {path.name} has shape {handle.shape}, '
                             f'expected {expected}')
        if handle.dtype != DESCRIPTOR_DTYPE:
            raise ValueError(f'trajectory cache shard {path.name} has dtype {handle.dtype}')
        if self.verify_checksums and sha256_file(path) != str(entry['sha256']):
            raise ValueError(f'trajectory cache shard {path.name} does not match its '
                             f'manifest hash')
        self._handles[shard] = handle
        return handle

    def __getitem__(self, position):
        index = int(position)
        if index < 0 or index >= self.total_positions:
            raise IndexError(f'position {index} is outside the trajectory cache '
                             f'(0..{self.total_positions - 1})')
        shard, offset = divmod(index, self.shard_size)
        # A writable copy: the caller hands the row to ``torch.from_numpy``, and
        # the mapping itself must stay read-only.
        return np.array(self._handle(shard)[offset], dtype=DESCRIPTOR_DTYPE, copy=True)

    def close(self):
        self._handles = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_handles'] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._handles = {}
        self._pid = os.getpid()


# ---------------------------------------------------------------------------
# Offline construction
# ---------------------------------------------------------------------------

def position_trajectory(source, stream, *, position, seed, sigma, ratio, identity=None):
    """The exact ``[31,5]`` trajectory the online path stores for one position.

    This is the whole point of the cache: it takes the same route the training
    stream takes -- ``build_mcl_ph_view`` draws the motif mask and then the
    coordinate noise from the shared stream -- and then stops at the trajectory,
    before the expert relations, the targets and the model.
    """
    index = stream.index_at(int(position))
    topology, trimer, smiles = source[index]
    static = source.static_for(index)
    key = source.samples[index][0].hex()
    shared = build_mcl_ph_view(topology, trimer, smiles, seed=seed, key=key,
                               position=int(position), sigma=sigma, ratio=ratio,
                               identity=identity)
    if not bool(static['geometry_valid']):
        # With a frozen static row the online ``geometry_valid`` *is* this flag,
        # so an invalid row stores the declared all-zero trajectory and never
        # enters the descriptor path.
        return np.zeros((len(ROUTER_RADII), DESCRIPTOR_COLUMNS), dtype=DESCRIPTOR_DTYPE)
    validate_geometry_carrier(topology, trimer)
    from .glt_dual_static import materialize_dual_geometry

    # The same raise conditions the online path hits on its way to the expert
    # relations: a presentation that would stop the training run stops the cache
    # build instead of being stored as something else.
    materialize_dual_geometry(static, shared.changed.trimer_pos)
    return five_descriptors(shared.field_view.positions.numpy()).astype(DESCRIPTOR_DTYPE)


class TrajectoryCacheBuilder:
    """One cache-building process: the frozen source plus one shard writer.

    A builder instance is opened once per worker process.  It resolves the same
    subset, the same stream and the same static rows the training run uses, so a
    range of positions written by any worker holds the bytes the training process
    would have computed for them.
    """

    def __init__(self, *, cohort_root, cache_root, dual_static_root, sample_index_artifact,
                 sample_index_split, output, shard_size, total_positions, seed, sigma,
                 ratio, identity):
        from src.training.glt_dual_runtime import (IndexedFrozenDualSource,
                                                   load_sample_index_artifact,
                                                   open_source, OrderedSampleStream)

        split = load_sample_index_artifact(sample_index_artifact, sample_index_split)
        if split is None:
            raise ValueError('the trajectory cache is built on the frozen P_train subset')
        self.split_sha256 = str(split['sha256'])
        self.source_base, _ = open_source(cohort_root, cache_root,
                                          dual_static_root=dual_static_root)
        self.source = IndexedFrozenDualSource(self.source_base, split['indices'])
        self.output = Path(output)
        self.shard_size = int(shard_size)
        self.total_positions = int(total_positions)
        self.seed = int(seed)
        self.sigma = float(sigma)
        self.ratio = ratio
        self.identity = dict(identity)
        self.bounds = shard_bounds(self.shard_size, self.total_positions)
        self.stream = OrderedSampleStream(len(self.source), self.seed)

    def close(self):
        self.source_base.close()

    def build_chunk(self, shard, first_position, positions):
        """Fill ``positions`` rows of one shard; writers touch disjoint row ranges.

        The segment marker is written only after the rows are flushed, so a
        segment whose computation raised keeps no marker and is rebuilt rather
        than trusted.
        """
        shard = int(shard)
        first, rows = self.bounds[shard]
        path = self.output / SHARDS_DIRNAME / shard_file_name(first, rows)
        if not path.is_file():
            raise FileNotFoundError(f'trajectory cache shard file was not created: {path}')
        offset = int(first_position) - first
        if offset < 0 or offset + int(positions) > rows:
            raise ValueError(f'chunk {first_position}+{positions} leaves shard {shard}')
        handle = np.load(path, mmap_mode='r+', allow_pickle=False)
        try:
            if handle.shape != (rows, len(ROUTER_RADII), DESCRIPTOR_COLUMNS):
                raise ValueError(f'{path.name} has an unexpected shape {handle.shape}')
            for step in range(int(positions)):
                handle[offset + step] = self._trajectory(int(first_position) + step)
            handle.flush()
        finally:
            del handle
        write_segment_marker(self.output, shard, first_position, positions)
        return int(positions)

    def _trajectory(self, position):
        return position_trajectory(self.source, self.stream, position=position,
                                   seed=self.seed, sigma=self.sigma, ratio=self.ratio)
