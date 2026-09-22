"""Contract tests for the offline noisy-topology trajectory cache (r10R2).

Two layers.  The layout, identity and fail-closed behaviour of the reader are
checked against a synthetic cache written by the same functions the real builder
uses, so they run on CPU in seconds and never depend on the frozen cohort.  The
equivalence layer -- cached trajectory vs the online path, and worker-count
invariance -- needs the real frozen cohort and is skipped (reported as skipped,
never as covered) when it is absent.
"""
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from src.dataset.mcl_ph_trajectory_cache import (BUILD_STATE_DIRNAME, CACHE_SCHEMA,
                                                 DESCRIPTOR_DTYPE, IDENTITY_FIELDS,
                                                 SEGMENTS_DIRNAME, SHARDS_DIRNAME,
                                                 MCLPHTrajectoryCache,
                                                 adopt_written_segments, cache_identity,
                                                 create_shard_files, describe_shard,
                                                 position_chunks, read_segment_markers,
                                                 segment_key, shard_bounds, shard_file_name,
                                                 write_manifest, write_segment_marker)
from src.dataset.mcl_ph_view import (DESCRIPTOR_COLUMNS, ROUTER_RADII,
                                     cached_topology_trajectory)
import scripts.build_mcl_ph_trajectory_cache as cli

SHAPE = (len(ROUTER_RADII), DESCRIPTOR_COLUMNS)
ROWS = 10
SHARD_SIZE = 4

BASE_IDENTITY = {'seed': 42, 'noise_sigma': 0.03, 'mask_ratio': 0.3, 'global_batch': 1008,
                 'sample_index_split': 'train', 'sample_index_artifact_sha256': 'a' * 64,
                 'cohort_manifest_hash': 'b' * 64, 'dual_static_manifest_hash': 'c' * 64}


def row_value(shard, first, index):
    """A distinct, exactly representable float32 per position."""
    position = first + index
    return np.full(SHAPE, np.float32(position) + np.float32(0.5), dtype=np.float32)


def write_cache(root, *, rows=ROWS, shard_size=SHARD_SIZE, identity=None, fill=row_value,
                dtype='float32', schema=CACHE_SCHEMA, shards=None, complete=True):
    root = Path(root)
    create_shard_files(root, shard_size, rows)
    manifest = dict(schema=schema, dtype=dtype, shape_per_item=list(SHAPE),
                    total_positions=int(rows), shard_size=int(shard_size),
                    max_updates=1, requested_total_positions=int(rows),
                    router_radii=list(ROUTER_RADII), descriptor_columns=DESCRIPTOR_COLUMNS,
                    builder='test', complete=bool(complete), shards=[])
    manifest.update(dict(identity or BASE_IDENTITY))
    entries = []
    for shard, (first, count) in enumerate(shard_bounds(shard_size, rows)):
        path = root / 'shards' / shard_file_name(first, count)
        handle = np.load(path, mmap_mode='r+', allow_pickle=False)
        for index in range(count):
            handle[index] = fill(shard, first, index)
        handle.flush()
        del handle
        if shards is None or shard in shards:
            entries.append(describe_shard(root, shard, first, count))
    manifest['shards'] = entries
    write_manifest(root, manifest)
    return manifest


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def test_shard_bounds_cover_every_position_exactly_once():
    bounds = shard_bounds(100_000, 5_040_000)
    assert len(bounds) == 51
    assert bounds[0] == (0, 100_000)
    assert bounds[-1] == (5_000_000, 40_000)
    assert sum(rows for _, rows in bounds) == 5_040_000
    assert [first for first, _ in bounds] == sorted(first for first, _ in bounds)


def test_position_chunks_never_cross_a_shard_boundary():
    tasks = position_chunks(4, ROWS, 3)
    assert tasks == [(0, 0, 3), (0, 3, 1), (1, 4, 3), (1, 7, 1), (2, 8, 2)]
    for shard, first, rows in tasks:
        bound_first, bound_rows = shard_bounds(4, ROWS)[shard]
        assert bound_first <= first and first + rows <= bound_first + bound_rows


# ---------------------------------------------------------------------------
# Reader: values, dtype and lazy opening
# ---------------------------------------------------------------------------

def test_reader_returns_the_declared_float32_rows(tmp_path):
    write_cache(tmp_path)
    cache = MCLPHTrajectoryCache(tmp_path)
    for position in range(ROWS):
        value = cache[position]
        assert value.shape == SHAPE
        assert value.dtype == np.float32
        assert value.flags.writeable, 'a cache row must be writable for torch.from_numpy'
        assert np.array_equal(value, np.full(SHAPE, np.float32(position) + np.float32(0.5),
                                             dtype=np.float32))
    assert cache.total_positions == ROWS and cache.complete


def test_reader_rejects_a_position_outside_the_cache(tmp_path):
    write_cache(tmp_path)
    cache = MCLPHTrajectoryCache(tmp_path)
    for position in (-1, ROWS):
        with pytest.raises(IndexError):
            cache[position]


def test_reader_survives_pickle_and_drops_its_mappings(tmp_path):
    write_cache(tmp_path)
    cache = MCLPHTrajectoryCache(tmp_path)
    assert cache[0] is not None
    assert cache._handles
    restored = pickle.loads(pickle.dumps(cache))
    assert restored._handles == {}
    assert np.array_equal(restored[ROWS - 1], cache[ROWS - 1])


def test_reader_opens_shards_lazily(tmp_path):
    write_cache(tmp_path)
    cache = MCLPHTrajectoryCache(tmp_path)
    cache[3]
    assert set(cache._handles) == {0}
    cache[9]
    assert set(cache._handles) == {0, 2}


# ---------------------------------------------------------------------------
# Reader: identity and coverage (fail closed, never fallback)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('field, value', [
    ('seed', 41),
    ('noise_sigma', 0.031),
    ('mask_ratio', 0.5),
    ('global_batch', 512),
    ('cohort_manifest_hash', 'd' * 64),
    ('dual_static_manifest_hash', 'e' * 64),
    ('sample_index_artifact_sha256', 'f' * 64),
    ('sample_index_split', 'validation'),
])
def test_reader_rejects_an_identity_mismatch(tmp_path, field, value):
    write_cache(tmp_path)
    expected = dict(BASE_IDENTITY)
    expected[field] = value
    with pytest.raises(ValueError, match='identity mismatch'):
        MCLPHTrajectoryCache(tmp_path, expected=expected)


def test_reader_accepts_the_matching_identity(tmp_path):
    write_cache(tmp_path)
    assert MCLPHTrajectoryCache(tmp_path, expected=dict(BASE_IDENTITY)).total_positions == ROWS


def test_identity_helper_refuses_an_empty_field():
    with pytest.raises(ValueError, match='is empty'):
        cache_identity(**{**BASE_IDENTITY, 'cohort_manifest_hash': ''})
    assert cache_identity(**BASE_IDENTITY).keys() == set(IDENTITY_FIELDS)


def test_reader_rejects_a_run_that_needs_more_positions(tmp_path):
    write_cache(tmp_path)
    cache = MCLPHTrajectoryCache(tmp_path)
    with pytest.raises(ValueError, match='needs'):
        cache.require_positions(ROWS + 1)
    assert cache.require_positions(ROWS)


def test_reader_rejects_positions_covered_only_by_an_unbuilt_shard(tmp_path):
    write_cache(tmp_path, shards=(0,), complete=False)
    cache = MCLPHTrajectoryCache(tmp_path)
    assert cache.completed_shards == 1 and cache.complete is False
    assert cache.require_positions(4)
    with pytest.raises(ValueError, match='incomplete'):
        cache.require_positions(ROWS)
    assert np.array_equal(cache[0], row_value(0, 0, 0))
    with pytest.raises(ValueError, match='is not built'):
        cache[ROWS - 1]


def test_reader_rejects_a_missing_shard(tmp_path):
    write_cache(tmp_path)
    (tmp_path / 'shards' / shard_file_name(4, 4)).unlink()
    cache = MCLPHTrajectoryCache(tmp_path)
    with pytest.raises(FileNotFoundError):
        cache[4]


def test_reader_rejects_a_corrupt_shard(tmp_path):
    write_cache(tmp_path)
    path = tmp_path / 'shards' / shard_file_name(0, 4)
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 0xFF
    path.write_bytes(bytes(payload))
    cache = MCLPHTrajectoryCache(tmp_path)
    with pytest.raises(ValueError, match='manifest hash'):
        cache[0]


def test_reader_rejects_a_truncated_shard(tmp_path):
    write_cache(tmp_path)
    path = tmp_path / 'shards' / shard_file_name(0, 4)
    path.write_bytes(path.read_bytes()[:-64])
    cache = MCLPHTrajectoryCache(tmp_path)
    with pytest.raises((ValueError, OSError)):
        cache[0]


def test_reader_rejects_a_manifest_that_lies_about_shape_or_dtype(tmp_path):
    write_cache(tmp_path, dtype='float64')
    with pytest.raises(ValueError, match='dtype'):
        MCLPHTrajectoryCache(tmp_path)
    write_cache(tmp_path / 'second', schema='mcl-ph-noisy-trajectory-cache-v0')
    with pytest.raises(ValueError, match='schema'):
        MCLPHTrajectoryCache(tmp_path / 'second')


def test_reader_rejects_a_shard_that_disagrees_with_the_layout(tmp_path):
    manifest = write_cache(tmp_path)
    manifest['shards'][0]['rows'] = 3
    write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match='disagrees with the layout'):
        MCLPHTrajectoryCache(tmp_path)


def test_reader_rejects_a_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        MCLPHTrajectoryCache(tmp_path / 'absent')


def test_provenance_records_the_manifest_identity(tmp_path):
    write_cache(tmp_path)
    cache = MCLPHTrajectoryCache(tmp_path)
    block = cache.provenance(required_positions=ROWS)
    assert block['mode'] == 'cached' and block['schema'] == CACHE_SCHEMA
    assert block['manifest_identity'] == BASE_IDENTITY
    assert block['required_positions'] == ROWS and block['complete'] is True
    payload = json.loads(json.dumps(block))
    assert payload['manifest_identity']['dual_static_manifest_hash'] == 'c' * 64


# ---------------------------------------------------------------------------
# The override a sample accepts
# ---------------------------------------------------------------------------

def test_cached_trajectory_validation_is_strict():
    good = np.zeros(SHAPE, dtype=np.float32)
    assert cached_topology_trajectory(good) is good
    with pytest.raises(ValueError, match='shape'):
        cached_topology_trajectory(np.zeros((30, 5), dtype=np.float32))
    with pytest.raises(ValueError, match='dtype'):
        cached_topology_trajectory(np.zeros(SHAPE, dtype=np.float64))
    with pytest.raises(ValueError, match='dtype'):
        cached_topology_trajectory(np.zeros(SHAPE, dtype=np.float16))
    bad = np.zeros(SHAPE, dtype=np.float32)
    bad[7, 3] = np.inf
    with pytest.raises(ValueError, match='NaN/Inf'):
        cached_topology_trajectory(bad)


# ---------------------------------------------------------------------------
# Equivalence against the online path (needs the frozen cohort)
# ---------------------------------------------------------------------------

COHORT = ROOT / 'data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1'
CACHE = ROOT / 'data/processed/mips_trimer_scage'
STATIC = ROOT / 'data/processed/glt_dual_v2/pi1m/dual_static_v1'
SPLIT = ROOT / 'results/glt_pred_20260918/s3b_prep/pretrain_split_v1.json'
STATS = ROOT / 'results/mcl_ph_20260921/p0/statistics.npz'
FROZEN = all(path.exists() for path in (COHORT, CACHE, STATIC, SPLIT, STATS))

POSITIONS = (0, 1, 500, 1007)


def _open_frozen():
    from src.training.glt_dual_runtime import (IndexedFrozenDualSource,
                                               load_sample_index_artifact, open_source)

    split = load_sample_index_artifact(str(SPLIT), 'train')
    source, _ = open_source(str(COHORT), str(CACHE), dual_static_root=str(STATIC))
    return source, IndexedFrozenDualSource(source, split['indices'])


@pytest.mark.skipif(not FROZEN, reason='frozen P_train cache is not available in this checkout')
def test_cached_trajectory_equals_the_online_trajectory_at_real_positions():
    import torch

    from src.dataset.mcl_ph_trajectory_cache import position_trajectory
    from src.dataset.mcl_ph_view import load_geometric_statistics, prepare_mcl_ph_sample
    from src.training.glt_dual_runtime import OrderedSampleStream

    statistics = load_geometric_statistics(STATS)
    source, subset = _open_frozen()
    stream = OrderedSampleStream(len(subset), 42)
    try:
        for position in POSITIONS:
            cached = position_trajectory(subset, stream, position=position, seed=42,
                                         sigma=0.03, ratio=0.3)
            index = stream.index_at(position)
            key = subset.samples[index][0].hex()
            online, online_labels = prepare_mcl_ph_sample(
                *subset[index], seed=42, key=key, position=position, sigma=0.03, ratio=0.3,
                static=subset.static_for(index), statistics=statistics)
            assert np.array_equal(cached, online.mcl_trajectory.numpy())
            with_override, override_labels = prepare_mcl_ph_sample(
                *subset[index], seed=42, key=key, position=position, sigma=0.03, ratio=0.3,
                static=subset.static_for(index), statistics=statistics,
                trajectory_override=cached)
            assert torch.equal(with_override.mcl_trajectory, online.mcl_trajectory)
            assert torch.equal(with_override.mcl_pos, online.mcl_pos)
            assert torch.equal(with_override.mcl_masked, online.mcl_masked)
            assert torch.equal(with_override.mcl_edge_index, online.mcl_edge_index)
            assert torch.equal(with_override.mcl_edge_distance, online.mcl_edge_distance)
            assert torch.equal(with_override.mcl_edge_type, online.mcl_edge_type)
            for name, value in online_labels.items():
                if torch.is_tensor(value):
                    assert torch.equal(override_labels[name], value), name
    finally:
        source.close()


@pytest.mark.skipif(not FROZEN, reason='frozen P_train cache is not available in this checkout')
def test_a_rejected_override_is_never_silently_recomputed():
    import torch

    from src.dataset.mcl_ph_view import load_geometric_statistics, prepare_mcl_ph_sample
    from src.training.glt_dual_runtime import OrderedSampleStream

    statistics = load_geometric_statistics(STATS)
    source, subset = _open_frozen()
    stream = OrderedSampleStream(len(subset), 42)
    try:
        position = POSITIONS[0]
        index = stream.index_at(position)
        arguments = dict(seed=42, key=subset.samples[index][0].hex(), position=position,
                         sigma=0.03, ratio=0.3, static=subset.static_for(index),
                         statistics=statistics)
        for bad in (np.zeros((30, 5), dtype=np.float32),
                    np.zeros(SHAPE, dtype=np.float64),
                    np.full(SHAPE, np.nan, dtype=np.float32)):
            with pytest.raises(ValueError):
                prepare_mcl_ph_sample(*subset[index], trajectory_override=bad, **arguments)
        # An invalid-geometry row stores the declared zeros, so a cached row that
        # claims anything else is a mismatch between the cache and this run.
        invalid = dict(arguments['static'])
        invalid['geometry_valid'] = False
        invalid['geometry_invalid_reason'] = 'geometry_invalid'
        invalid_arguments = {**arguments, 'static': invalid}
        zeros = np.zeros(SHAPE, dtype=np.float32)
        data, labels = prepare_mcl_ph_sample(*subset[index], trajectory_override=zeros,
                                             **invalid_arguments)
        assert not bool(labels['geometry_valid'])
        assert torch.equal(data.mcl_trajectory, torch.from_numpy(zeros))
        with pytest.raises(ValueError, match='invalid geometry'):
            prepare_mcl_ph_sample(*subset[index],
                                  trajectory_override=np.full(SHAPE, 0.25, dtype=np.float32),
                                  **invalid_arguments)
    finally:
        source.close()


@pytest.mark.skipif(not FROZEN, reason='frozen P_train cache is not available in this checkout')
def test_worker_count_does_not_change_a_presented_sample():
    import torch

    from src.dataset.mcl_ph_view import MCLPHMicrobatchStream, load_geometric_statistics

    statistics = load_geometric_statistics(STATS)
    source, subset = _open_frozen()
    try:
        reference = None
        for workers in (0, 1, 4, 12):
            dataset = MCLPHMicrobatchStream(
                subset, seed=42, world=1, rank=0, microbatch=2, accumulation=1,
                start_step=0, max_steps=2, sigma=0.03, ratio=0.3, statistics=statistics)
            assert len(dataset) == 2
            if workers == 0:
                items = [dataset[item] for item in range(len(dataset))]
            else:
                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=None, num_workers=workers, prefetch_factor=2,
                    persistent_workers=False)
                items = list(loader)
            digests = []
            for batch, labels in items:
                digests.append((list(labels['sample_key']),
                                batch.mcl_trajectory.clone(),
                                batch.mcl_pos.clone(),
                                batch.mcl_masked.clone(),
                                batch.mcl_edge_index.clone(),
                                labels['mcl_nonbond_target'].clone(),
                                labels['atom_mask'].clone()))
            if reference is None:
                reference = digests
                continue
            assert len(digests) == len(reference)
            for (keys, trajectory, positions, mask, edges, nonbond, atom), expected in \
                    zip(digests, reference):
                assert keys == expected[0]
                assert torch.equal(trajectory, expected[1])
                assert torch.equal(positions, expected[2])
                assert torch.equal(mask, expected[3])
                assert torch.equal(edges, expected[4])
                assert torch.equal(nonbond, expected[5])
                assert torch.equal(atom, expected[6])
    finally:
        source.close()


# ---------------------------------------------------------------------------
# The build driver: segment markers, resume, recovery and failures
# ---------------------------------------------------------------------------

class _FakeBuilder:
    """A CPU-only stand-in for ``TrajectoryCacheBuilder`` writing synthetic rows.

    ``fail_at``  -- positions whose segment always raises inside ``build_chunk``.
    ``fail_once``-- a sentinel path: the first attempt raises and creates it, so a
                    later attempt through the driver's requeue succeeds.
    ``calls``    -- a path every built segment appends ``shard:first`` to, which is
                    how a test proves exactly which segments a run recomputed.
    """

    def __init__(self, **payload):
        self.output = Path(payload['output'])
        self.bounds = shard_bounds(payload['shard_size'], payload['total_positions'])
        self.fail_at = {int(value) for value in (payload.get('fail_at') or ())}
        self.fail_once = Path(payload['fail_once']) if payload.get('fail_once') else None
        self.calls = Path(payload['calls']) if payload.get('calls') else None

    def build_chunk(self, shard, first_position, positions):
        shard, first_position = int(shard), int(first_position)
        if self.calls is not None:
            with self.calls.open('a', encoding='utf-8') as handle:
                handle.write(f'{shard}:{first_position}\n')
        if first_position in self.fail_at:
            raise RuntimeError(f'synthetic failure at position {first_position}')
        if self.fail_once is not None and not self.fail_once.exists():
            self.fail_once.write_text('seen', encoding='utf-8')
            raise RuntimeError(f'synthetic transient failure at position {first_position}')
        bound_first, _ = self.bounds[shard]
        path = self.output / SHARDS_DIRNAME / shard_file_name(*self.bounds[shard])
        handle = np.load(path, mmap_mode='r+', allow_pickle=False)
        try:
            offset = first_position - bound_first
            for step in range(int(positions)):
                handle[offset + step] = row_value(shard, first_position, step)
            handle.flush()
        finally:
            del handle
        write_segment_marker(self.output, shard, first_position, positions)
        return int(positions)


def _build(root, *, rows=ROWS, shard_size=SHARD_SIZE, chunk=3, workers=2, **kwargs):
    """Drive a synthetic build through the production ``build_cache``."""
    log, extra = [], kwargs.pop('extra', {})
    payload = dict(output=str(root), shard_size=shard_size, total_positions=rows,
                   identity=BASE_IDENTITY, **extra)
    manifest = cli.build_cache(builder_payload=payload, root=root, shard_size=shard_size,
                               total_positions=rows, chunk_size=chunk, workers=workers,
                               requested_total_positions=rows, max_updates=1, log=log.append,
                               progress_seconds=999.0, poll_seconds=0.05, **kwargs)
    return manifest, log


@pytest.fixture()
def synthetic_builder(monkeypatch):
    def install(**payload_extra):
        monkeypatch.setattr(cli, 'BUILDER_CLASS', _FakeBuilder)
        return payload_extra

    return install


def test_driver_builds_and_marks_every_segment(tmp_path, synthetic_builder):
    calls = tmp_path / 'calls.txt'
    extra = synthetic_builder(calls=str(calls))
    manifest, _ = _build(tmp_path, extra=extra)
    assert manifest['complete'] and manifest['completed_shards'] == 3
    assert len(manifest['shards']) == 3
    markers = read_segment_markers(tmp_path)
    assert sorted(markers) == sorted((shard, first) for shard, first, _ in position_chunks(4, ROWS, 3))
    assert all(entry['rows'] > 0 for entry in markers.values())
    cache = MCLPHTrajectoryCache(tmp_path, required_positions=ROWS)
    assert cache.complete
    for position in range(ROWS):
        assert np.array_equal(cache[position],
                              np.full(SHAPE, np.float32(position) + np.float32(0.5),
                                      dtype=np.float32))
    assert len(calls.read_text(encoding='utf-8').split()) == len(markers)


def test_a_second_run_rebuilds_nothing(tmp_path, synthetic_builder):
    calls = tmp_path / 'calls.txt'
    extra = synthetic_builder(calls=str(calls))
    _build(tmp_path, extra=extra)
    before = calls.read_text(encoding='utf-8').split()
    manifest, log = _build(tmp_path, extra=extra)
    assert calls.read_text(encoding='utf-8').split() == before
    assert manifest['complete']
    assert any('NOTHING TO DO' in line for line in log)


def test_driver_recovers_a_manifest_lost_by_the_driver(tmp_path, synthetic_builder):
    calls = tmp_path / 'calls.txt'
    extra = synthetic_builder(calls=str(calls))
    _build(tmp_path, extra=extra)
    before = calls.read_text(encoding='utf-8').split()
    (tmp_path / 'manifest.json').unlink()
    manifest, log = _build(tmp_path, extra=extra)
    assert manifest['complete'] and manifest['completed_shards'] == 3
    assert calls.read_text(encoding='utf-8').split() == before, 'markers must not be rebuilt'
    assert any('RECOVERED' in line for line in log)
    MCLPHTrajectoryCache(tmp_path, required_positions=ROWS)


def test_a_recorded_shard_without_markers_is_rebuilt(tmp_path, synthetic_builder):
    calls = tmp_path / 'calls.txt'
    extra = synthetic_builder(calls=str(calls))
    _build(tmp_path, extra=extra)
    markers = tmp_path / BUILD_STATE_DIRNAME / SEGMENTS_DIRNAME
    for path in markers.glob('*.json'):
        path.unlink()
    manifest, _ = _build(tmp_path, extra=extra)
    assert manifest['complete']
    assert len(calls.read_text(encoding='utf-8').split()) == 2 * len(position_chunks(4, ROWS, 3))


def test_a_failing_segment_is_reported_without_hanging(tmp_path, synthetic_builder):
    extra = synthetic_builder(fail_at=[3])
    with pytest.raises(RuntimeError, match='stalled'):
        _build(tmp_path, extra=extra, max_requeues=0, stall_seconds=0.3)
    assert segment_key(0, 3) not in read_segment_markers(tmp_path)


def test_a_transient_failure_is_retried_and_the_build_completes(tmp_path, synthetic_builder):
    sentinel = tmp_path / 'transient.sentinel'
    extra = synthetic_builder(fail_once=str(sentinel))
    manifest, log = _build(tmp_path, extra=extra, max_requeues=1, stall_seconds=0.3)
    assert sentinel.exists()
    assert manifest['complete']
    assert any('SEGMENT FAILED' in line for line in log)
    assert any('STALLED' in line for line in log)
    MCLPHTrajectoryCache(tmp_path, required_positions=ROWS)


def test_adoption_marks_written_rows_and_leaves_the_rest_to_build(tmp_path, synthetic_builder):
    create_shard_files(tmp_path, SHARD_SIZE, ROWS)
    path = tmp_path / SHARDS_DIRNAME / shard_file_name(0, SHARD_SIZE)
    handle = np.load(path, mmap_mode='r+', allow_pickle=False)
    for index in range(3):
        handle[index] = row_value(0, 0, index)
    handle.flush()
    del handle
    summary = adopt_written_segments(tmp_path, SHARD_SIZE, ROWS, chunk_size=3)
    assert summary['adopted'] == 1 and summary['missing_count'] == 4
    assert summary['zero_rows'] > 0
    assert sorted(read_segment_markers(tmp_path)) == [segment_key(0, 0)]

    calls = tmp_path / 'calls.txt'
    extra = synthetic_builder(calls=str(calls))
    manifest, _ = _build(tmp_path, extra=extra, adopt_written=True)
    assert manifest['complete'] and manifest['adopted_segments'] == 1
    written = calls.read_text(encoding='utf-8').split()
    assert '0:0' not in written, 'an adopted segment must not be recomputed'
    assert len(written) == len(position_chunks(4, ROWS, 3)) - 1
    cache = MCLPHTrajectoryCache(tmp_path, required_positions=ROWS)
    assert np.array_equal(cache[0], np.full(SHAPE, np.float32(0.5), dtype=np.float32))
    assert np.array_equal(cache[3], np.full(SHAPE, np.float32(3.5), dtype=np.float32))


def test_a_single_zero_row_keeps_a_segment_unbuilt(tmp_path):
    create_shard_files(tmp_path, SHARD_SIZE, ROWS)
    path = tmp_path / SHARDS_DIRNAME / shard_file_name(0, SHARD_SIZE)
    handle = np.load(path, mmap_mode='r+', allow_pickle=False)
    for index in range(3):
        handle[index] = row_value(0, 0, index)
    handle[1] = np.zeros(SHAPE, dtype=np.float32)
    handle.flush()
    del handle
    summary = adopt_written_segments(tmp_path, SHARD_SIZE, ROWS, chunk_size=3)
    assert summary['adopted'] == 0 and summary['missing_count'] == 5
    assert read_segment_markers(tmp_path) == {}


def test_segment_markers_round_trip_and_reject_a_malformed_file(tmp_path):
    write_segment_marker(tmp_path, 2, 8, 2)
    markers = read_segment_markers(tmp_path)
    assert markers[segment_key(2, 8)]['rows'] == 2
    directory = tmp_path / BUILD_STATE_DIRNAME / SEGMENTS_DIRNAME
    broken = directory / 'segment_00003_000000012.json'
    broken.write_text(json.dumps({'shard': 3}), encoding='utf-8')
    with pytest.raises(ValueError, match='missing'):
        read_segment_markers(tmp_path)
    broken.write_text(json.dumps({'shard': 3, 'first_position': 12, 'rows': 0,
                                  'finished_at': 'now', 'pid': 1}), encoding='utf-8')
    with pytest.raises(ValueError, match='0 rows'):
        read_segment_markers(tmp_path)


def test_no_resume_discards_shards_and_markers(tmp_path, synthetic_builder):
    calls = tmp_path / 'calls.txt'
    extra = synthetic_builder(calls=str(calls))
    _build(tmp_path, extra=extra)
    manifest, _ = _build(tmp_path, extra=extra, resume=False)
    assert manifest['complete']
    assert len(calls.read_text(encoding='utf-8').split()) == 2 * len(position_chunks(4, ROWS, 3))
