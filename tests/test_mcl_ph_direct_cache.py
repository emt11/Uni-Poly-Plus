from types import SimpleNamespace
import numpy as np
import pytest
from src.dataset.mcl_ph_trajectory_cache import (
    MCLPHTrajectoryCache, cache_identity, create_shard_files, describe_shard,
    require_direct_source_rows, write_manifest, CACHE_SCHEMA)


class Source:
    def __init__(self, keys=(b'a'*32, b'b'*32)):
        self.samples = [(key, 'fixture') for key in keys]
        self.entries = [{'pretrain_role': 'train'} for key in keys]
    def __len__(self): return len(self.samples)
    def source_index_at(self, index): return index


def test_direct_cache_rejects_same_count_different_keys_and_roles(tmp_path):
    source = Source()
    require_direct_source_rows(tmp_path, source, publish=True)
    require_direct_source_rows(tmp_path, source)
    with pytest.raises(ValueError, match='binding differs'):
        require_direct_source_rows(tmp_path, Source((b'c'*32, b'b'*32)))
    source.entries[0]['pretrain_role'] = 'validation'
    with pytest.raises(ValueError, match='non-train'):
        require_direct_source_rows(tmp_path, source)


def test_direct_cache_has_no_digest_and_rejects_nonfinite_values(tmp_path):
    identity = cache_identity(seed=42, noise_sigma=.03, mask_ratio=.3, global_batch=1008,
        sample_index_split='train', direct_source=dict(source_root='/frozen/fixture', source_rows=2, selected_rows=2))
    assert not any('hash' in field or 'sha256' in field for field in identity)
    create_shard_files(tmp_path, 2, 2)
    path = next((tmp_path/'shards').glob('*.npy'))
    data = np.load(path, mmap_mode='r+'); data[:] = 1; data.flush()
    entry = describe_shard(tmp_path, 0, 0, 2, direct=True)
    assert 'sha256' not in entry
    write_manifest(tmp_path, dict(schema=CACHE_SCHEMA, dtype='float32', shape_per_item=[31,5], total_positions=2, shard_size=2, shards=[entry], **identity))
    cache = MCLPHTrajectoryCache(tmp_path, expected=identity, verify_checksums=False)
    np.testing.assert_array_equal(cache[0], np.ones((31,5), dtype=np.float32))
    data[0,0,0] = np.nan; data.flush()
    with pytest.raises(ValueError, match='NaN/Inf'):
        cache[0]
    with pytest.raises(ValueError, match='finite values'):
        describe_shard(tmp_path, 0, 0, 2, direct=True)
    cache.close()


def test_missing_legacy_digest_is_rejected():
    with pytest.raises(ValueError, match='missing a digest'):
        cache_identity(seed=42, noise_sigma=.03, mask_ratio=.3, global_batch=1008, sample_index_split='train')


@pytest.mark.parametrize('bad_field,bad_values', [
    ('train_source_indices', np.array([0., 2.])),
    ('train_source_indices', np.array([2, 0], dtype=np.int64)),
    ('train_source_indices', np.array([0, 3], dtype=np.int64)),
    ('validation_source_indices', np.array([0], dtype=np.int64)),
    ('fixed_validation_source_indices', np.array([2], dtype=np.int64)),
])
def test_direct_index_rejects_float_range_order_and_leakage(tmp_path, bad_field, bad_values):
    import json
    from src.training.glt_dual_runtime import load_sample_index_artifact
    summary = dict(schema='mcl-ph-next-direct-index-v1', pi1m_rows=3)
    path = tmp_path/'summary.json'; path.write_text(json.dumps(summary))
    payload = dict(train_source_indices=np.array([0,2], dtype=np.int64),
                   validation_source_indices=np.array([1], dtype=np.int64),
                   fixed_validation_source_indices=np.array([1], dtype=np.int64))
    payload[bad_field] = bad_values
    np.savez(tmp_path/'pretrain_indices.npz', **payload)
    with pytest.raises(ValueError):
        load_sample_index_artifact(path, 'train')
