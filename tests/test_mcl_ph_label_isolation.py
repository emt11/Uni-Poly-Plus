import json
import hashlib

import numpy as np

from scripts.finetune_mcl_ph import compact_train_validation_indices
import src.dataset.glt_dual_cache as dual_cache
import src.training.glt_dual_runtime as dual_runtime
from src.utils import scale_targets


class _Targets:
    def __init__(self, values):
        self.raw_targets = np.asarray(values, dtype=np.float64)
        self.targets = self.raw_targets.copy()

    def set_target_override(self, values):
        self.targets = np.asarray(values, dtype=np.float64)


def _frozen_cohort_with_opaque_outer_test(tmp_path, monkeypatch):
    cohort_root = tmp_path / 'downstream' / 'cohort'
    cohort_root.mkdir(parents=True)
    union_root = cohort_root.parent / 'union_outer5_inner20'
    union_root.mkdir()
    split_hash = 'fixed-split-sha256'
    normalized = [f'polymer-{index}' for index in range(4)]
    keys = [hashlib.sha256(value.encode()).digest() for value in normalized]
    key_array = np.frombuffer(b''.join(keys), dtype=np.uint8).reshape(-1, 32)
    np.save(cohort_root / 'keys.npy', key_array)

    rows = [
        {'task': 'xc', 'original_row': index, 'label': label,
         'sample_key': keys[index].hex(), 'source_smiles': normalized[index],
         'normalized_smiles': normalized[index]}
        for index, label in enumerate((10.0, None, 20.0, 14.0))
        if index != 1
    ]
    records_path = cohort_root / 'records.jsonl'
    records_path.write_bytes(
        (json.dumps(rows[0], separators=(',', ':')) + '\n').encode()
        + b'{"task":"xc","original_row":1,"label":THIS_OUTER_TEST_LABEL_MUST_NOT_BE_DECODED\xff}\n'
        + (json.dumps(rows[1], separators=(',', ':')) + '\n').encode()
        + (json.dumps(rows[2], separators=(',', ':')) + '\n').encode()
    )

    task_counts = {'xc': 4}
    union_manifest = {'task_order': ['xc'], 'task_counts': task_counts,
                      'row_count': 4, 'split_manifest_sha256': {'xc': split_hash}}
    (union_root / 'manifest.json').write_text(json.dumps(union_manifest))
    bindings = {
        'main_bundle_hash': 'b' * 64,
        'source_manifest_hash': 'source-hash',
        'topology_manifest_hash': 'topology-hash',
        'trimer_manifest_hash': 'trimer-hash',
    }
    manifest = {
        **bindings,
        'identity_policy': 'dataset_rows_with_reuse',
        'ordering_policy': 'task_order_then_original_row_preserving_fixed_property_rows',
        'sample_count': 4,
        'task_counts': task_counts,
        'split_manifest_sha256': {'xc': split_hash},
        'union_manifest_hash': dual_cache.json_hash(union_manifest),
        'keys_file_sha256': dual_cache.sha256_file(cohort_root / 'keys.npy'),
        'records_file_sha256': dual_cache.sha256_file(records_path),
        'ordered_sample_key_hash': dual_cache.ordered_key_hash(key_array),
        'duplicate_count': 0,
    }
    (cohort_root / 'manifest.json').write_text(json.dumps(manifest))
    (cohort_root / '.frozen').write_text(
        json.dumps({'manifest_hash': dual_cache.json_hash(manifest)})
    )
    monkeypatch.setattr(dual_cache, 'load_active_dual_store', lambda _: {
        'bundle_hash': bindings['main_bundle_hash'],
        'source': {'source_manifest_hash': bindings['source_manifest_hash']},
        'artifacts': {
            'topology': {'manifest_hash': bindings['topology_manifest_hash']},
            'trimer': {'manifest_hash': bindings['trimer_manifest_hash']},
        },
    })
    monkeypatch.setattr(dual_cache, 'sample_key_from_normalized',
                        lambda value: hashlib.sha256(value.encode()).digest())
    return cohort_root, split_hash, keys


def test_open_source_decodes_only_train_validation_labels(tmp_path, monkeypatch):
    cohort_root, split_hash, keys = _frozen_cohort_with_opaque_outer_test(
        tmp_path, monkeypatch)
    class _Source:
        def __init__(self, _cache_root, cohort, **_kwargs):
            self.samples = [(bytes.fromhex(row['sample_key']), row['source_smiles'])
                            for row in cohort['records']]

        def __len__(self):
            return len(self.samples)

        def close(self):
            pass

    monkeypatch.setattr(dual_runtime, 'FrozenDualLayerSource', _Source)
    monkeypatch.setattr(
        dual_runtime, 'load_dual_cohort',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError('full cohort loader must not run')),
    )
    selected, train_local, validation_local = compact_train_validation_indices(
        train_indices=[0, 3], validation_indices=[2])
    source, frame = dual_runtime.open_source(
        cohort_root, tmp_path / 'cache', task='xc', selected_indices=selected,
        expected_task_rows=4, expected_split_sha256=split_hash)
    assert len(source) == 3
    assert frame['original_row'].tolist() == selected
    assert frame['sample_key'].tolist() == [keys[index].hex() for index in selected]
    assert frame['label'].tolist() == [10.0, 20.0, 14.0]


def test_train_only_scaler_uses_compact_training_indices():
    selected, train_local, validation_local = compact_train_validation_indices(
        train_indices=[0, 3], validation_indices=[2])
    assert selected == [0, 2, 3]

    # The compact target array contains train+validation only. The scaler fit
    # uses only compact train indices; validation is transformed after fitting.
    targets = _Targets([10.0, 20.0, 14.0])
    scaler = scale_targets(targets, 'eat', train_indices=train_local,
                           transform_mode='standard')
    assert train_local == [0, 2]
    assert validation_local == [1]
    assert scaler.scaler.mean_[0] == 12.0
    np.testing.assert_allclose(targets.targets, [-1.0, 4.0, 1.0])
