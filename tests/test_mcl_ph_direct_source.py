"""Direct row identity checks must preserve selected-label isolation."""
import json

import numpy as np
import pytest
from torch_geometric.data import Data

from src.dataset.cache_lifecycle import StagingWriter, select_record_fields
from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.glt_dual_static import build_dual_static
from src.dataset.mcl_ph_direct import DirectMCLSource
from src.dataset.trimer_mcl import attach_unavailable_trimer_identity
from src.dataset.lmdb_cache import sample_key_from_normalized
from src.training.glt_dual_runtime import load_sample_index_artifact
from src.training.glt_dual_runtime import IndexedFrozenDualSource, pretrain_source_fields


def fixture(tmp_path, *, corrupt_center=False):
    root = tmp_path / 'source.staging'
    key = bytes(range(32))
    topology = build_canonical_periodic_topology('*CC*')
    trimer = select_record_fields('trimer', attach_unavailable_trimer_identity(
        topology.clone(), '*CC*', 'TIMEOUT'))
    if corrupt_center:
        # Both atoms are carbon, so an element-only assertion cannot detect it.
        trimer.mips_to_trimer_central_index = trimer.mips_to_trimer_central_index.flip(0)
    writer = StagingWriter(root, {'schema': 'fixture'})
    writer.put(key, Data(topology=topology, trimer=trimer,
        static=build_dual_static(topology, trimer, '*CC*'),
        normalized_smiles=topology.normalized_canonical_smiles,
        final_geometry_status='TIMEOUT'))
    writer.close()
    manifest = {'schema': 'mcl-ph-direct-source-v1', 'kind': 'downstream',
                'conclusion': 'PASS', 'protocol': 'paper5_outer',
                'coordinate_generation_calls': 0, 'identity_records': 1,
                'task_rows': {'ei': 3}}
    (root / 'direct_source.json').write_text(json.dumps(manifest))
    (root / '.frozen').write_text('fixture')
    offsets = []
    with (root / 'ei.jsonl').open('wb') as stream:
        for index, role in enumerate(('train', 'validation', 'test')):
            offsets.append(stream.tell())
            # The unselected test label deliberately cannot be JSON-decoded.
            if index == 2:
                stream.write(b'UNSELECTED TEST MUST NOT BE DECODED\n')
                continue
            row = {'task': 'ei', 'original_row': index,
                   'sample_key': key.hex(), 'source_smiles': '*CC*', 'label': float(index),
                   'normalized_smiles': topology.normalized_canonical_smiles,
                   'geometry_status': 'TIMEOUT',
                   'fold_membership': [{'fold': fold, 'role': role} for fold in range(5)]}
            stream.write((json.dumps(row) + '\n').encode())
    np.save(root / 'ei_offsets.npy', np.asarray(offsets, dtype=np.int64))
    split = {'task': 'ei', 'protocol': 'paper5_outer', 'sample_count': 3,
             'folds': [{'fold': fold, 'train_indices': [0],
                        'validation_indices': [1], 'test_indices': [2]} for fold in range(5)]}
    return root, split


def test_direct_reader_ignores_unselected_label_bytes(tmp_path):
    root, split = fixture(tmp_path)
    source = DirectMCLSource(root, task='ei', selected_indices=[0, 1], split_manifest=split)
    try:
        assert len(source) == 2
        assert not source[0][1].trimer_geometry_valid
        assert not source.static_for(1)['geometry_valid']
    finally:
        source.close()


@pytest.mark.parametrize('with_identity', [False, True])
def test_pretrain_fields_decode_once_and_preserve_subset_row(tmp_path, monkeypatch, with_identity):
    root, split = fixture(tmp_path)
    base = DirectMCLSource(root, task='ei', selected_indices=[0, 1], split_manifest=split)
    source = IndexedFrozenDualSource(base, np.array([1], dtype=np.int64))
    calls = []
    record = base.record

    def observe(index):
        calls.append(index)
        return record(index)

    monkeypatch.setattr(base, 'record', observe)
    try:
        fields = pretrain_source_fields(source, 0, include_identity=with_identity)
        topology, trimer, smiles, static, target = fields[:5]
        assert calls == [1]
        assert smiles == '*CC*' and topology.graph_available
        assert not trimer.trimer_geometry_valid and not static['geometry_valid']
        assert target is None
        if with_identity:
            assert len(fields) == 6
            assert fields[5]['normalized_smiles'] == topology.normalized_canonical_smiles
    finally:
        source.close()


@pytest.mark.parametrize('with_identity', [False, True])
def test_pretrain_fields_keep_center_identity_check(tmp_path, with_identity):
    root, split = fixture(tmp_path, corrupt_center=True)
    source = DirectMCLSource(root, task='ei', selected_indices=[0], split_manifest=split)
    try:
        with pytest.raises(ValueError, match='center mapping'):
            pretrain_source_fields(source, 0, include_identity=with_identity)
    finally:
        source.close()


def test_direct_reader_rejects_role_conflicts_and_unfrozen_source(tmp_path):
    root, split = fixture(tmp_path)
    split['folds'][0]['train_indices'], split['folds'][0]['validation_indices'] = [1], [0]
    with pytest.raises(ValueError, match='roles differ'):
        DirectMCLSource(root, task='ei', selected_indices=[0], split_manifest=split)
    (root / '.frozen').unlink()
    with pytest.raises(ValueError, match='not verified and frozen'):
        DirectMCLSource(root, task='ei', selected_indices=[0], split_manifest=split)


def test_direct_reader_rejects_same_element_center_swap(tmp_path):
    root, split = fixture(tmp_path, corrupt_center=True)
    source = DirectMCLSource(root, task='ei', selected_indices=[0], split_manifest=split)
    try:
        with pytest.raises(ValueError, match='center mapping'):
            source[0]
    finally:
        source.close()


def test_pi1m_direct_reader_and_index_contract(tmp_path):
    root = tmp_path / 'pi1m.staging'
    key = sample_key_from_normalized('*CC*')
    topology = build_canonical_periodic_topology('*CC*')
    trimer = select_record_fields('trimer', attach_unavailable_trimer_identity(
        topology.clone(), '*CC*', 'TIMEOUT'))
    writer = StagingWriter(root, {'schema': 'fixture'})
    from src.dataset.glt_dual_static import build_pretrain_target
    writer.put(key, Data(topology=topology, trimer=trimer,
        static=build_dual_static(topology, trimer, '*CC*'),
        target=build_pretrain_target(topology.normalized_canonical_smiles),
        normalized_smiles=topology.normalized_canonical_smiles,
        final_geometry_status='TIMEOUT', source_index=0))
    writer.close()
    manifest = {'schema': 'mcl-ph-direct-source-v1', 'kind': 'pi1m',
                'conclusion': 'PASS', 'coordinate_generation_calls': 0,
                'identity_records': 1, 'source_rows': 2}
    (root / 'direct_source.json').write_text(json.dumps(manifest))
    (root / '.frozen').write_text('fixture')
    rows = [
        {'source_index': 0, 'sample_key': key.hex(), 'source_smiles': '*CC*',
         'normalized_smiles': topology.normalized_canonical_smiles, 'pretrain_role': 'train'},
        {'source_index': 1, 'sample_key': (bytes([1]) * 32).hex(), 'source_smiles': '*CO*',
         'normalized_smiles': '*CO*', 'pretrain_role': 'excluded'},
    ]
    offsets = []
    with (root / 'records.jsonl').open('wb') as stream:
        for row in rows:
            offsets.append(stream.tell())
            stream.write((json.dumps(row) + '\n').encode())
    np.save(root / 'record_offsets.npy', np.asarray(offsets, dtype=np.int64))
    source = DirectMCLSource(root, task=None, selected_indices=[0], split_manifest=None)
    try:
        assert len(source) == 1
        assert source.source_index_at(0) == 0
        assert source.entries[0]['pretrain_role'] == 'train'
        assert set(source.target_for(0)) == {'brics_groups'}
    finally:
        source.close()


def test_direct_index_summary_does_not_require_source_sha(tmp_path):
    root = tmp_path / 'indices'
    root.mkdir()
    summary = {'schema': 'mcl-ph-next-direct-index-v1',
               'pretrain_train_count': 2, 'pi1m_rows': 3}
    (root / 'summary.json').write_text(json.dumps(summary))
    np.savez(root / 'pretrain_indices.npz', train_source_indices=np.asarray([0, 2]),
             validation_source_indices=np.asarray([1]), fixed_validation_source_indices=np.asarray([1]))
    loaded = load_sample_index_artifact(root / 'summary.json', 'train')
    assert loaded['direct_contract'] and loaded['indices'].tolist() == [0, 2]
