"""Data repair regressions. All geometries here are synthetic, NOT real fixtures.

These tests are delivered unexecuted in the current environment.
"""
import json
import sys

import pytest
import torch
from rdkit import Chem

from test_complete_trimer_glt import _toy_pair
from src.dataset.graph_data import build_periodic_multimer_mol
from src.dataset.periodic_line_glt_complete import build_complete_trimer_glt_sample
from src.dataset.glt_dual import FrozenDualLayerSource, build_dual_sample, dual_glt_collate
from src.dataset.canonical_periodic import build_canonical_periodic_topology
from scripts import validate_dual_glt as audit


@pytest.mark.parametrize('smiles', ['*COC*', '*C*'])
def test_missing_physical_bond_is_invalid_even_for_n0(smiles):
    top, tri = _toy_pair(smiles)
    tri.trimer_edge_index = tri.trimer_edge_index[:, 2:]
    tri.trimer_bond_type = tri.trimer_bond_type[2:]
    row = build_complete_trimer_glt_sample(top, tri, smiles)
    assert not row['geometry_valid']
    assert row['invalid_reason'].startswith('physical_bond_set_mismatch:missing=')


def test_undirected_storage_and_joint_renumbering_are_supported():
    top, tri = _toy_pair('*COC*')
    tri.trimer_edge_index = tri.trimer_edge_index[:, ::2]
    tri.trimer_bond_type = tri.trimer_bond_type[::2]
    assert build_complete_trimer_glt_sample(top, tri, '*COC*')['geometry_valid']
    order = torch.arange(tri.trimer_pos.size(0) - 1, -1, -1)
    inverse = torch.argsort(order)
    for name in ('trimer_pos', 'trimer_atomic_number', 'trimer_base_ru_atom_id', 'trimer_ru_offset'):
        setattr(tri, name, getattr(tri, name)[order])
    tri.trimer_edge_index = inverse[tri.trimer_edge_index]
    assert build_complete_trimer_glt_sample(top, tri, '*COC*')['geometry_valid']


@pytest.mark.parametrize('corruption', ['side_element', 'duplicate_identity', 'bond_type'])
def test_complete_identity_and_types(corruption):
    top, tri = _toy_pair('*COC*')
    if corruption == 'side_element':
        tri.trimer_atomic_number[0] = 7
    elif corruption == 'duplicate_identity':
        tri.trimer_base_ru_atom_id[0] = 1
    else:
        tri.trimer_bond_type[:2] = 2
    assert not build_complete_trimer_glt_sample(top, tri, '*COC*')['geometry_valid']


@pytest.mark.parametrize('smiles', ['*C/C=C/C*', '*C/C=C\\C*', '*/C=C/*', '*/C(C)=C(C)/*'])
def test_pristine_stereo_audit_and_lost_stereo_detection(smiles):
    source = Chem.MolFromSmiles(smiles)
    for original in (source, Chem.RenumberAtoms(source, list(reversed(range(source.GetNumAtoms()))))):
        mol, meta = build_periodic_multimer_mol(original, 3, close_periodic=False)
        result = audit.audit_source_stereo(original, mol, meta)
        assert result['status'] == 'PASS', result
        assert result['center_specified'] == 1
        assert result['retained_copies'] > 0
        center = set(meta['unit_atoms'][1])
        double = next(b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE
                      and b.GetBeginAtomIdx() in center and b.GetEndAtomIdx() in center)
        double.SetStereo(Chem.BondStereo.STEREONONE)
        assert audit.audit_source_stereo(original, mol, meta)['status'] == 'ANOMALY'


def test_pristine_audit_detects_direction_corruption():
    original = Chem.MolFromSmiles('*C/C=C/C*')
    mol, meta = build_periodic_multimer_mol(original, 3, close_periodic=False)
    bond = next(b for b in mol.GetBonds() if b.GetBondDir() != Chem.BondDir.NONE)
    bond.SetBondDir(Chem.BondDir.NONE)
    assert audit.audit_source_stereo(original, mol, meta)['status'] == 'ANOMALY'


def test_audit_early_failure_has_explicit_check_statuses():
    top, tri = _toy_pair()
    report = audit.audit_record(top, tri, 'CC')  # valid SMILES, missing two stars
    assert report['checks']['chemistry']['status'] == 'ANOMALY'
    assert report['checks']['model_input']['status'] == 'NOT_RUN'
    assert report['checks']['source_stereo']['status'] == 'NOT_RUN'


def test_degenerate_stereo_axis_has_explicit_failure():
    _, tri = _toy_pair('*C/C=C/C*')
    tri.trimer_pos.zero_()
    with pytest.raises(ValueError, match='degenerate stereo axis'):
        audit.audit_frozen_stereo(tri, '*C/C=C/C*')


def test_missing_model_field_is_not_geometry_fallback(monkeypatch):
    from src.dataset import glt_dual
    _, tri = _toy_pair('*COC*')
    top = build_canonical_periodic_topology('*COC*')
    monkeypatch.setattr(glt_dual, 'build_complete_trimer_glt_sample', lambda *args: {})
    with pytest.raises(KeyError):
        build_dual_sample(top, tri, '*COC*')


def _entry(expected, actual, stereo):
    return dict(checks={
        'source_stereo': dict(status='PASS', expected_center_bonds=expected, center_specified=stereo),
        'model_input': dict(status='PASS', center_bonds=actual)})


def test_fixture_roles_use_source_counts_and_bound_stereo():
    assert audit.fixture_coverage([_entry(0, 0, 0), _entry(2, 2, 1)])['status'] == 'PASS'
    assert audit.fixture_coverage([_entry(1, 0, 1), _entry(2, 2, 0)])['status'] == 'ANOMALY'


def test_partial_lmdb_open_failure_closes_first_layer(monkeypatch):
    from src.dataset import lmdb_cache
    closed = []
    class FakeStore:
        def __init__(self, path):
            if path == 'second':
                raise OSError('open failed')
        def close(self):
            closed.append(True)
    monkeypatch.setattr(lmdb_cache, 'LmdbLayerStore', FakeStore)
    with pytest.raises(OSError, match='open failed'):
        FrozenDualLayerSource('first', 'second', [])
    assert closed == [True]


def test_schema_failure_closes_both_layers(monkeypatch):
    from src.dataset import lmdb_cache
    closed = []
    class FakeStore:
        schema = 'wrong schema'
        def __init__(self, path):
            self.path = path
        def close(self):
            closed.append(self.path)
    monkeypatch.setattr(lmdb_cache, 'LmdbLayerStore', FakeStore)
    with pytest.raises(ValueError, match='semantics'):
        FrozenDualLayerSource('first', 'second', [])
    assert closed == ['first', 'second']


def test_close_attempts_both_layers():
    calls = []
    class Store:
        def __init__(self, fail):
            self.fail = fail
        def close(self):
            calls.append(self.fail)
            if self.fail:
                raise OSError('close failed')
    source = FrozenDualLayerSource.__new__(FrozenDualLayerSource)
    source.topology, source.trimer = Store(True), Store(False)
    with pytest.raises(OSError, match='close failed'):
        source.close()
    assert calls == [True, False]
    source.close()  # idempotent


def test_multipath_batch_offsets_with_n0():
    items = []
    for smiles in ('*C1CCC1*', '*C*'):
        _, tri = _toy_pair(smiles)
        top = build_canonical_periodic_topology(smiles)
        items.append(build_dual_sample(top, tri, smiles))
    batch = dual_glt_collate(items)
    a, b = items
    offset = a.line_path.size(0)
    torch.testing.assert_close(batch.line_path_group[offset:], b.line_path_group + a.line_source.numel())
    expected = b.line_path.clone()
    expected[expected >= 0] += a.bond_distance.numel()
    torch.testing.assert_close(batch.line_path[offset:], expected)


@pytest.mark.parametrize('failure', [None, 'close', 'report', 'model'])
def test_audit_json_status_and_failure_reporting(monkeypatch, capsys, tmp_path, failure):
    class Source:
        def __init__(self, *args):
            pass
        def __getitem__(self, index):
            return None, None, str(index)
        def close(self):
            if failure == 'close':
                raise OSError('close failure')
    def record(top, tri, smiles):
        result = _entry(0, 0, 0) if smiles == '0' else _entry(2, 2, 1)
        result['checks']['connection_policy'] = dict(status='REVIEW')
        return result
    monkeypatch.setattr(audit, 'FrozenDualLayerSource', Source)
    monkeypatch.setattr(audit, 'audit_record', record)
    destination = tmp_path / 'audit.json'
    if failure == 'report':
        destination = tmp_path / 'missing-parent' / 'audit.json'
    if failure == 'model':
        def fail_dataset(*args):
            raise RuntimeError('model validation fixture failure')
        monkeypatch.setattr(audit, 'DualGLTDataset', fail_dataset)
    argv = ['audit', '--topology-root', 'top', '--trimer-root', 'tri',
        '--sample', '00' * 32, '*C*', '--sample', '11' * 32, '*CC*',
        '--report-json', str(destination)]
    if failure != 'model':
        argv.append('--audit-only')
    monkeypatch.setattr(sys, 'argv', argv)
    if failure in ('close', 'report'):
        with pytest.raises(RuntimeError):
            audit.main()
    else:
        assert audit.main() == (2 if failure == 'model' else 0)
    report = json.loads(capsys.readouterr().out)
    assert report['outcome'] == ('REVIEW' if failure is None else
                                 'MODEL_FAILURE' if failure == 'model' else 'SCRIPT_ERROR')
    assert report['model']['status'] == ('FAIL' if failure == 'model' else 'NOT_RUN')
    if failure != 'report':
        assert json.loads(destination.read_text(encoding='utf-8')) == report
