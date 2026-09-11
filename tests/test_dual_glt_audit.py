"""Data repair regressions. All geometries here are synthetic, NOT real fixtures.

Real frozen-record coverage is kept in ``scripts/validate_dual_glt.py`` and is
never inferred from these synthetic cases.
"""
import json
import sys

import pytest
import torch
from rdkit import Chem

from test_complete_trimer_glt import _toy_pair
from src.dataset.graph_data import build_periodic_multimer_mol, build_star_linking_mol
from src.dataset.periodic_line_glt_complete import build_complete_trimer_glt_sample
from src.dataset.glt_dual import FrozenDualLayerSource, build_dual_sample, dual_glt_collate
from src.dataset.canonical_periodic import (
    build_canonical_periodic_topology, resolve_normalized_identity,
)
from src.dataset.glt_dual_pretrain import (
    chemical_targets, motif_mask, prepare_pretrain_sample, sample_generator,
)
from src.modules.glt_dual import build_dual_glt_model
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


def test_source_stereo_audit_allows_canonical_endpoint_reversal():
    # Canonical RDKit output reverses this asymmetric Z bond's adjacent slash
    # directions.  The chemical stereo is unchanged, so the audit must compare
    # BondDir in the mapped bond orientation instead of raw enum equality.
    smiles = '*N(C)C/C=C\\C*'
    top = build_canonical_periodic_topology(smiles)
    identity = resolve_normalized_identity(top, smiles, require_fields=True)
    mol, meta = build_periodic_multimer_mol(
        identity['normalized_smiles'], 3, close_periodic=False
    )
    result = audit.audit_source_stereo(
        identity['source_molecule'], mol, meta,
        identity['source_to_normalized_base'],
        identity['source_to_normalized_atom'], identity['normalized_molecule'],
    )
    assert result['status'] == 'PASS', result


def test_source_stereo_audit_maps_dummy_reference_after_normalization():
    # The canonical reparse reverses the two dummy sites.  The terminal E/Z
    # reference must follow the normalized dummy side, not the caller's dummy
    # list position.
    smiles = '*N(C)/C=C/*'
    top = build_canonical_periodic_topology(smiles)
    identity = resolve_normalized_identity(top, smiles, require_fields=True)
    mol, meta = build_periodic_multimer_mol(
        identity['normalized_smiles'], 3, close_periodic=False
    )
    result = audit.audit_source_stereo(
        identity['source_molecule'], mol, meta,
        identity['source_to_normalized_base'],
        identity['source_to_normalized_atom'], identity['normalized_molecule'],
    )
    assert result['status'] == 'PASS', result
    assert result['retained_copies'] == 2
    assert result['terminal_undefined'] == 1


def test_audit_early_failure_has_explicit_check_statuses():
    top, tri = _toy_pair()
    report = audit.audit_record(top, tri, 'CC')  # valid SMILES, missing two stars
    assert report['checks']['chemistry']['status'] == 'ANOMALY'
    assert report['checks']['model_input']['status'] == 'NOT_RUN'
    assert report['checks']['source_stereo']['status'] == 'NOT_RUN'


def test_degenerate_stereo_axis_has_explicit_failure():
    _, tri = _toy_pair('*C/C=C/C*')
    tri.trimer_pos.zero_()
    with pytest.raises(ValueError, match='3 frozen stereo coordinate checks failed'):
        audit.audit_frozen_stereo(tri, '*C/C=C/C*')


def test_missing_model_field_is_not_geometry_fallback(monkeypatch):
    from src.dataset import glt_dual
    _, tri = _toy_pair('*COC*')
    top = build_canonical_periodic_topology('*COC*')
    monkeypatch.setattr(
        glt_dual, 'build_complete_trimer_glt_sample',
        lambda *args, **kwargs: {},
    )
    with pytest.raises(KeyError):
        build_dual_sample(top, tri, '*COC*')


def test_structural_trimer_failure_is_not_geometry_fallback():
    top = build_canonical_periodic_topology('*COC*')
    _, tri = _toy_pair('*COC*')
    tri.trimer_edge_index = tri.trimer_edge_index[:, 2:]
    tri.trimer_bond_type = tri.trimer_bond_type[2:]
    with pytest.raises(ValueError, match='physical_bond_set_mismatch'):
        build_dual_sample(top, tri, '*COC*')


def test_missing_identity_is_not_geometry_fallback_when_geometry_is_invalid():
    top = build_canonical_periodic_topology('*COC*')
    tri = type('SparseTrimer', (), {
        'trimer_geometry_valid': False,
        'trimer_geometry_is_3d': False,
        'trimer_2d_fallback': False,
    })()
    with pytest.raises(ValueError, match='trimer_identity_missing'):
        build_dual_sample(top, tri, '*COC*')


def test_normalized_identity_keeps_paths_brics_noise_and_prediction_inputs_equal():
    raw = '*N(C)C/C=C/C*'
    normalized = Chem.MolToSmiles(Chem.MolFromSmiles(raw), canonical=True)
    alternate = Chem.MolToSmiles(
        Chem.RenumberAtoms(Chem.MolFromSmiles(raw), list(reversed(range(8)))),
        canonical=False,
    )
    top = build_canonical_periodic_topology(raw)
    _, trimer = _toy_pair(normalized)
    raw_item = build_dual_sample(top, trimer, raw)
    normalized_item = build_dual_sample(top, trimer, normalized)
    alternate_item = build_dual_sample(top, trimer, alternate)
    for name in (
        'mips_x', 'mips_backbone_mask', 'lga_edge_index', 'lga_spd',
        'lga_path_index', 'lga_path_mask', 'lga_path_shift',
        'lga_source_image_shift', 'bond_path_features', 'bond_path_mask',
        'bond_z_a', 'bond_z_b', 'bond_distance', 'bond_type', 'bond_center',
        'line_path', 'line_path_group', 'line_source', 'line_target',
        'line_angle', 'line_mask', 'line_is_self',
    ):
        torch.testing.assert_close(getattr(raw_item, name), getattr(normalized_item, name))
        torch.testing.assert_close(getattr(raw_item, name), getattr(alternate_item, name))
    assert raw_item.geometry_valid == normalized_item.geometry_valid == alternate_item.geometry_valid

    groups_raw, fingerprint_raw = chemical_targets(raw)
    groups_norm, fingerprint_norm = chemical_targets(normalized)
    assert groups_raw == groups_norm
    torch.testing.assert_close(fingerprint_raw, fingerprint_norm)
    mask_raw, fallback_raw = motif_mask(
        top, groups_raw, sample_generator(42, 'identity', 7)
    )
    mask_norm, fallback_norm = motif_mask(
        top, groups_norm, sample_generator(42, 'identity', 7)
    )
    assert fallback_raw == fallback_norm
    assert torch.equal(mask_raw, mask_norm)

    raw_noisy, raw_labels = prepare_pretrain_sample(
        top, trimer, raw, seed=42, key='same-physical-noise', position=11,
        sigma=0.03, ratio=0.3,
    )
    norm_noisy, norm_labels = prepare_pretrain_sample(
        top, trimer, normalized, seed=42, key='same-physical-noise', position=11,
        sigma=0.03, ratio=0.3,
    )
    for name in raw_noisy.keys():
        left, right = getattr(raw_noisy, name), getattr(norm_noisy, name)
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right)
        else:
            assert left == right
    for name in raw_labels:
        left, right = raw_labels[name], norm_labels[name]
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right)
        else:
            assert left == right

    for mode in ('concat', 'kfuse'):
        torch.manual_seed(123)
        model = build_dual_glt_model(mode, dropout=0).eval()
        raw_prediction = model(dual_glt_collate([raw_noisy]))
        normalized_prediction = model(dual_glt_collate([norm_noisy]))
        torch.testing.assert_close(raw_prediction, normalized_prediction)


def test_normalized_identity_rejects_damaged_cached_mapping():
    raw = '*N(C)C/C=C/C*'
    normalized = Chem.MolToSmiles(Chem.MolFromSmiles(raw), canonical=True)
    top = build_canonical_periodic_topology(raw)
    _, trimer = _toy_pair(normalized)
    broken = top.clone()
    broken.source_to_normalized_canonical_atom_id = torch.roll(
        top.source_to_normalized_canonical_atom_id, 1, 0
    )
    with pytest.raises(ValueError, match='cached source-to-normalized base mapping'):
        build_dual_sample(broken, trimer, raw)
    with pytest.raises(ValueError, match='cached source-to-normalized base mapping'):
        build_complete_trimer_glt_sample(broken, trimer, raw)
    broken = top.clone()
    broken.canonical_to_trimer_base_atom_id = torch.tensor(
        [0, 0, 2, 3, 4, 5]
    )
    with pytest.raises(ValueError, match='permutation'):
        build_dual_sample(broken, trimer, raw)


def test_star_linking_reports_declared_and_actual_edge_attributes():
    _, matching = build_star_linking_mol('*c1ccccc1*', return_mapping=True)
    assert matching['connection_policy'] == 'matching_attachment_type'
    assert matching['declared_attachment_bond_type_left'] == 'SINGLE'
    assert matching['declared_attachment_bond_type_right'] == 'SINGLE'
    # The two boundary atoms already share an aromatic ring bond; no second
    # physical edge is added and the post-sanitize edge remains aromatic.
    assert matching['actual_link_added'] is False
    assert matching['actual_link_bond_type'] == 'AROMATIC'
    assert matching['actual_link_conjugated'] is True
    assert matching['actual_link_ring'] is True

    _, added = build_star_linking_mol('*c1ccc(*)cc1', return_mapping=True)
    assert added['actual_link_added'] is True
    assert added['actual_link_bond_type'] == 'SINGLE'
    with pytest.raises(ValueError, match='attachment bond types'):
        build_star_linking_mol('*C(C#*)CC', return_mapping=True)


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
    assert audit.main() == (2 if failure in ('close', 'report', 'model') else 0)
    report = json.loads(capsys.readouterr().out)
    assert report['outcome'] == ('REVIEW' if failure is None else
                                 'MODEL_FAILURE' if failure == 'model' else 'SCRIPT_ERROR')
    assert report['model']['status'] == ('FAIL' if failure == 'model' else 'NOT_RUN')
    if failure != 'report':
        assert json.loads(destination.read_text(encoding='utf-8')) == report


def test_audit_open_failure_emits_two_complete_records(monkeypatch, capsys, tmp_path):
    class Source:
        def __init__(self, *args):
            raise FileNotFoundError('missing frozen layer')

    monkeypatch.setattr(audit, 'FrozenDualLayerSource', Source)
    destination = tmp_path / 'open_failure.json'
    argv = ['audit', '--topology-root', 'top', '--trimer-root', 'tri',
            '--sample', '00' * 32, '*C*', '--sample', '11' * 32, '*CC*',
            '--audit-only', '--report-json', str(destination)]
    monkeypatch.setattr(sys, 'argv', argv)
    assert audit.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report['outcome'] == 'DATA_ANOMALY'
    assert report['model']['status'] == 'NOT_RUN'
    assert report['fixture_coverage']['status'] == 'ANOMALY'
    assert len(report['samples']) == 2
    assert all(set(entry['checks']) == set(audit.AUDIT_CHECK_NAMES)
               for entry in report['samples'])
    assert json.loads(destination.read_text(encoding='utf-8')) == report
