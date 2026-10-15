import json
import numpy as np
import pytest
from src.dataset.mcl_ph_view import load_geometric_statistics, validate_direct_statistics_source


class Source:
    def __init__(self, root):
        self.root = root
        self.entries = [{'pretrain_role': 'train'} for i in range(3)]
    def __len__(self): return 2
    def source_index_at(self, index): return [0,2][index]


def artifact(root):
    path = root/'statistics.npz'
    np.savez(path, router_mean=np.zeros((31,5)), router_std=np.ones((31,5)),
             length_mu=[1.], length_sigma=[.2], nonbond_mu=[1.], nonbond_sigma=[.3],
             samples=[2], seed=[42], sigma=[.03],
             identity_contract=np.asarray('mcl-ph-train-statistics-direct-v1'),
             fit_source_indices=np.array([0,2], dtype=np.int64),
             valid_source_indices=np.array([0,2], dtype=np.int64))
    metadata = dict(scope='full', conclusion='PASS', source_root=str(root), mask_ratio=.30)
    path.with_suffix('.json').write_text(json.dumps(metadata))
    return path, metadata


def test_full_statistics_are_bound_to_actual_train_rows(tmp_path):
    path, metadata = artifact(tmp_path)
    statistics = load_geometric_statistics(path)
    source = Source(tmp_path)
    validate_direct_statistics_source(path, statistics, source, seed=42, sigma=.03)
    source.entries[2]['pretrain_role'] = 'validation'
    with pytest.raises(ValueError, match='non-train'):
        validate_direct_statistics_source(path, statistics, source, seed=42, sigma=.03)


@pytest.mark.parametrize('change,value', [('scope','probe'), ('source_root','/other/source')])
def test_probe_or_other_source_statistics_cannot_enter_training(tmp_path, change, value):
    path, metadata = artifact(tmp_path)
    metadata[change] = value
    path.with_suffix('.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='full fit'):
        validate_direct_statistics_source(path, load_geometric_statistics(path), Source(tmp_path), seed=42, sigma=.03)


def test_statistics_noise_recipe_cannot_change_on_reuse(tmp_path):
    path, metadata = artifact(tmp_path)
    with pytest.raises(ValueError, match='noise recipe'):
        validate_direct_statistics_source(path, load_geometric_statistics(path), Source(tmp_path), seed=43, sigma=.03)
    with pytest.raises(ValueError, match='noise recipe'):
        validate_direct_statistics_source(path, load_geometric_statistics(path), Source(tmp_path), seed=42, sigma=.03, ratio=.20)


def test_valid_statistics_indices_match_final_geometry_ledger(tmp_path):
    import sqlite3
    from scripts.audit_mcl_ph_statistics import validate_final_geometry
    ledger=tmp_path/'rows.sqlite'
    with sqlite3.connect(ledger) as conn:
        conn.execute('CREATE TABLE rows(source_index INTEGER, sample_key TEXT, final_status TEXT)')
        conn.executemany('INSERT INTO rows VALUES(?,?,?)',[(0,'a','PASS'),(1,'b','TIMEOUT')])
    entries=[{'sample_key':'a'},{'sample_key':'b'}]
    counts=validate_final_geometry(np.array([0,1]),np.array([0]),entries,ledger)
    assert counts=={'PASS':1,'TIMEOUT':1}
    with pytest.raises(AssertionError):validate_final_geometry(np.array([0,1]),np.array([1]),entries,ledger)
    entries[0]['sample_key']='other-key'
    with pytest.raises(ValueError,match='key differs'):validate_final_geometry(np.array([0,1]),np.array([0]),entries,ledger)
