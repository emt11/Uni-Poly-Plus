"""PI1M failures retain atom masking targets and cannot acquire coordinates."""
import sqlite3

import lmdb
import numpy as np

from scripts import publish_mcl_ph_pi1m_direct as publisher
from src.dataset.trimer_mcl import get_etkdg_call_count, reset_etkdg_call_counter


def test_pi1m_failure_materialization_has_targets_and_no_embedding(tmp_path):
    geometry = tmp_path / 'geometry'
    geometry.mkdir()
    key = bytes(range(32))
    db = sqlite3.connect(geometry / 'rows.sqlite')
    db.execute('CREATE TABLE rows(source_index INTEGER, sample_key TEXT, final_status TEXT)')
    db.execute('INSERT INTO rows VALUES (0, ?, ?)', (key.hex(), 'TIMEOUT'))
    db.commit()
    db.close()
    path = geometry / 'data.lmdb'
    path.mkdir()
    env = lmdb.open(str(path), map_size=64 * 1024**2)
    env.close()
    reset_etkdg_call_counter()
    publisher.initialize(geometry)
    try:
        index, actual_key, record, error = publisher.materialize((0, '*CC*', key.hex()))
        assert error is None
        assert index == 0 and actual_key == key
        assert not record.trimer.trimer_geometry_valid
        assert record.trimer.trimer_pos.shape == (0, 3)
        assert not record.static['geometry_valid']
        assert set(record.target) == {'brics_groups'} and record.target['brics_groups']
        assert get_etkdg_call_count() == 0
    finally:
        publisher.SOURCE_ENV.close()
        publisher.SOURCE_DB.close()
