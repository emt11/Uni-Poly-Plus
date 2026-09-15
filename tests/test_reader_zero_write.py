"""Frozen-reader policy: reader bookkeeping never counts as a cache write.

A frozen/published bundle may only be read with ``lock=False``.  Even if a
reader (or a legacy code path) does touch LMDB ``lock.mdb``, that is reader
bookkeeping: it carries no artifact content and no identity binding, so it is
outside the formal zero-write set.  Writers are refused outright on frozen
artifacts so nothing can silently mutate a published bundle.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.cache_lifecycle import (  # noqa: E402
    READER_BOOKKEEPING_NAMES,
    snapshot_tree,
    zero_write_snapshot,
)


def test_lock_mdb_is_bookkeeping_but_content_files_are_not(tmp_path):
    (tmp_path / "data.mdb").write_bytes(b"artifact content")
    (tmp_path / "lock.mdb").write_bytes(b"reader lock")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    before = zero_write_snapshot(tmp_path)
    assert READER_BOOKKEEPING_NAMES == frozenset({"lock.mdb"})
    assert "lock.mdb" not in before
    assert "data.mdb" in before and "manifest.json" in before

    # A lock.mdb touch is invisible to the zero-write definition ...
    (tmp_path / "lock.mdb").write_bytes(b"reader lock changed")
    assert zero_write_snapshot(tmp_path) == before
    # ... while the raw snapshot still records it.
    assert snapshot_tree(tmp_path) != before

    # Any artifact-defining change is a real write.
    (tmp_path / "manifest.json").write_text('{"a": 1}', encoding="utf-8")
    assert zero_write_snapshot(tmp_path) != before


def test_writer_refuses_a_frozen_artifact(tmp_path):
    from src.dataset.lmdb_cache import LmdbLayerWriter

    layer = tmp_path / "topology"
    layer.mkdir()
    (layer / ".frozen").write_text(
        json.dumps({"manifest_hash": "0" * 64}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="frozen artifact"):
        LmdbLayerWriter(layer, {"schema": "unused-because-frozen"})
