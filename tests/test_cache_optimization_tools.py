import numpy as np
import pytest
import sys
import json

from src.dataset.cache_lifecycle import CacheLifecycleError


def test_parity_comparison_accepts_integer_container_representations():
    from scripts.verify_glt_dual_static_parity import _compare

    differences, floats = _compare(
        {"groups": [np.asarray([1, 2], dtype=np.int32)]},
        {"groups": ((1, 2),)},
        "target",
    )
    assert differences == []
    assert floats == []


def test_parity_comparison_rejects_nonfinite_and_value_changes():
    from scripts.verify_glt_dual_static_parity import _compare

    differences, _ = _compare(
        np.asarray([1, 2], dtype=np.int32),
        np.asarray([1, 3], dtype=np.int64),
        "indices",
    )
    assert differences and differences[0]["reason"] == "value mismatch"
    differences, _ = _compare(
        np.asarray([np.nan], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        "geometry",
    )
    assert differences and differences[0]["reason"] == "nonfinite"


def test_parity_temp_root_and_budget_are_explicit(tmp_path):
    from scripts.verify_glt_dual_static_parity import _check_budget, _new_temp_root

    existing = tmp_path / "already-there"
    existing.mkdir()
    with pytest.raises(CacheLifecycleError, match="new path"):
        _new_temp_root(existing)
    fresh = _new_temp_root(tmp_path / "fresh")
    (fresh / "payload.bin").write_bytes(b"0123456789")
    with pytest.raises(CacheLifecycleError, match="exceeds budget"):
        _check_budget(fresh, 4)


def test_benchmark_counter_counts_only_chunk_cache_misses(monkeypatch):
    import src.dataset.glt_dual_static as static_module
    from scripts import benchmark_glt_dual_read as benchmark

    original_reader_load = static_module._ChunkReader._load
    original_np_load = np.load

    def fake_load(self, chunk_id):
        self._cache.setdefault(chunk_id, object())
        return self._cache[chunk_id]

    monkeypatch.setattr(static_module._ChunkReader, "_load", fake_load)
    counts = benchmark._instrument()
    reader = type("Reader", (), {"_cache": {}})()
    static_module._ChunkReader._load(reader, 0)
    static_module._ChunkReader._load(reader, 0)
    static_module._ChunkReader._load(reader, 1)
    assert counts["chunk_cache_misses"] == 2
    assert counts["array_open_calls"] == 0
    monkeypatch.setattr(static_module._ChunkReader, "_load", original_reader_load)
    monkeypatch.setattr(np, "load", original_np_load)


def test_parity_main_writes_classified_failure_report(monkeypatch, tmp_path):
    from scripts import verify_glt_dual_static_parity as parity

    cache = tmp_path / "cache"
    cache.mkdir()
    report_path = tmp_path / "report.json"
    temp_path = tmp_path / "temporary"

    def missing_route(*_args, **_kwargs):
        raise parity.MissingFixture("required real category is absent")

    monkeypatch.setattr(parity, "_route", missing_route)
    monkeypatch.setattr(sys, "argv", [
        "verify_glt_dual_static_parity.py",
        "--pi1m-cache-root", str(cache), "--pi1m-cohort-root", str(cache),
        "--pi1m-reference-static", str(cache), "--pi1m-reference-targets", str(cache),
        "--downstream-cache-root", str(cache), "--downstream-cohort-root", str(cache),
        "--downstream-reference-static", str(cache), "--temp-root", str(temp_path),
        "--report-json", str(report_path),
    ])
    assert parity.main() == 1
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "MISSING_FIXTURE"
    assert payload["error_type"] == "MissingFixture"
    assert payload["model_status"] == "NOT_RUN"
