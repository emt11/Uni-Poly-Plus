import numpy as np
import pytest
import sys
import json
import hashlib

from src.dataset.cache_lifecycle import CacheLifecycleError


def _write_key_fixture(tmp_path):
    entries = {"pi1m": ["a" * 64], "downstream": ["b" * 64]}
    payload = {"format": "test", "source": "synthetic"}
    for route, keys in entries.items():
        payload[route] = {
            "keys": keys,
            "sha256": hashlib.sha256("\n".join(keys).encode("ascii")).hexdigest(),
        }
    path = tmp_path / "expected_keys.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, entries


def _patch_fake_route(monkeypatch, parity, expected, result_keys=None):
    def fake_route(*_args, **kwargs):
        route = "pi1m" if kwargs["targets"] else "downstream"
        return {"keys": (result_keys or expected)[route]}

    monkeypatch.setattr(parity, "_route", fake_route)


def _base_parity_argv(parity, tmp_path, key_fixture, budget=1024):
    cache = tmp_path / "cache"
    cache.mkdir()
    return [
        "verify_glt_dual_static_parity.py",
        "--pi1m-cache-root", str(cache), "--pi1m-cohort-root", str(cache),
        "--pi1m-reference-static", str(cache), "--pi1m-reference-targets", str(cache),
        "--downstream-cache-root", str(cache), "--downstream-cohort-root", str(cache),
        "--downstream-reference-static", str(cache), "--temp-root", str(tmp_path / "temporary"),
        "--report-json", str(tmp_path / "report.json"),
        "--expected-key-json", str(key_fixture), "--limit-pi1m", "1",
        "--limit-downstream", "1", "--size-budget-bytes", str(budget),
    ]


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


def test_benchmark_exposes_performance_gate_without_claiming_acceptance():
    from scripts import benchmark_glt_dual_read as benchmark

    assert benchmark._performance_gate_passed(1.10, 1.05)
    assert not benchmark._performance_gate_passed(1.09, 1.00)
    source = open(benchmark.__file__, encoding="utf-8").read()
    assert '"performance_gate_passed"' in source
    assert '"overall_recommendation"' in source
    assert '"accepted"' not in source


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


def test_parity_main_fails_closed_when_active_cache_changes(monkeypatch, tmp_path):
    from scripts import verify_glt_dual_static_parity as parity

    key_fixture, expected = _write_key_fixture(tmp_path)
    _patch_fake_route(monkeypatch, parity, expected)
    states = iter([
        {"snapshot": 1}, {"snapshot": 1},
        {"snapshot": 2}, {"snapshot": 1},
    ])
    monkeypatch.setattr(parity, "zero_write_snapshot", lambda _path: next(states))
    monkeypatch.setattr(sys, "argv", _base_parity_argv(parity, tmp_path, key_fixture))
    assert parity.main() == 1
    payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert payload["status"] == "ACTIVE_CACHE_MODIFIED"
    assert payload["error_type"] == "ActiveCacheModified"


def test_parity_main_fails_closed_when_temporary_budget_is_exceeded(monkeypatch, tmp_path):
    from scripts import verify_glt_dual_static_parity as parity

    key_fixture, expected = _write_key_fixture(tmp_path)
    _patch_fake_route(monkeypatch, parity, expected)
    monkeypatch.setattr(parity, "zero_write_snapshot", lambda _path: {"snapshot": 1})
    monkeypatch.setattr(parity, "_temporary_bytes", lambda _path: 5)
    monkeypatch.setattr(sys, "argv", _base_parity_argv(parity, tmp_path, key_fixture, budget=4))
    assert parity.main() == 1
    payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert payload["status"] == "BUDGET_EXCEEDED"
    assert payload["error_type"] == "BudgetExceeded"


def test_parity_main_rejects_changed_fixed_key_list(monkeypatch, tmp_path):
    from scripts import verify_glt_dual_static_parity as parity

    key_fixture, expected = _write_key_fixture(tmp_path)
    wrong = {"pi1m": ["c" * 64], "downstream": expected["downstream"]}
    _patch_fake_route(monkeypatch, parity, expected, result_keys=wrong)
    monkeypatch.setattr(parity, "zero_write_snapshot", lambda _path: {"snapshot": 1})
    monkeypatch.setattr(sys, "argv", _base_parity_argv(parity, tmp_path, key_fixture))
    assert parity.main() == 1
    payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert payload["status"] == "UNRESOLVED_PROVENANCE"
    assert payload["error_type"] == "ExpectedKeyMismatch"
