"""Relaxed per-sample failure policy tests (production cases A-F).

A/B/C: known sample-local RU failures — degenerate dummy-only repeat units
("*=*", "**") and the real no-validated-isomorphism boundary sample — become
per-sample RU failures; the build continues, downstream layers are
SKIPPED_PARENT_FAILED (not failed), and the bundle still publishes.
D: an unknown struct exception becomes UNCLASSIFIED_SAMPLE_FAILURE and the
build continues with the following samples (E).
F: a writer/storage failure is SYSTEM_FATAL: the whole build stops and
nothing is published.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_ru_build_boundary import (  # noqa: E402
    BOUNDARY_KEY,
    BOUNDARY_SMILES,
)

DEGENERATE_STAR_BOND_STAR = "*=*"
DEGENERATE_STAR_STAR = "**"
LEDGER_REQUIRED_FIELDS = (
    "sample_key", "source_row", "canonical_smiles", "layer",
    "failure_code", "exception_type", "exception_message",
    "worker_pid", "elapsed_seconds",
)


def _source_csv(path: Path, extra_rows):
    smiles_rows = [
        DEGENERATE_STAR_BOND_STAR, DEGENERATE_STAR_STAR, BOUNDARY_SMILES,
    ]
    smiles_rows += [
        f"*C{'C' * a}(C){'O' * b}C*" for a in range(4) for b in range(5)
    ]
    smiles_rows += list(extra_rows)
    path.write_text(
        "smiles\n" + "\n".join(smiles_rows) + "\n", encoding="utf-8"
    )
    return smiles_rows


def _run_build(cache_root: Path, source_csv: Path, workers: int = 4,
               env_extra: dict | None = None, timeout: int = 900):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "."
    if env_extra:
        environment.update(env_extra)
    return subprocess.run(
        [sys.executable, "scripts/build_mts_cache.py",
         "--cache-root", str(cache_root),
         "--source-csv", str(source_csv),
         "--limit", "0", "--workers", str(workers)],
        capture_output=True, text=True, timeout=timeout, env=environment,
        cwd=str(ROOT),
    )


def test_known_ru_failures_continue_downstream_skipped_and_publish(tmp_path):
    """Cases A + B + C + E on the real multi-worker spawn path."""
    from src.dataset.lmdb_cache import (
        normalize_polymer_smiles,
        sample_key_from_normalized,
    )

    source_csv = tmp_path / "relaxed.csv"
    _source_csv(source_csv, extra_rows=[])
    cache_root = tmp_path / "cache"
    key_a = sample_key_from_normalized(
        normalize_polymer_smiles(DEGENERATE_STAR_BOND_STAR)[0]
    ).hex()
    key_b = sample_key_from_normalized(
        normalize_polymer_smiles(DEGENERATE_STAR_STAR)[0]
    ).hex()
    failed_keys = {key_a, key_b, BOUNDARY_KEY}

    result = _run_build(cache_root, source_csv)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]

    store = json.loads((cache_root / "store.json").read_text(encoding="utf-8"))
    bundle = cache_root / "builds" / store["bundle_hash"]
    ru_manifest = json.loads((bundle / "ru_base" / "manifest.json").read_text())
    topo_manifest = json.loads(
        (bundle / "topology" / "manifest.json").read_text()
    )
    tri_manifest = json.loads((bundle / "trimer" / "manifest.json").read_text())

    # Cases A/B/C: exactly three known RU failures, build continued
    assert ru_manifest["source_count"] == 23
    assert ru_manifest["rejected_count"] == 3
    assert ru_manifest["accepted_count"] == 20
    assert ru_manifest["source_count"] == (
        ru_manifest["accepted_count"] + ru_manifest["rejected_count"]
    )
    ru_rejections = [json.loads(line) for line in (
        bundle / "ru_base" / "rejections.jsonl"
    ).read_text(encoding="utf-8").splitlines() if line.strip()]
    codes = sorted(row["failure_code"] for row in ru_rejections)
    assert codes == ["RU_BUILD_UNSUPPORTED", "RU_DEGENERATE_DUMMY_ONLY",
                     "RU_DEGENERATE_DUMMY_ONLY"]
    assert {row["sample_key"] for row in ru_rejections} == failed_keys
    for row in ru_rejections:
        for field in LEDGER_REQUIRED_FIELDS:
            assert field in row
        assert row["layer"] == "ru_base"

    # Case E: samples after the failures were still built and accepted
    assert topo_manifest["record_count"] == 20
    assert topo_manifest["rejected_count"] == 0
    assert tri_manifest["source_count"] == 20
    assert tri_manifest["accepted_count"] + tri_manifest["rejected_count"] == 20
    assert tri_manifest["accepted_count"] >= 19

    # RU failure means downstream SKIPPED_PARENT_FAILED, not failed
    topo_rejections = (
        bundle / "topology" / "rejections.jsonl"
    ).read_text(encoding="utf-8")
    tri_rejections = (
        bundle / "trimer" / "rejections.jsonl"
    ).read_text(encoding="utf-8")
    for key in failed_keys:
        assert key not in topo_rejections
        assert key not in tri_rejections

    # the failed samples have no record in any published layer
    import numpy as np
    from src.dataset.cache_lifecycle import ReadonlyArtifact
    for layer in ("ru_base", "topology", "trimer"):
        artifact = ReadonlyArtifact(cache_root, store["artifacts"][layer])
        try:
            for key in failed_keys:
                assert bytes.fromhex(key) not in artifact
        finally:
            artifact.close()
    rejected = np.load(bundle / "ru_base" / "rejected_keys.npy")
    assert {bytes(row).hex() for row in rejected} == failed_keys


def test_unclassified_struct_failure_continues_and_publishes(tmp_path):
    """Case D: an unknown sample-local struct exception is recorded as
    UNCLASSIFIED_SAMPLE_FAILURE and the build continues (Case E)."""
    from src.dataset.cache_lifecycle import load_source_rows

    source_csv = tmp_path / "tiny.csv"
    smiles_rows = [
        f"*C{'C' * a}(C){'O' * b}C*" for a in range(3) for b in range(4)
    ]
    source_csv.write_text(
        "smiles\n" + "\n".join(smiles_rows) + "\n", encoding="utf-8"
    )
    rows, _ = load_source_rows(source_csv, 0)
    victim = rows[2]["sample_key"]
    cache_root = tmp_path / "cache"

    result = _run_build(
        cache_root, source_csv,
        env_extra={"MTS_CACHE_FAULT_UNCLASSIFIED_KEY": victim},
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]

    store = json.loads((cache_root / "store.json").read_text(encoding="utf-8"))
    bundle = cache_root / "builds" / store["bundle_hash"]
    ru_manifest = json.loads((bundle / "ru_base" / "manifest.json").read_text())
    assert ru_manifest["source_count"] == len(rows)
    assert ru_manifest["rejected_count"] == 1
    assert ru_manifest["accepted_count"] == len(rows) - 1
    assert ru_manifest["unclassified_failure_count"] == 1

    ru_rejections = (bundle / "ru_base" / "rejections.jsonl").read_text(
        encoding="utf-8"
    )
    entry = next(
        json.loads(line) for line in ru_rejections.splitlines()
        if victim in line
    )
    assert entry["failure_code"] == "UNCLASSIFIED_SAMPLE_FAILURE"
    assert entry["exception_type"] == "ValueError"
    assert "unexpected sample-local condition" in entry["exception_message"]
    assert entry["layer"] == "ru_base"
    tracebacks = (bundle / "ru_base" / "failure_tracebacks.jsonl").read_text(
        encoding="utf-8"
    )
    assert "unexpected sample-local condition" in tracebacks

    # downstream skipped for the victim; everyone else terminal
    topo_manifest = json.loads(
        (bundle / "topology" / "manifest.json").read_text()
    )
    assert topo_manifest["record_count"] == len(rows) - 1
    assert topo_manifest["rejected_count"] == 0
    tri_manifest = json.loads((bundle / "trimer" / "manifest.json").read_text())
    assert tri_manifest["source_count"] == len(rows) - 1
    assert (
        tri_manifest["accepted_count"] + tri_manifest["rejected_count"]
        == len(rows) - 1
    )


def test_writer_failure_is_system_fatal(tmp_path, monkeypatch):
    """Case F: a writer/storage failure is SYSTEM_FATAL — the whole build
    stops with an error and nothing is published."""
    import scripts.build_mts_cache as builder

    source_csv = tmp_path / "tiny.csv"
    smiles_rows = [
        f"*C{'C' * a}(C){'O' * b}C*" for a in range(3) for b in range(4)
    ]
    source_csv.write_text(
        "smiles\n" + "\n".join(smiles_rows) + "\n", encoding="utf-8"
    )
    cache_root = tmp_path / "cache"

    def broken_put(self, key, data):
        raise OSError("simulated writer/storage failure")

    monkeypatch.setattr(builder.StagingWriter, "put", broken_put)
    with pytest.raises(OSError, match="simulated writer/storage failure"):
        builder.main([
            "--cache-root", str(cache_root),
            "--source-csv", str(source_csv),
            "--limit", "0", "--workers", "1",
        ])
    assert not (cache_root / "store.json").exists()
    builds = cache_root / "builds"
    published = (
        [p for p in builds.iterdir() if not p.name.endswith(".staging")]
        if builds.is_dir() else []
    )
    assert not published, "SYSTEM_FATAL must never publish"
