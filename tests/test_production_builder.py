"""Production-builder integration tests.

These prove, on the real multi-worker spawn path:

1. a sample-local Trimer failure (including former contract errors) is
   recorded as UNCLASSIFIED_SAMPLE_FAILURE and the build continues and
   publishes (relaxed per-sample failure policy);
2. SIGTERM preserves staging without publishing and a re-run resumes;
3. the full-source manifest mode counts raw/unique/duplicate/invalid rows
   explicitly instead of assuming a size;
4. progress lines carry the per-layer accounting fields.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tiny_source_csv(path: Path, count: int = 120) -> list[str]:
    # 120 guaranteed-unique tiny P-SMILES (chain length x polyether length
    # decides the canonical identity); all embed in well under the budget
    smiles_rows = []
    for a in range(10):
        for b in range(12):
            smiles_rows.append(f"*C{'C' * a}(C){'O' * b}C*")
    assert len(smiles_rows) >= count
    smiles_rows = smiles_rows[:count]
    path.write_text(
        "smiles\n" + "\n".join(smiles_rows) + "\n", encoding="utf-8"
    )
    return smiles_rows


def _run_build(cache_root: Path, source_csv: Path, workers: int = 4,
               env_extra: dict | None = None, timeout: int = 600):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "."
    if env_extra:
        environment.update(env_extra)
    return subprocess.run(
        [sys.executable, "scripts/build_mts_cache.py",
         "--cache-root", str(cache_root),
         "--source-csv", str(source_csv),
         "--limit", "120", "--workers", str(workers)],
        capture_output=True, text=True, timeout=timeout, env=environment,
        cwd=str(ROOT),
    )


def _assert_unpublished(cache_root: Path):
    assert not (cache_root / "store.json").exists(), \
        "store.json must not exist after an aborted build"
    builds = cache_root / "builds"
    published = [p for p in builds.iterdir() if not p.name.endswith(".staging")] \
        if builds.is_dir() else []
    assert not published, "nothing may be published after an aborted build"


def test_trimer_sample_failure_continues_and_publishes(tmp_path):
    from src.dataset.cache_lifecycle import load_source_rows

    source_csv = tmp_path / "tiny.csv"
    _tiny_source_csv(source_csv)
    rows, _ = load_source_rows(source_csv, 120)
    fault_key = rows[3]["sample_key"]
    cache_root = tmp_path / "cache"

    result = _run_build(cache_root, source_csv,
                        env_extra={"MTS_CACHE_FAULT_SAMPLE_KEY": fault_key})
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    store = json.loads((cache_root / "store.json").read_text(encoding="utf-8"))
    bundle = cache_root / "builds" / store["bundle_hash"]
    trimer_manifest = json.loads(
        (bundle / "trimer" / "manifest.json").read_text(encoding="utf-8")
    )
    assert trimer_manifest["source_count"] == 120
    assert trimer_manifest["source_count"] == (
        trimer_manifest["accepted_count"] + trimer_manifest["rejected_count"]
    )
    assert trimer_manifest["rejected_count"] >= 1
    # the faulted sample has an explicit terminal state, never silently
    # dropped, and the failure is classified as an unclassified sample failure
    runtime = (bundle / "trimer" / "runtime.jsonl").read_text(encoding="utf-8")
    assert fault_key in runtime
    rejections = (bundle / "trimer" / "rejections.jsonl").read_text(
        encoding="utf-8"
    )
    entry = next(
        json.loads(line) for line in rejections.splitlines()
        if fault_key in line
    )
    assert entry["failure_code"] == "UNCLASSIFIED_SAMPLE_FAILURE"
    assert entry["exception_type"] == "ValueError"
    assert entry["layer"] == "trimer"
    # full traceback stored once in the dedup sidecar
    tracebacks = (bundle / "trimer" / "failure_tracebacks.jsonl").read_text(
        encoding="utf-8"
    )
    assert "unexpected sample-local condition" in tracebacks


def test_sigterm_preserves_staging_and_resume_publishes(tmp_path):
    source_csv = tmp_path / "tiny.csv"
    _tiny_source_csv(source_csv)
    cache_root = tmp_path / "cache"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "."

    process = subprocess.Popen(
        [sys.executable, "scripts/build_mts_cache.py",
         "--cache-root", str(cache_root), "--source-csv", str(source_csv),
         "--limit", "120", "--workers", "4"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=environment, cwd=str(ROOT),
    )
    # wait until the build has actually entered a phase that owns staging,
    # so the signal lands on the builder and not on interpreter imports
    deadline = time.monotonic() + 90
    staged = False
    while time.monotonic() < deadline:
        builds = cache_root / "builds"
        if builds.is_dir() and any(builds.glob("*.staging")):
            staged = True
            break
        if process.poll() is not None:
            break
        time.sleep(0.5)
    process.send_signal(signal.SIGTERM)
    try:
        process.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        process.kill()
        pytest.fail("builder ignored SIGTERM")
    assert process.returncode != 0
    _assert_unpublished(cache_root)
    if staged:
        assert any((cache_root / "builds").glob("*.staging")), \
            "staging must survive SIGTERM"

    resumed = _run_build(cache_root, source_csv)
    assert resumed.returncode == 0, resumed.stdout[-2000:]
    assert (cache_root / "store.json").is_file()


def test_full_source_manifest_counts_rows_explicitly(tmp_path):
    from src.dataset.cache_lifecycle import load_source_rows

    # 8 raw rows: 1 invalid, 1 canonical duplicate of a valid row
    rows = [
        "*CC*", "*CCC*", "not-a-valid-smiles!!!", "*CC*", "*CCCC*",
        "*CCCCC*", "bad)))smiles", "*CC(C)C*",
    ]
    source_csv = tmp_path / "mixed.csv"
    source_csv.write_text("smiles\n" + "\n".join(rows) + "\n", encoding="utf-8")
    records, manifest = load_source_rows(source_csv, 0)
    selection = manifest["selection"]
    assert manifest["source_count"] == len(records) == 5
    assert selection["policy"] == "full_unique_valid_canonical_production"
    assert selection["duplicate_candidates_excluded"] == 1
    assert selection["invalid_candidates_excluded"] == 2
    assert len({row["sample_key"] for row in records}) == 5
    # pilot (bounded) mode keeps its original manifest shape
    records_pilot, manifest_pilot = load_source_rows(source_csv, 5)
    assert manifest_pilot["selection"]["policy"] == \
        "first_unique_valid_canonical_pilot"
    assert manifest_pilot["source_count"] == 5


def test_progress_lines_are_emitted(tmp_path):
    source_csv = tmp_path / "tiny.csv"
    _tiny_source_csv(source_csv)
    cache_root = tmp_path / "cache"
    result = _run_build(cache_root, source_csv)
    assert result.returncode == 0
    assert '"phase": "struct"' in result.stdout
    assert '"phase": "trimer"' in result.stdout
    assert '"unresolved"' in result.stdout
