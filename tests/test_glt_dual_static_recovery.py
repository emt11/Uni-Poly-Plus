"""Fault-injection tests for static build recovery, publish identity and freezing.

Synthetic two-chunk fixtures only: no real cache, no conformer generation.
Covers the interruption points and refusal conditions listed in the cache
optimization plan (chunk half-written, chunk written without its completion
marker, static published while targets are not, same keys with a different
parent/configuration, missing/truncated/mis-aligned payloads, repeated calls,
concurrent writers and finalize-on-frozen).
"""

import json
import multiprocessing as mp
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from src.dataset.cache_lifecycle import CacheLifecycleError, json_hash
from src.dataset.glt_dual_cache import ordered_key_hash
from src.dataset.glt_dual_static import DualStaticCache, load_chunk_payload, write_chunk


def _row(*, geometry_valid=True, reason="", n_line=2, n_token=3, n_bond=2,
         n_distance=1, n_angle=1):
    return {
        "bond_path_features": np.zeros((n_bond, 2, 14), dtype=np.float32),
        "bond_path_mask": np.ones((n_bond, 2), dtype=bool),
        "token_pos_index_a": np.arange(n_token, dtype=np.int32),
        "token_pos_index_b": np.arange(n_token, dtype=np.int32),
        "token_z_a": np.ones(n_token, dtype=np.uint8),
        "token_z_b": np.ones(n_token, dtype=np.uint8),
        "token_bond_type": np.ones(n_token, dtype=np.uint8),
        "token_center_mask": np.zeros(n_token, dtype=bool),
        "line_source": np.zeros(n_line, dtype=np.int32),
        "line_target": np.zeros(n_line, dtype=np.int32),
        "line_path": np.zeros((n_line, 3), dtype=np.int32),
        "line_path_mask": np.ones((n_line, 2), dtype=bool),
        "line_path_group": np.zeros(n_line, dtype=np.int32),
        "line_is_self": np.zeros(n_line, dtype=bool),
        "angle_pos_triplet": np.zeros((n_line, 2, 3), dtype=np.int32),
        "distance_token_index": np.zeros(n_distance, dtype=np.int32),
        "angle_pairs": np.zeros((n_angle, 2), dtype=np.int32),
        "geometry_valid": geometry_valid,
        "geometry_invalid_reason": reason,
    }


def _target_row():
    return {"brics_groups": [np.asarray([0, 1], dtype=np.int32)],
            "fingerprint_packed": np.zeros(256, dtype=np.uint8)}


def _hold_build_lock(artifact_root, ready, release):
    from scripts.build_glt_dual_static_cache import _acquire_build_lock, _release_build_lock

    lock = _acquire_build_lock(artifact_root)
    ready.send(True)
    release.recv()
    _release_build_lock(lock)


def _run_builder_process(args_payload, result_pipe, ready=None, proceed=None):
    """Run the real builder in a child, optionally pausing before lock acquire."""

    from scripts import build_glt_dual_static_cache as builder

    if ready is not None:
        original_acquire = builder._acquire_build_lock

        def pause_before_lock(root):
            ready.send("preflight_complete")
            proceed.recv()
            return original_acquire(root)

        builder._acquire_build_lock = pause_before_lock
    try:
        result_pipe.send(("ok", builder.build(SimpleNamespace(**args_payload))))
    except BaseException as exc:  # pragma: no cover - surfaced by the parent assertion
        result_pipe.send(("error", type(exc).__name__, str(exc)))


def _publish(root, chunk_counts, *, fmt="glt-dual-static-v1", targets=False,
             cohort=None, params=None, keys=None):
    """Write chunks plus a frozen manifest exactly like the builder does."""

    root = Path(root)
    cohort = cohort or _cohort()
    keys = keys or [bytes([97 + index]) * 32 for index in range(sum(chunk_counts))]
    items = []
    start = 0
    for count in chunk_counts:
        rows = [_target_row() for _ in range(count)] if targets else [_row() for _ in range(count)]
        manifest = write_chunk(root, start, rows, targets=targets)
        items.append({"start": start, "count": count,
                      "path": f"chunks/chunk_{start:08d}", "arrays": manifest["arrays"]})
        start += count
    key_array = np.frombuffer(b"".join(keys), dtype=np.uint8).reshape(-1, 32)
    np.save(root / "sample_keys.npy", key_array)
    manifest = {"format": fmt, "sample_count": sum(chunk_counts), "chunks": items,
                "parent_bundle_hash": cohort["manifest"]["main_bundle_hash"],
                "cohort_manifest_hash": cohort["manifest_hash"],
                "build_parameters": params if params is not None else {},
                "ordered_sample_key_hash": ordered_key_hash(key_array)}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / ".frozen").write_text(json.dumps({"manifest_hash": json_hash(manifest)}), encoding="utf-8")
    return manifest


def _cohort(manifest_hash="b" * 64, parent="a" * 64):
    return {"manifest": {"main_bundle_hash": parent}, "manifest_hash": manifest_hash}


def test_write_chunk_is_atomic_and_quarantines_only_owned_leftovers(tmp_path):
    root = tmp_path / "artifact.staging"
    manifest = write_chunk(root, 0, [_row(), _row()])
    chunk = root / "chunks" / "chunk_00000000"
    assert (chunk / ".complete").is_file()
    assert not list((root / "chunks").glob(".tmp_chunk_*"))

    # An interrupted payload write and a chunk written without its marker are
    # both refused when their ownership is not confirmed...
    leftover = root / "chunks" / ".tmp_chunk_00000002"
    leftover.mkdir(parents=True)
    np.save(leftover / "bond_path_features.npy", np.zeros(1, dtype=np.float32))
    with pytest.raises(FileExistsError):
        write_chunk(root, 2, [_row()])
    # ...and quarantined (never deleted) once the caller owns this staging.
    manifest2 = write_chunk(root, 2, [_row()], quarantine_root=root / ".interrupted")
    assert manifest2["count"] == 1
    quarantined = sorted(path.name for path in (root / ".interrupted").iterdir())
    assert any(name.startswith(".tmp_chunk_00000002.attempt_") for name in quarantined)

    # A completed chunk is never overwritten, not even with a quarantine root.
    with pytest.raises(FileExistsError):
        write_chunk(root, 0, [_row()], quarantine_root=root / ".interrupted")
    assert (chunk / "manifest.json").read_text(encoding="utf-8") == json.dumps(manifest, sort_keys=True) \
        or json.loads((chunk / "manifest.json").read_text(encoding="utf-8")) == manifest

    # A complete temporary directory left just before rename is promoted after
    # its payload contract is revalidated, rather than being rejected as a
    # duplicate or rebuilt blindly.
    source = tmp_path / "complete_source"
    write_chunk(source, 2, [_row()])
    recover = tmp_path / "recover"
    (recover / "chunks").mkdir(parents=True)
    shutil.copytree(source / "chunks" / "chunk_00000002",
                    recover / "chunks" / ".tmp_chunk_00000002")
    recovered = write_chunk(recover, 2, [_row()])
    assert recovered["start"] == 2
    assert (recover / "chunks" / "chunk_00000002" / ".complete").is_file()
    assert not (recover / "chunks" / ".tmp_chunk_00000002").exists()


def test_loader_rejects_missing_truncated_and_misaligned_payloads(tmp_path):
    root = tmp_path / "static"
    _publish(root, [2, 2])
    cache = DualStaticCache(root)
    assert cache.get(0)["geometry_valid"] is True

    other = tmp_path / "static_missing"
    _publish(other, [2])
    (other / "chunks" / "chunk_00000000" / "line_path.npy").unlink()
    with pytest.raises(CacheLifecycleError):
        DualStaticCache(other).get(0)

    truncated = tmp_path / "static_truncated"
    _publish(truncated, [2])
    np.save(truncated / "chunks" / "chunk_00000000" / "token_pos_index_a.npy",
            np.zeros(1, dtype=np.int32))
    with pytest.raises(CacheLifecycleError):
        DualStaticCache(truncated).get(0)

    misaligned = tmp_path / "static_offsets"
    _publish(misaligned, [2])
    item = json.loads((misaligned / "chunks" / "chunk_00000000" / "manifest.json").read_text())
    payload = misaligned / "chunks" / "chunk_00000000" / "bond_path_offsets.npy"
    bad = np.asarray([0, 1, 1], dtype=np.int64)  # last entry no longer covers the payload
    np.save(payload, bad)
    item["arrays"]["bond_path_offsets"]["shape"] = list(bad.shape)
    with pytest.raises(CacheLifecycleError):
        load_chunk_payload(misaligned / "chunks" / "chunk_00000000", item)

    nonmonotone = tmp_path / "static_nonmonotone"
    _publish(nonmonotone, [2])
    table = np.asarray([0, 2, 1], dtype=np.int64)
    np.save(nonmonotone / "chunks" / "chunk_00000000" / "bond_path_offsets.npy", table)
    with pytest.raises(CacheLifecycleError):
        DualStaticCache(nonmonotone).get(0)


def test_builder_resume_validates_completed_chunk_payload(tmp_path):
    from scripts.build_glt_dual_static_cache import _existing_chunk

    root = tmp_path / "resume"
    manifest = write_chunk(root, 0, [_row(), _row()])
    item = root / "chunks" / "chunk_00000000" / "token_pos_index_a.npy"
    np.save(item, np.zeros(1, dtype=np.int32))
    with pytest.raises(CacheLifecycleError):
        _existing_chunk(root, 0, 2, False)
    assert manifest["count"] == 2


def test_staging_resume_requires_matching_build_context(tmp_path):
    from scripts.build_glt_dual_static_cache import (
        _acquire_build_lock, _build_context, _initialize_staging, _prepare_artifact,
        _release_build_lock,
    )

    root = tmp_path / "out" / "static"
    staging = root.with_name(root.name + ".staging")
    keys = [b"k" * 32, b"j" * 32]
    params = {"chunk_size": 2, "unique": False, "limit": 0, "targets": False}
    _, staging, _ = _prepare_artifact(root, "glt-dual-static-v1", keys, _cohort(), params)
    lock = _acquire_build_lock(root)
    _initialize_staging(staging, keys, _build_context("glt-dual-static-v1", keys, _cohort(), params))
    _release_build_lock(lock)
    assert (staging / "build_context.json").is_file()

    # Same keys, different cohort or parent bundle must not resume.
    with pytest.raises(CacheLifecycleError):
        _prepare_artifact(root, "glt-dual-static-v1", keys, _cohort(manifest_hash="d" * 64), params)
    with pytest.raises(CacheLifecycleError):
        _prepare_artifact(root, "glt-dual-static-v1", keys, _cohort(parent="e" * 64), params)
    with pytest.raises(CacheLifecycleError):
        _prepare_artifact(root, "glt-dual-static-v1", keys, _cohort(),
                          dict(params, chunk_size=4))
    with pytest.raises(CacheLifecycleError):
        _prepare_artifact(root, "glt-dual-static-v1", keys, _cohort(),
                          dict(params, targets=True))
    # Identical identity resumes.
    _, _, published = _prepare_artifact(root, "glt-dual-static-v1", keys, _cohort(), params)
    assert published is None

    unknown = tmp_path / "unknown" / "static"
    unknown_staging = unknown.with_name(unknown.name + ".staging")
    unknown_staging.mkdir(parents=True)
    (unknown_staging / "mystery.bin").write_bytes(b"not a cache")
    with pytest.raises(CacheLifecycleError, match="no build context"):
        _prepare_artifact(unknown, "glt-dual-static-v1", keys, _cohort(), params)


def test_published_artifact_identity_is_checked_and_never_rewritten(tmp_path):
    from scripts.build_glt_dual_static_cache import _prepare_artifact, _published_identity

    keys = [b"k" * 32, b"j" * 32]
    params = {"chunk_size": 2, "unique": False, "limit": 0, "targets": False}
    root = tmp_path / "published"
    manifest = _publish(root, [2], keys=keys, params=params)
    identity = _published_identity(root)
    assert identity["cohort_manifest_hash"] == manifest["cohort_manifest_hash"]
    # The published side is reused only by a build with the same context.
    _, _, published = _prepare_artifact(
        root, "glt-dual-static-v1", keys, _cohort(), params)
    assert published is not None
    with pytest.raises(CacheLifecycleError):
        _prepare_artifact(root, "glt-dual-static-v1", keys, _cohort(parent="f" * 64), params)

    # A published tree whose .frozen does not bind its manifest is refused.
    broken = tmp_path / "broken"
    _publish(broken, [2])
    (broken / ".frozen").write_text(json.dumps({"manifest_hash": "0" * 64}), encoding="utf-8")
    with pytest.raises(CacheLifecycleError):
        _published_identity(broken)
    (broken / "manifest.json").unlink()
    with pytest.raises(CacheLifecycleError):
        _published_identity(broken)

    # A matching manifest/frozen pair is still not reusable when a published
    # payload is truncated.  The builder must validate the side before
    # idempotent or single-sided reuse.
    broken_payload = tmp_path / "broken_payload"
    _publish(broken_payload, [2])
    np.save(
        broken_payload / "chunks" / "chunk_00000000" / "token_pos_index_a.npy",
        np.zeros(1, dtype=np.int32),
    )
    with pytest.raises(CacheLifecycleError):
        _published_identity(broken_payload)

    broken_target = tmp_path / "broken_target"
    _publish(broken_target, [2], fmt="glt-dual-pretrain-targets-v1", targets=True)
    np.save(
        broken_target / "chunks" / "chunk_00000000" / "fingerprint_packed.npy",
        np.zeros((1, 256), dtype=np.uint8),
    )
    with pytest.raises(CacheLifecycleError):
        _published_identity(broken_target)


def test_single_writer_lock_blocks_concurrency_without_pid_reclaim_race(tmp_path):
    from scripts.build_glt_dual_static_cache import _acquire_build_lock, _release_build_lock

    staging = tmp_path / "staging"
    staging.mkdir()
    lock = _acquire_build_lock(staging)
    assert lock.is_file()
    assert lock.path == staging.with_name(staging.name + ".build.lock")
    with pytest.raises(CacheLifecycleError):
        _acquire_build_lock(staging)
    _release_build_lock(lock)
    assert lock.is_file()
    second = _acquire_build_lock(staging)
    assert second.is_file()
    _release_build_lock(second)

    # The stable anchor is never deleted or recreated to recover a lock.
    assert lock.path.is_file()
    third = _acquire_build_lock(staging)
    assert third.is_file()
    _release_build_lock(third)


def test_single_writer_lock_is_enforced_between_processes(tmp_path):
    from scripts.build_glt_dual_static_cache import _acquire_build_lock, _release_build_lock

    staging = tmp_path / "cross_process_staging"
    ready_parent, ready_child = mp.Pipe(duplex=False)
    release_child, release_parent = mp.Pipe(duplex=False)
    process = mp.get_context("fork").Process(
        target=_hold_build_lock, args=(str(staging), ready_child, release_child)
    )
    process.start()
    try:
        assert ready_parent.recv() is True
        with pytest.raises(CacheLifecycleError):
            _acquire_build_lock(staging)
        release_parent.send(True)
        process.join(timeout=5)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            release_parent.send(True)
            process.join(timeout=5)
    lock = _acquire_build_lock(staging)
    _release_build_lock(lock)


def test_static_then_targets_lock_order_releases_partial_acquisition(tmp_path, monkeypatch):
    from scripts import build_glt_dual_static_cache as builder

    static_root = tmp_path / "static"
    target_root = tmp_path / "targets"
    original = builder._acquire_build_lock
    calls = []

    def fail_targets(root):
        root = Path(root).resolve()
        calls.append(root)
        if root == target_root.resolve():
            raise CacheLifecycleError("synthetic target lock failure")
        return original(root)

    monkeypatch.setattr(builder, "_acquire_build_lock", fail_targets)
    with pytest.raises(CacheLifecycleError, match="synthetic target lock failure"):
        builder._acquire_side_locks(static_root, target_root)
    assert calls == [static_root.resolve(), target_root.resolve()]

    # The first lock was released after the second acquisition failed.
    lock = original(static_root)
    builder._release_build_lock(lock)


def test_snapshot_failure_releases_both_side_locks(tmp_path, monkeypatch):
    from scripts import build_glt_dual_static_cache as builder

    key = b"s" * 32
    key_array = np.frombuffer(key, dtype=np.uint8).reshape(1, 32)
    cohort = {
        "records": [{"sample_key": key.hex(), "source_smiles": "*CC*",
                     "normalized_smiles": "*CC*"}],
        "manifest": {"main_bundle_hash": "a" * 64,
                      "ordered_sample_key_hash": ordered_key_hash(key_array)},
        "manifest_hash": "b" * 64,
    }
    monkeypatch.setattr(builder, "load_dual_cohort", lambda *_args, **_kwargs: cohort)
    monkeypatch.setattr(builder, "load_active_dual_store",
                        lambda *_args, **_kwargs: {"bundle_hash": "a" * 64})

    def fail_snapshot(_root):
        raise OSError("synthetic snapshot failure")

    monkeypatch.setattr(builder, "zero_write_snapshot", fail_snapshot)
    cache_root = tmp_path / "main"
    cache_root.mkdir()
    static_root = tmp_path / "static"
    target_root = tmp_path / "targets"
    args = SimpleNamespace(
        cache_root=str(cache_root), cohort_root="unused", output_root=str(static_root),
        target_root=str(target_root), build_targets=True, workers=1, chunk_size=1,
        limit=0, unique=False, progress_chunks=1,
    )
    with pytest.raises(OSError, match="synthetic snapshot failure"):
        builder.build(args)

    # The snapshot failed immediately after both stable locks were acquired;
    # both must nevertheless be available again, without process teardown.
    static_lock = builder._acquire_build_lock(static_root)
    target_lock = builder._acquire_build_lock(target_root)
    builder._release_build_lock(target_lock)
    builder._release_build_lock(static_lock)


def test_build_rechecks_after_stable_lock_when_publish_wins_race(tmp_path, monkeypatch):
    """A preflight loser must return idempotent after another process publishes."""

    from scripts import build_glt_dual_static_cache as builder
    from test_complete_trimer_glt import _toy_pair
    from src.dataset.canonical_periodic import build_canonical_periodic_topology

    smiles = "*CC*"
    topology = build_canonical_periodic_topology(smiles)
    _, trimer = _toy_pair(smiles)
    key = b"r" * 32
    cohort = {
        "records": [{"sample_key": key.hex(), "source_smiles": smiles,
                     "normalized_smiles": smiles}],
        "manifest": {"main_bundle_hash": "a" * 64,
                      "ordered_sample_key_hash": ordered_key_hash(np.frombuffer(key, dtype=np.uint8).reshape(1, 32))},
        "manifest_hash": "b" * 64,
    }

    class FakeBundle:
        def __init__(self, *_args, **_kwargs):
            self.topology = {key: topology}
            self.trimer = {key: trimer}

        def close(self):
            pass

    monkeypatch.setattr(builder, "load_dual_cohort", lambda *_args, **_kwargs: cohort)
    monkeypatch.setattr(builder, "load_active_dual_store",
                        lambda *_args, **_kwargs: {"bundle_hash": "a" * 64})
    monkeypatch.setattr(builder, "DualFrozenBundle", FakeBundle)

    cache_root = tmp_path / "main"
    cache_root.mkdir()
    static_root = tmp_path / "race_static"
    args_payload = {
        "cache_root": str(cache_root), "cohort_root": "unused",
        "output_root": str(static_root), "target_root": None,
        "build_targets": False, "workers": 1, "chunk_size": 1,
        "limit": 0, "unique": False, "progress_chunks": 1,
    }
    context = mp.get_context("fork")
    ready_parent, ready_child = context.Pipe(duplex=False)
    proceed_child, proceed_parent = context.Pipe(duplex=False)
    second_result_parent, second_result_child = context.Pipe(duplex=False)
    first_result_parent, first_result_child = context.Pipe(duplex=False)
    second = context.Process(
        target=_run_builder_process,
        args=(args_payload, second_result_child, ready_child, proceed_child),
    )
    first = context.Process(
        target=_run_builder_process,
        args=(args_payload, first_result_child),
    )
    second.start()
    try:
        assert ready_parent.recv() == "preflight_complete"
        # The second process has completed its unlocked read.  The first now
        # publishes, deterministically, before the second acquires the stable
        # lock and performs its mandatory post-lock recheck.
        first.start()
        first.join(timeout=15)
        assert first.exitcode == 0
        first_message = first_result_parent.recv()
        assert first_message[0] == "ok", first_message
        assert first_message[1]["status"] == "PASS"
        manifest_before = (static_root / "manifest.json").read_bytes()
        frozen_before = (static_root / ".frozen").read_bytes()
        proceed_parent.send(True)
        second.join(timeout=15)
        assert second.exitcode == 0
        second_message = second_result_parent.recv()
        assert second_message[0] == "ok", second_message
        assert second_message[1]["status"] == "IDEMPOTENT"
        assert (static_root / "manifest.json").read_bytes() == manifest_before
        assert (static_root / ".frozen").read_bytes() == frozen_before
        assert not static_root.with_name(static_root.name + ".staging").exists()
        assert builder._build_lock_path(static_root).is_file()
    finally:
        if second.is_alive():
            proceed_parent.send(True)
            second.join(timeout=15)
        if first.is_alive():
            first.join(timeout=15)
        if second.is_alive():
            second.terminate()
            second.join(timeout=5)
        if first.is_alive():
            first.terminate()
            first.join(timeout=5)


def test_finalize_refuses_to_modify_a_frozen_artifact(tmp_path):
    from scripts.finalize_glt_dual_static_artifact import finalize

    root = tmp_path / "frozen"
    manifest = _publish(root, [2, 2])
    manifest["geometry_valid_count"] = 4
    manifest["geometry_invalid_reason_counts"] = {"": 4}
    manifest["build_spec_hash"] = "spec"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / ".frozen").write_text(json.dumps({"manifest_hash": json_hash(manifest)}), encoding="utf-8")
    before = (root / "manifest.json").read_text(encoding="utf-8")

    final, outcome = finalize(root)
    assert outcome == "IDEMPOTENT_ALREADY_FINAL"
    assert (root / "manifest.json").read_text(encoding="utf-8") == before

    # A frozen artifact that is not yet final must be refused, not rewritten.
    stale = dict(manifest)
    stale.pop("geometry_valid_count")
    stale.pop("geometry_invalid_reason_counts")
    (root / "manifest.json").write_text(json.dumps(stale), encoding="utf-8")
    (root / ".frozen").write_text(json.dumps({"manifest_hash": json_hash(stale)}), encoding="utf-8")
    protected_before = {
        name: (root / name).read_bytes() for name in ("manifest.json", ".frozen")
    }
    manifest_alias = tmp_path / "manifest_alias.json"
    manifest_alias.symlink_to(root / "manifest.json")
    root_alias = tmp_path / "artifact_alias"
    root_alias.symlink_to(root, target_is_directory=True)
    for invalid_report in (
        root / "manifest.json",
        root / ".frozen",
        root / "other-diagnostic.json",
        manifest_alias,
        root_alias / "diagnostic.json",
    ):
        with pytest.raises(CacheLifecycleError, match="protected artifact root"):
            finalize(root, diagnostic_report=invalid_report)
        assert {name: (root / name).read_bytes() for name in protected_before} == protected_before
    assert not (root / "other-diagnostic.json").exists()

    report = tmp_path / "refusal.json"
    with pytest.raises(CacheLifecycleError):
        finalize(root, diagnostic_report=report)
    assert json.loads((root / "manifest.json").read_text(encoding="utf-8")) == stale
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "REFUSED_FROZEN_ARTIFACT"
    assert "geometry_valid_count" in payload["differing_fields"]

    external_dir = tmp_path / "external-diagnostics"
    external_dir.mkdir()
    external_report = external_dir / "refusal.json"
    with pytest.raises(CacheLifecycleError):
        finalize(root, diagnostic_report=external_report)
    assert json.loads(external_report.read_text(encoding="utf-8"))["status"] == "REFUSED_FROZEN_ARTIFACT"
    assert {name: (root / name).read_bytes() for name in protected_before} == protected_before

    unfrozen = tmp_path / "unfrozen"
    _publish(unfrozen, [2])
    (unfrozen / ".frozen").unlink()
    finalized, outcome = finalize(unfrozen)
    assert outcome == "PUBLISHED_UNFROZEN"
    assert finalized["geometry_valid_count"] == 2
    assert (unfrozen / ".frozen").is_file()

    # Finalization validates payloads before creating a new frozen binding.
    broken_unfrozen = tmp_path / "broken_unfrozen"
    _publish(broken_unfrozen, [2])
    (broken_unfrozen / ".frozen").unlink()
    (broken_unfrozen / "chunks" / "chunk_00000000" / "line_path.npy").unlink()
    with pytest.raises(CacheLifecycleError):
        finalize(broken_unfrozen)
    assert not (broken_unfrozen / ".frozen").exists()

    # Legacy target manifests did not carry the outer target flag; finalizer
    # still validates their payload without rewriting the historical metadata.
    legacy_target = tmp_path / "legacy_target"
    _publish(legacy_target, [2], fmt="glt-dual-pretrain-targets-v1", targets=True)
    (legacy_target / ".frozen").unlink()
    finalized_target, target_outcome = finalize(legacy_target)
    assert target_outcome == "PUBLISHED_UNFROZEN"
    assert finalized_target["format"] == "glt-dual-pretrain-targets-v1"


def test_publish_state_covers_single_sided_publication(tmp_path):
    """Static and targets are separate roots: only the missing side is rebuilt."""

    from scripts.build_glt_dual_static_cache import _publish_plan

    assert _publish_plan(static_published=False, targets_requested=True,
                         targets_published=False) == ("none", True, True)
    assert _publish_plan(static_published=False, targets_requested=False,
                         targets_published=False) == ("none", True, False)
    # static published, targets not yet: resume only the target side
    assert _publish_plan(static_published=True, targets_requested=True,
                         targets_published=False) == ("static_only", False, True)
    # targets published, static not yet: resume only the static side
    assert _publish_plan(static_published=False, targets_requested=True,
                         targets_published=True) == ("target_only", True, False)
    # both published: idempotent, nothing left to build
    assert _publish_plan(static_published=True, targets_requested=True,
                         targets_published=True) == ("both", False, False)
    assert _publish_plan(static_published=True, targets_requested=False,
                         targets_published=False) == ("static_only", False, False)


def test_full_build_path_is_idempotent_and_resumes_missing_target(tmp_path, monkeypatch):
    """Exercise build() with deterministic workers, including static-only resume."""

    from scripts import build_glt_dual_static_cache as builder
    from test_complete_trimer_glt import _toy_pair
    from src.dataset.canonical_periodic import build_canonical_periodic_topology

    smiles = "*CC*"
    topology = build_canonical_periodic_topology(smiles)
    _, trimer = _toy_pair(smiles)
    invalid_trimer = trimer.clone()
    invalid_trimer.trimer_geometry_valid = False
    keys = [bytes([97 + index]) * 32 for index in range(2)]
    key_array = np.frombuffer(b"".join(keys), dtype=np.uint8).reshape(-1, 32)
    cohort = {
        "records": [{"sample_key": key.hex(), "source_smiles": smiles,
                     "normalized_smiles": smiles} for key in keys],
        "manifest": {"main_bundle_hash": "a" * 64,
                      "ordered_sample_key_hash": ordered_key_hash(key_array)},
        "manifest_hash": "b" * 64,
    }

    class FakeBundle:
        def __init__(self, *_args, **_kwargs):
            self.topology = {key: topology for key in keys}
            self.trimer = {keys[0]: trimer, keys[1]: invalid_trimer}

        def close(self):
            pass

    monkeypatch.setattr(builder, "load_dual_cohort", lambda *_args, **_kwargs: cohort)
    monkeypatch.setattr(builder, "load_active_dual_store",
                        lambda *_args, **_kwargs: {"bundle_hash": "a" * 64})
    monkeypatch.setattr(builder, "DualFrozenBundle", FakeBundle)
    cache_root = tmp_path / "main"
    cache_root.mkdir()
    static_root = tmp_path / "static"
    target_root = tmp_path / "targets"
    args_static = SimpleNamespace(
        cache_root=str(cache_root), cohort_root="unused", output_root=str(static_root),
        target_root=None, build_targets=False, workers=1, chunk_size=1, limit=0,
        unique=False, progress_chunks=1,
    )
    first = builder.build(args_static)
    assert first["status"] == "PASS"
    static_manifest = json.loads((static_root / "manifest.json").read_text(encoding="utf-8"))
    assert static_manifest["geometry_valid_count"] == 1
    assert static_manifest["geometry_invalid_reason_counts"] == {
        "": 1, "geometry_invalid": 1
    }
    static_manifest_before = (static_root / "manifest.json").read_bytes()
    static_frozen_before = (static_root / ".frozen").read_bytes()

    from scripts.finalize_glt_dual_static_artifact import finalize
    finalized, outcome = finalize(static_root)
    assert outcome == "IDEMPOTENT_ALREADY_FINAL"
    assert finalized == static_manifest
    assert (static_root / "manifest.json").read_bytes() == static_manifest_before
    assert (static_root / ".frozen").read_bytes() == static_frozen_before

    # A repeated complete build is a no-op, including no new staging tree.
    repeated = builder.build(args_static)
    assert repeated["status"] == "IDEMPOTENT"
    assert (static_root / "manifest.json").read_bytes() == static_manifest_before
    assert (static_root / ".frozen").read_bytes() == static_frozen_before
    assert not (static_root.with_name("static.staging")).exists()

    # Build the missing target side while reusing the immutable static side.
    args_both = SimpleNamespace(
        cache_root=str(cache_root), cohort_root="unused", output_root=str(static_root),
        target_root=str(target_root), build_targets=True, workers=1, chunk_size=1,
        limit=0, unique=False, progress_chunks=1,
    )
    resumed = builder.build(args_both)
    assert resumed["status"] == "PASS"
    assert resumed["published_at_start"] == {"static": True, "targets": False}
    assert (target_root / ".frozen").is_file()
    assert (static_root / "manifest.json").read_bytes() == static_manifest_before
    assert (static_root / ".frozen").read_bytes() == static_frozen_before
    target_manifest = json.loads((target_root / "manifest.json").read_text(encoding="utf-8"))
    assert "geometry_valid_count" not in target_manifest
    target_manifest_before = (target_root / "manifest.json").read_bytes()
    target_frozen_before = (target_root / ".frozen").read_bytes()
    _, target_outcome = finalize(target_root)
    assert target_outcome == "IDEMPOTENT_ALREADY_FINAL"
    assert (target_root / "manifest.json").read_bytes() == target_manifest_before
    assert (target_root / ".frozen").read_bytes() == target_frozen_before

    # Published-side reuse validates payloads before the idempotent branch;
    # either side being damaged is a hard refusal, not a target-only/static-only
    # continuation.
    target_payload_path = target_root / "chunks" / "chunk_00000000" / "fingerprint_packed.npy"
    target_payload = np.load(target_payload_path).copy()
    np.save(target_payload_path, np.zeros((2, 256), dtype=np.uint8))
    with pytest.raises(CacheLifecycleError):
        builder.build(args_both)
    np.save(target_payload_path, target_payload)

    static_payload_path = static_root / "chunks" / "chunk_00000000" / "token_pos_index_a.npy"
    static_payload = np.load(static_payload_path).copy()
    np.save(static_payload_path, np.zeros(1, dtype=np.int32))
    with pytest.raises(CacheLifecycleError):
        builder.build(args_both)
    np.save(static_payload_path, static_payload)
