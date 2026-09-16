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


def _hold_build_lock(staging, ready, release):
    from scripts.build_glt_dual_static_cache import _acquire_build_lock, _release_build_lock

    lock = _acquire_build_lock(staging)
    ready.send(True)
    release.recv()
    _release_build_lock(lock)


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
    lock = _acquire_build_lock(staging)
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


def test_single_writer_lock_blocks_concurrency_without_pid_reclaim_race(tmp_path):
    from scripts.build_glt_dual_static_cache import _acquire_build_lock, _release_build_lock

    staging = tmp_path / "staging"
    staging.mkdir()
    lock = _acquire_build_lock(staging)
    assert lock.is_file()
    with pytest.raises(CacheLifecycleError):
        _acquire_build_lock(staging)
    _release_build_lock(lock)
    assert lock.is_file()
    second = _acquire_build_lock(staging)
    assert second.is_file()
    _release_build_lock(second)

    # PID text is only provenance; stale text cannot make a live flock race
    # unsafe, and the kernel releases a held flock when its process dies.
    (staging / "build.lock").write_text("999999", encoding="utf-8")
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
    report = tmp_path / "refusal.json"
    with pytest.raises(CacheLifecycleError):
        finalize(root, diagnostic_report=report)
    assert json.loads((root / "manifest.json").read_text(encoding="utf-8")) == stale
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "REFUSED_FROZEN_ARTIFACT"
    assert "geometry_valid_count" in payload["differing_fields"]

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
            self.trimer = {key: trimer for key in keys}

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
    static_manifest_before = (static_root / "manifest.json").read_bytes()
    static_frozen_before = (static_root / ".frozen").read_bytes()

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
