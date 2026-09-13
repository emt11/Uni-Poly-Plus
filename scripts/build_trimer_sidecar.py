#!/usr/bin/env python3
"""Build, audit, freeze and publish one derived sidecar for a published MTS
bundle.

Sidecar: ``bond_angle`` — for every accepted Trimer record, all bond angles
(i, j, k) with centre atom j and bonded neighbours i < k, in degrees
(float32, ordered by centre atom then neighbour indices).  The sidecar is
content-addressed by its own build_spec and strictly bound to the parent
artifact hash, the parent accepted-key cohort and the parent build spec.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.cache_lifecycle import (  # noqa: E402
    CacheAuditError,
    CacheLifecycleError,
    PublishedCacheDataset,
    ReadonlyArtifact,
    atomic_json,
    json_hash,
    ordered_key_hash,
    sha256_file,
    sidecar_binding,
    snapshot_tree,
    validate_sidecar_binding,
)


def sidecar_build_spec() -> dict:
    return {
        "artifact_type": "bond_angle_sidecar",
        "parameters": {
            "source_records": "published_trimer_accepted",
            "angle_definition": "bonded_pair_around_centre_atom",
            "unit": "degrees",
            "dtype": "float32",
            "ordering": "centre_atom_then_neighbour_index",
            "cosine_formula": "1 - dot(u, v) where u/v are unit vectors",
        },
    }


def compute_angles(trimer) -> np.ndarray:
    positions = torch.as_tensor(trimer.trimer_pos, dtype=torch.float64)
    edges = torch.as_tensor(trimer.trimer_edge_index).long()
    if edges.numel() == 0:
        return np.zeros(0, dtype=np.float32)
    neighbours = {}
    for column in range(edges.size(1)):
        a, b = int(edges[0, column]), int(edges[1, column])
        neighbours.setdefault(a, []).append(b)
    values = []
    for centre in sorted(neighbours):
        bonded = sorted(set(neighbours[centre]))
        for left in range(len(bonded)):
            for right in range(left + 1, len(bonded)):
                u = positions[bonded[left]] - positions[centre]
                v = positions[bonded[right]] - positions[centre]
                norm_u = float(torch.linalg.vector_norm(u))
                norm_v = float(torch.linalg.vector_norm(v))
                if norm_u <= 0.0 or norm_v <= 0.0:
                    raise CacheAuditError("zero-length bond vector in sidecar")
                cosine = float(torch.dot(u / norm_u, v / norm_v))
                cosine = max(-1.0, min(1.0, cosine))
                values.append(math.degrees(math.acos(cosine)))
    return np.asarray(values, dtype=np.float32)


def _expected_metadata(store, accepted_key_hash):
    return sidecar_binding(
        store["artifacts"]["trimer"], accepted_key_hash, sidecar_build_spec()
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    args = parser.parse_args(argv)

    cache_root = args.cache_root.resolve()
    store = json.loads((cache_root / "store.json").read_text(encoding="utf-8"))
    sidecar_spec = sidecar_build_spec()
    spec_hash = json_hash(sidecar_spec)

    dataset = PublishedCacheDataset(cache_root)
    try:
        accepted_keys = np.asarray(
            np.load(dataset.trimer.root / "accepted_keys.npy", mmap_mode="r")
        )
        accepted_key_hash = ordered_key_hash(
            bytes(np.asarray(key)) for key in accepted_keys
        )
        expected_metadata = _expected_metadata(store, accepted_key_hash)

        final = args.sidecar_root.resolve() / spec_hash
        staging = args.sidecar_root.resolve() / f"{spec_hash}.staging"

        if final.is_dir():
            metadata = json.loads(
                (final / "metadata.json").read_text(encoding="utf-8")
            )
            validate_sidecar_binding(
                metadata, store["artifacts"]["trimer"], accepted_key_hash,
                sidecar_build_spec(),
            )
            frozen = json.loads((final / ".frozen").read_text(encoding="utf-8"))
            manifest = json.loads(
                (final / "manifest.json").read_text(encoding="utf-8")
            )
            if frozen != {"manifest_hash": json_hash(manifest)}:
                raise CacheLifecycleError("published sidecar .frozen mismatch")
            before = snapshot_tree(args.sidecar_root)
            values = np.load(final / "angles.f32.npy", mmap_mode="r")
            if values.shape[0] != manifest["angle_count"]:
                raise CacheLifecycleError("published sidecar data mismatch")
            del values
            if snapshot_tree(args.sidecar_root) != before:
                raise CacheLifecycleError("sidecar repeat run wrote files")
            print(json.dumps({
                "sidecar_repeat_zero_write_ok": True,
                "sidecar_hash": spec_hash,
                "record_count": manifest["record_count"],
            }))
            return 0

        staging.mkdir(parents=True, exist_ok=False)
        offsets = [0]
        angle_chunks = []
        for index in range(len(dataset)):
            key = bytes(np.asarray(accepted_keys[index]))
            trimer = dataset.trimer[key]
            angles = compute_angles(trimer)
            angle_chunks.append(angles)
            offsets.append(offsets[-1] + int(angles.size))
        keys_matrix = accepted_keys.reshape(-1, 32)
        for name, values in (
            ("sample_keys.npy", keys_matrix),
            ("angle_offsets.npy", np.asarray(offsets, dtype=np.int64)),
            ("angles.f32.npy", np.concatenate(angle_chunks) if angle_chunks
             else np.zeros(0, dtype=np.float32)),
        ):
            temporary = staging / f"{name}.tmp.{os.getpid()}"
            with temporary.open("wb") as handle:
                np.save(handle, values, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, staging / name)

        manifest = {
            "sidecar_build_spec_hash": spec_hash,
            "record_count": len(keys_matrix),
            "angle_count": int(offsets[-1]),
            "ordered_accepted_key_hash": accepted_key_hash,
            "sample_keys_file_sha256": sha256_file(staging / "sample_keys.npy"),
            "offsets_file_sha256": sha256_file(staging / "angle_offsets.npy"),
            "data_file_sha256": sha256_file(staging / "angles.f32.npy"),
        }
        # audit: coverage, monotonic offsets, finite values, parent binding
        observed_keys = np.load(staging / "sample_keys.npy", mmap_mode="r")
        if observed_keys.shape != (len(dataset), 32):
            raise CacheAuditError("sidecar key coverage failure")
        observed_offsets = np.load(staging / "angle_offsets.npy", mmap_mode="r")
        if observed_offsets.shape != (len(dataset) + 1,) \
                or int(observed_offsets[0]) != 0 \
                or not bool((np.diff(observed_offsets) >= 0).all()):
            raise CacheAuditError("sidecar offsets are not monotonic")
        observed_values = np.load(staging / "angles.f32.npy", mmap_mode="r")
        if observed_values.shape != (int(offsets[-1]),) \
                or observed_values.dtype != np.float32 \
                or not bool(np.isfinite(np.asarray(observed_values)).all()):
            raise CacheAuditError("sidecar values are invalid or non-finite")
        del observed_values
        atomic_json(staging / "metadata.json", expected_metadata)
        atomic_json(staging / "manifest.json", manifest)
        atomic_json(staging / ".frozen", {"manifest_hash": json_hash(manifest)})
        # post-freeze self-check
        reloaded = json.loads(
            (staging / "manifest.json").read_text(encoding="utf-8")
        )
        if json.loads((staging / ".frozen").read_text(encoding="utf-8")) \
                != {"manifest_hash": json_hash(reloaded)}:
            raise CacheLifecycleError("sidecar freeze self-check failed")
        validate_sidecar_binding(
            json.loads((staging / "metadata.json").read_text(encoding="utf-8")),
            store["artifacts"]["trimer"], accepted_key_hash,
            sidecar_build_spec(),
        )
        if final.exists():
            raise CacheLifecycleError(f"sidecar final exists: {final}")
        os.replace(staging, final)
        # negative check: a stale parent binding must be rejected
        stale = dict(expected_metadata)
        stale["parent_artifact_hash"] = "0" * 64
        try:
            validate_sidecar_binding(
                stale, store["artifacts"]["trimer"], accepted_key_hash,
                sidecar_build_spec(),
            )
        except CacheLifecycleError:
            pass
        else:
            raise CacheLifecycleError("stale parent binding was not rejected")
        print(json.dumps({
            "sidecar_published": True, "sidecar_hash": spec_hash,
            "record_count": manifest["record_count"],
            "angle_count": manifest["angle_count"],
            "stale_parent_rejected": True,
        }))
        return 0
    finally:
        dataset.close()


if __name__ == "__main__":
    raise SystemExit(main())
