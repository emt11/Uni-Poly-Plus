"""Continuous cos(angle) sidecar derived from frozen Trimer and angle caches."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from .mips_trimer_contract import CACHE_CONTINUOUS_ANGLE_SCHEMA
from .trimer_angle_cache import angle_cache_root, validate_angle_cache


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def continuous_angle_root(trimer_root, cohort_hash):
    return Path(trimer_root) / "derived" / "bond_angle_continuous" / str(cohort_hash)


def build_continuous_angle_cache(cohort, trimer_store, trimer_root, trimer_artifact_hash):
    manifest = cohort["manifest"]
    categorical_root = angle_cache_root(trimer_root, manifest["cohort_hash"])
    categorical = validate_angle_cache(
        categorical_root,
        cohort_hash=manifest["cohort_hash"],
        ordered_key_hash=manifest["ordered_sample_key_hash"],
        trimer_artifact_hash=trimer_artifact_hash,
        record_count=len(cohort["keys"]),
    )
    offsets = np.load(categorical_root / "angle_offsets.npy", mmap_mode="r")
    indices = np.load(categorical_root / "angle_indices.npy", mmap_mode="r")
    root = continuous_angle_root(trimer_root, manifest["cohort_hash"])
    root.mkdir(parents=True, exist_ok=True)
    values_tmp = root / "angle_cos.npy.tmp"
    values = np.lib.format.open_memmap(
        values_tmp, mode="w+", dtype=np.float32, shape=(len(indices),)
    )
    validation = np.zeros(len(cohort["keys"]), dtype=np.bool_)
    for row, key in enumerate(cohort["keys"]):
        start, end = int(offsets[row]), int(offsets[row + 1])
        validation[row] = int.from_bytes(bytes(key)[:8], "little") % 100 == 0
        if end <= start:
            continue
        data = trimer_store[key]
        xyz = torch.as_tensor(data.trimer_pos).float()
        triplets = torch.from_numpy(np.asarray(indices[start:end], dtype=np.int64))
        left = xyz[triplets[:, 0]] - xyz[triplets[:, 1]]
        right = xyz[triplets[:, 2]] - xyz[triplets[:, 1]]
        cosine = torch.nn.functional.cosine_similarity(left, right, dim=-1).clamp(-1, 1)
        if not bool(torch.isfinite(cosine).all()):
            raise RuntimeError(f"non-finite continuous angle at cohort row {row}")
        values[start:end] = cosine.numpy().astype(np.float32, copy=False)
    values.flush()
    del values
    os.replace(values_tmp, root / "angle_cos.npy")
    for filename in ("angle_offsets.npy", "angle_indices.npy"):
        temporary = root / f"{filename}.tmp"
        shutil.copyfile(categorical_root / filename, temporary)
        os.replace(temporary, root / filename)
    for filename, array in (("validation_mask.npy", validation),):
        temporary = root / f"{filename}.tmp"
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, root / filename)
    files = {
        name: _sha(root / name) for name in (
            "angle_offsets.npy", "angle_indices.npy", "angle_cos.npy",
            "validation_mask.npy",
        )
    }
    metadata = {
        "schema": CACHE_CONTINUOUS_ANGLE_SCHEMA,
        "cohort_hash": manifest["cohort_hash"],
        "ordered_sample_key_hash": manifest["ordered_sample_key_hash"],
        "trimer_artifact_hash": str(trimer_artifact_hash),
        "categorical_angle_artifact_hash": (
            categorical_root / ".done"
        ).read_text().strip(),
        "target": "cos_bond_angle",
        "validation_policy": "sample_hash_mod_100_eq_0",
        "record_count": len(cohort["keys"]),
        "angle_count": len(indices),
        "files": files,
    }
    artifact = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    temporary = root / "metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, root / "metadata.json")
    (root / ".done.tmp").write_text(artifact + "\n")
    os.replace(root / ".done.tmp", root / ".done")
    frozen = {"schema": CACHE_CONTINUOUS_ANGLE_SCHEMA, "artifact_hash": artifact}
    (root / ".frozen.tmp").write_text(json.dumps(frozen, sort_keys=True) + "\n")
    os.replace(root / ".frozen.tmp", root / ".frozen")
    return root, metadata
