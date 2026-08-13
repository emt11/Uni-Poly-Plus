#!/usr/bin/env python3
"""Build frozen relation-wise Star-RBF v2 sidecars from frozen LMDB layers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_mips_trimer_cache import _specs  # noqa: E402
from src.dataset.lmdb_cache import LmdbLayerStore, load_cohort  # noqa: E402
from src.dataset.mips_trimer_contract import (  # noqa: E402
    STAR_RBF_V2_BUILDER_VERSION,
    STAR_RBF_V2_SIDECAR_SCHEMA,
)
from src.dataset.mts_star_rbf_v2 import (  # noqa: E402
    ARRAY_DTYPES,
    ARRAY_NAMES,
    build_star_rbf_v2_sample,
    rbf_upper_from_distances,
)

_TOPOLOGY = None
_TRIMER = None


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _cohort(cache_root, name):
    pointer = cache_root / "cohorts" / name / "current.json"
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    return load_cohort(pointer.parent / payload["cohort_hash"], load_text=False, verify_integrity=True)


def _source_identity():
    specs = _specs(PROJECT_ROOT)
    output = {}
    for name in ("topology", "trimer"):
        root = Path(specs[name]["root"])
        if not all((root / item).is_file() for item in (".done", ".frozen", "metadata.json", "manifest.json")):
            raise RuntimeError(f"frozen {name} cache is incomplete: {root}")
        output[name] = {
            "root": str(root),
            "done_artifact_hash": (root / ".done").read_text(encoding="utf-8").strip(),
            "done_sha256": _sha256_file(root / ".done"),
            "frozen_sha256": _sha256_file(root / ".frozen"),
            "metadata_sha256": _sha256_file(root / "metadata.json"),
            "manifest_sha256": _sha256_file(root / "manifest.json"),
        }
    return output


def _init_worker(topology_root, trimer_root):
    global _TOPOLOGY, _TRIMER
    torch.set_num_threads(1)
    _TOPOLOGY = LmdbLayerStore(topology_root, require_done=True)
    _TRIMER = LmdbLayerStore(trimer_root, require_done=True)


def _work(keys):
    return [build_star_rbf_v2_sample(bytes(key), _TOPOLOGY[key], _TRIMER[key]) for key in keys]


def _bounded_results(executor, chunks, max_pending):
    """Yield chunk results in input order with bounded executor prefetch."""
    iterator = iter(chunks)
    pending = []
    for _ in range(max_pending):
        try: pending.append(executor.submit(_work, next(iterator)))
        except StopIteration: break
    while pending:
        future = pending.pop(0)
        yield future.result()
        try: pending.append(executor.submit(_work, next(iterator)))
        except StopIteration: pass


def _append(handle, values, dtype):
    value = np.asarray(values, dtype=dtype)
    if value.size:
        handle.write(value.tobytes(order="C"))


def _materialize(raw, output, dtype, shape):
    temporary = output.with_suffix(".npy.tmp")
    result = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=shape)
    total = int(np.prod(shape, dtype=np.int64))
    offset = 0
    with raw.open("rb") as handle:
        while offset < total:
            values = np.fromfile(handle, dtype=dtype, count=min(total - offset, 1 << 20))
            if not values.size:
                raise RuntimeError(f"truncated staging payload: {raw}")
            result.reshape(-1)[offset:offset + values.size] = values
            offset += values.size
    result.flush()
    del result
    os.replace(temporary, output)


def _record_arrays(record):
    relations = record["relations"]
    pairs = record["pairs"]
    return {
        "sample_keys": np.frombuffer(record["sample_key"], dtype=np.uint8),
        "relation_row": [item["row"] for item in relations],
        "relation_pair_index": [item["pair_index"] for item in relations],
        "relation_spd": [item["spd"] for item in relations],
        "pair_key_src": [item["key"][0] for item in pairs],
        "pair_key_dst": [item["key"][1] for item in pairs],
        "pair_key_shift": [item["key"][2] for item in pairs],
        "pair_spd": [item["spd"] for item in pairs],
        "pair_relation_multiplicity": [item["multiplicity"] for item in pairs],
        "pair_path_signature_hash": np.asarray([
            np.frombuffer(item["path_signature_hash"], dtype=np.uint8) for item in pairs
        ], dtype=np.uint8).reshape(-1, 32),
        "pair_observation_distances": np.asarray([
            item["distances"] for item in pairs
        ], dtype=np.float32).reshape(-1, 2),
        "pair_observation_count": [item["observation_count"] for item in pairs],
        "pair_valid": [item["valid"] for item in pairs],
        "pair_invalid_reason_code": [item["reason_code"] for item in pairs],
        "pair_geometry_source": [item["geometry_source"] for item in pairs],
        "pair_absolute_asymmetry": [item["absolute_asymmetry"] for item in pairs],
        "pair_relative_asymmetry": [item["relative_asymmetry"] for item in pairs],
    }


def build(cohort_name, *, workers, chunk_size, max_samples=None):
    cache_root = PROJECT_ROOT / "data/processed/mips_trimer_scage"
    cohort = _cohort(cache_root, cohort_name)
    identity = _source_identity()
    keys = [bytes(row) for row in np.asarray(cohort["keys"], dtype=np.uint8)]
    if max_samples is not None:
        keys = keys[:int(max_samples)]
    binding = {
        "schema": STAR_RBF_V2_SIDECAR_SCHEMA,
        "builder_version": STAR_RBF_V2_BUILDER_VERSION,
        "cohort": cohort_name,
        "cohort_hash": cohort["manifest"]["cohort_hash"],
        "ordered_sample_key_hash": cohort["manifest"]["ordered_sample_key_hash"],
        "record_count": len(keys),
        "selection": "full_cohort" if max_samples is None else f"prefix_{len(keys)}",
        "source_identity": identity,
        "geometry_contract": {
            "definition": "trimer_periodic_relation_rbf_v2",
            "max_spd": 2, "max_abs_shift": 2,
            "shift0": "central_direct", "shift1": "dual_observation_rbf_mean",
            "shift2": "outer_trimer_direct", "trivial_self": "zero_bias",
            "asymmetry": {"stored": True, "model_input": False, "hard_filter": False},
        },
    }
    source_hash = _digest(binding)
    root = cache_root / "star_rbf_v2" / source_hash / cohort_name / cohort["manifest"]["cohort_hash"]
    if root.exists():
        if (root / ".done").is_file():
            return root
        raise RuntimeError(f"incomplete same-identity Star-RBF v2 artifact: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{cohort_name}.star-v2-", dir=root.parent))
    handles = {name: (staging / f"{name}.raw").open("wb") for name in ARRAY_NAMES}
    relation_total = pair_total = 0
    distance_raw = staging / "valid_observation_distances.raw"
    distance_handle = distance_raw.open("wb")
    valid_distance_count = 0
    try:
        _append(handles["sample_relation_offsets"], [0], ARRAY_DTYPES["sample_relation_offsets"])
        _append(handles["sample_pair_offsets"], [0], ARRAY_DTYPES["sample_pair_offsets"])
        chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker,
            initargs=(identity["topology"]["root"], identity["trimer"]["root"]),
        ) as executor:
            for records in _bounded_results(executor, chunks, max_pending=max(1, 2 * workers)):
                for record in records:
                    arrays = _record_arrays(record)
                    for name, values in arrays.items():
                        _append(handles[name], values, ARRAY_DTYPES[name])
                    relation_total += len(record["relations"])
                    pair_total += len(record["pairs"])
                    _append(handles["sample_relation_offsets"], [relation_total], ARRAY_DTYPES["sample_relation_offsets"])
                    _append(handles["sample_pair_offsets"], [pair_total], ARRAY_DTYPES["sample_pair_offsets"])
                    for pair in record["pairs"]:
                        if pair["valid"] and pair["geometry_source"] != 1:
                            values = pair["distances"][:pair["observation_count"]]
                            _append(distance_handle, values, np.dtype("<f4"))
                            valid_distance_count += len(values)
        for handle in handles.values():
            handle.close()
        distance_handle.close()
        shapes = {}
        for name in ARRAY_NAMES:
            if name == "sample_keys": shape = (len(keys), 32)
            elif name in {"sample_relation_offsets", "sample_pair_offsets"}: shape = (len(keys) + 1,)
            elif name in {"relation_row", "relation_pair_index", "relation_spd"}: shape = (relation_total,)
            elif name == "pair_path_signature_hash": shape = (pair_total, 32)
            elif name == "pair_observation_distances": shape = (pair_total, 2)
            else: shape = (pair_total,)
            shapes[name] = shape
            _materialize(staging / f"{name}.raw", staging / f"{name}.npy", ARRAY_DTYPES[name], shape)
        if cohort_name == "PI1M_v2":
            if valid_distance_count <= 0:
                raise RuntimeError("PI1M_v2 has no valid Star-RBF v2 distances")
            distance_mmap = np.memmap(
                distance_raw, dtype=np.dtype("<f4"), mode="r",
                shape=(valid_distance_count,),
            )
            upper = rbf_upper_from_distances(distance_mmap)
            del distance_mmap
        else:
            upper_file = cache_root / "star_rbf_v2" / "PI1M_v2_RBF.json"
            if not upper_file.is_file():
                raise RuntimeError("build PI1M_v2 Star-RBF v2 sidecar before downstream_union")
            upper = float(json.loads(upper_file.read_text(encoding="utf-8"))["upper"])
        semantic = dict(binding["geometry_contract"])
        semantic["rbf"] = {"num_rbf": 32, "lower": 0.0, "upper": upper, "gamma": "0.5/spacing^2"}
        metadata = dict(binding)
        metadata["model_semantic_hash"] = _digest(semantic)
        metadata["rbf"] = semantic["rbf"]
        metadata["arrays"] = {name: {
            "path": f"{name}.npy", "dtype": ARRAY_DTYPES[name].str,
            "shape": list(shapes[name]), "sha256": _sha256_file(staging / f"{name}.npy"),
        } for name in ARRAY_NAMES}
        metadata["artifact_hash"] = _digest(metadata)
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        (staging / ".done").write_text(metadata["artifact_hash"] + "\n")
        (staging / ".frozen").write_text(json.dumps({
            "schema": STAR_RBF_V2_SIDECAR_SCHEMA, "artifact_hash": metadata["artifact_hash"]
        }, sort_keys=True) + "\n")
        for raw in staging.glob("*.raw"):
            raw.unlink()
        os.replace(staging, root)
        if cohort_name == "PI1M_v2" and max_samples is None:
            pointer = cache_root / "star_rbf_v2" / "PI1M_v2_RBF.json"
            temporary = pointer.with_suffix(".tmp")
            temporary.write_text(json.dumps({"upper": upper, "artifact_hash": metadata["artifact_hash"]}, indent=2) + "\n")
            os.replace(temporary, pointer)
        return root
    except Exception:
        for handle in handles.values():
            if not handle.closed: handle.close()
        if not distance_handle.closed: distance_handle.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("PI1M_v2", "downstream_union"), required=True)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_size < 1:
        parser.error("workers and chunk-size must be positive")
    print(build(args.cohort, workers=args.workers, chunk_size=args.chunk_size, max_samples=args.max_samples))


if __name__ == "__main__":
    main()
