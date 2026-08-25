#!/usr/bin/env python3
"""Build GraphGate central-policy line-geometry sidecars from frozen caches."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_mips_trimer_cache import _specs  # noqa: E402
from src.dataset.lmdb_cache import LmdbLayerStore, load_cohort  # noqa: E402
from src.dataset.periodic_line_glt_central import (  # noqa: E402
    ARRAY_DTYPES, ARRAY_NAMES, BUILDER_VERSION, SIDECAR_SCHEMA,
    build_periodic_line_central_sample,
)

_TOPOLOGY = _TRIMER = None


def _cohort(cache_root, name):
    pointer = cache_root / "cohorts" / name / "current.json"
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    return load_cohort(pointer.parent / payload["cohort_hash"], load_text=False, verify_integrity=False)


def _init_worker(topology_root, trimer_root):
    global _TOPOLOGY, _TRIMER
    torch.set_num_threads(1)
    _TOPOLOGY = LmdbLayerStore(topology_root, require_done=True)
    _TRIMER = LmdbLayerStore(trimer_root, require_done=True)


def _work(keys):
    return [build_periodic_line_central_sample(bytes(key), _TOPOLOGY[key], _TRIMER[key]) for key in keys]


def _append(handle, values, dtype):
    array = np.asarray(values, dtype=dtype)
    if array.size:
        handle.write(array.tobytes(order="C"))


def _materialize(raw, output, dtype, shape):
    temporary = output.with_suffix(".npy.tmp")
    target = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=shape)
    flat = target.reshape(-1)
    offset = 0
    with raw.open("rb") as handle:
        while offset < flat.size:
            values = np.fromfile(handle, dtype=dtype, count=min(flat.size - offset, 1 << 20))
            if not values.size:
                raise RuntimeError(f"truncated staging payload: {raw}")
            flat[offset:offset + values.size] = values
            offset += values.size
    target.flush()
    del target
    os.replace(temporary, output)


def _record_arrays(record):
    tokens, relations = record["tokens"], record["relations"]
    return {
        "sample_keys": np.frombuffer(record["sample_key"], dtype=np.uint8),
        "graph_base_mapping_valid": [record["base_mapping_valid"]],
        "graph_runtime_valid": [record["runtime_valid"]],
        "graph_invalid_reason": [record["invalid_reason"]],
        "token_atom_a": [x["atom_a"] for x in tokens],
        "token_atom_b": [x["atom_b"] for x in tokens],
        "token_shift": [x["shift"] for x in tokens],
        "token_endpoint_z_a": [x["z_a"] for x in tokens],
        "token_endpoint_z_b": [x["z_b"] for x in tokens],
        "token_bond_type": [x["bond_type"] for x in tokens],
        "token_label": [x["label"] for x in tokens],
        "token_observation_distances": np.asarray([x["distances"] for x in tokens], np.float32).reshape(-1, 3),
        "token_observation_valid": np.asarray([x["observation_valid"] for x in tokens], bool).reshape(-1, 3),
        "token_observation_translation": np.asarray([x["translations"] for x in tokens], np.int8).reshape(-1, 3),
        "token_observation_q_a": np.asarray([x["q_a"] for x in tokens], np.int8).reshape(-1, 3),
        "token_observation_q_b": np.asarray([x["q_b"] for x in tokens], np.int8).reshape(-1, 3),
        "token_observation_count": [x["observation_count"] for x in tokens],
        "token_runtime_valid": [x["runtime_valid"] for x in tokens],
        "relation_source": [x["source"] for x in relations],
        "relation_target": [x["target"] for x in relations],
        "relation_center_atom": [x["center_atom"] for x in relations],
        "relation_outer_offset_a": [x["outer_offset_a"] for x in relations],
        "relation_outer_offset_b": [x["outer_offset_b"] for x in relations],
        "relation_span": [x["span"] for x in relations],
        "relation_multiplicity": [x["multiplicity"] for x in relations],
        "relation_observation_angles": np.asarray([x["angles"] for x in relations], np.float32).reshape(-1, 3),
        "relation_observation_valid": np.asarray([x["observation_valid"] for x in relations], bool).reshape(-1, 3),
        "relation_observation_translation": np.asarray([x["translations"] for x in relations], np.int8).reshape(-1, 3),
        "relation_observation_count": [x["observation_count"] for x in relations],
        "relation_runtime_valid": [x["runtime_valid"] for x in relations],
        "relation_is_fallback": [x["fallback"] for x in relations],
    }


def build(cohort_name, *, workers=48, chunk_size=256, max_samples=None, output_root=None):
    cache_root = ROOT / "data/processed/mips_trimer_scage"
    cohort = _cohort(cache_root, cohort_name)
    keys = [bytes(row) for row in np.asarray(cohort["keys"], dtype=np.uint8)]
    if max_samples is not None:
        keys = keys[:int(max_samples)]
    specs = _specs(ROOT)
    destination = Path(output_root or cache_root / "periodic_line_glt_central_v1" / cohort_name).resolve()
    if destination.exists():
        if (destination / ".done").is_file():
            return destination
        raise RuntimeError(f"incomplete central line sidecar exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{cohort_name}.central-glt-", dir=destination.parent))
    handles = {name: (staging / f"{name}.raw").open("wb") for name in ARRAY_NAMES}
    token_total = relation_total = 0
    try:
        _append(handles["sample_token_offsets"], [0], ARRAY_DTYPES["sample_token_offsets"])
        _append(handles["sample_relation_offsets"], [0], ARRAY_DTYPES["sample_relation_offsets"])
        chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(str(specs["topology"]["root"]), str(specs["trimer"]["root"]))) as executor:
            for records in executor.map(_work, chunks):
                for record in records:
                    for name, values in _record_arrays(record).items():
                        _append(handles[name], values, ARRAY_DTYPES[name])
                    token_total += len(record["tokens"])
                    relation_total += len(record["relations"])
                    _append(handles["sample_token_offsets"], [token_total], ARRAY_DTYPES["sample_token_offsets"])
                    _append(handles["sample_relation_offsets"], [relation_total], ARRAY_DTYPES["sample_relation_offsets"])
        for handle in handles.values():
            handle.close()
        shapes = {}
        token_matrices = {"token_observation_distances", "token_observation_valid", "token_observation_translation", "token_observation_q_a", "token_observation_q_b"}
        relation_matrices = {"relation_observation_angles", "relation_observation_valid", "relation_observation_translation"}
        for name in ARRAY_NAMES:
            if name == "sample_keys": shape = (len(keys), 32)
            elif name in {"sample_token_offsets", "sample_relation_offsets"}: shape = (len(keys) + 1,)
            elif name.startswith("graph_"): shape = (len(keys),)
            elif name in token_matrices: shape = (token_total, 3)
            elif name in relation_matrices: shape = (relation_total, 3)
            elif name.startswith("token_"): shape = (token_total,)
            else: shape = (relation_total,)
            shapes[name] = shape
            _materialize(staging / f"{name}.raw", staging / f"{name}.npy", ARRAY_DTYPES[name], shape)
        metadata = {
            "schema": SIDECAR_SCHEMA, "builder_version": BUILDER_VERSION,
            "cohort": cohort_name, "record_count": len(keys),
            "selection": "full_cohort" if max_samples is None else f"prefix_{len(keys)}",
            "policy": {"distance": "central_shift0_dual_shift1", "angle": "canonical_span", "observation_identity": "explicit_slots"},
            "arrays": {name: {"dtype": ARRAY_DTYPES[name].str, "shape": list(shapes[name])} for name in ARRAY_NAMES},
        }
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / ".done").write_text("complete\n", encoding="utf-8")
        for raw in staging.glob("*.raw"): raw.unlink()
        os.replace(staging, destination)
        return destination
    except Exception:
        for handle in handles.values():
            if not handle.closed: handle.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("PI1M_v2", "downstream_union"), required=True)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output-root")
    args = parser.parse_args()
    print(build(args.cohort, workers=args.workers, chunk_size=args.chunk_size,
                max_samples=args.max_samples, output_root=args.output_root))


if __name__ == "__main__":
    main()
