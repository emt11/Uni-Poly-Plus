#!/usr/bin/env python3
"""Build the MTS-GLT-v2 periodic line-geometry sidecar from frozen caches."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_mips_trimer_cache import _specs  # noqa: E402
from src.dataset.lmdb_cache import LmdbLayerStore, load_cohort  # noqa: E402
from src.dataset.periodic_line_glt import (  # noqa: E402
    ARRAY_DTYPES,
    ARRAY_NAMES,
    BUILDER_VERSION,
    SIDECAR_SCHEMA,
    build_periodic_line_sample,
)

_TOPOLOGY = None
_TRIMER = None


def _cohort(cache_root, name):
    pointer = cache_root / "cohorts" / name / "current.json"
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    return load_cohort(
        pointer.parent / payload["cohort_hash"],
        load_text=False,
        verify_integrity=False,
    )


def _init_worker(topology_root, trimer_root):
    global _TOPOLOGY, _TRIMER
    torch.set_num_threads(1)
    _TOPOLOGY = LmdbLayerStore(topology_root, require_done=True)
    _TRIMER = LmdbLayerStore(trimer_root, require_done=True)


def _work(keys):
    return [
        build_periodic_line_sample(bytes(key), _TOPOLOGY[key], _TRIMER[key])
        for key in keys
    ]


def _append(handle, values, dtype):
    array = np.asarray(values, dtype=dtype)
    if array.size:
        handle.write(array.tobytes(order="C"))


def _materialize(raw, output, dtype, shape):
    temporary = output.with_suffix(".npy.tmp")
    target = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=dtype, shape=shape
    )
    total = int(np.prod(shape, dtype=np.int64))
    offset = 0
    with raw.open("rb") as handle:
        while offset < total:
            values = np.fromfile(
                handle, dtype=dtype, count=min(total - offset, 1 << 20)
            )
            if not values.size:
                raise RuntimeError(f"truncated staging payload: {raw}")
            target.reshape(-1)[offset:offset + values.size] = values
            offset += values.size
    target.flush()
    del target
    os.replace(temporary, output)


def _record_arrays(record):
    tokens, relations = record["tokens"], record["relations"]
    return {
        "sample_keys": np.frombuffer(record["sample_key"], dtype=np.uint8),
        "graph_geometry_valid": [record["geometry_valid"]],
        "token_atom_a": [item["atom_a"] for item in tokens],
        "token_atom_b": [item["atom_b"] for item in tokens],
        "token_shift": [item["shift"] for item in tokens],
        "token_endpoint_z_a": [item["z_a"] for item in tokens],
        "token_endpoint_z_b": [item["z_b"] for item in tokens],
        "token_bond_type": [item["bond_type"] for item in tokens],
        "token_label": [item["label"] for item in tokens],
        "token_observation_distances": np.asarray(
            [item["distances"] for item in tokens], dtype=np.float32
        ).reshape(-1, 3),
        "token_observation_count": [item["observation_count"] for item in tokens],
        "token_valid": [item["valid"] for item in tokens],
        "relation_source": [item["source"] for item in relations],
        "relation_target": [item["target"] for item in relations],
        "relation_center_atom": [item["center_atom"] for item in relations],
        "relation_multiplicity": [item["multiplicity"] for item in relations],
        "relation_observation_angles": np.asarray(
            [item["angles"] for item in relations], dtype=np.float32
        ).reshape(-1, 3),
        "relation_observation_count": [item["observation_count"] for item in relations],
        "relation_valid": [item["valid"] for item in relations],
        "relation_is_fallback": [item["fallback"] for item in relations],
    }


def build(cohort_name, *, workers, chunk_size, max_samples=None, output_root=None):
    cache_root = PROJECT_ROOT / "data/processed/mips_trimer_scage"
    cohort = _cohort(cache_root, cohort_name)
    keys = [bytes(row) for row in np.asarray(cohort["keys"], dtype=np.uint8)]
    if max_samples is not None:
        keys = keys[:int(max_samples)]
    specs = _specs(PROJECT_ROOT)
    topology_root = Path(specs["topology"]["root"])
    trimer_root = Path(specs["trimer"]["root"])
    root = Path(output_root) if output_root else (
        cache_root / "periodic_line_glt_v1" / cohort_name
    )
    root = root.resolve()
    if root.exists():
        if (root / ".done").is_file():
            return root
        raise RuntimeError(f"incomplete periodic line GLT sidecar exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{cohort_name}.glt-", dir=root.parent))
    handles = {name: (staging / f"{name}.raw").open("wb") for name in ARRAY_NAMES}
    token_total = relation_total = 0
    try:
        _append(handles["sample_token_offsets"], [0], ARRAY_DTYPES["sample_token_offsets"])
        _append(handles["sample_relation_offsets"], [0], ARRAY_DTYPES["sample_relation_offsets"])
        chunks = [keys[index:index + chunk_size] for index in range(0, len(keys), chunk_size)]
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(str(topology_root), str(trimer_root)),
        ) as executor:
            for records in executor.map(_work, chunks):
                for record in records:
                    arrays = _record_arrays(record)
                    for name, values in arrays.items():
                        _append(handles[name], values, ARRAY_DTYPES[name])
                    token_total += len(record["tokens"])
                    relation_total += len(record["relations"])
                    _append(handles["sample_token_offsets"], [token_total], ARRAY_DTYPES["sample_token_offsets"])
                    _append(handles["sample_relation_offsets"], [relation_total], ARRAY_DTYPES["sample_relation_offsets"])
        for handle in handles.values():
            handle.close()
        shapes = {}
        token_matrix = {"token_observation_distances"}
        relation_matrix = {"relation_observation_angles"}
        for name in ARRAY_NAMES:
            if name == "sample_keys":
                shape = (len(keys), 32)
            elif name in {"sample_token_offsets", "sample_relation_offsets"}:
                shape = (len(keys) + 1,)
            elif name == "graph_geometry_valid":
                shape = (len(keys),)
            elif name in token_matrix:
                shape = (token_total, 3)
            elif name in relation_matrix:
                shape = (relation_total, 3)
            elif name.startswith("token_"):
                shape = (token_total,)
            else:
                shape = (relation_total,)
            shapes[name] = shape
            _materialize(
                staging / f"{name}.raw", staging / f"{name}.npy",
                ARRAY_DTYPES[name], shape,
            )
        metadata = {
            "schema": SIDECAR_SCHEMA,
            "builder_version": BUILDER_VERSION,
            "cohort": cohort_name,
            "record_count": len(keys),
            "selection": "full_cohort" if max_samples is None else f"prefix_{len(keys)}",
            "geometry": {
                "line_scope": "true_chemical_bonds",
                "relation_scope": "shared_center_1hop",
                "distance_aggregation": "encode_then_mean",
                "angle_aggregation": "encode_then_mean",
                "bond_type_model_input": False,
            },
            "arrays": {
                name: {"dtype": ARRAY_DTYPES[name].str, "shape": list(shapes[name])}
                for name in ARRAY_NAMES
            },
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging / ".done").write_text("complete\n", encoding="utf-8")
        for raw in staging.glob("*.raw"):
            raw.unlink()
        os.replace(staging, root)
        return root
    except Exception:
        for handle in handles.values():
            if not handle.closed:
                handle.close()
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
    if args.workers < 1 or args.chunk_size < 1:
        parser.error("workers and chunk-size must be positive")
    print(build(
        args.cohort, workers=args.workers, chunk_size=args.chunk_size,
        max_samples=args.max_samples, output_root=args.output_root,
    ))


if __name__ == "__main__":
    main()
