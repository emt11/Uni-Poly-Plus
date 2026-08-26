#!/usr/bin/env python3
"""Build filtered periodic spatial-contact sidecars from frozen MTS caches."""

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
from src.dataset.periodic_spatial_contact import (  # noqa: E402
    ARRAY_DTYPES, ARRAY_NAMES, BUILDER_VERSION, SIDECAR_SCHEMA,
    build_spatial_contact_sample,
)

_TOPOLOGY = None
_TRIMER = None


def _cohort(cache_root, name):
    pointer = cache_root / "cohorts" / name / "current.json"
    payload = json.loads(pointer.read_text())
    return load_cohort(pointer.parent / payload["cohort_hash"], load_text=False, verify_integrity=False)


def _init_worker(topology_root, trimer_root):
    global _TOPOLOGY, _TRIMER
    torch.set_num_threads(1)
    _TOPOLOGY = LmdbLayerStore(topology_root, require_done=True)
    _TRIMER = LmdbLayerStore(trimer_root, require_done=True)


def _work(keys):
    return [build_spatial_contact_sample(bytes(key), _TOPOLOGY[key], _TRIMER[key]) for key in keys]


def _append(handle, values, dtype):
    array = np.asarray(values, dtype=dtype)
    if array.size:
        handle.write(array.tobytes(order="C"))


def _materialize(raw, output, dtype, shape):
    temporary = output.with_suffix(".npy.tmp")
    target = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=shape)
    total, offset = int(np.prod(shape, dtype=np.int64)), 0
    with raw.open("rb") as handle:
        while offset < total:
            values = np.fromfile(handle, dtype=dtype, count=min(total - offset, 1 << 20))
            if not values.size:
                raise RuntimeError(f"truncated staging payload: {raw}")
            target.reshape(-1)[offset:offset + values.size] = values
            offset += values.size
    target.flush()
    del target
    os.replace(temporary, output)


def _record_arrays(record):
    pairs = record["pairs"]
    return {
        "sample_keys": np.frombuffer(record["sample_key"], dtype=np.uint8),
        "graph_valid": [record["graph_valid"]],
        "sample_atom_count": [record["atom_count"]],
        "pair_atom_a": [row["atom_a"] for row in pairs],
        "pair_atom_b": [row["atom_b"] for row in pairs],
        "pair_shift": [row["shift"] for row in pairs],
        "pair_observation_distances": np.asarray([row["distances"] for row in pairs], dtype=np.float32).reshape(-1, 3),
        "pair_observation_valid": np.asarray([row["observation_valid"] for row in pairs], dtype=bool).reshape(-1, 3),
        "pair_observation_count": [row["observation_count"] for row in pairs],
        "pair_shell_id": [row["shell_id"] for row in pairs],
        "pair_periodic_self": [row["periodic_self"] for row in pairs],
        "pair_valid": [row["valid"] for row in pairs],
        "pair_spd": [row["spd"] for row in pairs],
        "pair_raw_mean_distance": [row["raw_mean_distance"] for row in pairs],
        "pair_raw_variance": [row["raw_variance"] for row in pairs],
    }


def build(cohort_name, *, workers, chunk_size, max_samples=None, output_root=None):
    cache_root = ROOT / "data/processed/mips_trimer_scage"
    cohort = _cohort(cache_root, cohort_name)
    keys = [bytes(row) for row in np.asarray(cohort["keys"], dtype=np.uint8)]
    if max_samples is not None:
        keys = keys[:int(max_samples)]
    specs = _specs(ROOT)
    root = Path(output_root) if output_root else cache_root / "spatial_contact_v1" / cohort_name
    root = root.resolve()
    if root.exists():
        if (root / ".done").is_file():
            return root
        raise RuntimeError(f"incomplete periodic spatial sidecar exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{cohort_name}.spatial-", dir=root.parent))
    handles = {name: (staging / f"{name}.raw").open("wb") for name in ARRAY_NAMES}
    pair_total = invalid_graphs = 0
    shell_counts = [0, 0]
    try:
        _append(handles["sample_pair_offsets"], [0], ARRAY_DTYPES["sample_pair_offsets"])
        chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker,
            initargs=(str(specs["topology"]["root"]), str(specs["trimer"]["root"])),
        ) as executor:
            for records in executor.map(_work, chunks):
                for record in records:
                    arrays = _record_arrays(record)
                    for name, values in arrays.items():
                        _append(handles[name], values, ARRAY_DTYPES[name])
                    pair_total += len(record["pairs"])
                    invalid_graphs += int(not record["graph_valid"])
                    for row in record["pairs"]:
                        shell_counts[int(row["shell_id"])] += 1
                    _append(handles["sample_pair_offsets"], [pair_total], ARRAY_DTYPES["sample_pair_offsets"])
        for handle in handles.values():
            handle.close()
        shapes = {}
        for name in ARRAY_NAMES:
            if name == "sample_keys": shape = (len(keys), 32)
            elif name == "sample_pair_offsets": shape = (len(keys) + 1,)
            elif name in {"graph_valid", "sample_atom_count"}: shape = (len(keys),)
            elif name in {"pair_observation_distances", "pair_observation_valid"}: shape = (pair_total, 3)
            else: shape = (pair_total,)
            shapes[name] = shape
            _materialize(staging / f"{name}.raw", staging / f"{name}.npy", ARRAY_DTYPES[name], shape)
        metadata = {
            "schema": SIDECAR_SCHEMA,
            "builder_version": BUILDER_VERSION,
            "cohort": cohort_name,
            "record_count": len(keys),
            "pair_count": pair_total,
            "invalid_graph_count": invalid_graphs,
            "shell_pair_counts": {"s4": shell_counts[0], "s45": shell_counts[1]},
            "selection": "full_cohort" if max_samples is None else f"prefix_{len(keys)}",
            "geometry": {
                "spd_min": 4, "core_cutoff_A": 4.0, "outer_cutoff_A": 5.0,
                "shell_assignment": "raw_observation_mean",
                "canonical_equivalence": "(i,j,s)==(j,i,-s)",
            },
            "arrays": {name: {"dtype": ARRAY_DTYPES[name].str, "shape": list(shapes[name])} for name in ARRAY_NAMES},
        }
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        (staging / ".done").write_text("complete\n")
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
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output-root")
    args = parser.parse_args()
    print(build(args.cohort, workers=args.workers, chunk_size=args.chunk_size, max_samples=args.max_samples, output_root=args.output_root))


if __name__ == "__main__":
    main()
