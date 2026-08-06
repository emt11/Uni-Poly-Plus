#!/usr/bin/env python
"""Create the read-only topology-cost array used by the balanced sampler.

The LMDB values are PyTorch/PyG objects.  Repeated ``torch.load`` calls retain
allocator arenas in a long-lived process, so a million-row scan can grow RSS
even though only two uint32 values are kept per row.  This producer processes
bounded chunks in short-lived workers and keeps the parent limited to an output
mmap plus a bool progress map.  It never changes an LMDB record or its hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.lmdb_cache import LmdbLayerStore


def _read_cost_chunk(cohort_root, topology_root, start, end, part_path):
    """Read one bounded chunk and write it before the worker exits."""
    keys = np.load(Path(cohort_root) / "sample_keys.npy", mmap_mode="r")
    values = np.empty((int(end) - int(start), 2), dtype=np.uint32)
    store = LmdbLayerStore(str(topology_root))
    try:
        for local, row in enumerate(range(int(start), int(end))):
            item = store[bytes(np.asarray(keys[row], dtype=np.uint8).tobytes())]
            values[local, 0] = int(item.mips_x.size(0))
            values[local, 1] = int(item.lga_edge_index.size(1))
    finally:
        store.close()
    # Passing an open handle prevents numpy from silently appending another
    # ``.npy`` suffix to the process-specific part path.
    with open(part_path, "wb") as handle:
        np.save(handle, values)


def _open_or_create_progress(path, size):
    if path.is_file():
        try:
            progress = np.load(path, mmap_mode="r+")
            if progress.shape == (size,) and progress.dtype == np.bool_:
                return progress
            del progress
        except (OSError, ValueError):
            pass
    temporary = path.with_suffix(path.suffix + ".tmp")
    progress = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.bool_, shape=(size,)
    )
    progress[:] = False
    progress.flush()
    del progress
    os.replace(temporary, path)
    return np.load(path, mmap_mode="r+")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-root", required=True)
    parser.add_argument("--topology-root", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--chunk-size", type=int, default=16384,
        help="Rows per short-lived worker; smaller values bound allocator RSS.",
    )
    args = parser.parse_args()
    if int(args.chunk_size) <= 0:
        raise ValueError("--chunk-size must be positive")

    cohort_root = Path(args.cohort_root).resolve()
    manifest = json.loads((cohort_root / "manifest.json").read_text())
    keys = np.load(cohort_root / "sample_keys.npy", mmap_mode="r")
    if len(keys) != int(manifest["unique_count"]):
        raise RuntimeError("cohort key count does not match manifest")
    output = (
        Path(args.output).resolve()
        if args.output else cohort_root / "topology_cost.npy"
    )
    metadata_path = output.with_name("topology_cost_metadata.json")
    artifact_hash = Path(args.topology_root, ".done").read_text().strip()
    if output.is_file() and metadata_path.is_file():
        try:
            observed = json.loads(metadata_path.read_text())
            if (
                observed.get("schema") == "mips-trimer-scage-topology-cost-v1"
                and observed.get("cohort_hash") == manifest["cohort_hash"]
                and observed.get("ordered_sample_key_hash")
                == manifest["ordered_sample_key_hash"]
                and observed.get("topology_artifact_hash") == artifact_hash
                and observed.get("shape") == [len(keys), 2]
                and observed.get("dtype") == "uint32"
                and observed.get("sha256")
                == hashlib.sha256(output.read_bytes()).hexdigest()
            ):
                print(json.dumps(observed, sort_keys=True))
                return
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    temporary = output.with_suffix(output.suffix + ".building")
    progress_path = output.with_name(output.stem + ".progress.npy")
    size = len(keys)

    # A prior interrupted build is reusable only if its NumPy header matches.
    if temporary.is_file():
        try:
            existing = np.load(temporary, mmap_mode="r+")
            valid = existing.shape == (size, 2) and existing.dtype == np.uint32
            del existing
            if not valid:
                temporary.replace(temporary.with_name(temporary.name + ".invalid"))
        except (OSError, ValueError):
            temporary.replace(temporary.with_name(temporary.name + ".invalid"))
    values = np.lib.format.open_memmap(
        temporary,
        mode="r+" if temporary.is_file() else "w+",
        dtype=np.uint32,
        shape=(size, 2),
    )
    progress = _open_or_create_progress(progress_path, size)
    context = mp.get_context("spawn")
    topology_root = str(Path(args.topology_root).resolve())
    completed = int(np.count_nonzero(progress))

    for start in range(0, size, int(args.chunk_size)):
        end = min(size, start + int(args.chunk_size))
        if bool(progress[start:end].all()):
            continue
        part_path = Path(f"{temporary}.part.{start}")
        process = context.Process(
            target=_read_cost_chunk,
            args=(str(cohort_root), topology_root, start, end, str(part_path)),
        )
        process.start()
        process.join()
        if process.exitcode != 0 or not part_path.is_file():
            raise RuntimeError(
                f"topology cost worker failed for rows {start}:{end} "
                f"(exitcode={process.exitcode})"
            )
        part = np.load(part_path, mmap_mode="r")
        if part.shape != (end - start, 2) or part.dtype != np.uint32:
            raise RuntimeError(f"invalid topology cost chunk {part_path}")
        values[start:end] = part
        values.flush()
        del part
        # Keep an explicit completion marker so a killed parent never treats
        # an uncommitted part as complete.  The tiny .done markers are
        # harmless and make interrupted builds diagnosable.
        part_path.replace(part_path.with_name(part_path.name + ".done"))
        progress[start:end] = True
        progress.flush()
        completed += end - start
        print(f"[topology_cost] completed={completed}/{size}", flush=True)

    del progress
    values.flush()
    del values
    os.replace(temporary, output)
    if progress_path.exists():
        progress_path.replace(progress_path.with_name(progress_path.name + ".done"))
    # Part files are only a crash-recovery aid; once the complete array is
    # atomically published they have no read-side meaning.  Remove only files
    # under this explicitly selected output prefix.
    for part_file in output.parent.glob(f"{temporary.name}.part.*"):
        try:
            part_file.unlink()
        except OSError:
            pass
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata = {
        "schema": "mips-trimer-scage-topology-cost-v1",
        "cohort_hash": manifest["cohort_hash"],
        "ordered_sample_key_hash": manifest["ordered_sample_key_hash"],
        "topology_artifact_hash": artifact_hash,
        "shape": [size, 2],
        "dtype": "uint32",
        "sha256": digest,
        "chunk_size": int(args.chunk_size),
    }
    metadata_temporary = metadata_path.with_suffix(".tmp")
    metadata_temporary.write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n"
    )
    os.replace(metadata_temporary, metadata_path)
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
