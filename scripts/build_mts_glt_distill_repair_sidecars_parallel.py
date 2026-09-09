#!/usr/bin/env python3
"""Parallel shard builder and ordered merger for revision-2 line sidecars."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_mts_glt_distill_sidecars import TOKEN_DTYPES, RELATION_DTYPES  # noqa: E402
from src.dataset.periodic_line_distill_v2 import SCHEMA_PREFIX  # noqa: E402


def merge(parts_root: Path, output_root: Path, processes: int):
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    for version in ("n_plus_2", "n_plus_1"):
        sources = [parts_root / f"part_{i:02d}" / version for i in range(processes)]
        counts = [len(np.load(source / "geometry_valid.npy", mmap_mode="r")) for source in sources]
        token_counts = [int(np.load(source / "token_offsets.npy", mmap_mode="r")[-1]) for source in sources]
        relation_counts = [int(np.load(source / "relation_offsets.npy", mmap_mode="r")[-1]) for source in sources]
        count, token_total, relation_total = sum(counts), sum(token_counts), sum(relation_counts)
        target = output_root / version
        target.mkdir()
        keys = np.lib.format.open_memmap(target / "sample_keys.npy", mode="w+", dtype=np.uint8, shape=(count, 32))
        valid = np.lib.format.open_memmap(target / "geometry_valid.npy", mode="w+", dtype=np.bool_, shape=(count,))
        token_offsets = np.lib.format.open_memmap(target / "token_offsets.npy", mode="w+", dtype=np.int64, shape=(count + 1,))
        relation_offsets = np.lib.format.open_memmap(target / "relation_offsets.npy", mode="w+", dtype=np.int64, shape=(count + 1,))
        arrays = {
            name: np.lib.format.open_memmap(
                target / f"{name}.npy", mode="w+", dtype=dtype,
                shape=((token_total if name.startswith("token_") else relation_total),),
            ) for name, dtype in {**TOKEN_DTYPES, **RELATION_DTYPES}.items()
        }
        row_at = token_at = relation_at = 0
        token_offsets[0] = relation_offsets[0] = 0
        for source, rows, token_count, relation_count in zip(sources, counts, token_counts, relation_counts):
            keys[row_at:row_at + rows] = np.load(source / "sample_keys.npy", mmap_mode="r")
            valid[row_at:row_at + rows] = np.load(source / "geometry_valid.npy", mmap_mode="r")
            local_t = np.load(source / "token_offsets.npy", mmap_mode="r")
            local_r = np.load(source / "relation_offsets.npy", mmap_mode="r")
            token_offsets[row_at + 1:row_at + rows + 1] = local_t[1:] + token_at
            relation_offsets[row_at + 1:row_at + rows + 1] = local_r[1:] + relation_at
            for name in TOKEN_DTYPES:
                arrays[name][token_at:token_at + token_count] = np.load(source / f"{name}.npy", mmap_mode="r")
            for name in RELATION_DTYPES:
                arrays[name][relation_at:relation_at + relation_count] = np.load(source / f"{name}.npy", mmap_mode="r")
            row_at += rows; token_at += token_count; relation_at += relation_count
        for value in (keys, valid, token_offsets, relation_offsets, *arrays.values()):
            value.flush()
        metadata = {
            "schema": SCHEMA_PREFIX + version, "geometry_revision": 2,
            "geometry_semantics": version, "sample_count": count,
            "valid_count": int(valid.sum()), "token_count": token_total,
            "relation_count": relation_total,
            "source": "frozen_trimer_coordinates_direct",
            "source_sidecars": ["periodic_line_glt_image_v1/PI1M_v2 (chemistry only)"],
            "relation_semantics": "distinct_physical_bonds_incident_on_center_ru",
            "build_processes": processes,
        }
        (target / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        (target / ".done").write_text("complete\n")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--output-root", default="data/processed/mips_trimer_scage/periodic_line_glt_distill_v2")
    parser.add_argument("--sample-count", type=int, default=0)
    args = parser.parse_args(argv)
    processes = int(args.processes)
    output = Path(args.output_root)
    parts = output.with_name(output.name + ".parts")
    if output.exists() or parts.exists():
        raise FileExistsError(f"output/parts conflict: {output}, {parts}")
    parts.mkdir(parents=True)
    sample_count = int(args.sample_count) or int(np.load(
        ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_image_v1/PI1M_v2/sample_keys.npy",
        mmap_mode="r",
    ).shape[0])
    width = math.ceil(sample_count / processes)
    children = []
    handles = []
    try:
        for index in range(processes):
            start, stop = index * width, min(sample_count, (index + 1) * width)
            if start >= stop:
                continue
            log = parts / f"part_{index:02d}.log"
            handle = log.open("w")
            handles.append(handle)
            command = [
                sys.executable, "scripts/build_mts_glt_distill_repair_sidecars.py",
                "--workers", "1", "--start", str(start), "--stop", str(stop),
                "--output-root", str(parts / f"part_{index:02d}"),
            ]
            children.append((index, subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)))
        failures = []
        for index, child in children:
            status = child.wait()
            if status:
                failures.append((index, status))
        for handle in handles:
            handle.close()
        if failures:
            raise RuntimeError(f"sidecar shard builders failed: {failures}")
        merge(parts, output, processes)
        print(f"merged {processes} parts into {output}", flush=True)
    finally:
        for _index, child in children:
            if child.poll() is None:
                child.terminate()
        for handle in handles:
            if not handle.closed:
                handle.close()


if __name__ == "__main__":
    main()
