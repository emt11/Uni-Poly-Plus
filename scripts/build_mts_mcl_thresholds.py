#!/usr/bin/env python3
"""Build or reuse the content-bound MCL threshold sidecars for MTS."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_mips_trimer_cache import _specs  # noqa: E402
from src.dataset.lmdb_cache import (  # noqa: E402
    LmdbLayerStore,
    build_or_load_cohort,
    compute_mcl_thresholds,
    finalize_mcl_threshold_mmap,
    mcl_thresholds_cached,
)


_WORKER_TRIMER = None


def _init_worker(trimer_root):
    global _WORKER_TRIMER
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    _WORKER_TRIMER = LmdbLayerStore(trimer_root, require_done=True)


def _chunk_worker(chunk, index):
    values = np.empty((len(chunk), 2), dtype=np.float32)
    for row, key in enumerate(chunk):
        values[row] = compute_mcl_thresholds(_WORKER_TRIMER[key])
    return index, values


def _cohort_names(value):
    return [item for item in str(value).replace(",", " ").split() if item]


def _load_cohorts(cache_root, names):
    cohorts = []
    for name in names:
        filename = "smi_all.csv" if name == "downstream_union" else f"{name}.csv"
        cohorts.append(
            build_or_load_cohort(
                cache_root,
                name,
                PROJECT_ROOT / "data" / "raw" / filename,
                load_text=False,
                verify_integrity=True,
            )
        )
    return cohorts


def _done_hash(trimer_root):
    return (Path(trimer_root) / ".done").read_text(encoding="utf-8").strip()


def _build_one(cohort, trimer_root, workers, chunk_size):
    shape = (len(cohort["keys"]), 2)
    cached = mcl_thresholds_cached(cohort, trimer_root, shape)
    if cached is not None:
        path, metadata = cached
        return {
            "cohort": cohort["manifest"]["dataset_name"],
            "cohort_hash": cohort["manifest"]["cohort_hash"],
            "path": path,
            "reused": True,
            "metadata": metadata,
        }

    cohort_root = Path(cohort["root"])
    cohort_root.mkdir(parents=True, exist_ok=True)
    values_path = cohort_root / "mcl_thresholds.npy"
    temporary = values_path.with_suffix(values_path.suffix + ".tmp")
    mmap = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
    try:
        keys = cohort["keys"]
        chunks = [keys[i : i + chunk_size] for i in range(0, len(keys), chunk_size)]
        if workers > 1 and chunks:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=_init_worker,
                initargs=(str(trimer_root),),
            ) as executor:
                futures = {
                    executor.submit(_chunk_worker, chunk, index): index
                    for index, chunk in enumerate(chunks)
                }
                for future in as_completed(futures):
                    index, values = future.result()
                    start = index * chunk_size
                    mmap[start : start + len(values)] = values
        else:
            for index, key in enumerate(keys):
                mmap[index] = compute_mcl_thresholds(trimer_store[key])
        mmap.flush()
    finally:
        del mmap
    path, metadata = finalize_mcl_threshold_mmap(cohort, trimer_root, temporary)
    return {
        "cohort": cohort["manifest"]["dataset_name"],
        "cohort_hash": cohort["manifest"]["cohort_hash"],
        "path": path,
        "reused": False,
        "metadata": metadata,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohorts", default="PI1M_v2 downstream_union")
    parser.add_argument("--cache-root", default="data/processed/mips_trimer_scage")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--batch-chunk", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    del args.resume  # Sidecar reuse is always content-bound and safe.
    cache_root = (PROJECT_ROOT / args.cache_root).resolve()
    specs = _specs(PROJECT_ROOT)
    trimer_root = Path(specs["trimer"]["root"]).resolve()
    if not (trimer_root / ".done").is_file() or not (trimer_root / ".frozen").is_file():
        raise SystemExit(f"Trimer cache is not frozen: {trimer_root}")
    cohorts = _load_cohorts(cache_root, _cohort_names(args.cohorts))
    workers = max(1, int(args.workers))
    chunk_size = max(1, int(args.batch_chunk))
    global trimer_store
    trimer_store = LmdbLayerStore(trimer_root, expected_meta=specs["trimer"]["meta"])
    try:
        outputs = [_build_one(cohort, trimer_root, workers, chunk_size) for cohort in cohorts]
    finally:
        trimer_store.close()
    print(json.dumps({"command": "build-mcl-thresholds", "outputs": outputs}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
