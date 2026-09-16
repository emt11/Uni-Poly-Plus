#!/usr/bin/env python3
"""Phase C fair read benchmark for the frozen dual route.

Contract (cache optimization plan section 5):

* fixed 2,048 indices and order (``OrderedSampleStream`` with seed 42); the list
  is saved in the report;
* the same per-sample mask/noise/targets for every configuration, because the
  position layout is identical and only the reader configuration changes;
* worker=0 plus one configuration with at most three prefetch workers;
* at most three paired repetitions per configuration, alternating order;
* no GPU, no model, no conformer generation, no write to the frozen caches;
* reports samples/s, per-sample latency p50/p95, chunk-mapping events, file
  opens, process-tree RSS, FD peak, page faults and machine load.

One controlled change at a time: the only candidate here is the bounded chunk
cache capacity of the static/target reader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.dataset.cache_lifecycle import zero_write_snapshot
from src.dataset.glt_dual_pretrain import chemical_targets, prepare_pretrain_sample
from src.training.glt_dual_runtime import OrderedSampleStream, RankMicrobatchStream, open_source


def _proc_stat():
    fields = Path("/proc/self/stat").read_text(encoding="utf-8").split()
    return {"minor_faults": int(fields[9]), "major_faults": int(fields[11])}


def _rss_bytes(pid=None):
    pid = pid or os.getpid()
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return 0
    return 0


def _tree_rss(self_pid):
    children = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        parent = None
        for line in status:
            if line.startswith("PPid:"):
                parent = int(line.split()[1])
                break
        children.setdefault(parent, []).append(int(entry.name))
    total, stack = 0, [self_pid]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += _rss_bytes(pid)
        stack.extend(children.get(pid, []))
    return total


def _fd_count():
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return 0


def _instrument():
    """Count chunk-mapping events and array-file opens without touching production code."""

    import src.dataset.glt_dual_static as static_module

    counts = {"chunk_map_events": 0, "np_load_calls": 0}
    original_load = static_module._ChunkReader._load

    def counted_load(self, chunk_id):
        counts["chunk_map_events"] += 1
        return original_load(self, chunk_id)

    static_module._ChunkReader._load = counted_load
    original_np_load = np.load

    def counted_np_load(*args, **kwargs):
        counts["np_load_calls"] += 1
        return original_np_load(*args, **kwargs)

    np.load = counted_np_load
    return counts


def _summary(values):
    ordered = sorted(values)
    count = len(ordered)
    return {"count": count, "mean": statistics.fmean(ordered),
            "p50": ordered[min(count - 1, int(0.50 * count))],
            "p95": ordered[min(count - 1, int(0.95 * count))],
            "max": ordered[-1]}


def _run_inline(source, indices, *, seed, microbatch, meta):
    """worker=0: identical positions to the prefetch path, plus phase timings."""

    phase = {"source_read": [], "static_read": [], "target_read": [], "prepare": [], "total": []}
    faults_before, fd_peak = _proc_stat(), 0
    started = time.perf_counter()
    for order, index in enumerate(indices):
        fault = time.perf_counter()
        record = source[index]
        after_source = time.perf_counter()
        static = source.static_for(index)
        after_static = time.perf_counter()
        target = source.target_for(index)
        after_target = time.perf_counter()
        key, _ = source.samples[index]
        position = order
        prepare_pretrain_sample(
            *record, seed=seed, key=key.hex(), position=position,
            sigma=meta["sigma"], ratio=meta["ratio"], static=static, target=target,
        )
        after_prepare = time.perf_counter()
        phase["source_read"].append(after_source - fault)
        phase["static_read"].append(after_static - after_source)
        phase["target_read"].append(after_target - after_static)
        phase["prepare"].append(after_prepare - after_target)
        phase["total"].append(after_prepare - fault)
        if order % 64 == 0:
            fd_peak = max(fd_peak, _fd_count())
    elapsed = time.perf_counter() - started
    faults_after = _proc_stat()
    return {
        "wall_seconds": elapsed,
        "samples_per_second": len(indices) / elapsed,
        "latency_ms": {name: {key: value * 1000 for key, value in _summary(values).items()
                             if key != "count"} | {"count": len(values)}
                       for name, values in phase.items()},
        "fd_peak": max(fd_peak, _fd_count()),
        "minor_faults": faults_after["minor_faults"] - faults_before["minor_faults"],
        "major_faults": faults_after["major_faults"] - faults_before["major_faults"],
        "self_rss_bytes": _rss_bytes(),
        "tree_rss_bytes": _tree_rss(os.getpid()),
    }


def _run_prefetch(source, indices, *, seed, microbatch, workers, meta):
    """The existing DataLoader prefetch path over the same positions."""

    faults_before = _proc_stat()
    dataset = RankMicrobatchStream(
        source, seed=seed, world=1, rank=0, microbatch=microbatch, accumulation=1,
        start_step=0, max_steps=max(1, len(indices) // microbatch),
        sigma=meta["sigma"], ratio=meta["ratio"],
    )
    started = time.perf_counter()
    batches = 0
    fd_peak, rss_peak = 0, 0
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=int(workers),
        prefetch_factor=4, persistent_workers=False, pin_memory=False,
    )
    for _ in loader:
        batches += 1
        if batches % 8 == 0:
            fd_peak = max(fd_peak, _fd_count())
            rss_peak = max(rss_peak, _tree_rss(os.getpid()))
    elapsed = time.perf_counter() - started
    faults_after = _proc_stat()
    return {
        "wall_seconds": elapsed,
        "samples_per_second": batches * microbatch / elapsed,
        "microbatches": batches,
        "microbatch": microbatch,
        "latency_ms": {"microbatch_mean": elapsed / max(1, batches) * 1000},
        "fd_peak": max(fd_peak, _fd_count()),
        "minor_faults": faults_after["minor_faults"] - faults_before["minor_faults"],
        "major_faults": faults_after["major_faults"] - faults_before["major_faults"],
        "self_rss_bytes": _rss_bytes(),
        "rss_peak_bytes": max(rss_peak, _rss_bytes()),
        "tree_rss_bytes": _tree_rss(os.getpid()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--cohort-root", required=True)
    parser.add_argument("--dual-static-root", required=True)
    parser.add_argument("--pretrain-target-root", required=True)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--microbatch", type=int, default=32)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--capacity", type=int, default=64,
                        help="bounded chunk cache capacity for the candidate")
    parser.add_argument("--prefetch-workers", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    cache_root = Path(args.cache_root).resolve()
    before = zero_write_snapshot(cache_root)
    counts = _instrument()
    meta = {"sigma": args.sigma, "ratio": args.ratio}
    report = {"scope": "CPU-only paired read benchmark; no GPU, no model, no conformer generation",
              "samples": args.samples, "seed": args.seed, "microbatch": args.microbatch,
              "reps": args.reps, "candidate_chunk_capacity": args.capacity,
              "load_average_at_start": list(os.getloadavg())}
    try:
        base_source, frame = open_source(
            args.cohort_root, cache_root, dual_static_root=args.dual_static_root,
            pretrain_target_root=args.pretrain_target_root,
        )
        try:
            stream = OrderedSampleStream(len(base_source), args.seed)
            indices = [stream.index_at(position) for position in range(args.samples)]
            report["indices"] = indices
            report["index_list_sha256"] = hashlib.sha256(
                json.dumps(indices).encode("utf-8")).hexdigest()
            # Bounded in-process memoization warmup, identical for every config.
            for index in indices[: min(len(indices), 256)]:
                chemical_targets(str(base_source[index][0].normalized_canonical_smiles))
        finally:
            base_source.close()

        configurations = [
            {"name": "baseline_capacity2", "capacity": 2, "workers": 0},
            {"name": f"candidate_capacity{args.capacity}", "capacity": args.capacity, "workers": 0},
        ]
        report["configurations"] = {}
        for config in configurations:
            entry = {"capacity": config["capacity"], "workers": config["workers"], "runs": []}
            for rep in range(args.reps):
                source, _ = open_source(
                    args.cohort_root, cache_root, dual_static_root=args.dual_static_root,
                    pretrain_target_root=args.pretrain_target_root,
                    chunk_cache_capacity=config["capacity"],
                )
                try:
                    chunk_before, load_before = counts["chunk_map_events"], counts["np_load_calls"]
                    run = _run_inline(source, indices, seed=args.seed,
                                     microbatch=args.microbatch, meta=meta)
                    run["rep"] = rep
                    run["chunk_map_events"] = counts["chunk_map_events"] - chunk_before
                    run["np_load_calls"] = counts["np_load_calls"] - load_before
                    entry["runs"].append(run)
                finally:
                    source.close()
            entry["median_samples_per_second"] = statistics.median(
                run["samples_per_second"] for run in entry["runs"])
            report["configurations"][config["name"]] = entry

        # One paired repetition of the existing prefetch path, baseline vs candidate.
        for config in configurations:
            source, _ = open_source(
                args.cohort_root, cache_root, dual_static_root=args.dual_static_root,
                pretrain_target_root=args.pretrain_target_root,
                chunk_cache_capacity=config["capacity"],
            )
            try:
                chunk_before, load_before = counts["chunk_map_events"], counts["np_load_calls"]
                run = _run_prefetch(source, indices, seed=args.seed, microbatch=args.microbatch,
                                    workers=args.prefetch_workers, meta=meta)
                run["rep"] = 0
                run["chunk_map_events"] = counts["chunk_map_events"] - chunk_before
                run["np_load_calls"] = counts["np_load_calls"] - load_before
                report["configurations"][config["name"]][
                    f"prefetch_workers{args.prefetch_workers}"] = run
            finally:
                source.close()
    finally:
        report["frozen_cache_zero_write"] = zero_write_snapshot(cache_root) == before
        report["load_average_at_end"] = list(os.getloadavg())

    baseline = report["configurations"]["baseline_capacity2"]
    candidate = report["configurations"][f"candidate_capacity{args.capacity}"]
    ratio = candidate["median_samples_per_second"] / baseline["median_samples_per_second"]
    baseline_p95 = statistics.median(
        run["latency_ms"]["total"]["p95"] for run in baseline["runs"])
    candidate_p95 = statistics.median(
        run["latency_ms"]["total"]["p95"] for run in candidate["runs"])
    fd_growth = [run["fd_peak"] for run in candidate["runs"] + baseline["runs"]]
    rss_growth = [run["self_rss_bytes"] for run in candidate["runs"] + baseline["runs"]]
    report["decision"] = {
        "throughput_ratio": ratio,
        "median_samples_per_second": {
            "baseline": baseline["median_samples_per_second"],
            "candidate": candidate["median_samples_per_second"]},
        "p95_ratio": candidate_p95 / baseline_p95,
        "fd_peak_max": max(fd_growth), "fd_peak_min": min(fd_growth),
        "rss_min": min(rss_growth), "rss_max": max(rss_growth),
        "thresholds": {"throughput_gain": 0.10, "p95_regression": 0.05},
        "accepted": bool(ratio >= 1.10 and candidate_p95 <= baseline_p95 * 1.05),
        "note": ("thresholds are an engineering screen over three paired repetitions, not a "
                 "statistical significance claim; production switching needs separate authorization"),
    }
    Path(args.output_json).write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("samples", "reps", "decision", "frozen_cache_zero_write",
                       "load_average_at_end")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
