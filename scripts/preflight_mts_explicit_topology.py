#!/usr/bin/env python
"""Correctness/throughput preflight for corrected explicit k-RU topology.

This tool never opens an LMDB writer and never builds Trimer geometry.  It is
the required gate before materialising the independent explicit topology
cache for the exact PI1M/downstream union.
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from rdkit import rdBase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.dataset import _compute_ru_base_layer, _compute_topology_layer
from src.dataset.lmdb_cache import build_or_load_cohort
from src.dataset.mips_trimer_contract import TOPOLOGY_CANONICAL, TOPOLOGY_EXPLICIT


def _relation_counter(data, *, explicit=False, include_shift=True):
    values = Counter()
    canonical_count = int(
        torch.as_tensor(data.canonical_ru_atom_index).max().item()
    ) + 1
    repeat_units = int(getattr(data, "mips_repeat_units", 1))
    for row in range(int(data.lga_spd.numel())):
        source = int(data.lga_edge_index[0, row])
        target = int(data.lga_edge_index[1, row])
        if explicit:
            source_copy = int(data.ru_copy_index[source])
            target_copy = int(data.ru_copy_index[target])
            shift = source_copy - target_copy
            if shift > repeat_units // 2:
                shift -= repeat_units
            elif shift < -(repeat_units // 2):
                shift += repeat_units
            source = int(data.canonical_ru_atom_index[source])
            target = int(data.canonical_ru_atom_index[target])
        else:
            shift = int(data.lga_source_image_shift[row])
        key = (
            (
                source, target, shift, int(data.lga_spd[row]),
                bool(data.lga_star_edge_mask[row]),
            )
            if include_shift else
            (
                source, target, int(data.lga_spd[row]),
                bool(data.lga_star_edge_mask[row]),
            )
        )
        values[key] += 1
    return values, canonical_count, repeat_units


def _serialized_size(data):
    buffer = io.BytesIO()
    torch.save(data, buffer)
    return int(buffer.tell())


def _worker(index_and_smiles):
    sample_index, smiles = index_and_smiles
    with rdBase.BlockLogs():
        try:
            ru_base = _compute_ru_base_layer(smiles)
            canonical = _compute_topology_layer(
                smiles, ru_base, max_hops=2,
                topology_representation=TOPOLOGY_CANONICAL,
            )
            explicit = _compute_topology_layer(
                smiles, ru_base, max_hops=2,
                topology_representation=TOPOLOGY_EXPLICIT,
            )
            # k=1/2 finite rings preserve relation multiplicity but copy IDs
            # cannot distinguish +1 from -1.  That shift is diagnostic only
            # and never embedded by the explicit model.  Check the full
            # shift-labelled multiset whenever k>=3; for k<3 compare the
            # source/target/SPD multiset while retaining every duplicate row.
            repeat_units = int(explicit.mips_repeat_units)
            include_shift = repeat_units >= 3
            canonical_rows, canonical_count, _ = _relation_counter(
                canonical, include_shift=include_shift
            )
            explicit_rows, explicit_count, repeat_units = _relation_counter(
                explicit, explicit=True, include_shift=include_shift
            )
            raw_parity = (
                canonical_count == explicit_count
                and explicit_rows == Counter({
                    key: repeat_units * count
                    for key, count in canonical_rows.items()
                })
            )
            canonical_identity = torch.as_tensor(
                explicit.canonical_ru_atom_index
            ).long()
            identity_ok = (
                canonical_identity.numel() == int(explicit.num_nodes)
                and canonical_identity.numel()
                == repeat_units * canonical_count
                and torch.equal(
                    canonical_identity,
                    torch.arange(canonical_count).repeat(repeat_units),
                )
            )
            star_self = bool((
                explicit.lga_star_edge_mask.bool()
                & (explicit.lga_edge_index[0] == explicit.lga_edge_index[1])
            ).any())
            available = bool(explicit.graph_available)
            # The canonical quotient may remain available when materialising
            # the minimum explicit chain would exceed 384 atoms.  That is the
            # specified explicit-only unavailable fallback, not a parity bug.
            parity = bool(raw_parity or not available)
            boundary_ok = (
                not available
                or (
                    bool(explicit.mips_condition_valid)
                    and int(explicit.mips_boundary_distance) > 5
                    and int(explicit.num_nodes) <= 384
                )
            )
            return {
                "sample_index": int(sample_index),
                "sample_smiles": str(smiles),
                "ok": bool(
                    parity and identity_ok
                    and not star_self and boundary_ok
                ),
                "available": available,
                "parity": bool(parity),
                "shift_parity_checked": bool(include_shift),
                "identity": bool(identity_ok),
                "star_self": star_self,
                "boundary": int(explicit.mips_boundary_distance),
                "repeat_units": repeat_units,
                "nodes": int(explicit.num_nodes),
                "relations": int(explicit.lga_spd.numel()),
                "bytes": _serialized_size(explicit),
                "error": "",
            }
        except Exception as exc:
            # Invalid chemistry is a supported graph-unavailable outcome in
            # the LMDB builder, but direct construction exceptions are still
            # reported so the preflight cannot hide an internal invariant.
            return {
                "sample_index": int(sample_index),
                "sample_smiles": str(smiles),
                "ok": False,
                "available": False,
                "parity": False,
                "identity": False,
                "star_self": False,
                "boundary": -1,
                "repeat_units": 0,
                "nodes": 0,
                "relations": 0,
                "bytes": 0,
                "error": f"{type(exc).__name__}:{exc}"[:400],
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PI1M_v2")
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.limit <= 0 or args.workers <= 0:
        raise ValueError("limit/workers must be positive")
    cache_root = PROJECT_ROOT / "data/processed/mips_trimer_scage"
    source = PROJECT_ROOT / f"data/raw/{args.dataset}.csv"
    cohort = build_or_load_cohort(
        cache_root, args.dataset, source,
        load_text=True, verify_integrity=False,
    )
    smiles = list(cohort["smiles"][: min(args.limit, len(cohort["smiles"]))])
    started = time.monotonic()
    ctx = mp.get_context("spawn")
    reports = []
    with ctx.Pool(
        processes=min(args.workers, len(smiles)),
        maxtasksperchild=512,
    ) as pool:
        for index, item in enumerate(pool.imap(
            _worker, enumerate(smiles), chunksize=max(1, args.chunk_size)
        ), start=1):
            reports.append(item)
            if index % 500 == 0 or index == len(smiles):
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"[explicit-preflight] {index}/{len(smiles)} "
                    f"records_per_s={index / elapsed:.2f}"
                )
    elapsed = max(time.monotonic() - started, 1e-9)
    failure_counts = Counter(item["error"] for item in reports if item["error"])
    total_bytes = sum(item["bytes"] for item in reports)
    estimated_full_count = 999224
    result = {
        "schema": "mts-explicit-kru-preflight-v1",
        "dataset": args.dataset,
        "sample_count": len(reports),
        "workers": int(args.workers),
        "chunk_size": int(args.chunk_size),
        "elapsed_seconds": elapsed,
        "records_per_second": len(reports) / elapsed,
        "available": sum(int(item["available"]) for item in reports),
        "unavailable": sum(int(not item["available"]) for item in reports),
        "parity_failures": sum(int(not item["parity"]) for item in reports),
        "shift_parity_checked": sum(
            int(item.get("shift_parity_checked", False)) for item in reports
        ),
        "identity_failures": sum(int(not item["identity"]) for item in reports),
        "star_self_failures": sum(int(item["star_self"]) for item in reports),
        "boundary_failures": sum(int(
            item["available"] and item["boundary"] <= 5
        ) for item in reports),
        "internal_failures": sum(int(not item["ok"]) for item in reports),
        "failure_counts": dict(failure_counts.most_common(20)),
        "first_failures": [
            item for item in reports if not item["ok"]
        ][:10],
        "mean_repeat_units": (
            sum(item["repeat_units"] for item in reports) / max(len(reports), 1)
        ),
        "max_repeat_units": max((item["repeat_units"] for item in reports), default=0),
        "max_nodes": max((item["nodes"] for item in reports), default=0),
        "max_relations": max((item["relations"] for item in reports), default=0),
        "bytes_per_sample": total_bytes / max(len(reports), 1),
        "estimated_full_hours": (
            estimated_full_count / max(len(reports) / elapsed, 1e-9) / 3600.0
        ),
        "estimated_full_gib": (
            total_bytes / max(len(reports), 1) * estimated_full_count / 1024 ** 3
        ),
        "peak_parent_rss_gib": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
        ),
    }
    result["passed"] = bool(
        result["internal_failures"] == 0
        and result["parity_failures"] == 0
        and result["identity_failures"] == 0
        and result["star_self_failures"] == 0
        and result["estimated_full_hours"] < 12.0
        and result["estimated_full_gib"] < 250.0
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
