#!/usr/bin/env python
"""Read-only G-precheck for the frozen canonical MTS cache.

The precheck deliberately stops at aggregated statistics.  It does not create
relation sidecars, open an LMDB writer, change a cohort manifest, or attach
geometry to a Dataset.  A relation is considered *complete* only when every
enumerated two-edge shortest path can be mapped to the open Trimer and both
real chemical bonds and finite 3-D geometry are available.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import heapq
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_mips_trimer_cache import _specs  # noqa: E402
from src.dataset.lmdb_cache import LmdbLayerStore, load_cohort  # noqa: E402
from src.dataset.mts_relation_geometry import (  # noqa: E402
    enumerate_two_edge_paths as _shared_enumerate_two_edge_paths,
    path_geometry as _shared_path_geometry,
    prepare_topology as _shared_prepare_topology,
    prepare_trimer as _shared_prepare_trimer,
)

enumerate_two_edge_paths = _shared_enumerate_two_edge_paths
_prepare_topology = _shared_prepare_topology
_prepare_trimer = _shared_prepare_trimer
path_geometry = _shared_path_geometry


SCHEMA = "mts-g-precheck-v1"
SELECTION_SEED = 42
PI1M_MAX_SAMPLES = 50_000
RESERVOIR_LIMIT = 20_000
DISTANCE_BINS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0, float("inf"))
COSINE_BINS = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
SUPPORT_THRESHOLDS = (2, 5, 20, 50)

_WORKER_TOPOLOGY = None
_WORKER_TRIMER = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _as_bool(value, default=False) -> bool:
    if value is None:
        return bool(default)
    try:
        return bool(torch.as_tensor(value).reshape(-1)[0].item())
    except (IndexError, RuntimeError, TypeError, ValueError):
        return bool(value)


def _cohort_read_only(cache_root: Path, name: str):
    """Load an existing v2 cohort without allowing an integrity upgrade write."""

    pointer = cache_root / "cohorts" / str(name) / "current.json"
    if not pointer.is_file():
        raise RuntimeError(f"cohort pointer is missing: {pointer}")
    pointer_payload = json.loads(pointer.read_text(encoding="utf-8"))
    cohort_dir = pointer.parent / str(pointer_payload["cohort_hash"])
    manifest_path = cohort_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("integrity_schema") != "mips-trimer-scage-cohort-manifest-v2":
        raise RuntimeError(
            f"read-only G-precheck refuses a cohort needing manifest upgrade: {name}"
        )
    return load_cohort(cohort_dir, load_text=False, verify_integrity=True)


def _cache_identity(project_root: Path):
    specs = _specs(project_root)
    identity = {}
    for name in ("topology", "trimer"):
        root = Path(specs[name]["root"])
        required = [root / ".done", root / ".frozen", root / "metadata.json", root / "manifest.json"]
        if not all(path.is_file() for path in required):
            raise RuntimeError(f"frozen {name} cache is incomplete: {root}")
        identity[name] = {
            "root": str(root),
            "feature_config_hash": specs[name]["meta"].get("feature_config_hash"),
            "done_artifact_hash": root.joinpath(".done").read_text(encoding="utf-8").strip(),
            "done_file_sha256": _sha256_file(root / ".done"),
            "frozen_file_sha256": _sha256_file(root / ".frozen"),
            "metadata_file_sha256": _sha256_file(root / "metadata.json"),
            "manifest_file_sha256": _sha256_file(root / "manifest.json"),
            "metadata": json.loads((root / "metadata.json").read_text(encoding="utf-8")),
        }
    return identity


def select_sample_keys(
    cohort,
    cohort_name: str,
    *,
    max_samples=PI1M_MAX_SAMPLES,
    seed=SELECTION_SEED,
    limit_to_max=False,
):
    """Select by sample-key hash, independent of LMDB/manifest physical order."""

    keys = [bytes(row) for row in np.asarray(cohort["keys_array"], dtype=np.uint8)]
    scored = [
        (
            hashlib.sha256(f"{int(seed)}:".encode("ascii") + key).digest(),
            key,
        )
        for key in keys
    ]
    scored.sort(key=lambda item: (item[0], item[1]))
    if str(cohort_name) == "PI1M_v2" or bool(limit_to_max):
        scored = scored[: min(int(max_samples), len(scored))]
    selected = [key for _, key in scored]
    ordered_hash = hashlib.sha256(b"".join(selected)).hexdigest()
    unordered_hash = hashlib.sha256(b"".join(sorted(selected))).hexdigest()
    return selected, {
        "algorithm": "sha256(seed-ascii-colon-plus-sample-key), ascending digest then key",
        "seed": int(seed),
        "max_samples": int(max_samples) if (str(cohort_name) == "PI1M_v2" or bool(limit_to_max)) else None,
        "selected_count": len(selected),
        "source_unique_count": len(keys),
        "ordered_sample_key_hash": ordered_hash,
        "unordered_sample_key_hash": unordered_hash,
        "source_manifest_ordered_sample_key_hash": cohort["manifest"].get("ordered_sample_key_hash"),
    }


def stratum_key(
    source_atomic_number: int,
    target_atomic_number: int,
    intermediate_atomic_number: int,
    bond_types: Iterable[int],
    signed_source_shift: int,
):
    """The fixed chemistry/shift control stratum from the handoff."""

    return (
        (int(source_atomic_number), int(target_atomic_number)),
        int(intermediate_atomic_number),
        tuple(sorted(int(value) for value in bond_types)),
        int(signed_source_shift),
    )


def _stratum_text(value) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=False)


def _metric():
    return {
        "count": 0,
        "sum": 0.0,
        "sumsq": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
        "hist": [0 for _ in range(0)],
        # A max-heap (via negative keys) keeps the smallest deterministic
        # SHA-256 scores in O(log K) per value.  The previous dictionary plus
        # ``max`` scan made a large formal cohort needlessly quadratic.
        "reservoir": [],
    }


def _metric_with_hist(bin_count: int):
    value = _metric()
    value["hist"] = [0] * int(bin_count - 1)
    return value


def _metric_add(metric, value: float, token: bytes, bins: Sequence[float]):
    value = float(value)
    if not math.isfinite(value):
        return
    metric["count"] += 1
    metric["sum"] += value
    metric["sumsq"] += value * value
    metric["min"] = min(float(metric["min"]), value)
    metric["max"] = max(float(metric["max"]), value)
    index = int(np.searchsorted(np.asarray(bins, dtype=float), value, side="right") - 1)
    index = max(0, min(index, len(metric["hist"]) - 1))
    metric["hist"][index] += 1
    score = int.from_bytes(hashlib.sha256(token).digest(), "big")
    reservoir = metric["reservoir"]
    item = (-score, score, value)
    if len(reservoir) < RESERVOIR_LIMIT:
        heapq.heappush(reservoir, item)
    elif score < reservoir[0][1]:
        heapq.heapreplace(reservoir, item)


def _metric_merge(target, source):
    target["count"] += int(source["count"])
    target["sum"] += float(source["sum"])
    target["sumsq"] += float(source["sumsq"])
    if int(source["count"]):
        target["min"] = min(float(target["min"]), float(source["min"]))
        target["max"] = max(float(target["max"]), float(source["max"]))
    for index, value in enumerate(source["hist"]):
        target["hist"][index] += int(value)
    for _, score, value in source["reservoir"]:
        item = (-int(score), int(score), float(value))
        if len(target["reservoir"]) < RESERVOIR_LIMIT:
            heapq.heappush(target["reservoir"], item)
        elif int(score) < target["reservoir"][0][1]:
            heapq.heapreplace(target["reservoir"], item)


def _metric_summary(metric):
    count = int(metric["count"])
    if not count:
        return {"count": 0, "mean": None, "variance_population": None, "min": None, "max": None, "quantiles": {}}
    values = np.asarray([item[2] for item in metric["reservoir"]], dtype=float)
    variance = max(0.0, float(metric["sumsq"]) / count - (float(metric["sum"]) / count) ** 2)
    return {
        "count": count,
        "mean": float(metric["sum"]) / count,
        "variance_population": variance,
        "min": float(metric["min"]),
        "max": float(metric["max"]),
        "quantiles": {
            str(p): float(np.quantile(values, p))
            for p in (0.05, 0.25, 0.50, 0.75, 0.95)
        } if values.size else {},
        "quantile_source": "exact" if count <= RESERVOIR_LIMIT else f"deterministic_sha256_reservoir_{RESERVOIR_LIMIT}",
        "fixed_bin_counts": list(metric["hist"]),
    }


def _json_edges(values):
    return ["inf" if math.isinf(float(value)) and float(value) > 0 else float(value) for value in values]


def _new_aggregate():
    return {
        "sample_count": 0,
        "graph_available_samples": 0,
        "geometry_3d_samples": 0,
        "relation_count": 0,
        "relation_complete_geometry_count": 0,
        "relation_partial_geometry_count": 0,
        "relation_mixed_path_chemistry_count": 0,
        "path_count": 0,
        "valid_path_count": 0,
        "samples_without_spd2_relation": 0,
        "sample_failure_reasons": Counter(),
        "relation_invalid_reasons": Counter(),
        "path_invalid_reasons": Counter(),
        "distance": _metric_with_hist(len(DISTANCE_BINS)),
        "cosine": _metric_with_hist(len(COSINE_BINS)),
        "strata": {},
    }


def _new_stratum():
    return {
        "path_count": 0,
        "path_sum": 0.0,
        "path_sumsq": 0.0,
        "relation_count": 0,
        "relation_sum": 0.0,
        "relation_sumsq": 0.0,
        "relation_min": float("inf"),
        "relation_max": float("-inf"),
        "relation_range_sum": 0.0,
        "relation_range_sumsq": 0.0,
        "relation_range_min": float("inf"),
        "relation_range_max": float("-inf"),
        "relation_distance_sum": 0.0,
        "relation_distance_sumsq": 0.0,
        "relation_distance_min": float("inf"),
        "relation_distance_max": float("-inf"),
        "relation_distance_range_sum": 0.0,
        "relation_distance_range_sumsq": 0.0,
        "relation_distance_range_min": float("inf"),
        "relation_distance_range_max": float("-inf"),
    }


def _add_stratum_path(entry, cosine):
    value = float(cosine)
    entry["path_count"] += 1
    entry["path_sum"] += value
    entry["path_sumsq"] += value * value


def _add_stratum_relation(
    entry, mean_cos, range_cos, mean_distance, range_distance
):
    mean_cos = float(mean_cos)
    range_cos = float(range_cos)
    entry["relation_count"] += 1
    entry["relation_sum"] += mean_cos
    entry["relation_sumsq"] += mean_cos * mean_cos
    entry["relation_min"] = min(float(entry["relation_min"]), mean_cos)
    entry["relation_max"] = max(float(entry["relation_max"]), mean_cos)
    entry["relation_range_sum"] += range_cos
    entry["relation_range_sumsq"] += range_cos * range_cos
    entry["relation_range_min"] = min(float(entry["relation_range_min"]), range_cos)
    entry["relation_range_max"] = max(float(entry["relation_range_max"]), range_cos)
    mean_distance = float(mean_distance)
    range_distance = float(range_distance)
    entry["relation_distance_sum"] += mean_distance
    entry["relation_distance_sumsq"] += mean_distance * mean_distance
    entry["relation_distance_min"] = min(float(entry["relation_distance_min"]), mean_distance)
    entry["relation_distance_max"] = max(float(entry["relation_distance_max"]), mean_distance)
    entry["relation_distance_range_sum"] += range_distance
    entry["relation_distance_range_sumsq"] += range_distance * range_distance
    entry["relation_distance_range_min"] = min(float(entry["relation_distance_range_min"]), range_distance)
    entry["relation_distance_range_max"] = max(float(entry["relation_distance_range_max"]), range_distance)


def _merge_aggregate(target, source):
    for name in (
        "sample_count", "graph_available_samples", "geometry_3d_samples",
        "relation_count", "relation_complete_geometry_count",
        "relation_partial_geometry_count", "path_count", "valid_path_count",
        "samples_without_spd2_relation", "relation_mixed_path_chemistry_count",
    ):
        target[name] += int(source[name])
    target["sample_failure_reasons"].update(source["sample_failure_reasons"])
    target["relation_invalid_reasons"].update(source["relation_invalid_reasons"])
    target["path_invalid_reasons"].update(source["path_invalid_reasons"])
    _metric_merge(target["distance"], source["distance"])
    _metric_merge(target["cosine"], source["cosine"])
    for key, source_entry in source["strata"].items():
        entry = target["strata"].setdefault(key, _new_stratum())
        for name in ("path_count", "relation_count"):
            entry[name] += int(source_entry[name])
        for name in (
            "path_sum", "path_sumsq", "relation_sum", "relation_sumsq",
            "relation_range_sum", "relation_range_sumsq",
            "relation_distance_sum", "relation_distance_sumsq",
            "relation_distance_range_sum", "relation_distance_range_sumsq",
        ):
            entry[name] += float(source_entry[name])
        for name in (
            "relation_min", "relation_max", "relation_range_min", "relation_range_max",
            "relation_distance_min", "relation_distance_max",
            "relation_distance_range_min", "relation_distance_range_max",
        ):
            if source_entry["relation_count"]:
                if name.endswith("min"):
                    entry[name] = min(float(entry[name]), float(source_entry[name]))
                else:
                    entry[name] = max(float(entry[name]), float(source_entry[name]))


def _record_aggregate(key: bytes, topology, trimer):
    aggregate = _new_aggregate()
    aggregate["sample_count"] = 1
    if _as_bool(getattr(topology, "graph_available", False)):
        aggregate["graph_available_samples"] += 1
    if _as_bool(getattr(trimer, "trimer_geometry_valid", False)) and _as_bool(getattr(trimer, "trimer_geometry_is_3d", False)):
        aggregate["geometry_3d_samples"] += 1
    prepared, topology_reason = _prepare_topology(topology)
    trimer_info, trimer_reason = _prepare_trimer(trimer, topology)
    if topology_reason:
        aggregate["sample_failure_reasons"][topology_reason] += 1
    if not _as_bool(getattr(topology, "graph_available", False)):
        aggregate["sample_failure_reasons"]["graph_unavailable"] += 1
    if trimer_reason:
        aggregate["sample_failure_reasons"][trimer_reason] += 1
    if prepared is None:
        return aggregate
    rows = torch.nonzero(prepared["spd"] == 2, as_tuple=False).flatten().tolist()
    if not rows:
        aggregate["samples_without_spd2_relation"] = 1
        return aggregate
    for row in rows:
        aggregate["relation_count"] += 1
        relation_reason = topology_reason or trimer_reason
        if not _as_bool(getattr(topology, "graph_available", False)):
            relation_reason = "graph_unavailable"
        paths = []
        if relation_reason is None:
            source = (int(prepared["edge"][0, row]), int(prepared["shift"][row]))
            target = (int(prepared["edge"][1, row]), 0)
            paths = enumerate_two_edge_paths(
                source, target, prepared["internal_edges"], prepared["left"], prepared["right"]
            )
            if not paths:
                relation_reason = "no_two_edge_paths"
        aggregate["path_count"] += len(paths)
        valid = []
        invalid = []
        for path_index, path in enumerate(paths):
            geometry, reason = path_geometry(path, trimer_info)
            if reason:
                invalid.append(reason)
                aggregate["path_invalid_reasons"][reason] += 1
                continue
            valid.append((path_index, path, geometry))
        if relation_reason is None and not valid:
            relation_reason = invalid[0] if invalid else "no_valid_geometry"
        if valid:
            aggregate["valid_path_count"] += len(valid)
            source_z = int(prepared["z"][int(prepared["edge"][0, row])])
            target_z = int(prepared["z"][int(prepared["edge"][1, row])])
            relation_values = {}
            relation_strata = Counter()
            for path_index, path, geometry in valid:
                source_local = geometry["source_local"]
                middle_local = geometry["middle_local"]
                target_local = geometry["target_local"]
                key_stratum = stratum_key(
                    source_z,
                    target_z,
                    int(trimer_info["atomic"][middle_local]),
                    geometry["bond_types"],
                    int(path[0][1]),
                )
                key_text = _stratum_text(key_stratum)
                relation_strata[key_text] += 1
                entry = aggregate["strata"].setdefault(key_text, _new_stratum())
                _add_stratum_path(entry, geometry["cosine"])
                relation_values.setdefault(key_text, []).append(
                    (geometry["cosine"], geometry["distance"])
                )
                token = key + int(row).to_bytes(4, "little") + int(path_index).to_bytes(4, "little")
                _metric_add(aggregate["distance"], geometry["distance"], token + b"d", DISTANCE_BINS)
                _metric_add(aggregate["cosine"], geometry["cosine"], token + b"c", COSINE_BINS)
            # A relation can legitimately have multiple chemistry strata when
            # distinct shortest paths use different intermediate atoms or
            # bond types.  Keep a relation-level mean/range inside each
            # stratum and expose the mixed count instead of collapsing paths.
            if len(relation_strata) > 1:
                aggregate["relation_mixed_path_chemistry_count"] += 1
            for stratum, count in relation_strata.items():
                values = relation_values[stratum]
                cosine_values = [value[0] for value in values]
                distance_values = [value[1] for value in values]
                entry = aggregate["strata"][stratum]
                _add_stratum_relation(
                    entry,
                    float(np.mean(cosine_values)),
                    float(np.max(cosine_values) - np.min(cosine_values)),
                    float(np.mean(distance_values)),
                    float(np.max(distance_values) - np.min(distance_values)),
                )
            if not invalid:
                aggregate["relation_complete_geometry_count"] += 1
            else:
                aggregate["relation_partial_geometry_count"] += 1
                relation_reason = "partial_path_geometry"
        if relation_reason is not None:
            aggregate["relation_invalid_reasons"][relation_reason] += 1
    return aggregate


def _init_worker(topology_root: str, trimer_root: str):
    global _WORKER_TOPOLOGY, _WORKER_TRIMER
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _WORKER_TOPOLOGY = LmdbLayerStore(topology_root, require_done=True)
    _WORKER_TRIMER = LmdbLayerStore(trimer_root, require_done=True)
    atexit.register(_close_worker)


def _close_worker():
    global _WORKER_TOPOLOGY, _WORKER_TRIMER
    if _WORKER_TOPOLOGY is not None:
        _WORKER_TOPOLOGY.close()
        _WORKER_TOPOLOGY = None
    if _WORKER_TRIMER is not None:
        _WORKER_TRIMER.close()
        _WORKER_TRIMER = None


def _analyze_chunk(keys):
    output = _new_aggregate()
    for key in keys:
        key = bytes(key)
        try:
            topology = _WORKER_TOPOLOGY[key]
            trimer = _WORKER_TRIMER[key]
            _merge_aggregate(output, _record_aggregate(key, topology, trimer))
        except Exception as exc:  # retain a stable sample-level reason
            output["sample_count"] += 1
            output["sample_failure_reasons"][f"record_read_error:{type(exc).__name__}"] += 1
    return output


def _variance_rows(aggregate):
    rows = []
    for unit_name, value_getter in (
        ("path", lambda item: (item["path_count"], item["path_sum"], item["path_sumsq"])),
        ("relation_mean", lambda item: (item["relation_count"], item["relation_sum"], item["relation_sumsq"])),
    ):
        totals = []
        for item in aggregate["strata"].values():
            count, total, total_sq = value_getter(item)
            if count:
                totals.append((int(count), float(total), float(total_sq)))
        total_count = sum(item[0] for item in totals)
        if not total_count:
            continue
        total_sum = sum(item[1] for item in totals)
        total_sq = sum(item[2] for item in totals)
        total_mean = total_sum / total_count
        total_variance = max(0.0, total_sq / total_count - total_mean * total_mean)
        for threshold in SUPPORT_THRESHOLDS:
            supported = [item for item in totals if item[0] >= threshold]
            count = sum(item[0] for item in supported)
            if not count:
                rows.append({"unit": unit_name, "support_n": threshold, "total_variance": total_variance, "within_variance": None, "between_variance": None, "within_over_total": None, "supported_count": 0, "supported_strata": 0})
                continue
            supported_sum = sum(item[1] for item in supported)
            supported_sq = sum(item[2] for item in supported)
            mean = supported_sum / count
            total = max(0.0, supported_sq / count - mean * mean)
            within = sum(max(0.0, item[2] - item[1] * item[1] / item[0]) for item in supported) / count
            between = max(0.0, total - within)
            rows.append({
                "unit": unit_name,
                "support_n": threshold,
                "total_variance": total,
                "within_variance": within,
                "between_variance": between,
                "within_over_total": within / total if total > 0 else 0.0,
                "supported_count": count,
                "supported_strata": len(supported),
                "all_valid_count": total_count,
                "all_strata": len(totals),
            })
    return rows


def _finalize_summary(aggregate, *, cohort_name, selection, source_identity, mode, elapsed_seconds, workers, chunk_size):
    strata = {}
    for key, item in sorted(aggregate["strata"].items()):
        path_count = int(item["path_count"])
        relation_count = int(item["relation_count"])
        strata[key] = {
            "stratum": json.loads(key),
            "path_count": path_count,
            "relation_count": relation_count,
            "path_cosine_mean": item["path_sum"] / path_count if path_count else None,
            "path_cosine_variance_population": max(0.0, item["path_sumsq"] / path_count - (item["path_sum"] / path_count) ** 2) if path_count else None,
            "relation_cosine_mean_mean": item["relation_sum"] / relation_count if relation_count else None,
            "relation_cosine_mean_variance_population": max(0.0, item["relation_sumsq"] / relation_count - (item["relation_sum"] / relation_count) ** 2) if relation_count else None,
            "relation_cosine_mean_min": item["relation_min"] if relation_count else None,
            "relation_cosine_mean_max": item["relation_max"] if relation_count else None,
            "relation_cosine_range_mean": item["relation_range_sum"] / relation_count if relation_count else None,
            "relation_cosine_range_min": item["relation_range_min"] if relation_count else None,
            "relation_cosine_range_max": item["relation_range_max"] if relation_count else None,
            "relation_distance_mean": item["relation_distance_sum"] / relation_count if relation_count else None,
            "relation_distance_variance_population": max(0.0, item["relation_distance_sumsq"] / relation_count - (item["relation_distance_sum"] / relation_count) ** 2) if relation_count else None,
            "relation_distance_min": item["relation_distance_min"] if relation_count else None,
            "relation_distance_max": item["relation_distance_max"] if relation_count else None,
            "relation_distance_range_mean": item["relation_distance_range_sum"] / relation_count if relation_count else None,
            "relation_distance_range_min": item["relation_distance_range_min"] if relation_count else None,
            "relation_distance_range_max": item["relation_distance_range_max"] if relation_count else None,
        }
    relation_count = int(aggregate["relation_count"])
    valid_path_count = int(aggregate["valid_path_count"])
    relation_stratum_observation_count = sum(
        int(item["relation_count"]) for item in aggregate["strata"].values()
    )
    return {
        "schema": SCHEMA,
        "analysis_mode": str(mode),
        "cohort": str(cohort_name),
        "source_identity": source_identity,
        "selection": selection,
        "sample_count": int(aggregate["sample_count"]),
        "graph_available_samples": int(aggregate["graph_available_samples"]),
        "geometry_3d_samples": int(aggregate["geometry_3d_samples"]),
        "relation_count": relation_count,
        "relation_complete_geometry_count": int(aggregate["relation_complete_geometry_count"]),
        "relation_partial_geometry_count": int(aggregate["relation_partial_geometry_count"]),
        "relation_mixed_path_chemistry_count": int(aggregate["relation_mixed_path_chemistry_count"]),
        "relation_stratum_observation_count": relation_stratum_observation_count,
        "relation_geometry_complete_rate": (aggregate["relation_complete_geometry_count"] / relation_count if relation_count else None),
        "path_count": int(aggregate["path_count"]),
        "valid_path_count": valid_path_count,
        "path_geometry_valid_rate": (valid_path_count / aggregate["path_count"] if aggregate["path_count"] else None),
        "samples_without_spd2_relation": int(aggregate["samples_without_spd2_relation"]),
        "sample_failure_reasons": dict(sorted(aggregate["sample_failure_reasons"].items())),
        "relation_invalid_reasons": dict(sorted(aggregate["relation_invalid_reasons"].items())),
        "path_invalid_reasons": dict(sorted(aggregate["path_invalid_reasons"].items())),
        "endpoint_distance": _metric_summary(aggregate["distance"]),
        "angle_cosine": _metric_summary(aggregate["cosine"]),
        "fixed_bins": {
            "endpoint_distance_edges": _json_edges(DISTANCE_BINS),
            "angle_cosine_edges": _json_edges(COSINE_BINS),
        },
        "strata": strata,
        "support_coverage": [
            {
                "support_n": threshold,
                "supported_strata": sum(1 for item in strata.values() if item["relation_count"] >= threshold),
                "relation_coverage": (sum(item["relation_count"] for item in strata.values() if item["relation_count"] >= threshold) / relation_stratum_observation_count if relation_stratum_observation_count else None),
                "path_coverage": (sum(item["path_count"] for item in strata.values() if item["relation_count"] >= threshold) / valid_path_count if valid_path_count else None),
            }
            for threshold in SUPPORT_THRESHOLDS
        ],
        "conditional_variance": _variance_rows(aggregate),
        "runtime": {
            "elapsed_seconds": float(elapsed_seconds),
            "samples_per_second": int(aggregate["sample_count"]) / max(float(elapsed_seconds), 1e-9),
            "workers": int(workers),
            "chunk_size": int(chunk_size),
            "gpu_used": False,
        },
    }


def _write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _run_one(project_root: Path, cache_root: Path, cohort_name: str, *, output_dir: Path, mode: str, max_samples: int, workers: int, chunk_size: int):
    cohort = _cohort_read_only(cache_root, cohort_name)
    selected, selection = select_sample_keys(
        cohort,
        cohort_name,
        max_samples=max_samples,
        limit_to_max=(mode == "preflight_smoke"),
    )
    source_identity = _cache_identity(project_root)
    chunks = [selected[index:index + int(chunk_size)] for index in range(0, len(selected), int(chunk_size))]
    aggregate = _new_aggregate()
    started = time.monotonic()
    print(f"[g-precheck] cohort={cohort_name} mode={mode} samples={len(selected)} workers={workers} chunks={len(chunks)}", flush=True)
    if chunks and int(workers) == 0:
        # A single-process read-only path is useful as a reproducibility
        # oracle for the worker implementation and never opens an LMDB writer.
        _init_worker(
            source_identity["topology"]["root"],
            source_identity["trimer"]["root"],
        )
        try:
            iterator = (_analyze_chunk(chunk) for chunk in chunks)
            for completed, result in enumerate(iterator, start=1):
                _merge_aggregate(aggregate, result)
                if completed == 1 or completed == len(chunks) or completed % max(1, len(chunks) // 20) == 0:
                    print(f"[g-precheck] cohort={cohort_name} chunks={completed}/{len(chunks)} relations={aggregate['relation_count']} paths={aggregate['path_count']}", flush=True)
        finally:
            _close_worker()
    elif chunks:
        with ProcessPoolExecutor(
            max_workers=int(workers),
            initializer=_init_worker,
            initargs=(source_identity["topology"]["root"], source_identity["trimer"]["root"]),
        ) as executor:
            for completed, result in enumerate(executor.map(_analyze_chunk, chunks), start=1):
                _merge_aggregate(aggregate, result)
                if completed == 1 or completed == len(chunks) or completed % max(1, len(chunks) // 20) == 0:
                    print(f"[g-precheck] cohort={cohort_name} chunks={completed}/{len(chunks)} relations={aggregate['relation_count']} paths={aggregate['path_count']}", flush=True)
    elapsed = max(time.monotonic() - started, 1e-9)
    summary = _finalize_summary(
        aggregate,
        cohort_name=cohort_name,
        selection=selection,
        source_identity=source_identity,
        mode=mode,
        elapsed_seconds=elapsed,
        workers=workers,
        chunk_size=chunk_size,
    )
    out_dir = output_dir / str(cohort_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(out_dir / "manifest.json", {
        "schema": SCHEMA,
        "analysis_mode": mode,
        "cohort": cohort_name,
        "source_cohort_manifest": cohort["manifest"],
        "selection": selection,
        "source_identity": source_identity,
        "sample_count": summary["sample_count"],
        "relation_count": summary["relation_count"],
        "path_count": summary["path_count"],
        "created_at": time.time(),
    })
    _atomic_json(out_dir / "relation_summary.json", summary)
    _write_csv(
        out_dir / "conditional_variance.csv",
        summary["conditional_variance"],
        ["unit", "support_n", "total_variance", "within_variance", "between_variance", "within_over_total", "supported_count", "supported_strata", "all_valid_count", "all_strata"],
    )
    invalid_rows = []
    for kind, values in (("sample", summary["sample_failure_reasons"]), ("relation", summary["relation_invalid_reasons"]), ("path", summary["path_invalid_reasons"])):
        for reason, count in values.items():
            invalid_rows.append({"scope": kind, "reason": reason, "count": count})
    _write_csv(out_dir / "invalid_reasons.csv", invalid_rows, ["scope", "reason", "count"])
    return summary


def _main(args):
    project_root = PROJECT_ROOT
    cache_root = (project_root / args.cache_root).resolve()
    output_root = (project_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    names = [item.strip() for item in str(args.cohorts).split(",") if item.strip()]
    if set(names) != {"PI1M_v2", "downstream_union"}:
        raise SystemExit("--cohorts must contain exactly PI1M_v2,downstream_union")
    summaries = {}
    for name in ("PI1M_v2", "downstream_union"):
        summaries[name] = _run_one(
            project_root,
            cache_root,
            name,
            output_dir=output_root / ("preflight" if args.preflight else ""),
            mode="preflight_smoke" if args.preflight else "formal_read_only",
            max_samples=min(int(args.max_samples), 100) if args.preflight else int(args.max_samples),
            workers=int(args.workers),
            chunk_size=int(args.chunk_size),
        )
    if not args.preflight:
        comparison = {
            "schema": SCHEMA,
            "analysis_mode": "formal_read_only",
            "route_decision": "pending_codex_review",
            "cohorts": summaries,
            "note": "Objective statistics only; Claude Code does not decide proceed_to_g_prep/stop_geometry_route/inconclusive.",
            "created_at": time.time(),
        }
        _atomic_json(output_root / "comparison_summary.json", comparison)
    else:
        _atomic_json(output_root / "preflight_summary.json", {
            "schema": SCHEMA,
            "analysis_mode": "preflight_smoke",
            "cohorts": summaries,
            "note": "Smoke only; excluded from the formal scientific conclusion.",
            "created_at": time.time(),
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", default="PI1M_v2,downstream_union")
    parser.add_argument("--cache-root", default="data/processed/mips_trimer_scage")
    parser.add_argument("--output-root", default="results/mts_multiscale_topology/g_precheck_v1")
    parser.add_argument("--max-samples", type=int, default=PI1M_MAX_SAMPLES)
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 1))))
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if int(args.workers) < 0 or int(args.chunk_size) < 1:
        raise SystemExit("workers must be zero or positive; chunk-size must be positive")
    _main(args)


if __name__ == "__main__":
    main()
