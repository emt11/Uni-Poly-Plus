#!/usr/bin/env python3
"""Audit frozen Trimer geometry without rebuilding or mutating any cache."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_mips_trimer_cache import _specs  # noqa: E402
from src.analysis.mts_trimer_validation import trimer_sample_metrics  # noqa: E402
from src.dataset.lmdb_cache import LmdbLayerStore, load_cohort, sample_key_from_smiles  # noqa: E402
from src.dataset.periodic_line_glt_central import PeriodicLineGLTCentralSidecar  # noqa: E402


TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
SIDECAR_ROOT = ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_central_v1"
DEFAULT_OUTPUT = ROOT / "results/mts_glt_graphgate_v1/trimer_validation_v1"


class StreamingDistribution:
    """Exact moments with deterministic bounded samples for quantiles."""

    def __init__(self, stride=1):
        self.stride = max(1, int(stride))
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.maximum = -float("inf")
        self.minimum = float("inf")
        self.samples = []

    def add(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if not values.size:
            return
        offset = self.count
        self.count += int(values.size)
        self.total += float(values.sum(dtype=np.float64))
        self.total_square += float(np.square(values).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        positions = np.arange(values.size, dtype=np.int64) + offset
        selected = values[(positions % self.stride) == 0]
        if selected.size:
            self.samples.append(selected.copy())

    def summary(self):
        if not self.count:
            return {
                "count": 0, "mean": None, "std": None, "p1": None,
                "p10": None, "p50": None, "p90": None, "p99": None,
                "max": None, "quantile_sample_count": 0,
            }
        sample = np.concatenate(self.samples) if self.samples else np.empty(0)
        variance = max(0.0, self.total_square / self.count - (self.total / self.count) ** 2)
        quantiles = np.quantile(sample, (0.01, 0.10, 0.50, 0.90, 0.99))
        return {
            "count": int(self.count),
            "mean": float(self.total / self.count),
            "std": float(math.sqrt(variance)),
            "p1": float(quantiles[0]), "p10": float(quantiles[1]),
            "p50": float(quantiles[2]), "p90": float(quantiles[3]),
            "p99": float(quantiles[4]), "max": float(self.maximum),
            "quantile_sample_count": int(sample.size),
        }


def _atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path, payload):
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _cohort(name):
    root = ROOT / "data/processed/mips_trimer_scage/cohorts" / name
    pointer = json.loads((root / "current.json").read_text(encoding="utf-8"))
    return load_cohort(root / pointer["cohort_hash"], load_text=False, verify_integrity=False)


def _sidecar_distributions(sidecar, chunk_size=1_000_000, target_samples=500_000):
    arrays = sidecar.arrays
    token_total = int(arrays["token_shift"].shape[0])
    relation_total = int(arrays["relation_span"].shape[0])
    token_stride = max(1, token_total // target_samples)
    relation_stride = max(1, relation_total // target_samples)
    token_stats = {
        f"shift_{shift}_{kind}": StreamingDistribution(token_stride)
        for shift in (0, 1) for kind in ("raw_distance", "policy_scalar_mean")
    }
    boundary_abs = StreamingDistribution(max(1, len(sidecar) // target_samples))
    boundary_rel = StreamingDistribution(max(1, len(sidecar) // target_samples))
    distance_groups = defaultdict(lambda: StreamingDistribution(token_stride))

    for start in range(0, token_total, chunk_size):
        end = min(token_total, start + chunk_size)
        shift = np.abs(np.asarray(arrays["token_shift"][start:end], dtype=np.int16))
        valid = np.asarray(arrays["token_runtime_valid"][start:end], dtype=bool)
        observation_valid = np.asarray(arrays["token_observation_valid"][start:end], dtype=bool)
        distances = np.asarray(arrays["token_observation_distances"][start:end], dtype=np.float64)
        z_a = np.asarray(arrays["token_endpoint_z_a"][start:end], dtype=np.int16)
        z_b = np.asarray(arrays["token_endpoint_z_b"][start:end], dtype=np.int16)
        bond = np.asarray(arrays["token_bond_type"][start:end], dtype=np.int16)
        for shift_class in (0, 1):
            selected = valid & (shift == shift_class)
            raw_mask = selected[:, None] & observation_valid
            token_stats[f"shift_{shift_class}_raw_distance"].add(distances[raw_mask])
            denominator = observation_valid[selected].sum(axis=1)
            means = (distances[selected] * observation_valid[selected]).sum(axis=1) / np.maximum(denominator, 1)
            token_stats[f"shift_{shift_class}_policy_scalar_mean"].add(means)
        selected = valid & (shift == 1) & observation_valid[:, 0] & observation_valid[:, 1]
        left, right = distances[selected, 0], distances[selected, 1]
        difference = np.abs(left - right)
        boundary_abs.add(difference)
        boundary_rel.add(difference / (0.5 * (left + right) + 1e-8))

        valid_rows = np.nonzero(valid)[0]
        if valid_rows.size:
            group_table = np.stack((
                np.minimum(z_a[valid_rows], z_b[valid_rows]),
                np.maximum(z_a[valid_rows], z_b[valid_rows]),
                bond[valid_rows], shift[valid_rows],
            ), axis=1)
            for group in np.unique(group_table, axis=0):
                row_mask = np.all(group_table == group, axis=1)
                rows = valid_rows[row_mask]
                values = distances[rows][observation_valid[rows]]
                key = f"Z{int(group[0])}-Z{int(group[1])}/bond{int(group[2])}/shift{int(group[3])}"
                distance_groups[key].add(values)

    relation_stats = {
        f"span_{span}_{kind}": StreamingDistribution(relation_stride)
        for span in (0, 1, 2) for kind in ("raw_angle_rad", "policy_scalar_mean_rad")
    }
    span1_abs = StreamingDistribution(relation_stride)
    for start in range(0, relation_total, chunk_size):
        end = min(relation_total, start + chunk_size)
        span = np.asarray(arrays["relation_span"][start:end], dtype=np.int8)
        source = np.asarray(arrays["relation_source"][start:end], dtype=np.int64)
        target = np.asarray(arrays["relation_target"][start:end], dtype=np.int64)
        fallback = np.asarray(arrays["relation_is_fallback"][start:end], dtype=bool)
        runtime_valid = np.asarray(arrays["relation_runtime_valid"][start:end], dtype=bool)
        observation_valid = np.asarray(arrays["relation_observation_valid"][start:end], dtype=bool)
        angles = np.asarray(arrays["relation_observation_angles"][start:end], dtype=np.float64)
        canonical_direction = source <= target
        for span_class in (0, 1, 2):
            selected = (~fallback) & runtime_valid & canonical_direction & (span == span_class)
            raw_mask = selected[:, None] & observation_valid
            relation_stats[f"span_{span_class}_raw_angle_rad"].add(angles[raw_mask])
            denominator = observation_valid[selected].sum(axis=1)
            means = (angles[selected] * observation_valid[selected]).sum(axis=1) / np.maximum(denominator, 1)
            relation_stats[f"span_{span_class}_policy_scalar_mean_rad"].add(means)
        selected = (
            (~fallback) & runtime_valid & canonical_direction & (span == 1)
            & observation_valid[:, 0] & observation_valid[:, 1]
        )
        span1_abs.add(np.abs(angles[selected, 0] - angles[selected, 1]))

    result = {
        "quantiles": "deterministic_stride_sample; moments/count/max are exact",
        "tokens": {key: value.summary() for key, value in token_stats.items()},
        "boundary_distance_absolute_asymmetry": boundary_abs.summary(),
        "boundary_distance_relative_asymmetry": boundary_rel.summary(),
        "relations": {key: value.summary() for key, value in relation_stats.items()},
        "span1_angle_absolute_asymmetry_rad": span1_abs.summary(),
        "distance_by_atom_pair_bond_shift": {
            key: value.summary() for key, value in sorted(distance_groups.items())
        },
    }
    degree = dict(result["span1_angle_absolute_asymmetry_rad"])
    for key in ("mean", "std", "p1", "p10", "p50", "p90", "p99", "max"):
        if degree[key] is not None:
            degree[key] = math.degrees(degree[key])
    result["span1_angle_absolute_asymmetry_deg"] = degree
    return result


def _pi1m_sample_rows(sidecar, count=10_000):
    valid = np.nonzero(np.asarray(sidecar.arrays["graph_runtime_valid"], dtype=bool))[0]
    token_counts = np.diff(np.asarray(sidecar.arrays["sample_token_offsets"], dtype=np.int64))
    ordered = valid[np.lexsort((valid, token_counts[valid]))]
    selected = []
    for bucket in np.array_split(ordered, 10):
        take = min(1000, len(bucket))
        positions = np.linspace(0, len(bucket) - 1, take, dtype=np.int64)
        selected.extend(bucket[positions].tolist())
    selected = np.asarray(sorted(set(selected)), dtype=np.int64)
    if selected.size != min(count, valid.size):
        raise RuntimeError(f"deterministic PI1M selection produced {selected.size} rows")
    return selected


def _task_membership():
    membership = defaultdict(list)
    for task in TASKS:
        frame = pd.read_csv(ROOT / "data/raw" / f"smi_{task}.csv")
        for smiles in frame["smiles"].astype(str):
            membership[sample_key_from_smiles(smiles)].append(task)
    return membership


def _sample_metrics(cohort_name, sidecar, rows, topology_store, trimer_store, membership=None):
    records = []
    keys = sidecar.arrays["sample_keys"]
    for position, row in enumerate(rows):
        key = bytes(keys[int(row)])
        metrics = trimer_sample_metrics(
            topology_store[key], trimer_store[key], sidecar.qc_row(int(row))
        )
        metrics.update({
            "cohort": cohort_name, "row": int(row), "sample_key": key.hex(),
            "tasks": ";".join(sorted(set((membership or {}).get(key, [])))),
        })
        records.append(metrics)
        if (position + 1) % 1000 == 0:
            print(f"{cohort_name}: sample metrics {position + 1}/{len(rows)}", flush=True)
    return pd.DataFrame(records)


def _metric_distributions(frame):
    excluded = {"cohort", "row", "sample_key", "tasks", "geometry_valid", "failure_reason"}
    result = {}
    for column in frame.columns:
        if column in excluded or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = frame.loc[frame["geometry_valid"].astype(bool), column].to_numpy(dtype=np.float64)
        stats = StreamingDistribution(1)
        stats.add(values)
        result[column] = stats.summary()
    return result


def write_report(payload, output):
    lines = [
        "# MTS Trimer 3D Geometry Consistency", "",
        "> Read-only audit of the frozen open Trimer and existing central GLT sidecars.", "",
    ]
    for cohort in ("PI1M_v2", "downstream_union"):
        item = payload["sidecar_distributions"][cohort]
        distance = item["boundary_distance_relative_asymmetry"]
        angle = item["span1_angle_absolute_asymmetry_deg"]
        lines += [
            f"## {cohort}", "",
            f"- Boundary relative distance asymmetry p50/p90/p99: `{distance['p50']:.6f}` / `{distance['p90']:.6f}` / `{distance['p99']:.6f}`",
            f"- Span-1 angle asymmetry (degree) p50/p90/p99: `{angle['p50']:.6f}` / `{angle['p90']:.6f}` / `{angle['p99']:.6f}`",
            "",
        ]
    for cohort in ("PI1M_v2_sample", "downstream_union"):
        item = payload["sample_distributions"][cohort]
        lines += [f"## {cohort} raw Trimer sample", ""]
        for name in (
            "internal_bond_outer_relative_mean", "span0_angle_outer_abs_deg_mean",
            "kabsch_rmsd_over_rg_mean", "mmff_energy_per_heavy_atom",
        ):
            row = item.get(name, {})
            lines.append(
                f"- {name} p50/p90/p99: `{row.get('p50')}` / `{row.get('p90')}` / `{row.get('p99')}`"
            )
        lines.append("")
    _atomic_text(output / "geometry_qc_report.md", "\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--skip-full-sidecar", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    pi1m = PeriodicLineGLTCentralSidecar(SIDECAR_ROOT / "PI1M_v2")
    downstream = PeriodicLineGLTCentralSidecar(SIDECAR_ROOT / "downstream_union")
    sidecar_payload = {}
    if not args.skip_full_sidecar:
        print("scanning full PI1M central sidecar", flush=True)
        sidecar_payload["PI1M_v2"] = _sidecar_distributions(pi1m)
        print("scanning full downstream central sidecar", flush=True)
        sidecar_payload["downstream_union"] = _sidecar_distributions(downstream)
    else:
        existing = json.loads((output / "geometry_distribution.json").read_text(encoding="utf-8"))
        sidecar_payload = existing["sidecar_distributions"]

    specs = _specs(ROOT)
    topology_store = LmdbLayerStore(specs["topology"]["root"], require_done=True)
    trimer_store = LmdbLayerStore(specs["trimer"]["root"], require_done=True)
    try:
        pi1m_rows = _pi1m_sample_rows(pi1m)
        downstream_rows = np.arange(len(downstream), dtype=np.int64)
        pi1m_frame = _sample_metrics(
            "PI1M_v2", pi1m, pi1m_rows, topology_store, trimer_store
        )
        downstream_frame = _sample_metrics(
            "downstream_union", downstream, downstream_rows,
            topology_store, trimer_store, membership=_task_membership(),
        )
    finally:
        topology_store.close()
        trimer_store.close()

    pi1m_frame.to_csv(output / "pi1m_sample_geometry_metrics.csv", index=False)
    downstream_frame.to_csv(output / "downstream_geometry_metrics.csv", index=False)
    payload = {
        "schema": "mts-trimer-geometry-validation-v1",
        "sidecar_distributions": sidecar_payload,
        "sample_selection": {
            "PI1M_v2": "1000 deterministic ordered rows from each token-count rank decile",
            "PI1M_v2_count": int(len(pi1m_frame)),
            "downstream_union": "all unique rows",
            "downstream_union_count": int(len(downstream_frame)),
        },
        "sample_distributions": {
            "PI1M_v2_sample": _metric_distributions(pi1m_frame),
            "downstream_union": _metric_distributions(downstream_frame),
        },
    }
    _atomic_json(output / "geometry_distribution.json", payload)
    write_report(payload, output)
    print(output / "geometry_qc_report.md")


if __name__ == "__main__":
    main()
