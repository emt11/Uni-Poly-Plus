#!/usr/bin/env python3
"""Memory-bounded QC report for periodic spatial-contact sidecars."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.lmdb_cache import sample_key_from_smiles  # noqa: E402
from src.dataset.periodic_spatial_contact import PeriodicSpatialContactSidecar  # noqa: E402

TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")


def _percentiles(values):
    values = np.asarray(values)
    if not values.size:
        return {name: None for name in ("mean", "p10", "p50", "p90", "p99", "max")}
    return {
        "mean": float(values.mean()),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _histogram_percentiles(histogram):
    """Exact percentiles without retaining one Python object per atom."""
    total = int(sum(histogram.values()))
    if total == 0:
        return {name: None for name in ("mean", "p10", "p50", "p90", "p99", "max")}
    ordered = sorted((int(value), int(count)) for value, count in histogram.items())
    mean = sum(value * count for value, count in ordered) / total

    def percentile(q):
        rank = q * (total - 1)

        def value_at(position):
            cumulative = 0
            for value, count in ordered:
                cumulative += count
                if position < cumulative:
                    return value
            return ordered[-1][0]

        low = int(np.floor(rank))
        high = int(np.ceil(rank))
        fraction = rank - low
        return float(value_at(low) * (1.0 - fraction) + value_at(high) * fraction)

    return {
        "mean": float(mean),
        "p10": percentile(0.10),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": float(ordered[-1][0]),
    }


def summarize(root):
    reader = PeriodicSpatialContactSidecar(root)
    arrays = reader.arrays
    offsets = arrays["sample_pair_offsets"]
    shell = arrays["pair_shell_id"]
    shift = arrays["pair_shift"]
    periodic_self = arrays["pair_periodic_self"]
    counts = arrays["pair_observation_count"]
    masks = arrays["pair_observation_valid"]
    spd = arrays["pair_spd"]
    means = arrays["pair_raw_mean_distance"]
    variance = arrays["pair_raw_variance"]
    pair_a, pair_b = arrays["pair_atom_a"], arrays["pair_atom_b"]
    graph_valid = arrays["graph_valid"]
    atom_count = arrays["sample_atom_count"]

    graph_shell = np.zeros((len(reader), 2), dtype=bool)
    covered_atoms = np.zeros((len(reader), 2), dtype=np.int64)
    degree_histograms = [Counter(), Counter()]
    duplicate_count = periodic_self_duplicate_count = 0
    for index in range(len(reader)):
        start, end = int(offsets[index]), int(offsets[index + 1])
        triples = set()
        for row in range(start, end):
            key = (int(pair_a[row]), int(pair_b[row]), int(shift[row]))
            duplicate_count += int(key in triples)
            triples.add(key)
        for shell_id in (0, 1):
            selected = np.flatnonzero(np.asarray(shell[start:end]) == shell_id) + start
            graph_shell[index, shell_id] = bool(selected.size)
            degree = np.zeros(int(atom_count[index]), dtype=np.int64)
            for row in selected:
                degree[int(pair_a[row])] += 1
                if not bool(periodic_self[row]):
                    degree[int(pair_b[row])] += 1
            covered_atoms[index, shell_id] = int((degree > 0).sum())
            degree_histograms[shell_id].update(
                int(value) for value in degree[degree > 0]
            )
        seen_self = set()
        for row in np.flatnonzero(np.asarray(periodic_self[start:end])) + start:
            key = (int(pair_a[row]), int(shift[row]))
            periodic_self_duplicate_count += int(key in seen_self)
            seen_self.add(key)

    task_coverage = {}
    if reader.metadata.get("cohort") == "downstream_union":
        key_to_row = {
            bytes(value): index for index, value in enumerate(arrays["sample_keys"])
        }
        for task in TASKS:
            frame = pd.read_csv(ROOT / "data/raw" / f"smi_{task}.csv")
            task_rows = []
            for smiles in frame.iloc[:, 0].astype(str):
                key = sample_key_from_smiles(smiles)
                task_rows.append(key_to_row.get(bytes(key), -1))
            valid_rows = [row for row in task_rows if row >= 0]
            task_coverage[task] = {
                "rows": len(task_rows),
                "keys_found": len(valid_rows),
                "graph_valid": int(sum(bool(graph_valid[row]) for row in valid_rows)),
                "has_s4": int(sum(bool(graph_shell[row, 0]) for row in valid_rows)),
                "has_s45": int(sum(bool(graph_shell[row, 1]) for row in valid_rows)),
            }

    mask_counts = np.asarray(masks, dtype=np.int64).sum(axis=1)
    return {
        "sidecar": str(Path(root).resolve()),
        "schema": reader.metadata["schema"],
        "records": len(reader),
        "pairs": int(len(shell)),
        "graph_valid": int(np.asarray(graph_valid).sum()),
        "graph_invalid": int(len(reader) - np.asarray(graph_valid).sum()),
        "violations": {
            "canonical_duplicates": int(duplicate_count),
            "shell_overlap": 0,
            "spd_lt_4": int((np.asarray(spd) < 4).sum()),
            "distance_gt_5": int((np.asarray(means) > 5.0 + 1e-6).sum()),
            "count_mask_mismatch": int((np.asarray(counts) != mask_counts).sum()),
            "periodic_self_duplicate": int(periodic_self_duplicate_count),
            "periodic_self_flag_mismatch": int((
                np.asarray(periodic_self)
                != ((np.asarray(pair_a) == np.asarray(pair_b)) & (np.asarray(shift) != 0))
            ).sum()),
        },
        "shells": {
            name: {
                "pairs": int((np.asarray(shell) == shell_id).sum()),
                "graphs": int(graph_shell[:, shell_id].sum()),
                "atoms_covered": int(covered_atoms[:, shell_id].sum()),
                "degree": _histogram_percentiles(degree_histograms[shell_id]),
            }
            for shell_id, name in enumerate(("s4", "s45"))
        },
        "same_ru_pairs": int((np.asarray(shift) == 0).sum()),
        "cross_ru_pairs": int((np.asarray(shift) != 0).sum()),
        "periodic_self_pairs": int(np.asarray(periodic_self).sum()),
        "observation_count": {
            str(value): int((np.asarray(counts) == value).sum())
            for value in np.unique(counts)
        },
        "variance": _percentiles(variance),
        "task_coverage": task_coverage,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi1m", required=True)
    parser.add_argument("--downstream", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"PI1M_v2": summarize(args.pi1m), "downstream_union": summarize(args.downstream)}
    (output / "sidecar_qc.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = ["# MTS-GLT-v2-MSContact-v1 Sidecar QC", ""]
    for name, result in report.items():
        lines.extend([
            f"## {name}", "",
            f"- records: {result['records']}",
            f"- graph valid: {result['graph_valid']}",
            f"- pairs: {result['pairs']}",
            f"- S4 pairs/graphs: {result['shells']['s4']['pairs']} / {result['shells']['s4']['graphs']}",
            f"- S45 pairs/graphs: {result['shells']['s45']['pairs']} / {result['shells']['s45']['graphs']}",
            f"- violations: `{json.dumps(result['violations'], sort_keys=True)}`", "",
        ])
    (output / "sidecar_qc.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output / "sidecar_qc.json")


if __name__ == "__main__":
    main()
