#!/usr/bin/env python3
"""Write the pre-training coverage report for GraphGate central sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.lmdb_cache import sample_key_from_smiles  # noqa: E402
from src.dataset.periodic_line_glt_central import (  # noqa: E402
    INVALID_ANGLE, INVALID_BASE_MAPPING, INVALID_DISTANCE,
    PeriodicLineGLTCentralSidecar,
)
from src.dataset.periodic_line_glt import PeriodicLineGLTSidecar  # noqa: E402

TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")


def _summary(sidecar):
    arrays = sidecar.arrays
    reasons = np.asarray(arrays["graph_invalid_reason"])
    runtime = np.asarray(arrays["graph_runtime_valid"], dtype=bool)
    token_valid = np.asarray(arrays["token_runtime_valid"], dtype=bool)
    relation_real = ~np.asarray(arrays["relation_is_fallback"], dtype=bool)
    relation_valid = np.asarray(arrays["relation_runtime_valid"], dtype=bool)[relation_real]
    return {
        "graph_count": len(sidecar),
        "graph_runtime_valid_count": int(runtime.sum()),
        "graph_runtime_valid_rate": float(runtime.mean()) if runtime.size else 0.0,
        "invalid_base_mapping": int((reasons == INVALID_BASE_MAPPING).sum()),
        "invalid_distance": int((reasons == INVALID_DISTANCE).sum()),
        "invalid_angle": int((reasons == INVALID_ANGLE).sum()),
        "token_count": int(token_valid.size),
        "token_runtime_valid_rate": float(token_valid.mean()) if token_valid.size else 0.0,
        "real_relation_count": int(relation_valid.size),
        "relation_runtime_valid_rate": float(relation_valid.mean()) if relation_valid.size else 0.0,
        "reason_total_reconciles": int(runtime.sum()) + int((reasons != 0).sum()) == len(sidecar),
    }


def _task_coverage(sidecar):
    key_to_valid = {
        bytes(key): bool(valid)
        for key, valid in zip(sidecar.arrays["sample_keys"], sidecar.arrays["graph_runtime_valid"])
    }
    result = {}
    for task in TASKS:
        frame = pd.read_csv(ROOT / "data/raw" / f"smi_{task}.csv")
        keys = [sample_key_from_smiles(value) for value in frame["smiles"].astype(str)]
        present = [key for key in keys if key in key_to_valid]
        valid = sum(key_to_valid[key] for key in present)
        result[task] = {
            "rows": len(keys), "rows_present": len(present),
            "runtime_valid_rows": int(valid),
            "runtime_valid_rate": float(valid / len(present)) if present else 0.0,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi1m", default="data/processed/mips_trimer_scage/periodic_line_glt_central_v1/PI1M_v2")
    parser.add_argument("--downstream", default="data/processed/mips_trimer_scage/periodic_line_glt_central_v1/downstream_union")
    parser.add_argument("--output-root", default="results/mts_glt_graphgate_v1/sidecar_qc")
    args = parser.parse_args()
    pi1m = PeriodicLineGLTCentralSidecar(ROOT / args.pi1m)
    downstream = PeriodicLineGLTCentralSidecar(ROOT / args.downstream)
    payload = {
        "schema": "mts-glt-graphgate-v1-sidecar-coverage-v1",
        "PI1M_v2": _summary(pi1m),
        "downstream_union": {**_summary(downstream), "task_coverage": _task_coverage(downstream)},
    }
    for cohort in ("PI1M_v2", "downstream_union"):
        old_root = ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_v1" / cohort
        if (old_root / ".done").is_file():
            old = PeriodicLineGLTSidecar(old_root)
            old_rate = float(np.asarray(old.arrays["graph_geometry_valid"], dtype=bool).mean())
            payload[cohort]["legacy_graph_valid_rate"] = old_rate
            payload[cohort]["runtime_valid_rate_delta_vs_legacy"] = payload[cohort]["graph_runtime_valid_rate"] - old_rate
    for cohort in ("PI1M_v2", "downstream_union"):
        if not payload[cohort]["reason_total_reconciles"]:
            raise RuntimeError(f"{cohort} invalid reasons do not reconcile")
    output = ROOT / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    (output / "coverage_report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# MTS-GLT-GraphGate-v1 Sidecar Coverage", ""]
    for cohort in ("PI1M_v2", "downstream_union"):
        item = payload[cohort]
        lines += [f"## {cohort}", "", f"- Graph runtime valid: {item['graph_runtime_valid_count']}/{item['graph_count']} ({item['graph_runtime_valid_rate']:.4%})",
                  f"- Invalid base/distance/angle: {item['invalid_base_mapping']}/{item['invalid_distance']}/{item['invalid_angle']}",
                  f"- Token valid rate: {item['token_runtime_valid_rate']:.4%}",
                  f"- Relation valid rate: {item['relation_runtime_valid_rate']:.4%}", ""]
        if "legacy_graph_valid_rate" in item:
            lines.insert(len(lines) - 1, f"- Legacy graph valid / delta: {item['legacy_graph_valid_rate']:.4%} / {item['runtime_valid_rate_delta_vs_legacy']:+.4%}")
    lines += ["## Downstream tasks", "", "| Task | Present rows | Runtime valid | Rate |", "|---|---:|---:|---:|"]
    for task, item in payload["downstream_union"]["task_coverage"].items():
        lines.append(f"| {task} | {item['rows_present']} | {item['runtime_valid_rows']} | {item['runtime_valid_rate']:.4%} |")
    (output / "coverage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output / "coverage_report.json")


if __name__ == "__main__":
    main()
