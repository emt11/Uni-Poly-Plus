#!/usr/bin/env python3
"""Apply the documented MTS-GLT-v2 architecture/InfoNCE ranking rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _metrics(result_root):
    rows = [
        json.loads(line) for line in (result_root / "training_metrics.jsonl")
        .read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise RuntimeError(f"empty metrics: {result_root}")
    return {
        "samples_per_second": float(np.median([
            row["samples_per_second"] for row in rows[-min(200, len(rows)):]
        ])),
        "peak_memory_bytes": int(max(row["peak_memory_bytes"] for row in rows)),
        "finite": all(
            np.isfinite(row[name]) for row in rows
            for name in ("masked_atom_loss", "masked_line_loss", "infonce_loss")
        ),
    }


def _candidate(name):
    pretrain = ROOT / "results/mts_glt_v2/architecture_screen" / name
    downstream = (
        ROOT / "results/mts_glt_v2/downstream"
        / f"architecture_screen_{name}" / "paired_summary.json"
    )
    summary = json.loads(downstream.read_text(encoding="utf-8"))
    return {
        "name": name,
        **_metrics(pretrain),
        "fused_macro3": float(summary["macro_fused"]),
        "delta_median": float(summary["median_task_delta"]),
        "positive_tasks": int(summary["positive_tasks"]),
        "macro_delta": float(summary["macro_delta"]),
    }


def _rank(candidates):
    candidates = [row for row in candidates if row["finite"]]
    if not candidates:
        raise RuntimeError("no finite MTS-GLT-v2 candidate")
    candidates.sort(
        key=lambda row: (
            row["fused_macro3"], row["delta_median"], row["positive_tasks"]
        ), reverse=True,
    )
    # Within the documented 0.002 performance tie, prefer throughput and then
    # lower memory without inventing another scientific score.
    best_macro = candidates[0]["fused_macro3"]
    tied = [row for row in candidates if best_macro - row["fused_macro3"] <= 0.002]
    tied.sort(
        key=lambda row: (row["samples_per_second"], -row["peak_memory_bytes"]),
        reverse=True,
    )
    return tied + [row for row in candidates if row not in tied]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates", nargs="+", default=["a6_h_2k", "a12_h_2k", "a12_p_2k"]
    )
    parser.add_argument(
        "--output", default="results/mts_glt_v2/architecture_screen/selection.json"
    )
    args = parser.parse_args()
    ranked = _rank([_candidate(name) for name in args.candidates])
    payload = {
        "schema": "mts-glt-v2-architecture-selection-v1",
        "ranked": ranked,
        "top_two": [row["name"] for row in ranked[:2]],
    }
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({"top_two": payload["top_two"]}, sort_keys=True))


if __name__ == "__main__":
    main()
