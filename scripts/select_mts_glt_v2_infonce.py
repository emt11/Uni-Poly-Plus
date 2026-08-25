#!/usr/bin/env python3
"""Select the MTS-GLT-v2 InfoNCE weight from matched 5k probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _candidate(
    weight: float, report: Path, checkpoint: Path, config: Path
) -> dict:
    payload = json.loads(report.read_text(encoding="utf-8"))
    if payload.get("schema") != "mts-glt-v2-paired-downstream-summary-v1":
        raise RuntimeError(f"unexpected paired report schema: {report}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not config.is_file():
        raise FileNotFoundError(config)
    return {
        "weight": float(weight),
        "report": str(report),
        "checkpoint": str(checkpoint),
        "config": str(config),
        "run_name": str(payload["run_name"]),
        "macro_o8": float(payload["macro_o8"]),
        "macro_fused": float(payload["macro_fused"]),
        "macro_delta": float(payload["macro_delta"]),
        "median_task_delta": float(payload["median_task_delta"]),
        "positive_tasks": int(payload["positive_tasks"]),
    }


def select(candidates: list[dict], *, tie_tolerance: float = 0.002) -> dict:
    if not candidates:
        raise ValueError("at least one InfoNCE candidate is required")
    ranked = sorted(
        candidates,
        key=lambda row: (
            row["macro_fused"], row["median_task_delta"],
            row["positive_tasks"], -row["weight"],
        ),
        reverse=True,
    )
    best_macro = ranked[0]["macro_fused"]
    tied = [
        row for row in ranked
        if best_macro - row["macro_fused"] <= float(tie_tolerance)
    ]
    selected = min(tied, key=lambda row: row["weight"])
    ordered = [selected] + [row for row in ranked if row is not selected]
    paired_gate = bool(
        selected["macro_delta"] >= -0.002
        and selected["positive_tasks"] >= 2
    )
    return {
        "schema": "mts-glt-v2-infonce-selection-v1",
        "tie_tolerance": float(tie_tolerance),
        "ranked": ordered,
        "selected": selected,
        "paired_gate_passed": paired_gate,
        "requires_frozen_concat_fallback": bool(not paired_gate),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", action="append", nargs=4, required=True,
        metavar=("WEIGHT", "REPORT", "CHECKPOINT", "CONFIG"),
        help=(
            "repeat for each weight: WEIGHT paired_summary.json "
            "checkpoint.pth source_config.json"
        ),
    )
    parser.add_argument(
        "--output",
        default="results/mts_glt_v2/infonce_screen/selection.json",
    )
    args = parser.parse_args()
    candidates = [
        _candidate(
            float(weight),
            (ROOT / report).resolve(),
            (ROOT / checkpoint).resolve(),
            (ROOT / config).resolve(),
        )
        for weight, report, checkpoint, config in args.candidate
    ]
    payload = select(candidates)
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({
        "selected_weight": payload["selected"]["weight"],
        "paired_gate_passed": payload["paired_gate_passed"],
        "requires_frozen_concat_fallback": payload[
            "requires_frozen_concat_fallback"
        ],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
