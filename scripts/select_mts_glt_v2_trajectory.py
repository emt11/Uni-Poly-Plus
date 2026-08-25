#!/usr/bin/env python3
"""Select the earliest near-best checkpoint from matched trajectory probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def select(candidates: list[dict], *, tie_tolerance: float = 0.002) -> dict:
    if not candidates:
        raise ValueError("at least one trajectory candidate is required")
    ranked = sorted(candidates, key=lambda row: row["macro_fused"], reverse=True)
    best_macro = float(ranked[0]["macro_fused"])
    tied = [
        row for row in ranked
        if best_macro - float(row["macro_fused"]) <= float(tie_tolerance)
    ]
    selected = min(tied, key=lambda row: int(row["step"]))
    return {
        "schema": "mts-glt-v2-trajectory-selection-v1",
        "tie_tolerance": float(tie_tolerance),
        "ranked": ranked,
        "selected": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", action="append", nargs=4, required=True,
        metavar=("STEP", "REPORT", "CHECKPOINT", "RUN_NAME"),
    )
    parser.add_argument(
        "--output",
        default="results/mts_glt_v2/formal/trajectory_selection.json",
    )
    args = parser.parse_args()
    candidates = []
    for step_text, report_text, checkpoint_text, run_name in args.candidate:
        report = (ROOT / report_text).resolve()
        checkpoint = (ROOT / checkpoint_text).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        payload = json.loads(report.read_text(encoding="utf-8"))
        if payload.get("schema") != "mts-glt-v2-paired-downstream-summary-v1":
            raise RuntimeError(f"unexpected paired report schema: {report}")
        candidates.append({
            "step": int(step_text),
            "report": str(report),
            "checkpoint": str(checkpoint),
            "run_name": str(run_name),
            "macro_o8": float(payload["macro_o8"]),
            "macro_fused": float(payload["macro_fused"]),
            "macro_delta": float(payload["macro_delta"]),
            "median_task_delta": float(payload["median_task_delta"]),
            "positive_tasks": int(payload["positive_tasks"]),
        })
    result = select(candidates)
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(result["selected"], sort_keys=True))


if __name__ == "__main__":
    main()
