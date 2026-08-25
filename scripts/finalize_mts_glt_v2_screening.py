#!/usr/bin/env python3
"""Resolve the documented MTS-GLT-v2 20k and Compact19 gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def decide(infonce: dict, compact: dict, descriptor: dict | None) -> dict:
    selected = dict(infonce["selected"])
    paired = bool(infonce["paired_gate_passed"])
    concat_delta = float(compact["frozen_concat_macro_delta"])
    concat_fallback = bool(concat_delta > 0)
    descriptor_admitted = bool(compact["admitted"])
    descriptor_enabled = False
    if descriptor_admitted:
        if descriptor is None:
            raise ValueError("admitted Compact19 requires the matched finetune report")
        descriptor_enabled = bool(
            float(descriptor["macro_delta"]) > 0
            and int(descriptor["positive_tasks"]) >= 2
        )
    weight = float(selected["weight"])
    weight_tag = {0.0: "w0", 0.25: "w025", 1.0: "w1"}[weight]
    return {
        "schema": "mts-glt-v2-screening-decision-v1",
        "selected_weight": weight,
        "selected_checkpoint": selected["checkpoint"],
        "selected_config": selected["config"],
        "selected_probe_run_name": selected["run_name"],
        "formal_run_name": f"formal/a6_h_{weight_tag}_20k",
        "paired_gate_passed": paired,
        "frozen_concat_macro_delta": concat_delta,
        "frozen_concat_fallback_passed": concat_fallback,
        "go_20k": bool(paired or concat_fallback),
        "compact19_admitted": descriptor_admitted,
        "compact19_enabled": descriptor_enabled,
        "descriptor_probe_macro_delta": (
            None if descriptor is None else float(descriptor["macro_delta"])
        ),
        "descriptor_probe_positive_tasks": (
            None if descriptor is None else int(descriptor["positive_tasks"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infonce-selection", required=True, type=Path)
    parser.add_argument("--compact-report", required=True, type=Path)
    parser.add_argument("--descriptor-report", type=Path)
    parser.add_argument(
        "--output",
        default="results/mts_glt_v2/screening_decision.json",
        type=Path,
    )
    args = parser.parse_args()
    infonce_path = (ROOT / args.infonce_selection).resolve()
    compact_path = (ROOT / args.compact_report).resolve()
    infonce = json.loads(infonce_path.read_text(encoding="utf-8"))
    compact = json.loads(compact_path.read_text(encoding="utf-8"))
    descriptor = None
    if args.descriptor_report is not None:
        descriptor_path = (ROOT / args.descriptor_report).resolve()
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    payload = decide(infonce, compact, descriptor)
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
