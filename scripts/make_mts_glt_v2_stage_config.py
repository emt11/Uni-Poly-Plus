#!/usr/bin/env python3
"""Derive a new independent GLT-v2 stage config from an explicit candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stop-after-steps", type=int, required=True)
    parser.add_argument("--infonce-weight", type=float, choices=(0.0, 0.25, 1.0))
    args = parser.parse_args()
    source = args.source.resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "mts-glt-v2":
        raise SystemExit("source must be an MTS-GLT-v2 config")
    stop = int(args.stop_after_steps)
    if stop not in {5000, 10000, 20000}:
        raise SystemExit("stage stop must be 5k, 10k, or 20k")
    payload["experiment_id"] = f"mts_glt_v2_{args.run_name}_seed42"
    payload["result_root"] = f"results/mts_glt_v2/{args.run_name}"
    payload["output_path"] = f"pretrained_models/mts_glt_v2/{args.run_name}.pth"
    payload["stop_after_steps"] = stop
    payload["probe_steps"] = [stop]
    if stop == 20000:
        payload["probe_steps"] = [5000, 10000, 20000]
    if args.infonce_weight is not None:
        payload["infonce_loss_weight"] = float(args.infonce_weight)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"refusing to overwrite stage config: {output}")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
