#!/usr/bin/env python3
"""Derive a GraphGate trajectory stage without changing scientific settings."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="configs/mts/glt_graphgate_v1/graphgate_500.json")
    parser.add_argument("--stop-after", type=int, required=True)
    parser.add_argument("--experiment-id")
    parser.add_argument("--result-root")
    parser.add_argument("--output-path")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.base).read_text(encoding="utf-8"))
    payload["stop_after_steps"] = int(args.stop_after)
    if not 1 <= args.stop_after <= int(payload["max_optimizer_steps"]):
        parser.error("stop-after is outside the 20k scheduler horizon")
    payload["probe_steps"] = [int(args.stop_after)]
    if args.experiment_id:
        payload["experiment_id"] = args.experiment_id
    if args.result_root:
        payload["result_root"] = args.result_root
    if args.output_path:
        payload["output_path"] = args.output_path
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
