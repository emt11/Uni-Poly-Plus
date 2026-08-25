#!/usr/bin/env python3
"""Publish a probe-selected MTS-GLT-v1 encoder as the formal final model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modules import MTSGraphLineModel  # noqa: E402
from src.training.common.checkpoint import save_final_state  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    payload = torch.load(args.probe, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise SystemExit("probe checkpoint must contain state_dict")
    encoder_state = {
        str(key)[len("model."):]: value
        for key, value in payload["state_dict"].items()
        if str(key).startswith("model.")
    }
    reference = MTSGraphLineModel()
    reference.load_state_dict(encoder_state, strict=True)
    final_state = {"model." + key: value for key, value in encoder_state.items()}

    def strict_load(candidate):
        for mode in ("o8_only", "o8_glt"):
            model = MTSGraphLineModel()
            model.downstream_mode = mode
            normalized = {
                str(key)[len("model."):]: value
                for key, value in candidate.items()
                if str(key).startswith("model.")
            }
            model.load_state_dict(normalized, strict=True)

    save_final_state(final_state, args.output, strict_load=strict_load)
    print(Path(args.output).resolve())


if __name__ == "__main__":
    main()
