#!/usr/bin/env python3
"""Create one deterministic GraphGate step-0 state for paired validation arms."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modules import MTSGraphGateModel  # noqa: E402
from src.modules.periodic_line_glt_graphgate import NUM_LINE_LABELS  # noqa: E402
from src.training.pretrain.engine import _atomic_torch_save  # noqa: E402
from src.training.pretrain.glt_graphgate_engine import (  # noqa: E402
    MTSGraphGatePretrainContainer, _trained_state,
)
from src.training.pretrain.glt_graphgate_objectives import (  # noqa: E402
    load_line_label_counts,
)
from src.utils import set_global_seed  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--line-label-counts",
        default="results/mts_glt_graphgate_v1/sidecar_qc/line_label_counts.json",
    )
    parser.add_argument(
        "--output",
        default=(
            "pretrained_models/mts_glt_graphgate_v1/trimer_validation_v1/"
            "shared_step0_seed42.pth"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    set_global_seed(args.seed)
    frequencies = load_line_label_counts(
        ROOT / args.line_label_counts, label_count=NUM_LINE_LABELS
    )
    model = MTSGraphGateModel(layers=6)
    for parameter in model.o8_encoder.md_residual.parameters():
        parameter.requires_grad = False
    container = MTSGraphGatePretrainContainer(model, frequencies, 256)
    payload = {
        "schema": "mts-glt-graphgate-v1-shared-step0-v1",
        "seed": int(args.seed),
        "trained_state": _trained_state(container),
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(payload, output)
    reloaded = torch.load(output, map_location="cpu", weights_only=True)
    if set(reloaded["trained_state"]) != set(payload["trained_state"]):
        raise RuntimeError("shared step-0 strict key verification failed")
    for key, value in payload["trained_state"].items():
        if not torch.equal(value, reloaded["trained_state"][key]):
            raise RuntimeError(f"shared step-0 tensor changed during save: {key}")
    print(output)


if __name__ == "__main__":
    main()
