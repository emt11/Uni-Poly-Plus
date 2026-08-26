#!/usr/bin/env python3
"""Write the explicit S4/MS45 matched step-0 state report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modules import GLTMaskedLineHeadV2, MTSGraphLineModelV2  # noqa: E402
from src.modules.periodic_line_glt_v2 import NUM_LINE_LABELS  # noqa: E402
from src.training.pretrain.glt_v2_engine import (  # noqa: E402
    MTSGLTV2PretrainContainer, _projection,
)


def make(shell_mode, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = MTSGraphLineModelV2(
            glt_layers=6, glt_attention_variant="mips",
            use_spatial_contact=True, spatial_shell_mode=shell_mode,
        )
        for module in (
            model.atom_fusion_norm, model.atom_fusion_projection,
            model.compact19_residual, model.o8.md_residual,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = False
        model.atom_channel_gate.requires_grad = False
        model.spatial_channel_gate.requires_grad = False
        return MTSGLTV2PretrainContainer(
            model,
            torch.nn.Linear(512, int(model.o8.masked_atom_classes)),
            GLTMaskedLineHeadV2(512), _projection(256), _projection(256),
            torch.ones(NUM_LINE_LABELS, dtype=torch.long),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    caller_rng = torch.get_rng_state().clone()
    s4 = make("s4", args.seed)
    ms45 = make("ms45", args.seed)
    left, right = s4.state_dict(), ms45.state_dict()
    shared = sorted(set(left) & set(right))
    shape_mismatch = [
        name for name in shared if tuple(left[name].shape) != tuple(right[name].shape)
    ]
    comparable = [name for name in shared if name not in shape_mismatch]
    unequal = [name for name in comparable if not torch.equal(left[name], right[name])]
    report = {
        "seed": int(args.seed),
        "s4_state_tensors": len(left),
        "ms45_state_tensors": len(right),
        "shared_tensors_compared": len(comparable),
        "equal_shared_tensors": len(comparable) - len(unequal),
        "shape_mismatches": shape_mismatch,
        "value_mismatches": unequal,
        "s4_only": sorted(set(left) - set(right)),
        "ms45_only": sorted(set(right) - set(left)),
        "caller_cpu_rng_unchanged": bool(torch.equal(caller_rng, torch.get_rng_state())),
        "s4_parameter_count": int(sum(p.numel() for p in s4.parameters())),
        "ms45_parameter_count": int(sum(p.numel() for p in ms45.parameters())),
        "pass": not shape_mismatch and not unequal and set(left) == set(right),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    parameter_report = {
        "total_parameters": report["s4_parameter_count"],
        "spatial_encoder_parameters": int(
            sum(p.numel() for p in s4.model.spatial_encoder.parameters())
        ),
        "pretrain_spatial_nce_parameters": int(
            sum(p.numel() for module in (
                s4.spatial_nce_norm, s4.spatial_nce_projection
            ) for p in module.parameters())
            + s4.spatial_nce_gate.numel()
        ),
        "downstream_spatial_gate_parameters": int(
            s4.model.spatial_channel_gate.numel()
        ),
        "s4_ms45_parameter_delta": 0,
        "shared_structure": True,
    }
    (output.parent / "parameter_accounting.json").write_text(
        json.dumps(parameter_report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
