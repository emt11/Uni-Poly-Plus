#!/usr/bin/env python3
"""Strictly publish a selected B0-v2 probe as a downstream checkpoint.

The published ``final.pth`` is the complete downstream B0-v2 model state
(O8 / Star ON / MCL OFF / MD200 ON / upper=3.75) in which every learned
topology module comes from the selected probe and everything else keeps the
deterministic downstream initialization.  Masked-atom head, coordinate
decoder and the fold-specific MD200 residual are
excluded from the transferred state.  A fresh downstream model must load the
assembled state with ``strict=True`` before the atomic write.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.modules.periodic_coordinate_decoder import PeriodicCoordinateDecoder
from src.training.common.checkpoint import save_final_state
from src.training.finetune.engine import build_mts_downstream_model
from src.training.pretrain.config import parse_arguments
from src.training.pretrain.engine import MIPSPretrainContainer, _build_b0_model

# Modules actually trained by the B0-v2 objective.  The MD200 residual is
# fold-seeded downstream, so it is not transferred.
_TRAINED_TOPOLOGY_PREFIXES = (
    "encoders.graph.encoder.atom_embedding.",
    "encoders.graph.encoder.spd_embedding.",
    "encoders.graph.encoder.path_bias.",
    "encoders.graph.encoder.layers.",
    "encoders.graph.encoder.star_distance_bias.",
)


def _finetune_args_for_b0(sidecar_root: str):
    import sys as _sys

    from src.training.finetune.config import parse_arguments as parse_finetune

    old = _sys.argv
    try:
        _sys.argv = ["train.py",
                     "--config_schema", "mts-config-v3",
                     "--graph_encoder_type", "mips_trimer_scage",
                     "--topology_attention_variant", "o8",
                     "--use_star_rbf", "--no-use_mcl", "--use_md200",
                     "--star_rbf_upper", "3.75",
                     "--star_rbf_v2_sidecar", sidecar_root]
        return parse_finetune()
    finally:
        _sys.argv = old


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--output", required=True)
    args_cli = parser.parse_args(argv)
    args = parse_arguments(["--b0_config", args_cli.config])
    probe_payload = torch.load(args_cli.probe, map_location="cpu", weights_only=False)
    if not isinstance(probe_payload, dict) or not isinstance(probe_payload.get("state_dict"), dict):
        raise RuntimeError("probe must contain a state_dict")
    state_dict = probe_payload["state_dict"]
    model = _build_b0_model(args)
    graph_encoder = model.encoders["graph"].encoder
    atom_head = nn.Linear(graph_encoder.emb_dim, int(graph_encoder.masked_atom_classes))
    coordinate_decoder = PeriodicCoordinateDecoder(int(graph_encoder.emb_dim))
    container = MIPSPretrainContainer(
        model, {"mips_atom": atom_head, "coordinate": coordinate_decoder}
    )
    container.load_state_dict(state_dict, strict=True)
    for value in state_dict.values():
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise RuntimeError("probe contains non-finite parameters")

    # The pretraining container stores the UniEncoder under ``model.``.
    pretrain_encoder = {
        str(key)[len("model."):]: value
        for key, value in state_dict.items()
        if str(key).startswith("model.")
    }
    if not pretrain_encoder:
        raise RuntimeError("probe has no model.* encoder state")

    downstream_args = _finetune_args_for_b0(str(args.star_rbf_v2_sidecar))
    downstream = build_mts_downstream_model(downstream_args)
    assembled = downstream.state_dict()
    transferred = {}
    for key, value in pretrain_encoder.items():
        if not any(key.startswith(prefix) for prefix in _TRAINED_TOPOLOGY_PREFIXES):
            continue
        if key not in assembled:
            raise RuntimeError(f"transferred key missing in downstream model: {key}")
        transferred[key] = value
    if not transferred:
        raise RuntimeError("no transferable topology keys found in probe")
    assembled.update(transferred)

    def strict_load(candidate):
        probe = downstream
        probe.load_state_dict(candidate, strict=True)
        for value in candidate.values():
            if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
                raise RuntimeError("published checkpoint contains non-finite parameters")
        return probe

    strict_load(assembled)
    save_final_state(assembled, args_cli.output, strict_load=strict_load)
    marker = Path(str(args_cli.output) + ".complete.json")
    print({
        "output": str(args_cli.output),
        "marker": str(marker),
        "source_probe": args_cli.probe,
        "step": probe_payload.get("step"),
        "transferred_tensors": len(transferred),
        "star_rbf_upper": 3.75,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
