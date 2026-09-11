#!/usr/bin/env python3
"""Read-only validation of the complete-Trimer GLT contract.

The command reads existing frozen feature-cache records, builds complete-Trimer
rows in memory, collates at most ``--limit`` rows, and checks a CPU
forward/backward pass.  It never writes a sidecar or generates coordinates.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_mts_glt_v3_sidecars import dataset_for_build  # noqa: E402
from src.dataset.dataloader import mips_trimer_collate  # noqa: E402
from src.dataset.periodic_line_glt_complete import (  # noqa: E402
    build_complete_trimer_glt_sample,
)
from src.modules.periodic_line_glt_v3 import CompleteTrimerGLTEncoder  # noqa: E402


def _attach(data, row):
    data = copy.copy(data)
    data.glt3_geometry_valid = bool(row["geometry_valid"])
    for name, value in row["tokens"].items():
        dtype = (
            torch.float32
            if name == "token_distance" or name == "token_bond_features"
            else torch.bool
            if name in {"token_valid", "token_center_internal"}
            else torch.long
        )
        setattr(data, "glt3_" + name, torch.as_tensor(value.copy(), dtype=dtype))
    for name, value in row["relations"].items():
        dtype = (
            torch.float32
            if name == "relation_angle"
            else torch.bool
            if name == "relation_valid"
            else torch.long
        )
        setattr(data, "glt3_" + name, torch.as_tensor(value.copy(), dtype=dtype))
    return data


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default="data")
    parser.add_argument("--dataset", default="PI1M_v2")
    parser.add_argument("--indices", type=int, nargs="+", default=[1, 10])
    args = parser.parse_args(argv)
    dataset = dataset_for_build(args)
    rows = []
    reports = []
    for index in args.indices:
        data = dataset[int(index)]
        row = build_complete_trimer_glt_sample(data, data, str(data.smiles))
        if not row["geometry_valid"]:
            raise RuntimeError(f"index {index} has invalid complete geometry: {row['invalid_reason']}")
        rows.append(_attach(data, row))
        reports.append({
            "index": int(index),
            "tokens": int(len(row["tokens"]["token_atom_a"])),
            "center_tokens": int(row["tokens"]["token_center_internal"].sum()),
            "relations": int(len(row["relations"]["relation_source"])),
            "bond_features": tuple(row["tokens"]["token_bond_features"].shape),
        })
    batch = mips_trimer_collate(rows)
    model = CompleteTrimerGLTEncoder(dropout=0.0)
    model.train()
    output = model(batch)
    loss = output["graph_geometry"].square().mean()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("complete-Trimer GLT forward produced NaN/Inf")
    loss.backward()
    gradient_names = (
        "endpoint_projection.weight",
        "bond_feature_projection.weight",
        "distance_projection.weight",
        "angle_projection.weight",
        "layers.0.attention.qkv.weight",
    )
    gradients = {}
    for name in gradient_names:
        parameter = model.get_parameter(name)
        gradients[name] = (
            float(parameter.grad.abs().sum()) if parameter.grad is not None else 0.0
        )
    if not all(value > 0.0 and torch.isfinite(torch.tensor(value)) for value in gradients.values()):
        raise RuntimeError(f"missing complete-Trimer GLT gradients: {gradients}")
    print({
        "status": "pass",
        "samples": reports,
        "batch_tokens": int(batch.glt3_token_atom_a.numel()),
        "batch_relations": int(batch.glt3_relation_source.numel()),
        "graph_output": tuple(output["graph_geometry"].shape),
        "loss": float(loss.detach()),
        "gradients": gradients,
        "writes": False,
    })


if __name__ == "__main__":
    main()
