#!/usr/bin/env python3
"""Two-update implementation smoke for the MTS-GLT-v3 joint and downstream paths."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_mts_glt_v3_sidecars import dataset_for_build  # noqa: E402
from src.dataset.dataloader import mips_trimer_collate  # noqa: E402
from src.dataset.periodic_line_glt_image import build_periodic_line_image_sample  # noqa: E402
from src.modules.mts_glt_v3 import MTSGraphLineModelV3  # noqa: E402
from src.training.pretrain.glt_v3_engine import MTSGLTV3PretrainContainer  # noqa: E402


def attach(data, row):
    data.mips_md = torch.linspace(0, 1, 200)
    data.mips_md_valid = True
    data.glt3_geometry_valid = bool(row["geometry_valid"])
    for name, value in row["tokens"].items():
        dtype = torch.float32 if name == "token_distance" else (torch.bool if name == "token_valid" else torch.long)
        setattr(data, "glt3_" + name, torch.as_tensor(value, dtype=dtype))
    for name, value in row["relations"].items():
        dtype = torch.float32 if name == "relation_angle" else (torch.bool if name == "relation_valid" else torch.long)
        setattr(data, "glt3_" + name, torch.as_tensor(value, dtype=dtype))
    return data


def real_batch(device):
    args = argparse.Namespace(cache_root="data", dataset="PI1M_v2")
    dataset = dataset_for_build(args)
    items = []
    for index in range(min(128, len(dataset))):
        data = dataset[index]
        row = build_periodic_line_image_sample(data, data, str(data.smiles))
        if row["geometry_valid"]:
            items.append(attach(data, row))
        if len(items) == 2:
            break
    if len(items) != 2:
        raise RuntimeError("smoke could not locate two valid frozen Trimers")
    return mips_trimer_collate(items).to(device)


def joint_smoke(batch):
    model = MTSGLTV3PretrainContainer(
        MTSGraphLineModelV3(glt_readout_mode="mips_concat")
    ).to(batch.mips_x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.98))
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        generator = torch.Generator(device=batch.mips_x.device).manual_seed(42 + step)
        output = model(batch, stream_step=step, generator=generator)
        loss = output["atom_sum"] / output["atom_count"] + output["line_loss"] + output["infonce"]
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite joint smoke loss")
        loss.backward()
        gradients = {}
        for name, prefix in (("o8", "model.o8."), ("md", "model.md_residual."), ("glt", "model.glt.")):
            gradients[name] = sum(float(p.grad.detach().abs().sum()) for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None)
            if gradients[name] <= 0:
                raise RuntimeError(f"joint smoke missing {name} gradient")
        optimizer.step()
        print(f"joint step={step + 1} loss={float(loss.detach()):.6f} gradients={gradients}", flush=True)


def downstream_smoke(batch, mode):
    encoder = MTSGraphLineModelV3(glt_readout_mode=mode).to(batch.mips_x.device)
    head = torch.nn.Linear(512, 1).to(batch.mips_x.device)
    optimizer = torch.optim.Adam([*encoder.parameters(), *head.parameters()], lr=1e-5)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = head(encoder(batch)).flatten()
        loss = prediction.square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite {mode} downstream loss")
        loss.backward(); optimizer.step()
        print(f"downstream mode={mode} step={step + 1} loss={float(loss.detach()):.6f}", flush=True)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(42)
    device = torch.device("cuda", 0)
    batch = real_batch(device)
    joint_smoke(batch)
    downstream_smoke(batch, "galformer")
    downstream_smoke(batch, "mips_concat")


if __name__ == "__main__":
    main()
