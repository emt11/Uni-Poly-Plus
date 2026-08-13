#!/usr/bin/env python
"""Finite T1 readiness benchmark and one-task fine-tuning smoke.

The outputs are screening evidence only.  The script uses real canonical
topology construction and a small subset of the ``eat`` fold; it never writes
``best_result.csv`` or any formal checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.dataloader import mips_trimer_collate
from src.modules.mips_local_graph import (
    MIPSLocalGraphEncoder,
    add_function_preserving_t1_parameters,
)


ROOT = Path(__file__).resolve().parents[1]


def _batch(smiles):
    return mips_trimer_collate([
        build_canonical_periodic_topology(str(value)) for value in smiles
    ])


def _models(device):
    torch.manual_seed(20260811)
    t0 = MIPSLocalGraphEncoder().to(device)
    t1 = MIPSLocalGraphEncoder(topology_attention_variant="msta_last2").to(device)
    t1.load_state_dict(add_function_preserving_t1_parameters(t0.state_dict()), strict=True)
    return t0, t1


def _step(model, batch, device):
    batch = batch.to(device)
    model.zero_grad(set_to_none=True)
    graph, _ = model._forward_impl(
        batch, use_star=False, use_geometry=False, use_md=False
    )
    loss = graph.square().mean()
    loss.backward()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite T1 smoke loss")
    finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    if not finite:
        raise FloatingPointError("non-finite T1 smoke gradient")
    return float(loss.detach().cpu())


def run_benchmark(output: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("T1 GPU benchmark requires CUDA")
    device = torch.device("cuda:0")
    batch = _batch(["*CCO*", "*c1ccccc1*"]).to(device)
    results = {}
    for name, model in zip(("T0", "T1"), _models(device)):
        model.train()
        for _ in range(2):
            _step(model, batch, device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        losses = []
        for _ in range(5):
            losses.append(_step(model, batch, device))
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        results[name] = {
            "samples": 2,
            "iterations": 5,
            "wall_seconds": elapsed,
            "samples_per_second": 10.0 / elapsed,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "loss_finite": all(torch.isfinite(torch.tensor(losses))),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "mts-t1-benchmark-v1",
        "screening_only": True,
        "device": torch.cuda.get_device_name(device),
        "results": results,
        "batch": {"samples": 2, "smiles": ["*CCO*", "*c1ccccc1*"]},
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _fold_train_indices():
    split = json.loads(
        (ROOT / "data/splits/mips_shared5/eat.json").read_text(encoding="utf-8")
    )
    fold = next(item for item in split["folds"] if int(item["fold"]) == 0)
    return [int(value) for value in fold["train_indices"]]


def run_finetune_smoke(output: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("T1 fine-tuning smoke requires CUDA")
    frame = pd.read_csv(ROOT / "data/raw/smi_eat.csv")
    indices = _fold_train_indices()[:4]
    rows = frame.iloc[indices]
    batch = _batch(rows["smiles"].tolist())
    labels = torch.tensor(rows["Eat"].to_numpy(), dtype=torch.float32)
    device = torch.device("cuda:0")
    model = MIPSLocalGraphEncoder(topology_attention_variant="msta_last2").to(device)
    head = nn.Linear(model.emb_dim, 1).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()), lr=1e-4
    )
    batch = batch.to(device)
    labels = labels.to(device)
    losses = []
    start = time.perf_counter()
    for epoch in range(2):
        model.train()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        graph, _ = model._forward_impl(
            batch, use_star=False, use_geometry=False, use_md=False
        )
        prediction = head(graph).reshape(-1)
        loss = torch.nn.functional.smooth_l1_loss(prediction, labels)
        loss.backward()
        if not torch.isfinite(loss) or not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in list(model.parameters()) + list(head.parameters())
        ):
            raise FloatingPointError("non-finite T1 fine-tuning smoke state")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    torch.cuda.synchronize(device)
    payload = {
        "schema": "mts-t1-finetune-smoke-v1",
        "screening_only": True,
        "task": "eat",
        "fold": 0,
        "epochs": 2,
        "samples": len(indices),
        "indices": indices,
        "wall_seconds": time.perf_counter() - start,
        "losses": losses,
        "finite": True,
        "checkpoint_written": False,
        "best_result_updated": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["benchmark", "finetune-smoke"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "benchmark":
        run_benchmark(args.output)
    else:
        run_finetune_smoke(args.output)


if __name__ == "__main__":
    main()
