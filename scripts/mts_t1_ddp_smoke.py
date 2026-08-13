#!/usr/bin/env python
"""Three-rank finite DDP smoke for the T1 MSTA topology path."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.dataloader import mips_trimer_collate
from src.modules.mips_local_graph import MIPSLocalGraphEncoder


ROOT = Path(__file__).resolve().parents[1]


class _RelationPretextWrapper(nn.Module):
    """Expose the relation-pretext path through a normal DDP forward call."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, data):
        graph, _ = self.encoder.forward_relation_pretext(data)
        return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("DDP smoke requires CUDA")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 3:
        raise RuntimeError(f"T1 DDP smoke requires world size 3, got {world}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    rank_smiles = ["*CCO*", "*c1ccccc1*", "*CC(=O)O*"][rank]
    data = mips_trimer_collate([
        build_canonical_periodic_topology(rank_smiles),
    ]).to(device)
    torch.manual_seed(20260811)
    torch.cuda.manual_seed_all(20260811)
    encoder = MIPSLocalGraphEncoder(topology_attention_variant="msta_last2")
    model = _RelationPretextWrapper(encoder).to(device)
    ddp = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(ddp.parameters(), lr=1e-4)
    selected_name = "module.encoder.layers.5.attention.local_output.weight"
    selected_before = dict(ddp.named_parameters())[selected_name].detach().clone()
    start = time.perf_counter()
    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        graph = ddp(data)
        loss = graph.square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite DDP loss")
        loss.backward()
        local_grad = dict(ddp.named_parameters())[selected_name].grad
        if local_grad is None or not torch.isfinite(local_grad).all():
            raise FloatingPointError("non-finite or missing MSTA local gradient")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    elapsed = time.perf_counter() - start
    selected = dict(ddp.named_parameters())[selected_name].detach().clone()
    if not torch.isfinite(selected).all():
        raise FloatingPointError("non-finite synchronized T1 parameter")
    if torch.count_nonzero(selected - selected_before).item() == 0:
        raise FloatingPointError("selected T1 parameter did not update")
    gathered_params = [torch.empty_like(selected) for _ in range(world)]
    dist.all_gather(gathered_params, selected)
    max_parameter_delta = max(
        float((candidate - gathered_params[0]).abs().max().item())
        for candidate in gathered_params
    )
    input_smiles = [None for _ in range(world)] if rank == 0 else None
    dist.gather_object(rank_smiles, input_smiles, dst=0)
    local_payload = {
        "rank": rank,
        "input_smiles": rank_smiles,
        "losses": losses,
        "wall_seconds": elapsed,
        "finite": True,
        "local_grad_nonzero": bool(
            torch.count_nonzero(dict(ddp.named_parameters())[selected_name].grad).item()
            > 0
        ),
        "parameter_checksum": float(selected.float().sum().item()),
        "max_parameter_delta_across_ranks": max_parameter_delta,
    }
    gathered = [None for _ in range(world)] if rank == 0 else None
    dist.gather_object(local_payload, gathered, dst=0)
    if rank == 0:
        payload = {
            "schema": "mts-t1-ddp-smoke-v2",
            "screening_only": True,
            "world_size": world,
            "ranks": gathered,
            "finite": all(item["finite"] for item in gathered),
            "local_grad_nonzero": all(
                item["local_grad_nonzero"] for item in gathered
            ),
            "ddp_forward_via_wrapper": True,
            "rank_inputs": input_smiles,
            "rank_inputs_distinct": len(set(input_smiles)) == world,
            "parameters_synchronized": max_parameter_delta <= 1e-7,
            "max_parameter_delta_across_ranks": max_parameter_delta,
            "checkpoint_written": False,
        }
        if not payload["rank_inputs_distinct"]:
            raise RuntimeError("DDP smoke rank inputs are not distinct")
        if not payload["parameters_synchronized"]:
            raise RuntimeError(
                "DDP smoke parameters diverged across ranks: "
                f"max_delta={max_parameter_delta}"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(payload, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
