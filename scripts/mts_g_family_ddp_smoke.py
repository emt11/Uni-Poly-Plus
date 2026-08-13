#!/usr/bin/env python3
"""Extremely short three-rank DDP readiness smoke for G0--G3."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from scripts.mts_g_family_readiness_smoke import _sample
from src.dataset.dataloader import mips_trimer_collate
from src.modules.mips_local_graph import MIPSLocalGraphEncoder


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="results/mts_multiscale_topology/g_family_readiness_v1/ddp_smoke.json")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("DDP smoke requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    results = {}
    try:
        for arm in ("g0", "g1", "g2", "g3"):
            torch.manual_seed(42)
            model = MIPSLocalGraphEncoder(
                topology_attention_variant="msta_last2",
                graph_geometry_mode=arm,
                g_family_arm=arm,
                use_star_rbf=False,
                use_mcl=False,
            ).to(device)
            wrapped = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
            batch = mips_trimer_collate([_sample(1 + rank, arm), _sample(2 + rank, arm)]).to(device)
            wrapped.train()
            output, _ = wrapped(batch)
            loss = output.float().square().mean()
            loss.backward()
            finite = torch.tensor([bool(torch.isfinite(loss))], device=device, dtype=torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            results[arm] = {"loss_finite_all_ranks": bool(finite.item()), "loss": float(loss.detach().cpu())}
            del wrapped, model, batch, output, loss
            torch.cuda.empty_cache()
        if rank == 0:
            report = {"schema": "mts-g-family-ddp-smoke-v1", "world_size": world_size, "arms": results, "devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
            path = Path(args.report)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
