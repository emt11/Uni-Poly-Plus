#!/usr/bin/env python3
"""Three-rank proof for uneven valid-target global sum/count normalization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from src.modules.glt_dual_pretrain import global_objective
from src.training.glt_dual_runtime import require_tmux, write_json


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, values):
        return self.scale * values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args()
    require_tmux()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != 3 or not torch.cuda.is_available():
        raise RuntimeError("this validation requires exactly three CUDA ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    # Each row is [chemistry, geometry, fingerprint].  Zeros model a rank
    # without a valid target for that objective.
    local_counts_table = (
        (2.0, 0.0, 1.0),
        (0.0, 3.0, 1.0),
        (1.0, 1.0, 0.0),
    )
    local_constant_sums = (
        (4.0, 0.0, 2.0),
        (0.0, 9.0, 4.0),
        (5.0, 7.0, 0.0),
    )
    counts = torch.tensor(local_counts_table[rank], device=device)
    global_counts = counts.clone()
    dist.all_reduce(global_counts)
    constants = torch.tensor(local_constant_sums[rank], device=device)
    probe = Probe().to(device)
    model = DistributedDataParallel(probe, device_ids=[local_rank])
    sums = model(constants)
    weights = (1.0, 1.0, 0.1)
    loss = global_objective(sums, global_counts, world, weights)
    loss.backward()
    global_sums = constants.clone()
    dist.all_reduce(global_sums)
    expected = (global_sums * torch.tensor(weights, device=device)
                / global_counts).sum()
    observed = probe.scale.grad
    if not torch.isfinite(observed) or not torch.allclose(observed, expected):
        raise RuntimeError(f"DDP global sum/count gradient mismatch: {observed} != {expected}")
    rows = [None] * world
    dist.all_gather_object(rows, {
        "rank": rank,
        "local_counts": list(local_counts_table[rank]),
        "local_sums": list(local_constant_sums[rank]),
        "observed_gradient": float(observed),
    })
    if rank == 0:
        write_json(args.report_json, {
            "status": "PASS", "world_size": world,
            "global_counts": global_counts.tolist(),
            "global_sums": global_sums.tolist(),
            "expected_global_mean_gradient": float(expected),
            "ranks": rows,
        })
        print(json.dumps(rows, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
