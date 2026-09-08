#!/usr/bin/env python3
"""Three-rank parity check for v3 global InfoNCE and target reductions."""

from __future__ import annotations

from datetime import timedelta
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.pretrain.engine import _differentiable_mean
from src.training.pretrain.glt_v3_objectives import distributed_bidirectional_infonce


def reference_infonce(first, second, valid, temperature=0.1):
    first = F.normalize(first[valid].float(), dim=-1)
    second = F.normalize(second[valid].float(), dim=-1)
    logits = first @ second.T / temperature
    labels = torch.arange(len(logits), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=5), device_id=torch.device("cuda", local_rank))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != 3:
            raise RuntimeError("parity check requires three ranks")
        device = torch.device("cuda", local_rank)
        generator = torch.Generator().manual_seed(42)
        all_o8 = torch.randn(12, 256, generator=generator).to(device)
        all_glt = torch.randn(12, 256, generator=generator).to(device)
        all_valid = torch.tensor([1, 1, 0, 1] * 3, dtype=torch.bool, device=device)
        sl = slice(rank * 4, (rank + 1) * 4)
        actual, pool = distributed_bidirectional_infonce(all_o8[sl], all_glt[sl], all_valid[sl], 0.1)
        expected = reference_infonce(all_o8, all_glt, all_valid, 0.1)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        local_count = rank + 2
        local_sum = torch.tensor(float((rank + 1) * local_count), device=device, requires_grad=True)
        scaled, global_count, global_sum = _differentiable_mean(local_sum, local_count, device)
        averaged = scaled.detach().clone()
        dist.all_reduce(averaged)
        averaged /= world
        expected_mean = sum((r + 1) * (r + 2) for r in range(world)) / sum(r + 2 for r in range(world))
        torch.testing.assert_close(
            averaged,
            torch.tensor(expected_mean, device=device, dtype=averaged.dtype),
            atol=1e-6, rtol=0,
        )
        assert global_count == 9 and global_sum == 20.0 and pool == 9
        if rank == 0:
            print(f"PASS infonce={float(actual):.8f} targets_mean={float(averaged):.8f} pool={pool}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
