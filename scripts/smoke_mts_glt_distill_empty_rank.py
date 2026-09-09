#!/usr/bin/env python3
"""Three-rank collective smoke for empty distillation candidate sets."""

import os
from pathlib import Path
import sys
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.pretrain.glt_distill_engine import multi_positive_infonce


def run_case(local_count):
    rank = dist.get_rank()
    value = torch.randn(local_count, 256, device="cuda", requires_grad=True)
    teacher = torch.randn(local_count, 256, device="cuda")
    identities = torch.arange(local_count, device="cuda", dtype=torch.long) + rank * 100
    loss, pool = multi_positive_infonce(value, teacher, identities)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite empty-rank InfoNCE")
    loss.backward()
    if value.grad is None or not torch.isfinite(value.grad).all():
        raise RuntimeError("empty-rank InfoNCE backward failed")
    return pool


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    try:
        partial = run_case(0 if dist.get_rank() == 0 else 2)
        empty = run_case(0)
        result = torch.tensor([partial, empty], device="cuda")
        gathered = [torch.empty_like(result) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, result)
        if dist.get_rank() == 0:
            print("EMPTY_RANK_COLLECTIVE_PASS", [row.tolist() for row in gathered], flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
