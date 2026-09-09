#!/usr/bin/env python3
"""Three-rank real-data smoke with one rank containing only N=0 samples."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.dataloader import mips_trimer_collate  # noqa: E402
from src.modules.mts_glt_distill import NPlusGLTTeacher  # noqa: E402
from src.training.pretrain.glt_distill_engine import (  # noqa: E402
    StudentContainer, _dataset,
)


N_ZERO_SAMPLE_INDEX = 10
ORDINARY_SAMPLE_INDEX = 1
N_ZERO_SAMPLE_KEY_HEX = "2320ad8ec663f0ca98eb48191b15e8b28b0f1c401a5142be68a286e990fbcfd1"


def _global_count(value: int, device: torch.device) -> int:
    count = torch.tensor(int(value), dtype=torch.long, device=device)
    dist.all_reduce(count)
    return int(count)


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    try:
        if dist.get_world_size() != 3:
            raise RuntimeError("N=0 rank smoke requires exactly three ranks")
        config_path = ROOT / "configs/mts/glt_distill_repair_c1.json"
        config = json.loads(config_path.read_text())
        config["config_path"] = str(config_path)
        dataset = _dataset(config)
        rank = dist.get_rank()
        index = N_ZERO_SAMPLE_INDEX if rank == 0 else ORDINARY_SAMPLE_INDEX
        data = mips_trimer_collate([dataset[index]]).to(device)

        expected_hash = (
            int.from_bytes(bytes.fromhex(N_ZERO_SAMPLE_KEY_HEX)[:8], "little")
            & ((1 << 63) - 1)
        )
        center_count = int(
            (data.glt3_token_center_internal.bool() & data.glt3_token_valid.bool()).sum()
        )
        if rank == 0:
            if int(data.mts_sample_hash64[0]) != expected_hash:
                raise RuntimeError("rank-0 N=0 sample identity mismatch")
            if center_count != 0:
                raise RuntimeError("rank-0 sample is not N=0")
        elif center_count <= 0:
            raise RuntimeError("ordinary control rank has no center target")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        container = StudentContainer(NPlusGLTTeacher()).to(device)
        module = torch.nn.parallel.DistributedDataParallel(
            container, device_ids=[local_rank], find_unused_parameters=False,
        )
        generator = torch.Generator(device=device).manual_seed(9000 + rank)
        output = module(data, 2000, generator=generator)

        if output["atom_count"] <= 0 or not torch.isfinite(output["atom_sum"]):
            raise RuntimeError("student atom task is absent or non-finite")
        if rank == 0 and (
            output["local_count"] != 0
            or output["valid_graphs"] != 0
            or output["global_pool"] != 2
        ):
            raise RuntimeError("N=0 rank leaked into distillation denominators")
        if not all(torch.isfinite(output[name]) for name in ("local_sum", "global")):
            raise RuntimeError("N=0 rank produced non-finite distillation loss")

        total_atom = _global_count(output["atom_count"], device)
        total_local = _global_count(output["local_count"], device)
        total_valid = _global_count(output["valid_graphs"], device)
        loss = dist.get_world_size() * (
            output["atom_sum"] / max(1, total_atom)
            + 0.1 * output["local_sum"] / max(1, total_local)
        ) + 0.1 * (output["global_pool"] / max(1, total_valid)) * output["global"]
        if not torch.isfinite(loss):
            raise RuntimeError("combined N=0 rank loss is non-finite")
        loss.backward()
        atom_grads = [
            parameter.grad for parameter in container.atom_head.parameters()
            if parameter.requires_grad
        ]
        if not atom_grads or any(
            gradient is None or not bool(torch.isfinite(gradient).all())
            for gradient in atom_grads
        ):
            raise RuntimeError("student atom-task backward failed")

        observed = torch.tensor(
            [center_count, output["atom_count"], output["local_count"], output["valid_graphs"]],
            dtype=torch.long, device=device,
        )
        gathered = [torch.empty_like(observed) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, observed)
        if rank == 0:
            print(
                "N_ZERO_RANK_DDP_PASS "
                f"sample_key={N_ZERO_SAMPLE_KEY_HEX} "
                f"counts={[value.cpu().tolist() for value in gathered]}",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
