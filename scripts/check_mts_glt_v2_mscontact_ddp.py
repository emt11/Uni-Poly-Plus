#!/usr/bin/env python3
"""Three-rank, two-step synthetic DDP smoke for MSContact."""

from __future__ import annotations

from datetime import timedelta
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modules import GLTMaskedLineHeadV2, MTSGraphLineModelV2  # noqa: E402
from src.modules.periodic_line_glt_v2 import NUM_LINE_LABELS  # noqa: E402
from src.training.pretrain.glt_v2_engine import (  # noqa: E402
    MTSGLTV2PretrainContainer, _projection,
)


def fixture_batch():
    path = ROOT / "tests/test_mts_glt_v2_mscontact.py"
    spec = importlib.util.spec_from_file_location("mscontact_ddp_fixture", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module._batch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shell-mode", choices=("s4", "ms45", "c5_mixed"), default="ms45"
    )
    parser.add_argument(
        "--output", default="results/mts_glt_v2/mscontact_v1/ddp_smoke.json"
    )
    args_cli = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", timeout=timedelta(minutes=10),
        device_id=torch.device("cuda", local_rank),
    )
    try:
        if dist.get_world_size() != 3:
            raise RuntimeError("MSContact DDP smoke requires three ranks")
        torch.manual_seed(42)
        model = MTSGraphLineModelV2(
            glt_layers=1, use_spatial_contact=True,
            spatial_shell_mode=args_cli.shell_mode,
        )
        for module in (
            model.atom_fusion_norm, model.atom_fusion_projection,
            model.compact19_residual, model.o8.md_residual,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = False
        model.atom_channel_gate.requires_grad = False
        model.spatial_channel_gate.requires_grad = False
        container = MTSGLTV2PretrainContainer(
            model, torch.nn.Linear(512, int(model.o8.masked_atom_classes)),
            GLTMaskedLineHeadV2(512), _projection(256), _projection(256),
            torch.ones(NUM_LINE_LABELS, dtype=torch.long),
        ).cuda()
        ddp = torch.nn.parallel.DistributedDataParallel(
            container, device_ids=[local_rank], find_unused_parameters=False
        )
        optimizer = torch.optim.Adam(
            [p for p in container.parameters() if p.requires_grad], lr=2e-4
        )
        args = type("Args", (), {
            "seed": 42, "graph_mask_ratio": 0.3,
            "glt_line_mask_ratio": 0.4,
            "glt_infonce_temperature": 0.1,
        })()
        metrics = []
        for step in range(2):
            batch = fixture_batch().cuda()
            generator = torch.Generator(device="cuda").manual_seed(
                42 + step * dist.get_world_size() + dist.get_rank()
            )
            optimizer.zero_grad(set_to_none=True)
            output = ddp(batch, args, step, generator=generator)
            loss = output["atom_sum"] + output["line_sum"] + output["infonce"]
            if not torch.isfinite(loss):
                raise RuntimeError("MSContact DDP loss is non-finite")
            loss.backward()
            optimizer.step()
            metrics.append({
                "step": step + 1, "loss": float(loss.detach()),
                "infonce_pool_size": int(output["infonce_pool_size"]),
            })
        dist.barrier()
        if dist.get_rank() == 0:
            output = ROOT / args_cli.output
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({
                "world_size": 3, "steps": metrics, "pass": True,
            }, indent=2) + "\n")
            print(output)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
