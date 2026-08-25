#!/usr/bin/env python3
"""Three-rank two-step MTS-GLT-v1 smoke with unequal valid-3D counts."""

from __future__ import annotations

from datetime import timedelta
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.canonical_periodic import build_canonical_periodic_topology  # noqa: E402
from src.dataset.dataloader import mips_trimer_collate  # noqa: E402
from src.dataset.lmdb_cache import sample_key_from_smiles  # noqa: E402
from src.dataset.periodic_line_glt import build_periodic_line_sample  # noqa: E402
from src.dataset.trimer_mcl import attach_finite_trimer_mcl  # noqa: E402
from src.modules import GLTMaskedLineHead, MTSGraphLineModel  # noqa: E402
from src.training.pretrain.engine import _b0_differentiable_mean  # noqa: E402
from src.training.pretrain.glt_engine import MTSGLTPretrainContainer  # noqa: E402


def _attach(data, record):
    data.glt_geometry_valid = bool(record["geometry_valid"])
    tokens, relations = record["tokens"], record["relations"]
    for name, field in {
        "token_atom_a": "atom_a", "token_atom_b": "atom_b", "token_shift": "shift",
        "token_endpoint_z_a": "z_a", "token_endpoint_z_b": "z_b",
        "token_label": "label", "token_observation_count": "observation_count",
    }.items():
        setattr(data, f"glt_{name}", torch.tensor([item[field] for item in tokens]))
    data.glt_token_valid = torch.tensor([item["valid"] for item in tokens], dtype=torch.bool)
    data.glt_token_observation_distances = torch.tensor(
        [item["distances"] for item in tokens], dtype=torch.float32
    )
    for name, field in {
        "relation_source": "source", "relation_target": "target",
        "relation_center_atom": "center_atom", "relation_multiplicity": "multiplicity",
        "relation_observation_count": "observation_count",
    }.items():
        setattr(data, f"glt_{name}", torch.tensor([item[field] for item in relations]))
    data.glt_relation_valid = torch.tensor([item["valid"] for item in relations], dtype=torch.bool)
    data.glt_relation_is_fallback = torch.tensor([item["fallback"] for item in relations], dtype=torch.bool)
    data.glt_relation_observation_angles = torch.tensor(
        [item["angles"] for item in relations], dtype=torch.float32
    )
    data.mips_md = torch.zeros(200)
    data.mips_md_valid = torch.tensor(False)
    return data


def _sample(smiles):
    data = build_canonical_periodic_topology(smiles)
    attach_finite_trimer_mcl(data, smiles, num_candidates=1)
    return _attach(
        data,
        build_periodic_line_sample(sample_key_from_smiles(smiles), data, data),
    )


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", timeout=timedelta(minutes=10),
        device_id=torch.device("cuda", local_rank),
    )
    rank = dist.get_rank()
    try:
        device = torch.device("cuda", local_rank)
        torch.manual_seed(42)
        samples = [_sample("*CCO*"), _sample("*COC*")]
        if rank > 0:
            samples[1].glt_geometry_valid = False
            samples[1].glt_token_valid.zero_()
            samples[1].glt_relation_valid.zero_()
        batch = mips_trimer_collate(samples).to(device)
        model = MTSGraphLineModel()
        model.glt_fusion_norm.requires_grad_(False)
        model.glt_fusion_projection.requires_grad_(False)
        model.glt_gate.requires_grad = False
        model.o8.md_residual.requires_grad_(False)
        container = MTSGLTPretrainContainer(
            model,
            nn.Linear(512, model.o8.masked_atom_classes),
            GLTMaskedLineHead(512),
            nn.Linear(512, 64),
            nn.Linear(512, 64),
        ).to(device)
        ddp = torch.nn.parallel.DistributedDataParallel(
            container, device_ids=[local_rank], find_unused_parameters=False
        )
        optimizer = torch.optim.Adam(
            [parameter for parameter in container.parameters() if parameter.requires_grad],
            lr=2e-4,
        )
        args = SimpleNamespace(
            seed=42, graph_mask_ratio=0.30, glt_line_mask_ratio=0.40,
            glt_infonce_temperature=0.10,
        )
        pools = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            generator = torch.Generator(device=device).manual_seed(42 + step * 3 + rank)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = ddp(batch, args, step, generator=generator)
                atom, _, _ = _b0_differentiable_mean(
                    output["atom_sum"], output["atom_count"], device
                )
                line, _, _ = _b0_differentiable_mean(
                    output["line_sum"], output["line_count"], device
                )
                loss = atom + line + output["infonce"]
            loss.backward()
            optimizer.step()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite GLT smoke loss")
            pools.append(int(output["infonce_pool_size"]))
        if pools != [4, 4]:
            raise RuntimeError(f"unexpected gathered InfoNCE pools: {pools}")
        if rank == 0:
            path = ROOT / "results/mts_glt_v1/ddp_smoke/synthetic_state.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(container.state_dict(), path)
            clone = MTSGLTPretrainContainer(
                MTSGraphLineModel(),
                nn.Linear(512, model.o8.masked_atom_classes),
                GLTMaskedLineHead(512),
                nn.Linear(512, 64),
                nn.Linear(512, 64),
            )
            clone.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            print("MTS-GLT-v1 3-rank smoke passed; pools=4,4", flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
