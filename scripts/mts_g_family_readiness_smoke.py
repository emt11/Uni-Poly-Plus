#!/usr/bin/env python3
"""Two-sample CPU readiness smoke for the T1 G-family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.data import Data

from src.dataset.dataloader import mips_trimer_collate
from src.dataset.mips_trimer_contract import FEATURE_SCHEMA, TOPOLOGY_CANONICAL
from src.modules.mips_local_graph import MIPSLocalGraphEncoder


def _sample(seed, arm):
    generator = torch.Generator().manual_seed(int(seed))
    n, edges = 3, 4
    data = Data(
        x=torch.zeros((n, 1)),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        mips_x=torch.randn((n, 137), generator=generator),
        mips_backbone_mask=torch.zeros(n, dtype=torch.long),
        lga_edge_index=torch.tensor([[0, 1, 2, 0], [0, 1, 2, 2]]),
        lga_spd=torch.tensor([0, 0, 0, 2]),
        lga_path_index=torch.full((edges, 3), -1, dtype=torch.long),
        lga_path_mask=torch.zeros((edges, 3), dtype=torch.bool),
        lga_path_bond_hist=torch.zeros((edges, 2, 6)),
        lga_star_edge_mask=torch.zeros(edges, dtype=torch.bool),
        polymer_link_mask=torch.zeros(edges, dtype=torch.bool),
        lga_source_image_shift=torch.zeros(edges, dtype=torch.long),
        lga_path_shift=torch.zeros((edges, 3), dtype=torch.long),
        canonical_ru_atom_index=torch.arange(n),
        canonical_graph_index=torch.zeros(n, dtype=torch.long),
        canonical_to_trimer_base_atom_id=torch.arange(n),
        atomic_numbers=torch.tensor([6, 6, 6]),
        graph_available=torch.tensor(True),
        mips_boundary_distance=torch.tensor(6),
        mips_condition_valid=torch.tensor(True),
        feature_schema=FEATURE_SCHEMA,
        topology_representation=TOPOLOGY_CANONICAL,
        mips_local_lga_schema_version=2,
        mts_canonical_periodic=True,
        mips_md=torch.randn((200,), generator=generator),
        mips_md_valid=torch.tensor(True),
        y=torch.tensor([0.0]),
        smiles=f"C{seed}",
        mts_sample_hash64=torch.tensor(seed),
    )
    data.mts_relation_geometry_arm = arm
    data.mts_relation_geometry_sidecar_artifact = "a" * 64
    data.mts_relation_geometry_cohort_hash = "b" * 64
    if arm != "g0":
        data.mts_relation_geometry_relation_row = torch.tensor([3])
        data.mts_relation_geometry_valid = torch.tensor([True])
        data.mts_relation_geometry_reason_code = torch.tensor([0], dtype=torch.int16)
        data.mts_relation_geometry_path_offsets = torch.tensor([0, 1])
        data.mts_relation_geometry_path_valid = torch.tensor([True])
        data.mts_relation_geometry_path_cos_angle = torch.tensor([0.2])
        data.mts_relation_geometry_endpoint_distance = torch.tensor([2.0])
    return data


def _finite_gradients(model):
    values = [p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]
    return bool(values) and bool(torch.isfinite(torch.cat(values)).all())


def run():
    outputs = {}
    shared = None
    for arm in ("g0", "g1", "g2", "g3"):
        torch.manual_seed(42)
        model = MIPSLocalGraphEncoder(
            topology_attention_variant="msta_last2",
            graph_geometry_mode=arm,
            g_family_arm=arm,
            use_star_rbf=False,
            use_mcl=False,
        ).eval()
        batch = mips_trimer_collate([_sample(1, arm), _sample(2, arm)])
        value, _ = model(batch)
        if shared is None:
            shared = value.detach()
        outputs[arm] = {
            "shape": list(value.shape),
            "finite": bool(torch.isfinite(value).all()),
            "step0_max_abs_delta": float((value.detach() - shared).abs().max()),
        }
        model.train()
        value, _ = model(batch)
        loss = value.float().square().mean()
        loss.backward()
        geometry_grads = [
            p.grad for p in model.relation_geometry_bias.parameters()
            if p.grad is not None
        ]
        outputs[arm].update({
            "loss_finite": bool(torch.isfinite(loss)),
            "gradients_finite": _finite_gradients(model),
            "geometry_gradients_present": bool(geometry_grads),
            "geometry_gradients_finite": bool(geometry_grads) and bool(torch.isfinite(torch.cat([g.reshape(-1) for g in geometry_grads])).all()),
            "g0_geometry_gradients_absent": arm == "g0" and not geometry_grads,
        })
    if any(item["step0_max_abs_delta"] > 1e-6 for item in outputs.values()):
        raise RuntimeError("G-family shared step-0 forward parity failed")
    if not all(item["finite"] and item["loss_finite"] and item["gradients_finite"] for item in outputs.values()):
        raise RuntimeError("G-family finite forward/backward gate failed")
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="results/mts_multiscale_topology/g_family_readiness_v1/direct_smoke.json")
    args = parser.parse_args(argv)
    report = {"schema": "mts-g-family-readiness-smoke-v1", "outputs": run()}
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
