"""Sym-MCL v1 real-batch regression: old vs new mask, forward/backward.

Run: PYTHONPATH=. python scripts/sym_mcl_regression.py
"""
import sys

import torch

from src.dataset.dataloader import custom_collate
from src.dataset.dataset import _compute_smiles_features_from_config
from src.dataset.mts_star_rbf_v2 import build_star_rbf_v2_sample
from src.modules.mips_local_graph import MIPSLocalGraphEncoder
from src.modules.trimer_mcl import build_sym_mcl_mask


def graph_data(smiles="*CCCCCCC*"):
    data = _compute_smiles_features_from_config(
        smiles,
        "./pretrained_models/encoders/PubChem10M_SMILES_BPE_450k",
        32, "star_linking", "repeat_unit", "disabled",
        graph_encoder_type="scage",
        mips_core="paper_corrected",
        mips_max_hops=2,
        mips_use_descriptors=True,
        spatial_mode="trimer_scage",
        graph_geometry_mode="trimer_scage_mcl",
        trimer_num_candidates=4,
        trimer_max_heavy_atoms=384,
    )
    record = build_star_rbf_v2_sample(b"k" * 32, data, data)
    data.mts_star_v2_relation_row = torch.tensor(
        [item["row"] for item in record["relations"]], dtype=torch.long
    )
    data.mts_star_v2_relation_pair_index = torch.tensor(
        [item["pair_index"] for item in record["relations"]], dtype=torch.long
    )
    data.mts_star_v2_relation_spd = torch.tensor(
        [item["spd"] for item in record["relations"]], dtype=torch.long
    )
    data.mts_star_v2_pair_observation_distances = torch.tensor(
        [item["distances"] for item in record["pairs"]], dtype=torch.float
    )
    data.mts_star_v2_pair_observation_count = torch.tensor(
        [item["observation_count"] for item in record["pairs"]], dtype=torch.long
    )
    data.mts_star_v2_pair_valid = torch.tensor(
        [item["valid"] for item in record["pairs"]], dtype=torch.bool
    )
    data.mts_star_v2_pair_geometry_source = torch.tensor(
        [item["geometry_source"] for item in record["pairs"]], dtype=torch.long
    )
    data.mts_star_v2_sidecar_artifact = "a" * 64
    data.mts_star_v2_model_semantic_hash = "b" * 64
    data.mts_star_v2_rbf_upper = 6.0
    data.y = torch.zeros(1)
    return data


def main():
    samples = [graph_data("*CCCCCCC*"), graph_data("*CCO*")]
    batch = custom_collate(samples)
    encoder = MIPSLocalGraphEncoder().eval()
    encoder.trimer_mcl.geometry_gate.data.fill_(0.2)
    mcl_params = sum(p.numel() for p in encoder.trimer_mcl.parameters())
    print(f"MCL parameter count: {mcl_params}")

    trimer_batch = batch.trimer_batch.long()
    valid = []
    for graph_id in range(2):
        valid.append(
            bool(batch.trimer_geometry_valid[graph_id])
            and bool(batch.trimer_geometry_is_3d[graph_id])
            and not bool(batch.trimer_2d_fallback[graph_id])
        )
    print(f"graphs mcl-eligible: {valid}")

    for graph_id in range(2):
        if not valid[graph_id]:
            continue
        atoms = torch.nonzero(trimer_batch == graph_id, as_tuple=False).flatten()
        central_mask = batch.trimer_central_ru_mask[atoms].bool()
        central = atoms[central_mask]
        c = int(central.numel())
        assert atoms.numel() == 3 * c, (atoms.numel(), c)
        positions = batch.trimer_pos[atoms].float()
        d = torch.cdist(positions[central_mask], positions)  # [C, 3C]
        thresholds = batch.trimer_mcl_thresholds[graph_id]
        for scale_name, q in (("q20", float(thresholds[0])), ("q50", float(thresholds[1]))):
            old = (d <= q)
            new = build_sym_mcl_mask(
                d.unsqueeze(0), q, torch.tensor([c]), None
            )[0]
            assert new.shape == old.shape, (new.shape, old.shape)
            assert not bool(torch.isnan(d).any())
            assert bool((new >= old).all())
            old_ratio = float(old.sum()) / float(old.numel())
            new_ratio = float(new.sum()) / float(new.numel())
            m_plus = new[:, 2 * c : 3 * c]
            m_minus = new[:, 0:c]
            invariant = torch.equal(m_plus, m_minus.transpose(-1, -2))
            print(
                f"graph {graph_id} C={c} {scale_name}: old={old_ratio:.4f} "
                f"new={new_ratio:.4f} invariant={invariant}"
            )

    with torch.no_grad():
        graph_out, nodes = encoder(batch)
    print(f"encoder output finite: {bool(torch.isfinite(graph_out).all())}, "
          f"shape={tuple(graph_out.shape)}")
    assert bool(torch.isfinite(graph_out).all())

    encoder.train()
    graph_out, _ = encoder(batch)
    loss = graph_out.square().mean()
    loss.backward()
    grads = [
        (name, param.grad)
        for name, param in encoder.trimer_mcl.named_parameters()
        if param.requires_grad
    ]
    all_finite = all(
        grad is not None and bool(torch.isfinite(grad).all()) for _, grad in grads
    )
    any_nonzero = any(
        grad is not None and bool((grad.abs().max() > 0).item()) for _, grad in grads
    )
    print(f"backward: loss={float(loss.item()):.6f} finite={all_finite} "
          f"nonzero_grad={any_nonzero}")
    assert all_finite and any_nonzero
    print("SYM-MCL REGRESSION OK")


if __name__ == "__main__":
    sys.exit(main())
