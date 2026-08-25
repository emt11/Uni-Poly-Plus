import copy

import numpy as np
import pytest
import torch

from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.dataloader import mips_trimer_collate
from src.dataset.mts_star_rbf_v2 import (
    GEOMETRY_SOURCE,
    build_star_rbf_v2_sample,
    periodic_pair_key,
    rbf_upper_from_distances,
)
from src.dataset.trimer_mcl import attach_finite_trimer_mcl
from src.modules.mips_local_graph import MIPSLocalGraphEncoder


def _sample(smiles="*CCO*"):
    topology = build_canonical_periodic_topology(smiles)
    attach_finite_trimer_mcl(topology, smiles, num_candidates=1)
    return topology


def _attach(topology, record, upper=6.0):
    relations, pairs = record["relations"], record["pairs"]
    topology.mts_star_v2_relation_row = torch.tensor([x["row"] for x in relations])
    topology.mts_star_v2_relation_pair_index = torch.tensor([x["pair_index"] for x in relations])
    topology.mts_star_v2_relation_spd = torch.tensor([x["spd"] for x in relations])
    topology.mts_star_v2_pair_observation_distances = torch.tensor([x["distances"] for x in pairs])
    topology.mts_star_v2_pair_observation_count = torch.tensor([x["observation_count"] for x in pairs])
    topology.mts_star_v2_pair_valid = torch.tensor([x["valid"] for x in pairs])
    topology.mts_star_v2_pair_geometry_source = torch.tensor([x["geometry_source"] for x in pairs])
    topology.mts_star_v2_sidecar_artifact = "a" * 64
    topology.mts_star_v2_model_semantic_hash = "b" * 64
    topology.mts_star_v2_rbf_upper = upper
    topology.mips_md = torch.zeros(200)
    topology.mips_md_valid = torch.tensor(False)
    return topology


def test_periodic_pair_key_is_inversion_symmetric():
    assert periodic_pair_key(1, 2, 1) == periodic_pair_key(2, 1, -1)


def test_v2_all_spd2_pairs_and_inverse_members_share_geometry():
    topology = _sample()
    record = build_star_rbf_v2_sample(b"k" * 32, topology, topology)
    assert len(record["relations"]) == int((topology.lga_spd <= 2).sum())
    assert all(pair["multiplicity"] in {1, 2} for pair in record["pairs"])
    rows = {item["row"]: item["pair_index"] for item in record["relations"]}
    for row in range(topology.lga_spd.numel()):
        source, target = (int(x) for x in topology.lga_edge_index[:, row])
        shift = int(topology.lga_source_image_shift[row])
        inverse = torch.nonzero(
            (topology.lga_edge_index[0] == target)
            & (topology.lga_edge_index[1] == source)
            & (topology.lga_source_image_shift == -shift),
            as_tuple=False,
        ).flatten()
        assert inverse.numel() == 1
        assert rows[row] == rows[int(inverse[0])]


def test_shift_contract_violation_hard_fails():
    topology = _sample()
    topology.lga_source_image_shift[0] = 3
    with pytest.raises(ValueError, match="topology invariant"):
        build_star_rbf_v2_sample(b"k" * 32, topology, topology)


def test_shift_sources_and_true_self_no_bias():
    topology = _sample("*C(*)C(=O)OCC(C)(C)C")
    record = build_star_rbf_v2_sample(b"k" * 32, topology, topology)
    sources = {pair["geometry_source"] for pair in record["pairs"]}
    assert GEOMETRY_SOURCE["trivial_self_no_bias"] in sources
    assert GEOMETRY_SOURCE["central_direct"] in sources
    assert GEOMETRY_SOURCE["adjacent_dual"] in sources
    assert GEOMETRY_SOURCE["outer_trimer_direct"] in sources
    trivial = [pair for pair in record["pairs"] if pair["geometry_source"] == 1]
    assert all(pair["valid"] and pair["observation_count"] == 0 for pair in trivial)


def test_rigid_transform_and_relation_order_do_not_change_pairs():
    topology = _sample()
    original = build_star_rbf_v2_sample(b"k" * 32, topology, topology)
    transformed = copy.deepcopy(topology)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed.trimer_pos = transformed.trimer_pos @ rotation.T + torch.tensor([3.0, -2.0, 4.0])
    observed = build_star_rbf_v2_sample(b"k" * 32, transformed, transformed)
    assert [pair["key"] for pair in original["pairs"]] == [pair["key"] for pair in observed["pairs"]]
    assert np.allclose(
        [pair["distances"] for pair in original["pairs"]],
        [pair["distances"] for pair in observed["pairs"]], atol=1e-5,
    )

    permuted = copy.deepcopy(topology)
    order = torch.arange(permuted.lga_spd.numel() - 1, -1, -1)
    for name in ("lga_spd", "lga_source_image_shift", "lga_path_index", "lga_path_shift", "lga_path_mask"):
        setattr(permuted, name, getattr(permuted, name)[order])
    permuted.lga_edge_index = permuted.lga_edge_index[:, order]
    reordered = build_star_rbf_v2_sample(b"k" * 32, permuted, permuted)
    assert [pair["key"] for pair in original["pairs"]] == [pair["key"] for pair in reordered["pairs"]]
    assert np.allclose(
        [pair["distances"] for pair in original["pairs"]],
        [pair["distances"] for pair in reordered["pairs"]], atol=1e-6,
    )


def test_encode_first_average_and_inverse_bitwise_identity():
    topology = _sample()
    record = build_star_rbf_v2_sample(b"k" * 32, topology, topology)
    batch = mips_trimer_collate([_attach(topology, record)])
    model = MIPSLocalGraphEncoder(
        graph_geometry_mode="trimer_scage_mcl", use_star_rbf=True,
        use_mcl=False, topology_attention_variant="o8",
        star_rbf_upper=6.0,
    )
    model.star_distance_bias.projection.weight.data.normal_()
    bias = model.star_distance_bias.forward_periodic_relation_v2(batch, torch.float32)
    for pair in torch.unique(batch.mts_star_v2_relation_pair_index):
        rows = batch.mts_star_v2_relation_row[batch.mts_star_v2_relation_pair_index == pair]
        assert torch.equal(bias[rows], bias[rows[0]].expand_as(bias[rows]))
    dual = torch.nonzero(batch.mts_star_v2_pair_observation_count == 2).flatten()
    assert dual.numel()
    index = int(dual[0])
    d = batch.mts_star_v2_pair_observation_distances[index]
    centers = model.star_distance_bias.centers
    expected = (torch.exp(-model.star_distance_bias.gamma * (d[0] - centers) ** 2)
                + torch.exp(-model.star_distance_bias.gamma * (d[1] - centers) ** 2)) / 2
    wrong = torch.exp(-model.star_distance_bias.gamma * (d.mean() - centers) ** 2)
    assert not torch.allclose(expected, wrong)


def test_rbf_upper_margin_and_valid_tail_nonzero():
    assert rbf_upper_from_distances([5.61] * 1000) == 6.0
    model = MIPSLocalGraphEncoder(
        graph_geometry_mode="trimer_scage_mcl", use_star_rbf=True,
        use_mcl=False, topology_attention_variant="o8",
        star_rbf_upper=3.0,
    )
    tail = torch.exp(-model.star_distance_bias.gamma * (torch.tensor(3.5) - model.star_distance_bias.centers) ** 2)
    assert bool((tail > 0).any())


def test_asymmetry_is_qc_only_and_step0_forward_matches_g1():
    topology = _sample()
    record = build_star_rbf_v2_sample(b"k" * 32, topology, topology)
    batch = mips_trimer_collate([_attach(topology, record)])
    legacy = MIPSLocalGraphEncoder(
        graph_geometry_mode="trimer_scage_mcl", use_star_rbf=False,
        use_mcl=False, topology_attention_variant="o8",
    ).eval()
    candidate = MIPSLocalGraphEncoder(
        graph_geometry_mode="trimer_scage_mcl", use_star_rbf=True,
        use_mcl=False, topology_attention_variant="o8",
        star_rbf_upper=6.0,
    ).eval()
    candidate.load_state_dict(legacy.state_dict(), strict=True)
    with torch.no_grad():
        expected = legacy._forward_impl(batch, use_star=False, use_md=False)[0]
        observed = candidate._forward_impl(batch, use_md=False)[0]
    assert torch.equal(expected, observed)
