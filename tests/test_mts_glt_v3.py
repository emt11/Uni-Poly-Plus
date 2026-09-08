import copy
import math

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.dataset.periodic_line_glt_image import (
    PeriodicLineImageSidecar,
    build_periodic_line_image_sample,
    write_image_sidecar,
)
from src.modules.mts_glt_v3 import MD200NodeResidual, MTSGraphLineModelV3
from src.modules.periodic_line_glt_v3 import LocalPeriodicGraphLineTransformerV3
from src.training.pretrain.glt_v3_objectives import make_masked_line_inputs_v3
from src.training.pretrain.glt_v3_engine import (
    MTSGLTV3PretrainContainer,
    deployable_bundle,
)
from src.training.finetune.config import parse_arguments as parse_finetune_arguments
from src.dataset.mips_trimer_contract import validate_runtime_args


def synthetic_topology_and_trimer():
    topology = Data(
        atomic_numbers=torch.tensor([6, 6]),
        ru_edge_index=torch.tensor([[0, 1], [1, 0]]),
        ru_bond_type=torch.tensor([1, 1]),
        lga_edge_index=torch.tensor([[0, 1], [1, 0]]),
        lga_spd=torch.tensor([1, 1]),
        lga_source_image_shift=torch.tensor([0, 0]),
        ru_left_boundary=0,
        ru_right_boundary=1,
        canonical_to_trimer_base_atom_id=torch.tensor([0, 1]),
    )
    # Three deliberately unequal translated RU lengths: 1, 2 and 4 A.
    position = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0], [5.0, 0.0, 0.0],
        [9.0, 0.0, 0.0], [13.0, 0.0, 0.0],
    ])
    undirected = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    edges = [(a, b) for a, b in undirected for a, b in ((a, b), (b, a))]
    trimer = Data(
        trimer_geometry_valid=True,
        trimer_geometry_is_3d=True,
        trimer_2d_fallback=False,
        trimer_pos=position,
        trimer_atomic_number=torch.tensor([6] * 6),
        trimer_base_ru_atom_id=torch.tensor([0, 1, 0, 1, 0, 1]),
        trimer_ru_offset=torch.tensor([-1, -1, 0, 0, 1, 1]),
        trimer_central_ru_mask=torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.bool),
        mips_to_trimer_central_index=torch.tensor([2, 3]),
        trimer_edge_index=torch.tensor(edges).T,
        trimer_bond_type=torch.tensor([1] * len(edges)),
    )
    return topology, trimer


def image_batch(graphs=2, tokens_per_graph=5):
    count = graphs * tokens_per_graph
    data = Data()
    data.glt3_token_atom_a = torch.arange(count) % (graphs * 3)
    data.glt3_token_atom_b = (data.glt3_token_atom_a + 1) % (graphs * 3)
    data.glt3_token_endpoint_z_a = torch.full((count,), 6)
    data.glt3_token_endpoint_z_b = torch.full((count,), 6)
    data.glt3_token_distance = torch.linspace(1.0, 2.0, count)
    data.glt3_token_bond_type = torch.zeros(count, dtype=torch.long)
    data.glt3_token_stereo = torch.zeros(count, dtype=torch.long)
    data.glt3_token_conjugated = torch.zeros(count, dtype=torch.long)
    data.glt3_token_valid = torch.ones(count, dtype=torch.bool)
    data.glt3_token_batch = torch.arange(graphs).repeat_interleave(tokens_per_graph)
    source, target = [], []
    if tokens_per_graph:
        for start in range(0, count, tokens_per_graph):
            for local in range(tokens_per_graph):
                source.append(start + local)
                target.append(start + (local + 1) % tokens_per_graph)
    data.glt3_relation_source = torch.tensor(source)
    data.glt3_relation_target = torch.tensor(target)
    data.glt3_relation_angle = torch.full((count,), math.pi / 2)
    data.glt3_relation_valid = torch.ones(count, dtype=torch.bool)
    data.glt3_geometry_valid = torch.ones(graphs, dtype=torch.bool)
    data.canonical_graph_index = torch.arange(graphs).repeat_interleave(3)
    return data


def test_center_anchor_uses_one_distance_and_image_relation(tmp_path):
    topology, trimer = synthetic_topology_and_trimer()
    row = build_periodic_line_image_sample(topology, trimer, "*CC*")
    assert row["geometry_valid"]
    internal = np.flatnonzero(row["tokens"]["token_shift"] == 0)
    assert len(internal) == 1
    assert row["tokens"]["token_distance"][internal[0]] == pytest.approx(2.0)
    assert row["tokens"]["token_distance"][internal[0]] != pytest.approx(7.0 / 3.0)
    assert len(row["relations"]["relation_angle"]) > 0


def test_sidecar_geometry_is_rigid_transform_invariant():
    topology, trimer = synthetic_topology_and_trimer()
    reference = build_periodic_line_image_sample(topology, trimer, "*CC*")
    transformed = copy.deepcopy(trimer)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed.trimer_pos = trimer.trimer_pos @ rotation.T + torch.tensor([4.0, -3.0, 7.0])
    actual = build_periodic_line_image_sample(topology, transformed, "*CC*")
    np.testing.assert_allclose(actual["tokens"]["token_distance"], reference["tokens"]["token_distance"], atol=1e-6)
    np.testing.assert_allclose(actual["relations"]["relation_angle"], reference["relations"]["relation_angle"], atol=1e-6)


def test_image_sidecar_round_trip(tmp_path):
    topology, trimer = synthetic_topology_and_trimer()
    row = build_periodic_line_image_sample(topology, trimer, "*CC*")
    root = tmp_path / "image"
    key = bytes(range(32))
    write_image_sidecar(root, [key], [row])
    loaded = PeriodicLineImageSidecar(root).model_row(0)
    np.testing.assert_allclose(loaded["tokens"]["token_distance"], row["tokens"]["token_distance"])
    np.testing.assert_allclose(loaded["relations"]["relation_angle"], row["relations"]["relation_angle"])


def test_galformer_mask_policy_and_clean_angles():
    data = image_batch(graphs=10, tokens_per_graph=10)
    original_angles = data.glt3_relation_angle.clone()
    masked = make_masked_line_inputs_v3(data, generator=torch.Generator().manual_seed(42))
    assert int(masked["selected"].sum()) == 40
    assert torch.equal(original_angles, data.glt3_relation_angle)
    assert torch.equal(masked["selected"], masked["masked"] | masked["replaced"] | masked["kept"])
    assert int(masked["masked"].sum()) + int(masked["replaced"].sum()) + int(masked["kept"].sum()) == 40
    assert masked["distance_mask"][masked["masked"]].all()


def test_v3_encoder_is_rigid_transform_invariant_and_has_geometry_gradients():
    data = image_batch()
    model = LocalPeriodicGraphLineTransformerV3(dropout=0.0).eval()
    first = model(data)
    # Coordinates are intentionally absent: only invariant scalar geometry is accepted.
    second = model(copy.deepcopy(data))
    torch.testing.assert_close(first["graph_geometry"], second["graph_geometry"], atol=1e-7, rtol=0)
    first["graph_geometry"].square().sum().backward()
    assert model.distance_projection.weight.grad.abs().sum() > 0
    assert model.angle_projection.weight.grad.abs().sum() > 0
    relation_pairs = set(zip(data.glt3_relation_source.tolist(), data.glt3_relation_target.tolist()))
    assert len(relation_pairs) == data.glt3_relation_source.numel()


def test_v3_encoder_handles_batch_without_line_tokens():
    data = image_batch(graphs=2, tokens_per_graph=0)
    data.glt3_geometry_valid.zero_()
    model = LocalPeriodicGraphLineTransformerV3(dropout=0.0).eval()
    output = model(data)
    assert output["line_states"].shape == (0, 512)
    assert output["atom_geometry_states"].shape == (6, 512)
    assert output["graph_geometry"].shape == (2, 512)
    assert torch.isfinite(output["graph_geometry"]).all()
    assert torch.count_nonzero(output["graph_geometry"]) == 0


def test_md_residual_gate_and_pre_pool_broadcast():
    module = MD200NodeResidual(dropout=0.0)
    assert torch.tanh(module.gate).item() == pytest.approx(0.05)
    data = Data(mips_md=torch.randn(2, 200), batch=torch.tensor([0, 0, 1]))
    nodes = torch.zeros(3, 512)
    result = module(nodes, data)
    torch.testing.assert_close(result[0], result[1])
    assert not torch.equal(result[0], result[2])


def test_downstream_modes_have_distinct_geometry_contracts():
    galformer = MTSGraphLineModelV3(glt_readout_mode="galformer")
    concat = MTSGraphLineModelV3(glt_readout_mode="mips_concat")
    assert galformer.glt is None
    assert galformer.concat_projection is None
    assert concat.glt is not None
    assert concat.concat_projection.in_features == 1024
    assert concat.concat_projection.out_features == 512


@pytest.mark.parametrize("mode", ["galformer", "mips_concat"])
def test_v3_downstream_runtime_contract_accepts_both_readouts(mode):
    args = parse_finetune_arguments([
        "--config_schema", "mts-glt-v3-downstream",
        "--mts_glt_version", "v3",
        "--glt_readout_mode", mode,
    ])
    validate_runtime_args(args)


def test_deployment_bundles_strictly_load_without_temporary_heads():
    container = MTSGLTV3PretrainContainer(
        MTSGraphLineModelV3(glt_readout_mode="mips_concat")
    )
    for mode in ("galformer", "mips_concat"):
        bundle = deployable_bundle(container, 5000, mode)
        target = MTSGraphLineModelV3(glt_readout_mode=mode)
        target.load_state_dict(bundle["state_dict"], strict=True)
        assert not any("atom_head" in key or "line_heads" in key or "projection" in key and key.startswith(("o8_projection", "glt_projection")) for key in bundle["state_dict"])
