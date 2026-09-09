import math

import numpy as np
import pytest
import torch

from src.dataset.periodic_line_distill import transform_row
from src.dataset.periodic_line_glt import PeriodicLineGLTSidecar
from src.dataset.periodic_line_glt_image import PeriodicLineImageSidecar
from src.modules.mts_glt_distill import (
    AtomicConditionedMD200, NPlusGLTTeacher, SourceQPreLNAttention,
)


@pytest.fixture(scope="module")
def aligned_row():
    old = PeriodicLineGLTSidecar("data/processed/mips_trimer_scage/periodic_line_glt_v1/PI1M_v2")
    image = PeriodicLineImageSidecar("data/processed/mips_trimer_scage/periodic_line_glt_image_v1/PI1M_v2")
    index = next(i for i in range(100) if image.model_row(i)["geometry_valid"])
    return image.model_row(index), old.model_row(index)


def test_n_plus_token_counts_and_cross_lengths(aligned_row):
    image, old = aligned_row
    one = transform_row(image, old, "n_plus_1")
    two = transform_row(image, old, "n_plus_2")
    n = int((image["tokens"]["token_shift"] == 0).sum())
    assert len(one["tokens"]["token_shift"]) == n + 1
    assert len(two["tokens"]["token_shift"]) == n + 2
    assert one["tokens"]["token_center_internal"].sum() == n
    assert two["tokens"]["token_center_internal"].sum() == n
    cross = int(np.flatnonzero(image["tokens"]["token_shift"] != 0)[0])
    values = old["tokens"]["token_observation_distances"][cross, :2]
    assert one["tokens"]["token_distance"][-1] == pytest.approx(float(values.mean()))
    np.testing.assert_allclose(two["tokens"]["token_distance"][-2:], values)


def test_n_plus_2_relations_are_centered_deduplicated_and_bidirectional(aligned_row):
    row = transform_row(*aligned_row, "n_plus_2")
    rel = row["relations"]
    keys = list(zip(rel["relation_source"], rel["relation_target"], rel["relation_center_atom"]))
    assert len(keys) == len(set(keys))
    assert all((target, source, center) in set(keys) for source, target, center in keys)
    assert set(row["tokens"]["token_shift"][-2:]) == {-1, 1}


def test_n_plus_1_shared_state_preserves_left_right_relation_multiplicity(aligned_row):
    row = transform_row(*aligned_row, "n_plus_1")
    relation = row["relations"]
    shared = len(row["tokens"]["token_shift"]) - 1
    from_shared = relation["relation_source"] == shared
    assert set(relation["relation_source_image_shift"][from_shared].tolist()) == {-1, 1}
    assert row["tokens"]["token_shift"][shared] == 2


def test_source_q_attention_normalizes_incoming_and_has_gradients():
    layer = SourceQPreLNAttention(hidden=8, heads=2, dropout=0.0)
    states = torch.randn(3, 8, requires_grad=True)
    edges = torch.tensor([[0, 1, 2], [2, 2, 2]])
    output = layer(states, edges, torch.zeros(3, 2))
    assert output.shape == states.shape
    output.square().sum().backward()
    assert states.grad is not None and torch.isfinite(states.grad).all()


def test_atomic_md_initial_gate_and_invalid_exact_zero():
    module = AtomicConditionedMD200(dropout=0.0).eval()
    assert torch.sigmoid(module.bias).item() == pytest.approx(0.05)
    assert torch.count_nonzero(module.query.weight) == 0
    data = type("Batch", (), {})()
    data.mips_md = torch.randn(2, 200)
    data.mips_md_valid = torch.tensor([True, False])
    data.canonical_graph_index = torch.tensor([0, 0, 1])
    states = torch.randn(3, 512)
    output = module(states, data)
    torch.testing.assert_close(output[2], states[2])
    assert not torch.equal(output[0], states[0])


def test_teacher_is_endpoint_exchange_invariant_and_returns_center_only():
    data = type("Batch", (), {})()
    data.glt3_token_atom_a = torch.tensor([0, 1])
    data.glt3_token_atom_b = torch.tensor([1, 2])
    data.glt3_token_endpoint_z_a = torch.tensor([6, 8])
    data.glt3_token_endpoint_z_b = torch.tensor([8, 7])
    data.glt3_token_bond_type = torch.tensor([0, 1])
    data.glt3_token_stereo = torch.tensor([0, 0])
    data.glt3_token_conjugated = torch.tensor([False, True])
    data.glt3_token_distance = torch.tensor([1.2, 1.4])
    data.glt3_token_valid = torch.tensor([True, True])
    data.glt3_token_center_internal = torch.tensor([True, False])
    data.glt3_token_batch = torch.tensor([0, 0])
    data.glt3_relation_source = torch.tensor([0, 1])
    data.glt3_relation_target = torch.tensor([1, 0])
    data.glt3_relation_angle = torch.tensor([1.1, 1.1])
    data.glt3_relation_valid = torch.tensor([True, True])
    model = NPlusGLTTeacher(dropout=0.0).eval()
    original = model(data)
    data.glt3_token_atom_a, data.glt3_token_atom_b = (
        data.glt3_token_atom_b.clone(), data.glt3_token_atom_a.clone()
    )
    data.glt3_token_endpoint_z_a, data.glt3_token_endpoint_z_b = (
        data.glt3_token_endpoint_z_b.clone(), data.glt3_token_endpoint_z_a.clone()
    )
    exchanged = model(data)
    torch.testing.assert_close(original["line_states"], exchanged["line_states"])
    assert original["center_projected"].shape == (1, 256)
