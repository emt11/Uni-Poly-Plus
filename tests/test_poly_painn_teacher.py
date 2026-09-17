import math

import pytest
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader

from src.dataset.poly_painn_teacher import build_teacher_sample, validate_central_mapping
from src.modules.poly_painn_teacher import (
    PolyPaiNNTeacher, load_teacher_deployment, teacher_deployment_package,
    teacher_global_objective,
)
from src.training.glt_dual_runtime import restore_rng, rng_state


def _input(pos=None):
    if pos is None:
        pos = torch.tensor([
            [0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [0.0, 1.5, 0.2],
            [0.2, 0.3, 2.1], [7.0, 0.0, 0.0],
        ])
    return Data(
        z=torch.tensor([6, 8, 1, 7, 6]), pos=pos,
        batch=torch.zeros(pos.size(0), dtype=torch.long),
        central_index=torch.tensor([0, 1, 2, 3]),
        central_batch=torch.zeros(4, dtype=torch.long),
    )


def test_translation_and_rotation_equivariance():
    torch.manual_seed(7)
    model = PolyPaiNNTeacher(hidden_channels=12, num_layers=2, rbf_dim=10).eval()
    data = _input()
    reference = model(data)
    translated = _input(data.pos + torch.tensor([11.0, -4.0, 2.5]))
    moved = model(translated)
    assert torch.allclose(reference["central_scalar_states"], moved["central_scalar_states"], atol=2e-5)
    assert torch.allclose(reference["predicted_noise"], moved["predicted_noise"], atol=2e-5)
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(q) < 0:
        q[:, 0] *= -1
    rotated = _input(data.pos @ q.T)
    turned = model(rotated)
    assert torch.allclose(reference["central_scalar_states"], turned["central_scalar_states"], atol=2e-5)
    assert torch.allclose(turned["central_vector_states"], reference["central_vector_states"] @ q.T, atol=3e-5)
    assert torch.allclose(turned["predicted_noise"], reference["predicted_noise"] @ q.T, atol=3e-5)


def test_permutation_equivariance_with_inverse_mapping():
    torch.manual_seed(8)
    model = PolyPaiNNTeacher(hidden_channels=12, num_layers=2, rbf_dim=10).eval()
    data = _input()
    ref = model(data)
    permutation = torch.tensor([3, 0, 4, 2, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    perm = Data(
        z=data.z[permutation], pos=data.pos[permutation],
        batch=data.batch[permutation], central_index=inverse[data.central_index],
        central_batch=data.central_batch,
    )
    got = model(perm)
    # Central order is kept canonical, so outputs should match directly.
    assert torch.allclose(ref["central_scalar_states"], got["central_scalar_states"], atol=3e-5)
    assert torch.allclose(ref["central_vector_states"], got["central_vector_states"], atol=3e-5)


def _top_trimer(mapping=None, z=None):
    topology = Data(mips_x=torch.zeros(3, 137), z=torch.tensor([6, 8, 7]))
    atomic = torch.tensor([6, 8, 7, 1, 6, 8])
    if z is not None:
        atomic = torch.as_tensor(z)
    trimer = Data(
        trimer_pos=torch.tensor([
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
            [0., 0., 1.], [6., 0., 0.], [6., 1., 0.],
        ]),
        trimer_heavy_indices=torch.tensor([0, 1, 2, 3, 4, 5]),
        trimer_atomic_number=atomic,
        trimer_central_ru_mask=torch.tensor([True, True, True, False, False, False]),
        mips_to_trimer_central_index=torch.tensor([0, 1, 2]),
        trimer_geometry_valid=True, trimer_geometry_is_3d=True, trimer_2d_fallback=False,
    )
    if mapping is not None:
        trimer.mips_to_trimer_central_index = torch.as_tensor(mapping)
    return topology, trimer


def test_central_mapping_and_heavy_subset_are_strict():
    topology, trimer = _top_trimer()
    checked = validate_central_mapping(topology, trimer)
    assert checked["central_heavy_index"].tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="duplicates"):
        validate_central_mapping(topology, _top_trimer([0, 0, 2])[1])
    with pytest.raises(ValueError, match="atomic-number"):
        validate_central_mapping(topology, _top_trimer(z=[6, 9, 7, 1, 6, 8])[1])
    bad = _top_trimer()[1]
    bad.trimer_central_ru_mask[2] = False
    with pytest.raises(ValueError, match="centre RU"):
        validate_central_mapping(topology, bad)
    topology_h = Data(mips_x=torch.zeros(3, 137), z=torch.tensor([6, 1, 7]))
    trimer_h = _top_trimer()[1]
    trimer_h.trimer_atomic_number[1] = 1
    trimer_h.mips_to_trimer_central_index = torch.tensor([0, 1, 2])
    trimer_h.o8_heavy_mask = torch.tensor([True, False, True])
    checked_h = validate_central_mapping(topology_h, trimer_h)
    assert checked_h["canonical_heavy_indices"].tolist() == [0, 2]
    assert checked_h["central_heavy_index"].tolist() == [0, 2]


def test_deterministic_noise_uses_seed_key_and_absolute_position():
    topology, trimer = _top_trimer()
    a, la = build_teacher_sample(topology, trimer, seed=42, key="aa", position=11)
    b, lb = build_teacher_sample(topology, trimer, seed=42, key="aa", position=11)
    c, lc = build_teacher_sample(topology, trimer, seed=42, key="aa", position=12)
    assert torch.equal(a.pos, b.pos)
    assert torch.equal(la["epsilon"], lb["epsilon"])
    assert not torch.equal(a.pos, c.pos)
    assert la["position"] == 11 and lc["position"] == 12
    invalid = _top_trimer()[1]
    invalid.trimer_geometry_valid = False
    with pytest.raises(ValueError, match="valid frozen 3D"):
        build_teacher_sample(topology, invalid, seed=42, key="aa", position=11)


def test_radius_cutoff_and_neighbor_cap_are_reported():
    model = PolyPaiNNTeacher(hidden_channels=8, num_layers=1, rbf_dim=8,
                             cutoff=1.5, max_num_neighbors=2)
    data = Data(
        z=torch.tensor([6, 6, 6, 6]),
        pos=torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [5., 0., 0.]]),
        batch=torch.zeros(4, dtype=torch.long),
        central_index=torch.arange(4), central_batch=torch.zeros(4, dtype=torch.long),
    )
    result = model(data)
    assert int(result["edge_count"]) >= 4
    assert int(result["neighbor_hist"].sum()) == int(data.num_nodes)
    assert int(result["neighbor_hist"][-1]) >= 0
    assert result["neighbor_count"][-1].item() == 0


def test_deployment_excludes_denoising_head_and_loads_strictly():
    model = PolyPaiNNTeacher(hidden_channels=8, num_layers=1, rbf_dim=8)
    package = teacher_deployment_package(model, 5)
    assert package["schema"] == "poly-painn-teacher-deployment-v1"
    assert not any(name.startswith("noise_head.") for name in package["encoder"])
    restored = PolyPaiNNTeacher(hidden_channels=8, num_layers=1, rbf_dim=8)
    load_teacher_deployment(restored, package, expected_step=5)
    for name, value in model.state_dict().items():
        if name.startswith("noise_head."):
            continue
        assert torch.equal(value, restored.state_dict()[name])


def test_geometry_noise_gradients_are_finite():
    torch.manual_seed(9)
    model = PolyPaiNNTeacher(hidden_channels=8, num_layers=1, rbf_dim=8)
    data = _input()
    out = model(data)
    target = torch.zeros_like(out["predicted_noise"])
    loss = (out["predicted_noise"] - target).square().mean()
    loss.backward()
    assert math.isfinite(float(loss))
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_ddp_graph_sum_count_formula():
    local = torch.tensor([2.0, 4.0])
    # Two ranks each contribute two graph means; DDP averages gradients, hence
    # the explicit world-size factor recovers the global graph mean.
    assert torch.equal(teacher_global_objective(local, torch.tensor(8.0), 2), torch.tensor(1.5))
    with pytest.raises(ValueError):
        teacher_global_objective(torch.empty(0), 1.0)


def test_worker_iterator_rng_restore_is_exact():
    class Values(torch.utils.data.Dataset):
        def __len__(self):
            return 8
        def __getitem__(self, index):
            return int(index)

    torch.manual_seed(1234)
    state = rng_state()
    loader = DataLoader(Values(), batch_size=None, num_workers=1, persistent_workers=False)
    iterator = iter(loader)
    next(iterator)
    restore_rng(state)
    after_prefetch = torch.rand(16)
    restore_rng(state)
    reference = torch.rand(16)
    assert torch.equal(after_prefetch, reference)
    del iterator, loader
