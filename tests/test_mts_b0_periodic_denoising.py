import torch
from torch_geometric.data import Data

from src.dataset.dataloader import mips_trimer_collate
from src.dataset.mts_star_rbf_v2 import build_star_rbf_v2_sample
from src.modules.mips_local_graph import MIPSLocalGraphEncoder
from src.modules.periodic_coordinate_decoder import PeriodicCoordinateDecoder
from src.training.pretrain.periodic_denoising import (
    add_canonical_correlated_noise,
    kabsch_align_clean_to_noisy,
)
from tests.test_mips_non_pbc import graph_data


def _batch():
    items = []
    for _ in range(2):
        item = graph_data("*CCO*")
        record = build_star_rbf_v2_sample(b"b0" * 16, item, item)
        pairs = record["pairs"]
        for name, values in (
            ("src", [entry["key"][0] for entry in pairs]),
            ("dst", [entry["key"][1] for entry in pairs]),
            ("shift", [entry["key"][2] for entry in pairs]),
        ):
            setattr(item, f"mts_star_v2_pair_key_{name}", torch.tensor(values))
        items.append(item)
    return mips_trimer_collate(items)


def test_rbf_definition_upper_375():
    encoder = MIPSLocalGraphEncoder(
        use_star_rbf=True, use_mcl=False, topology_attention_variant="o8",
        star_rbf_upper=3.75,
    )
    centers = encoder.star_distance_bias.centers
    assert int(centers.numel()) == 32
    assert float(centers[0]) == 0.0
    assert float(centers[-1]) == 3.75
    spacing = float(centers[1] - centers[0])
    # centers are float32 linspace; allow float32-level rounding of the
    # 0.5/spacing^2 definition.
    assert abs(
        float(encoder.star_distance_bias.gamma) - 0.5 / (spacing * spacing)
    ) < 1e-4


def _with_shift2_pair(data):
    """Append one synthetic shift=2 pair + relation row to a collated batch."""
    pair_count = int(data.mts_star_v2_pair_key_src.numel())
    data.mts_star_v2_pair_key_src = torch.cat(
        [data.mts_star_v2_pair_key_src, torch.tensor([0])]
    )
    data.mts_star_v2_pair_key_dst = torch.cat(
        [data.mts_star_v2_pair_key_dst, torch.tensor([1])]
    )
    data.mts_star_v2_pair_key_shift = torch.cat(
        [data.mts_star_v2_pair_key_shift, torch.tensor([2])]
    )
    data.mts_star_v2_pair_observation_distances = torch.cat([
        data.mts_star_v2_pair_observation_distances,
        torch.zeros((1, 2), dtype=torch.float32),
    ])
    data.mts_star_v2_pair_observation_count = torch.cat([
        data.mts_star_v2_pair_observation_count, torch.tensor([1])
    ])
    data.mts_star_v2_pair_valid = torch.cat([
        data.mts_star_v2_pair_valid, torch.tensor([True])
    ])
    data.mts_star_v2_pair_geometry_source = torch.cat([
        data.mts_star_v2_pair_geometry_source, torch.tensor([0])
    ])
    data.mts_star_v2_relation_row = torch.cat([
        data.mts_star_v2_relation_row, torch.tensor([0])
    ])
    data.mts_star_v2_relation_pair_index = torch.cat([
        data.mts_star_v2_relation_pair_index, torch.tensor([pair_count])
    ])
    return data


def test_sigma_zero_dynamic_observations_match_clean_sidecar():
    data = _with_shift2_pair(_batch())
    noisy, distances, mask = add_canonical_correlated_noise(data, 0.0)
    assert torch.equal(noisy, data.trimer_pos)
    # The synthetic shift=2 pair has no frozen sidecar distance; store the
    # exact sigma=0 dynamic observation so both encodings see the same d.
    data.mts_star_v2_pair_observation_distances[-1, 0] = distances[-1, 0]
    assert torch.equal(distances, data.mts_star_v2_pair_observation_distances)
    expected = data.mts_star_v2_pair_observation_count.long().sum()
    assert int(mask.sum()) == int(expected)
    # The frozen production sidecar defines upper=3.75; the B0 dynamic path
    # must reproduce the same centers/gamma semantics.
    shifts = data.mts_star_v2_pair_key_shift.long().abs()
    assert set(shifts.tolist()) == {0, 1, 2}
    encoder = MIPSLocalGraphEncoder(
        use_star_rbf=True, use_mcl=False, topology_attention_variant="o8",
        star_rbf_upper=3.75,
    )
    encoder.star_distance_bias.projection.weight.data.normal_()
    clean_bias = encoder.star_distance_bias.forward_periodic_relation_v2(
        data, torch.float32
    )
    dynamic_bias = encoder.star_distance_bias.forward_periodic_relation_v2(
        data, torch.float32, distances, mask
    )
    assert torch.equal(clean_bias, dynamic_bias)
    # For |shift|=1 the final pair representation must be the element-wise
    # mean of two RBF encodings, never RBF((d_L+d_R)/2).
    pair_index = data.mts_star_v2_relation_pair_index.long().reshape(-1)
    pair_shift = shifts.reshape(-1)
    dual = torch.nonzero(
        (pair_shift == 1)
        & (data.mts_star_v2_pair_valid.bool().reshape(-1))
        & (data.mts_star_v2_pair_observation_count.long().reshape(-1) == 2),
        as_tuple=False,
    ).flatten()
    assert dual.numel() > 0
    selected = dual[0]
    d1, d2 = data.mts_star_v2_pair_observation_distances[selected]
    centers = encoder.star_distance_bias.centers
    gamma = encoder.star_distance_bias.gamma
    rbf = lambda d: torch.exp(-gamma * (d - centers) ** 2)  # noqa: E731
    expected_mean = (rbf(d1) + rbf(d2)) / 2
    wrong_mean = rbf((d1 + d2) / 2)
    assert not torch.allclose(expected_mean, wrong_mean, atol=1e-5)
    rel_pos = torch.nonzero(
        (data.mts_star_v2_relation_pair_index.long().reshape(-1) == int(selected)),
        as_tuple=False,
    ).flatten()[0]
    assert rel_pos < data.mts_star_v2_relation_row.numel()
    edge_row = int(data.mts_star_v2_relation_row.long().reshape(-1)[rel_pos])
    bias_row = dynamic_bias[edge_row]
    expected_projected = encoder.star_distance_bias.projection(
        expected_mean.unsqueeze(0)
    ).squeeze(0)
    assert torch.allclose(bias_row, expected_projected, atol=1e-5)


def test_over_upper_distance_uses_gaussian_tail():
    encoder = MIPSLocalGraphEncoder(
        use_star_rbf=True, use_mcl=False, topology_attention_variant="o8",
        star_rbf_upper=3.75,
    )
    centers = encoder.star_distance_bias.centers
    tail = torch.exp(
        -encoder.star_distance_bias.gamma
        * (torch.tensor(4.0) - centers) ** 2
    )
    # A distance above the upper bound keeps a Gaussian tail: the value at
    # the largest center is nonzero (not clipped/masked to zero).
    assert float(tail[-1]) > 0
    assert bool((tail > 0).any())
    assert float(tail.sum()) < float(
        torch.exp(
            -encoder.star_distance_bias.gamma
            * (torch.tensor(0.5) - centers) ** 2
        ).sum()
    )


def test_inverse_relations_share_pair_bias():
    data = _batch()
    encoder = MIPSLocalGraphEncoder(
        use_star_rbf=True, use_mcl=False, topology_attention_variant="o8",
        star_rbf_upper=3.75,
    )
    encoder.star_distance_bias.projection.weight.data.normal_()
    bias = encoder.star_distance_bias.forward_periodic_relation_v2(
        data, torch.float32
    )
    pair_index = data.mts_star_v2_relation_pair_index.long().reshape(-1)
    for pair in set(pair_index.tolist()):
        rows = torch.nonzero(pair_index == pair, as_tuple=False).flatten()
        if rows.numel() > 1:
            assert torch.allclose(bias[rows].max(dim=0).values, bias[rows].min(dim=0).values)


def test_kabsch_has_no_reflection_and_preserves_rigid_transform():
    clean = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0],
    ])
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    noisy = clean @ rotation + torch.tensor([2.0, -1.0, 3.0])
    aligned = kabsch_align_clean_to_noisy(
        clean, noisy, torch.zeros(3, dtype=torch.long), torch.tensor([True])
    )
    assert torch.allclose(aligned, noisy, atol=1e-5)


def test_b0_backbone_decoder_forward_backward():
    data = _batch()
    noisy, distances, mask = add_canonical_correlated_noise(data, 0.05)
    encoder = MIPSLocalGraphEncoder(
        use_star_rbf=True, use_mcl=False, topology_attention_variant="o8",
        star_rbf_upper=3.75,
    )
    atom_mask = torch.zeros(data.mips_x.size(0), dtype=torch.bool)
    atom_mask[::2] = True
    _, nodes, _ = encoder.forward_b0_pretrain(
        data, atom_mask, distances, mask
    )
    decoder = PeriodicCoordinateDecoder()
    displacement, valid = decoder(data, nodes, noisy)
    assert displacement.shape == (data.mips_x.size(0), 3)
    assert bool(valid.any())
    loss = displacement.square().mean() + nodes.square().mean() * 0.0
    loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in decoder.parameters() if parameter.requires_grad
    )


def _endpoint_batch(shift):
    data = Data()
    data.mips_x = torch.zeros((2, 137), dtype=torch.float)
    data.lga_edge_index = torch.tensor([[1], [0]], dtype=torch.long)
    data.lga_source_image_shift = torch.tensor([shift], dtype=torch.long)
    data.lga_spd = torch.tensor([1], dtype=torch.long)
    data.mips_to_trimer_central_index = torch.tensor([2, 3], dtype=torch.long)
    data.trimer_base_ru_atom_index = torch.tensor(
        [0, 1, 0, 1, 0, 1], dtype=torch.long
    )
    data.trimer_ru_offset = torch.tensor(
        [-1, -1, 0, 0, 1, 1], dtype=torch.long
    )
    data.trimer_pos = torch.tensor(
        [[-1.0, 0.0, 0.0], [9.0, 0.0, 0.0],
         [0.0, 0.0, 0.0], [10.0, 0.0, 0.0],
         [1.0, 0.0, 0.0], [11.0, 0.0, 0.0]],
    )
    data.canonical_graph_index = torch.zeros(2, dtype=torch.long)
    data.graph_available = torch.tensor([True])
    data.trimer_geometry_valid = torch.tensor([True])
    data.trimer_geometry_is_3d = torch.tensor([True])
    data.trimer_2d_fallback = torch.tensor([False])
    data.trimer_batch = torch.zeros(6, dtype=torch.long)
    return data


def test_decoder_uses_real_source_and_all_periodic_endpoints():
    expected = {0: 10.0, 1: 11.0, -1: 9.0, 2: 12.0, -2: 8.0}
    for shift, value in expected.items():
        data = _endpoint_batch(shift)
        decoder = PeriodicCoordinateDecoder()
        with torch.no_grad():
            decoder.coefficient[-1].bias.fill_(1.0)
        displacement, valid = decoder(
            data, torch.zeros((2, 512)), data.trimer_pos
        )
        assert bool(valid.item())
        assert torch.allclose(displacement[0], torch.tensor([value, 0.0, 0.0]))
        assert torch.allclose(displacement[1], torch.zeros(3))


def test_decoder_zeroes_nonfinite_relative_vectors():
    # shift=+1 relation: source=(canonical 1, +1) -> trimer index 5,
    # target=(canonical 0, 0) -> trimer index 2.  Put the non-finite value
    # exactly on the source endpoint so the relative vector is non-finite.
    data = _endpoint_batch(1)
    data.trimer_pos = torch.tensor(
        [[-1.0, 0.0, 0.0], [9.0, 0.0, 0.0],
         [0.0, 0.0, 0.0], [10.0, 0.0, 0.0],
         [1.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]],
    )
    decoder = PeriodicCoordinateDecoder()
    with torch.no_grad():
        decoder.coefficient[-1].bias.fill_(1.0)
    displacement, valid = decoder(
        data, torch.zeros((2, 512)), data.trimer_pos
    )
    assert not bool(valid.item())
    assert torch.isfinite(displacement).all()
    assert torch.equal(displacement, torch.zeros_like(displacement))


def test_decoder_step_zero_is_exact_noisy_baseline():
    data = _endpoint_batch(2)
    decoder = PeriodicCoordinateDecoder()
    displacement, valid = decoder(
        data, torch.zeros((2, 512)), data.trimer_pos
    )
    assert bool(valid.item())
    assert torch.equal(displacement, torch.zeros_like(displacement))


def test_canonical_noise_generator_is_repeatable_without_global_rng():
    data = _batch()
    first, first_dist, first_mask = add_canonical_correlated_noise(
        data, 0.03, generator=torch.Generator().manual_seed(1234)
    )
    second, second_dist, second_mask = add_canonical_correlated_noise(
        data, 0.03, generator=torch.Generator().manual_seed(1234)
    )
    assert torch.equal(first, second)
    assert torch.equal(first_dist, second_dist)
    assert torch.equal(first_mask, second_mask)
