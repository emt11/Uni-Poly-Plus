import copy
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.dataloader import mips_trimer_collate
from src.dataset.lmdb_cache import sample_key_from_smiles
from src.dataset.periodic_line_glt import (
    build_periodic_line_sample,
    canonical_line_token,
    masked_line_label,
)
from src.dataset.trimer_mcl import attach_finite_trimer_mcl
from src.modules.periodic_line_glt import LocalPeriodicGraphLineTransformer
from src.modules.mts_glt import MTSGraphLineModel
from src.modules.mts_glt_v2 import MTSGraphLineModelV2
from src.modules.compact_trimer_descriptors import compact_trimer_descriptors
from src.modules.periodic_line_glt_v2 import (
    LearnedGaussianMoments,
    LocalPeriodicGraphLineTransformerV2,
    NUM_LINE_LABELS,
)
from src.training.pretrain.glt_objectives import distributed_bidirectional_infonce
from src.training.pretrain.glt_v2_objectives import make_masked_line_inputs_v2
from src.analysis.mts_glt_postmortem import (
    _ridge_path_predictions,
    contrastive_cosines,
    linear_cka,
    residual_probe,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def _sample(smiles="*CCO*"):
    topology = build_canonical_periodic_topology(smiles)
    attach_finite_trimer_mcl(topology, smiles, num_candidates=1)
    return topology


def _attach(data, record):
    data.glt_geometry_valid = bool(record["geometry_valid"])
    tokens, relations = record["tokens"], record["relations"]
    token_fields = {
        "token_atom_a": [item["atom_a"] for item in tokens],
        "token_atom_b": [item["atom_b"] for item in tokens],
        "token_shift": [item["shift"] for item in tokens],
        "token_endpoint_z_a": [item["z_a"] for item in tokens],
        "token_endpoint_z_b": [item["z_b"] for item in tokens],
        "token_label": [item["label"] for item in tokens],
        "token_observation_count": [item["observation_count"] for item in tokens],
        "token_valid": [item["valid"] for item in tokens],
    }
    for name, values in token_fields.items():
        dtype = torch.bool if name == "token_valid" else torch.long
        setattr(data, f"glt_{name}", torch.tensor(values, dtype=dtype))
    data.glt_token_observation_distances = torch.tensor(
        [item["distances"] for item in tokens], dtype=torch.float32
    )
    relation_fields = {
        "relation_source": [item["source"] for item in relations],
        "relation_target": [item["target"] for item in relations],
        "relation_center_atom": [item["center_atom"] for item in relations],
        "relation_multiplicity": [item["multiplicity"] for item in relations],
        "relation_observation_count": [item["observation_count"] for item in relations],
        "relation_valid": [item["valid"] for item in relations],
        "relation_is_fallback": [item["fallback"] for item in relations],
    }
    for name, values in relation_fields.items():
        dtype = torch.bool if name in {"relation_valid", "relation_is_fallback"} else torch.long
        setattr(data, f"glt_{name}", torch.tensor(values, dtype=dtype))
    data.glt_relation_observation_angles = torch.tensor(
        [item["angles"] for item in relations], dtype=torch.float32
    )
    data.mips_md = torch.zeros(200)
    data.mips_md_valid = torch.tensor(False)
    return data


def test_line_token_canonicalization_and_observation_counts():
    assert canonical_line_token(1, 0, 2, 1) == canonical_line_token(2, 1, 1, 0)
    topology = _sample()
    record = build_periodic_line_sample(sample_key_from_smiles("*CCO*"), topology, topology)
    assert record["geometry_valid"]
    assert sum(item["shift"] == 0 for item in record["tokens"]) == 2
    assert sum(abs(item["shift"]) == 1 for item in record["tokens"]) == 1
    assert sorted(item["observation_count"] for item in record["tokens"]) == [2, 3, 3]
    assert {item["observation_count"] for item in record["relations"]} <= {1, 2, 3}
    relations = {
        (item["source"], item["target"], item["center_atom"]): item
        for item in record["relations"] if not item["fallback"]
    }
    for (source, target, center), item in relations.items():
        inverse = relations[(target, source, center)]
        assert item["angles"] == inverse["angles"]
        assert item["observation_count"] == inverse["observation_count"]


def test_rigid_transform_preserves_line_distance_and_angle_observations():
    topology = _sample()
    key = sample_key_from_smiles("*CCO*")
    expected = build_periodic_line_sample(key, topology, topology)
    transformed = copy.deepcopy(topology)
    rotation = torch.tensor([
        [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]
    ])
    transformed.trimer_pos = transformed.trimer_pos @ rotation.T + torch.tensor([4.0, -3.0, 2.0])
    observed = build_periodic_line_sample(key, transformed, transformed)
    assert np.allclose(
        [item["distances"] for item in expected["tokens"]],
        [item["distances"] for item in observed["tokens"]], atol=1e-5,
    )
    assert np.allclose(
        [item["angles"] for item in expected["relations"]],
        [item["angles"] for item in observed["relations"]], atol=1e-5,
    )


def test_collate_offsets_and_glt_forward_do_not_consume_bond_type():
    first = _sample("*CCO*")
    second = _sample("*COC*")
    records = [
        build_periodic_line_sample(sample_key_from_smiles(smiles), sample, sample)
        for smiles, sample in (("*CCO*", first), ("*COC*", second))
    ]
    batch = mips_trimer_collate([
        _attach(first, records[0]), _attach(second, records[1])
    ])
    model = LocalPeriodicGraphLineTransformer(layers=2, hidden_size=64, heads=8).eval()
    with torch.no_grad():
        expected, states = model(batch)
        batch.glt_token_bond_type = torch.randint(0, 6, batch.glt_token_label.shape)
        observed, changed_states = model(batch)
    assert expected.shape == (2, 64)
    assert states.shape == changed_states.shape
    assert torch.equal(expected, observed)
    assert torch.equal(states, changed_states)
    assert int(batch.glt_relation_source.max()) < int(batch.glt_token_label.numel())
    assert masked_line_label(6, 1, 8) != masked_line_label(6, 2, 8)


def test_encode_then_mean_is_not_mean_then_encode():
    topology = _sample()
    record = build_periodic_line_sample(sample_key_from_smiles("*CCO*"), topology, topology)
    batch = mips_trimer_collate([_attach(topology, record)])
    model = LocalPeriodicGraphLineTransformer(layers=1, hidden_size=64, heads=8)
    dual = torch.nonzero(batch.glt_token_observation_count == 2).flatten()
    assert dual.numel() == 1
    index = int(dual[0])
    values = batch.glt_token_observation_distances[index:index + 1]
    encoded = model.distance_basis(values, torch.tensor([2]))
    wrong_values = values[:, :2].mean(dim=1, keepdim=True).expand(-1, 3)
    wrong = model.distance_basis(wrong_values, torch.tensor([1]))
    assert not torch.allclose(encoded, wrong)


def test_parallel_o8_glt_forward_and_invalid_fusion_fallback():
    first = _sample("*CCO*")
    second = _sample("*COC*")
    records = [
        build_periodic_line_sample(sample_key_from_smiles(smiles), sample, sample)
        for smiles, sample in (("*CCO*", first), ("*COC*", second))
    ]
    batch = mips_trimer_collate([
        _attach(first, records[0]), _attach(second, records[1])
    ])
    batch.glt_geometry_valid[1] = False
    model = MTSGraphLineModel(glt_layers=1).eval()
    model.glt_gate.data.fill_(1.0)
    with torch.no_grad():
        o8_only = model.forward_downstream(batch, "o8_only")
        fused = model.forward_downstream(batch, "o8_glt")
        views = model.encode_views(batch)
    assert o8_only.shape == fused.shape == (2, 512)
    assert torch.equal(o8_only[1], fused[1])
    assert not torch.equal(o8_only[0], fused[0])
    # MD200 is invalid in this fixture, so the ordinary fused forward must be
    # exactly the opt-in diagnostic decomposition.
    assert torch.allclose(fused, views["z_o8"] + views["delta_z_3d"])


def _infonce_worker(rank, init_file):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        base = torch.arange(12, dtype=torch.float32).reshape(3, 4) + rank
        first = base.clone().requires_grad_(True)
        second = (base + 0.5).clone().requires_grad_(True)
        valid = torch.tensor([True, rank == 0, False])
        loss, pool = distributed_bidirectional_infonce(first, second, valid)
        assert pool == 3
        loss.backward()
        assert torch.isfinite(loss)
        assert first.grad is not None and bool(torch.isfinite(first.grad).all())
        assert second.grad is not None and bool(torch.isfinite(second.grad).all())
    finally:
        dist.destroy_process_group()


def test_ddp_infonce_gathers_fixed_shapes_before_filter(tmp_path):
    init_file = tmp_path / "infonce_init"
    mp.spawn(_infonce_worker, args=(str(init_file),), nprocs=2, join=True)


def test_finetune_builds_both_glt_modes_without_star_rbf(monkeypatch):
    from src.dataset.mips_trimer_contract import validate_runtime_args
    from src.training.finetune.config import parse_arguments
    from src.training.finetune.engine import (
        build_mts_downstream_model,
        select_mts_glt_graph_state,
    )

    monkeypatch.setattr(sys, "argv", ["train"])
    args = parse_arguments()
    args.modalities = ["graph"]
    args.config_schema = "mts-glt-v1-downstream"
    args.use_star_rbf = False
    args.mts_glt_mode = "o8_glt"
    validate_runtime_args(args)
    model = build_mts_downstream_model(args)
    encoder = model.encoders["graph"].encoder
    assert encoder.downstream_mode == "o8_glt"
    assert encoder.use_star_rbf is False
    args.mts_glt_mode = "o8_only"
    validate_runtime_args(args)
    model = build_mts_downstream_model(args)
    assert model.encoders["graph"].encoder.downstream_mode == "o8_only"
    checkpoint_state = {
        "model." + key: value.clone()
        for key, value in model.encoders["graph"].encoder.state_dict().items()
    }
    mapped = select_mts_glt_graph_state(model.state_dict(), checkpoint_state)
    merged = model.state_dict()
    merged.update(mapped)
    model.load_state_dict(merged, strict=True)


def test_fusion_warm_cli_requires_o8_glt_and_parses_fixed_controls():
    from src.training.finetune.config import parse_arguments

    args = parse_arguments([
        "--mts_glt_mode", "o8_glt",
        "--mts_glt_fusion_strategy", "fusion_warm",
        "--mts_glt_fusion_warm_epochs", "5",
        "--mts_glt_initial_alpha", "0.1",
    ])
    assert args.mts_glt_fusion_strategy == "fusion_warm"
    assert args.mts_glt_fusion_warm_epochs == 5
    assert np.isclose(args.mts_glt_initial_alpha, 0.1)
    assert args.warmup_epochs == 0


def test_residual_probe_uses_cross_fitted_not_in_sample_residuals():
    rng = np.random.default_rng(7)
    size = 30
    # An identity design can memorize in-sample targets but cannot predict an
    # inner-heldout basis row.  A non-zero OOF residual therefore proves the
    # implementation did not subtract in-sample predictions.
    o8 = np.eye(size, dtype=np.float64)
    glt = rng.normal(size=(size, 3))
    target = 3.0 * glt[:, 0] + rng.normal(scale=0.01, size=size)
    observed = residual_probe(
        o8[:24], glt[:24], target[:24],
        o8[24:], glt[24:], target[24:], seed=19,
    )
    assert observed["oof_residual_rms"] > 0.5
    assert np.isfinite(observed["r2"])


def test_heldout_cka_and_multishift_cosine_are_deterministic():
    rng = np.random.default_rng(11)
    first = rng.normal(size=(20, 8))
    second = first.copy()
    assert np.isclose(linear_cka(first, second), 1.0)
    observed = contrastive_cosines(first, second, shifts=(1, 7, 21, 40))
    assert observed["legal_shifts"] == [1, 7]
    assert 0 not in observed["legal_shifts"]
    assert observed["matched"]["count"] == 20
    assert observed["unmatched"]["count"] == 40


def test_fast_ridge_path_matches_standard_ridge():
    rng = np.random.default_rng(23)
    train_x = rng.normal(size=(30, 6))
    test_x = rng.normal(size=(8, 6))
    train_y = rng.normal(size=(30, 4))
    scaler = StandardScaler().fit(train_x)
    scaled_train, scaled_test = scaler.transform(train_x), scaler.transform(test_x)
    alphas = (0.1, 10.0)
    observed = _ridge_path_predictions(
        scaled_train, train_y, scaled_test, alphas
    )
    for alpha, prediction in zip(alphas, observed):
        expected = Ridge(alpha=alpha).fit(scaled_train, train_y).predict(scaled_test)
        assert np.allclose(prediction, expected, atol=1e-10, rtol=1e-10)


def test_glt_v2_observation_moments_and_identity_self_relations():
    topology = _sample()
    record = build_periodic_line_sample(
        sample_key_from_smiles("*CCO*"), topology, topology
    )
    batch = mips_trimer_collate([_attach(topology, record)])
    basis = LearnedGaussianMoments(16, 0.0, 3.75)
    observations = batch.glt_token_observation_distances[:1]
    count = batch.glt_token_observation_count[:1]
    expected = basis(observations, count)
    permuted = basis(observations.flip(1), count)
    assert torch.allclose(expected[0], permuted[0])
    assert torch.allclose(expected[1], permuted[1])
    single_mean, single_variance = basis(observations, torch.ones_like(count))
    assert torch.isfinite(single_mean).all()
    assert torch.equal(single_variance, torch.zeros_like(single_variance))

    for variant in ("mips", "paper"):
        model = LocalPeriodicGraphLineTransformerV2(
            layers=1, hidden_size=64, heads=8, attention_variant=variant
        )
        output = model(batch)
        output["graph_geometry"].sum().backward()
        token_count = int(batch.glt_token_label.numel())
        source, target = output["relation_source"], output["relation_target"]
        identity = (source[-token_count:] == target[-token_count:])
        assert bool(identity.all())
        assert output["atom_geometry_states"].shape[0] == int(
            batch.canonical_graph_index.numel()
        )
        assert bool(output["atom_geometry_valid"].any())
        assert any(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )


def test_glt_v2_bond_type_is_not_input_and_invalid_fusion_is_exact_o8():
    first = _sample("*CCO*")
    second = _sample("*COC*")
    records = [
        build_periodic_line_sample(sample_key_from_smiles(smiles), sample, sample)
        for smiles, sample in (("*CCO*", first), ("*COC*", second))
    ]
    batch = mips_trimer_collate([
        _attach(first, records[0]), _attach(second, records[1])
    ])
    encoder = LocalPeriodicGraphLineTransformerV2(
        layers=1, hidden_size=64, heads=8
    ).eval()
    with torch.no_grad():
        expected = encoder(batch)["graph_geometry"]
        batch.glt_token_bond_type = torch.randint(0, 6, batch.glt_token_label.shape)
        observed = encoder(batch)["graph_geometry"]
    assert torch.equal(expected, observed)

    batch.glt_geometry_valid[1] = False
    model = MTSGraphLineModelV2(glt_layers=1).eval()
    with torch.no_grad():
        o8 = model.forward_downstream(batch, "o8_only")
        fused = model.forward_downstream(batch, "o8_glt_atom")
    assert torch.equal(o8[1], fused[1])
    assert not torch.equal(o8[0], fused[0])
    model.train()
    model.zero_grad(set_to_none=True)
    model.forward_downstream(batch, "o8_glt_atom").sum().backward()
    assert model.atom_channel_gate.grad is not None
    assert bool(torch.isfinite(model.atom_channel_gate.grad).all())
    assert float(model.atom_channel_gate.grad.abs().sum()) > 0.0

    boundary = torch.nonzero(
        batch.glt_token_shift.long().abs() == 1, as_tuple=False
    ).flatten()
    assert int(boundary.numel()) > 0
    for token in boundary.tolist():
        assert int(batch.glt_token_atom_a[token]) != int(batch.glt_token_atom_b[token])


def test_glt_v2_balanced_masking_uses_exact_count_and_whole_real_tokens():
    topology = _sample()
    record = build_periodic_line_sample(
        sample_key_from_smiles("*CCO*"), topology, topology
    )
    batch = mips_trimer_collate([_attach(topology, record)])
    frequencies = torch.ones(NUM_LINE_LABELS, dtype=torch.long)
    generator = torch.Generator().manual_seed(5)
    inputs = make_masked_line_inputs_v2(batch, 0.40, frequencies, generator)
    expected = max(1, round(0.40 * int(batch.glt_token_valid.sum())))
    assert int(inputs["selected"].sum()) == expected
    assert torch.isfinite(inputs["observation_distances"]).all()

    synthetic = copy.copy(batch)
    synthetic.glt_token_valid = torch.ones(100, dtype=torch.bool)
    synthetic.glt_token_label = torch.zeros(100, dtype=torch.long)
    synthetic.glt_token_label[-1] = 1
    synthetic.glt_token_endpoint_z_a = torch.full((100,), 6, dtype=torch.long)
    synthetic.glt_token_endpoint_z_b = torch.full((100,), 6, dtype=torch.long)
    synthetic.glt_token_observation_distances = torch.ones((100, 3))
    synthetic.glt_token_observation_count = torch.ones(100, dtype=torch.long)
    synthetic.glt_token_shift = torch.zeros(100, dtype=torch.long)
    skewed = torch.ones(NUM_LINE_LABELS, dtype=torch.long)
    skewed[0] = 10000
    rare_selected = 0
    for seed in range(20):
        observed = make_masked_line_inputs_v2(
            synthetic, 0.40, skewed, torch.Generator().manual_seed(seed)
        )
        rare_selected += int(observed["selected"][-1])
    assert rare_selected >= 18


def test_finetune_builds_glt_v2_atom_modes_and_strictly_maps_probe_state():
    from src.dataset.mips_trimer_contract import validate_runtime_args
    from src.training.finetune.config import parse_arguments
    from src.training.finetune.engine import (
        build_mts_downstream_model,
        select_mts_glt_graph_state,
    )

    args = parse_arguments([
        "--config_schema", "mts-glt-v2-downstream",
        "--graph_encoder_type", "mips_trimer_scage",
        "--no-use_star_rbf",
        "--mts_glt_version", "v2",
        "--mts_glt_layers", "6",
        "--mts_glt_attention_variant", "mips",
        "--mts_glt_mode", "o8_glt_atom",
    ])
    args.modalities = ["graph"]
    validate_runtime_args(args)
    model = build_mts_downstream_model(args)
    encoder = model.encoders["graph"].encoder
    assert isinstance(encoder, MTSGraphLineModelV2)
    assert encoder.downstream_mode == "o8_glt_atom"
    checkpoint_state = {
        "model." + key: value.clone() for key, value in encoder.state_dict().items()
    }
    mapped = select_mts_glt_graph_state(model.state_dict(), checkpoint_state)
    merged = model.state_dict()
    merged.update(mapped)
    model.load_state_dict(merged, strict=True)


def test_compact_trimer_descriptor_is_19d_finite_and_invalid_is_zero():
    first = _sample("*CCO*")
    second = _sample("*COC*")
    records = [
        build_periodic_line_sample(sample_key_from_smiles(smiles), sample, sample)
        for smiles, sample in (("*CCO*", first), ("*COC*", second))
    ]
    batch = mips_trimer_collate([
        _attach(first, records[0]), _attach(second, records[1])
    ])
    batch.glt_geometry_valid[1] = False
    values, valid = compact_trimer_descriptors(batch)
    assert values.shape == (2, 19)
    assert bool(torch.isfinite(values).all())
    assert bool(valid[0])
    assert not bool(valid[1])
    assert torch.equal(values[1], torch.zeros_like(values[1]))
