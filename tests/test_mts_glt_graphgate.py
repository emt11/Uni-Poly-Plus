import copy

import torch

from src.analysis.mts_trimer_validation import (
    graphgate_variant_view,
    kabsch_summary,
    trimer_sample_metrics,
)
from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.dataloader import mips_trimer_collate
from src.dataset.lmdb_cache import sample_key_from_smiles
from src.dataset.periodic_line_glt_central import build_periodic_line_central_sample
from src.dataset.trimer_mcl import attach_finite_trimer_mcl
from src.modules.periodic_line_glt_graphgate import LocalPeriodicGraphLineTransformerGraphGate
from src.training.pretrain.glt_graphgate_objectives import make_masked_line_states
from src.training.pretrain.glt_graphgate_engine import (
    MTSGraphGatePretrainContainer, _probe_payload,
)
from src.modules.mts_glt_graphgate import MTSGraphGateModel
from src.training.finetune.engine import select_mts_glt_graph_state


def _sample(smiles="*CCO*"):
    topology = build_canonical_periodic_topology(smiles)
    attach_finite_trimer_mcl(topology, smiles, num_candidates=1)
    return topology


def _attach(data, record):
    data.glt_geometry_valid = bool(record["runtime_valid"])
    data.glt_query_valid = bool(record["runtime_valid"] and record["tokens"])
    tokens, relations = record["tokens"], record["relations"]
    token_fields = {
        "token_atom_a": ("atom_a", torch.long), "token_atom_b": ("atom_b", torch.long),
        "token_shift": ("shift", torch.long), "token_endpoint_z_a": ("z_a", torch.long),
        "token_endpoint_z_b": ("z_b", torch.long), "token_label": ("label", torch.long),
        "token_observation_count": ("observation_count", torch.long),
        "token_valid": ("runtime_valid", torch.bool),
    }
    for name, (key, dtype) in token_fields.items():
        setattr(data, f"glt_{name}", torch.tensor([item[key] for item in tokens], dtype=dtype))
    for name, key, dtype in (
        ("token_observation_distances", "distances", torch.float32),
        ("token_observation_valid", "observation_valid", torch.bool),
        ("token_observation_translation", "translations", torch.long),
        ("token_observation_q_a", "q_a", torch.long),
        ("token_observation_q_b", "q_b", torch.long),
    ):
        setattr(data, f"glt_{name}", torch.tensor([item[key] for item in tokens], dtype=dtype))
    relation_fields = {
        "relation_source": ("source", torch.long), "relation_target": ("target", torch.long),
        "relation_center_atom": ("center_atom", torch.long),
        "relation_outer_offset_a": ("outer_offset_a", torch.long),
        "relation_outer_offset_b": ("outer_offset_b", torch.long),
        "relation_span": ("span", torch.long),
        "relation_multiplicity": ("multiplicity", torch.long),
        "relation_observation_count": ("observation_count", torch.long),
        "relation_valid": ("runtime_valid", torch.bool),
        "relation_is_fallback": ("fallback", torch.bool),
    }
    for name, (key, dtype) in relation_fields.items():
        setattr(data, f"glt_{name}", torch.tensor([item[key] for item in relations], dtype=dtype))
    for name, key, dtype in (
        ("relation_observation_angles", "angles", torch.float32),
        ("relation_observation_valid", "observation_valid", torch.bool),
        ("relation_observation_translation", "translations", torch.long),
    ):
        setattr(data, f"glt_{name}", torch.tensor([item[key] for item in relations], dtype=dtype))
    data.mips_md = torch.zeros(200); data.mips_md_valid = torch.tensor(False)
    return data


def test_central_distance_and_span_policy_keep_slot_identity():
    topology = _sample()
    record = build_periodic_line_central_sample(sample_key_from_smiles("*CCO*"), topology, topology)
    assert record["runtime_valid"]
    for token in record["tokens"]:
        if token["shift"] == 0:
            assert token["observation_valid"] == [True, False, False]
            assert token["translations"][0] == 0
        else:
            assert token["observation_valid"][:2] == [True, True]
            assert token["observation_count"] == 2
    for relation in record["relations"]:
        if relation["fallback"]:
            continue
        assert relation["multiplicity"] == {0: 1, 1: 2, 2: 1}[relation["span"]]
        assert sum(relation["observation_valid"]) == relation["multiplicity"]


def test_central_rigid_invariance_and_span_not_count_driven():
    topology = _sample(); key = sample_key_from_smiles("*CCO*")
    expected = build_periodic_line_central_sample(key, topology, topology)
    transformed = copy.deepcopy(topology)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    transformed.trimer_pos = transformed.trimer_pos @ rotation.T + torch.tensor([2., 3., -1.])
    observed = build_periodic_line_central_sample(key, transformed, transformed)
    assert torch.allclose(torch.tensor([x["distances"] for x in expected["tokens"]]), torch.tensor([x["distances"] for x in observed["tokens"]]), atol=1e-5)
    assert torch.allclose(torch.tensor([x["angles"] for x in expected["relations"]]), torch.tensor([x["angles"] for x in observed["relations"]]), atol=1e-5)


def test_graphgate_query_masks_invalid_after_final_norm_and_bond_is_not_input():
    first, second = _sample("*CCO*"), _sample("*COC*")
    records = [build_periodic_line_central_sample(sample_key_from_smiles(s), d, d) for s, d in (("*CCO*", first), ("*COC*", second))]
    batch = mips_trimer_collate([_attach(first, records[0]), _attach(second, records[1])])
    batch.glt_query_valid[1] = False
    model = LocalPeriodicGraphLineTransformerGraphGate(layers=2).eval()
    with torch.no_grad():
        clean = model.clean_line_inputs(batch)
        first_output = model(batch)
        batch.glt_token_label = torch.randint_like(batch.glt_token_label, 0, 10)
        batch.glt_token_observation_count.zero_()
        batch.glt_relation_observation_count.zero_()
        second_output = model(batch)
    assert torch.equal(first_output["graph_geometry"][1], torch.zeros(512))
    assert torch.equal(first_output["graph_geometry"], second_output["graph_geometry"])
    assert clean.shape[1] == 512


def test_graphgate_invalid_fusion_is_exact_o8():
    first, second = _sample("*CCO*"), _sample("*COC*")
    records = [build_periodic_line_central_sample(sample_key_from_smiles(s), d, d) for s, d in (("*CCO*", first), ("*COC*", second))]
    batch = mips_trimer_collate([_attach(first, records[0]), _attach(second, records[1])])
    batch.glt_query_valid[1] = False
    model = MTSGraphGateModel(layers=1).eval()
    with torch.no_grad():
        o8 = model.forward_downstream(batch, "o8_only")
        fused = model.forward_downstream(batch, "o8_glt_graph")
    assert torch.equal(o8[1], fused[1])
    assert not torch.equal(o8[0], fused[0])


def test_trimer_validation_metrics_are_rigid_invariant():
    topology = _sample("*CCO*")
    expected = trimer_sample_metrics(topology, topology)
    transformed = copy.deepcopy(topology)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    transformed.trimer_pos = transformed.trimer_pos @ rotation.T + torch.tensor([4., -2., 1.])
    observed = trimer_sample_metrics(transformed, transformed)
    assert expected["geometry_valid"] and observed["geometry_valid"]
    for key, value in expected.items():
        if key in {"geometry_valid", "failure_reason"}:
            continue
        torch.testing.assert_close(
            torch.tensor(float(value)), torch.tensor(float(observed[key])),
            rtol=1e-4, atol=1e-5,
        )


def test_single_atom_kabsch_is_translation_only():
    observed = kabsch_summary(
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 2.0, 2.0]]),
    )
    assert observed == {"translation": 3.0, "rotation_rad": 0.0, "rmsd": 0.0}


def test_graphgate_validation_variants_only_change_observation_masks():
    first, second = _sample("*CCO*"), _sample("*COC*")
    records = [
        build_periodic_line_central_sample(sample_key_from_smiles(smiles), data, data)
        for smiles, data in (("*CCO*", first), ("*COC*", second))
    ]
    batch = mips_trimer_collate([_attach(first, records[0]), _attach(second, records[1])])
    token_before = batch.glt_token_observation_valid.clone()
    relation_before = batch.glt_relation_observation_valid.clone()
    full = graphgate_variant_view(batch, "full")
    distance = graphgate_variant_view(batch, "distance_only")
    angle = graphgate_variant_view(batch, "angle_only")
    atom = graphgate_variant_view(batch, "atom_line_only")
    left = graphgate_variant_view(batch, "left_only")
    right = graphgate_variant_view(batch, "right_only")
    assert torch.equal(full.glt_token_observation_valid, token_before)
    assert torch.equal(full.glt_relation_observation_valid, relation_before)
    assert torch.equal(distance.glt_token_observation_valid, token_before)
    assert not distance.glt_relation_observation_valid.any()
    assert not angle.glt_token_observation_valid.any()
    assert torch.equal(angle.glt_relation_observation_valid, relation_before)
    assert not atom.glt_token_observation_valid.any()
    assert not atom.glt_relation_observation_valid.any()
    shift_one = batch.glt_token_shift.abs() == 1
    assert torch.all(
        batch.glt_token_observation_translation[left.glt_token_observation_valid & shift_one[:, None]] == -1
    )
    assert torch.all(
        batch.glt_token_observation_translation[right.glt_token_observation_valid & shift_one[:, None]] == 0
    )
    assert torch.equal(batch.glt_token_observation_valid, token_before)
    assert torch.equal(batch.glt_relation_observation_valid, relation_before)


def test_graphgate_full_validation_view_preserves_forward():
    sample = _sample("*CCO*")
    record = build_periodic_line_central_sample(
        sample_key_from_smiles("*CCO*"), sample, sample
    )
    batch = mips_trimer_collate([_attach(sample, record)])
    model = LocalPeriodicGraphLineTransformerGraphGate(layers=1).eval()
    with torch.no_grad():
        expected = model(batch)["graph_geometry"]
        observed = model(graphgate_variant_view(batch, "full"))["graph_geometry"]
        atom_only = model(graphgate_variant_view(batch, "atom_line_only"))["graph_geometry"]
    assert torch.equal(expected, observed)
    assert torch.isfinite(atom_only).all()
    assert not torch.equal(expected, atom_only)


def test_graphgate_geometry_off_keeps_topology_and_self_bias_but_zeros_geometry():
    sample = _sample("*CCO*")
    record = build_periodic_line_central_sample(
        sample_key_from_smiles("*CCO*"), sample, sample
    )
    batch = mips_trimer_collate([_attach(sample, record)])
    model = LocalPeriodicGraphLineTransformerGraphGate(layers=1).eval()
    clean_full = model.clean_line_inputs(batch, geometry_mode="full")
    clean_off = model.clean_line_inputs(batch, geometry_mode="off")
    source_full, target_full, bias_full = model.relation_graph(
        batch, clean_full.dtype, geometry_mode="full"
    )
    source_off, target_off, bias_off = model.relation_graph(
        batch, clean_off.dtype, geometry_mode="off"
    )
    real_count = int((~batch.glt_relation_is_fallback.bool()).sum())
    assert torch.equal(source_full, source_off)
    assert torch.equal(target_full, target_off)
    assert torch.equal(bias_off[:real_count], torch.zeros_like(bias_off[:real_count]))
    assert torch.equal(bias_full[real_count:], bias_off[real_count:])
    assert not torch.equal(clean_full, clean_off)
    output = model(batch, geometry_mode="off")["graph_geometry"]
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert model.distance_basis.centers.grad is not None
    assert model.angle_basis.centers.grad is not None


def test_masking_replaces_only_clean_self_representation_and_falls_back():
    data = type("Batch", (), {})()
    data.glt_token_valid = torch.tensor([True, True])
    data.glt_query_valid = torch.tensor([True])
    data.glt_token_batch = torch.tensor([0, 0])
    data.glt_token_label = torch.tensor([7, 7])
    clean = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    frequencies = torch.zeros(20, dtype=torch.long); frequencies[7] = 2
    result = make_masked_line_states(data, clean, torch.full((4,), -1.0), 1.0, frequencies, torch.Generator().manual_seed(5))
    assert torch.equal(data.glt_token_label, torch.tensor([7, 7]))
    assert result["selected"].all()
    assert torch.isfinite(result["line_states"]).all()


def test_probe_checkpoint_has_only_trained_namespaces_and_strict_encoder_reload():
    model = MTSGraphGateModel(layers=1)
    frequencies = torch.ones(40000, dtype=torch.long)
    container = MTSGraphGatePretrainContainer(model, frequencies)
    payload = _probe_payload(container, 500)
    assert set(payload["namespaces"]) == {
        "o8_encoder", "glt_line_encoder", "query_pool", "masked_atom_head",
        "masked_line_head", "o8_contrastive_head", "glt_contrastive_head",
    }
    assert not any("fusion" in key or "channel_gate" in key for values in payload["namespaces"].values() for key in values)
    clone = MTSGraphGateModel(layers=1)
    o8_state = clone.o8_encoder.state_dict()
    expected_o8 = {
        key for key in o8_state
        if not key.startswith("md_residual.") and not key.startswith("star_distance_bias.")
    }
    assert set(payload["namespaces"]["o8_encoder"]) == expected_o8
    o8_state.update(payload["namespaces"]["o8_encoder"])
    clone.o8_encoder.load_state_dict(o8_state, strict=True)
    glt_state = dict(payload["namespaces"]["glt_line_encoder"])
    glt_state.update({"query_pool." + key: value for key, value in payload["namespaces"]["query_pool"].items()})
    clone.glt_line_encoder.load_state_dict(glt_state, strict=True)
    prefix = "encoders.graph.encoder."
    wrapper_state = {prefix + key: value for key, value in clone.state_dict().items()}
    checkpoint_state = {}
    for key, value in payload["namespaces"]["o8_encoder"].items():
        checkpoint_state["model.o8_encoder." + key] = value
    for key, value in payload["namespaces"]["glt_line_encoder"].items():
        checkpoint_state["model.glt_line_encoder." + key] = value
    for key, value in payload["namespaces"]["query_pool"].items():
        checkpoint_state["model.glt_line_encoder.query_pool." + key] = value
    mapped = select_mts_glt_graph_state(wrapper_state, checkpoint_state, graphgate=True)
    assert set(mapped) == {
        key for key in wrapper_state
        if (key.startswith(prefix + "o8_encoder.")
            and not key.startswith(prefix + "o8_encoder.md_residual.")
            and not key.startswith(prefix + "o8_encoder.star_distance_bias."))
        or key.startswith(prefix + "glt_line_encoder.")
    }
