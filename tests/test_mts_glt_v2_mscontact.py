import copy
import importlib.util
from pathlib import Path

import torch
import numpy as np

from src.dataset.dataloader import mips_trimer_collate
from src.dataset.periodic_spatial_contact import build_spatial_contact_sample
from src.modules.mts_glt_v2 import MTSGraphLineModelV2
from src.modules.periodic_spatial_contact import PeriodicSpatialContactEncoder
from src.training.finetune.engine import select_mts_glt_graph_state
from src.training.pretrain.glt_v2_engine import (
    MTSGLTV2PretrainContainer, _checkpoint_payload, _probe_state_dict,
    _projection,
)
from src.modules.periodic_line_glt_v2 import GLTMaskedLineHeadV2, NUM_LINE_LABELS
from src.utils import _build_downstream_optimizer, _configure_mts_glt_fusion_stage


def _fixture():
    path = Path(__file__).with_name("test_mts_periodic_line_glt.py")
    spec = importlib.util.spec_from_file_location("mscontact_fixture", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _attach_spatial(data, record):
    pairs = record["pairs"]
    data.spatial_pair_index = torch.tensor(
        [[row["atom_a"] for row in pairs], [row["atom_b"] for row in pairs]],
        dtype=torch.long,
    ).reshape(2, -1)
    data.spatial_pair_shift = torch.tensor([row["shift"] for row in pairs])
    data.spatial_obs_distances = torch.from_numpy(
        np.asarray([row["distances"] for row in pairs], dtype=np.float32)
    ).reshape(-1, 3)
    data.spatial_obs_mask = torch.from_numpy(
        np.asarray([row["observation_valid"] for row in pairs], dtype=bool)
    ).reshape(-1, 3)
    data.spatial_obs_count = torch.tensor(
        [row["observation_count"] for row in pairs]
    )
    data.spatial_shell_id = torch.tensor([row["shell_id"] for row in pairs])
    data.spatial_periodic_self = torch.tensor(
        [row["periodic_self"] for row in pairs], dtype=torch.bool
    )
    data.spatial_pair_valid = torch.tensor(
        [row["valid"] for row in pairs], dtype=torch.bool
    )
    data.spatial_graph_valid = bool(record["graph_valid"])
    return data


def _batch():
    fixture = _fixture()
    samples = [fixture._sample("*CCO*"), fixture._sample("*CCCC*")]
    output = []
    for smiles, sample in zip(("*CCO*", "*CCCC*"), samples):
        key = fixture.sample_key_from_smiles(smiles)
        line = fixture.build_periodic_line_sample(key, sample, sample)
        spatial = build_spatial_contact_sample(key, sample, sample)
        output.append(_attach_spatial(fixture._attach(sample, line), spatial))
    return mips_trimer_collate(output)


def test_builder_filters_and_canonicalizes_contacts():
    batch = _batch()
    assert (batch.spatial_shell_id >= 0).all()
    assert (batch.spatial_shell_id <= 1).all()
    assert torch.equal(
        batch.spatial_obs_count, batch.spatial_obs_mask.sum(dim=1)
    )
    keys = list(zip(
        batch.spatial_pair_batch.tolist(),
        batch.spatial_pair_index[0].tolist(),
        batch.spatial_pair_index[1].tolist(),
        batch.spatial_pair_shift.tolist(),
    ))
    assert len(keys) == len(set(keys))
    periodic = batch.spatial_periodic_self
    assert torch.equal(
        periodic,
        (batch.spatial_pair_index[0] == batch.spatial_pair_index[1])
        & (batch.spatial_pair_shift != 0),
    )


def test_builder_is_rigid_transform_invariant():
    fixture = _fixture()
    sample = fixture._sample("*CCCC*")
    key = fixture.sample_key_from_smiles("*CCCC*")
    expected = build_spatial_contact_sample(key, sample, sample)
    transformed = copy.deepcopy(sample)
    rotation = torch.tensor([
        [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]
    ])
    transformed.trimer_pos = transformed.trimer_pos @ rotation.T + torch.tensor(
        [4.0, -3.0, 2.0]
    )
    actual = build_spatial_contact_sample(key, transformed, transformed)
    assert [(r["atom_a"], r["atom_b"], r["shift"]) for r in actual["pairs"]] == [
        (r["atom_a"], r["atom_b"], r["shift"]) for r in expected["pairs"]
    ]
    np.testing.assert_allclose(
        [r["distances"] for r in actual["pairs"]],
        [r["distances"] for r in expected["pairs"]], atol=1e-6, rtol=0.0,
    )


def test_observation_reordering_and_s4_outer_insensitivity():
    batch = _batch()
    encoder = PeriodicSpatialContactEncoder(dropout=0.0).eval()
    expected = encoder(batch, "s4")["atom_spatial_states"]
    reordered = copy.deepcopy(batch)
    reordered.spatial_obs_distances = torch.flip(
        reordered.spatial_obs_distances, dims=(1,)
    )
    reordered.spatial_obs_mask = torch.flip(reordered.spatial_obs_mask, dims=(1,))
    actual = encoder(reordered, "s4")["atom_spatial_states"]
    torch.testing.assert_close(actual, expected)
    changed = copy.deepcopy(batch)
    outer = changed.spatial_shell_id == 1
    changed.spatial_obs_distances[outer] = 99.0
    changed.spatial_obs_mask[outer] = torch.tensor([True, False, False])
    changed.spatial_obs_count[outer] = 1
    changed.spatial_pair_shift[outer] = 2
    changed.spatial_pair_valid[outer] = False
    changed.spatial_periodic_self[outer] = False
    changed.spatial_pair_index[:, outer] = 0
    actual = encoder(changed, "s4")["atom_spatial_states"]
    torch.testing.assert_close(actual, expected)


def test_distance_is_encoded_per_observation_before_moments():
    encoder = PeriodicSpatialContactEncoder(dropout=0.0).eval()
    observations = torch.tensor([[2.0, 4.0, 0.0]])
    counts = torch.tensor([2])
    z = torch.tensor([6])
    mean, variance = encoder.distance_basis(
        observations, counts, z_a=z, z_b=z
    )
    encoded_mean, _ = encoder.distance_basis(
        torch.tensor([[3.0, 0.0, 0.0]]), torch.tensor([1]), z_a=z, z_b=z
    )
    assert not torch.allclose(mean, encoded_mean)
    assert float(variance.detach().abs().sum()) > 0


def test_spatial_invalid_is_exact_zero_and_full_model_falls_back():
    batch = _batch()
    batch.spatial_graph_valid[:] = False
    spatial = PeriodicSpatialContactEncoder(dropout=0.0).eval()(batch, "ms45")
    assert torch.equal(
        spatial["atom_spatial_states"],
        torch.zeros_like(spatial["atom_spatial_states"]),
    )
    torch.manual_seed(31)
    baseline = MTSGraphLineModelV2(glt_layers=1).eval()
    torch.manual_seed(31)
    candidate = MTSGraphLineModelV2(
        glt_layers=1, use_spatial_contact=True, spatial_shell_mode="ms45"
    ).eval()
    candidate_state = candidate.state_dict()
    candidate_state.update({
        key: value for key, value in baseline.state_dict().items()
        if key in candidate_state and candidate_state[key].shape == value.shape
    })
    candidate.load_state_dict(candidate_state, strict=True)
    with torch.no_grad():
        expected = baseline.forward_downstream(batch, "o8_glt_atom")
        actual = candidate.forward_downstream(batch, "o8_glt_atom_spatial")
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0.0)


def test_shell_arms_are_state_dict_matched_and_spatial_has_gradients():
    torch.manual_seed(42)
    s4 = MTSGraphLineModelV2(
        glt_layers=1, use_spatial_contact=True, spatial_shell_mode="s4"
    )
    torch.manual_seed(42)
    ms45 = MTSGraphLineModelV2(
        glt_layers=1, use_spatial_contact=True, spatial_shell_mode="ms45"
    )
    torch.manual_seed(42)
    c5 = MTSGraphLineModelV2(
        glt_layers=1, use_spatial_contact=True, spatial_shell_mode="c5_mixed"
    )
    assert s4.state_dict().keys() == ms45.state_dict().keys() == c5.state_dict().keys()
    assert all(
        torch.equal(s4.state_dict()[key], ms45.state_dict()[key])
        and torch.equal(ms45.state_dict()[key], c5.state_dict()[key])
        for key in s4.state_dict()
    )
    batch = _batch()
    output = ms45.spatial_encoder(batch, "ms45")["graph_spatial"]
    output.square().mean().backward()
    parameters = (
        ms45.spatial_encoder.distance_basis.centers,
        ms45.spatial_encoder.distance_mean_projection.weight,
        ms45.spatial_encoder.distance_variance_projection.weight,
        ms45.spatial_encoder.message_mlp[0].weight,
    )
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and float(parameter.grad.abs().sum()) > 0
        for parameter in parameters
    )


def test_c5_mixed_ignores_shell_identity_and_is_one_unified_mean():
    batch = _batch()
    encoder = PeriodicSpatialContactEncoder(dropout=0.0).eval()
    expected = encoder(batch, "c5_mixed")["atom_spatial_states"]

    swapped = copy.deepcopy(batch)
    swapped.spatial_shell_id = 1 - swapped.spatial_shell_id
    actual = encoder(swapped, "c5_mixed")["atom_spatial_states"]
    torch.testing.assert_close(actual, expected)

    all_core = copy.deepcopy(batch)
    all_core.spatial_shell_id.zero_()
    unified_reference = encoder(all_core, "ms45")["atom_spatial_states"]
    torch.testing.assert_close(expected, unified_reference)


def test_probe_excludes_downstream_spatial_gate_and_strict_mapping_accepts_it():
    model = MTSGraphLineModelV2(
        glt_layers=1, use_spatial_contact=True, spatial_shell_mode="s4"
    )
    container = MTSGLTV2PretrainContainer(
        model, torch.nn.Linear(512, int(model.o8.masked_atom_classes)),
        GLTMaskedLineHeadV2(512), _projection(256), _projection(256),
        torch.ones(NUM_LINE_LABELS, dtype=torch.long),
    )
    checkpoint = _probe_state_dict(container)
    assert "model.spatial_channel_gate" not in checkpoint
    assert "model.atom_channel_gate" not in checkpoint
    assert not any(key.startswith("model.atom_fusion_") for key in checkpoint)
    assert not any(key.startswith("model.o8.md_residual.") for key in checkpoint)
    assert any(key.startswith("model.spatial_encoder.") for key in checkpoint)
    assert any(key.startswith("spatial_nce_projection.") for key in checkpoint)
    downstream_state = {
        "encoders.graph.encoder." + key: value
        for key, value in model.state_dict().items()
    }
    mapped = select_mts_glt_graph_state(downstream_state, checkpoint)
    assert "encoders.graph.encoder.spatial_channel_gate" not in mapped
    merged = dict(downstream_state)
    merged.update(mapped)
    assert merged.keys() == downstream_state.keys()


def test_train_state_records_exact_sampler_position():
    model = MTSGraphLineModelV2(
        glt_layers=1, use_spatial_contact=True, spatial_shell_mode="s4"
    )
    container = MTSGLTV2PretrainContainer(
        model, torch.nn.Linear(512, int(model.o8.masked_atom_classes)),
        GLTMaskedLineHeadV2(512), _projection(256), _projection(256),
        torch.ones(NUM_LINE_LABELS, dtype=torch.long),
    )
    optimizer = torch.optim.Adam(container.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    args = type("Args", (), {
        "gradient_accumulation_steps": 1, "glt_layers": 1,
        "glt_attention_variant": "mips", "glt_metadata_mode": "full",
        "use_spatial_contact": True, "spatial_shell_mode": "s4",
    })()
    payload = _checkpoint_payload(
        container, optimizer, scheduler, 500, args, 3,
        sampler_epoch=0, batches_in_epoch=500,
    )
    assert payload["optimizer_steps_completed"] == 500
    assert payload["sampler_epoch"] == 0
    assert payload["batches_in_epoch"] == 500


def test_warm0_optimizer_covers_spatial_encoder_and_gate_once():
    class GraphModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = MTSGraphLineModelV2(
                glt_layers=1, use_spatial_contact=True,
                spatial_shell_mode="ms45",
            )
            self.encoder.downstream_mode = "o8_glt_atom_spatial"
            self.norm = torch.nn.LayerNorm(512)
            self.projection = torch.nn.Linear(512, 512)

    class Downstream(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.modality_list = ("graph",)
            self.encoders = torch.nn.ModuleDict({"graph": GraphModule()})
            self.mlp = torch.nn.Linear(512, 1)
            self.modality_heads = torch.nn.ModuleDict()
            self.cross_task_aux_heads = torch.nn.ModuleDict()
            self.residual_modality_gates = torch.nn.ParameterDict()

    model = Downstream()
    encoder = model.encoders["graph"].encoder
    _configure_mts_glt_fusion_stage(model, "joint")
    optimizer = _build_downstream_optimizer(
        model, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-4, 0.02,
        mts_o8_lr=1e-5, mts_geometry_lr=1e-5,
        mts_glt_fusion_stage="joint",
    )
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert groups["spatial_encoder/decay"]["lr"] == 1e-5
    assert groups["spatial_channel_gate/no_decay"]["lr"] == 1e-5
    parameter_ids = [
        id(parameter) for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert id(encoder.spatial_channel_gate) in parameter_ids
    assert all(
        id(parameter) in parameter_ids
        for parameter in encoder.spatial_encoder.parameters()
    )
