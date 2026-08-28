"""Focused unit coverage for the retained downstream baseline."""

from types import SimpleNamespace

import torch
from torch import nn

from src.modules.mts_glt_v2 import MTSGraphLineModelV2
from src.training.finetune.engine import build_mts_downstream_model, select_mts_glt_graph_state
from src.training.finetune.config import parse_arguments
from src.utils import (
    _build_downstream_optimizer,
    _configure_mts_trainability,
    finetune_bf16_parity_gate,
)


def _baseline_args():
    return parse_arguments([
        "--config_schema", "mts-glt-v2-downstream",
        "--graph_encoder_type", "mips_trimer_scage",
        "--no-use_star_rbf",
        "--mts_glt_version", "v2",
        "--mts_glt_layers", "6",
        "--mts_glt_attention_variant", "mips",
        "--mts_glt_mode", "o8_glt_atom",
    ])


def test_baseline_model_build_and_strict_graph_checkpoint_mapping():
    args = _baseline_args()
    model = build_mts_downstream_model(args)
    encoder = model.encoders["graph"].encoder
    assert isinstance(encoder, MTSGraphLineModelV2)
    assert encoder.downstream_mode == "o8_glt_atom"
    checkpoint_state = {
        "model." + key: value.clone()
        for key, value in encoder.state_dict().items()
    }
    mapped = select_mts_glt_graph_state(model.state_dict(), checkpoint_state)
    merged = model.state_dict()
    merged.update(mapped)
    model.load_state_dict(merged, strict=True)


def test_baseline_trainability_and_optimizer_have_no_duplicate_parameters():
    model = build_mts_downstream_model(_baseline_args())
    _configure_mts_trainability(model)
    encoder = model.encoders["graph"].encoder
    assert all(
        parameter.requires_grad
        for name, parameter in encoder.o8.named_parameters()
        if not name.startswith("star_distance_bias.")
    )
    assert all(
        not parameter.requires_grad
        for parameter in encoder.o8.star_distance_bias.parameters()
    )
    assert all(parameter.requires_grad for parameter in encoder.glt.parameters())
    assert all(not parameter.requires_grad for parameter in encoder.compact19_residual.parameters())
    optimizer = _build_downstream_optimizer(
        model,
        graph_lr=1e-5,
        head_lr=1e-4,
        weight_decay=0.02,
        mts_o8_lr=1e-5,
        mts_geometry_lr=1e-5,
        mts_adapter_lr=1e-5,
    )
    parameter_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert {group["name"].split("/")[0] for group in optimizer.param_groups} >= {
        "o8", "glt", "graph_adapter", "regression_head"
    }


def test_bf16_gate_fails_closed_without_cuda():
    passed, details = finetune_bf16_parity_gate(
        nn.Linear(2, 1), None, nn.MSELoss(), torch.device("cpu")
    )
    assert not passed
    assert details["reason"] == "cuda_bf16_unavailable"


def test_downstream_config_rejects_nonbaseline_modalities():
    try:
        parse_arguments(["--modalities", "graph", "fp"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("non-baseline modality was accepted")
