"""新版 DistillStudent 的 GELU O8 与 Original-MIPS predictor 合同。"""

from pathlib import Path

import pytest
import torch
from torch import nn

from src.modules.mts_glt_distill import PreLNO8Encoder
from src.training.finetune.config import parse_arguments
from src.training.finetune.engine import (
    _student_architecture_metadata,
    build_mts_downstream_model,
)
from src.utils import _build_downstream_optimizer, _configure_mts_trainability


def _student_args():
    return parse_arguments([
        "--config_schema", "mts-glt-distill-repair-downstream",
        "--mts_glt_version", "distill_repair",
        "--distill_repair_version", "none",
        "--head_dropout", "0.25",
    ])


def test_all_six_source_q_o8_ffns_use_exact_gelu_and_dropout():
    encoder = PreLNO8Encoder()

    assert len(encoder.layers) == 6
    for layer in encoder.layers:
        assert isinstance(layer.ffn[1], nn.GELU)
        assert layer.ffn[1].approximate == "none"
        assert layer.ffn[2].p == 0.1
        assert layer.ffn[4].p == 0.1
        assert layer.attention.dropout.p == 0.1
        assert layer.attention.attention_dropout.p == 0.1


def test_distill_student_uses_identity_wrapper_and_mips_predictor():
    args = _student_args()
    model = build_mts_downstream_model(args)
    wrapper = model.encoders["graph"]
    predictor = model.mlp

    assert args.head_dropout == 0.1
    assert model.joint_embedding_dim == 512
    assert isinstance(wrapper.norm, nn.Identity)
    assert isinstance(wrapper.projection, nn.Identity)
    assert predictor[0].in_features == 512
    assert predictor[0].out_features == 512
    assert isinstance(predictor[1], nn.GELU)
    assert predictor[1].approximate == "none"
    assert isinstance(predictor[2], nn.Dropout)
    assert predictor[2].p == 0.1
    assert predictor[3].in_features == 512
    assert predictor[3].out_features == 1


def test_retired_glt_downstream_mode_is_rejected():
    args = parse_arguments([
        "--config_schema", "mts-glt-v2-downstream",
        "--mts_glt_version", "v2",
        "--mts_glt_mode", "o8_only",
        "--head_dropout", "0.25",
    ])
    with pytest.raises(ValueError, match="retired"):
        build_mts_downstream_model(args)


def test_distill_optimizer_covers_trainable_parameters_without_identity_params():
    model = build_mts_downstream_model(_student_args())
    _configure_mts_trainability(model)
    optimizer = _build_downstream_optimizer(
        model,
        graph_lr=1e-5,
        head_lr=1e-4,
        weight_decay=0.02,
        mts_o8_lr=1e-5,
        mts_geometry_lr=1e-5,
        mts_adapter_lr=1e-5,
    )
    trainable = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    listed = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert listed == trainable
    assert len(listed) == sum(len(group["params"]) for group in optimizer.param_groups)
    assert not list(model.encoders["graph"].norm.parameters())
    assert not list(model.encoders["graph"].projection.parameters())
    group_names = {group["name"].split("/")[0] for group in optimizer.param_groups}
    assert group_names >= {
        "o8", "md200_atom_residual", "regression_head"
    }
    assert "graph_adapter" not in group_names


def test_student_architecture_metadata_is_explicit_and_route_scoped():
    metadata = _student_architecture_metadata("distill_repair")
    assert metadata == {
        "o8_ffn_activation": "GELU(approximate='none')",
        "o8_ffn_hidden": "512->2048->512",
        "graph_adapter": "identity",
        "predictor": "512->512->1",
        "predictor_dropout": 0.1,
    }
    assert _student_architecture_metadata("v2") == {}


def test_student_state_strict_round_trip_preserves_eval_predictor_output(tmp_path):
    args = _student_args()
    source = build_mts_downstream_model(args).eval()
    inputs = torch.randn(2, 512)
    with torch.no_grad():
        expected = source.mlp(inputs)

    path = Path(tmp_path) / "student_model.pt"
    torch.save(source.state_dict(), path)
    restored = build_mts_downstream_model(args)
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    restored.eval()
    with torch.no_grad():
        actual = restored.mlp(inputs)
    torch.testing.assert_close(actual, expected)
