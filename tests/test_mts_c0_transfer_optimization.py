import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.training.c0_transfer import fit_probe_fold
from src.utils import TargetScaler, _set_staged_trainability, _build_stage1_optimizer, _audit_optimizer_parameters
from src.training.finetune.config import parse_arguments
from src.training.finetune.engine import build_mts_downstream_model


def test_probe_uses_train_scalers_and_validation_tie_prefers_larger_alpha():
    rng = np.random.default_rng(7)
    features = np.zeros((20, 6), dtype=np.float32)
    targets = np.exp(rng.normal(size=20))
    split = {"train_indices": list(range(12)), "validation_indices": list(range(12, 16)), "test_indices": list(range(16, 20))}
    metrics, state, truth, prediction, train, validation, test = fit_probe_fold(features, targets, split, "eps", alphas=(1.0, 10.0))
    expected_feature = StandardScaler().fit(features[train].astype(np.float64))
    expected_target = TargetScaler("eps", StandardScaler(), transform_mode="recommended")
    expected_target.scaler.fit(expected_target._pre_transform(targets[train]))
    assert np.array_equal(state["feature_mean"], expected_feature.mean_)
    assert np.array_equal(state["label_mean"], expected_target.scaler.mean_)
    assert metrics["selected_alpha"] == 10.0
    assert np.isfinite(prediction).all()
    assert np.array_equal(truth, targets[test])


def test_staged_trainability_freezes_only_deployment_encoder_in_stage1():
    args = parse_arguments(["--mts_glt_version", "distill_repair", "--distill_repair_version", "none", "--mips_norm_mode", "pre"])
    model = build_mts_downstream_model(args)
    _set_staged_trainability(model, encoder_trainable=False)
    encoder = model.encoders["graph"].encoder
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.encoders["graph"].projection.parameters())
    assert all(parameter.requires_grad for parameter in model.mlp.parameters())
    modules = [model.encoders["graph"].norm, model.encoders["graph"].projection, model.mlp]
    optimizer = _build_stage1_optimizer(modules, 1e-4, 0.02)
    _audit_optimizer_parameters(model, optimizer)
    assert {group["name"] for group in optimizer.param_groups} == {"task_head/decay", "task_head/no_decay"}
    assert {group["weight_decay"] for group in optimizer.param_groups} == {0.0, 0.02}
    _set_staged_trainability(model, encoder_trainable=True)
    assert any(parameter.requires_grad for parameter in encoder.o8.parameters())
    assert any(parameter.requires_grad for parameter in encoder.md_residual.parameters())


def test_staged_cli_contract():
    args = parse_arguments(["--finetune_strategy", "staged_head10", "--stage1_epochs", "10", "--stage2_epochs", "90"])
    assert args.finetune_strategy == "staged_head10"
    assert args.stage1_epochs + args.stage2_epochs == 100
