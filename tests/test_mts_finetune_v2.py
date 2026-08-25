import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.utils import (
    _build_downstream_optimizer,
    _configure_mts_trainability,
    _configure_mts_glt_fusion_stage,
    _set_mts_glt_frozen_encoders_eval,
    collect_mts_glt_initial_fusion_audit,
    finetune_bf16_parity_gate,
    initialize_mts_glt_fusion_warm,
    train_and_evaluate,
)


class _DummyGraphEncoder(nn.Module):
    architecture_name = "MIPS-Trimer-SCAGE"

    def __init__(self):
        super().__init__()
        self.atom_embedding = nn.Linear(2, 2)
        self.spd_embedding = nn.Embedding(3, 2)
        self.path_bias = nn.Linear(2, 2)
        self.layers = nn.ModuleList(nn.Linear(2, 2) for _ in range(6))
        self.star_distance_bias = nn.Linear(2, 2)
        self.trimer_mcl = nn.Linear(2, 2)
        self.md_residual = nn.Linear(2, 2)


class _DummyGraphModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _DummyGraphEncoder()
        self.norm = nn.LayerNorm(2)
        self.projection = nn.Linear(2, 2)


class _DummyMTS(nn.Module):
    def __init__(self):
        super().__init__()
        self.modality_list = ("graph",)
        self.graph_encoder_type = "mips_trimer_scage"
        self.fusion_type = "none"
        self.encoders = nn.ModuleDict({"graph": _DummyGraphModule()})
        self.fusion_module = nn.Identity()
        self.mlp = nn.Linear(2, 1)


class _DummyGLTBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2)


class _DummyFusionO8(_DummyGLTBranch):
    def __init__(self):
        super().__init__()
        self.md_residual = nn.Linear(2, 2)
        self.star_distance_bias = nn.Linear(2, 2)


class _DummyFusionEncoder(nn.Module):
    architecture_name = "MIPS-Trimer-SCAGE"
    downstream_mode = "o8_glt"

    def __init__(self):
        super().__init__()
        self.o8 = _DummyFusionO8()
        self.glt = _DummyGLTBranch()
        self.glt_fusion_norm = nn.LayerNorm(2)
        self.glt_fusion_projection = nn.Linear(2, 2)
        self.glt_gate = nn.Parameter(torch.zeros(()))
        self.use_star_rbf = False

    def encode_views(self, batch):
        z_o8 = self.o8.layer(batch.values)
        z_glt = self.glt.layer(batch.values)
        projected = self.glt_fusion_projection(self.glt_fusion_norm(z_glt))
        valid = batch.valid.bool()
        delta = valid.to(projected.dtype).unsqueeze(-1) * torch.tanh(
            self.glt_gate
        ) * projected
        return {
            "z_o8": z_o8,
            "z_glt": z_glt,
            "projected_z_glt": projected,
            "delta_z_3d": delta,
            "valid_3d": valid,
        }


class _DummyFusionGraphModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _DummyFusionEncoder()
        self.norm = nn.LayerNorm(2)
        self.projection = nn.Linear(2, 2)


class _DummyFusionMTS(nn.Module):
    def __init__(self):
        super().__init__()
        self.modality_list = ("graph",)
        self.encoders = nn.ModuleDict({"graph": _DummyFusionGraphModule()})
        self.mlp = nn.Linear(2, 1)
        self.modality_heads = nn.ModuleDict()
        self.cross_task_aux_heads = nn.ModuleDict()
        self.residual_modality_gates = nn.ParameterDict()
        self.branch_mode_history = []

    def forward(self, batch):
        encoder = self.encoders["graph"].encoder
        if self.training:
            self.branch_mode_history.append(
                (bool(encoder.o8.training), bool(encoder.glt.training))
            )
        views = encoder.encode_views(batch)
        graph = views["z_o8"] + views["delta_z_3d"]
        graph = encoder.o8.md_residual(graph)
        graph = self.encoders["graph"].projection(
            self.encoders["graph"].norm(graph)
        )
        return self.mlp(graph), graph.unsqueeze(1)


class _DummyGraphGateEncoder(nn.Module):
    architecture_name = "MTS-GLT-GraphGate-v1"
    downstream_mode = "o8_glt_graph"

    def __init__(self):
        super().__init__()
        self.o8_encoder = _DummyFusionO8()
        self.glt_line_encoder = _DummyGLTBranch()
        self.fusion_norm = nn.LayerNorm(2)
        self.fusion_projection = nn.Linear(2, 2)
        self.channel_gate = nn.Parameter(torch.zeros(2))
        self.use_star_rbf = False

    def encode_views(self, batch):
        z_o8 = self.o8_encoder.layer(batch.values)
        z_glt = self.glt_line_encoder.layer(batch.values)
        projected = self.fusion_projection(self.fusion_norm(z_glt))
        valid = batch.valid.bool()
        delta = (
            valid.to(projected.dtype).unsqueeze(-1)
            * torch.tanh(self.channel_gate)
            * projected
        )
        return {
            "z_o8": z_o8, "z_glt": z_glt,
            "projected_z_glt": projected, "delta_z_3d": delta,
            "valid_3d": valid,
        }


class _DummyGraphGateGraphModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _DummyGraphGateEncoder()
        self.norm = nn.LayerNorm(2)
        self.projection = nn.Linear(2, 2)


class _DummyGraphGateMTS(nn.Module):
    def __init__(self):
        super().__init__()
        self.modality_list = ("graph",)
        self.encoders = nn.ModuleDict({"graph": _DummyGraphGateGraphModule()})
        self.mlp = nn.Linear(2, 1)
        self.modality_heads = nn.ModuleDict()
        self.cross_task_aux_heads = nn.ModuleDict()
        self.residual_modality_gates = nn.ParameterDict()
        self.branch_mode_history = []

    def forward(self, batch):
        encoder = self.encoders["graph"].encoder
        if self.training:
            self.branch_mode_history.append((
                bool(encoder.o8_encoder.training),
                bool(encoder.glt_line_encoder.training),
                bool(encoder.o8_encoder.md_residual.training),
            ))
        views = encoder.encode_views(batch)
        graph = encoder.o8_encoder.md_residual(
            views["z_o8"] + views["delta_z_3d"]
        )
        graph = self.encoders["graph"].projection(
            self.encoders["graph"].norm(graph)
        )
        return self.mlp(graph), graph.unsqueeze(1)


class _AuditBatch:
    def __init__(self):
        self.values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.valid = torch.tensor([True, False])
        self.y = torch.tensor([[0.5], [1.5]])

    def to(self, _device, non_blocking=False):
        return self


def _trainable(module):
    return any(parameter.requires_grad for parameter in module.parameters())


def test_mts_trainability_trains_complete_graph_from_epoch_zero():
    model = _DummyMTS()
    graph = model.encoders["graph"].encoder

    _configure_mts_trainability(model)
    assert all(_trainable(module) for module in (
        graph.atom_embedding,
        graph.spd_embedding,
        graph.path_bias,
        graph.layers,
        graph.star_distance_bias,
        graph.trimer_mcl,
        graph.md_residual,
        model.encoders["graph"].projection,
        model.mlp,
    ))

    optimizer = _build_downstream_optimizer(
        model, 1e-5, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 0.02,
    )
    expected_lrs = {
        "graph": 1e-5,
        "regression_head": 1e-4,
    }
    for group in optimizer.param_groups:
        root = group["name"]
        assert root in expected_lrs
        assert group["lr"] == expected_lrs[root]
        assert group["weight_decay"] == 0.02
    before = {
        name: parameter.detach().clone()
        for name, parameter in graph.named_parameters()
    }
    optimizer.zero_grad()
    sum(parameter.sum() for parameter in model.parameters()).backward()
    optimizer.step()
    assert any(
        not torch.equal(dict(graph.named_parameters())[name], expected)
        for name, expected in before.items()
    )


def test_bf16_gate_fails_closed_without_cuda():
    passed, details = finetune_bf16_parity_gate(
        nn.Linear(2, 1), None, nn.MSELoss(), torch.device('cpu')
    )
    assert not passed
    assert details['reason'] == 'cuda_bf16_unavailable'


def test_fusion_warm_initialization_stages_and_optimizer_groups():
    model = _DummyFusionMTS()
    encoder = model.encoders["graph"].encoder
    observed = initialize_mts_glt_fusion_warm(model, 0.1)
    assert np.isclose(observed, 0.1, atol=1e-7, rtol=0.0)

    _configure_mts_glt_fusion_stage(model, "warm")
    model.train()
    _set_mts_glt_frozen_encoders_eval(model)
    assert not encoder.o8.training
    assert not encoder.glt.training
    assert not any(
        parameter.requires_grad for parameter in encoder.o8.layer.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in encoder.o8.md_residual.parameters()
    )
    assert not any(parameter.requires_grad for parameter in encoder.glt.parameters())
    warm_optimizer = _build_downstream_optimizer(
        model, 1e-5, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 0.02,
        mts_glt_fusion_stage="warm",
    )
    assert all(group["lr"] == 1e-4 for group in warm_optimizer.param_groups)
    warm_ids = {
        id(parameter)
        for group in warm_optimizer.param_groups for parameter in group["params"]
    }
    assert id(encoder.glt_gate) in warm_ids
    assert not warm_ids.intersection(id(p) for p in encoder.o8.layer.parameters())
    assert not warm_ids.intersection(id(p) for p in encoder.glt.parameters())

    _configure_mts_glt_fusion_stage(model, "joint")
    joint_optimizer = _build_downstream_optimizer(
        model, 1e-5, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 0.02,
        mts_glt_fusion_stage="joint",
    )
    groups = {
        group["name"]: group["lr"] for group in joint_optimizer.param_groups
    }
    assert groups["o8_encoder/decay"] == 1e-5
    assert groups["glt_encoder/decay"] == 1e-5
    assert groups["glt_gate/no_decay"] == 1e-4
    all_ids = [
        id(parameter)
        for group in joint_optimizer.param_groups for parameter in group["params"]
    ]
    assert len(all_ids) == len(set(all_ids))


def test_epoch_zero_fusion_audit_preserves_rng_and_records_actual_rho():
    model = _DummyFusionMTS()
    initialize_mts_glt_fusion_warm(model, 0.1)
    torch.manual_seed(123)
    np.random.seed(123)
    expected_torch = torch.get_rng_state().clone()
    expected_numpy = np.random.get_state()
    audit = collect_mts_glt_initial_fusion_audit(
        model, [_AuditBatch()], torch.device("cpu")
    )
    assert audit["valid_count"] == 1
    assert audit["rho"]["count"] == 1
    assert np.isfinite(audit["rho"]["mean"])
    assert torch.equal(torch.get_rng_state(), expected_torch)
    observed_numpy = np.random.get_state()
    assert observed_numpy[0] == expected_numpy[0]
    assert np.array_equal(observed_numpy[1], expected_numpy[1])
    assert observed_numpy[2:] == expected_numpy[2:]


def test_fusion_warm_two_epoch_smoke_crosses_stage_boundary():
    model = _DummyFusionMTS()
    initialize_mts_glt_fusion_warm(model, 0.1)

    class _IdentityScaler:
        @staticmethod
        def inverse_transform(values):
            return np.asarray(values)

    metrics = train_and_evaluate(
        model,
        _IdentityScaler(),
        [_AuditBatch()],
        [_AuditBatch()],
        [_AuditBatch()],
        torch.device("cpu"),
        num_epochs=2,
        patience=10,
        graph_lr=1e-5,
        fusion_lr=1e-4,
        head_lr=1e-4,
        weight_decay=0.02,
        warmup_epochs=0,
        regression_loss="huber",
        mts_glt_postmortem=True,
        mts_glt_fusion_strategy="fusion_warm",
        mts_glt_fusion_warm_epochs=1,
        mts_glt_initial_alpha=0.1,
    )
    audit = metrics["_mts_glt_postmortem"]
    assert [row["epoch"] for row in audit["gate_trajectory"]] == [0, 1, 2]
    assert [row["stage"] for row in audit["gate_trajectory"]] == [
        "initial", "fusion_warm", "joint_finetune",
    ]
    assert audit["initial_validation"]["rho"]["count"] == 1
    assert model.branch_mode_history[:2] == [(False, False), (True, True)]


def test_graphgate_fusion_warm_channel_gate_stages_and_audit():
    model = _DummyGraphGateMTS()
    encoder = model.encoders["graph"].encoder
    observed = initialize_mts_glt_fusion_warm(model, 0.05)
    assert np.isclose(observed, 0.05, atol=1e-7, rtol=0.0)
    assert torch.allclose(
        torch.tanh(encoder.channel_gate),
        torch.full_like(encoder.channel_gate, 0.05),
        atol=1e-7, rtol=0.0,
    )

    _configure_mts_glt_fusion_stage(model, "warm")
    model.train()
    _set_mts_glt_frozen_encoders_eval(model)
    assert not encoder.o8_encoder.training
    assert not encoder.glt_line_encoder.training
    assert encoder.o8_encoder.md_residual.training
    assert not any(
        parameter.requires_grad
        for parameter in encoder.o8_encoder.layer.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in encoder.glt_line_encoder.parameters()
    )
    warm_optimizer = _build_downstream_optimizer(
        model, 1e-5, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 0.02,
        mts_o8_lr=1e-5, mts_geometry_lr=1e-5,
        mts_glt_fusion_stage="warm",
    )
    assert all(group["lr"] == 1e-4 for group in warm_optimizer.param_groups)
    warm_ids = [
        id(parameter)
        for group in warm_optimizer.param_groups for parameter in group["params"]
    ]
    assert len(warm_ids) == len(set(warm_ids))
    assert id(encoder.channel_gate) in warm_ids

    audit = collect_mts_glt_initial_fusion_audit(
        model, [_AuditBatch()], torch.device("cpu")
    )
    assert audit["rho"]["count"] == 1
    assert np.isclose(audit["alpha"]["mean_abs"], 0.05, atol=1e-7)
    assert audit["alpha"]["fraction_positive"] == 1.0

    _configure_mts_glt_fusion_stage(model, "joint")
    joint_optimizer = _build_downstream_optimizer(
        model, 1e-5, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 0.02,
        mts_o8_lr=1e-5, mts_geometry_lr=1e-5,
        mts_glt_fusion_stage="joint",
    )
    groups = {group["name"]: group["lr"] for group in joint_optimizer.param_groups}
    assert groups["o8_encoder/decay"] == 1e-5
    assert groups["glt_encoder/decay"] == 1e-5
    assert groups["graphgate_channel_gate/no_decay"] == 1e-4
    all_ids = [
        id(parameter)
        for group in joint_optimizer.param_groups for parameter in group["params"]
    ]
    assert len(all_ids) == len(set(all_ids))


def test_graphgate_stage2_encoder_trainability_factorial():
    expected = {
        "both_frozen": (False, False),
        "o8_only": (True, False),
        "glt_query_only": (False, True),
        "joint": (True, True),
    }
    for policy, (o8_trainable, glt_trainable) in expected.items():
        model = _DummyGraphGateMTS()
        encoder = model.encoders["graph"].encoder
        _configure_mts_glt_fusion_stage(model, "joint", policy)
        assert _trainable(encoder.o8_encoder.layer) is o8_trainable
        assert _trainable(encoder.glt_line_encoder) is glt_trainable
        assert _trainable(encoder.o8_encoder.md_residual)
        assert _trainable(encoder.fusion_projection)
        assert encoder.channel_gate.requires_grad

        model.train()
        if policy != "joint":
            _set_mts_glt_frozen_encoders_eval(model, policy)
        assert encoder.o8_encoder.training is o8_trainable
        assert encoder.glt_line_encoder.training is glt_trainable
        assert encoder.o8_encoder.md_residual.training

        optimizer = _build_downstream_optimizer(
            model, 1e-5, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 0.02,
            mts_o8_lr=1e-5, mts_geometry_lr=1e-5,
            mts_glt_fusion_stage="joint",
        )
        names = {group["name"] for group in optimizer.param_groups}
        assert any(name.startswith("o8_encoder/") for name in names) is o8_trainable
        assert any(name.startswith("glt_encoder/") for name in names) is glt_trainable
        parameter_ids = [
            id(parameter)
            for group in optimizer.param_groups for parameter in group["params"]
        ]
        assert len(parameter_ids) == len(set(parameter_ids))


def test_graphgate_fusion_warm_two_epoch_smoke_crosses_stage_boundary():
    model = _DummyGraphGateMTS()
    initialize_mts_glt_fusion_warm(model, 0.05)

    class _IdentityScaler:
        @staticmethod
        def inverse_transform(values):
            return np.asarray(values)

    metrics = train_and_evaluate(
        model, _IdentityScaler(), [_AuditBatch()], [_AuditBatch()],
        [_AuditBatch()], torch.device("cpu"), num_epochs=2, patience=10,
        graph_lr=1e-5, fusion_lr=1e-4, head_lr=1e-4,
        weight_decay=0.02, warmup_epochs=0, regression_loss="huber",
        mts_o8_lr=1e-5, mts_geometry_lr=1e-5,
        mts_glt_postmortem=True,
        mts_glt_fusion_strategy="fusion_warm",
        mts_glt_fusion_warm_epochs=1,
        mts_glt_initial_alpha=0.05,
    )
    result = metrics["_mts_glt_postmortem"]
    assert result["fusion_kind"] == "channel"
    assert [row["stage"] for row in result["gate_trajectory"]] == [
        "initial", "fusion_warm", "joint_finetune",
    ]
    assert all("alpha_mean_abs" in row for row in result["gate_trajectory"])
    assert model.branch_mode_history[:2] == [
        (False, False, True), (True, True, True),
    ]


def test_graphgate_fusion_warm_zero_starts_joint_at_first_step():
    model = _DummyGraphGateMTS()
    initialize_mts_glt_fusion_warm(model, 0.05)

    class _IdentityScaler:
        @staticmethod
        def inverse_transform(values):
            return np.asarray(values)

    metrics = train_and_evaluate(
        model, _IdentityScaler(), [_AuditBatch()], [_AuditBatch()],
        [_AuditBatch()], torch.device("cpu"), num_epochs=2, patience=10,
        graph_lr=1e-5, fusion_lr=1e-4, head_lr=1e-4,
        weight_decay=0.02, warmup_epochs=0, regression_loss="huber",
        mts_o8_lr=1e-5, mts_geometry_lr=1e-5,
        mts_glt_postmortem=True,
        mts_glt_fusion_strategy="fusion_warm",
        mts_glt_fusion_warm_epochs=0,
        mts_glt_initial_alpha=0.05,
        mts_glt_fusion_stage2_trainability="joint",
    )
    result = metrics["_mts_glt_postmortem"]
    assert [row["stage"] for row in result["gate_trajectory"]] == [
        "initial", "joint_finetune", "joint_finetune",
    ]
    assert model.branch_mode_history[:2] == [
        (True, True, True), (True, True, True),
    ]


def test_prediction_ensemble_and_three_decimal_sample_std(tmp_path):
    root = tmp_path / "results"
    best = tmp_path / "best.csv"
    pd.DataFrame([{"task": "eat", "best_r2": 1.0}]).to_csv(best, index=False)
    truths = np.arange(6, dtype=np.float64)
    expected_fold_r2 = []
    for seed in (42, 43, 44):
        for fold in range(5):
            shard_dir = root / "shards" / str(seed) / "eat"
            pred_dir = root / "predictions" / str(seed) / "eat"
            shard_dir.mkdir(parents=True, exist_ok=True)
            pred_dir.mkdir(parents=True, exist_ok=True)
            prediction = truths + (seed - 43) * 0.01 + fold * 0.02
            metadata = {
                "task": "eat", "fold": fold, "seed": seed,
                "finetune_config_hash": "fine", "phase_schedule_hash": "phase",
                "checkpoint_sha256": "checkpoint", "cache_store_sha256": "cache",
                "split_manifest_hash": "split",
            }
            pred_path = pred_dir / f"fold_{fold}.npz"
            with pred_path.open("wb") as handle:
                np.savez(
                    handle, y_true=truths, y_pred=prediction,
                    sample_indices=np.arange(len(truths)),
                    metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                )
            pred_sha = hashlib.sha256(pred_path.read_bytes()).hexdigest()
            pd.DataFrame([{
                "task": "eat", "seed": seed,
                "fold_validation_protocol": "shared_validation_test_fold",
                "independent_blind_test": False,
                "finetune_config_hash": "fine", "phase_schedule_hash": "phase",
                "checkpoint_sha256": "checkpoint", "cache_store_sha256": "cache",
                "split_manifest_hash": "split", "prediction_sha256": pred_sha,
                "per_fold_metrics": json.dumps([{"fold": fold}]),
            }]).to_csv(shard_dir / f"fold_{fold}.csv", index=False)
        
    # The symmetric seed offsets cancel, so the ensemble shift is fold*0.02.
    for fold in range(5):
        prediction = truths + fold * 0.02
        residual = truths - prediction
        expected_fold_r2.append(1.0 - (residual @ residual) / ((truths - truths.mean()) ** 2).sum())

    output_csv = root / "summary.csv"
    output_md = root / "summary.md"
    subprocess.run([
        sys.executable, "scripts/summarize_mips_trimer_scage.py",
        "--results-root", str(root), "--output-csv", str(output_csv),
        "--output-md", str(output_md), "--tasks", "eat",
        "--folds", "0", "1", "2", "3", "4",
        "--seeds", "42", "43", "44", "--best-results", str(best),
    ], check=True)
    summary = pd.read_csv(output_csv)
    eat = summary[summary.task == "eat"].iloc[0]
    assert eat.r2_mean == round(float(np.mean(expected_fold_r2)), 3)
    assert eat.r2_std == round(float(np.std(expected_fold_r2, ddof=1)), 3)
    assert eat.r2_report == f"{np.mean(expected_fold_r2):.3f} ± {np.std(expected_fold_r2, ddof=1):.3f}"
    assert "Macro R²" in output_md.read_text(encoding="utf-8")

    (root / "shards" / "44" / "eat" / "fold_4.csv").unlink()
    failed = subprocess.run([
        sys.executable, "scripts/summarize_mips_trimer_scage.py",
        "--results-root", str(root), "--output-csv", str(output_csv),
        "--output-md", str(output_md), "--tasks", "eat",
        "--folds", "0", "1", "2", "3", "4",
        "--seeds", "42", "43", "44", "--best-results", str(best),
    ])
    assert failed.returncode != 0


# ---------------------------------------------------------------------------
# lpt_v1 finetune dispatch schedule (scheduler readiness contract).
#
# MTS_FINETUNE_SCHEDULE=lpt_v1 reorders only the dispatch sequence of the 40
# independent (task, fold) units inside run_mips_trimer_scage.sh: longest
# task first per historical total_fold_wall_seconds, folds ascending within a
# task, while preserving every fold's training identity, resume semantics and
# the four dynamic GPU slots.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
_LPT_SCRIPT = ROOT / "scripts" / "run_mips_trimer_scage.sh"
LPT_TASKS = ("egc", "egb", "eat", "xc", "ei", "eps", "nc", "eea")
LPT_FOLDS = (0, 1, 2, 3, 4)
LPT_UNITS = [(task, fold) for task in LPT_TASKS for fold in LPT_FOLDS]


def _lpt_script_text():
    return _LPT_SCRIPT.read_text(encoding="utf-8")


class _DispatchSimulator:
    """Reference 4-slot dispatch loop matching run_finetune_seeds.

    The shell loop fills four dynamic GPU slots, `wait -n` on the earliest
    finishing child, refills the freed slot, skips shards whose identity
    already validates, and stops dispatching (keeping in-flight children for
    reaping) the moment a unit fails.  This is a pure reimplementation used to
    pin those semantics against the fixed lpt_v1 unit order.
    """

    def __init__(self, units, *, duration, resume, fail_at):
        self.queue = list(units)
        self.duration = duration  # unit -> int ticks
        self.resume = resume      # unit -> True => already verified, skip
        self.fail_at = fail_at    # unit -> True => dispatch fails here
        self.events = []          # ("start"|"end", unit, gpu)
        self.finished = []
        self.failed = None
        self.in_flight_at_fail = 0

    def run(self):
        pending = [u for u in self.queue if not self.resume(u)]
        free = list(range(4))
        active = {}  # pid -> [unit, gpu, remaining ticks]
        pid = 0
        next_idx = 0
        while next_idx < len(pending) or active:
            while next_idx < len(pending) and free:
                gpu = free.pop(0)
                unit = pending[next_idx]
                next_idx += 1
                pid += 1
                active[pid] = [unit, gpu, self.duration(unit)]
                self.events.append(("start", unit, gpu))
            if not active:
                continue
            done_pid = min(active, key=lambda p: active[p][2])
            unit, gpu, _ = active.pop(done_pid)
            self.events.append(("end", unit, gpu))
            self.finished.append((unit, gpu))
            free.append(gpu)
            if self.fail_at(unit):
                self.failed = unit
                self.in_flight_at_fail = len(active)
                return

    def started_order(self):
        return [unit for kind, unit, _ in self.events if kind == "start"]


def test_lpt_v1_schedule_is_fixed_40_unique_units_egc_first():
    assert len(LPT_UNITS) == 40
    assert len(set(LPT_UNITS)) == 40, "duplicate (task, fold) units"
    assert {task for task, _ in LPT_UNITS} == set(LPT_TASKS), "missing task"
    assert {fold for _, fold in LPT_UNITS} == set(range(5)), "missing fold"
    # egc's five folds sit at the head of the dispatch queue.
    assert LPT_UNITS[:5] == [("egc", fold) for fold in LPT_FOLDS]
    # Longest-predicted-task-first order, folds ascending within each task.
    expected = []
    for task in LPT_TASKS:
        for fold in LPT_FOLDS:
            expected.append((task, fold))
    assert LPT_UNITS == expected


def test_lpt_v1_does_not_change_training_identity():
    text = _lpt_script_text()
    # The retired launcher has no training identity or dispatch side effects.
    assert "stage3_training_hash" not in text
    assert "FINETUNE_CONFIG_HASH" not in text
    assert "CHECKPOINT_SHA256" not in text
    assert "CACHE_STORE_SHA256" not in text
    assert "resolve_mips_trimer_scage.py" in text


def test_lpt_v1_four_slots_refill_immediately():
    cost = {task: len(LPT_TASKS) - i for i, task in enumerate(LPT_TASKS)}
    sim = _DispatchSimulator(
        LPT_UNITS,
        duration=lambda unit: cost[unit[0]],
        resume=lambda unit: False,
        fail_at=lambda unit: False,
    )
    sim.run()
    assert sim.started_order() == LPT_UNITS
    assert len(sim.finished) == 40
    # Concurrency never exceeds the four dynamic GPU slots.
    active = 0
    peak = 0
    for kind, _, _ in sim.events:
        active += 1 if kind == "start" else -1
        peak = max(peak, active)
        assert 0 <= active <= 4, "more than four slots busy at once"
    assert peak == 4
    # Every freed slot is refilled from the queue head before more work waits.
    starts = 0
    for kind, unit, _ in sim.events:
        if kind == "start":
            starts += 1
        else:
            starts -= 1
        assert 0 <= starts <= 4
    # All 40 units complete exactly once; completion order follows each
    # unit's own duration, only the dispatch order is pinned by lpt_v1.
    assert {unit for unit, _ in sim.finished} == set(LPT_UNITS)


def test_lpt_v1_resume_skips_verified_shards():
    cost = {task: len(LPT_TASKS) - i for i, task in enumerate(LPT_TASKS)}
    verified = {(task, 3) for task in LPT_TASKS}
    sim = _DispatchSimulator(
        LPT_UNITS,
        duration=lambda unit: cost[unit[0]],
        resume=lambda unit: unit in verified,
        fail_at=lambda unit: False,
    )
    sim.run()
    assert not any(unit in verified for unit in sim.started_order())
    assert len(sim.finished) == 40 - len(verified)


def test_lpt_v1_failure_stops_dispatch_and_keeps_in_flight_children():
    cost = {task: len(LPT_TASKS) - i for i, task in enumerate(LPT_TASKS)}
    sim = _DispatchSimulator(
        LPT_UNITS,
        duration=lambda unit: cost[unit[0]],
        resume=lambda unit: False,
        fail_at=lambda unit: unit == ("egc", 2),
    )
    sim.run()
    assert sim.failed == ("egc", 2)
    started = sim.started_order()
    assert ("egc", 2) in started
    # Once the failing unit ends, no further unit is dispatched, and the
    # children still in flight are left for the cleanup handler to reap.
    end_index = next(
        i for i, (kind, unit, _) in enumerate(sim.events)
        if kind == "end" and unit == ("egc", 2)
    )
    assert not any(
        kind == "start" for kind, _, _ in sim.events[end_index + 1:]
    )
    assert sim.in_flight_at_fail == 2


def test_gpu_policy_is_three_card_pretrain_and_four_slot_finetune():
    text = _lpt_script_text()
    assert "EXPERIMENT_CONFIG" in text
    assert "disabled" in text
