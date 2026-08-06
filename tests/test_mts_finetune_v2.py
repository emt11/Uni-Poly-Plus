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
    _configure_legacy_mts_trainability,
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


def _trainable(module):
    return any(parameter.requires_grad for parameter in module.parameters())


def test_mts_legacy_profile_trains_complete_graph_from_epoch_zero():
    model = _DummyMTS()
    graph = model.encoders["graph"].encoder

    _configure_legacy_mts_trainability(model)
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
        mts_finetune_profile="legacy_mts_huber_v1",
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
