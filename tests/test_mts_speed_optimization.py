import subprocess
import sys
import runpy
from types import SimpleNamespace
from pathlib import Path

import pytest
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_mts_pretrain import (
    _passes_gate as pretrain_candidate_passes,
    _parse_worker_sweep,
    _select_candidate as select_pretrain_candidate,
)
from scripts.benchmark_mts_finetune import (
    _select_eval_batch,
    _steady_training_metrics,
)
from src.utils import evaluate, test_model as run_test_model
from src.modules.mips_local_graph import MD200GraphResidual
from src.dataset.dataloader import mips_trimer_collate


class _Batch:
    def __init__(self, x, y):
        self.x = torch.as_tensor(x, dtype=torch.float32).reshape(-1, 1)
        self.y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)

    def to(self, device, non_blocking=False):
        self.x = self.x.to(device, non_blocking=non_blocking)
        self.y = self.y.to(device, non_blocking=non_blocking)
        return self


class _LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        with torch.no_grad():
            self.linear.weight.fill_(2.0)
            self.linear.bias.fill_(0.5)

    def forward(self, batch):
        output = self.linear(batch.x)
        return output, output


class _IdentityScaler:
    @staticmethod
    def inverse_transform(values):
        return np.asarray(values)


def test_launcher_is_explicitly_fail_closed_without_configuration():
    text = (ROOT / "scripts/run_mips_trimer_scage.sh").read_text(
        encoding="utf-8"
    )
    assert "EXPERIMENT_CONFIG" in text
    assert "resolve_mips_trimer_scage.py" in text


def test_pretrain_benchmark_requires_new_configuration(tmp_path):
    output = tmp_path / "benchmark.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_mts_pretrain.py",
            "--dry-run",
            "--batches",
            "500",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert completed.returncode != 0, completed.stdout
    assert "No active MTS configuration" in completed.stdout
    assert not output.exists()


def test_worker_sweep_parser_rejects_negative_and_duplicate_values():
    assert _parse_worker_sweep("8, 4,6") == (8, 4, 6)
    with pytest.raises(SystemExit, match="non-negative"):
        _parse_worker_sweep("4,-1,8")
    with pytest.raises(SystemExit, match="unique"):
        _parse_worker_sweep("4,6,4")


def test_worker_sweep_requires_new_configuration(tmp_path):
    output = tmp_path / "worker_sweep.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_mts_pretrain.py",
            "--worker-sweep",
            "4,6,8",
            "--dry-run",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert completed.returncode != 0, completed.stdout
    assert "No active MTS configuration" in completed.stdout
    assert not output.exists()


def test_pretrain_candidate_gate_requires_finite_memory_wait_and_shm():
    candidate = {
        "returncode": 0,
        "loss_finite": True,
        "gradient_finite": True,
        "parameters_finite": True,
        "peak_memory_fraction": 0.5,
        "samples_per_second": 100.0,
        "rank_wait_fraction": 0.1,
        "shared_memory_minimum_available_bytes": 1024,
    }
    assert pretrain_candidate_passes(candidate)
    for key in ("loss_finite", "gradient_finite", "parameters_finite"):
        rejected = dict(candidate, **{key: False})
        assert not pretrain_candidate_passes(rejected)
    assert not pretrain_candidate_passes(
        dict(candidate, shared_memory_minimum_available_bytes=0)
    )


def test_pretrain_two_percent_tie_prefers_lower_shared_memory():
    common = {
        "peak_memory_fraction": 0.1,
        "shared_memory_available_before_bytes": 1000,
        "loader_workers_requested": 4,
    }
    lower_resource = dict(
        common,
        samples_per_second=100.0,
        shared_memory_minimum_available_bytes=900,
        loader_prefetch_factor_requested=2,
    )
    slightly_faster = dict(
        common,
        samples_per_second=101.5,
        shared_memory_minimum_available_bytes=700,
        loader_prefetch_factor_requested=4,
    )
    assert select_pretrain_candidate([lower_resource, slightly_faster]) is lower_resource


def test_eval_batch_size_and_epoch_sync_do_not_change_fp32_predictions(monkeypatch):
    model = _LinearModel()
    values = np.arange(12, dtype=np.float32)
    targets = values * 2.0 + 0.5
    small = [_Batch(values[i:i + 3], targets[i:i + 3]) for i in range(0, 12, 3)]
    large = [_Batch(values[i:i + 6], targets[i:i + 6]) for i in range(0, 12, 6)]
    criterion = nn.MSELoss()
    small_eval = evaluate(model, small, criterion, "cpu")
    large_eval = evaluate(model, large, criterion, "cpu")
    np.testing.assert_allclose(small_eval[3], large_eval[3], rtol=0, atol=1e-5)
    monkeypatch.setenv("MTS_BENCHMARK_LEGACY_SYNC", "1")
    legacy_eval = evaluate(model, small, criterion, "cpu")
    np.testing.assert_allclose(small_eval[3], legacy_eval[3], rtol=0, atol=1e-5)
    monkeypatch.delenv("MTS_BENCHMARK_LEGACY_SYNC")
    small_test = run_test_model(
        model, small, _IdentityScaler(), "cpu", return_predictions=True
    )
    large_test = run_test_model(
        model, large, _IdentityScaler(), "cpu", return_predictions=True
    )
    np.testing.assert_allclose(
        small_test["_y_pred"], large_test["_y_pred"], rtol=0, atol=1e-5
    )


def test_finetune_payload_includes_precision_and_eval_batch_without_hash_identity():
    train_text = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    assert "'eval_batch_size': int(args.eval_batch_size)" in train_text
    assert "'amp_dtype': args.amp_dtype" in train_text
    assert "prediction_sha256" not in train_text


def test_mts_collate_fields_and_sample_order_match_with_two_workers():
    helpers = runpy.run_path(str(ROOT / "tests" / "test_mips_non_pbc.py"))
    make_graph = helpers["graph_data"]
    records = [make_graph("*CCO*"), make_graph("*CCCCCCC*")]
    records[0].mts_sample_hash64 = torch.tensor(11, dtype=torch.int64)
    records[1].mts_sample_hash64 = torch.tensor(22, dtype=torch.int64)

    def load(workers):
        loader = DataLoader(
            records,
            batch_size=2,
            shuffle=False,
            num_workers=workers,
            persistent_workers=workers > 0,
            prefetch_factor=2 if workers > 0 else None,
            collate_fn=mips_trimer_collate,
        )
        return next(iter(loader))

    direct = load(0)
    worker = load(2)
    assert set(direct.keys()) == set(worker.keys())
    assert direct.mts_sample_hash64.tolist() == [11, 22]
    assert worker.mts_sample_hash64.tolist() == [11, 22]
    for key in direct.keys():
        left = getattr(direct, key)
        right = getattr(worker, key)
        if torch.is_tensor(left):
            assert torch.equal(left, right), key


def test_finetune_benchmark_extracts_steady_step_rate_and_bf16_gate(tmp_path):
    for task in ("egc", "xc"):
        (tmp_path / f"finetune_seed42_{task}_fold0.log").write_text(
            "Training: 100%|x| 10/10 [00:05<00:00, 2.00it/s]\n"
            "Training: 100%|x| 10/10 [00:04<00:00, 2.50it/s]\n"
            "Fine-tune BF16 parity gate: "
            "{'relative_loss_delta': 0.01, 'gradient_cosine': 0.99, "
            "'finite': True}, pass=True\n",
            encoding="utf-8",
        )
    metrics = _steady_training_metrics(tmp_path, "egc xc")
    assert metrics["steady_training_steps"] == 40
    assert metrics["steady_training_seconds"] == 18
    assert metrics["steady_optimizer_steps_per_second"] == pytest.approx(40 / 18)
    assert all(item["passed"] for item in metrics["bf16_parity_gates"].values())


def test_md200_residual_casts_autocast_output_to_graph_dtype():
    module = MD200GraphResidual(dim=4, dropout=0.0)
    graph = torch.zeros((2, 4), dtype=torch.bfloat16)
    data = SimpleNamespace(
        mips_md=torch.randn((2, 200), dtype=torch.float32),
        mips_md_valid=torch.tensor([True, False]),
    )
    output = module(graph, data)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output.float()).all()


def test_eval_batch_selector_chooses_fastest_safe_batch_not_largest():
    def task(seconds):
        return {"eval_seconds": seconds}

    selected, evidence = _select_eval_batch({
        "64": {
            "passed": True,
            "tasks": {
                "egc": task(0.8), "egb": task(0.7),
                "eat": task(0.8), "xc": task(0.7),
            },
        },
        "128": {
            "passed": True,
            "tasks": {
                "egc": task(1.0), "egb": task(0.9),
                "eat": task(0.9), "xc": task(0.8),
            },
        },
        "256": {
            "passed": True,
            "tasks": {
                "egc": task(1.1), "egb": task(1.0),
                "eat": task(1.0), "xc": task(0.9),
            },
        },
    })
    assert selected == 64
    assert evidence["candidate_totals"]["256"] > evidence["candidate_totals"]["64"]
    assert evidence["tie_policy"].startswith("none")
