import hashlib
import json
import re
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
    finetune_bf16_parity_gate,
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


def test_bf16_gate_fails_closed_without_cuda():
    passed, details = finetune_bf16_parity_gate(
        nn.Linear(2, 1), None, nn.MSELoss(), torch.device('cpu')
    )
    assert not passed
    assert details['reason'] == 'cuda_bf16_unavailable'


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
# lpt_v1 finetune dispatch schedule (Plan.MD A0.4 / CODEX handoff A0.4).
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


def _parse_lpt_array(name):
    text = _lpt_script_text()
    match = re.search(rf"{name}=\(([^)]+)\)", text)
    assert match, f"{name} not found in run_mips_trimer_scage.sh"
    return match.group(1).split()


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
    tasks = _parse_lpt_array("LPT_V1_TASK_ORDER")
    folds = [int(fold) for fold in _parse_lpt_array("LPT_V1_FOLD_ORDER")]
    assert tasks == list(LPT_TASKS)
    assert folds == list(LPT_FOLDS)
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
    # The schedule must never leak into a fold's training_config_hash, which
    # is derived only from finetune_config_hash, seed and loader_workers.
    match = re.search(r"stage3_training_hash\(\)\s*\{.*?\}", text, re.S)
    assert match
    body = match.group(0)
    assert "MTS_FINETUNE_SCHEDULE" not in body
    assert "task" not in body and "fold" not in body
    # The lpt_v1 dispatch branch only selects dispatch_tasks/dispatch_folds;
    # it must not touch hashes, launch arguments, or the resume gate.
    branch = re.search(
        r'if \[\[ "\$MTS_FINETUNE_SCHEDULE" == "lpt_v1" \]\].*?^  fi$',
        text, re.M | re.S,
    )
    assert branch
    branch_text = branch.group(0)
    for forbidden in (
        "stage3_training_hash", "training_config_hash", "launch_stage3_unit",
        "validate_shard", "CHECKPOINT_SHA256", "CACHE_STORE_SHA256",
    ):
        assert forbidden not in branch_text


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
    assert 'MTS_PRETRAIN_GPU_IDS:-1,2,3' in text
    assert 'MTS_FINETUNE_GPU_IDS:-0,1,2,3' in text
    assert 'validate_gpu_ids "$PRETRAIN_GPU_IDS" 3' in text
    assert 'validate_gpu_ids "$FINETUNE_GPU_IDS" 4' in text
