"""Final-only tests for the r3 initialization and execution-policy changes."""
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pretrain_mcl_ph import load_initial_state, router_stop_required
from scripts.run_mcl_ph_opt import execution_paths


def test_complete_initial_state_and_schema_failure(tmp_path):
    model = nn.Linear(2, 1)
    path = tmp_path / 'step0.pt'
    state = {'weight': torch.tensor([[1., 2.]]), 'bias': torch.tensor([3.])}
    torch.save(state, path)
    load_initial_state(model, path)
    assert torch.equal(model(torch.tensor([[4., 5.]])), torch.tensor([[17.]]))
    torch.save({'weight': state['weight']}, path)
    with pytest.raises(RuntimeError, match='Missing key'):
        load_initial_state(model, path)


def test_low_util_observation_can_continue_without_changing_threshold():
    config = {'router_hard_min': .01, 'stop_on_router_low_util': False}
    assert not router_stop_required(config, [45, 0, 0])
    assert config['router_hard_min'] == .01
    assert router_stop_required(dict(config, stop_on_router_low_util=True), [3, 0, 0])
    assert not router_stop_required(dict(config, stop_on_router_low_util=True), [2, 0, 0])


def test_r3_cannot_use_old_execution_roots_or_omit_initial_state():
    out = ROOT / 'results/mcl_ph_opt_20261006/r3'
    log = ROOT / 'logs/mcl_ph_opt_20261006/r3'
    initial = ROOT / 'results/mcl_ph_opt_20261006/R2/pretrain/M0/initial_model.pt'
    assert execution_paths('r3', out, log, initial) == (out, log, initial)
    with pytest.raises(ValueError):
        execution_paths('r3', out.parent, log, initial)
    with pytest.raises(ValueError):
        execution_paths('r3', out, log, None)
