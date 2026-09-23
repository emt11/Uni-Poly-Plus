"""Targeted, training-free checks for the r10R3 preflight repairs."""

import json
from types import SimpleNamespace

import pytest
import torch

from scripts import pretrain_mcl_ph
from scripts.finetune_mcl_ph import O8OnlyArm
from src.dataset.mcl_ph_view import clean_nonbond_distances


def test_clean_nonbond_pairs_use_heavy_view_indices_after_an_explicit_hydrogen():
    # Source indices 0, 2, 3 are heavy; source index 1 is an explicit H.
    trimer = SimpleNamespace(trimer_pos=torch.tensor([
        [0.0, 0.0, 0.0], [99.0, 0.0, 0.0],
        [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))
    result = clean_nonbond_distances(trimer, torch.tensor([0, 2, 3]),
                                     torch.tensor([[1, 2], [0, 1]]))
    assert result.tolist() == [3.0, 2.0]


def test_checkpoint_cadence_includes_each_milestone_once_and_final_smoke_step():
    due = [step for step in range(1, 5001)
           if pretrain_mcl_ph.checkpoint_due(step, 5000, 1000)]
    assert due == [1000, 2000, 3000, 4000, 5000]
    assert pretrain_mcl_ph.checkpoint_due(2, 2, 1000)
    with pytest.raises(ValueError, match='save_every'):
        pretrain_mcl_ph.checkpoint_due(1, 5000, 0)


def test_each_rank_enters_the_rng_gather_before_rank_zero_writes(monkeypatch):
    calls = []
    monkeypatch.setattr(pretrain_mcl_ph, 'rng_state', lambda: {'rank': len(calls)})

    def gather(states, local):
        calls.append(local)
        states[:] = [{'rank': rank} for rank in range(4)]

    monkeypatch.setattr(pretrain_mcl_ph.dist, 'all_gather_object', gather)
    for rank in range(4):
        states = pretrain_mcl_ph.checkpoint_rng_states(4)
        assert len(states) == 4
        assert states[rank] == {'rank': rank}
    assert len(calls) == 4


def test_an_incomplete_checkpoint_does_not_promote_runtime_to_pass(tmp_path, monkeypatch):
    runtime = tmp_path / 'runtime.json'
    runtime.write_text(json.dumps({'status': 'TRAINING_COMPLETE',
                                   'cleanup': 'pending', 'main_returned': False}))
    monkeypatch.setattr(pretrain_mcl_ph, '_OUTPUT', [tmp_path])
    monkeypatch.setenv('RANK', '0')
    pretrain_mcl_ph.finalize_runtime_record()
    assert json.loads(runtime.read_text())['status'] == 'TRAINING_COMPLETE'


def test_o8_only_uses_the_same_single_head_as_the_new_arms():
    arm = O8OnlyArm()
    assert list(dict(arm.named_children())) == ['o8', 'head']
    assert arm.head.norm.normalized_shape == (512,)
