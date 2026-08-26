from pathlib import Path

import torch

from src.modules.mts_glt_v2 import Atom3DConditionalUpdate
from src.training.finetune.engine import (
    _atomic_save_best_checkpoint,
    select_mts_glt_graph_state,
)


def _block(seed=7):
    torch.manual_seed(seed)
    return Atom3DConditionalUpdate(hidden_size=8)


def test_s3_x23_modules_are_parameter_matched_and_identically_initialized():
    s3 = _block()
    x23 = _block()
    assert sum(p.numel() for p in s3.parameters()) == sum(
        p.numel() for p in x23.parameters()
    )
    for left, right in zip(s3.state_dict().values(), x23.state_dict().values()):
        assert torch.equal(left, right)


def test_zero_gate_is_exact_identity_and_invalid_is_always_identity():
    block = _block()
    h2 = torch.randn(4, 8)
    h3 = torch.randn(4, 8)
    valid = torch.tensor([True, True, False, False])
    updated, delta, _, _ = block(h2, h3, valid)
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(updated, h3)
    with torch.no_grad():
        block.gate.fill_(0.3)
    updated, delta, _, _ = block(h2, h3, valid)
    assert torch.equal(delta[~valid], torch.zeros_like(delta[~valid]))
    assert torch.equal(updated[~valid], h3[~valid])


def test_condition_is_detached_but_geometry_content_has_gradient():
    block = _block()
    with torch.no_grad():
        block.gate.fill_(0.3)
    h2 = torch.randn(3, 8, requires_grad=True)
    h3 = torch.randn(3, 8, requires_grad=True)
    updated, delta, condition, content = block(
        h2, h3, torch.ones(3, dtype=torch.bool)
    )
    assert not torch.equal(condition, block(h3.detach(), h3.detach(), torch.ones(3, dtype=torch.bool))[2])
    assert torch.isfinite(delta).all() and torch.isfinite(content).all()
    updated.sum().backward()
    assert h2.grad is None
    assert h3.grad is not None
    assert torch.isfinite(h3.grad).all()
    assert float(h3.grad.abs().sum()) > 0


def test_checkpoint_mapping_allows_only_downstream_interaction_parameters():
    prefix = 'encoders.graph.encoder.'
    model_state = {
        prefix + 'o8.weight': torch.ones(2),
        prefix + 'interaction_update.gate': torch.zeros(2),
    }
    checkpoint = {'model.o8.weight': torch.ones(2)}
    mapped = select_mts_glt_graph_state(model_state, checkpoint)
    assert set(mapped) == {prefix + 'o8.weight'}
    try:
        select_mts_glt_graph_state(model_state, {})
    except RuntimeError as exc:
        assert 'o8.weight' in str(exc)
    else:
        raise AssertionError('missing shared pretrained tensor was accepted')


def test_atomic_best_checkpoint_contains_exact_state(tmp_path: Path):
    model = torch.nn.Linear(3, 2)
    target = tmp_path / 'best.pth'
    _atomic_save_best_checkpoint(model, target, {'best_epoch': 4})
    payload = torch.load(target, map_location='cpu')
    assert payload['best_epoch'] == 4
    for key, value in model.state_dict().items():
        assert torch.equal(payload['state_dict'][key], value)

