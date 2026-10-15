"""Final-stage checks for the two changes in MCL-PH-OPT; no hash checks."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.modules.mcl_ph_pretrain import MCLPHPretrainer
from src.training.sota_ft_contract import recipe, reference, key, check_scope
from src.training.sota_ft_runtime import reserve, account, read


class LengthDecoder(nn.Module):
    def lengths(self, states, bonds):
        return states[bonds[:, 0], 0] + states[bonds[:, 1], 0]


@pytest.mark.parametrize('mode', ['expert', 'expert_mixed_50'])
def test_local_closed_form_and_router_gradient(mode):
    # Exercise the production local objective; bypass unrelated O8/PH work.
    model = MCLPHPretrainer.__new__(MCLPHPretrainer)
    nn.Module.__init__(model)
    model.local_objective = mode
    model.local_decoder = LengthDecoder()
    states = [torch.full((2, 128), value, requires_grad=True) for value in (.5, 1., 2.)]
    logits = torch.tensor([-.7, .4, 1.2], requires_grad=True)
    dense = logits.softmax(0)
    keep = logits.topk(2).indices
    mask = torch.zeros_like(dense).scatter(0, keep, 1.)
    alpha = dense * mask / (dense * mask).sum()
    mixed = sum(weight * state for weight, state in zip(alpha, states))
    atoms = torch.zeros((2, 512), requires_grad=True)
    labels = dict(mcl_length_pair=torch.tensor([[0, 1]]), mcl_angle_pair=torch.empty((0, 2), dtype=torch.long),
                  mcl_length_target=torch.zeros(1), mcl_angle_target=torch.empty(0),
                  mcl_length_graph=torch.zeros(1, dtype=torch.long), mcl_angle_graph=torch.empty(0, dtype=torch.long),
                  mcl_nonbond_pair=torch.empty((0, 2), dtype=torch.long))
    encoded = dict(expert_states=states, mixed=mixed, atom_states=atoms)
    local, nonbond, valid = model._geometry_terms(
        SimpleNamespace(mcl_bond_index=torch.tensor([[0, 1]])), labels, encoded, 2)
    expert_loss = (0.375 + 0.875 + 1.875) / 3
    prediction = float(sum(a * (2 * v) for a, v in zip(alpha.detach(), (.5, 1., 2.))))
    mixed_loss = .5 * prediction - .125
    expected = expert_loss if mode == 'expert' else .5 * expert_loss + .5 * mixed_loss
    assert local['value'].tolist() == pytest.approx([expected, 0.], abs=1e-6)
    assert valid.tolist() == [True, False]
    assert not nonbond['valid'].any() and not nonbond['value'].any()
    gradients = torch.autograd.grad(local['value'].sum(), [*states, logits, atoms], allow_unused=True)
    assert all(g is not None and bool(g.abs().sum() > 0) for g in gradients[:3])
    assert gradients[-1] is None  # local supervision does not enter O8/XATTN.
    if mode == 'expert':
        assert gradients[3] is None
    else:
        assert gradients[3] is not None and bool(gradients[3].abs().sum() > 0)


def test_r1_recipe_scope_and_unique_output():
    fold = 2
    r = recipe('development', 'nc', 0, 'SP')
    r.update(plan='MCL-PH-OPT-20261006-01', revision='r2', fold=fold, seed=44,
             head_initial=str(reference('nc', fold) / 'head_initial.pt'), K=0,
             ensemble='disabled; single best only')
    k = key('development', 'nc', fold, 'SP')
    scope = dict(plan=r['plan'], status='AUTHORIZED_R1', units={k:dict(recipe=r)})
    folder = ROOT / 'results/mcl_ph_opt_20261006/R1' / k
    assert check_scope(scope, 'development', 'nc', fold, 'SP', folder)['recipe']['alpha'] == .001
    with pytest.raises(ValueError):
        check_scope(scope, 'development', 'nc', fold, 'SP', folder / 'other')
    with pytest.raises(ValueError):
        check_scope(scope, 'development', 'nc', fold, 'F0', folder)


def test_r1_budget_cannot_expand_or_restart(tmp_path):
    ledger = tmp_path / 'budget.json'
    for fold in (2, 3, 4):
        folder = tmp_path / str(fold)
        folder.mkdir()
        reserve(folder, 'development', str(fold), 70, 770, ledger, caps=(3, 210, 2310))
    with pytest.raises(ValueError, match='already reserved'):
        reserve(tmp_path, 'development', '2', 70, 770, ledger, caps=(3, 210, 2310))
    with pytest.raises(ValueError, match='exceeded'):
        reserve(tmp_path, 'development', '5', 70, 770, ledger, caps=(3, 210, 2310))
    account('2', 'attempted_updates', ledger=ledger, total_cap=2310)
    assert read(ledger)['units']['2']['attempted_updates'] == 1
