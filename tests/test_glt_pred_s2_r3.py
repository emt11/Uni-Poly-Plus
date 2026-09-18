"""S2 r3 semantic references for true-Trimer FGR and ALIGN."""

import copy
import inspect

import torch
from rdkit import Chem
from torch.nn import functional as F

from test_complete_trimer_glt import _toy_pair
from src.dataset.canonical_periodic import build_canonical_periodic_topology, resolve_normalized_identity
from src.dataset.graph_data import build_periodic_multimer_mol
from src.dataset.glt_dual_pretrain import fgr_pairs, prepare_pretrain_sample
from src.modules.glt_dual_pretrain import AlignmentProjection, FGRDecoder, alignment_loss


def _fgr_fixture(smiles='*CCCC*'):
    topology = build_canonical_periodic_topology(smiles)
    _, trimer = _toy_pair(smiles)
    identity = resolve_normalized_identity(topology, smiles, require_fields=True)
    return topology, trimer, identity


def _true_trimer_reference(topology, identity):
    molecule, metadata = build_periodic_multimer_mol(
        identity['normalized_smiles'], 3, close_periodic=False
    )
    central = metadata['unit_atoms'][1]
    inverse = {
        int(base): canonical
        for canonical, base in enumerate(identity['canonical_to_normalized_base'].tolist())
    }
    expected = {}
    for left_offset, left in enumerate(central):
        if molecule.GetAtomWithIdx(int(left)).GetAtomicNum() <= 1:
            continue
        for right in central[left_offset + 1:]:
            if molecule.GetAtomWithIdx(int(right)).GetAtomicNum() <= 1:
                continue
            hop = len(Chem.GetShortestPath(molecule, int(left), int(right))) - 1
            if hop in (2, 3):
                pair = tuple(sorted((inverse[left_offset], inverse[central.index(right)])))
                expected[pair] = hop
    return expected


def test_fgr_uses_independent_true_trimer_shortest_paths_and_clean_labels():
    topology, trimer, identity = _fgr_fixture()
    pair_index, target, spd = fgr_pairs(
        topology, trimer, '*CCCC*', identity=identity,
        seed=42, key='r3-fgr', position=11, max_pairs=32,
    )
    expected = _true_trimer_reference(topology, identity)
    observed = {tuple(pair.tolist()): int(hop) for pair, hop in zip(pair_index, spd)}
    assert observed == expected
    assert pair_index.unique(dim=0).size(0) == pair_index.size(0)
    assert bool((pair_index[:, 0] < pair_index[:, 1]).all())
    central = trimer.mips_to_trimer_central_index
    distances = torch.linalg.vector_norm(
        trimer.trimer_pos[central[pair_index[:, 0]]] -
        trimer.trimer_pos[central[pair_index[:, 1]]], dim=-1
    )
    torch.testing.assert_close(target, torch.log1p(distances))
    assert bool(torch.isfinite(target).all())


def test_fgr_ignores_lifted_duplicates_and_rejects_bad_mapping():
    topology, trimer, identity = _fgr_fixture()
    baseline = fgr_pairs(
        topology, trimer, '*CCCC*', identity=identity,
        seed=7, key='r3-duplicate', position=3,
    )
    altered = copy.copy(topology)
    altered.lga_edge_index = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)
    altered.lga_spd = torch.tensor([3, 3], dtype=torch.long)
    changed = fgr_pairs(
        altered, trimer, '*CCCC*', identity=identity,
        seed=7, key='r3-duplicate', position=3,
    )
    for left, right in zip(baseline, changed):
        torch.testing.assert_close(left, right)
    broken = copy.copy(topology)
    broken.canonical_to_trimer_base_atom_id = torch.zeros_like(
        topology.canonical_to_trimer_base_atom_id
    )
    try:
        fgr_pairs(broken, trimer, '*CCCC*', identity=identity,
                  seed=7, key='r3-broken', position=0)
    except ValueError as exc:
        assert 'mapping' in str(exc).lower()
    else:
        raise AssertionError('bad canonical mapping was accepted')


def test_fgr_invalid_geometry_is_empty_but_sample_is_retained():
    topology, trimer, _ = _fgr_fixture('*CCCC*')
    trimer.trimer_geometry_valid = False
    noisy, labels = prepare_pretrain_sample(
        topology, trimer, '*CCCC*', seed=42, key='r3-invalid', position=0,
        third_task='fgr',
    )
    assert noisy is not None
    assert labels['fgr_pair_index'].shape == (0, 2)
    assert labels['fgr_target'].numel() == 0
    assert labels['fgr_valid'] is False


def test_fgr_decoder_has_no_clean_coordinate_or_raw_target_input():
    parameters = inspect.signature(FGRDecoder.forward).parameters
    assert list(parameters) == ['self', 'left', 'right', 'context']
    assert not any('coord' in name or 'target' in name for name in parameters)


def _alignment_reference(z2, z3, identities, valid, temperature=0.1):
    valid = valid.bool()
    distinct = {identity for identity, keep in zip(identities, valid.tolist()) if keep}
    if len(distinct) < 2 or not bool(valid.any()):
        return z2.sum() * 0 + z3.sum() * 0, torch.tensor(0., device=z2.device)
    valid_columns = valid.unsqueeze(0)
    logits23 = z2 @ z3.t() / temperature
    logits32 = z3 @ z2.t() / temperature
    positive = torch.tensor(
        [[left == right and keep for right, keep in zip(identities, valid.tolist())]
         for left in identities], device=z2.device, dtype=torch.bool
    )
    anchor = valid & torch.tensor(
        [identity in distinct for identity in identities], device=z2.device
    )
    def directional(logits):
        denominator = torch.logsumexp(logits.masked_fill(~valid_columns, -torch.inf), dim=1)
        count = positive.sum(1).clamp_min(1).float()
        return -(logits.masked_fill(~positive, 0).sum(1) / count - denominator)
    return (directional(logits23)[anchor] + directional(logits32)[anchor]).sum() / 2, anchor.sum().float()


def test_alignment_math_reference_multi_positive_and_invalid_padding_exclusion():
    torch.manual_seed(123)
    raw2 = torch.randn(3, 128, requires_grad=True)
    raw3 = torch.randn(3, 128, requires_grad=True)
    z2, z3 = F.normalize(raw2, dim=-1), F.normalize(raw3, dim=-1)
    identities = ['A', 'A', 'B']
    valid = torch.tensor([True, True, True])
    actual_sum, actual_count = alignment_loss(z2, z3, identities, valid)
    expected_sum, expected_count = _alignment_reference(z2, z3, identities, valid)
    torch.testing.assert_close(actual_sum, expected_sum)
    torch.testing.assert_close(actual_count, expected_count)
    actual_sum.backward()
    assert raw2.grad is not None and raw3.grad is not None
    assert torch.isfinite(raw2.grad).all() and torch.isfinite(raw3.grad).all()
    assert raw2.grad.abs().sum() > 0 and raw3.grad.abs().sum() > 0

    base2 = torch.randn(3, 128)
    base3 = torch.randn(3, 128)
    reference = None
    for padding in (torch.zeros(128), torch.full((128,), 1e4), torch.randn(128)):
        trial2 = torch.cat([base2[:2], padding.unsqueeze(0)]).requires_grad_()
        trial3 = torch.cat([base3[:2], padding.unsqueeze(0)]).requires_grad_()
        trial2 = F.normalize(trial2, dim=-1)
        trial3 = F.normalize(trial3, dim=-1)
        value, count = alignment_loss(
            trial2, trial3, ['A', 'B', 'padding'], torch.tensor([True, True, False])
        )
        if reference is None:
            reference = (value.detach(), count.detach())
        else:
            torch.testing.assert_close(value, reference[0])
            torch.testing.assert_close(count, reference[1])


def test_alignment_all_same_identity_and_single_valid_identity_are_safe_zero():
    for valid, identities in (
        (torch.tensor([True, True, True]), ['A', 'A', 'A']),
        (torch.tensor([True, False, False]), ['A', 'B', 'C']),
    ):
        raw2 = torch.randn(3, 128, requires_grad=True)
        raw3 = torch.randn(3, 128, requires_grad=True)
        value, count = alignment_loss(
            F.normalize(raw2, dim=-1), F.normalize(raw3, dim=-1), identities, valid
        )
        assert value.item() == 0.0
        assert count.item() == 0.0
        value.backward()
        assert torch.isfinite(raw2.grad).all() and torch.isfinite(raw3.grad).all()


def test_alignment_projection_gradients_are_finite():
    torch.manual_seed(9)
    projection2, projection3 = AlignmentProjection(), AlignmentProjection()
    input2, input3 = torch.randn(3, 512), torch.randn(3, 512)
    z2, z3 = projection2(input2), projection3(input3)
    value, count = alignment_loss(z2, z3, ['A', 'A', 'B'], torch.ones(3, dtype=torch.bool))
    assert count.item() == 3
    value.backward()
    for module in (projection2, projection3):
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
