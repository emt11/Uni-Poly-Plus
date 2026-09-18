"""Bounded S2 target/decoder checks; fixtures are synthetic and not science results."""

import torch

from test_complete_trimer_glt import _toy_pair
from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.glt_dual_pretrain import prepare_pretrain_sample, pretrain_collate
from src.modules.glt_dual_pretrain import DualPretrainer, global_objective


def _row(key='row', position=0, task='none'):
    smiles = '*COC*'
    topology = build_canonical_periodic_topology(smiles)
    _, trimer = _toy_pair(smiles)
    return prepare_pretrain_sample(
        topology, trimer, smiles, seed=42, key=key, position=position,
        third_task=task, fgr_max_pairs=32,
    )


def test_fp_free_objectives_only_build_common_three_ru(monkeypatch):
    import src.dataset.glt_dual_pretrain as module

    actual = module.build_periodic_multimer_mol
    repeats = []

    def watched(*args, **kwargs):
        repeats.append(int(args[1]))
        return actual(*args, **kwargs)

    monkeypatch.setattr(module, 'build_periodic_multimer_mol', watched)
    _row(task='none')
    assert repeats == [3]


def test_fgr_pair_sampling_is_deterministic_bounded_and_clean_labelled():
    first = _row('fgr-key', 7, 'fgr')
    second = _row('fgr-key', 7, 'fgr')
    pair_index, target, spd = first[1]['fgr_pair_index'], first[1]['fgr_target'], first[1]['fgr_spd']
    assert pair_index.shape[0] == target.numel() == spd.numel() <= 32
    assert pair_index.numel() == 0 or torch.all(pair_index[:, 0] != pair_index[:, 1])
    assert set(spd.tolist()).issubset({2, 3})
    assert torch.equal(pair_index, second[1]['fgr_pair_index'])
    torch.testing.assert_close(target, second[1]['fgr_target'])


def test_fgr_collate_offsets_and_empty_geometry():
    first = _row('fgr-a', 0, 'fgr')
    second = _row('fgr-b', 1, 'fgr')
    data, labels = pretrain_collate([first, second])
    n = first[1]['fgr_pair_index'].size(0)
    offset = first[0].mips_x.size(0)
    if n:
        assert torch.equal(labels['fgr_pair_index'][:n], first[1]['fgr_pair_index'])
        assert torch.equal(labels['fgr_pair_index'][n:], second[1]['fgr_pair_index'] + offset)
    smiles = '*COC*'
    topology = build_canonical_periodic_topology(smiles)
    _, trimer = _toy_pair(smiles)
    trimer.trimer_geometry_valid = False
    invalid = prepare_pretrain_sample(topology, trimer, smiles, seed=42,
                                      key='invalid', position=2, third_task='fgr')
    _, invalid_labels = pretrain_collate([invalid])
    assert invalid_labels['fgr_pair_index'].shape == (0, 2)
    assert not bool(invalid_labels['fgr_graph_valid'][0])


def test_none_fgr_and_align_have_safe_sums_and_expected_gradients():
    none = _row('none', 0, 'none')
    data, labels = pretrain_collate([none])
    model = DualPretrainer('concat', third_task='none')
    assert model.fp_head is None
    result = model(data, labels)
    assert result['counts'][2].item() == 0
    global_objective(result['sums'], result['counts']).backward()
    assert all(torch.isfinite(parameter.grad).all()
               for parameter in model.parameters() if parameter.grad is not None)

    fgr = _row('fgr', 0, 'fgr')
    data, labels = pretrain_collate([fgr])
    model = DualPretrainer('concat', third_task='fgr')
    result = model(data, labels)
    assert torch.isfinite(result['sums']).all()
    global_objective(result['sums'], result['counts']).backward()
    assert model.fgr_head.mlp[0].weight.grad is not None
    assert torch.isfinite(model.fgr_head.mlp[0].weight.grad).all()

    rows = [_row('align-a', 0, 'align'), _row('align-b', 1, 'align')]
    data, labels = pretrain_collate(rows)
    model = DualPretrainer('concat', third_task='align')
    result = model(data, labels)
    assert result['counts'][2].item() == 2
    assert torch.isfinite(result['sums']).all()
    global_objective(result['sums'], result['counts']).backward()
    for projection in (model.align_proj2, model.align_proj3):
        assert projection.net[0].weight.grad is not None
        assert torch.isfinite(projection.net[0].weight.grad).all()
