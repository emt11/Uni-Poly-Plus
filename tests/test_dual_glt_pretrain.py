"""Atom/geometry checks only. Synthetic coordinates are not real fixtures."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
from rdkit import Chem

from tests.test_dual_glt import _toy_pair
from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.glt_dual import build_dual_sample, dual_glt_collate
from src.dataset.glt_dual_pretrain import (chemical_groups, motif_mask, sample_generator,
                                         prepare_pretrain_sample, pretrain_collate)
from src.dataset.graph_data import build_periodic_multimer_mol
from src.modules.glt_dual import build_dual_glt_model
from src.modules.glt_dual_pretrain import DualPretrainer, per_graph, global_objective, deployment_package, load_deployment
from src.training.glt_dual_runtime import rng_state, restore_rng, scheduled_lr, OrderedSampleStream
from src.utils import scale_targets
from scripts.finetune_glt_dual import fixed_manifest


def record(smiles='*COC*', position=0, **kwargs):
    top = build_canonical_periodic_topology(smiles)
    _, tri = _toy_pair(smiles)
    return prepare_pretrain_sample(top, tri, smiles, seed=42, key=smiles, position=position, **kwargs)


def test_motif_partition_fallback_single_atom_and_replay():
    for smiles in ('*COC*', '*CC(=O)NCC*', '*C*'):
        top = build_canonical_periodic_topology(smiles)
        groups = chemical_groups(smiles)
        assert sorted(i for group in groups for i in group) == list(range(top.mips_x.size(0)))
        a, fallback = motif_mask(top, groups, sample_generator(42, smiles, 7))
        b, _ = motif_mask(top, groups, sample_generator(42, smiles, 7))
        assert torch.equal(a, b)
        assert int(a.sum()) < a.numel()
        if a.numel() > 1:
            assert a.any()
        else:
            assert not a.any() and not fallback
    top = build_canonical_periodic_topology('*COC*')
    mask, fallback = motif_mask(top, [(0, 1, 2)], sample_generator(42, 'x', 0))
    assert fallback and int(mask.sum()) == 1


def test_noise_all_features_match_one_coordinate_draw_and_targets_are_clean():
    top = build_canonical_periodic_topology('*COC*')
    _, tri = _toy_pair('*COC*')
    original = tri.trimer_pos.clone()
    noisy, labels = prepare_pretrain_sample(top, tri, '*COC*', seed=42, key='x', position=3)
    generator = sample_generator(42, 'x', 3)
    motif_mask(top, chemical_groups('*COC*'), generator)
    changed = copy.copy(tri)
    changed.trimer_pos = original + .03 * torch.randn(original.shape, generator=generator)
    expected = build_dual_sample(top, changed, '*COC*')
    clean = build_dual_sample(top, tri, '*COC*')
    torch.testing.assert_close(noisy.bond_distance, expected.bond_distance)
    torch.testing.assert_close(noisy.line_angle, expected.line_angle)
    torch.testing.assert_close(labels['distance'], clean.bond_distance[clean.bond_center])
    assert torch.equal(tri.trimer_pos, original)
    assert labels['angle_pairs'].unique(dim=0).size(0) == labels['angle_pairs'].size(0)
    assert clean.bond_center[labels['angle_pairs']].all()
    assert (labels['angle_pairs'][:, 0] < labels['angle_pairs'][:, 1]).all()
    replay = prepare_pretrain_sample(top, tri, '*COC*', seed=42, key='x', position=3)
    assert torch.equal(replay[0].bond_distance, noisy.bond_distance)
    # Isotropic noise is invariant in distribution; rotate the SAME draw for
    # a pointwise invariant check, not a newly generated lab-frame draw.
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    changed.trimer_pos = changed.trimer_pos @ rotation + 5
    rotated = build_dual_sample(top, changed, '*COC*')
    torch.testing.assert_close(rotated.bond_distance, noisy.bond_distance, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rotated.line_angle, noisy.line_angle, atol=1e-4, rtol=1e-4)


def test_atom_mask_target_uses_only_three_ru_and_has_no_fp(monkeypatch):
    import src.dataset.glt_dual_pretrain as module
    actual_builder = module.build_periodic_multimer_mol
    repeats = []
    def watched(*args, **kwargs):
        repeats.append(args[1])
        assert kwargs['close_periodic'] is False
        return actual_builder(*args, **kwargs)
    monkeypatch.setattr(module, 'build_periodic_multimer_mol', watched)
    chemical_groups.cache_clear()
    assert chemical_groups('*C*') == ((0,),)
    assert repeats == [3]
    from src.dataset.glt_dual_static import build_pretrain_target
    assert set(build_pretrain_target('*C*')) == {'brics_groups'}


def test_per_graph_and_ddp_gradient_reference():
    values = torch.tensor([1., 3., 8.], requires_grad=True)
    means, valid = per_graph(values, torch.tensor([0, 0, 2]), 4)
    torch.testing.assert_close(means, torch.tensor([2., 0., 8., 0.]))
    assert valid.tolist() == [True, False, True, False]
    # Unequal valid counts per rank: average DDP gradients, not local means.
    weight = torch.tensor(2., requires_grad=True)
    counts = torch.tensor([3., 1., 4.])
    a = global_objective(weight * torch.tensor([2., 0., 1.]), counts, 2)
    b = global_objective(weight * torch.tensor([5., 4., 3.]), counts, 2)
    ddp = torch.autograd.grad((a + b) / 2, weight)[0]
    torch.testing.assert_close(ddp, torch.tensor(7/3 + 4 + .1))


@pytest.mark.parametrize('mode', ['concat', 'kfuse'])
def test_atom_geometry_gradients_loading_and_mask_leakage(mode):
    rows = [record(), record('*C*')]
    data, labels = pretrain_collate(rows)
    model = DualPretrainer(mode).eval()
    result = model(data, labels)
    assert result['counts'].tolist() == [1, 1, 0]
    assert 'fingerprint' not in labels
    assert not hasattr(model, 'fp_head')
    assert not any(name.startswith('fp_head.') for name in model.state_dict())
    loss = global_objective(result['sums'], result['counts'])
    chem_grad = torch.autograd.grad(result['sums'][0], model.encoder.glt.endpoint.weight,
                                    allow_unused=True, retain_graph=True)[0]
    assert chem_grad is None or chem_grad.abs().sum() == 0
    loss.backward()
    for name in ('encoder.o8.atom_embedding.projection.weight', 'encoder.glt.endpoint.weight',
                 'atom_head.head.weight', 'length_head.2.weight', 'angle_head.2.weight'):
        grad = model.get_parameter(name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    before = model.encoder.o8(data, labels['atom_mask'])
    changed = copy.deepcopy(data)
    changed.mips_x[labels['atom_mask']] = 123.
    changed.mips_backbone_mask[labels['atom_mask']] = ~changed.mips_backbone_mask[labels['atom_mask']].bool()
    after = model.encoder.o8(changed, labels['atom_mask'])
    for a, b in zip(before, after):
        torch.testing.assert_close(a, b)
    package = deployment_package(model, 5000)
    assert not any('head' in name or 'predictor' in name for name in package['state_dict'] if not name.startswith('glt.angle_bias.'))
    downstream = build_dual_glt_model(mode).eval()
    original_head = copy.deepcopy(downstream.predictor.state_dict())
    load_deployment(downstream, package)
    for key, value in original_head.items():
        torch.testing.assert_close(value, downstream.predictor.state_dict()[key])
    encoded = model.encoder.encode(data)
    torch.testing.assert_close(model.encoder.fuse(encoded), downstream.fuse(downstream.encode(data)))
    bad = dict(package, fusion_mode='kfuse' if mode == 'concat' else 'concat')
    with pytest.raises(ValueError, match='mismatch'):
        load_deployment(downstream, bad)


def test_empty_geometry_and_batch_target_offsets():
    first, second = record(), record()
    top = build_canonical_periodic_topology('*COC*')
    _, tri = _toy_pair('*COC*')
    tri.trimer_geometry_valid = False
    invalid = prepare_pretrain_sample(top, tri, '*COC*', seed=1, key='x', position=0)
    data, labels = pretrain_collate([first, second, invalid])
    n = first[1]['angle_pairs'].size(0)
    torch.testing.assert_close(labels['angle_pairs'][n:2*n], second[1]['angle_pairs'] + first[0].bond_distance.numel())
    assert data.geometry_valid.tolist() == [True, True, False]
    result = DualPretrainer()(*pretrain_collate([invalid]))
    assert result['counts'].tolist() == [1, 0, 0]
    assert torch.isfinite(result['sums']).all()


def test_removed_fp_cannot_be_selected_and_old_common_is_initialization_only(tmp_path):
    with pytest.raises(ValueError,match='unsupported third'):
        DualPretrainer(third_task='fp')
    with pytest.raises(ValueError,match='unsupported third'):
        record(third_task='fp')
    from src.training.glt_dual_runtime import apply_common_initialization
    model=DualPretrainer()
    state={k:v.clone() for k,v in model.state_dict().items()}
    old=dict(state)
    for name in ('fp_head.0.weight','fp_head.0.bias','fp_head.3.weight','fp_head.3.bias'):
        old[name]=torch.ones(1)
    path=tmp_path/'legacy_common.pt'
    torch.save({'common_state_dict':old},path)
    info=apply_common_initialization(model,path,direct=True)
    assert set(info['tensor_names'])==set(state)
    assert info['fixed_seed_unshared_tensor_names']==[]
    for name,value in state.items():torch.testing.assert_close(value,model.state_dict()[name])
    # Only initialization projects out the retired head; resume is strict.
    with pytest.raises(RuntimeError,match='Unexpected key'):
        model.load_state_dict(old,strict=True)


@pytest.mark.parametrize('third_task',['align','fgr'])
def test_other_optional_objectives_remain_trainable_without_fp(third_task):
    data,labels=pretrain_collate([record('*COC*',third_task=third_task),record('*CCN*',third_task=third_task)])
    model=DualPretrainer(third_task=third_task).eval()
    result=model(data,labels)
    assert 'fingerprint' not in labels and not hasattr(model,'fp_head')
    assert result['counts'][2]>0
    global_objective(result['sums'],result['counts']).backward()
    head=model.align_proj2.net[0] if third_task=='align' else model.fgr_head.mlp[0]
    assert head.weight.grad is not None and head.weight.grad.abs().sum()>0


def test_rng_and_schedule_resume():
    state = rng_state()
    expected = torch.rand(4)
    restore_rng(state)
    torch.testing.assert_close(expected, torch.rand(4))
    cfg = dict(lr=2e-4, warmup_steps=2000, schedule_total_steps=20000, end_lr=1e-9)
    assert scheduled_lr(4999, **cfg) > 0
    assert scheduled_lr(2000, **cfg) == cfg['lr']
    stream = OrderedSampleStream(7, 42)
    reference = [stream.index_at(p) for p in range(20)]
    resumed = OrderedSampleStream(7, 42)
    assert [resumed.index_at(p) for p in range(9, 20)] == reference[9:]
    assert sorted(reference[:7]) == list(range(7))


def test_validation_selects_weights_and_test_runs_once(monkeypatch):
    import scripts.finetune_glt_dual as runtime
    encoder = torch.nn.Linear(1, 1, bias=False)
    wrapper = runtime.EvaluationAdapter(encoder)
    scores, calls = iter([.4, .8, .5]), []
    def train(*args, epoch, **kwargs):
        with torch.no_grad():
            encoder.weight.fill_(epoch)
        calls.append(('train', epoch))
    def validate(*args, **kwargs):
        calls.append(('validation', int(encoder.weight.item())))
        return 1., next(scores), None, None
    def test(*args, **kwargs):
        calls.append(('test', int(encoder.weight.item())))
        assert encoder.weight.item() == 2
        return {'test_r2': .1}
    monkeypatch.setattr(runtime, 'train_epoch', train)
    monkeypatch.setattr(runtime, 'evaluate', validate)
    monkeypatch.setattr(runtime, 'test_model', test)
    best, score, epoch, result = runtime.fit_select_and_test(wrapper, None, None, None, None,
        'cpu', None, None, {'epochs': 5, 'patience': 1}, task='eat', fold_id=0)
    assert score == .8 and epoch == 2 and best['weight'].item() == 2
    assert calls == [('train', 1), ('validation', 1), ('train', 2), ('validation', 2),
                     ('train', 3), ('validation', 3), ('test', 2)]


def test_optimizer_and_rng_roundtrip(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    def update():
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.randn(3, 2)).square().mean()
        loss.backward()
        optimizer.step()
        return loss.detach().clone()
    update()
    path = tmp_path / 'resume.pt'
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state()), path)
    expected_loss = update()
    expected = copy.deepcopy(model.state_dict())
    state = torch.load(path, weights_only=False)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'])
    torch.testing.assert_close(update(), expected_loss)
    for name, value in expected.items():
        torch.testing.assert_close(model.state_dict()[name], value)


def test_fixed_300_split_and_scaler(tmp_path):
    csv = tmp_path / 'smi_eat.csv'
    pd.DataFrame({'smiles': [f'sample{i}' for i in range(300)], 'y': np.arange(300)}).to_csv(csv, index=False)
    path = tmp_path / 'eat.json'
    manifest = fixed_manifest('eat', csv, path)
    visits = np.zeros(300, dtype=int)
    class Targets:
        raw_targets = np.arange(300, dtype=float)
        def set_target_override(self, targets):
            self.targets = targets
    for fold in manifest['folds']:
        tr, va, te = [fold[f'{s}_indices'] for s in ('train', 'validation', 'test')]
        assert [len(tr), len(va), len(te)] == [192, 48, 60]
        assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
        visits[te] += 1
        scaler = scale_targets(Targets(), 'eat', train_indices=tr, transform_mode='standard')
        assert scaler.scaler.mean_[0] == np.mean(np.asarray(tr))
    assert (visits == 1).all()
    assert fixed_manifest('eat', csv, path) == manifest
    manifest['folds'][0]['validation_indices'] = manifest['folds'][0]['test_indices']
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='differs'):
        fixed_manifest('eat', csv, path)
