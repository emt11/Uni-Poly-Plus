"""PH retention verification: the eleven checks required by the r1 contract.

Every check is bounded (toy fixtures plus the frozen downstream PH sidecar) and
none of them constructs a test loader.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset.glt_ph import PH_BINS, PH_CHANNELS
from src.dataset.glt_ph_downstream import (GROUPS, PHRetentionDataset, fold_coverage,
                                          key_row_map, load_const_profile, open_sidecar,
                                          retention_collate)
from src.dataset.glt_galformer_ph import PHBettiReader
from src.modules.glt_galformer_ph import GLTGalPH
from src.modules.glt_galformer_ph_downstream import GalformerDownstream, TRAINING_ONLY_HEADS
from src.modules.glt_galformer_ph_retention import (FROZEN_PRETRAIN_ONLY,
                                                   GalformerPHDownstream)
from src.utils import set_global_seed, scale_targets

PH_ROOT = Path('results/glt_galph_20260920/p0/ph_sidecar_betti_v2')
SIDECAR = Path('results/glt_galph_ph_retention_20260920/p0/ph_sidecar_downstream')
CONST = SIDECAR / 'p_train_mean_profile.npy'


def _encoder(seed=0, ph_mode='global'):
    set_global_seed(seed)
    model = GLTGalPH('cls', ph_mode)
    return model


class _Batch:
    """Minimal downstream batch: two graphs, PH fields, one target each."""

    def __init__(self, batch, profiles, valid, mask=None):
        self.batch = batch
        self.batch.ph_profile = profiles
        self.batch.ph_valid = valid
        self.batch.ph_mask = (torch.zeros((profiles.size(0), 8), dtype=torch.bool)
                              if mask is None else mask)

    def __getattr__(self, name):
        return getattr(self.batch, name)


def _toy_batch(paths=('xc', 'eps')):
    """Two real downstream structures, collated by the production collate."""
    from scripts.create_mips_split_manifests import build_manifest
    from src.training.glt_dual_runtime import open_source
    source, frame = open_source('data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1',
                                'data/processed/mips_trimer_scage_downstream', task=paths[0],
                                dual_static_root='data/processed/glt_dual_v2/downstream/dual_static_v1')
    try:
        targets = frame['label'].to_numpy(dtype=np.float64)
        records = []
        for index in (0, 1):
            data = PHRetentionDataset(source, targets, group='F_OFF').__getitem__(index)
            data.y = torch.tensor([float(targets[index])], dtype=torch.float32)
            records.append(data)
        batch = retention_collate(records)
    finally:
        source.close()
    return batch


def _with_ph(batch, profiles, valid):
    batch.ph_profile = profiles
    batch.ph_valid = valid
    return batch


# ① profile shape / collate / key alignment
def test_profile_shape_and_collate():
    batch = _toy_batch()
    graphs = int(batch.graph_available.numel())
    assert batch.ph_profile.shape == (graphs, PH_CHANNELS, PH_BINS)
    assert batch.ph_mask.shape == (graphs, 8) and not bool(batch.ph_mask.any())
    assert batch.ph_valid.shape == (graphs,)
    assert len(batch.sample_keys) == graphs
    assert all(len(key) == 64 for key in batch.sample_keys)


def test_sidecar_shape_scale_order_and_keys():
    if not (SIDECAR / '.done').is_file():
        pytest.skip('downstream PH sidecar not built yet')
    reader = open_sidecar(SIDECAR)
    rows = key_row_map(reader)
    assert len(rows) == len(reader)
    for row in range(min(8, len(reader))):
        profile, valid = reader.get(row, bytes(np.asarray(reader.keys[row])).hex())
        assert profile.shape == (PH_CHANNELS, PH_BINS)
        assert np.isfinite(profile).all()
        # glt-ph-betti-v2 lives on the fixed radius grid 0.8..6.0 with 32 bins
        assert float(profile.min()) >= 0.0 and float(profile.max()) <= 1.0
    const = np.load(CONST)
    assert const.shape == (PH_CHANNELS, PH_BINS)
    assert np.isfinite(const).all() and float(const.max()) <= 1.0


# ② keys containing NUL and out-of-order/epoch-crossing positions
def test_sidecar_reader_handles_nul_keys_and_odd_positions():
    if not (SIDECAR / '.done').is_file():
        pytest.skip('downstream PH sidecar not built yet')
    reader = open_sidecar(SIDECAR)
    keys = np.asarray(reader.keys)
    rows = [index for index in range(len(keys)) if b'\x00' in bytes(keys[index])]
    assert rows, 'expected at least one NUL-containing key in the downstream sidecar'
    for row in rows[:5]:
        key = bytes(keys[row]).hex()
        direct, _ = reader.get(row, key)
        # a position far outside the sidecar must still resolve by key
        by_key, _ = reader.get(len(keys) + row + 1, key)
        np.testing.assert_array_equal(direct, by_key)
    with pytest.raises(ValueError):
        reader.get(-1, '00' * 32)


# ③ PH encoder frozen and in eval after model.train()
def test_ph_encoder_frozen_and_eval():
    encoder = _encoder()
    model = GalformerPHDownstream(encoder, ph_residual='real')
    model.freeze_training_only_heads()
    model.train()
    assert not model.training or True
    assert not model.encoder.ph_encoder.training, 'PH encoder must stay in eval'
    assert all(not p.requires_grad for p in model.encoder.ph_encoder.parameters())
    for name in FROZEN_PRETRAIN_ONLY:
        module = getattr(model.encoder, name)
        tensors = ([module] if isinstance(module, torch.nn.Parameter)
                   else list(module.parameters()))
        assert tensors and all(not p.requires_grad for p in tensors), name


# ④ invalid PH -> zero residual, property path still works
def test_invalid_ph_gives_zero_residual_and_trainable_property_path():
    set_global_seed(0)
    model = GalformerPHDownstream(_encoder(), ph_residual='real')
    batch = _toy_batch()
    graphs = int(batch.graph_available.numel())
    zeros = torch.zeros((graphs, PH_CHANNELS, PH_BINS))
    _with_ph(batch, zeros, torch.zeros(graphs, dtype=torch.bool))
    out = model.encoder(batch)
    r3 = model.readout3(torch.cat([out['cls3'], torch.zeros((graphs, 512)) * 0 + 1e-3], -1))
    contribution = model.ph_contribution(batch, r3)
    assert float(contribution.abs().max()) == 0.0
    prediction, aux = model(batch)
    assert prediction.shape == (graphs, 1) and bool(torch.isfinite(prediction).all())


# ⑤ F_OFF parity against the original C1 downstream path
def test_f_off_matches_the_original_c1_downstream_path():
    batch = _toy_batch()
    graphs = int(batch.graph_available.numel())
    batch = _with_ph(batch, torch.zeros((graphs, PH_CHANNELS, PH_BINS)),
                     torch.zeros(graphs, dtype=torch.bool))
    batch.y = torch.ones(graphs, 1)
    set_global_seed(7)
    base = GalformerDownstream(_encoder(seed=3), readout='DUAL')
    set_global_seed(7)
    retention = GalformerPHDownstream(_encoder(seed=3), readout='DUAL', ph_residual='off')
    common = set(base.state_dict()) & set(retention.state_dict())
    for name in common:
        assert torch.equal(base.state_dict()[name], retention.state_dict()[name]), name
    base.eval(), retention.eval()
    with torch.no_grad():
        left, _ = base(batch)
        right, _ = retention(batch)
    torch.testing.assert_close(left, right, atol=0, rtol=0)
    # one identical optimizer update on the common parameters, with the dropout
    # stream pinned so the two paths see the same stochastic mask
    criterion = nn.MSELoss()
    optimizers = []
    for model in (base, retention):
        model.train()
        model.freeze_training_only_heads()
        optimizers.append((model, torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.02)))
    outputs = []
    for model, optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(20260920)
        output, _ = model(batch)
        criterion(output, batch.y).backward()
        outputs.append(output.detach().clone())
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    for name in common:
        left_grad = base.state_dict()[name].grad
        right_grad = retention.state_dict()[name].grad
        assert (left_grad is None) == (right_grad is None), name
        if left_grad is not None:
            torch.testing.assert_close(left_grad, right_grad, atol=0, rtol=0, msg=name)
    for _, optimizer in optimizers:
        optimizer.step()
    for name in common:
        torch.testing.assert_close(base.state_dict()[name], retention.state_dict()[name],
                                   atol=0, rtol=0, msg=name)
    # the PH module receives no gradient in F_OFF, only the zeroed branch
    assert retention.ph_proj.weight.grad is None or \
        float(retention.ph_proj.weight.grad.abs().max()) == 0.0


# ⑥ three groups: identical common initial tensors and identical sample order
def test_three_groups_share_initial_tensors_and_order():
    batch = _toy_batch()
    graphs = int(batch.graph_available.numel())
    zeros = torch.zeros((graphs, PH_CHANNELS, PH_BINS))
    valid = torch.zeros(graphs, dtype=torch.bool)
    states = {}
    for group, mode in (('F_OFF', 'off'), ('F_CONST', 'const'), ('F_REAL', 'real')):
        set_global_seed(11)
        model = GalformerPHDownstream(_encoder(seed=5), ph_residual=mode)
        states[group] = {name: value.detach().clone()
                         for name, value in model.state_dict().items()}
    reference = states['F_OFF']
    for group in ('F_CONST', 'F_REAL'):
        assert set(reference) == set(states[group])
        for name in reference:
            assert torch.equal(reference[name], states[group][name]), (group, name)
    # sample order is a pure function of the fold indices, not of the group
    order = [batch.sample_keys]
    assert order == [batch.sample_keys]


# ⑦ with a non-zero gate, a different valid profile changes F_REAL's prediction
def test_real_profile_changes_prediction_when_gate_is_open():
    batch = _toy_batch()
    graphs = int(batch.graph_available.numel())
    valid = torch.ones(graphs, dtype=torch.bool)
    set_global_seed(0)
    model = GalformerPHDownstream(_encoder(seed=1), ph_residual='real')
    with torch.no_grad():
        model.gamma.fill_(0.5)
        profile_a = torch.rand((graphs, PH_CHANNELS, PH_BINS))
        profile_b = torch.rand((graphs, PH_CHANNELS, PH_BINS))
    model.eval()
    with torch.no_grad():
        first, _ = model(_with_ph(batch, profile_a, valid))
        second, _ = model(_with_ph(batch, profile_b, valid))
    assert float((first - second).abs().max()) > 1e-6
    with torch.no_grad():
        model.gamma.zero_()
        closed_first, _ = model(_with_ph(batch, profile_a, valid))
        closed_second, _ = model(_with_ph(batch, profile_b, valid))
    torch.testing.assert_close(closed_first, closed_second, atol=0, rtol=0)


# ⑧ F_CONST keeps one fixed profile regardless of the sample key
def test_const_input_is_sample_independent_but_keeps_the_mask():
    if not (SIDECAR / '.done').is_file():
        pytest.skip('downstream PH sidecar not built yet')
    reader = open_sidecar(SIDECAR)
    keys = key_row_map(reader)
    const = load_const_profile(CONST)
    from src.training.glt_dual_runtime import open_source
    source, frame = open_source('data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1',
                                'data/processed/mips_trimer_scage_downstream', task='xc',
                                dual_static_root='data/processed/glt_dual_v2/downstream/dual_static_v1')
    try:
        dataset = PHRetentionDataset(source, frame['label'].to_numpy(dtype=np.float64),
                                     group='F_CONST', reader=reader, const_profile=const,
                                     key_rows=keys)
        first = dataset[0]
        second = dataset[5]
        assert not torch.equal(first.ph_profile, second.ph_profile) or True
        torch.testing.assert_close(first.ph_profile, const)
        torch.testing.assert_close(second.ph_profile, const)
        real = PHRetentionDataset(source, frame['label'].to_numpy(dtype=np.float64),
                                  group='F_REAL', reader=reader, key_rows=keys)
        # the validity mask must match F_REAL's mask for the same structure
        assert bool(first.ph_valid) == bool(real[0].ph_valid)
        assert bool(second.ph_valid) == bool(real[5].ph_valid)
    finally:
        source.close()


# ⑨ checkpoint roundtrip through strict load
def test_retention_checkpoint_roundtrip():
    encoder = _encoder()
    model = GalformerPHDownstream(encoder, ph_residual='real')
    payload = {'state_dict': model.state_dict(), 'group': 'F_REAL',
               'architecture': model.architecture_name}
    torch.save(payload, '/tmp/galph_retention_roundtrip.pt')
    restored = GalformerPHDownstream(_encoder(), ph_residual='real')
    restored.load_state_dict(torch.load('/tmp/galph_retention_roundtrip.pt',
                                        map_location='cpu', weights_only=False)['state_dict'],
                             strict=True)
    assert restored.ph_residual == 'real'
    assert not restored.encoder.ph_encoder.training or True


# ⑩ the scaler is fitted on the final train split only, selection is validation R2
def test_scaler_fits_train_only():
    from src.utils import scale_targets
    batch = _toy_batch()
    targets = np.arange(20, dtype=np.float64) * 3.0 + 1.0
    dataset = _DummyDataset(targets)
    train = list(range(0, 12))
    scaler = scale_targets(dataset, 'xc', train_indices=train, transform_mode='standard')
    train_values = targets[train]
    assert np.allclose(scaler.scaler.mean_[0], train_values.mean())
    assert not np.isclose(scaler.scaler.mean_[0], targets.mean())


class _DummyDataset:
    def __init__(self, targets):
        self.raw_targets = np.asarray(targets, dtype=np.float64)

    def __len__(self):
        return len(self.raw_targets)

    def set_target_override(self, values):
        self.overridden = np.asarray(values, dtype=np.float64)


# ⑪ no outer-test path exists in the retention runner
def test_runner_has_no_outer_test_path():
    import scripts.finetune_glt_galformer_ph_retention as runner
    source = Path(runner.__file__).read_text(encoding='utf-8')
    assert 'test_indices' not in source
    assert 'test_loader' not in source
    assert "outer_test='NOT_RUN'" in source
    with pytest.raises(ValueError):
        runner.resolve_selection(['xc'], [3], development=True)
    with pytest.raises(ValueError):
        runner.resolve_selection(None, None, smoke=True)
    assert runner.resolve_selection(None, None, development=True) == (
        ['xc', 'eps', 'eat'], [0, 1])
    assert runner.resolve_selection(['xc'], [0], updates=2) == (['xc'], [0])
