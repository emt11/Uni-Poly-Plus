"""r5 wiring checks: the new C1_REPAIR_5K deployment and the three retention arms.

The r1/r2 suite already covers the data layer, the F_OFF gradient parity and the
aggregator's numeric checks.  This file adds what only the r5 contract needs:

* the committed identity record describes the checkpoint that is actually on
  disk, and any other checkpoint (in particular the previous cycle's degenerate
  C1 deployment) is refused;
* the real deployment loads strictly into the retention encoder and the frozen
  PH encoder still carries observable sample discrimination on the fixed probe
  set;
* the three arms built from that deployment start from bit-identical tensors;
* with a non-zero gate, F_REAL's property path really consumes the PH profile;
* the frozen encoder stays in eval after ``model.train()``;
* the invalid path gives a zero residual, and a saved unit reloads with the
  identity intact.
"""
import json

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.create_mips_split_manifests import build_manifest
from src.dataset.glt_ph_downstream import PHRetentionDataset, retention_collate
from src.modules.glt_galformer_ph import GLTGalPH, PH_CHANNELS, PH_BINS
from src.modules.glt_galformer_ph_downstream import GalformerDownstream
from src.modules.glt_galformer_ph_pretrain import load_galformer_deployment
from src.modules.glt_galformer_ph_retention import (FROZEN_PRETRAIN_ONLY,
                                                   GalformerPHDownstream)
from src.modules.glt_galph_checkpoint_identity import (DEFAULT_IDENTITY, file_sha256,
                                                       frozen_checkpoint_sha256,
                                                       load_identity, verify_checkpoint)
from src.training.glt_dual_runtime import open_source
from src.utils import set_global_seed

IDENTITY = load_identity(DEFAULT_IDENTITY)
CHECKPOINT = Path(IDENTITY['checkpoint'])
OLD_C1 = Path('results/glt_galph_20260920/p2/c1/pretrain/deploy_05000.pt')
DOWNSTREAM = ('data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1',
              'data/processed/mips_trimer_scage_downstream',
              'data/processed/glt_dual_v2/downstream/dual_static_v1')
PROBE = Path('results/glt_galph_ph_retention_20260920/p1/ph_probe_profiles.npy')


def _package():
    if not CHECKPOINT.is_file():
        pytest.skip(f'the C1_REPAIR_5K deployment is not present: {CHECKPOINT}')
    return torch.load(CHECKPOINT, map_location='cpu', weights_only=False)


def _encoder(package):
    encoder = GLTGalPH(str(package['summary_mode']), package.get('ph_mode'))
    load_galformer_deployment(encoder, package, int(IDENTITY['step']))
    return encoder


def _retention(package, mode):
    set_global_seed(11)
    return GalformerPHDownstream(_encoder(package), readout='DUAL', ph_residual=mode)


def _toy_batch(task='xc'):
    source, frame = open_source(DOWNSTREAM[0], DOWNSTREAM[1], task=task,
                                dual_static_root=DOWNSTREAM[2])
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


# ① the record describes the file on disk, field by field
def test_identity_record_matches_the_checkpoint_on_disk():
    if not CHECKPOINT.is_file():
        pytest.skip('deployment absent')
    package = _package()
    verified = verify_checkpoint(IDENTITY, CHECKPOINT, package)
    assert verified['sha256'] == IDENTITY['sha256']
    assert verified['step'] == 5000 and IDENTITY['step'] == 5000
    assert (verified['summary_mode'], verified['ph_mode']) == ('cls', 'global')
    assert verified['ph_encoder_version'] == 'scale-interaction-v2'
    assert CHECKPOINT.stat().st_size == int(IDENTITY['bytes'])
    assert file_sha256(CHECKPOINT) == IDENTITY['sha256']
    assert frozen_checkpoint_sha256() == IDENTITY['sha256']


def test_encoder_state_claims_belong_to_the_recorded_deployment():
    assert IDENTITY['encoder_state']['probe_rows'] == 64
    assert IDENTITY['encoder_state']['summary_spread'] > 0.1
    assert IDENTITY['encoder_state']['real_vs_const_max_abs'] > 0.1
    # the superseded deployment is named explicitly as *not* this round's control
    assert 'b709389895554e924884df22bd98f2b7b51cb7b54c199ebc92b3e95417fc2817' in \
        IDENTITY['supersedes']
    assert IDENTITY['sha256'] not in IDENTITY['supersedes']


# ② the previous cycle's checkpoint cannot be passed as this round's trunk
def test_wrong_checkpoint_identity_is_refused(tmp_path):
    if not CHECKPOINT.is_file():
        pytest.skip('deployment absent')
    package = _package()
    if OLD_C1.is_file():
        with pytest.raises(ValueError, match='identity record'):
            verify_checkpoint(IDENTITY, OLD_C1, package)
    tampered = json.loads(Path(DEFAULT_IDENTITY).read_text(encoding='utf-8'))
    tampered['sha256'] = '0' * 64
    path = tmp_path / 'tampered.json'
    path.write_text(json.dumps(tampered), encoding='utf-8')
    with pytest.raises(ValueError, match='identity record'):
        verify_checkpoint(load_identity(path), CHECKPOINT, package)
    # a package whose own metadata disagrees is refused even with the right sha
    broken = dict(package, ph_encoder_version='scale-interaction-v1')
    with pytest.raises(ValueError, match='ph_encoder_version'):
        verify_checkpoint(IDENTITY, CHECKPOINT, broken)
    assert load_identity(DEFAULT_IDENTITY)['sha256'] == IDENTITY['sha256']


# ③ the real deployment loads strictly and the frozen encoder still discriminates
def test_frozen_encoder_discriminates_on_fixed_probe_set():
    if not (CHECKPOINT.is_file() and PROBE.is_file()):
        pytest.skip('deployment or probe set absent')
    package = _package()
    encoder = _encoder(package)
    probe = torch.as_tensor(np.load(PROBE), dtype=torch.float32)
    assert probe.shape[0] == IDENTITY['encoder_state']['probe_rows']
    # the recorded real-vs-const number was measured against the P_train mean
    # profile, not against the probe set's own mean
    const = np.load(IDENTITY['encoder_state']['const_profile'])
    const = torch.as_tensor(const, dtype=torch.float32).unsqueeze(0).expand_as(probe)
    encoder.eval()
    with torch.no_grad():
        mask = torch.zeros((probe.size(0), 8), dtype=torch.bool)
        summary = encoder.ph_encoder.summarize(
            encoder.ph_encoder(probe, mask)).float()
        fixed = encoder.ph_encoder.summarize(
            encoder.ph_encoder(const.contiguous(), mask)).float()
    spread = float((summary.max(0).values - summary.min(0).values).max())
    observed = float((summary - fixed).abs().max())
    assert spread > 1e-3, 'the frozen encoder collapsed again: no cross-sample spread'
    assert observed > 1e-3, 'the frozen encoder no longer separates real from const'
    tolerance = float(IDENTITY['encoder_state']['tolerance']['bf16'])
    assert abs(spread - float(IDENTITY['encoder_state']['summary_spread'])) <= tolerance
    assert abs(observed - float(IDENTITY['encoder_state']['real_vs_const_max_abs'])) <= tolerance


# ④ the three arms start from bit-identical tensors
def test_three_arms_share_the_deployment_initial_tensors():
    if not CHECKPOINT.is_file():
        pytest.skip('deployment absent')
    package = _package()
    states = {group: {name: value.detach().clone()
                      for name, value in _retention(package, mode).state_dict().items()}
              for group, mode in (('F_OFF', 'off'), ('F_CONST', 'const'), ('F_REAL', 'real'))}
    reference = states['F_OFF']
    for group in ('F_CONST', 'F_REAL'):
        assert set(reference) == set(states[group])
        for name in reference:
            assert torch.equal(reference[name], states[group][name]), (group, name)
    for group in states:
        assert 'ph_proj.weight' in states[group], 'the new projection must be instantiated'
        assert float(states[group]['gamma']) == 0.0, 'gamma must start at zero'


# ⑤ with a non-zero gate, the property path consumes the PH profile
def test_real_arm_consumes_ph_when_gate_is_open():
    if not CHECKPOINT.is_file():
        pytest.skip('deployment absent')
    batch = _toy_batch()
    graphs = int(batch.graph_available.numel())
    valid = torch.ones(graphs, dtype=torch.bool)
    model = _retention(_package(), 'real')
    model.freeze_training_only_heads()
    model.eval()
    with torch.no_grad():
        model.gamma.fill_(0.5)
        generator = torch.Generator().manual_seed(3)
        first_profile = torch.rand((graphs, PH_CHANNELS, PH_BINS), generator=generator)
        second_profile = torch.rand((graphs, PH_CHANNELS, PH_BINS), generator=generator)
        batch.ph_valid, batch.ph_mask = valid, torch.zeros((graphs, 8), dtype=torch.bool)
        batch.ph_profile = first_profile
        first, _ = model(batch)
        batch.ph_profile = second_profile
        second, _ = model(batch)
        assert float(model.last_ph_stats['residual_norm']) > 0.0
        assert float((first - second).abs().max()) > 1e-6, \
            'a different valid profile did not reach the prediction'
        model.gamma.zero_()
        batch.ph_profile = first_profile
        closed_first, _ = model(batch)
        batch.ph_profile = second_profile
        closed_second, _ = model(batch)
    torch.testing.assert_close(closed_first, closed_second, atol=0, rtol=0)


# ⑥ F_OFF is an exact fallback to the base downstream path
def test_f_off_parity_on_the_real_deployment():
    if not CHECKPOINT.is_file():
        pytest.skip('deployment absent')
    package = _package()
    batch = _toy_batch()
    graphs = int(batch.graph_available.numel())
    batch.ph_profile = torch.zeros((graphs, PH_CHANNELS, PH_BINS))
    batch.ph_valid = torch.zeros(graphs, dtype=torch.bool)
    batch.ph_mask = torch.zeros((graphs, 8), dtype=torch.bool)
    batch.y = torch.ones(graphs, 1)
    set_global_seed(7)
    base = GalformerDownstream(_encoder(package), readout='DUAL')
    # the same seed before each construction: the shared modules are drawn from
    # the same random stream, and F_OFF's extra modules are drawn after them
    set_global_seed(7)
    retention = GalformerPHDownstream(_encoder(package), readout='DUAL', ph_residual='off')
    base.eval(), retention.eval()
    with torch.no_grad():
        left, _ = base(batch)
        right, _ = retention(batch)
    torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert float(retention.ph_contribution(batch, torch.zeros(graphs, 512)).abs().max()) == 0.0


# ⑦ the frozen encoder stays in eval, the pretraining-only path stays frozen
def test_frozen_modules_and_eval_state_on_the_real_deployment():
    if not CHECKPOINT.is_file():
        pytest.skip('deployment absent')
    model = _retention(_package(), 'real')
    frozen = model.freeze_training_only_heads()
    model.train()
    assert model.training and not model.encoder.ph_encoder.training
    for name in FROZEN_PRETRAIN_ONLY:
        module = getattr(model.encoder, name)
        tensors = ([module] if isinstance(module, torch.nn.Parameter)
                   else list(module.parameters()))
        assert tensors and all(not tensor.requires_grad for tensor in tensors), name
    assert 'ph_head' in frozen and 'head_2d' in frozen
    assert model.ph_proj.weight.requires_grad and model.gamma.requires_grad


# ⑧ invalid PH gives zero residual; a saved unit reloads with its identity
def test_invalid_path_and_unit_roundtrip():
    if not CHECKPOINT.is_file():
        pytest.skip('deployment absent')
    package = _package()
    model = _retention(package, 'real')
    batch = _toy_batch()
    graphs = int(batch.graph_available.numel())
    batch.ph_profile = torch.rand((graphs, PH_CHANNELS, PH_BINS))
    batch.ph_valid = torch.zeros(graphs, dtype=torch.bool)
    batch.ph_mask = torch.zeros((graphs, 8), dtype=torch.bool)
    with torch.no_grad():
        model.gamma.fill_(0.3)
        contribution = model.ph_contribution(batch, torch.ones(graphs, 512))
    assert float(contribution.abs().max()) == 0.0, 'invalid PH must not contribute'
    assert model.last_ph_stats['ph_valid_fraction'] == 0.0
    payload = {'state_dict': model.state_dict(), 'group': 'F_REAL',
               'architecture': model.architecture_name, 'epochs_run': 30,
               'checkpoint_sha256': IDENTITY['sha256']}
    torch.save(payload, '/tmp/galph_retention_r5_roundtrip.pt')
    restored = _retention(package, 'real')
    restored.load_state_dict(torch.load('/tmp/galph_retention_r5_roundtrip.pt',
                                        map_location='cpu', weights_only=False)['state_dict'],
                             strict=True)
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name]), name


# ⑨ the aggregator refuses a unit that does not carry this round's identity
def _unit(group, task, fold, identity, **overrides):
    row = {'group': group, 'task': task, 'fold': fold,
           'protocol': 'ph_retention_development', 'outer_test': 'NOT_RUN',
           'validation_only': True, 'best_validation_r2': 0.7, 'best_epoch': 5,
           'epochs_configured': 30, 'epochs_run': 5,
           'checkpoint_sha256': identity['sha256'],
           'checkpoint_identity': {key: identity[key] for key in
                                   ('record', 'record_path', 'checkpoint', 'sha256', 'step',
                                    'summary_mode', 'ph_mode', 'ph_encoder_version')},
           'ph_encoder_version': identity['ph_encoder_version'],
           'pretrain_summary_mode': identity['summary_mode'],
           'pretrain_ph_mode': identity['ph_mode'],
           'readout': 'DUAL', 'adaptation': 'full',
           'coverage': {'samples': 10, 'valid': 10, 'invalid': 0, 'missing': 0},
           'diagnostics': {'batches': 4, 'retention_gate_tanh': 0.01,
                           'residual_relative_norm': 1e-3, 'ph_valid_fraction': 1.0},
           'wall_seconds': 1.0}
    row.update(overrides)
    return row


def _write_units(root, identity, mutate=None):
    for group in ('F_OFF', 'F_CONST', 'F_REAL'):
        for task in ('xc', 'eps', 'eat'):
            for fold in (0, 1):
                row = _unit(group, task, fold, identity)
                if mutate is not None:
                    mutate(row)
                folder = Path(root) / group / task / f'fold{fold}'
                folder.mkdir(parents=True, exist_ok=True)
                (folder / 'metrics.json').write_text(json.dumps(row), encoding='utf-8')


def test_aggregator_uses_the_shared_identity_record(tmp_path):
    import scripts.aggregate_glt_galph_ph_retention as aggregator
    assert aggregator.frozen_c1_sha256() == IDENTITY['sha256']
    assert aggregator.DEFAULT_IDENTITY == DEFAULT_IDENTITY
    _write_units(tmp_path, IDENTITY)
    clean = aggregator.summarize(tmp_path)
    assert clean['checkpoint_identity']['sha256'] == IDENTITY['sha256']
    assert clean['risk_flags'] == []
    assert set(clean['arm_gate_tanh']) == {'F_OFF', 'F_CONST', 'F_REAL'}

    def wrong_encoder(row):
        if row['task'] == 'eat':
            row['ph_encoder_version'] = 'scale-interaction-v1'
    _write_units(tmp_path, IDENTITY, wrong_encoder)
    with pytest.raises(ValueError, match='ph_encoder_version'):
        aggregator.summarize(tmp_path)

    def wrong_record(row):
        if row['task'] == 'eps' and row['group'] == 'F_CONST':
            row['checkpoint_identity']['record_path'] = 'configs/mts/other.json'
    _write_units(tmp_path, IDENTITY, wrong_record)
    with pytest.raises(ValueError, match='identity record'):
        aggregator.summarize(tmp_path)

    def missing_coverage(row):
        if row['group'] == 'F_OFF' and row['task'] == 'xc':
            row['coverage'] = {'samples': 10, 'valid': 9, 'invalid': 1, 'missing': 1}
    _write_units(tmp_path, IDENTITY, missing_coverage)
    with pytest.raises(ValueError, match='coverage'):
        aggregator.summarize(tmp_path)

    def epochs_impossible(row):
        if row['group'] == 'F_REAL' and row['task'] == 'xc':
            row['epochs_run'] = 3
    _write_units(tmp_path, IDENTITY, epochs_impossible)
    with pytest.raises(ValueError, match='epochs_run'):
        aggregator.summarize(tmp_path)


# ⑩ the runner refuses a checkpoint outside the record and records the identity
def test_runner_identity_wiring_is_explicit():
    import scripts.finetune_glt_galformer_ph_retention as runner
    source = Path(runner.__file__).read_text(encoding='utf-8')
    assert 'verify_checkpoint' in source and 'checkpoint_identity' in source
    assert runner.DEFAULT_IDENTITY == DEFAULT_IDENTITY
    assert 'frozen_arm_deployment' not in source, \
        'the previous cycle version record must no longer gate this round'
    history = runner.EpochHistory()
    history.update(validation_r2=0.5)
    history.update(validation_r2=0.6)
    assert len(history.history) == 2 and history.history[-1]['validation_r2'] == 0.6


def test_split_manifest_matches_the_contract():
    for task in ('xc', 'eps', 'eat'):
        manifest = json.loads(Path(f'data/splits/mips_outer5_inner20/{task}.json')
                              .read_text(encoding='utf-8'))
        expected = build_manifest(task, f'data/raw/smi_{task}.csv', 'outer5_inner20')
        assert manifest['sample_order_hash'] == expected['sample_order_hash']
        assert manifest['validation_is_test'] is False
        assert len(manifest['folds']) == 5


class _DiagnosticBatch:
    """Carries only what ``ph_diagnostics`` reads, with one graph."""

    def __init__(self, profile):
        self.ph_profile = profile
        self.ph_mask = torch.zeros((profile.size(0), 8), dtype=torch.bool)

    def to(self, device, **kwargs):
        return self


class _DiagnosticModel:
    """Minimal stand-in: the frozen encoder contract plus the reported stats."""

    class _PHEncoder:
        def __call__(self, profile, mask):
            return profile

        def summarize(self, tokens):
            return tokens.mean(1)

    def __init__(self, group):
        self.ph_residual = group
        self.encoder = type('Encoder', (), {'ph_encoder': self._PHEncoder()})()
        self.last_ph_stats = {'gate_tanh': 0.0, 'residual_relative_norm': 0.0,
                              'residual_norm': 0.0, 'ph_valid_fraction': 1.0}

    def __call__(self, batch):
        return None, {'gate_mean': 0.5}

    def eval(self):
        return self


# ⑪ the reported PH diagnostics survive a const profile on another device
def test_ph_diagnostics_handles_a_const_profile_on_another_device():
    """The r5 smoke died on exactly this: const on CPU, batch profiles on CUDA."""
    import scripts.finetune_glt_galformer_ph_retention as runner
    if not torch.cuda.is_available():
        pytest.skip('a second device is needed to reproduce the mixing')
    profile = torch.ones((2, PH_CHANNELS, PH_BINS), dtype=torch.float32)
    profile[1].add_(1.0)
    const = torch.full((PH_CHANNELS, PH_BINS), 0.5).cuda()
    batches = [_DiagnosticBatch(value) for value in (profile.clone(), profile.clone())]
    stats = runner.ph_diagnostics(_DiagnosticModel('real'), batches,
                                  torch.device('cpu'), const)
    assert stats['profile_source'] == 'sample_own_frozen'
    # the stub summarises by channel mean: rows are 1.0 and 2.0, const is 0.5
    assert stats['encoder_summary_own_vs_const_max_abs'] == pytest.approx(1.5, abs=1e-9)
    assert stats['profile_vs_const_max_abs'] == pytest.approx(1.5, abs=1e-9)
    assert stats['profile_cross_sample_spread'] == pytest.approx(1.0, abs=1e-9)
    zeros = [_DiagnosticBatch(torch.zeros((2, PH_CHANNELS, PH_BINS)))]
    assert runner.ph_diagnostics(_DiagnosticModel('off'), zeros,
                                 torch.device('cpu'), const)['profile_source'] \
        == 'zero_placeholder'
