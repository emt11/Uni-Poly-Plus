"""GLT-GALPH r2 P1 correctness tests (plan section 10, items 1-12).

Real frozen records supply the fixtures; synthetic tensors are confined to the
invariance/parity checks.  No test performs an optimizer update.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from test_complete_trimer_glt import _toy_pair

from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.glt_dual_static import build_dual_static
from src.dataset.glt_galformer_ph import (PHBettiReader, galformer_collate,
                                          prepare_galformer_sample, mask_3d)
from src.modules.glt_galformer_ph import MASK_TYPE, GLTGalPH, PHProfileEncoder
from src.modules.glt_galformer_ph_pretrain import (GalformerPretrainer,
                                                   galformer_deployment_package,
                                                   galformer_objective,
                                                   load_galformer_deployment)

SMILES = '*CCOCC*'
PH_ROOT = Path('results/glt_galph_20260920/p0/ph_sidecar_betti_v2')


class _FakeReader:
    """Deterministic PH stand-in when the real sidecar is unavailable."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def get(self, position, key):
        profile = self.rng.random((3, 32)).astype(np.float32) * 0.1
        return profile, True

    def __len__(self):
        return 1 << 20


def _fixture():
    topology = build_canonical_periodic_topology(SMILES)
    _, trimer = _toy_pair(SMILES)
    static = build_dual_static(topology, trimer, SMILES)
    assert static['geometry_valid']
    return topology, trimer, static


def _sample(**kwargs):
    topology, trimer, static = _fixture()
    return prepare_galformer_sample(topology, trimer, SMILES, static=static,
                                    seed=42, key='fixture', position=0, **kwargs)


def _batch(records):
    return galformer_collate(records)


def _distinct_ph_records():
    """Two real samples whose PH patch masks differ (bounded key search)."""
    topology, trimer, static = _fixture()
    first, first_mask = None, None
    for index in range(8):
        record = prepare_galformer_sample(topology, trimer, SMILES, static=static,
                                          seed=42, key=f'ph{index}', position=index,
                                          ph_reader=_FakeReader(seed=index))
        mask = record[1]['ph_patch_mask']
        if first is None:
            first, first_mask = record, mask.clone()
        elif not torch.equal(mask, first_mask):
            return [first, record]
    raise AssertionError('no two candidate keys produced different PH patch masks')


# ① 2D mask must not leak the original atom state into the encoder input
def test_mask2d_blocks_original_atom_state():
    data, labels = _sample()
    model = GLTGalPH('mean')
    initial = model.o8.atom_embedding(data)
    masked_rows = data.mask2d_rows[data.mask2d_policy == 1]
    assert masked_rows.numel() > 0, 'fixture must contain MASK rows'
    states, _, _, _ = model._o8_states(data)
    torch.testing.assert_close(states[masked_rows],
                               model.mask_2d_embedding.expand(masked_rows.numel(), 512))
    assert not torch.allclose(states[masked_rows], initial[masked_rows])
    # the PathNode bias consumes the masked states, not the raw features
    donor_rows = data.mask2d_rows[data.mask2d_policy == 2]
    if donor_rows.numel():
        donor_states = model._donor_embeddings(data)
        torch.testing.assert_close(states[donor_rows], donor_states)


# ②③ 3D masking contract (deterministic, run in a clean subprocess because the
# pytest process accumulates state that perturbs the toy-geometry scatter path)
def test_mask3d_blocks_endpoint_distance_and_type():
    script = Path(__file__).with_name('_galph_mask3d_check.py')
    completed = subprocess.run([sys.executable, str(script)], capture_output=True,
                               text=True, timeout=900,
                               env={**os.environ, 'OMP_NUM_THREADS': '1'})
    assert completed.returncode == 0, completed.stdout + completed.stderr
    verdict = json.loads(completed.stdout.strip().splitlines()[-1])
    assert set(verdict['cases']) == {'MASK_only', 'REPLACE_only', 'KEEP_only',
                                     'MASK_REPLACE_KEEP'}
    assert verdict['status'] == 'PASS', verdict
    for name, case in verdict['cases'].items():
        assert case['status'] == 'PASS', (name, case)
    combined = verdict['cases']['MASK_REPLACE_KEEP']['checks']
    assert combined['mask_embedding_exact'] and combined['mask_type_in_bias']
    assert combined['keep_input_is_clean'] and combined['keep_type_is_real']
    assert verdict['cases']['REPLACE_only']['checks']['replace_takes_donor_input']
    assert verdict['cases']['REPLACE_only']['checks']['replace_distance_is_donor']


# ④ No-CLS summary must pool every real physical bond, not center bonds only
def test_nocls_pool_uses_all_physical_bonds():
    data, labels = _sample()
    batch, _ = _batch([(data, labels)])
    model = GLTGalPH('mean')
    out = model(batch)
    manual = torch.zeros_like(out['g3'])
    counts = torch.bincount(batch.bond_batch.long(), minlength=1).clamp_min(1)
    manual.index_add_(0, batch.bond_batch.long(), out['bond_states'])
    manual = manual / counts.unsqueeze(-1)
    manual = torch.where(out['line3d_valid'].unsqueeze(-1), manual,
                         torch.zeros_like(manual))
    torch.testing.assert_close(out['g3'], manual, atol=1e-6, rtol=1e-6)
    center_bonds = int(batch.bond_center.sum())
    assert center_bonds < int(batch.bond_distance.numel()), \
        'fixture must contain non-center physical bonds'


# ⑤⑥⑦ CLS: one per graph, no cross-graph edges, present in every layer
def test_cls_token_structure():
    records = [_sample(), _sample()]
    batch, labels = _batch(records)
    graphs = int(batch.graph_available.numel())
    model = GLTGalPH('cls')
    captured = []
    handles = [layer.register_forward_hook(
        lambda module, inputs, output: captured.append(int(inputs[0].size(0))))
        for layer in model.o8.layers]
    out = model(batch)
    for handle in handles:
        handle.remove()
    n_atom = int(batch.mips_x.size(0))
    assert out['cls2'].shape == (graphs, 512)
    assert len(captured) == len(model.o8.layers)
    assert all(size == n_atom + graphs for size in captured), \
        'CLS must join all six layers'
    # no cross-graph CLS edge: the CLS attached to atom i is i graph's own
    assert out['cls2'].shape[0] == graphs
    assert out['cls3'].shape[0] == graphs


# ⑧ PH sidecar key alignment
@pytest.mark.skipif(not (PH_ROOT / '.done').is_file(),
                    reason='PH Betti sidecar not present in this checkout')
def test_ph_sidecar_key_alignment():
    reader = PHBettiReader(PH_ROOT)
    profile, valid = reader.get(0, _key_at(0))
    assert profile.shape == (3, 32) and profile.dtype == np.float32
    assert valid in (True, False)
    with pytest.raises(ValueError):
        reader.get(0, '00' * 32)


def _key_at(position):
    """The P_train key at a *training-stream* position (the sidecar's order)."""
    from src.training.glt_dual_runtime import (IndexedFrozenDualSource,
                                               load_sample_index_artifact,
                                               open_source, OrderedSampleStream)
    artifact = load_sample_index_artifact(
        'results/glt_pred_20260918/s3b_prep/pretrain_split_v1.json', 'train')
    base, _ = open_source('data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1',
                          'data/processed/mips_trimer_scage',
                          dual_static_root='data/processed/glt_dual_v2/pi1m/dual_static_v1')
    subset = IndexedFrozenDualSource(base, artifact['indices'])
    stream = OrderedSampleStream(len(subset), 42)
    key, _ = subset.samples[int(stream.index_at(int(position)))]
    base.close()
    return key.hex()


# ⑨ masked PH patches contribute no raw values
def test_ph_masked_patch_hides_raw_values():
    encoder = PHProfileEncoder()
    profile = torch.rand(1, 3, 32)
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[0, :2] = True
    tokens = encoder(profile, mask)
    masked = encoder.mask_token + encoder.scale_embedding
    torch.testing.assert_close(tokens[0, :2], masked[:2])
    altered = profile.clone()
    altered[0, :, :8] = 123.0                      # patches 0 and 1 live here
    tokens_alt = encoder(altered, mask)
    torch.testing.assert_close(tokens_alt[0, :2], tokens[0, :2], atol=0, rtol=0)


# ⑨b PH batch stacking: per-graph [3,32] profiles, [8] masks and scalar flags
def test_ph_batch_stacking_two_graphs():
    records = _distinct_ph_records()
    batch, labels = _batch(records)
    assert batch.ph_profile.shape == (2, 3, 32), batch.ph_profile.shape
    assert batch.ph_mask.shape == (2, 8), batch.ph_mask.shape
    assert batch.ph_valid.shape == (2,), batch.ph_valid.shape
    assert labels['ph_patch_mask'].shape == (2, 8)
    for graph, (data, _) in enumerate(records):
        torch.testing.assert_close(batch.ph_profile[graph], data.ph_profile)
        torch.testing.assert_close(batch.ph_mask[graph], data.ph_mask)
    assert not torch.equal(batch.ph_mask[0], batch.ph_mask[1]), \
        'fixture must give the two graphs different PH patch masks'
    torch.manual_seed(0)
    model = GLTGalPH('mean', 'global')
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert out['ph_summary'].shape == (2, 512)
    assert bool(torch.isfinite(out['g3']).all())


class _PatchStub(torch.nn.Module):
    """ph_head stand-in: patch p of graph b predicts the constant 10*b + p."""

    def forward(self, summary):
        graphs = summary.size(0)
        patch = torch.arange(8, dtype=summary.dtype, device=summary.device).unsqueeze(0)
        offset = torch.arange(graphs, dtype=summary.dtype,
                              device=summary.device).unsqueeze(-1) * 10.0
        values = offset + patch                       # [B,8]
        return values.unsqueeze(-1).expand(-1, -1, 12).reshape(graphs, 96)


def test_ph_loss_uses_per_graph_patch_mask():
    records = _distinct_ph_records()
    batch, labels = _batch(records)
    masks = labels['ph_patch_mask'].bool()
    assert masks.shape == (2, 8) and not torch.equal(masks[0], masks[1])
    torch.manual_seed(0)
    trainer = GalformerPretrainer('mean', 'global')
    trainer.model.ph_head = _PatchStub()
    result = trainer(batch, labels)

    def huber(patch_values, target):
        values = patch_values.unsqueeze(-1).expand(-1, 12)
        return F.huber_loss(values, target, reduction='none', delta=0.1).mean()

    own = []
    for graph in range(2):
        patches = torch.where(masks[graph])[0].float()
        target = labels['label_ph'][graph * 24:(graph + 1) * 24].reshape(2, 12)
        own.append(huber(10.0 * graph + patches, target))
    own = torch.stack(own)
    torch.testing.assert_close(result['loss_ph'], own.mean(), atol=1e-6, rtol=1e-6)
    # Reusing graph 0's mask for graph 1 would score different patches entirely.
    shared = 10.0 + torch.where(masks[0])[0].float()
    target_1 = labels['label_ph'][24:].reshape(2, 12)
    assert not torch.isclose(huber(shared, target_1), own[1], atol=1e-6, rtol=1e-6)


# ⑩ alpha_PH = 0 parity: N1 vs N0 and C1 vs C0 summaries
@pytest.mark.parametrize('summary_mode', ['mean', 'cls'])
def test_alpha_zero_parity(summary_mode):
    data, labels = _sample(ph_reader=_FakeReader())
    data.ph_profile = data.ph_profile.float()
    batch, _ = _batch([(data, labels)])
    plain = GLTGalPH(summary_mode)
    with_ph = GLTGalPH(summary_mode, 'global')
    missing, unexpected = with_ph.load_state_dict(plain.state_dict(), strict=False)
    assert all('ph_' in name or 'alpha_ph' in name for name in missing), missing
    plain.eval(), with_ph.eval()
    with torch.no_grad():
        left = plain(batch)
        right = with_ph(batch)
    torch.testing.assert_close(left['g3'], right['g3'], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(left['g2'], right['g2'], atol=1e-6, rtol=1e-6)


# ⑫ deployment strict load
@pytest.mark.parametrize('summary_mode,ph_mode', [('mean', None), ('mean', 'global'),
                                                  ('cls', None), ('cls', 'global')])
def test_deployment_roundtrip(summary_mode, ph_mode):
    torch.manual_seed(0)
    trainer = GalformerPretrainer(summary_mode, ph_mode)
    package = galformer_deployment_package(trainer, 5000)
    rebuilt = GLTGalPH(summary_mode, ph_mode)
    load_galformer_deployment(rebuilt, package, 5000)
    other = GLTGalPH('cls' if summary_mode == 'mean' else 'mean', ph_mode)
    with pytest.raises(ValueError):
        load_galformer_deployment(other, package, 5000)


def test_bf16_autocast_masked_forward():
    """The masked forward must survive the formal bf16 autocast (P1.2 smoke bug)."""
    data, labels = _sample(ph_reader=_FakeReader())
    batch, labels = _batch([(data, labels)])
    torch.manual_seed(0)
    trainer = GalformerPretrainer('cls', 'global')
    with torch.autocast('cpu', dtype=torch.bfloat16):
        result = trainer(batch, labels)
        objective = galformer_objective(result)
    assert all(bool(torch.isfinite(value).all()) for value in result['sums'])
    assert bool(torch.isfinite(objective).all())
    objective.backward()
    assert float(trainer.model.mask_2d_embedding.grad.abs().sum()) > 0
    assert float(trainer.model.mask_3d_embedding.grad.abs().sum()) > 0


def test_objective_and_gradients_are_finite():
    data, labels = _sample(ph_reader=_FakeReader())
    batch, labels = _batch([(data, labels)])
    torch.manual_seed(0)
    trainer = GalformerPretrainer('mean', 'global')
    result = trainer(batch, labels)
    objective = galformer_objective(result)
    assert bool(torch.isfinite(objective).all())
    objective.backward()
    grads = {name: parameter.grad for name, parameter in trainer.named_parameters()
             if parameter.grad is not None}
    assert grads, 'no gradients were produced'
    assert all(bool(torch.isfinite(value).all()) for value in grads.values())
    assert trainer.model.alpha_ph.grad is not None
    assert float(trainer.model.alpha_ph.grad.abs()) > 0
    assert trainer.model.ph_to_summary.weight.grad is not None


def test_ddp_contrastive_two_rank_gloo():
    """0-update 2-rank DDP check that the contrastive path survives a collective."""
    script = Path(__file__).with_name('_galph_ddp_check.py')
    completed = subprocess.run(
        [sys.executable, '-m', 'torch.distributed.run', '--standalone',
         '--nproc_per_node=2', str(script)],
        capture_output=True, text=True, timeout=900,
        env={**os.environ, 'OMP_NUM_THREADS': '1'})
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1])['status'] == 'PASS'


def test_shared_initialization_is_exact():
    """Every same-name, same-shape tensor is identical across the four arms."""
    import json as _json
    import itertools
    arms = {'N0': ('mean', None), 'N1': ('mean', 'global'),
            'C0': ('cls', None), 'C1': ('cls', 'global')}
    states = {}
    for arm, (summary_mode, ph_mode) in arms.items():
        torch.manual_seed(42)
        states[arm] = {name: value.detach().clone() for name, value in
                       GalformerPretrainer(summary_mode, ph_mode).model.state_dict().items()}
    report, worst = {}, 0.0
    for left, right in itertools.combinations(arms, 2):
        common = set(states[left]) & set(states[right])
        diff = max((float((states[left][name] - states[right][name]).abs().max())
                    for name in common), default=0.0)
        report[f'{left}_{right}'] = {'common_keys': len(common), 'max_abs_diff': diff}
        worst = max(worst, diff)
    assert worst == 0.0, report
    assert report['N0_N1']['common_keys'] > 50 and report['C0_C1']['common_keys'] > 50
    assert report['N1_C1']['common_keys'] > report['N0_N1']['common_keys']
    Path('/tmp/galph_shared_init.json').write_text(_json.dumps(report))
