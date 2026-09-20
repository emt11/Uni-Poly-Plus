"""r3 regression tests for the PH degeneration diagnostic.

The Codex review of the r2 diagnostic found two defects, both fixed here:

1. ``model_stats`` replaced ``batch.ph_profile`` in place, so the condition that
   ran first leaked its PH input into every later condition;
2. residual and head values were recomputed in fp32 *after* a bf16 forward.

The tests below pin the three properties the fix must guarantee: input purity
(the batch is never modified and a call is not influenced by earlier calls),
call-order independence (the ``own`` baseline is always the sample's own input,
whatever ran before) and model-state restoration (``alpha_ph`` bit-exact and the
training mode restored).  A miniature stub with the same gate semantics as the
C1 model keeps the tests free of GPU, data and checkpoints.
"""
import copy
import math
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import diagnose_glt_galph_ph_degeneration as diagnostic

DIM = 6


class _StubBatch:
    """Carries only the fields the diagnostic reads from a real batch."""

    def __init__(self, profile, valid=None, summary=None):
        self.ph_profile = profile
        self.ph_valid = (torch.ones(profile.size(0), dtype=torch.bool)
                         if valid is None else valid)
        self.graph_available = torch.ones(profile.size(0), dtype=torch.bool)
        self.line3d_valid = torch.ones(profile.size(0), dtype=torch.bool)
        self.base_summary = (torch.arange(1, profile.size(0) + 1, dtype=torch.float32)
                             .unsqueeze(-1) * torch.ones(1, DIM) if summary is None
                             else summary)

    def to(self, device, **kwargs):
        return self


class _StubModel(nn.Module):
    """Same residual semantics as ``GLTGalPH``, in miniature and on CPU."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.alpha_ph = nn.Parameter(torch.zeros(1))
        # [B, PH_BINS] stands in for the frozen encoder summary of the real model.
        self.ph_to_summary = nn.Linear(diagnostic.PH_BINS, DIM, bias=False)
        self.ph_head = nn.Linear(DIM, DIM, bias=False)
        self.cl_proj3 = nn.Linear(DIM, DIM, bias=False)
        for module in (self.ph_to_summary, self.ph_head, self.cl_proj3):
            nn.init.normal_(module.weight, std=0.3)

    def residual(self, profile, valid=None):
        tokens = profile.mean(dim=1)
        residual = torch.tanh(self.alpha_ph) * self.ph_to_summary(tokens)
        if valid is None:
            return residual
        return torch.where(valid.bool().unsqueeze(-1), residual, torch.zeros_like(residual))

    def forward(self, data):
        summary = data.ph_profile.mean(dim=1)
        return {'g3': data.base_summary + self.residual(data.ph_profile, data.ph_valid),
                'cls3': data.base_summary, 'line3d_valid': data.line3d_valid,
                'ph_summary': summary}


class _StubEncoder(nn.Module):
    """Same call contract as ``PHProfileEncoder`` (mask the content, then fuse)."""

    def __init__(self, dim=DIM, patches=diagnostic.PH_PATCHES):
        super().__init__()
        self.dim, self.patches = dim, patches
        self.mask_token = nn.Parameter(torch.zeros(dim))
        self.proj = nn.Linear(3 * 4, dim)      # 3 channels x 4 radii per patch

    def forward(self, profile, mask):
        shaped = profile.reshape(profile.size(0), 3, self.patches, -1)
        patches = shaped.permute(0, 2, 1, 3).reshape(profile.size(0), self.patches, -1)
        content = self.proj(patches)
        return torch.where(mask.unsqueeze(-1), self.mask_token.view(1, 1, -1), content)

    def summarize(self, tokens):
        return tokens.mean(1)


def _profiles(samples=4):
    torch.manual_seed(7)
    own = torch.rand((samples, 3, diagnostic.PH_BINS))
    const = own.mean(0)
    return own, const


def test_forward_never_mutates_the_batch_and_is_input_pure():
    own, const = _profiles()
    batch = _StubBatch(own)
    model = _StubModel()
    with torch.no_grad():
        model.alpha_ph.fill_(0.05)
    baseline = diagnostic._forward(model, batch, torch.device('cpu'), 'fp32')['g3']
    const_view = const.unsqueeze(0).expand_as(own).contiguous()
    diagnostic._forward(model, diagnostic._with_profile(batch, const_view),
                        torch.device('cpu'), 'fp32')
    after = diagnostic._forward(model, batch, torch.device('cpu'), 'fp32')['g3']
    assert torch.equal(baseline, after), 'a later call changed the unmodified batch result'
    assert torch.equal(batch.ph_profile, own), 'the batch profile was replaced in place'
    view = diagnostic._with_profile(batch, const_view)
    assert view is not batch and torch.equal(view.ph_profile, const_view)
    assert torch.equal(batch.ph_profile, own), 'the view wrote through to the batch'


def test_model_stats_row_pairing_is_call_order_independent():
    own, const = _profiles()
    batch = _StubBatch(own)
    model = _StubModel()
    with torch.no_grad():
        model.alpha_ph.fill_(0.03)
    checkpoint_gate = float(torch.tanh(model.alpha_ph.detach()))
    stats = diagnostic.model_stats(model, batch, const, torch.device('cpu'), 'fp32')
    for row in stats['rows']:
        gate = row['tanh_alpha']
        # Independent expectation: the baseline is the sample's own profile.
        with torch.no_grad():
            model.alpha_ph.fill_(math.atanh(gate))
            own_out = model(batch)
            const_view = copy.copy(batch)
            const_view.ph_profile = const.unsqueeze(0).expand_as(own).contiguous()
            const_out = model(const_view)
            reference = float((own_out['g3'] - model.residual(own, batch.ph_valid)).norm())
            expected = float((const_out['g3'] - own_out['g3']).norm()) / reference
            expected_residual = float(
                model.residual(own, batch.ph_valid).norm()) / reference
        assert row['g3_relative_change_const'] == pytest.approx(expected, rel=0, abs=0), \
            'the compared baseline is not the sample own PH input'
        # The same-context view reproduces the model's own expression exactly; the
        # g3-minus-summary view is a subtraction and is limited by cancellation.
        assert row['residual_relative_norm_direct'] == pytest.approx(
            expected_residual, rel=0, abs=0)
        assert row['residual_relative_norm'] == pytest.approx(
            expected_residual, rel=1e-4, abs=0)
        assert 0.0 <= row['residual_views_ratio'] < 1.0
    assert stats['checkpoint_tanh_alpha'] == checkpoint_gate
    assert sorted(row['tanh_alpha'] for row in stats['rows']) == \
        sorted({round(checkpoint_gate, 12), 0.02})


def test_model_stats_restores_gate_and_mode_exactly():
    own, const = _profiles()
    batch = _StubBatch(own)
    model = _StubModel()
    with torch.no_grad():
        model.alpha_ph.fill_(-0.017)
    original = model.alpha_ph.detach().clone()
    model.train()
    diagnostic.model_stats(model, batch, const, torch.device('cpu'), 'fp32')
    assert torch.equal(model.alpha_ph, original), 'alpha_ph was not restored bit-exactly'
    assert model.training, 'the training mode was not restored'
    model.eval()
    diagnostic.model_stats(model, batch, const, torch.device('cpu'), 'fp32')
    assert not model.training, 'eval mode was not restored'


def test_residual_and_heads_share_the_forward_precision():
    own, const = _profiles()
    batch = _StubBatch(own)
    model = _StubModel()
    with torch.no_grad():
        model.alpha_ph.fill_(0.02)
    for precision in ('fp32', 'bf16'):
        stats = diagnostic.model_stats(model, batch, const, torch.device('cpu'), precision)
        for row in stats['rows']:
            assert row['residual_dtype'] == row['g3_dtype'], \
                'residual was recomputed outside the forward precision'
            assert row['residual_direct_dtype'] == row['g3_dtype'], \
                'the residual views disagree in precision'
            assert row['head_dtype'] == ('torch.bfloat16' if precision == 'bf16'
                                         else 'torch.float32'), \
                'the heads did not run in the forward precision context'
        forward = diagnostic._forward(model, batch, torch.device('cpu'), precision)
        assert forward['residual'].dtype == forward['g3'].dtype
        assert forward['residual_direct'].dtype == forward['g3'].dtype
        assert all(torch.isfinite(value).all() for value in forward.values())


def test_mask_leakage_check_is_input_pure():
    own, _ = _profiles()
    batch = _StubBatch(own)
    encoder = _StubEncoder()
    original = batch.ph_profile.clone()
    result = diagnostic.mask_leakage(encoder, own, torch.device('cpu'), 'fp32')
    assert result['no_observable_leakage'], 'masked content reached the summary'
    assert result['changed_patches'] == diagnostic.PH_PATCHES - 1
    assert torch.equal(batch.ph_profile, original)
