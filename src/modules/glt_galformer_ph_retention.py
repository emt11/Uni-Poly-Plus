"""PH retention groups: the same C1 trunk with a matched downstream PH residual.

All three groups instantiate the identical new modules (``ph_proj``/``gamma``)
and load the identical frozen C1 ``scale-interaction-v2`` PH encoder, so their
capacity and common initial tensors match; only the PH input differs:

    r3_new = r3 + tanh(gamma) * W_ph * PH_encoder_frozen(profile)

* ``F_OFF``   -- the residual is explicitly zeroed by a branch, the modules stay
                 instantiated and identical to the other two groups
* ``F_CONST`` -- the fixed P_train mean profile for every sample
* ``F_REAL``  -- the sample's own profile

``gamma`` starts at zero, so the first update has no PH projection gradient but
does carry a gate gradient (``d/d gamma = (1 - tanh^2(gamma)) W_ph p_ph``); the
gate therefore trains and the projection follows from the second update on.
The PH encoder is frozen and kept in eval mode even after ``model.train()``.
The pretraining-only PH residual path (``ph_to_summary``/``alpha_ph``) is
disabled identically in all three groups and frozen, because the CLS readout
never reads ``g3``.
"""
import torch
from torch import nn

from .glt_galformer_ph_downstream import GalformerDownstream

PH_RESIDUAL_MODES = ('off', 'const', 'real')
FROZEN_PRETRAIN_ONLY = ('ph_encoder', 'ph_to_summary', 'alpha_ph')


class GalformerPHDownstream(GalformerDownstream):
    architecture_name = 'O8-GalformerTrimer-GalPH-Downstream-PHRetention'

    def __init__(self, encoder, readout='DUAL', ph_residual='off', hidden=256, dropout=0.1):
        super().__init__(encoder, readout=readout, hidden=hidden, dropout=dropout)
        if encoder.ph_mode != 'global':
            raise ValueError('PH retention requires a PH-conditioned encoder')
        if ph_residual not in PH_RESIDUAL_MODES:
            raise ValueError(f'ph_residual must be one of {PH_RESIDUAL_MODES}')
        self.ph_residual = ph_residual
        self.ph_proj = nn.Linear(512, 512)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.last_ph_stats = {}

    def train(self, mode=True):
        """The frozen PH encoder stays in eval mode whatever the parent asks."""
        super().train(mode)
        self.encoder.ph_encoder.eval()
        return self

    def freeze_training_only_heads(self):
        frozen = super().freeze_training_only_heads()
        for name in FROZEN_PRETRAIN_ONLY:
            module = getattr(self.encoder, name, None)
            if module is not None:
                module.requires_grad_(False)
                frozen.append(name)
        return frozen

    def ph_contribution(self, batch, reference):
        """The retention residual for this batch, before the validity mask."""
        tokens = self.encoder.ph_encoder(batch.ph_profile.float(), batch.ph_mask.bool())
        summary = self.encoder.ph_encoder.summarize(tokens)
        residual = torch.tanh(self.gamma) * self.ph_proj(summary)
        if self.ph_residual == 'off':
            # Explicit control branch: modules exist and are identical, output is zero.
            contribution = torch.zeros_like(residual)
        else:
            valid = batch.ph_valid.bool().unsqueeze(-1)
            contribution = torch.where(valid, residual, torch.zeros_like(residual))
        with torch.no_grad():
            residual_norm = float(contribution.detach().float().norm(dim=-1).mean())
            reference_norm = float(reference.detach().float().norm(dim=-1).mean())
        self.last_ph_stats = {
            'group': self.ph_residual,
            'gate_tanh': float(torch.tanh(self.gamma).detach()),
            'residual_norm': residual_norm,
            'reference_norm': reference_norm,
            'residual_relative_norm': (residual_norm / reference_norm) if reference_norm else 0.0,
            'ph_valid_fraction': (float(batch.ph_valid.float().mean())
                                  if self.ph_residual != 'off' else 0.0),
        }
        return contribution

    def adjust_r3(self, batch, r3, out):
        return r3 + self.ph_contribution(batch, r3)
