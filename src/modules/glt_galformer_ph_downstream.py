"""GLT-GALPH downstream head: gated DUAL fusion over the pretrained summaries.

Route-N (no CLS) reads the trunk's own graph means; Route-C adds a linear readout
over ``[CLS ; mean]`` for both modalities.  Fusion is the planned gated residual

    gate = sigmoid(MLP([r2, r3])),   r = LN(r2 + valid_3D * gate * W3 r3)

followed by the 512->256->GELU->Dropout->1 property head.  ``2D_ONLY`` keeps the
same interface with the 3D residual disabled so the geometry slot can be ablated
without a second implementation.

The pretrained trunk is loaded strictly (``load_galformer_deployment``); the
readout, gate, fusion projection and property head are new per downstream fold
and are never part of a pretraining checkpoint.  PH is a pretraining-side
modality: a downstream batch that carries no PH profile sets ``ph_valid=False``,
which zeroes the PH residual through the model's own validity gate.
"""
import torch
from torch import nn

from .glt_dual import mean_pool
from .glt_galformer_ph import PH_BINS, PH_CHANNELS, PH_PATCHES

READOUTS = ('DUAL', '2D_ONLY')
TRAINING_ONLY_HEADS = ('head_2d', 'head_3d', 'cl_proj2', 'cl_proj3', 'ph_head')


class GalformerDownstream(nn.Module):
    architecture_name = 'O8-GalformerTrimer-GalPH-Downstream'

    def __init__(self, encoder, readout='DUAL', hidden=256, dropout=0.1):
        super().__init__()
        if readout not in READOUTS:
            raise ValueError('readout must be DUAL or 2D_ONLY')
        self.encoder = encoder
        self.readout = readout
        self.cls_readout = encoder.summary_mode == 'cls'
        if self.cls_readout:
            self.readout2 = nn.Linear(1024, 512)
            self.readout3 = nn.Linear(1024, 512)
        self.gate = nn.Sequential(nn.Linear(1024, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.proj3 = nn.Linear(512, 512)
        self.norm = nn.LayerNorm(512)
        self.head = nn.Sequential(nn.Linear(512, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def freeze_training_only_heads(self):
        """Pretraining-only heads are imported but not part of the inference path."""
        frozen = []
        for name in TRAINING_ONLY_HEADS:
            module = getattr(self.encoder, name, None)
            if module is not None:
                module.requires_grad_(False)
                frozen.append(name)
        return frozen

    def _with_ph_placeholder(self, batch, graphs):
        """Downstream batches carry no PH profile: use the model's invalid path."""
        if self.encoder.ph_mode != 'global' or hasattr(batch, 'ph_profile'):
            return batch
        device = batch.mips_x.device
        batch.ph_profile = torch.zeros((graphs, PH_CHANNELS, PH_BINS),
                                       dtype=torch.float32, device=device)
        batch.ph_mask = torch.zeros((graphs, PH_PATCHES), dtype=torch.bool, device=device)
        batch.ph_valid = torch.zeros((graphs,), dtype=torch.bool, device=device)
        return batch

    def summarize(self, batch):
        """Return (fused, aux) for the downstream readout."""
        out = self.encoder(batch)
        graphs = int(batch.graph_available.numel())
        if self.cls_readout:
            mean2 = mean_pool(out['atom_states'], batch.canonical_graph_index.long(), graphs)
            mean3 = mean_pool(out['bond_states'], batch.bond_batch.long(), graphs)
            r2 = self.readout2(torch.cat([out['cls2'], mean2], -1))
            r3 = self.readout3(torch.cat([out['cls3'], mean3], -1))
        else:
            r2, r3 = out['g2'], out['g3']
        gate = torch.sigmoid(self.gate(torch.cat([r2, r3], -1)))
        if self.readout == '2D_ONLY':
            fused = self.norm(r2)
        else:
            valid = out['line3d_valid'].bool().unsqueeze(-1)
            fused = self.norm(r2 + valid * gate * self.proj3(r3))
        aux = {'readout': self.readout, 'gate_mean': float(gate.detach().mean()),
               'ph_downstream': ('absent_zero_residual'
                                 if self.encoder.ph_mode == 'global' else 'none')}
        return fused, aux

    def forward(self, batch):
        graphs = int(batch.graph_available.numel())
        batch = self._with_ph_placeholder(batch, graphs)
        fused, aux = self.summarize(batch)
        return self.head(fused), aux
