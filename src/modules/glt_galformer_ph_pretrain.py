"""GLT-GALPH r2 objectives: 2D/3D masked-token CE, dual-view InfoNCE, PH.

Route-N (mean) and Route-C (cls) share this module; the only differences live
in the model's summary route and in where PH enters.  The contrastive term
reuses the already-validated multi-positive, cross-rank differentiable
``alignment_loss`` instead of a new InfoNCE implementation.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .glt_dual_pretrain import alignment_loss
from .glt_galformer_ph import PH_PROFILE_DIM

CONTRASTIVE_TEMPERATURE = 0.1
PH_HUBER_DELTA = 0.1


class GalformerPretrainer(nn.Module):
    def __init__(self, summary_mode='mean', ph_mode=None, dropout=0.1,
                 temperature=CONTRASTIVE_TEMPERATURE):
        super().__init__()
        from .glt_galformer_ph import GLTGalPH
        self.summary_mode = summary_mode
        self.ph_mode = ph_mode
        self.temperature = float(temperature)
        self.model = GLTGalPH(summary_mode=summary_mode, ph_mode=ph_mode,
                              dropout=dropout)

    def forward(self, data, labels, world_size=1):
        out = self.model(data)
        total_2d = int(labels['label_2d'].numel())
        total_3d = int(labels['label_3d'].numel())

        if total_2d:
            logits_2d = self.model.head_2d(out['atom_states'][data.mask2d_rows])
            loss_2d = F.cross_entropy(logits_2d.float(), labels['label_2d'])
            sum_2d = F.cross_entropy(logits_2d.float(), labels['label_2d'],
                                     reduction='sum')
        else:
            logits_2d = None
            loss_2d = out['g2'].sum() * 0.0
            sum_2d = out['g2'].sum() * 0.0

        if total_3d:
            logits_3d = self.model.head_3d(out['bond_states'][data.mask3d_rows])
            loss_3d = F.cross_entropy(logits_3d.float(), labels['label_3d'])
            sum_3d = F.cross_entropy(logits_3d.float(), labels['label_3d'],
                                     reduction='sum')
        else:
            logits_3d = None
            loss_3d = out['g3'].sum() * 0.0
            sum_3d = out['g3'].sum() * 0.0

        valid = out['line3d_valid'].bool() & data.graph_available.bool()
        cl_sum, cl_count = alignment_loss(
            self.model.cl_proj2(out['g2']), self.model.cl_proj3(out['g3']),
            labels['identity'], valid, temperature=self.temperature)

        ph_sum = out['g3'].sum() * 0.0
        ph_count = torch.zeros((), dtype=torch.long, device=out['g2'].device)
        if self.ph_mode == 'global':
            summary = out['g3']
            prediction = self.model.ph_head(summary).float()
            patch_mask = labels['ph_patch_mask'].bool()
            target = labels['label_ph'].float().reshape(-1, 2, 12)
            predicted = prediction.reshape(-1, 8, 12)[:, patch_mask[0], :]
            valid_ph = data.ph_valid.bool() & data.graph_available.bool()
            per_graph = F.huber_loss(predicted, target, reduction='none',
                                     delta=PH_HUBER_DELTA).mean(-1).mean(-1)
            ph_sum = per_graph[valid_ph].sum()
            ph_count = valid_ph.sum()

        graph_count = data.graph_available.numel()
        return {
            'sums': torch.stack([sum_2d, sum_3d, cl_sum, ph_sum]),
            'counts': torch.stack([
                torch.tensor(total_2d, dtype=torch.long, device=out['g2'].device),
                torch.tensor(total_3d, dtype=torch.long, device=out['g2'].device),
                cl_count.to(torch.long), ph_count.to(torch.long)]),
            'loss_2d': loss_2d, 'loss_3d': loss_3d,
            'loss_cl': cl_sum / cl_count.clamp_min(1).to(cl_sum.dtype),
            'loss_ph': ph_sum / ph_count.clamp_min(1).to(ph_sum.dtype),
            'logits_2d': logits_2d, 'logits_3d': logits_3d,
            'summary_mode': self.summary_mode, 'ph_mode': self.ph_mode,
            'graph_count': graph_count,
        }


def galformer_objective(result, world_size=1, weights=(1.0, 1.0, 1.0, 1.0)):
    """Token-level global mean per term, DDP-corrected (gradients are averaged)."""
    sums = result['sums']
    counts = result['counts'].to(sums.dtype).clamp_min(1.0)
    weight = sums.new_tensor(weights[:sums.numel()])
    return (sums / counts * weight).sum() * float(world_size)


def galformer_deployment_package(pretrainer, step):
    """Inference package: encoders, CLS tokens, readout, PH modules.

    Training-only parts (mask heads, CL projection heads, PH reconstruction
    head) and the downstream fusion/property head are excluded.
    """
    model = pretrainer.model
    excluded = ('head_2d.', 'head_3d.', 'cl_proj2.', 'cl_proj3.', 'ph_head.')
    state = {name: value.detach().cpu().clone()
             for name, value in model.state_dict().items()
             if not name.startswith(excluded)}
    return dict(architecture=model.architecture_name,
                summary_mode=model.summary_mode, ph_mode=model.ph_mode,
                pretrain_objective='galformer_native_mask_cl',
                mask_candidate_rate=0.40, mask_policy='80_10_10',
                contrastive_temperature=pretrainer.temperature,
                ph_schema='glt-ph-betti-v2' if model.ph_mode else None,
                ph_channels=3 if model.ph_mode else None,
                ph_bins=32 if model.ph_mode else None,
                step=int(step), state_dict=state)


def load_galformer_deployment(model, package, expected_step=5000):
    if (package.get('architecture') != model.architecture_name
            or package.get('summary_mode') != model.summary_mode
            or package.get('ph_mode') != model.ph_mode
            or int(package.get('step', -1)) != int(expected_step)):
        raise ValueError('galformer checkpoint identity mismatch')
    current = model.state_dict()
    expected = {name for name in current
                if not name.startswith(('head_2d.', 'head_3d.', 'cl_proj2.',
                                        'cl_proj3.', 'ph_head.'))}
    if set(package['state_dict']) != expected:
        raise ValueError('deployment must contain exactly the inference tensors')
    current.update(package['state_dict'])
    model.load_state_dict(current, strict=True)
