"""Clean graph FP/ALIGN and optional local supervision; no feature fusion."""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F

from .glt_dual import Pure2DGLTModel, mean_pool, strict_module_state
from .glt_dual_pretrain import alignment_loss
from .ua2d_galformer_pretrain import (UA2DGalformerPretrainer, alignment_valid,
                                     balanced_fp_loss, task_sum, objective)

SCHEMA = 'ua2d_cleanfp_readout_distill_v1'


@dataclass(frozen=True)
class CleanConfig:
    readout: str = 'mean'
    local: bool = False
    weight_policy: str = 'ALL1'
    revision: str = SCHEMA
    strategy: str = 'mask_align'
    fusion: str = 'none'
    mask_ratio: float = .4
    noise_sigma: float = .03
    temperature: float = .1
    dropout: float = .1
    fp_enabled: bool = True

    def __post_init__(self):
        if (self.revision != SCHEMA or self.readout not in ('mean', 'query')
                or type(self.local) is not bool or (self.local and self.readout != 'query')
                or self.weight_policy not in ('ALL1', 'DYN') or self.strategy != 'mask_align'
                or self.fusion != 'none' or self.mask_ratio != .4 or self.noise_sigma != .03
                or self.temperature != .1 or self.dropout not in (0., .1) or self.fp_enabled is not True):
            raise ValueError('unsupported clean FP recipe')

    @property
    def reconstruct_3d_mask(self): return True

    @property
    def reconstruct_geometry(self): return False


class QueryReadout(nn.Module):
    """One query, four 32-dimensional heads; zero-gated residual on mean."""
    def __init__(self):
        super().__init__()
        self.query = nn.Parameter(torch.empty(4, 32))
        nn.init.normal_(self.query, std=.02)
        self.key = nn.Linear(512, 128)
        self.value = nn.Linear(512, 128)
        self.output = nn.Linear(128, 512)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, atoms, index, graphs):
        # Softmax separately inside each graph; no padding or cross-graph links.
        key = self.key(atoms).reshape(-1, 4, 32)
        value = self.value(atoms).reshape(-1, 4, 32)
        score = (key.float() * self.query.float()).sum(-1) / (32 ** .5)
        from torch_scatter import scatter_softmax, scatter_add
        attention = scatter_softmax(score, index, dim=0)
        pooled = scatter_add(attention[..., None] * value.float(), index, dim=0,
                             dim_size=graphs).reshape(graphs, 128)
        return mean_pool(atoms, index, graphs) + self.alpha.tanh() * self.output(pooled)


def local_targets(bonds, data):
    bond, atom = data.incidence_index.long()
    n = len(data.mips_x)
    if bond.numel():
        if (bond.min() < 0 or bond.max() >= len(bonds) or atom.min() < 0 or atom.max() >= n):
            raise ValueError('local incidence index out of range')
        if not torch.equal(data.bond_batch[bond], data.canonical_graph_index[atom]):
            raise ValueError('cross-graph local incidence')
    count = torch.bincount(atom, minlength=n)
    valid = (count > 0) & data.geometry_valid.bool()[data.canonical_graph_index]
    target = mean_pool(bonds[bond], atom, n)
    return target, valid


def local_loss(atoms, bonds, data, predictor):
    target, valid = local_targets(bonds.detach(), data)
    prediction = predictor(atoms[valid]).float()
    selected = target[valid].float().detach()
    if len(selected) and ((selected.norm(dim=-1) <= 1e-8).any() or not torch.isfinite(selected).all()):
        raise FloatingPointError('invalid local teacher representation')
    loss = (1 - (F.normalize(prediction, dim=-1) * F.normalize(selected, dim=-1)).sum(-1)).clamp_min(0)
    graphs = data.geometry_valid.numel()
    value, count = task_sum(loss, data.canonical_graph_index[valid], graphs)
    diagnostics = dict(local_atoms=int(valid.sum()), local_mean_cosine=float(1-loss.detach().mean()) if len(loss) else 0.,
                       teacher_variance=float(selected.var(0, unbiased=False).mean()) if len(selected) else 0.)
    return value, count, diagnostics


class CleanPretrainer(UA2DGalformerPretrainer):
    def __init__(self, config=CleanConfig()):
        super().__init__(config)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261020)
            self.readout = QueryReadout() if config.readout == 'query' else nn.Identity()
        if config.local:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20261021)
                self.local_predictor = nn.Sequential(nn.Linear(512,256), nn.GELU(), nn.Linear(256,512))

    @property
    def weights(self):
        return dict.fromkeys(('mask_2d','mask_3d','contrast','fp') + (('local',) if self.config.local else ()), 1.)

    def target_counts(self, data, labels):
        counts = super().target_counts(data, labels)
        if self.config.local:
            n = len(data.mips_x)
            valid = (torch.bincount(data.incidence_index[1].long(), minlength=n) > 0)
            valid &= data.geometry_valid.bool()[data.canonical_graph_index]
            ids = data.canonical_graph_index[valid]
            counts['local'] = torch.bincount(ids, minlength=data.geometry_valid.numel()).gt(0).sum().float()
        return counts

    def graph2(self, atoms, data):
        n = data.geometry_valid.numel()
        pooled = (self.readout(atoms, data.canonical_graph_index, n) if self.config.readout == 'query'
                  else mean_pool(atoms, data.canonical_graph_index, n))
        return self.norm2(pooled)

    def forward(self, masked, labels, clean):
        if labels['strategy'] != 'mask_align': raise ValueError('wrong masked strategy')
        # Fixed order in all arms; no FP/geometry fields are inputs to O8.
        masked_atoms, _ = self.o8(masked)
        atoms, _ = self.o8(clean)
        masked_bonds, _, _ = self.encoder3d(masked, labels)
        bonds, graph3, valid = self.encoder3d(clean, None)
        if not torch.equal(valid, alignment_valid(clean)): raise ValueError('ALIGN validity mismatch')
        graphs = clean.geometry_valid.numel()
        sums, counts = {}, {}
        selected = labels['atom_selected']
        values = F.cross_entropy(self.atom_head(masked_atoms[selected]).float(), labels['atom_label'][selected], reduction='none')
        sums['mask_2d'], counts['mask_2d'] = task_sum(values, masked.canonical_graph_index[selected], graphs)
        selected = labels['token_selected'] & (masked.bond_length_weight > 0)
        values = F.cross_entropy(self.line_head(masked_bonds[selected]).float(), labels['token_label'][selected], reduction='none')
        sums['mask_3d'], counts['mask_3d'] = task_sum(values, masked.bond_batch[selected], graphs, masked.bond_length_weight[selected])
        graph2 = self.graph2(atoms, clean)
        value, count = alignment_loss(self.proj2(graph2), self.proj3(self.norm3(graph3)), labels['identity'], valid,
                                      temperature=self.config.temperature)
        sums['contrast'], counts['contrast'] = 2 * value, count
        logits = self.fp_head(graph2).float()
        values = balanced_fp_loss(logits, labels['fp_target'])
        sums['fp'], counts['fp'] = values.sum(), values.new_tensor(graphs)
        with torch.no_grad():
            active, truth = logits >= 0, labels['fp_target'].bool()
            intersection = (active & truth).sum(-1)
            diagnostics = dict(fp_recall=float((intersection / truth.sum(-1)).mean()),
                fp_tanimoto=float((intersection / (active | truth).sum(-1)).mean()),
                fp_active_fraction=float(active.float().mean()))
        if self.config.local:
            sums['local'], counts['local'], diag = local_loss(atoms, bonds, clean, self.local_predictor)
            diagnostics.update(diag)
        return dict(sums=sums, counts=counts, diagnostics=diagnostics)


class CleanPure2D(Pure2DGLTModel):
    def __init__(self, readout='mean'):
        super().__init__()
        if readout not in ('mean','query'): raise ValueError('unknown readout')
        self.readout_kind = readout
        self.readout = QueryReadout() if readout == 'query' else nn.Identity()

    def features(self, data):
        atoms, _ = self.o8(data)
        graphs = data.graph_available.numel()
        pooled = (self.readout(atoms, data.canonical_graph_index, graphs) if self.readout_kind == 'query'
                  else mean_pool(atoms, data.canonical_graph_index, graphs))
        return self.norm2(pooled)


def restore_downstream(package, head_state):
    if package.get('schema') != SCHEMA or type(package.get('step')) is not int or package['step'] < 0:
        raise ValueError('unsupported clean FP checkpoint')
    config = CleanConfig(**package['config'])
    model = CleanPure2D(config.readout)
    for name in ('o8','norm2','readout'):
        prefix = name+'.'
        strict_module_state(getattr(model,name), {k[len(prefix):]:v for k,v in package['state_dict'].items() if k.startswith(prefix)})
    strict_module_state(model.head, head_state)
    return model


def deployment_package(model, step):
    return dict(schema=SCHEMA, config=asdict(model.config), step=int(step),
        state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, prediction_paths=['pure_2d'])
