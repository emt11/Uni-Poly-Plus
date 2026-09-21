"""GLT-PH end-to-end fusion candidates (family B first, R0/R2D references).

Family B keeps the O8 and GLT mathematics untouched and inserts a spatial
message block after GLT layer 3 and layer 5.  Every scale shares one message
MLP and one affine-free normalisation; the scale router is driven by the
*declared conditional input* of the arm (CONST / STAT / PH), whose only role is
to pick the scale weights.  The router's last layer is zero-initialised, so the
initial routing is exactly uniform and no learnable PH on/off gate exists.

``R0`` is the same trunk with no spatial block and no conditional input, and
``R2D`` is the O8/CLS2 trunk alone (no GLT, no CL, no PH): the two references
the plan compares against.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .glt_dual import BondPathO8, GalformerTrimer3D, mean_pool
from .glt_dual_pretrain import alignment_loss
from .glt_galformer_ph import CLS_SEED, GLTGalPH, PH_BINS, PH_CHANNELS, PH_PATCHES, _head
from ..dataset.glt_ph_fusion_inputs import ARMS, EDGE_FEATURE_DIM, edge_rbf

FAMILY_SEEDS = {'A': 20260923, 'B': 20260924, 'C': 20260922}
SPATIAL_SCALES = 3
SPATIAL_HIDDEN = 128
MESSAGE_INPUT_DIM = 2 * SPATIAL_HIDDEN + EDGE_FEATURE_DIM + SPATIAL_HIDDEN  # 417
INSERT_AFTER_LAYERS = (3, 5)
CONTRASTIVE_TEMPERATURE = 0.1
SPATIAL_WRITE_SCALE = 0.1
TRAINING_ONLY_PREFIXES = ('head_2d.', 'head_3d.', 'cl_proj2.', 'cl_proj3.', 'ph_head.')
TRAINING_ONLY_MODULES = ('head_2d', 'head_3d', 'cl_proj2', 'cl_proj3', 'ph_head')


def _xavier(module):
    for item in module.modules():
        if isinstance(item, nn.Linear):
            nn.init.xavier_uniform_(item.weight)
            if item.bias is not None:
                nn.init.zeros_(item.bias)


class ConditionalProfileEncoder(nn.Module):
    """[G,3,32] conditional profile -> [G,128] (plan section 3.2)."""

    def __init__(self, hidden=SPATIAL_HIDDEN, patches=8, patch_dim=12, heads=4,
                 ffn=512, dropout=0.1):
        super().__init__()
        self.patch = nn.Sequential(nn.Linear(patch_dim, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden))
        self.scale_embedding = nn.Parameter(torch.zeros(patches, hidden))
        self.norm = nn.LayerNorm(hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=ffn, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        _xavier(self)
        nn.init.normal_(self.scale_embedding, std=0.02)
        self.hidden = hidden

    def patchify(self, profile):
        batch = profile.reshape(-1, 3, 32)
        shaped = batch.reshape(batch.shape[0], 3, 8, 4)
        return shaped.permute(0, 2, 1, 3).reshape(batch.shape[0], 8, 12)

    def forward(self, profile):
        content = self.patch(self.patchify(profile.float()))
        token = self.norm(content + self.scale_embedding.unsqueeze(0))
        return self.encoder(token).mean(1)


def incidence_mean(bond_states, bond_atom_index, atom_count):
    """Mean of the incident bond states per atom; an atom with none gets zero."""
    if bond_states.numel() == 0 or bond_atom_index.numel() == 0:
        return bond_states.new_zeros((atom_count, bond_states.size(-1)))
    out = bond_states.new_zeros((atom_count, bond_states.size(-1)))
    count = bond_states.new_zeros((atom_count,))
    for row in (0, 1):
        index = bond_atom_index[row]
        out.index_add_(0, index, bond_states)
        count.index_add_(0, index, torch.ones_like(index, dtype=count.dtype))
    return out / count.clamp_min(1.0).unsqueeze(-1)


class SpatialMessageBlock(nn.Module):
    """Three nested scales, one shared message MLP, one scale router (section 5.2)."""

    def __init__(self, hidden=512, dim=SPATIAL_HIDDEN, scales=SPATIAL_SCALES,
                 dropout=0.1, chunk=16384):
        super().__init__()
        self.scales = int(scales)
        self.dim = int(dim)
        # Edges are processed in exact chunks: same edges, same radius, same
        # order, only the message MLP's working set is bounded.
        self.chunk = int(chunk)
        self.atom_proj = nn.Linear(hidden, dim)
        self.atom_norm = nn.LayerNorm(dim)
        self.message = nn.Sequential(nn.Linear(MESSAGE_INPUT_DIM, 256), nn.GELU(),
                                     nn.Linear(256, dim))
        self.message_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.scale_embedding = nn.Parameter(torch.zeros(self.scales, dim))
        self.router = nn.Sequential(nn.Linear(dim, dim), nn.GELU(),
                                    nn.Linear(dim, self.scales))
        self.bond_proj = nn.Linear(dim, hidden)
        _xavier(self)
        nn.init.normal_(self.scale_embedding, std=0.02)
        # Uniform initial routing: the router output starts at exactly zero.
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def forward(self, bond_states, bond_count, data, conditional):
        atom_count = int(data.atom_batch.numel())
        root = bond_states[:bond_count]
        atoms = self.atom_norm(self.atom_proj(
            incidence_mean(root, data.bond_atom_index.long(), atom_count)))
        edge = data.spatial_edge_index.long()
        scale = data.spatial_scale.long()
        rbf = edge_rbf(data.spatial_distance, data.spatial_bonded).to(root.dtype)
        messages = []
        for index in range(self.scales):
            selected = scale == index
            if int(selected.sum()) == 0:
                messages.append(root.new_zeros((atom_count, self.dim)))
                continue
            target, source = edge[1][selected], edge[0][selected]
            pooled = root.new_zeros((atom_count, self.dim))
            counts = root.new_zeros((atom_count,))
            for start in range(0, int(target.numel()), self.chunk):
                stop = min(start + self.chunk, int(target.numel()))
                rows = slice(start, stop)
                feature = torch.cat([
                    atoms[target[rows]], atoms[source[rows]], rbf[selected][rows],
                    self.scale_embedding[index].expand(stop - start, -1)], -1)
                value = self.message(feature)
                # Autocast may hand back bf16 while the accumulator is fp32.
                pooled.index_add_(0, target[rows], value.to(pooled.dtype))
                counts.index_add_(0, target[rows],
                                  torch.ones_like(target[rows], dtype=counts.dtype))
            messages.append(self.message_norm(pooled / counts.clamp_min(1.0).unsqueeze(-1)))
        routing = F.softmax(self.router(conditional.float()), dim=-1)
        weight = routing[data.atom_batch.long()].to(root.dtype)
        combined = sum(weight[:, index:index + 1] * messages[index]
                       for index in range(self.scales))
        endpoints = data.bond_atom_index.long()
        update = self.bond_proj(combined[endpoints[0]] + combined[endpoints[1]])
        updated = root + SPATIAL_WRITE_SCALE * update.to(root.dtype)
        with torch.no_grad():
            norms = [float(value.detach().norm(dim=-1).mean()) for value in messages]
            cosine = {}
            for left in range(self.scales):
                for right in range(left + 1, self.scales):
                    cosine[f'{left}-{right}'] = float(F.cosine_similarity(
                        messages[left].detach(), messages[right].detach(),
                        dim=-1).mean())
            metrics = {
                'routing_mean': routing.detach().mean(0).tolist(),
                'routing_graph_variance': routing.detach().var(0, unbiased=False).tolist(),
                'message_norms': norms, 'message_cosine': cosine,
                'message_norm': float(combined.detach().norm(dim=-1).mean()),
                'bond_update_relative': (float(update.detach().norm())
                                         / max(float(root.detach().norm()), 1e-12)),
                'spatial_edge_count': int(edge.size(1)),
            }
        return updated, metrics


class GLTFusionB(GLTGalPH):
    """R0 trunk plus the two spatial blocks (family B, three arms)."""

    architecture_name = 'O8-GalformerTrimer-PhFusionB'

    def __init__(self, arm, profile_stats, dropout=0.1, scales=SPATIAL_SCALES,
                 insert_after=INSERT_AFTER_LAYERS):
        if arm not in ARMS:
            raise ValueError(f'arm must be one of {ARMS}')
        missing = [name for name in ('mean', 'std') if name not in profile_stats]
        if missing:
            raise ValueError(f'profile statistics are missing {missing}')
        stats = {name: torch.as_tensor(profile_stats[name], dtype=torch.float32)
                 for name in ('mean', 'std')}
        super().__init__(summary_mode='cls', ph_mode=None, dropout=dropout)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(FAMILY_SEEDS['B'])
            self.conditional = ConditionalProfileEncoder()
            self.spatial = nn.ModuleList([SpatialMessageBlock(dropout=dropout)
                                          for _ in insert_after])
        self.arm = arm
        self.insert_after = tuple(int(value) for value in insert_after)
        self.register_buffer('profile_mean', stats['mean'].clone())
        self.register_buffer('profile_std', stats['std'].clone())

    def conditional_input(self, data):
        value = data.profile_input.float()
        return self.conditional((value - self.profile_mean) / self.profile_std)

    def forward(self, data):
        graphs = data.graph_available.numel()
        atom_graph = data.canonical_graph_index.long()
        atoms, bias2, source2, target2 = self._o8_states(data)
        atoms, bias2, source2, target2 = self._append_cls(
            atoms, atom_graph, graphs, bias2, source2, target2, self.cls_2d)
        for layer in self.o8.layers:
            atoms = layer(atoms, source2, target2, bias2)
        n_atom = data.mips_x.size(0)
        atom_hidden = atoms[:n_atom] * data.graph_available.bool()[atom_graph].unsqueeze(-1)
        cls2 = atoms[n_atom:]

        bonds, bias3, source3, target3, valid3 = self._glt_states(data, graphs)
        bond_graph = data.bond_batch.long()
        bonds, bias3, source3, target3 = self._append_cls(
            bonds, bond_graph, graphs, bias3, source3, target3, self.cls_3d)
        n_bond = int(data.bond_distance.numel())
        conditional = self.conditional_input(data)
        spatial_metrics = []
        for index, layer in enumerate(self.glt.layers):
            bonds = layer(bonds, source3, target3, bias3)
            position = index + 1
            if position in self.insert_after:
                block = self.spatial[self.insert_after.index(position)]
                updated, row = block(bonds, n_bond, data, conditional)
                bonds = torch.cat([updated, bonds[n_bond:]], 0)
                row['after_layer'] = position
                spatial_metrics.append(row)
        bond_hidden = bonds[:n_bond]
        cls3 = bonds[n_bond:]
        summary3 = torch.where(valid3.unsqueeze(-1), cls3, torch.zeros_like(cls3))
        return {'g2': cls2, 'g3': summary3, 'cls2': cls2, 'cls3': cls3,
                'atom_states': atom_hidden, 'bond_states': bond_hidden,
                'line3d_valid': valid3, 'ph_summary': None,
                'conditional': conditional, 'spatial_metrics': spatial_metrics}


class R2DModel(nn.Module):
    """O8/CLS2 trunk alone: the 2D-only reference (no GLT, no CL, no PH)."""

    architecture_name = 'O8-GalformerTrimer-R2D'
    summary_mode = 'cls'
    ph_mode = None

    def __init__(self, dropout=0.1):
        super().__init__()
        self.o8 = BondPathO8(dropout)
        self.mask_2d_embedding = nn.Parameter(torch.zeros(512))
        self.head_2d = _head(512, 512, 101)
        nn.init.normal_(self.mask_2d_embedding, std=0.02)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(CLS_SEED)
            self.cls_2d = nn.Parameter(torch.zeros(512))
            self.virtual_to_real_bias = nn.Parameter(torch.zeros(8))
            self.real_to_virtual_bias = nn.Parameter(torch.zeros(8))
            self.virtual_self_bias = nn.Parameter(torch.zeros(8))
            nn.init.normal_(self.cls_2d, std=0.02)

    def _o8_states(self, data):
        return GLTGalPH._o8_states(self, data)

    def _donor_embeddings(self, data):
        return GLTGalPH._donor_embeddings(self, data)

    def _append_cls(self, states, graph_index, graphs, bias, source, target, cls):
        return GLTGalPH._append_cls(self, states, graph_index, graphs, bias,
                                    source, target, cls)

    def forward(self, data):
        graphs = data.graph_available.numel()
        atom_graph = data.canonical_graph_index.long()
        atoms, bias2, source2, target2 = self._o8_states(data)
        atoms, bias2, source2, target2 = self._append_cls(
            atoms, atom_graph, graphs, bias2, source2, target2, self.cls_2d)
        for layer in self.o8.layers:
            atoms = layer(atoms, source2, target2, bias2)
        n_atom = data.mips_x.size(0)
        atom_hidden = atoms[:n_atom] * data.graph_available.bool()[atom_graph].unsqueeze(-1)
        cls2 = atoms[n_atom:]
        empty = atoms.new_zeros((0, atoms.size(-1)))
        return {'g2': cls2, 'g3': cls2, 'cls2': cls2, 'cls3': None,
                'atom_states': atom_hidden, 'bond_states': empty,
                'line3d_valid': torch.zeros(graphs, dtype=torch.bool,
                                            device=atoms.device),
                'ph_summary': None}


class FusionPretrainer(nn.Module):
    """The plan's main objective for R0/B (three terms) and R2D (2D CE only)."""

    def __init__(self, model, temperature=CONTRASTIVE_TEMPERATURE, objective='R0'):
        super().__init__()
        if objective not in ('R0', 'R2D'):
            raise ValueError('objective must be R0 or R2D')
        self.model = model
        self.objective = objective
        self.temperature = float(temperature)

    def forward(self, data, labels, world_size=1):
        out = self.model(data)
        zero = out['g2'].sum() * 0.0
        total_2d = int(labels['label_2d'].numel())
        if total_2d:
            logits_2d = self.model.head_2d(out['atom_states'][data.mask2d_rows])
            loss_2d = F.cross_entropy(logits_2d.float(), labels['label_2d'])
            sum_2d = F.cross_entropy(logits_2d.float(), labels['label_2d'], reduction='sum')
        else:
            logits_2d, loss_2d, sum_2d = None, zero, zero
        logits_3d = None
        loss_3d = zero
        sum_3d = zero
        total_3d = int(labels['label_3d'].numel()) if self.objective == 'R0' else 0
        if self.objective == 'R0' and total_3d:
            logits_3d = self.model.head_3d(out['bond_states'][data.mask3d_rows])
            loss_3d = F.cross_entropy(logits_3d.float(), labels['label_3d'])
            sum_3d = F.cross_entropy(logits_3d.float(), labels['label_3d'], reduction='sum')
        elif self.objective == 'R0':
            total_3d = 0
        cl_sum, cl_count = zero, torch.zeros((), dtype=torch.long, device=zero.device)
        if self.objective == 'R0':
            valid = out['line3d_valid'].bool() & data.graph_available.bool()
            cl_sum, cl_count = alignment_loss(
                self.model.cl_proj2(out['g2']), self.model.cl_proj3(out['g3']),
                labels['identity'], valid, temperature=self.temperature)
        return {
            'sums': torch.stack([sum_2d, sum_3d, cl_sum, zero]),
            'counts': torch.stack([
                torch.tensor(total_2d, dtype=torch.long, device=zero.device),
                torch.tensor(total_3d, dtype=torch.long, device=zero.device),
                cl_count.to(torch.long),
                torch.zeros((), dtype=torch.long, device=zero.device)]),
            'loss_2d': loss_2d, 'loss_3d': loss_3d,
            'loss_cl': cl_sum / cl_count.clamp_min(1).to(cl_sum.dtype),
            'loss_ph': zero,
            'logits_2d': logits_2d, 'logits_3d': logits_3d,
            'objective': self.objective, 'graph_count': data.graph_available.numel(),
        }


class FusionDownstream(nn.Module):
    """Plan section 7.1 readout: gated DUAL fusion, or the R2D 2D_ONLY branch."""

    architecture_name = 'PhFusion-Downstream'
    READOUTS = ('DUAL', '2D_ONLY')

    def __init__(self, encoder, readout='DUAL', hidden=256, dropout=0.1):
        super().__init__()
        if readout not in self.READOUTS:
            raise ValueError('readout must be DUAL or 2D_ONLY')
        self.encoder = encoder
        self.readout = readout
        self.cls_readout = getattr(encoder, 'summary_mode', None) == 'cls'
        if self.cls_readout:
            self.readout2 = nn.Linear(1024, 512)
            if readout == 'DUAL':
                self.readout3 = nn.Linear(1024, 512)
        if readout == 'DUAL':
            self.gate = nn.Sequential(nn.Linear(1024, hidden), nn.GELU(),
                                      nn.Linear(hidden, 1))
            self.proj3 = nn.Linear(512, 512)
            self.norm = nn.LayerNorm(512)
        else:
            self.norm = nn.LayerNorm(512)
        self.head = nn.Sequential(nn.Linear(512, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def freeze_training_only_heads(self):
        frozen = []
        for name in TRAINING_ONLY_MODULES:
            module = getattr(self.encoder, name, None)
            if module is not None:
                module.requires_grad_(False)
                frozen.append(name)
        return frozen

    def _with_conditional_placeholder(self, batch, graphs):
        """Supply the declared placeholders a downstream batch may not carry."""
        encoder = self.encoder
        if hasattr(encoder, 'arm'):
            if not hasattr(batch, 'profile_input'):
                raise ValueError('a fusion encoder requires the declared conditional input')
            if not hasattr(batch, 'profile_valid'):
                batch.profile_valid = torch.ones(graphs, dtype=torch.bool,
                                                 device=batch.profile_input.device)
        if getattr(encoder, 'ph_mode', None) == 'global' and not hasattr(batch, 'ph_profile'):
            # CURRENT anchor (r6 F_OFF architecture): a batch without PH carries
            # the invalid placeholder, which zeroes the pretraining-side residual.
            device = batch.mips_x.device
            batch.ph_profile = torch.zeros((graphs, PH_CHANNELS, PH_BINS),
                                           dtype=torch.float32, device=device)
            batch.ph_mask = torch.zeros((graphs, PH_PATCHES), dtype=torch.bool, device=device)
            batch.ph_valid = torch.zeros(graphs, dtype=torch.bool, device=device)
        return batch

    def summarize(self, batch):
        out = self.encoder(batch)
        graphs = int(batch.graph_available.numel())
        if self.cls_readout:
            mean2 = mean_pool(out['atom_states'], batch.canonical_graph_index.long(), graphs)
            r2 = self.readout2(torch.cat([out['cls2'], mean2], -1))
            if self.readout == 'DUAL':
                mean3 = mean_pool(out['bond_states'], batch.bond_batch.long(), graphs)
                r3 = self.readout3(torch.cat([out['cls3'], mean3], -1))
        else:
            r2 = out['g2']
            r3 = out['g3'] if self.readout == 'DUAL' else None
        if self.readout == '2D_ONLY':
            fused = self.norm(r2)
            aux = {'readout': '2D_ONLY',
                   'ph_downstream': 'not_used_by_2d_only'}
            return fused, aux
        gate = torch.sigmoid(self.gate(torch.cat([r2, r3], -1)))
        valid = out['line3d_valid'].bool().unsqueeze(-1)
        fused = self.norm(r2 + valid * gate * self.proj3(r3))
        aux = {'readout': 'DUAL', 'gate_mean': float(gate.detach().mean()),
               'ph_downstream': 'absent_zero_residual',
               'spatial_metrics': out.get('spatial_metrics')}
        return fused, aux

    def forward(self, batch):
        graphs = int(batch.graph_available.numel())
        batch = self._with_conditional_placeholder(batch, graphs)
        fused, aux = self.summarize(batch)
        return self.head(fused), aux


def fusion_deployment_package(pretrainer, step, *, arm, extra=None):
    """Inference package: encoders and CLS tokens, no training-only heads."""
    model = pretrainer.model
    state = {name: value.detach().cpu().clone()
             for name, value in model.state_dict().items()
             if not name.startswith(TRAINING_ONLY_PREFIXES)}
    package = dict(architecture=model.architecture_name, arm=arm, step=int(step),
                   objective=getattr(pretrainer, 'objective', None),
                   state_dict=state)
    package.update(extra or {})
    return package


def load_fusion_deployment(model, package, expected_step=None):
    """Strict load: exact key set, explicit architecture and step identity."""
    if package.get('architecture') != model.architecture_name:
        raise ValueError('fusion checkpoint architecture mismatch')
    if expected_step is not None and int(package.get('step', -1)) != int(expected_step):
        raise ValueError('fusion checkpoint step mismatch')
    current = model.state_dict()
    expected = {name for name in current if not name.startswith(TRAINING_ONLY_PREFIXES)}
    if set(package['state_dict']) != expected:
        missing = sorted(expected - set(package['state_dict']))
        extra = sorted(set(package['state_dict']) - expected)
        raise ValueError(f'fusion deployment tensor mismatch: missing={missing[:5]} '
                         f'extra={extra[:5]}')
    current.update(package['state_dict'])
    model.load_state_dict(current, strict=True)
    return package
