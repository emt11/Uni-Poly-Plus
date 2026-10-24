"""UA2D-3DSHIFT periodic image relation package (P) and local canonical alignment (L).

The module is intentionally isolated: legacy UA2D entry points never import it, and
every shared low-level block is reused read-only.  Nothing here switches the legacy
periodic template path on; the periodic image graph of this package is built from
the sidecar cache under ``results/ua2d_3dshift_20261022`` only.
"""
from dataclasses import asdict
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter

from .glt_dual import (BondPathO8, GalformerTrimer3D, Pure2DGLTModel, element_index, mean_pool, softmax,
                       strict_module_state, triplet_type)
from .glt_dual_opt import incident_pool
from .glt_dual_pretrain import AlignmentProjection, alignment_local_anchor_count, alignment_loss
from .ua2d_galformer_pretrain import (SCHEMA as BASE_SCHEMA, PretrainConfig, MaskedGalformer3D,
                                      alignment_valid, attention_record, balanced_fp_loss,
                                      package_attention, task_sum, UNIFORM_OBJECTIVE_POLICY)

SHIFT_SCHEMA = 'ua2d-3dshift-local-geometry-r1'
SHIFT_REVISION = 'r1'
PRETRAIN_UPDATES = 10000
GEOMETRY_REVISION = 'ua2d-3dshift-relation-geometry-r1'
SELF_RELATION_REVISION = 'ua2d-periodic-self-image-r1'
CONTEXT_DIM = 128
RBF = 16
LENGTH_WIDTH = 3.0 / (RBF - 1)
ANGLE_WIDTH = math.pi / (RBF - 1)
NEW_PREFIXES = ('encoder3d.geometry.', 'encoder3d.value_layers.')
VARIANTS = {'F_G10K': dict(encoder='base', local=False),
            'P_G': dict(encoder='shift', local=False),
            'F_L': dict(encoder='base', local=True),
            'P_L': dict(encoder='shift', local=True)}


def shift_record(variant='P_L'):
    """Explicit identity of the relative periodic image semantics."""
    if VARIANTS[variant]['encoder'] == 'base':
        return dict(geometry='finite_incident_trimer', relative_shift_only=False,
                    absolute_anchor_in_model=False, self_image_relations=False,
                    value_injection='original_masked_galformer')
    return dict(revision=GEOMETRY_REVISION, geometry='periodic_canonical_bond_image',
                relative_shift_only=True, absolute_anchor_in_model=False,
                self_image_relations=True, self_relation_revision=SELF_RELATION_REVISION,
                two_hop='factorized_one_hop_message', max_abs_shift=None, shift_encoding='relation_identity_only',
                rbf_bins=RBF, value_injection='per_layer_zero_initialized',
                value_dimension=512, layers=6)


def architecture_record(variant='P_L'):
    if VARIANTS[variant]['encoder'] == 'base':
        return dict(image_geometry='finite_incident_trimer', canonical_bond_state=False,
                    covering_graph=False, value_conditioning='original_masked_galformer',
                    local_alignment='physical_center_incidence_stop_gradient' if VARIANTS[variant]['local'] else False,
                    layers=6, dimension=512)
    return dict(image_geometry='relative_periodic_image_shift', canonical_bond_state=True,
                covering_graph='lifted_1_and_2_hop', value_conditioning='relation_geometry',
                local_alignment='canonical_incidence_stop_gradient' if VARIANTS[variant]['local'] else False,
                layers=6, dimension=512)


def _weighted_mean(values, index, weight, count):
    weight = weight.to(values.dtype)
    total = scatter(values * weight[:, None], index, dim=0, dim_size=count, reduce='sum')
    denominator = scatter(weight, index, dim=0, dim_size=count, reduce='sum')
    return total / denominator.clamp_min(1e-12)[:, None]




class RelationGeometry(nn.Module):
    """Encode one observed 1-hop relation geometry (d_source, d_target, angle)."""

    def __init__(self):
        super().__init__()
        self.register_buffer('length_centers', torch.linspace(0., 3., RBF))
        self.register_buffer('angle_centers', torch.linspace(0., math.pi, RBF))
        self.net = nn.Sequential(nn.Linear(3 * RBF, 128), nn.GELU(), nn.Linear(128, 128))
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=.02)
                nn.init.zeros_(layer.bias)

    def _rbf(self, value, centers, width):
        return torch.exp(-0.5 * ((value[:, None] - centers[None]) / width) ** 2)

    def forward(self, distance_source, distance_target, theta, valid):
        feature = torch.cat([self._rbf(distance_source, self.length_centers, LENGTH_WIDTH),
                             self._rbf(distance_target, self.length_centers, LENGTH_WIDTH),
                             self._rbf(theta, self.angle_centers, ANGLE_WIDTH)], -1)
        return self.net(feature) * valid.float()[:, None]


class ShiftedMaskedGalformer3D(MaskedGalformer3D):
    """Canonical bond states on the periodic covering graph with geometry-valued messages."""

    def __init__(self, dropout, *, attention_scale_denominator=512):
        super().__init__(dropout, attention_scale_denominator=attention_scale_denominator)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261022)
            self.geometry = RelationGeometry()
            self.value_layers = nn.ModuleList([nn.Linear(CONTEXT_DIM, 512, bias=False)
                                               for _ in range(6)])
            for layer in self.value_layers:
                nn.init.zeros_(layer.weight)

    def _relation_features(self,theta,source_type,target_type,valid,rel_index,relations,position):
        pair=torch.stack([source_type,target_type],-1)
        encoded=self.angle_bias.position[position](self.angle_bias.gaussian(theta,pair))
        encoded=encoded*valid.float()[:,None]
        denominator=scatter(valid.float(),rel_index,dim=0,dim_size=relations,reduce='sum').clamp_min(1.)[:,None]
        return scatter(encoded,rel_index,dim=0,dim_size=relations,reduce='sum')/denominator

    def _one_hop_bias(self,theta,source_type,target_type,valid,rel_index,relations):
        # Original per-observation position MLP -> heads -> observation pooling.
        encoded=self.angle_bias.position[0](self.angle_bias.gaussian(theta,torch.stack([source_type,target_type],-1)))
        values=self.angle_bias.heads(encoded)*valid.float()[:,None]
        denominator=scatter(valid.float(),rel_index,dim=0,dim_size=relations,reduce='sum').clamp_min(1.)[:,None]
        return scatter(values,rel_index,dim=0,dim_size=relations,reduce='sum')/denominator

    def _block(self, layer, states, source, target, bias, extra):
        attention = layer.attention
        hidden = layer.norm1(states)
        q, k, v = attention.qkv(hidden).reshape(-1, 3, 8, 64).unbind(1)
        score = (q[source].float() * k[target].float()).sum(-1) * attention.scale
        weight = softmax(score + bias.float(), target, num_nodes=states.size(0)).to(v.dtype)
        messages = attention.dropout(weight).unsqueeze(-1) * (v[source] + extra.reshape(-1, 8, 64))
        update = torch.zeros_like(v).index_add(0, target, messages).reshape(-1, 512)
        states = states + layer.dropout(layer.output(update))
        return states + layer.dropout(layer.ffn(layer.norm2(states)))

    def forward(self, data, labels=None):
        if labels is None:
            raise ValueError('3dshift encoder requires the masked labels of the same forward')
        za, zb = element_index(data.bond_z_a), element_index(data.bond_z_b)
        distance = data.bond_distance.float()
        if not torch.isfinite(distance).all() or (distance <= 0).any():
            raise ValueError('invalid input distance')
        endpoints = self.endpoint(F.one_hot(za, 101).float()) + self.endpoint(F.one_hot(zb, 101).float())
        radial = self.distance_projection(self.distance_basis(distance, torch.stack([za, zb], -1)))
        states = self.triplet(torch.cat([endpoints, radial], -1))
        types = triplet_type(data.bond_z_a, data.bond_z_b, data.bond_type)
        action, donor = labels['token_action'], labels['token_donor']
        states = torch.where((action == 2)[:, None], states[donor], states)
        types = torch.where(action == 2, types[donor], types)
        states = torch.where((action == 1)[:, None], self.mask_token[None].to(states), states)
        types = torch.where(action == 1, torch.full_like(types, 25857), types)
        real = (data.line_mask.sum(-1) == 1) & ~data.line_is_self
        angle = self.angle_bias.gaussian(data.line_angle[real, 0], types[data.line_path[real, :2]])
        states = states + mean_pool(self.angle_value(angle), data.line_target[data.line_path_group[real]], len(states))
        group = data.ds_token_group.long()
        canonical = _weighted_mean(states, group, data.ds_token_weight.float(), int(data.ds_group_count))

        positions = labels['physical_pos'].float()
        obs = data.ds_obs_pos.long()
        left = positions[obs[:, 0]] - positions[obs[:, 1]]
        right = positions[obs[:, 2]] - positions[obs[:, 1]]
        cosine = (left * right).sum(-1) / (left.norm(dim=-1) * right.norm(dim=-1)).clamp_min(1e-8)
        theta = torch.acos(cosine.clamp(-1. + 1e-6, 1. - 1e-6))
        obs_source, obs_target = data.ds_obs_source.long(), data.ds_obs_target.long()
        valid = ~((action[obs_source] == 1) | (action[obs_target] == 1))
        context_distance = torch.where(action == 2, distance[donor], distance)
        observed = self.geometry(context_distance[obs_source],
                                 context_distance[obs_target], theta, valid)
        relations = int(data.ds_rel1_count)
        rel_index = data.ds_obs_rel.long()
        total = scatter(observed, rel_index, dim=0, dim_size=relations, reduce='sum')
        denominator = scatter(valid.float(), rel_index, dim=0, dim_size=relations, reduce='sum')
        raw1 = total / denominator.clamp_min(1.)[:, None]
        context1 = raw1
        context1 = torch.where(data.ds_rel1_self.bool()[:, None], torch.zeros_like(context1), context1)
        # Original angle bias retains angles with corrupted MASK/donor TYPES.
        angle_valid = torch.ones_like(valid)
        bias1 = self._one_hop_bias(theta, types[obs_source], types[obs_target], angle_valid, rel_index, relations)
        feature0 = self._relation_features(theta,types[obs_source],types[obs_target],angle_valid,rel_index,relations,0)
        feature1 = self._relation_features(theta,types[obs_source],types[obs_target],angle_valid,rel_index,relations,1)
        bias1 = torch.where(data.ds_rel1_self.bool()[:, None], torch.zeros_like(bias1), bias1)

        second = int(data.ds_rel2_count)
        if second:
            counts = data.ds_rel2_path_count.long()
            path_matrix = data.ds_rel2_path.long()
            path_relation = torch.repeat_interleave(torch.arange(second, device=counts.device), counts)
            pair = .5 * (raw1[path_matrix[:, 0]] + raw1[path_matrix[:, 1]])
            raw2 = scatter(pair, path_relation, dim=0, dim_size=second, reduce='sum') / counts.clamp_min(1.)[:, None]
            # Preserve both path-position MLPs, combine BEFORE the nonlinear heads.
            bias2 = self.angle_bias.heads(.5 * (feature0[path_matrix[:, 0]] + feature1[path_matrix[:, 1]]))
            bias2 = scatter(bias2, path_relation, dim=0, dim_size=second, reduce='sum') / counts.clamp_min(1.)[:, None]
            context2 = raw2
            source = torch.cat([data.ds_rel1_source.long(), data.ds_rel2_source.long()])
            target = torch.cat([data.ds_rel1_target.long(), data.ds_rel2_target.long()])
            bias = torch.cat([bias1, bias2], 0)
            context = torch.cat([context1, context2], 0)
        else:
            source, target, bias, context = (data.ds_rel1_source.long(), data.ds_rel1_target.long(),
                                             bias1, context1)

        for index, layer in enumerate(self.layers):
            canonical = self._block(layer, canonical, source, target, bias, self.value_layers[index](context))
        physical = canonical[group]
        graph, available = incident_pool(canonical, torch.stack([data.ds_inc_bond,data.ds_inc_atom]), data.ds_atom_graph,
                                         data.geometry_valid.numel())
        return dict(token_states=physical, canonical_states=canonical, graph_3d=graph,
                    valid=available & data.geometry_valid.bool(), angle_bias=bias)


def _incidence(data, variant):
    if variant == 'F_L':
        return data.incidence_index[0].long(), data.incidence_index[1].long()
    return data.ds_inc_bond.long(), data.ds_inc_atom.long()


def local_atom_mask(data, labels, atoms, variant):
    """Eligible canonical atoms: selected, observed, and every incident bond action==0."""
    bond, atom = _incidence(data, variant)
    observed = torch.bincount(atom, minlength=atoms) > 0
    if variant == 'F_L':
        masked = torch.bincount(atom, weights=(labels['token_action'][bond] != 0).float(), minlength=atoms) > 0
    else:
        masked = torch.bincount(atom, weights=(data.ds_group_action[bond] != 0).float(), minlength=atoms) > 0
    return labels['atom_selected'].bool() & observed & ~masked


def canonical_teacher(bonds, data, variant, atoms):
    bond, atom = _incidence(data, variant)
    return scatter(bonds[bond], atom, dim=0, dim_size=atoms, reduce='mean')


def local_graph_count(data, labels, atoms, variant):
    mask = local_atom_mask(data, labels, atoms, variant)
    graph = data.canonical_graph_index.long() if variant == 'F_L' else data.ds_atom_graph.long()
    graphs = int(data.geometry_valid.numel())
    return torch.bincount(graph[mask], minlength=graphs).gt(0).sum().float()


class UA2D3DShiftPretrainer(nn.Module):
    """Pretraining model with the periodic image encoder (P) and local alignment (L)."""

    def __init__(self, config=PretrainConfig(), *, variant='P_G'):
        if variant not in VARIANTS:
            raise ValueError('unknown 3dshift variant')
        super().__init__()
        self.config = config
        self.variant = variant
        self.local_alignment = VARIANTS[variant]['local']
        self.completed_updates = 0
        if VARIANTS[variant]['encoder'] == 'base':
            # Reuse the exact frozen BASE encoder module (no new P parameter).
            self.encoder3d = MaskedGalformer3D(config.dropout,
                                               attention_scale_denominator=config.attention_scale_denominator)
        else:
            self.encoder3d = ShiftedMaskedGalformer3D(config.dropout,
                                                      attention_scale_denominator=config.attention_scale_denominator)
        self.o8 = BondPathO8(config.dropout, attention_scale_denominator=config.attention_scale_denominator)
        self.norm2 = nn.LayerNorm(512)
        self.norm3 = nn.LayerNorm(512)
        self.proj2, self.proj3 = AlignmentProjection(), AlignmentProjection()
        self.atom_head = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 101))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261019)
            if config.reconstruct_3d_mask:
                self.line_head = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 25755))
            if config.reconstruct_geometry:
                self.coordinate_head = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))
                nn.init.zeros_(self.coordinate_head[-1].weight)
                nn.init.zeros_(self.coordinate_head[-1].bias)
                self.length_head = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))
                self.angle_head = nn.Sequential(nn.Linear(1024, 128), nn.GELU(), nn.Linear(128, 1))
        if config.fp_enabled:
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(20261019)
                self.fp_head = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 2048))

    @property
    def weights(self):
        cfg = self.config
        weights = dict(mask_2d=cfg.mask_2d_weight, contrast=cfg.contrast_weight)
        if cfg.reconstruct_3d_mask:
            weights['mask_3d'] = cfg.mask_3d_weight
        if cfg.reconstruct_geometry:
            weights.update(coordinate=cfg.coordinate_weight, length=cfg.length_weight, angle=cfg.angle_weight)
        if cfg.objective_policy == UNIFORM_OBJECTIVE_POLICY:
            if cfg.fp_enabled:
                weights['fp'] = cfg.fp_weight
            if self.local_alignment:
                weights['local'] = 1.
            return weights
        ramp = min(1., self.completed_updates / cfg.weight_warmup)
        weights['contrast'] *= ramp
        if cfg.fp_enabled:
            weights['fp'] = cfg.fp_weight * ramp
        if self.local_alignment:
            weights['local'] = .1 * min(1., self.completed_updates / 1000.)
        return weights

    def target_counts(self, data, labels):
        graphs = data.geometry_valid.numel()

        def count(index):
            return torch.bincount(index, minlength=graphs).gt(0).sum().float()

        selected = labels['atom_selected']
        result = dict(mask_2d=count(data.canonical_graph_index[selected]),
                      contrast=alignment_local_anchor_count(labels['identity'], alignment_valid(data),
                                                            device=data.mips_x.device))
        if self.config.reconstruct_3d_mask:
            line = labels['token_selected'] & (data.bond_length_weight > 0)
            result['mask_3d'] = count(data.bond_batch[line])
        if self.config.reconstruct_geometry:
            result.update(coordinate=count(labels['physical_graph'][labels['physical_center']]),
                          length=count(data.bond_batch[data.bond_length_weight > 0]),
                          angle=count(labels['angle_graph']))
        if self.config.fp_enabled:
            result['fp'] = torch.tensor(float(graphs), device=data.mips_x.device)
        if self.local_alignment:
            atoms = int(selected.numel())
            result['local'] = local_graph_count(data, labels, atoms, self.variant)
        return result

    def forward(self, data, labels):
        if labels['strategy'] != self.config.strategy:
            raise ValueError('sample/model strategy mismatch')
        if not self.config.reconstruct_3d_mask:
            raise ValueError('3dshift requires the mask_align objectives')
        atoms, _ = self.o8(data)
        out = self.encoder3d(data, labels)
        if isinstance(out, dict):
            bonds, graph3, valid = out['token_states'], out['graph_3d'], out['valid']
            canonical_states = out['canonical_states']
        else:
            bonds, graph3, valid = out
            canonical_states = None
        graphs = data.geometry_valid.numel()
        graph2 = self.norm2(mean_pool(atoms, data.canonical_graph_index, graphs))
        graph3 = self.norm3(graph3)
        if not torch.equal(valid, alignment_valid(data)):
            raise ValueError('ALIGN forward/count validity mismatch')
        sums, counts = {}, {}
        selected = labels['atom_selected']
        loss = F.cross_entropy(self.atom_head(atoms[selected]).float(), labels['atom_label'][selected],
                               reduction='none')
        sums['mask_2d'], counts['mask_2d'] = task_sum(loss, data.canonical_graph_index[selected], graphs)
        selected3 = labels['token_selected'] & (data.bond_length_weight > 0)
        loss = F.cross_entropy(self.line_head(bonds[selected3]).float(), labels['token_label'][selected3],
                               reduction='none')
        sums['mask_3d'], counts['mask_3d'] = task_sum(loss, data.bond_batch[selected3], graphs,
                                                     data.bond_length_weight[selected3])
        value, count = alignment_loss(self.proj2(graph2), self.proj3(graph3), labels['identity'], valid,
                                      temperature=self.config.temperature)
        sums['contrast'], counts['contrast'] = self.config.alignment_multiplier * value, count
        if self.config.fp_enabled:
            target = labels['fp_target']
            logits = self.fp_head(graph2).float()
            loss = balanced_fp_loss(logits, target)
            sums['fp'], counts['fp'] = loss.sum(), loss.new_tensor(graphs)
        if self.local_alignment:
            atoms_count = atoms.size(0)
            mask = local_atom_mask(data, labels, atoms_count, self.variant)
            if mask.any():
                source = canonical_states if self.variant != 'F_L' else bonds
                teacher = canonical_teacher(source, data, self.variant, atoms_count)[mask]
                student = self.proj2(self.norm2(atoms[mask]))
                target = self.proj3(self.norm3(teacher)).detach()
                loss = 1. - (student * target).sum(-1)
                graph = (data.canonical_graph_index[mask].long() if self.variant == 'F_L'
                         else data.ds_atom_graph[mask].long())
                sums['local'], counts['local'] = task_sum(loss, graph, graphs)
            else:
                sums['local'] = atoms.sum() * 0.
                counts['local'] = atoms.new_tensor(0.)
        return dict(sums=sums, counts=counts, diagnostics={})


def strict_load_shift_initialization(model, directory):
    """Load the frozen fast_r4 common step0 + V1 mask head; only listed P tensors stay fresh."""
    from pathlib import Path
    directory = Path(directory)
    common = torch.load(directory / 'common_step0.pt', map_location='cpu', weights_only=False)
    head = torch.load(directory / 'task_heads' / 'V1.pt', map_location='cpu', weights_only=False)
    if (common.get('schema') != BASE_SCHEMA or common.get('step') != 0 or common.get('seed') != 42
            or common.get('plan') != 'UA2D-PRETRAIN-FP-20261019-01'):
        raise ValueError('invalid common step0 package')
    if head.get('schema') != BASE_SCHEMA or head.get('step') != 0 or head.get('strategy') != model.config.strategy:
        raise ValueError('original V1 mask-head identity mismatch')
    state = dict(common['state_dict'])
    state.update(head['state_dict'])
    current = model.state_dict()
    # Exactly the original independent FP-head stream, not part of common_step0.
    state.update({k:v for k,v in current.items() if k.startswith('fp_head.')})
    missing = [key for key in current if key not in state]
    allowed = [key for key in missing if key.startswith(NEW_PREFIXES)]
    if len(allowed) != len(missing):
        raise ValueError('common initialisation is missing a shared tensor: ' +
                         str(sorted(set(missing) - set(allowed))[:4]))
    for key in allowed:
        state[key] = current[key]
    extra = [key for key in state if key not in current]
    if extra:
        raise ValueError('common initialisation carries unknown tensors: ' + str(sorted(extra)[:4]))
    strict_module_state(model, state)
    return dict(copied=len(current) - len(allowed), fresh=sorted(allowed), fp_initialization='independent seed20261019', fp_keys=sorted(k for k in current if k.startswith('fp_head.')))


def shift_deployment_package(model, step):
    return dict(schema=SHIFT_SCHEMA, config=asdict(model.config), attention=attention_record(model.config),
                step=int(step), variant=model.variant, local_alignment=bool(model.local_alignment),
                architecture=architecture_record(model.variant), shift=shift_record(model.variant),
                state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                prediction_paths=['pure_2d'], outer_test='NOT_RUN')


def check_shift_package(package, where):
    if package.get('schema') != SHIFT_SCHEMA:
        raise ValueError('unsupported 3dshift package schema in ' + where)
    if package.get('variant') not in VARIANTS:
        raise ValueError('unknown 3dshift variant in ' + where)
    if package.get('step') != PRETRAIN_UPDATES:
        raise ValueError('3dshift package is not a completed step10000 checkpoint')
    if package.get('architecture') != architecture_record(package['variant']) or package.get('shift') != shift_record(package['variant']):
        raise ValueError('3dshift architecture/geometry record mismatch in ' + where)
    config = PretrainConfig(**package['config'])
    if config.attention_scale_denominator != 512:
        raise ValueError('3dshift keeps the original 512 attention denominator')
    package_attention(package)
    return config


def restore_downstream_shift(package, head_state):
    """Recreate the pure-2D downstream model from a P_G/P_L package (o8 + norm2 + head)."""
    config = check_shift_package(package, 'deployment package')
    attention = package_attention(package)
    model = Pure2DGLTModel(attention_scale_denominator=config.attention_scale_denominator)
    for name in ('o8', 'norm2'):
        prefix = name + '.'
        subset = {k[len(prefix):]: v for k, v in package['state_dict'].items() if k.startswith(prefix)}
        strict_module_state(getattr(model, name), subset)
    if model.attention_scale != attention['scale']:
        raise ValueError('restored 2D attention scale does not match the package record')
    for layer in model.o8.layers:
        if layer.attention.scale != attention['scale']:
            raise ValueError('restored 2D layer attention scale mismatch')
    strict_module_state(model.head, head_state)
    return model
