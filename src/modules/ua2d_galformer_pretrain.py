"""Galformer-style objectives adapted to atom-node O8 (not a line-graph replica)."""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

from .glt_dual import (BondPathO8, GalformerTrimer3D, Pure2DGLTModel, mean_pool,
                       triplet_type, element_index, strict_module_state)
from .glt_dual_opt import incident_pool
from .glt_dual_pretrain import AlignmentProjection, alignment_loss, alignment_local_anchor_count

SCHEMA = 'ua2d_galformer_objectives_v3'
ATTENTION_SCALE_REVISION = 'ua2d_attention_scale_r1'
LEGACY_OBJECTIVE_POLICY = 'weighted_align_sum_v1'
UNIFORM_OBJECTIVE_POLICY = 'uniform1_align_mean_v1'


@dataclass(frozen=True)
class PretrainConfig:
    strategy: str = 'mask_align'
    fusion: str = 'none'
    mask_ratio: float = .4
    noise_sigma: float = .03
    temperature: float = .1
    dropout: float = .1
    mask_2d_weight: float = 1.
    mask_3d_weight: float = 1.
    contrast_weight: float = .1
    weight_warmup: int = 1000
    coordinate_unit: str = 'angstrom_div_sigma'
    fp_enabled: bool = False
    fp_weight: float = .1
    revision: str = SCHEMA
    coordinate_weight: float = 1.
    length_weight: float = 1.
    angle_weight: float = 1.
    attention_scale_denominator: int = 512
    # Absent in historical checkpoint configs: preserve their original meaning.
    objective_policy: str = LEGACY_OBJECTIVE_POLICY

    def __post_init__(self):
        if self.revision != SCHEMA or self.coordinate_unit != 'angstrom_div_sigma':
            raise ValueError('invalid revision, schedule or coordinate unit')
        if self.objective_policy not in (LEGACY_OBJECTIVE_POLICY, UNIFORM_OBJECTIVE_POLICY):
            raise ValueError('unknown objective policy')
        expected_warmup = 0 if self.objective_policy == UNIFORM_OBJECTIVE_POLICY else 1000
        if type(self.weight_warmup) is not int or self.weight_warmup != expected_warmup:
            raise ValueError('objective policy and loss-weight warmup conflict')
        if type(self.fp_enabled) is not bool:
            raise ValueError('FP enabled must be boolean')
        if self.reconstruct_geometry and self.noise_sigma != .03:
            raise ValueError('normalized coordinate loss requires sigma .03 angstrom')
        if self.strategy not in ('mask_align', 'mask_align_geometry'):
            raise ValueError('unknown pretraining strategy')
        if type(self.attention_scale_denominator) is not int or self.attention_scale_denominator not in (64, 512):
            raise ValueError('attention scale denominator must be 64 (1/8) or 512 (1/sqrt(512))')
        if self.fusion != 'none':
            raise ValueError('V1 and V2 disable 3D-to-2D fusion')
        if not 0 < self.mask_ratio < 1 or not 0 <= self.dropout < 1:
            raise ValueError('invalid mask/dropout ratio')
        if not math.isfinite(self.noise_sigma) or self.noise_sigma < 0 or not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('invalid noise/temperature')
        for name, value in asdict(self).items():
            if name.endswith('_weight') and (not math.isfinite(value) or value <= 0):
                raise ValueError('loss weights must be finite and positive')
            if (self.objective_policy == UNIFORM_OBJECTIVE_POLICY
                    and name.endswith('_weight') and value != 1.):
                raise ValueError('uniform objective policy requires every task weight to equal one')

    @property
    def alignment_multiplier(self):
        return 1. if self.objective_policy == UNIFORM_OBJECTIVE_POLICY else 2.

    @property
    def reconstruct_geometry(self):
        return self.strategy == 'mask_align_geometry'

    @property
    def reconstruct_3d_mask(self):
        return self.strategy == 'mask_align'

    @property
    def attention_scale(self):
        return self.attention_scale_denominator ** -0.5


def attention_record(config):
    """Explicit record of the QK dot-product scale; structural bias is never scaled."""
    return dict(revision=ATTENTION_SCALE_REVISION, scope='qk_dot_product',
                scale_denominator=config.attention_scale_denominator, scale=config.attention_scale,
                bias_scaled=False, targets=['o8.layers[*].attention', 'encoder3d.layers[*].attention'])


def package_attention(package):
    """Config-derived record, rejecting a package whose stored record disagrees.

    A legacy 512-scale package may omit the record; a package that uses the new
    64-scale must carry it, so a silently defaulted config can never pass.
    """
    config = PretrainConfig(**package['config'])
    expected = attention_record(config)
    stored = package.get('attention')
    if stored is None:
        if expected['scale_denominator'] != 512:
            raise ValueError('missing attention scale record in package')
        return expected
    if stored != expected:
        raise ValueError('package attention scale metadata conflicts with its config')
    return expected


def check_attention_record(record, expected, where):
    """Require an explicit record; only a legacy 512 package may omit it."""
    if record is None:
        if expected['scale_denominator'] != 512:
            raise ValueError('missing attention scale record in ' + where)
        return
    if record != expected:
        raise ValueError('attention scale record mismatch in ' + where)


def assert_attention_scale(model):
    """Runtime invariant for every 2D and 3D self-attention block."""
    config = model.config
    layers = list(model.o8.layers) + list(model.encoder3d.layers)
    if len(layers) != 12:
        raise ValueError('unexpected attention block count')
    for layer in layers:
        if (layer.attention.scale != config.attention_scale
                or layer.attention.scale_denominator != config.attention_scale_denominator):
            raise ValueError('attention scale does not match the configured denominator')


class MaskedGalformer3D(GalformerTrimer3D):
    """Hide full line-token inputs and their type-dependent angle-bias identity."""
    def __init__(self, dropout, *, attention_scale_denominator=512):
        super().__init__(dropout, geometry_mode='finite_incident', angle_value_enabled=True,
                         attention_scale_denominator=attention_scale_denominator)
        self.mask_token = nn.Parameter(torch.zeros(512))
        nn.init.normal_(self.mask_token, std=.02)

    def forward(self, data, labels=None):
        za, zb = element_index(data.bond_z_a), element_index(data.bond_z_b)
        distance = data.bond_distance.float()
        if not torch.isfinite(distance).all() or (distance <= 0).any():
            raise ValueError('invalid input distance')
        endpoints = self.endpoint(F.one_hot(za, 101).float()) + self.endpoint(F.one_hot(zb, 101).float())
        radial = self.distance_projection(self.distance_basis(distance, torch.stack([za, zb], -1)))
        states = self.triplet(torch.cat([endpoints, radial], -1))
        types = triplet_type(data.bond_z_a, data.bond_z_b, data.bond_type)
        if labels is not None:
            action, donor = labels['token_action'], labels['token_donor']
            states = torch.where((action == 2)[:, None], states[donor], states)
            types = torch.where(action == 2, types[donor], types)
            states = torch.where((action == 1)[:, None], self.mask_token[None].to(states), states)
            types = torch.where(action == 1, torch.full_like(types, 25857), types)
        bias = mean_pool(self.angle_bias(types, data.line_path, data.line_angle, data.line_mask),
                         data.line_path_group, data.line_source.numel())
        real = (data.line_mask.sum(-1) == 1) & ~data.line_is_self
        angle = self.angle_bias.gaussian(data.line_angle[real, 0], types[data.line_path[real, :2]])
        states = states + mean_pool(self.angle_value(angle), data.line_target[data.line_path_group[real]], len(states))
        for layer in self.layers:
            states = layer(states, data.line_source, data.line_target, bias)
        graph, available = incident_pool(states, data.incidence_index, data.canonical_graph_index,
                                         data.geometry_valid.numel())
        return states, graph, available & data.geometry_valid.bool()


def alignment_valid(data):
    graphs = data.geometry_valid.numel()
    atoms = data.incidence_index[1].long()
    available = torch.bincount(data.canonical_graph_index[atoms], minlength=graphs) > 0
    return data.geometry_valid.bool() & available


def validate_fp(target, graphs):
    if target.shape != (graphs, 2048) or not ((target == 0) | (target == 1)).all():
        raise ValueError('invalid FP binary target')
    n = target.sum(-1)
    if not ((n > 0) & (n < 2048)).all():
        raise ValueError('all-zero/all-one FP target')


def balanced_fp_loss(logits, target):
    validate_fp(target, len(logits))
    positive = target.float()
    negative = 1 - positive
    return .5 * ((F.softplus(-logits.float()) * positive).sum(-1) / positive.sum(-1)
                 + (F.softplus(logits.float()) * negative).sum(-1) / negative.sum(-1))


def task_sum(values, index, count, weights=None):
    """One mean per valid molecule, then sum; retain differentiable empty losses."""
    weights = torch.ones_like(values) if weights is None else weights.to(values)
    total = values.new_zeros(count).index_add(0, index, values * weights)
    denominator = values.new_zeros(count).index_add(0, index, weights)
    valid = denominator > 0
    return (total / denominator.clamp_min(1e-12))[valid].sum(), valid.sum().float()


class UA2DGalformerPretrainer(nn.Module):
    def __init__(self, config=PretrainConfig()):
        super().__init__()
        self.config = config
        self.completed_updates = 0
        self.o8 = BondPathO8(config.dropout, attention_scale_denominator=config.attention_scale_denominator)
        self.encoder3d = MaskedGalformer3D(config.dropout,
                                           attention_scale_denominator=config.attention_scale_denominator)
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
                self.fp_head = nn.Sequential(nn.Linear(512,256), nn.GELU(), nn.Linear(256,2048))

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
            return weights
        ramp = min(1., self.completed_updates / cfg.weight_warmup)
        weights['contrast'] *= ramp
        if cfg.fp_enabled:
            weights['fp'] = cfg.fp_weight * ramp
        return weights

    def target_counts(self, data, labels):
        graphs = data.geometry_valid.numel()
        def count(index):
            return torch.bincount(index, minlength=graphs).gt(0).sum().float()
        selected = labels['atom_selected']
        result = dict(mask_2d=count(data.canonical_graph_index[selected]),
                      contrast=alignment_local_anchor_count(labels['identity'], alignment_valid(data), device=data.mips_x.device))
        if self.config.reconstruct_3d_mask:
            line = labels['token_selected'] & (data.bond_length_weight > 0)
            result['mask_3d'] = count(data.bond_batch[line])
        if self.config.reconstruct_geometry:
            result.update(coordinate=count(labels['physical_graph'][labels['physical_center']]),
                          length=count(data.bond_batch[data.bond_length_weight > 0]),
                          angle=count(labels['angle_graph']))
        if self.config.fp_enabled:
            validate_fp(labels['fp_target'], graphs)
            result['fp'] = torch.tensor(float(graphs), device=data.mips_x.device)
        return result

    def coordinate_prediction(self, bonds, labels, graphs):
        a, b = labels['physical_edges']
        pos = labels['physical_pos']
        relative = pos[a] - pos[b]
        update = self.coordinate_head(bonds).float() * relative
        vector = torch.zeros_like(pos).index_add(0, a, update).index_add(0, b, -update)
        # Same centering convention as target, also for partial physical graphs.
        return vector - mean_pool(vector, labels['physical_graph'], graphs)[labels['physical_graph']]

    def forward(self, data, labels):
        if labels['strategy'] != self.config.strategy:
            raise ValueError('sample/model strategy mismatch')
        atoms, _ = self.o8(data)
        bonds, graph3, valid = self.encoder3d(data, labels if self.config.reconstruct_3d_mask else None)
        graphs = data.geometry_valid.numel()
        graph2 = self.norm2(mean_pool(atoms, data.canonical_graph_index, graphs))
        graph3 = self.norm3(graph3)
        if not torch.equal(valid, alignment_valid(data)):
            raise ValueError('ALIGN forward/count validity mismatch')
        sums, counts = {}, {}
        selected = labels['atom_selected']
        loss = F.cross_entropy(self.atom_head(atoms[selected]).float(), labels['atom_label'][selected], reduction='none')
        sums['mask_2d'], counts['mask_2d'] = task_sum(loss, data.canonical_graph_index[selected], graphs)
        if self.config.reconstruct_3d_mask:
            selected3 = labels['token_selected'] & (data.bond_length_weight > 0)
            loss = F.cross_entropy(self.line_head(bonds[selected3]).float(), labels['token_label'][selected3], reduction='none')
            sums['mask_3d'], counts['mask_3d'] = task_sum(loss, data.bond_batch[selected3], graphs,
                                                     data.bond_length_weight[selected3])
        value, count = alignment_loss(self.proj2(graph2), self.proj3(graph3), labels['identity'], valid,
                                       temperature=self.config.temperature)
        # The new recipe keeps the helper's directional mean. Legacy configs
        # retain their historical directional sum for checkpoint compatibility.
        sums['contrast'], counts['contrast'] = self.config.alignment_multiplier * value, count
        if self.config.reconstruct_geometry:
            mask = data.bond_length_weight > 0
            loss = (self.length_head(bonds[mask]).flatten().float() - labels['length_target'][mask]).square()
            sums['length'], counts['length'] = task_sum(loss, data.bond_batch[mask], graphs, data.bond_length_weight[mask])
            left, right = labels['angle_pairs'].unbind(-1)
            feature = torch.cat([bonds[left] + bonds[right], (bonds[left] - bonds[right]).abs()], -1)
            loss = (self.angle_head(feature).flatten().float().tanh() - labels['angle_target']).square()
            sums['angle'], counts['angle'] = task_sum(loss, labels['angle_graph'], graphs)
            prediction = self.coordinate_prediction(bonds, labels, graphs)
            mask = labels['physical_center']
            loss = ((prediction[mask] - labels['coordinate_target'][mask]) / self.config.noise_sigma).square().mean(-1)
            sums['coordinate'], counts['coordinate'] = task_sum(loss, labels['physical_graph'][mask], graphs)
        diagnostics = {}
        if self.config.fp_enabled:
            target = labels['fp_target']
            logits = self.fp_head(graph2).float()
            loss = balanced_fp_loss(logits, target)
            sums['fp'], counts['fp'] = loss.sum(), loss.new_tensor(graphs)
            with torch.no_grad():
                active = logits >= 0
                positive = target.bool()
                intersection = (active & positive).sum(-1)
                diagnostics = dict(fp_recall=float((intersection / positive.sum(-1)).mean()),
                    fp_active_fraction=float(active.float().mean()),
                    fp_tanimoto=float((intersection / (active | positive).sum(-1)).mean()))
        return dict(sums=sums, counts=counts, diagnostics=diagnostics)


def objective(result, weights, denominators=None):
    """Correct global optimizer-window denominator under DDP gradient averaging."""
    import torch.distributed as dist
    world = dist.get_world_size() if dist.is_initialized() else 1
    if denominators is None:
        denominators = {k: v.detach().clone() for k, v in result['counts'].items()}
        if world > 1:
            for value in denominators.values():
                dist.all_reduce(value)
    if set(weights) != set(result['sums']) or set(denominators) != set(weights):
        raise ValueError('loss task mismatch')
    return sum(weights[k] * value * world / denominators[k].clamp_min(1) for k, value in result['sums'].items())


def deployment_package(model, step):
    return dict(schema=SCHEMA, config=asdict(model.config), attention=attention_record(model.config),
                step=int(step),
                state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                prediction_paths=['pure_2d'])


def initialize_from_ua_package(model, package):
    """Copy existing 512/6/8 UA backbone weights; new task heads stay fresh."""
    state = package['state_dict']
    mapping = [('o8', 'o8'), ('norm2', 'norm2'), ('norm3', 'norm3'), ('encoder3d', 'glt')]
    for target, source in mapping:
        module = getattr(model, target)
        subset = {k[len(source) + 1:]: v for k, v in state.items() if k.startswith(source + '.')}
        if target == 'encoder3d':
            # The one explicitly new tensor has no counterpart in old Galformer.
            subset['mask_token'] = module.mask_token.detach().clone()
        strict_module_state(module, subset)
    return dict(copied=[x for x, _ in mapping],
                new=['encoder3d.mask_token', 'atom_head', 'proj2', 'proj3'] +
                    (['line_head'] if model.config.reconstruct_3d_mask else []) +
                    (['coordinate_head', 'length_head', 'angle_head'] if model.config.reconstruct_geometry else []))


def restore_downstream(package, head_state, *, fused=False):
    if package.get('schema') != SCHEMA or type(package.get('step')) is not int or package['step'] < 0:
        raise ValueError('unsupported pretraining package')
    config = PretrainConfig(**package['config'])
    attention = package_attention(package)
    if fused:
        raise ValueError('V1 and V2 disable 3D-to-2D fusion')
    model = Pure2DGLTModel(attention_scale_denominator=config.attention_scale_denominator)
    if (model.attention_scale != attention['scale'] or len(model.o8.layers) != 6
            or any(layer.attention.scale != attention['scale'] for layer in model.o8.layers)):
        raise ValueError('restored attention scale does not match the package record')
    for name in ('o8', 'norm2'):
        prefix = name + '.'
        state = {k[len(prefix):]: v for k, v in package['state_dict'].items() if k.startswith(prefix)}
        strict_module_state(getattr(model, name), state)
    strict_module_state(model.head, head_state)
    return model
