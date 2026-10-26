"""UA2D P10K no-FP ALIGN-ablation objectives; isolated from every legacy campaign.

Every arm of UA2D-P10K-NOFP-ALIGN-20261024-01 removes FP from step0: this schema
has no FP field at all, no FP head exists, and constructing this config with an
``fp_enabled`` key is a hard error rather than a silently ignored extra.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

from .glt_dual import BondPathO8, mean_pool, strict_module_state
from .glt_dual_pretrain import (AlignmentProjection, alignment_loss,
                                alignment_local_anchor_count)
from .ua2d_galformer_pretrain import (ATTENTION_SCALE_REVISION, MaskedGalformer3D,
                                      SCHEMA as ORIGINAL_PT_SCHEMA, alignment_valid, task_sum)

SCHEMA = 'ua2d-p10k-nofp-align-objectives-r1'
# The preregistered objective policy keeps the historical weighted_align_sum_v1
# meaning: the configured lambda multiplies the directional MEAN, and the stored
# ALIGN sum itself still carries the historical factor of two.
LEGACY_OBJECTIVE_POLICY = 'weighted_align_sum_v1'
ALIGNMENT_MULTIPLIER = 2.
LAMBDA_GRID = (0., .05, .10, .20)

# Historical model bases MUST remain unchanged: legacy 3D(k) = 2D(k+2).
# Branch counters advance separately, but seed sets overlap (59998 of 60000).
# Data-side salts are unchanged; v2 only removes exact model seed reuse.
MASK_STREAM_SEED_SALT = 1_000_003
GEOMETRY_STREAM_SEED_SALT = 3_000_003
MODEL_STREAM2D_BASE = 86_400_011
MODEL_STREAM3D_BASE = 86_400_013
STREAM_MODULUS = 2 ** 63 - 1

LEGACY_STREAM_POLICY = 'legacy_offset_v1'
DISJOINT_STREAM_POLICY = 'disjoint_counter_v2'
MODEL_STREAM_BASE = 86_400_010
MODEL_STREAM_LIMIT = 60_000


def model_stream_seed(policy, branch, counter):
    if branch not in ('2d', '3d') or type(counter) is not int or counter < 0:
        raise ValueError('invalid model branch/counter')
    if policy == LEGACY_STREAM_POLICY:
        return ((MODEL_STREAM2D_BASE if branch == '2d' else MODEL_STREAM3D_BASE) + counter) % STREAM_MODULUS
    if policy != DISJOINT_STREAM_POLICY or counter >= MODEL_STREAM_LIMIT:
        raise ValueError('unknown policy or exhausted v2 forward domain')
    return MODEL_STREAM_BASE + 2 * counter + (branch == '3d')


def parse_model_stream_state(state, branch3d):
    """Missing policy means legacy only; never reinterpret an old checkpoint."""
    policy = state.get('policy', LEGACY_STREAM_POLICY)
    if policy not in (LEGACY_STREAM_POLICY, DISJOINT_STREAM_POLICY):
        raise ValueError('unknown model stream policy')
    positions = [state.get('two_d_position'), state.get('three_d_position')]
    if any(type(k) is not int or k < 0 for k in positions) or (not branch3d and positions[1] != 0):
        raise ValueError('invalid model stream positions')
    if policy == LEGACY_STREAM_POLICY:
        for key, expected in (('two_d_base', MODEL_STREAM2D_BASE), ('three_d_base', MODEL_STREAM3D_BASE)):
            if key in state and state[key] != expected:
                raise ValueError('legacy model stream base mismatch')
        if 'B' in state:
            raise ValueError('v2 base cannot describe legacy streams')
    else:
        if (state.get('B') != MODEL_STREAM_BASE or state.get('forward_counter_domain') != [0, MODEL_STREAM_LIMIT]
                or state.get('terminal_counter') != MODEL_STREAM_LIMIT or max(positions) > MODEL_STREAM_LIMIT
                or 'two_d_base' in state or 'three_d_base' in state):
            raise ValueError('v2 model stream metadata/domain mismatch')
    return policy, positions


@dataclass(frozen=True)
class NoFPAlignConfig:
    strategy: str = 'mask_align'
    fusion: str = 'none'
    mask_ratio: float = .4
    noise_sigma: float = .03
    temperature: float = .1
    dropout: float = .1
    align_lambda: float = .10
    weight_warmup: int = 1000
    branch3d: bool = True
    coordinate_unit: str = 'angstrom_div_sigma'
    attention_scale_denominator: int = 512
    revision: str = SCHEMA

    def __post_init__(self):
        if self.revision != SCHEMA or self.coordinate_unit != 'angstrom_div_sigma':
            raise ValueError('invalid revision or coordinate unit')
        if self.strategy != 'mask_align' or self.fusion != 'none':
            raise ValueError('this cycle keeps mask_align only and disables fusion')
        if type(self.align_lambda) is not float or self.align_lambda not in LAMBDA_GRID:
            raise ValueError('align lambda outside the preregistered grid')
        if type(self.branch3d) is not bool:
            raise ValueError('branch3d must be boolean')
        if not self.branch3d and self.align_lambda != 0.:
            raise ValueError('PURE2D has no ALIGN target to weight')
        if type(self.weight_warmup) is not int or self.weight_warmup != 1000:
            raise ValueError('loss-weight warmup must stay the historical 1000')
        if type(self.attention_scale_denominator) is not int or self.attention_scale_denominator not in (64, 512):
            raise ValueError('attention scale denominator must be 64 (1/8) or 512 (1/sqrt(512))')
        if not 0 < self.mask_ratio < 1 or not 0 <= self.dropout < 1:
            raise ValueError('invalid mask/dropout ratio')
        if (not math.isfinite(self.noise_sigma) or self.noise_sigma < 0
                or not math.isfinite(self.temperature) or self.temperature <= 0):
            raise ValueError('invalid noise/temperature')

    @property
    def align_enabled(self):
        return self.branch3d and self.align_lambda > 0.

    @property
    def alignment_multiplier(self):
        return ALIGNMENT_MULTIPLIER

    @property
    def attention_scale(self):
        return self.attention_scale_denominator ** -0.5


def arm_config(arm):
    """The five preregistered arms of this cycle, and nothing else."""
    arms = dict(D_A000=(0., True), D_A005=(.05, True), D_A010=(.10, True),
                D_A020=(.20, True), PURE2D=(0., False))
    if arm not in arms:
        raise ValueError('unregistered no-FP arm: ' + str(arm))
    value, branch3d = arms[arm]
    return NoFPAlignConfig(align_lambda=value, branch3d=branch3d)


def attention_record(config):
    """Branch-aware QK scale record; a pure-2D arm never fabricates 3D rows."""
    if not isinstance(config, NoFPAlignConfig):
        raise ValueError('attention record requires the no-FP config')
    if config.branch3d:
        targets = ['o8.layers[*].attention', 'encoder3d.layers[*].attention']
        blocks = 12
    else:
        targets = ['o8.layers[*].attention']
        blocks = 6
    return dict(revision=ATTENTION_SCALE_REVISION, scope='qk_dot_product',
                scale_denominator=config.attention_scale_denominator,
                scale=config.attention_scale, bias_scaled=False, targets=targets,
                attention_blocks=blocks, branch3d=bool(config.branch3d))


def package_attention_nofp(package):
    """Require an explicit stored record; unlike legacy, omission is an error."""
    config = NoFPAlignConfig(**package['config'])
    expected = attention_record(config)
    stored = package.get('attention')
    if stored is None:
        raise ValueError('missing attention record in no-FP package')
    if stored != expected:
        raise ValueError('package attention scale metadata conflicts with its config')
    return expected


def check_attention_record_nofp(record, expected, where):
    if record is None or record != expected:
        raise ValueError('attention scale record mismatch in ' + where)


def assert_attention_scale_nofp(model):
    """Runtime invariant: 6 O8 blocks always, 6 more 3D blocks only when present."""
    config = model.config
    layers = list(model.o8.layers) + (list(model.encoder3d.layers) if config.branch3d else [])
    if len(layers) != (12 if config.branch3d else 6):
        raise ValueError('unexpected attention block count')
    for layer in layers:
        if (layer.attention.scale != config.attention_scale
                or layer.attention.scale_denominator != config.attention_scale_denominator):
            raise ValueError('attention scale does not match the configured denominator')


@contextmanager
def branch_deterministic_rng(seed, device=None):
    """Seed one branch forward deterministically, then restore the outer stream."""
    devices = []
    if device is not None and device.type == 'cuda':
        index = device.index if device.index is not None else torch.cuda.current_device()
        devices = [torch.device('cuda', index)]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed) % STREAM_MODULUS)
        yield


class UA2DNoFPAlignPretrainer(nn.Module):
    """No FP anywhere; optional 3D branch; optional explicit ALIGN λ (0 = off)."""

    def __init__(self, config=NoFPAlignConfig(), *, stream_policy=LEGACY_STREAM_POLICY):
        super().__init__()
        if not isinstance(config, NoFPAlignConfig):
            raise ValueError('pretrainer requires the no-FP config')
        if stream_policy not in (LEGACY_STREAM_POLICY, DISJOINT_STREAM_POLICY):
            raise ValueError('unknown model stream policy')
        self.stream_policy = stream_policy
        self.config = config
        self.completed_updates = 0
        self.stream2d_position = 0
        self.stream3d_position = 0
        self.o8 = BondPathO8(config.dropout, attention_scale_denominator=config.attention_scale_denominator)
        self.norm2 = nn.LayerNorm(512)
        self.atom_head = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 101))
        if config.branch3d:
            self.encoder3d = MaskedGalformer3D(config.dropout,
                                               attention_scale_denominator=config.attention_scale_denominator)
            self.line_head = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 25755))
        if config.align_enabled:
            # norm3 only normalizes the 3D graph feature for ALIGN; with ALIGN
            # explicitly off it would be an optimizer parameter serving no
            # enabled objective, so it is not constructed at all.
            self.norm3 = nn.LayerNorm(512)
            self.proj2 = AlignmentProjection()
            self.proj3 = AlignmentProjection()

    def branch_device(self):
        return next(self.parameters()).device

    @property
    def weights(self):
        cfg = self.config
        ramp = min(1., self.completed_updates / cfg.weight_warmup)
        weights = dict(mask_2d=1.)
        if cfg.branch3d:
            weights['mask_3d'] = 1.
        if cfg.align_enabled:
            weights['contrast'] = cfg.align_lambda * ramp
        return weights

    def effective_align_coefficient(self):
        """Coefficient actually applied to the ALIGN directional mean."""
        cfg = self.config
        return cfg.alignment_multiplier * cfg.align_lambda * min(1., self.completed_updates / cfg.weight_warmup)

    def target_counts(self, data, labels):
        graphs = data.geometry_valid.numel()

        def count(index):
            return torch.bincount(index, minlength=graphs).gt(0).sum().float()
        selected = labels['atom_selected']
        result = dict(mask_2d=count(data.canonical_graph_index[selected]))
        if self.config.branch3d:
            line = labels['token_selected'] & (data.bond_length_weight > 0)
            result['mask_3d'] = count(data.bond_batch[line])
        if self.config.align_enabled:
            result['contrast'] = alignment_local_anchor_count(
                labels['identity'], alignment_valid(data), device=data.mips_x.device)
        return result

    def forward(self, data, labels):
        if labels['strategy'] != self.config.strategy:
            raise ValueError('sample/model strategy mismatch')
        # Validate both counters before either branch can advance.
        seed2 = model_stream_seed(self.stream_policy, '2d', self.stream2d_position)
        seed3 = model_stream_seed(self.stream_policy, '3d', self.stream3d_position) if self.config.branch3d else None
        device = self.branch_device()
        graphs = data.geometry_valid.numel()
        sums, counts = {}, {}
        with branch_deterministic_rng(seed2, device):
            atoms, _ = self.o8(data)
        self.stream2d_position += 1
        selected = labels['atom_selected']
        loss = F.cross_entropy(self.atom_head(atoms[selected]).float(),
                               labels['atom_label'][selected], reduction='none')
        sums['mask_2d'], counts['mask_2d'] = task_sum(
            loss, data.canonical_graph_index[selected], graphs)
        graph2 = self.norm2(mean_pool(atoms, data.canonical_graph_index, graphs))
        if self.config.branch3d:
            with branch_deterministic_rng(seed3, device):
                bonds, graph3, valid = self.encoder3d(data, labels)
            self.stream3d_position += 1
            selected3 = labels['token_selected'] & (data.bond_length_weight > 0)
            loss3 = F.cross_entropy(self.line_head(bonds[selected3]).float(),
                                    labels['token_label'][selected3], reduction='none')
            sums['mask_3d'], counts['mask_3d'] = task_sum(
                loss3, data.bond_batch[selected3], graphs, data.bond_length_weight[selected3])
            if self.config.align_enabled:
                if not torch.equal(valid, alignment_valid(data)):
                    raise ValueError('ALIGN forward/count validity mismatch')
                value, count = alignment_loss(self.proj2(graph2), self.proj3(self.norm3(graph3)),
                                              labels['identity'], valid,
                                              temperature=self.config.temperature)
                sums['contrast'] = self.config.alignment_multiplier * value
                counts['contrast'] = count
        return dict(sums=sums, counts=counts, diagnostics={})

    def stream_state(self):
        state = dict(policy=self.stream_policy, two_d_position=int(self.stream2d_position),
                     three_d_position=int(self.stream3d_position))
        if self.stream_policy == LEGACY_STREAM_POLICY:
            state.update(two_d_base=MODEL_STREAM2D_BASE, three_d_base=MODEL_STREAM3D_BASE,
                         semantics='legacy branch counters; 3D(k)=2D(k+2); 59998 shared seeds over 60000 forwards; continuous across 5K')
        else:
            state.update(B=MODEL_STREAM_BASE, forward_counter_domain=[0, MODEL_STREAM_LIMIT],
                         terminal_counter=MODEL_STREAM_LIMIT,
                         semantics='seed=B+2*k+branch; 0<=k<60000; counter60000 is save/restore only; continuous across 5K')
        parse_model_stream_state(state, self.config.branch3d)
        return state

    def restore_stream_state(self, state):
        policy, positions = parse_model_stream_state(state, self.config.branch3d)
        self.stream_policy = policy
        self.stream2d_position, self.stream3d_position = positions


def load_nofp_initialization(model, path):
    """Explicit module-subset load of the common step0 + V1 mask heads.

    The subset lists exactly what each arm owns; every excluded tensor belongs
    to a disabled branch (3D encoder, norm3, ALIGN projections or the 3D mask
    head).  Nothing is dropped silently through a permissive strict flag.
    """
    path = Path(path)
    common = torch.load(path, map_location='cpu', weights_only=False)
    if (common.get('schema') != ORIGINAL_PT_SCHEMA or common.get('step') != 0
            or common.get('seed') != 42 or common.get('plan') != 'UA2D-PRETRAIN-FP-20261019-01'):
        raise ValueError('original common step0 identity mismatch')
    heads = torch.load(path.parent / 'task_heads/V1.pt', map_location='cpu', weights_only=False)
    if (heads.get('schema') != ORIGINAL_PT_SCHEMA or heads.get('step') != 0
            or heads.get('strategy') != 'mask_align'):
        raise ValueError('original V1 mask head mismatch')
    source = dict(common['state_dict'], **heads['state_dict'])
    if any(name.startswith('fp_head.') for name in source):
        raise ValueError('common step0 unexpectedly contains FP parameters')
    wanted = {'o8.', 'norm2.', 'atom_head.'}
    if model.config.branch3d:
        wanted |= {'encoder3d.', 'line_head.'}
    if model.config.align_enabled:
        wanted |= {'norm3.', 'proj2.', 'proj3.'}
    subset = {name: value for name, value in source.items() if name.startswith(tuple(wanted))}
    excluded = sorted(name for name in source if name not in subset)
    strict_module_state(model, subset)
    return dict(source=str(path), mask_head=str(path.parent / 'task_heads/V1.pt'),
                loaded_modules=sorted({name.split('.')[0] for name in subset}),
                loaded_tensors=len(subset), excluded_tensors=excluded,
                excluded_semantics='disabled-branch parameters only; never dropped by a permissive loader')


def deployment_package(model, step):
    return dict(schema=SCHEMA, config=asdict(model.config), attention=attention_record(model.config),
                step=int(step), prediction_paths=['pure_2d'], stream_positions=model.stream_state(),
                state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
