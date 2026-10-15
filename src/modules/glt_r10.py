"""Explicit modules and contracts for the fixed r10 experiment arms."""
import copy
import json
from pathlib import Path
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/glt_optimization_20261011/r10'
COMMON = ROOT / 'results/glt_periodic_seam_20261009/r1/initialization/F_CE_step0.pt'
RUNTIME = 'glt_r10_e0_cache_global_loop_v1'
REVISION = 'glt_optimization_r10_v1'
TASKS = ('eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc')
PT_ARMS = {
    'A1': ('A1_FUSEDCE_2D', 'A1_RAWCE_2D'),
    'A2': ('A2_2D_ANCHOR',), 'P1': ('P1_BASE20K', 'P1_SCHED5K'),
    'F2': (), 'F3': (), 'G0': ('G0_BASE', 'G0_TORSION'),
    'G1': ('G1_NONBONDED',), 'Q': ('Q_WEIGHTED',),
}
FT_ARMS = dict(PT_ARMS, F2=('F2_BASE', 'F2_FILM'), F3=('F3_ATTENTION',))
REFERENCE = {'A1': ('A1', 'A1_FUSEDCE_2D'), 'A2': ('A1', 'A1_FUSEDCE_2D'),
             'P1': ('P1', 'P1_BASE20K'), 'F2': ('F2', 'F2_BASE'),
             'F3': ('F2', 'F2_BASE'), 'G0': ('G0', 'G0_BASE'),
             'G1': ('P1', 'P1_BASE20K'), 'Q': ('P1', 'P1_BASE20K')}
GEOMETRY = dict(geometry_revision='glt_periodic_seam_r1', geometry_mode='finite_incident',
                angle_value_enabled=True, readout='incident')


def pt_config(cycle, arm):
    if cycle not in PT_ARMS or arm not in PT_ARMS[cycle]:
        raise ValueError('undefined r10 PT arm')
    cfg = json.loads((COMMON.parent / 'F_CE.json').read_text())
    cfg.update(plan='GLT-OPTIMIZATION-20261011-01', revision='r10', cycle=cycle, arm=arm,
               r10_revision=REVISION, runtime_revision=RUNTIME, expected_world_size=2,
               accumulation=6, cycle_root=str(OUT / cycle), common_initialization=str(COMMON),
               initial_state=str(OUT / cycle / 'initialization' / (arm + '_step0.pt')),
               fusion_gate='none', fusion_attention_mode='global', incident_bias=0.0,
               endpoint_revision=None, quality_root=None, align_revision=None, align_init_seed=None,
               align_temperature=.10, align_weight=.10, align_warmup_steps=1000,
               align_direction='symmetric', nonbonded_revision=None, qc_weight_root=None)
    if cycle in ('A1', 'A2'):
        cfg.update(align_enabled=True, align_revision='glt_align_r10_' + cycle.lower(),
                   align_init_seed=314161)
        if arm == 'A1_RAWCE_2D':
            cfg['atom_ce_input'] = 'raw'
        if cycle == 'A2':
            cfg['align_direction'] = '2d_anchor'
    if arm == 'P1_SCHED5K':
        cfg['schedule_total_steps'] = 5000
    if cycle == 'G0':
        cfg['seed'] = 43
        if arm == 'G0_TORSION':
            from .glt_dual import PHYSICAL_TORSION
            cfg.update(PHYSICAL_TORSION)
    if cycle == 'G1':
        cfg['nonbonded_revision'] = 'glt_nonbonded_r10_g1'
    if cycle == 'Q':
        cfg['qc_weight_root'] = str(OUT / 'Q/qc')
    return cfg


def validate_pt_config(cfg):
    expected = pt_config(cfg.get('cycle'), cfg.get('arm'))
    if set(cfg) != set(expected) or any(type(cfg[k]) is not type(v) or cfg[k] != v
                                      for k, v in expected.items()):
        raise ValueError('r10 PT recipe/identity differs from its fixed arm')
    return cfg


def bind_package(package, cfg):
    validate_pt_config(cfg)
    result = dict(package, r10_revision=REVISION, runtime_revision=RUNTIME,
                  training_config=copy.deepcopy(cfg))
    for name in ('plan', 'revision', 'cycle', 'arm', 'common_initialization',
                 'expected_world_size', 'microbatch', 'accumulation', 'global_batch',
                 'seed', 'align_revision', 'align_init_seed', 'align_direction', 'nonbonded_revision', 'qc_weight_root'):
        result[name] = cfg[name]
    for name in ('fusion_attention_mode','incident_bias','endpoint_revision'):
        result[name]=cfg[name]
    if cfg['align_enabled']:
        result.update(pretrain_revision=cfg['align_revision'], align_revision=cfg['align_revision'])
    return result


def validate_package(package):
    cfg = package.get('training_config')
    if not isinstance(cfg, dict) or package.get('r10_revision') != REVISION:
        raise ValueError('missing/unknown r10 parent contract')
    validate_pt_config(cfg)
    for name in ('runtime_revision', 'plan', 'revision', 'cycle', 'arm',
                 'common_initialization', 'expected_world_size', 'microbatch',
                 'accumulation', 'global_batch', 'seed', 'atom_ce_input', 'align_enabled',
                 'align_revision', 'align_init_seed', 'align_direction', 'nonbonded_revision',
                 'qc_weight_root', 'fusion_mode', 'fusion_revision', 'fusion_gate',
                 'use_md200', 'local_enabled', 'third_task', 'torsion_modules',
                 'torsion_mode', 'geometry_revision', 'geometry_mode', 'angle_value_enabled',
                 'readout', 'opt_mode', 'quality_root', 'align_weight', 'align_temperature',
                 'align_warmup_steps'):
        if name not in package or type(package[name]) is not type(cfg[name]) or package[name] != cfg[name]:
            raise ValueError('r10 parent field mismatch: ' + name)
    for name in ('fusion_attention_mode','incident_bias','endpoint_revision'):
        if name not in package or type(package[name]) is not type(cfg[name]) or package[name]!=cfg[name]:
            raise ValueError('r10 connection mismatch: '+name)
    if cfg['align_enabled'] and package.get('pretrain_revision') != cfg['align_revision']:
        raise ValueError('r10 ALIGN revision mismatch')
    return cfg


def configure_runtime(encoder):
    # E1's bounded global case measured 0.589x; retain loop for this route.
    # Omit unused mean attention; do not promise end-to-end speedup.
    encoder.token_fusion.set_implementation('loop')
    encoder.collect_fusion_attention = False


class NonbondedInput(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(314173)
            self.net = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 512))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, data, count):
        if getattr(data, 'nonbonded_revision', None) != 'glt_nonbonded_r10_g1':
            raise ValueError('nonbonded model/input revision mismatch')
        index, distance, valid = data.nonbonded_bond_index, data.nonbonded_distance, data.nonbonded_mask
        if index.dtype != torch.long or valid.dtype != torch.bool or index.shape != distance.shape or valid.shape != index.shape:
            raise ValueError('nonbonded input schema mismatch')
        if index.numel() and (index.min() < 0 or index.max() >= count):
            raise ValueError('nonbonded central bond out of range')
        if not torch.isfinite(distance).all() or (distance[valid] <= 0).any():
            raise ValueError('invalid nonbonded distance')
        centers = torch.linspace(0, 6, 16, device=distance.device)
        values = self.net(torch.exp(-.5 * ((distance.float()[:, None] - centers) / .4).square()))
        result = values.new_zeros(count, 512).index_add(0, index[valid], values[valid])
        counts = values.new_zeros(count).index_add(0, index[valid], torch.ones_like(distance[valid]))
        return result / counts.clamp_min(1)[:, None]


class GraphFiLM(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(314179)
            self.net = nn.Sequential(nn.Linear(512, 64), nn.GELU(), nn.Linear(64, 1024))
            nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, valid):
        h2, h3 = features.split(512, dim=-1)
        gamma, beta = self.net(h3).split(512, dim=-1)
        update = torch.where(valid[:, None], gamma * h2 + beta, torch.zeros_like(h2))
        return torch.cat((h2 + update, h3), dim=-1)


class GraphAttention(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(314181)
            self.q = nn.Linear(512, 128); self.k = nn.Linear(512, 128); self.v = nn.Linear(512, 128)
            self.output = nn.Linear(128, 512)
            nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, features, valid):
        tokens = features.reshape(-1, 2, 512)
        q, k, v = [layer(tokens).reshape(-1, 2, 4, 32).transpose(1, 2)
                   for layer in (self.q, self.k, self.v)]
        weights = torch.softmax(q.float() @ k.float().transpose(-2, -1) / (32 ** .5), dim=-1)
        update = self.output((weights.to(v.dtype) @ v).transpose(1, 2).reshape(-1, 2, 128))
        update = torch.where(valid[:, None, None], update, torch.zeros_like(update))
        return (tokens + update).reshape(-1, 1024)
