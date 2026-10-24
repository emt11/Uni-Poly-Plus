"""Isolated UA-HOP-PH r1 model and complete restoration; legacy models unchanged."""
import torch
from torch import nn
from torch_geometric.utils import softmax
from torch_scatter import scatter

from .glt_dual import BondPathO8, mean_pool, strict_module_state
from .ua2d_galformer_pretrain import MaskedGalformer3D, package_attention, SCHEMA as PT_SCHEMA
from src.dataset.ua_hop_ph import SEMANTICS, SCHEMA as INPUT_SCHEMA

SCHEMA = 'ua-hop-ph-full-ft-r1'
ARCHITECTURE = dict(schema=SCHEMA, input=SEMANTICS, inherited_attention_denominator=512,
    branches=4, independent_ln_qkv=True, qkv_width=128, heads=4,
    adapter_attention_denominator=32, shared_bias=[18, 32, 4], shared_output=[128, 512],
    output_zero_init=True, router=[155, 128, 4], router_extra_parameters=20484,
    router_mode='dense', pooling='original_mean_norm2', dropout_encoder=0., dropout_head=.1)


class HopBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.qkv = nn.Linear(512, 384)

    def forward(self, states, edge, bias):
        if edge.shape[1] == 0:
            return states.new_zeros((len(states), 128))
        source, target = edge
        q, k, v = self.qkv(self.norm(states)).reshape(-1, 3, 4, 32).unbind(1)
        logits = (q[source]*k[target]).sum(-1)*(32**-.5)+bias
        weight = softmax(logits, target, num_nodes=len(states))
        return scatter(weight[..., None]*v[source], target, dim=0,
                       dim_size=len(states), reduce='sum').reshape(-1, 128)


class HopAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([HopBranch() for _ in range(4)])
        self.bias = nn.Sequential(nn.Linear(18, 32), nn.GELU(), nn.Linear(32, 4))
        self.W_out = nn.Linear(128, 512)
        nn.init.zeros_(self.W_out.weight); nn.init.zeros_(self.W_out.bias)

    def forward(self, states, data, weights, atoms):
        bias = self.bias(data.hop_features)
        outputs = []
        for k, branch in enumerate(self.branches, 1):
            mask = data.hop_distance == k
            outputs.append(branch(states, data.hop_edge_index[:, mask], bias[mask]))
        values = (torch.stack(outputs, 1)*weights[data.bond_batch, :, None]).sum(1)
        bonds, canonical = data.incidence_index
        pooled = mean_pool(values[bonds], canonical, atoms)
        counts = torch.bincount(canonical, minlength=atoms)
        valid = data.geometry_valid[data.canonical_graph_index].bool() & (counts > 0)
        projected = self.W_out(pooled)
        return torch.where(valid[:, None], projected, torch.zeros_like(projected))


class HopModel(nn.Module):
    def __init__(self, arm):
        super().__init__()
        if arm not in ('HOP_EQ', 'HOP_PH'): raise ValueError('unknown arm')
        self.arm = arm
        self.o8 = BondPathO8(0., attention_scale_denominator=512)
        self.encoder3d = MaskedGalformer3D(0., attention_scale_denominator=512)
        self.norm2 = nn.LayerNorm(512)
        self.head = nn.Sequential(nn.Linear(512, 512), nn.GELU(), nn.Dropout(.1), nn.Linear(512, 1))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261020)
            self.adapter = HopAdapter()
        if arm == 'HOP_PH':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20261020)
                self.router = nn.Sequential(nn.Linear(155, 128), nn.GELU(), nn.Linear(128, 4))
                nn.init.normal_(self.router[0].weight, std=.02); nn.init.zeros_(self.router[0].bias)
                nn.init.zeros_(self.router[-1].weight); nn.init.zeros_(self.router[-1].bias)
        self.stage = 'joint'

    def train(self, mode=True):
        super().train(mode)
        if mode and self.stage == 'head':
            self.o8.eval(); self.encoder3d.eval(); self.adapter.eval(); self.norm2.eval()
            if self.arm == 'HOP_PH': self.router.eval()
        return self

    def features(self, data):
        if data.hop_input_schema != INPUT_SCHEMA: raise ValueError('wrong input schema')
        atoms, _ = self.o8(data)
        weights = (torch.softmax(self.router(data.hop_ph.flatten(1)), -1) if self.arm == 'HOP_PH'
                   else atoms.new_full((len(data.geometry_valid), 4), .25))
        if data.bond_distance.numel():
            states, _, _ = self.encoder3d(data)
        else:
            states = atoms.new_empty((0, 512))
        delta = self.adapter(states, data, weights, len(atoms))
        return self.norm2(mean_pool(atoms+delta, data.canonical_graph_index, len(data.geometry_valid)))

    def forward(self, data):
        return self.head(self.features(data)).view(-1)


def initialize(parent, head, arm):
    if (parent.get('schema') != PT_SCHEMA or parent.get('step') != 5000
            or parent.get('config', {}).get('strategy') != 'mask_align'
            or parent['config'].get('fp_enabled') is not True
            or parent.get('execution', {}).get('revision') != 'r4'
            or parent.get('execution', {}).get('plan') != 'UA2D-PRETRAIN-FP-20261019-01'
            or package_attention(parent)['scale_denominator'] != 512):
        raise ValueError('not the declared B_FP parent')
    model = HopModel(arm)
    for name in ('o8', 'encoder3d', 'norm2'):
        prefix = name+'.'
        strict_module_state(getattr(model, name), {k[len(prefix):]: v for k, v in parent['state_dict'].items() if k.startswith(prefix)})
    strict_module_state(model.head, head)
    return model


def restore(saved, expected_config):
    if (saved.get('schema') != SCHEMA or saved.get('architecture') != ARCHITECTURE
            or saved.get('config') != expected_config or saved.get('best_stage') not in ('head', 'joint')):
        raise ValueError('complete FT schema/config mismatch')
    model = HopModel(expected_config['arm'])
    strict_module_state(model, saved['state_dict'])
    return model


def optimizer(model, stage):
    if stage not in ('head', 'joint'): raise ValueError('wrong stage')
    model.stage = stage
    for name, p in model.named_parameters(): p.requires_grad_(stage == 'joint' or name.startswith('head.'))
    groups = []
    if stage == 'joint':
        groups.extend([dict(name='inherited', params=[p for m in (model.o8, model.encoder3d, model.norm2) for p in m.parameters()], lr=1e-4),
                       dict(name='new', params=[p for name, p in model.named_parameters() if name.startswith(('adapter.', 'router.'))], lr=1e-3)])
    groups.append(dict(name='head', params=list(model.head.parameters()), lr=3e-4 if stage == 'head' else 1e-3))
    ids = [id(p) for g in groups for p in g['params']]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError('optimizer groups overlap or omit trainable parameters')
    evidence = [dict(name=g['name'], lr=g['lr'], tensors=len(g['params']), parameters=sum(p.numel() for p in g['params'])) for g in groups]
    return torch.optim.AdamW(groups, weight_decay=.02), evidence
