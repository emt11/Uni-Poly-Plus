"""O8 bond-path + Galformer physical-Trimer encoder, without MD200.

References (MIT): robbenplus/MolGT src/models/modeling.py (bond path);
peizhenbai/Galformer model/{module_utils,model_3d}.py (Gaussian/Residual).
Project adaptations: 512/6/8, physical line-hop <=2, center-only readout,
explicit post-MLP padding masks, and no virtual CLS node.
"""

import math

import torch
from torch import nn
from torch_geometric.utils import softmax
from torch_scatter import scatter

from .mips_local_graph import MIPSLocalAtomEmbedding, MIPSSinglePathNodeBias
from .original_mips_knowledge_fusion import OriginalMIPSAttentiveFusion


def mean_pool(states, index, count):
    return scatter(states, index.long(), dim=0, dim_size=count, reduce='mean')


class SharedBondPathBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.type_emb = nn.Embedding(5, 8)
        self.conj_emb = nn.Embedding(2, 8)
        self.ring_emb = nn.Embedding(2, 8)
        self.stereo_emb = nn.Embedding(7, 8)
        self.position = nn.Parameter(torch.empty(2, 8, 8))
        for parameter in self.parameters():
            nn.init.normal_(parameter, std=0.02)

    def forward(self, features, mask):
        if features.shape != (*mask.shape, 14) or mask.ndim != 2 or mask.size(1) != 2:
            raise ValueError('bond paths must be [R,2,14] with [R,2] mask')
        if not torch.isfinite(features).all():
            raise ValueError('nonfinite bond features')
        selected = features[mask]
        if selected.numel() and (not ((selected == 0) | (selected == 1)).all()
                                or not (selected[:, :5].sum(-1) == 1).all()
                                or not (selected[:, 7:].sum(-1) == 1).all()):
            raise ValueError('invalid fourteen-dimensional bond chemistry')
        z = (self.type_emb(features[..., :5].argmax(-1))
             + self.conj_emb(features[..., 5].long())
             + self.ring_emb(features[..., 6].long())
             + self.stereo_emb(features[..., 7:].argmax(-1)))
        transformed = torch.einsum('rki,kij->rkj', z, self.position)
        return (transformed * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


class SourceAttention512(nn.Module):
    """Source-Q, target-K; source values and incoming softmax. Isolated from legacy."""
    def __init__(self, dropout=0.1):
        super().__init__()
        self.qkv = nn.Linear(512, 1536)
        self.dropout = nn.Dropout(dropout)
        self.scale = 512 ** -0.5

    def forward(self, states, source, target, bias):
        q, k, v = self.qkv(states).reshape(-1, 3, 8, 64).unbind(1)
        score = (q[source].float() * k[target].float()).sum(-1) * self.scale
        weight = softmax(score + bias.float(), target, num_nodes=states.size(0)).to(v.dtype)
        messages = self.dropout(weight).unsqueeze(-1) * v[source]
        return torch.zeros_like(v).index_add(0, target, messages).reshape(-1, 512)


class TransformerBlock(nn.Module):
    def __init__(self, dropout=0.1, *, o8=False):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(512), nn.LayerNorm(512)
        self.attention = SourceAttention512(dropout)
        self.output = nn.Linear(512, 512)
        self.dropout = nn.Dropout(dropout)
        # O8 retains its original intermediate dropout; Galformer has only
        # the output feature dropout in the two-dense-layer FFN.
        self.ffn = nn.Sequential(nn.Linear(512, 2048), nn.GELU(),
                                 nn.Dropout(dropout) if o8 else nn.Identity(),
                                 nn.Linear(2048, 512))

    def forward(self, states, source, target, bias):
        states = states + self.dropout(self.output(self.attention(self.norm1(states), source, target, bias)))
        return states + self.dropout(self.ffn(self.norm2(states)))


class BondPathO8(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.atom_embedding = MIPSLocalAtomEmbedding(512)
        self.spd_embedding = nn.Embedding(3, 8)
        nn.init.zeros_(self.spd_embedding.weight)
        self.path_bias = MIPSSinglePathNodeBias(512, 8, 2)
        self.bond_bias = SharedBondPathBias()
        self.layers = nn.ModuleList([TransformerBlock(dropout, o8=True) for _ in range(6)])

    def forward(self, data, atom_mask=None):
        initial = self.atom_embedding(data, atom_mask=atom_mask)
        bias = (self.spd_embedding(data.lga_spd.long()) + self.path_bias(initial, data)
                + self.bond_bias(data.bond_path_features, data.bond_path_mask.bool()))
        bias = bias.masked_fill(data.lga_relation_mask.bool().unsqueeze(-1), 0)
        states = initial
        source, target = data.lga_edge_index.long()
        for layer in self.layers:
            states = layer(states, source, target, bias)
        states = states * data.graph_available[data.canonical_graph_index].unsqueeze(-1)
        return states, bias


def element_index(z):
    return torch.where((z >= 1) & (z <= 100), z.long() - 1, torch.full_like(z.long(), 100))


def triplet_type(z_a, z_b, bond_type):
    """Exact bonded portion of Galformer Vocab's low/type/high loop order."""
    low = torch.minimum(element_index(z_a), element_index(z_b))
    high = torch.maximum(element_index(z_a), element_index(z_b))
    return 5 * (101 * low - low * (low - 1) // 2) + bond_type.long() * (101 - low) + high - low


class TypeGaussian(nn.Module):
    def __init__(self, size, types):
        super().__init__()
        self.means = nn.Parameter(torch.empty(size))
        self.stds = nn.Parameter(torch.empty(size))
        self.mul = nn.Embedding(types, 1)
        self.bias = nn.Embedding(types, 1)
        nn.init.uniform_(self.means, 0, 3)
        nn.init.uniform_(self.stds, 0, 3)
        nn.init.ones_(self.mul.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, value, pair_type):
        mul = self.mul(pair_type).sum(-2)
        bias = self.bias(pair_type).sum(-2)
        x = mul * value.float().unsqueeze(-1) + bias
        std = self.stds.float().abs() + 1e-2
        return torch.exp(-0.5 * ((x - self.means.float()) / std).square()) / (math.sqrt(2 * 3.14159) * std)


class PathAngleBias(nn.Module):
    def __init__(self):
        super().__init__()
        # 25755 bonded + 101 unbonded + virtual/mask/padding categories.
        self.gaussian = TypeGaussian(128, 25859)
        self.position = nn.ModuleList([
            nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128)) for _ in range(2)])
        self.heads = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 8))

    def forward(self, types, path, angles, mask):
        if path.shape != (angles.size(0), 3) or angles.shape != mask.shape or angles.size(1) != 2:
            raise ValueError('line paths require 3 tokens and 2 angle positions')
        if not torch.isfinite(angles[mask]).all() or not mask.any(-1).all():
            raise ValueError('invalid or empty required angle path')
        extended = torch.cat([types, types.new_tensor([25858])])
        indices = torch.where(path >= 0, path, path.new_full(path.shape, types.numel()))
        path_types = extended[indices]
        features = []
        for k in range(2):
            encoded = self.gaussian(angles[:, k], path_types[:, k:k + 2])
            # Mask after the MLP: its learned bias must not contaminate padding.
            features.append(self.position[k](encoded) * mask[:, k:k + 1])
        pooled = sum(features) / mask.sum(-1, keepdim=True).clamp_min(1)
        return self.heads(pooled)


class GalformerTrimer3D(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.endpoint = nn.Linear(101, 512)
        self.distance_basis = TypeGaussian(256, 104)
        self.distance_projection = nn.Linear(256, 512)
        self.triplet = nn.Sequential(nn.Linear(1024, 512), nn.GELU(), nn.Linear(512, 512))
        self.angle_bias = PathAngleBias()
        self.layers = nn.ModuleList([TransformerBlock(dropout) for _ in range(6)])
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data):
        za, zb = element_index(data.bond_z_a), element_index(data.bond_z_b)
        distance = data.bond_distance.float()
        if not torch.isfinite(distance).all() or (distance <= 0).any():
            raise ValueError('invalid physical bond distance')
        a = torch.nn.functional.one_hot(za, 101).float()
        b = torch.nn.functional.one_hot(zb, 101).float()
        endpoints = self.endpoint(a) + self.endpoint(b)
        radial = self.distance_projection(self.distance_basis(distance, torch.stack([za, zb], -1)))
        states = self.triplet(torch.cat([endpoints, radial], -1))
        types = triplet_type(data.bond_z_a, data.bond_z_b, data.bond_type)
        path_bias = self.angle_bias(types, data.line_path, data.line_angle, data.line_mask)
        # Equal shortest paths contribute symmetrically to ONE attention edge.
        bias = mean_pool(path_bias, data.line_path_group, data.line_source.numel())
        for layer in self.layers:
            states = layer(states, data.line_source, data.line_target, bias)
        center = data.bond_center.bool()
        batch = data.bond_batch[center]
        count = data.geometry_valid.numel()
        numbers = torch.bincount(batch, minlength=count)
        valid = data.geometry_valid.bool() & (numbers > 0)
        graph = mean_pool(states[center], batch, count)
        graph = torch.where(valid.unsqueeze(-1), graph, torch.zeros_like(graph))
        return dict(bond_states=states, center_bond_states=states[center],
                    graph_3d=graph, geometry_valid=valid, angle_bias=bias)


class DualGLTModel(nn.Module):
    architecture_name = 'O8-BondPath-GalformerTrimer-Hop2'

    def __init__(self, fusion_mode='concat', dropout=0.1):
        super().__init__()
        if fusion_mode not in {'concat', 'kfuse'}:
            raise ValueError('fusion_mode must be concat or kfuse')
        self.fusion_mode = fusion_mode
        self.o8 = BondPathO8(dropout)
        self.glt = GalformerTrimer3D(dropout)
        if fusion_mode == 'concat':
            self.norm2, self.norm3 = nn.LayerNorm(512), nn.LayerNorm(512)
        else:
            self.kfuse = OriginalMIPSAttentiveFusion(
                knodes=('glt3d',), knowledge_dims={'glt3d': 512})
        self.predictor = nn.Sequential(nn.Linear(1024 if fusion_mode == 'concat' else 512, 512),
                                       nn.GELU(), nn.Dropout(dropout), nn.Linear(512, 1))

    def encode(self, data, *, atom_mask=None):
        atoms, bias = self.o8(data, atom_mask)
        result = self.glt(data)
        result.update(atom_states=atoms, bond_path_attention_bias=bias,
                      canonical_graph_index=data.canonical_graph_index,
                      graph_2d=mean_pool(atoms, data.canonical_graph_index, data.graph_available.numel()))
        return result

    def fuse(self, encoded):
        valid = encoded['geometry_valid']
        if self.fusion_mode == 'concat':
            geometry = self.norm3(encoded['graph_3d'])
            geometry = torch.where(valid.unsqueeze(-1), geometry, torch.zeros_like(geometry))
            features = torch.cat([self.norm2(encoded['graph_2d']), geometry], -1)
        else:
            atoms = encoded['atom_states']
            index = encoded['canonical_graph_index']
            fused = self.kfuse(atoms, {'glt3d': encoded['graph_3d']}, index)
            # Mask the entire projected update, including v_proj.bias.
            fused = torch.where(valid[index].unsqueeze(-1), fused, atoms)
            features = mean_pool(fused, index, encoded['graph_2d'].size(0))
        return features

    def forward(self, data, *, atom_mask=None):
        return self.predictor(self.fuse(self.encode(data, atom_mask=atom_mask)))


def build_dual_glt_model(fusion_mode='concat', *, dropout=0.1):
    return DualGLTModel(fusion_mode=fusion_mode, dropout=dropout)
