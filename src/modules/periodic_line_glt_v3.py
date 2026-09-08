"""Galformer-style center-image Graph Line Transformer for MTS-GLT-v3."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch_geometric.utils import softmax
from torch_scatter import scatter

from src.dataset.periodic_line_glt_image import (
    BOND_TYPE_MASK,
    NUM_BOND_TYPE_TARGETS,
    STEREO_MASK,
)


ELEMENT_UNKNOWN = 0
ELEMENT_MASK = 101
ELEMENT_INPUT_VOCAB = 102
ATOM_PAIR_TARGETS = 101 * 102 // 2
CONJUGATED_MASK = 2


def atom_pair_label(z_a, z_b):
    low = torch.minimum(z_a.long(), z_b.long()).clamp(0, 100)
    high = torch.maximum(z_a.long(), z_b.long()).clamp(0, 100)
    return high * (high + 1) // 2 + low


class PairConditionedGaussianRBF(nn.Module):
    """A learnable 256-D Gaussian basis conditioned on an unordered pair."""

    def __init__(self, size=256, lower=0.0, upper=3.75):
        super().__init__()
        centers = torch.linspace(float(lower), float(upper), int(size))
        spacing = float(centers[1] - centers[0])
        self.centers = nn.Parameter(centers)
        self.raw_width = nn.Parameter(torch.full_like(centers, math.log(math.expm1(spacing))))
        self.pair_affine = nn.Embedding(ELEMENT_INPUT_VOCAB ** 2, 2)
        with torch.no_grad():
            self.pair_affine.weight[:, 0].fill_(1.0)
            self.pair_affine.weight[:, 1].zero_()

    def forward(self, distance, z_a, z_b):
        low = torch.minimum(z_a.long(), z_b.long()).clamp(0, ELEMENT_INPUT_VOCAB - 1)
        high = torch.maximum(z_a.long(), z_b.long()).clamp(0, ELEMENT_INPUT_VOCAB - 1)
        affine = self.pair_affine(low * ELEMENT_INPUT_VOCAB + high)
        value = distance.float().reshape(-1, 1) * affine[:, :1] + affine[:, 1:]
        width = torch.nn.functional.softplus(self.raw_width.float()).clamp_min(1e-6)
        return torch.exp(-0.5 * ((value - self.centers.float()) / width) ** 2)


class GaussianRBF(nn.Module):
    def __init__(self, size=128, lower=0.0, upper=math.pi):
        super().__init__()
        centers = torch.linspace(float(lower), float(upper), int(size))
        spacing = float(centers[1] - centers[0])
        self.centers = nn.Parameter(centers)
        self.raw_width = nn.Parameter(torch.full_like(centers, math.log(math.expm1(spacing))))

    def forward(self, value):
        width = torch.nn.functional.softplus(self.raw_width.float()).clamp_min(1e-6)
        return torch.exp(-0.5 * ((value.float().reshape(-1, 1) - self.centers.float()) / width) ** 2)


class ImageLineAttention(nn.Module):
    def __init__(self, hidden=512, heads=8, dropout=0.1):
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(hidden) // self.heads
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.output = nn.Linear(hidden, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states, source, target, bias):
        count = int(states.size(0))
        q, k, v = self.qkv(states).chunk(3, dim=-1)
        q = q.view(count, self.heads, self.head_dim)
        k = k.view(count, self.heads, self.head_dim)
        v = v.view(count, self.heads, self.head_dim)
        scores = ((q[target].float() * k[source].float()).sum(-1) * self.head_dim ** -0.5)
        weights = softmax(scores + bias.float(), target, num_nodes=count).to(v.dtype)
        message = self.dropout(weights).unsqueeze(-1) * v[source]
        aggregate = torch.zeros_like(v)
        aggregate.index_add_(0, target, message)
        return self.output(aggregate.reshape(count, self.heads * self.head_dim))


class ImageLineLayer(nn.Module):
    def __init__(self, hidden=512, heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attention = ImageLineAttention(hidden, heads, dropout)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden), nn.Dropout(dropout),
        )

    def forward(self, states, source, target, bias):
        states = states + self.attention(self.norm1(states), source, target, bias)
        return states + self.ffn(self.norm2(states))


class LocalPeriodicGraphLineTransformerV3(nn.Module):
    """Six-layer GLT using one center-anchored length/angle per record."""

    def __init__(self, hidden=512, layers=6, heads=8, dropout=0.1):
        super().__init__()
        if (hidden, layers, heads) != (512, 6, 8):
            raise ValueError("MTS-GLT-v3 fixes hidden/layers/heads to 512/6/8")
        self.element_embedding = nn.Embedding(ELEMENT_INPUT_VOCAB, 256)
        self.bond_type_embedding = nn.Embedding(BOND_TYPE_MASK + 1, 256)
        self.stereo_embedding = nn.Embedding(STEREO_MASK + 1, 256)
        self.conjugated_embedding = nn.Embedding(CONJUGATED_MASK + 1, 256)
        self.distance_basis = PairConditionedGaussianRBF(256, 0.0, 3.75)
        self.distance_projection = nn.Linear(256, 256)
        self.distance_mask = nn.Parameter(torch.zeros(256))
        self.input_norm = nn.LayerNorm(512)
        self.angle_basis = GaussianRBF(128, 0.0, math.pi)
        self.angle_projection = nn.Linear(128, heads, bias=False)
        self.identity_self_bias = nn.Parameter(torch.zeros(heads))
        self.layers = nn.ModuleList([
            ImageLineLayer(hidden, heads, dropout) for _ in range(layers)
        ])
        self.final_norm = nn.LayerNorm(hidden)
        self.incidence_projection = nn.Linear(hidden, hidden, bias=False)
        self.atom_output_norm = nn.LayerNorm(hidden)
        self.readout_norm = nn.LayerNorm(hidden * 2)
        self.readout = nn.Linear(hidden * 2, hidden)

    def _line_inputs(self, data, overrides=None):
        values = overrides or {}
        z_a = values.get("z_a", data.glt3_token_endpoint_z_a).long()
        z_b = values.get("z_b", data.glt3_token_endpoint_z_b).long()
        bond = values.get("bond_type", data.glt3_token_bond_type).long()
        stereo = values.get("stereo", data.glt3_token_stereo).long()
        conjugated = values.get("conjugated", data.glt3_token_conjugated).long()
        chemical = (
            self.element_embedding(z_a) + self.element_embedding(z_b)
            + self.bond_type_embedding(bond) + self.stereo_embedding(stereo)
            + self.conjugated_embedding(conjugated)
        )
        distance = values.get("distance", data.glt3_token_distance)
        radial = self.distance_projection(self.distance_basis(distance, z_a, z_b))
        distance_mask = values.get("distance_mask")
        if distance_mask is not None:
            radial = torch.where(
                distance_mask.bool().unsqueeze(-1),
                self.distance_mask.to(radial.dtype).unsqueeze(0), radial,
            )
        return self.input_norm(torch.cat([chemical, radial.to(chemical.dtype)], dim=-1))

    def _relations(self, data, dtype):
        valid = data.glt3_relation_valid.bool()
        source = data.glt3_relation_source.long()[valid]
        target = data.glt3_relation_target.long()[valid]
        bias = self.angle_projection(
            self.angle_basis(data.glt3_relation_angle[valid])
        ).to(dtype)
        identity = torch.arange(data.glt3_token_atom_a.numel(), device=source.device)
        return (
            torch.cat([source, identity]),
            torch.cat([target, identity]),
            torch.cat([bias, self.identity_self_bias.to(dtype).expand(identity.numel(), -1)]),
        )

    def _readout(self, data, lines):
        token = torch.arange(lines.size(0), device=lines.device).repeat(2)
        atom = torch.cat([data.glt3_token_atom_a.long(), data.glt3_token_atom_b.long()])
        valid = data.glt3_token_valid.bool().repeat(2)
        messages = self.incidence_projection(lines[token]) * valid.to(lines.dtype).unsqueeze(-1)
        atom_count = int(data.canonical_graph_index.numel())
        atom_sum = scatter(messages, atom, dim=0, dim_size=atom_count, reduce="sum")
        denominator = scatter(valid.to(lines.dtype).unsqueeze(-1), atom, dim=0, dim_size=atom_count, reduce="sum")
        atom_valid = denominator.squeeze(-1) > 0
        atoms = self.atom_output_norm(atom_sum / denominator.clamp_min(1.0))
        atoms = atoms * atom_valid.to(atoms.dtype).unsqueeze(-1)
        graph_count = int(data.glt3_geometry_valid.numel())
        atom_graph = data.canonical_graph_index.long()
        z_atom = scatter(atoms, atom_graph, dim=0, dim_size=graph_count, reduce="mean")
        z_line = scatter(lines, data.glt3_token_batch.long(), dim=0, dim_size=graph_count, reduce="mean")
        valid_graph = data.glt3_geometry_valid.to(lines.dtype).unsqueeze(-1)
        z = self.readout(self.readout_norm(torch.cat([z_atom, z_line], dim=-1))) * valid_graph
        return atoms, atom_valid, z, z_atom, z_line

    def forward(self, data, *, token_overrides=None):
        states = self._line_inputs(data, token_overrides)
        source, target, bias = self._relations(data, states.dtype)
        for layer in self.layers:
            states = layer(states, source, target, bias)
        states = self.final_norm(states)
        atom_states, atom_valid, graph, z_atom, z_line = self._readout(data, states)
        return {
            "line_states": states, "atom_geometry_states": atom_states,
            "atom_geometry_valid": atom_valid, "graph_geometry": graph,
            "z_atom": z_atom, "z_line": z_line,
            "relation_source": source, "relation_target": target,
        }


__all__ = [
    "ELEMENT_MASK", "ATOM_PAIR_TARGETS", "CONJUGATED_MASK",
    "atom_pair_label", "LocalPeriodicGraphLineTransformerV3",
]
