"""Galformer-style local periodic GLT with a last-layer graph query."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch_geometric.utils import softmax
from torch_scatter import scatter

from src.dataset.periodic_line_glt import MAX_ATOMIC_NUMBER, NUM_BOND_TYPES


NUM_LINE_LABELS = ((MAX_ATOMIC_NUMBER + 1) * (MAX_ATOMIC_NUMBER + 2) // 2) * NUM_BOND_TYPES


def _inverse_softplus(value):
    return math.log(math.expm1(max(float(value), 1e-6)))


class PairConditionedGaussianBasis(nn.Module):
    def __init__(self, size=256, lower=0.0, upper=3.75):
        super().__init__()
        centers = torch.linspace(float(lower), float(upper), int(size))
        spacing = float(centers[1] - centers[0])
        self.centers = nn.Parameter(centers)
        self.raw_width = nn.Parameter(torch.full_like(centers, _inverse_softplus(spacing)))
        self.vocab = MAX_ATOMIC_NUMBER + 1
        self.affine = nn.Embedding(self.vocab * self.vocab, 2)
        with torch.no_grad():
            self.affine.weight[:, 0].fill_(1.0)
            self.affine.weight[:, 1].zero_()

    def forward(self, values, valid_mask, z_a, z_b):
        low = torch.minimum(z_a.long(), z_b.long()).clamp(0, self.vocab - 1)
        high = torch.maximum(z_a.long(), z_b.long()).clamp(0, self.vocab - 1)
        affine = self.affine(low * self.vocab + high)
        transformed = values.to(self.centers.dtype) * affine[:, :1] + affine[:, 1:]
        width = torch.nn.functional.softplus(self.raw_width).clamp_min(1e-6)
        encoded = torch.exp(-0.5 * ((transformed.unsqueeze(-1) - self.centers) / width) ** 2)
        weights = valid_mask.to(encoded.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return (encoded * weights).sum(dim=1) / denominator


class GaussianBasis(nn.Module):
    def __init__(self, size=128, lower=0.0, upper=math.pi):
        super().__init__()
        centers = torch.linspace(float(lower), float(upper), int(size))
        spacing = float(centers[1] - centers[0])
        self.centers = nn.Parameter(centers)
        self.raw_width = nn.Parameter(torch.full_like(centers, _inverse_softplus(spacing)))

    def forward(self, values, valid_mask):
        width = torch.nn.functional.softplus(self.raw_width).clamp_min(1e-6)
        encoded = torch.exp(-0.5 * ((values.to(self.centers.dtype).unsqueeze(-1) - self.centers) / width) ** 2)
        weights = valid_mask.to(encoded.dtype).unsqueeze(-1)
        return (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class GraphGateLineAttention(nn.Module):
    def __init__(self, hidden_size=512, heads=8, dropout=0.1):
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(hidden_size) // self.heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, normalized_states, source, target, bias):
        count = int(normalized_states.size(0))
        q, k, v = self.qkv(normalized_states).chunk(3, dim=-1)
        q = q.view(count, self.heads, self.head_dim)
        k = k.view(count, self.heads, self.head_dim)
        v = v.view(count, self.heads, self.head_dim)
        logits = (q[target] * k[source]).sum(-1) * (self.head_dim ** -0.5) + bias
        weights = softmax(logits, target, num_nodes=count).to(v.dtype)
        messages = self.dropout(weights).unsqueeze(-1) * v[source]
        aggregated = torch.zeros_like(v)
        aggregated.index_add_(0, target, messages)
        return self.out(aggregated.reshape(count, -1))


class GraphGateLineLayer(nn.Module):
    def __init__(self, hidden_size=512, heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attention = GraphGateLineAttention(hidden_size, heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size), nn.Dropout(dropout),
        )

    def forward(self, states, source, target, bias):
        normalized = self.norm1(states)
        states = states + self.attention(normalized, source, target, bias)
        return states + self.ffn(self.norm2(states))


class LastLayerQueryPool(nn.Module):
    def __init__(self, hidden_size=512, heads=8):
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(hidden_size) // self.heads
        self.query = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.query, std=hidden_size ** -0.5)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.line_norm = nn.LayerNorm(hidden_size)
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(self, line_states, token_batch, token_valid, query_valid):
        graph_count = int(query_valid.numel())
        normalized_lines = self.line_norm(line_states)
        query = self.query_norm(self.query)
        q = self.q(query).view(1, self.heads, self.head_dim).expand(graph_count, -1, -1)
        k = self.k(normalized_lines).view(-1, self.heads, self.head_dim)
        v = self.v(normalized_lines).view(-1, self.heads, self.head_dim)
        eligible = token_valid.bool() & query_valid.bool()[token_batch.long()]
        message = torch.zeros((graph_count, self.heads, self.head_dim), device=line_states.device, dtype=line_states.dtype)
        if bool(eligible.any()):
            indices = torch.nonzero(eligible, as_tuple=False).flatten()
            groups = token_batch.long()[indices]
            logits = (q[groups] * k[indices]).sum(-1) * (self.head_dim ** -0.5)
            weights = softmax(logits, groups, num_nodes=graph_count).to(v.dtype)
            message = scatter(weights.unsqueeze(-1) * v[indices], groups, dim=0, dim_size=graph_count, reduce="sum")
        pooled = self.final_norm(self.query.unsqueeze(0) + self.out(message.reshape(graph_count, -1)))
        return pooled * query_valid.to(pooled.dtype).unsqueeze(-1)


class LocalPeriodicGraphLineTransformerGraphGate(nn.Module):
    def __init__(self, hidden_size=512, layers=6, heads=8, dropout=0.1):
        super().__init__()
        self.atom_embedding = nn.Embedding(MAX_ATOMIC_NUMBER + 1, hidden_size)
        self.atom_projection = nn.Linear(hidden_size, hidden_size)
        self.distance_basis = PairConditionedGaussianBasis(256, 0.0, 3.75)
        self.input_mlp = nn.Sequential(
            nn.Linear(hidden_size + 256, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.input_norm = nn.LayerNorm(hidden_size)
        self.mask_embedding = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.mask_embedding, std=hidden_size ** -0.5)
        self.angle_basis = GaussianBasis(128, 0.0, math.pi)
        self.angle_projection = nn.Linear(128, heads, bias=False)
        self.self_bias = nn.Parameter(torch.zeros(heads))
        self.layers = nn.ModuleList([GraphGateLineLayer(hidden_size, heads, dropout) for _ in range(int(layers))])
        self.final_norm = nn.LayerNorm(hidden_size)
        self.query_pool = LastLayerQueryPool(hidden_size, heads)

    @staticmethod
    def _validate_geometry_mode(geometry_mode):
        mode = str(geometry_mode)
        if mode not in {"full", "off"}:
            raise ValueError(f"invalid GraphGate geometry mode: {mode}")
        return mode

    def clean_line_inputs(self, data, geometry_mode="full"):
        geometry_mode = self._validate_geometry_mode(geometry_mode)
        z_a, z_b = data.glt_token_endpoint_z_a.long(), data.glt_token_endpoint_z_b.long()
        h_atom = self.atom_projection(self.atom_embedding(z_a) + self.atom_embedding(z_b))
        h_dist = self.distance_basis(
            data.glt_token_observation_distances,
            data.glt_token_observation_valid,
            z_a, z_b,
        )
        if geometry_mode == "off":
            # Keep the exact GBF execution graph while removing its value.
            h_dist = h_dist * 0.0
        return self.input_norm(self.input_mlp(torch.cat([h_atom, h_dist], dim=-1)))

    def relation_graph(self, data, dtype, geometry_mode="full"):
        geometry_mode = self._validate_geometry_mode(geometry_mode)
        real = ~data.glt_relation_is_fallback.bool()
        source = data.glt_relation_source.long()[real]
        target = data.glt_relation_target.long()[real]
        encoded = self.angle_basis(
            data.glt_relation_observation_angles[real],
            data.glt_relation_observation_valid[real],
        )
        bias = self.angle_projection(encoded).to(dtype)
        bias = bias * data.glt_relation_valid.bool()[real].to(dtype).unsqueeze(-1)
        if geometry_mode == "off":
            # Identity self bias remains active; only real angle bias is off.
            bias = bias * 0.0
        token_count = int(data.glt_token_atom_a.numel())
        identity = torch.arange(token_count, device=source.device)
        source = torch.cat([source, identity])
        target = torch.cat([target, identity])
        bias = torch.cat([bias, self.self_bias.to(dtype).expand(token_count, -1)], dim=0)
        return source, target, bias

    def encode_lines(self, data, line_inputs=None, geometry_mode="full"):
        geometry_mode = self._validate_geometry_mode(geometry_mode)
        states = (
            self.clean_line_inputs(data, geometry_mode=geometry_mode)
            if line_inputs is None else line_inputs
        )
        source, target, bias = self.relation_graph(
            data, states.dtype, geometry_mode=geometry_mode
        )
        for layer in self.layers:
            states = layer(states, source, target, bias)
        states = self.final_norm(states)
        query_valid = data.glt_query_valid.bool()
        return {
            "line_states": states,
            "query_valid": query_valid,
            "relation_source": source,
            "relation_target": target,
        }

    def forward(self, data, line_inputs=None, geometry_mode="full"):
        encoded = self.encode_lines(
            data, line_inputs=line_inputs, geometry_mode=geometry_mode
        )
        states = encoded["line_states"]
        query_valid = encoded["query_valid"]
        graph = self.query_pool(states, data.glt_token_batch, data.glt_token_valid, query_valid)
        return {**encoded, "graph_geometry": graph}


class GLTMaskedLineHeadGraphGate(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.projection = nn.Linear(hidden_size, NUM_LINE_LABELS)

    def forward(self, states):
        return self.projection(states)


__all__ = [
    "GLTMaskedLineHeadGraphGate", "LocalPeriodicGraphLineTransformerGraphGate",
    "LastLayerQueryPool", "NUM_LINE_LABELS", "PairConditionedGaussianBasis",
]
