"""MIPS-aligned local periodic 3-D Graph Line Transformer v2.

The v2 encoder deliberately reuses the frozen v1 sidecar.  It derives
Gaussian observation moments, identity self-relations and line-to-atom
incidence at runtime, so no cached geometry is rewritten.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch_geometric.utils import softmax
from torch_scatter import scatter

from src.dataset.periodic_line_glt import MAX_ATOMIC_NUMBER, NUM_BOND_TYPES


NUM_LINE_LABELS = ((MAX_ATOMIC_NUMBER + 1) * (MAX_ATOMIC_NUMBER + 2) // 2) * NUM_BOND_TYPES
MASK_ATOMIC_NUMBER = MAX_ATOMIC_NUMBER + 1
MASK_SHIFT_CLASS = 2


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))


class LearnedGaussianMoments(nn.Module):
    """Encode observations with learned Gaussians, returning mean/variance."""

    def __init__(
        self,
        size: int,
        lower: float,
        upper: float,
        *,
        pair_conditioned: bool = False,
    ):
        super().__init__()
        size = int(size)
        if size < 2:
            raise ValueError("Gaussian basis size must be at least two")
        centers = torch.linspace(float(lower), float(upper), size)
        spacing = float(centers[1] - centers[0])
        self.centers = nn.Parameter(centers)
        self.raw_width = nn.Parameter(torch.full((size,), _inverse_softplus(spacing)))
        self.pair_conditioned = bool(pair_conditioned)
        self.pair_vocab = MAX_ATOMIC_NUMBER + 2
        if self.pair_conditioned:
            self.pair_affine = nn.Embedding(self.pair_vocab * self.pair_vocab, 2)
            with torch.no_grad():
                self.pair_affine.weight[:, 0].fill_(1.0)
                self.pair_affine.weight[:, 1].zero_()

    def _pair_index(self, z_a, z_b):
        low = torch.minimum(z_a.long(), z_b.long()).clamp(0, self.pair_vocab - 1)
        high = torch.maximum(z_a.long(), z_b.long()).clamp(0, self.pair_vocab - 1)
        return low * self.pair_vocab + high

    def forward(self, observations, counts, *, z_a=None, z_b=None):
        observations = observations.to(dtype=self.centers.dtype)
        counts = counts.long().reshape(-1)
        transformed = observations
        if self.pair_conditioned:
            if z_a is None or z_b is None:
                raise ValueError("pair-conditioned Gaussian basis requires endpoint types")
            affine = self.pair_affine(self._pair_index(z_a, z_b))
            scale = affine[:, :1]
            bias = affine[:, 1:]
            transformed = observations * scale + bias
        width = torch.nn.functional.softplus(self.raw_width).clamp_min(1e-6)
        encoded = torch.exp(
            -0.5
            * ((transformed.unsqueeze(-1) - self.centers) / width) ** 2
        )
        slots = torch.arange(observations.size(1), device=observations.device).unsqueeze(0)
        mask = slots < counts.unsqueeze(1)
        weights = mask.unsqueeze(-1).to(encoded.dtype)
        denominator = counts.clamp_min(1).to(encoded.dtype).unsqueeze(-1)
        mean = (encoded * weights).sum(dim=1) / denominator
        centered = (encoded - mean.unsqueeze(1)) * weights
        variance = centered.square().sum(dim=1) / denominator
        valid = (counts > 0).to(mean.dtype).unsqueeze(-1)
        return mean * valid, variance * valid


class PeriodicLineAttentionV2(nn.Module):
    def __init__(self, hidden_size=512, heads=8, dropout=0.1, variant="mips"):
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        if variant not in {"mips", "paper"}:
            raise ValueError("attention variant must be mips or paper")
        self.hidden_size = int(hidden_size)
        self.heads = int(heads)
        self.head_dim = self.hidden_size // self.heads
        self.variant = str(variant)
        self.scale = (
            self.head_dim ** -0.5 if self.variant == "mips"
            else self.hidden_size ** -0.5
        )
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states, source, target, relation_bias):
        node_count = int(states.size(0))
        q, k, v = self.qkv(states).chunk(3, dim=-1)
        q = q.view(node_count, self.heads, self.head_dim)
        k = k.view(node_count, self.heads, self.head_dim)
        v = v.view(node_count, self.heads, self.head_dim)
        if self.variant == "mips":
            left, right = q[target], k[source]
        else:
            left, right = q[source], k[target]
        scores = (left * right).sum(dim=-1) * self.scale + relation_bias
        weights = softmax(scores, target, num_nodes=node_count).to(v.dtype)
        messages = self.dropout(weights).unsqueeze(-1) * v[source]
        aggregated = torch.zeros_like(v)
        aggregated.index_add_(0, target, messages)
        return self.out(aggregated.reshape(node_count, self.hidden_size))


class PeriodicLineLayerV2(nn.Module):
    def __init__(self, hidden_size=512, heads=8, dropout=0.1, variant="mips"):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attention = PeriodicLineAttentionV2(hidden_size, heads, dropout, variant)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, states, source, target, relation_bias):
        states = states + self.attention(
            self.norm1(states), source, target, relation_bias
        )
        return states + self.ffn(self.norm2(states))


class LocalPeriodicGraphLineTransformerV2(nn.Module):
    """Local periodic line encoder with canonical atom incidence readout."""

    def __init__(
        self,
        hidden_size=512,
        layers=6,
        heads=8,
        dropout=0.1,
        attention_variant="mips",
        distance_basis=256,
        angle_basis=128,
        distance_upper=3.75,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.layers_count = int(layers)
        self.attention_variant = str(attention_variant)
        self.atom_embedding = nn.Embedding(MAX_ATOMIC_NUMBER + 2, hidden_size)
        self.distance_basis = LearnedGaussianMoments(
            distance_basis, 0.0, distance_upper, pair_conditioned=True
        )
        self.angle_basis = LearnedGaussianMoments(angle_basis, 0.0, math.pi)
        self.distance_mean_projection = nn.Linear(distance_basis, hidden_size)
        self.distance_variance_projection = nn.Linear(distance_basis, hidden_size)
        self.angle_mean_projection = nn.Linear(angle_basis, heads, bias=False)
        self.angle_variance_projection = nn.Linear(angle_basis, heads, bias=False)
        self.token_count_embedding = nn.Embedding(4, hidden_size)
        self.token_shift_embedding = nn.Embedding(3, hidden_size)
        self.relation_count_embedding = nn.Embedding(4, heads)
        self.relation_multiplicity_embedding = nn.Embedding(4, heads)
        self.identity_self_bias = nn.Parameter(torch.zeros(heads))
        self.input_norm = nn.LayerNorm(hidden_size)
        self.layers = nn.ModuleList([
            PeriodicLineLayerV2(
                hidden_size, heads, dropout, variant=self.attention_variant
            )
            for _ in range(self.layers_count)
        ])
        self.final_norm = nn.LayerNorm(hidden_size)
        self.incidence_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.incidence_shift_embedding = nn.Embedding(2, hidden_size)
        self.atom_output_norm = nn.LayerNorm(hidden_size)

    def line_inputs(
        self,
        data,
        *,
        endpoint_z_a=None,
        endpoint_z_b=None,
        observation_distances=None,
        observation_counts=None,
        token_shifts=None,
    ):
        z_a = data.glt_token_endpoint_z_a if endpoint_z_a is None else endpoint_z_a
        z_b = data.glt_token_endpoint_z_b if endpoint_z_b is None else endpoint_z_b
        distances = (
            data.glt_token_observation_distances
            if observation_distances is None else observation_distances
        )
        counts = (
            data.glt_token_observation_count
            if observation_counts is None else observation_counts
        )
        shifts = data.glt_token_shift if token_shifts is None else token_shifts
        dist_mean, dist_variance = self.distance_basis(
            distances, counts, z_a=z_a, z_b=z_b
        )
        endpoint = 0.5 * (self.atom_embedding(z_a) + self.atom_embedding(z_b))
        geometry = (
            self.distance_mean_projection(dist_mean)
            + self.distance_variance_projection(dist_variance)
        )
        count_feature = self.token_count_embedding(counts.long().clamp(0, 3))
        shift_class = shifts.long().abs().clamp(0, MASK_SHIFT_CLASS)
        shift_feature = self.token_shift_embedding(shift_class)
        valid = data.glt_token_valid.to(geometry.dtype).unsqueeze(-1)
        return self.input_norm(endpoint + valid * (geometry + count_feature + shift_feature))

    def _relations(self, data, dtype):
        real = ~data.glt_relation_is_fallback.bool()
        source = data.glt_relation_source.long()[real]
        target = data.glt_relation_target.long()[real]
        angles = data.glt_relation_observation_angles[real]
        counts = data.glt_relation_observation_count.long()[real]
        multiplicity = data.glt_relation_multiplicity.long()[real]
        valid = data.glt_relation_valid.bool()[real]
        angle_mean, angle_variance = self.angle_basis(angles, counts)
        bias = (
            self.angle_mean_projection(angle_mean)
            + self.angle_variance_projection(angle_variance)
            + self.relation_count_embedding(counts.clamp(0, 3))
            + self.relation_multiplicity_embedding(multiplicity.clamp(0, 3))
        )
        bias = bias * valid.to(bias.dtype).unsqueeze(-1)
        token_count = int(data.glt_token_atom_a.numel())
        identity = torch.arange(token_count, device=source.device)
        source = torch.cat([source, identity])
        target = torch.cat([target, identity])
        self_bias = self.identity_self_bias.to(dtype).unsqueeze(0).expand(token_count, -1)
        bias = torch.cat([bias.to(dtype), self_bias], dim=0)
        return source, target, bias

    def _atom_incidence(self, data, line_states):
        token_count = int(line_states.size(0))
        token_index = torch.arange(token_count, device=line_states.device)
        incidence_token = torch.cat([token_index, token_index])
        incidence_atom = torch.cat([
            data.glt_token_atom_a.long(), data.glt_token_atom_b.long()
        ])
        incidence_shift = torch.cat([
            data.glt_token_shift.long().abs(), data.glt_token_shift.long().abs()
        ]).clamp(0, 1)
        incidence_valid = torch.cat([
            data.glt_token_valid.bool(), data.glt_token_valid.bool()
        ])
        messages = (
            self.incidence_projection(line_states[incidence_token])
            + self.incidence_shift_embedding(incidence_shift)
        )
        messages = messages * incidence_valid.to(messages.dtype).unsqueeze(-1)
        atom_count = int(data.canonical_graph_index.numel())
        atom_sum = scatter(messages, incidence_atom, dim=0, dim_size=atom_count, reduce="sum")
        denominator = scatter(
            incidence_valid.to(messages.dtype).unsqueeze(-1),
            incidence_atom,
            dim=0,
            dim_size=atom_count,
            reduce="sum",
        )
        atom_valid = denominator.squeeze(-1) > 0
        atom_states = atom_sum / denominator.clamp_min(1.0)
        graph_valid = data.glt_geometry_valid.bool()[data.canonical_graph_index.long()]
        atom_valid = atom_valid & graph_valid
        atom_states = self.atom_output_norm(atom_states)
        atom_states = atom_states * atom_valid.to(atom_states.dtype).unsqueeze(-1)
        return atom_states, atom_valid

    def _graph_pool(self, data, atom_states, atom_valid):
        graph_index = data.canonical_graph_index.long()
        graph_count = int(data.glt_geometry_valid.numel())
        graph_sum = scatter(
            atom_states, graph_index, dim=0, dim_size=graph_count, reduce="sum"
        )
        denominator = scatter(
            atom_valid.to(atom_states.dtype).unsqueeze(-1),
            graph_index,
            dim=0,
            dim_size=graph_count,
            reduce="sum",
        )
        graph = graph_sum / denominator.clamp_min(1.0)
        return graph * data.glt_geometry_valid.to(graph.dtype).unsqueeze(-1)

    def forward(
        self,
        data,
        *,
        endpoint_z_a=None,
        endpoint_z_b=None,
        observation_distances=None,
        observation_counts=None,
        token_shifts=None,
    ):
        states = self.line_inputs(
            data,
            endpoint_z_a=endpoint_z_a,
            endpoint_z_b=endpoint_z_b,
            observation_distances=observation_distances,
            observation_counts=observation_counts,
            token_shifts=token_shifts,
        )
        source, target, relation_bias = self._relations(data, states.dtype)
        for layer in self.layers:
            states = layer(states, source, target, relation_bias)
        states = self.final_norm(states)
        atom_states, atom_valid = self._atom_incidence(data, states)
        graph = self._graph_pool(data, atom_states, atom_valid)
        return {
            "graph_geometry": graph,
            "line_states": states,
            "atom_geometry_states": atom_states,
            "atom_geometry_valid": atom_valid,
            "relation_source": source,
            "relation_target": target,
        }


class GLTMaskedLineHeadV2(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.projection = nn.Linear(hidden_size, NUM_LINE_LABELS)

    def forward(self, line_states):
        return self.projection(line_states)


__all__ = [
    "GLTMaskedLineHeadV2",
    "LearnedGaussianMoments",
    "LocalPeriodicGraphLineTransformerV2",
    "MASK_ATOMIC_NUMBER",
    "MASK_SHIFT_CLASS",
    "NUM_LINE_LABELS",
    "PeriodicLineAttentionV2",
]
