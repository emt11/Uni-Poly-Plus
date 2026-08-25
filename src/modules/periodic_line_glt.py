"""Strictly local periodic 3-D Graph Line Transformer for MTS-GLT-v1."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import softmax

from src.dataset.periodic_line_glt import MAX_ATOMIC_NUMBER, NUM_BOND_TYPES


NUM_LINE_LABELS = ((MAX_ATOMIC_NUMBER + 1) * (MAX_ATOMIC_NUMBER + 2) // 2) * NUM_BOND_TYPES
MASK_ATOMIC_NUMBER = MAX_ATOMIC_NUMBER + 1


class FixedGaussianBasis(nn.Module):
    """A fixed Gaussian basis with an explicit encode-then-average path."""

    def __init__(self, size: int, lower: float, upper: float):
        super().__init__()
        centers = torch.linspace(float(lower), float(upper), int(size))
        spacing = float(centers[1] - centers[0]) if int(size) > 1 else 1.0
        self.register_buffer("centers", centers)
        self.gamma = 0.5 / (spacing * spacing)

    def forward(self, observations, counts):
        observations = observations.to(dtype=self.centers.dtype)
        encoded = torch.exp(
            -self.gamma * (observations.unsqueeze(-1) - self.centers) ** 2
        )
        slots = torch.arange(
            observations.size(1), device=observations.device
        ).unsqueeze(0)
        mask = slots < counts.long().unsqueeze(1)
        encoded = encoded * mask.unsqueeze(-1).to(encoded.dtype)
        denominator = counts.clamp_min(1).to(encoded.dtype).unsqueeze(-1)
        return encoded.sum(dim=1) / denominator


class PeriodicLineAttention(nn.Module):
    def __init__(self, hidden_size=512, heads=8, dropout=0.1):
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        self.hidden_size = int(hidden_size)
        self.heads = int(heads)
        self.head_dim = self.hidden_size // self.heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states, source, target, angle_bias):
        n = int(states.size(0))
        q = self.q(states).view(n, self.heads, self.head_dim)
        k = self.k(states).view(n, self.heads, self.head_dim)
        v = self.v(states).view(n, self.heads, self.head_dim)
        scores = (q[target] * k[source]).sum(dim=-1) * self.scale + angle_bias
        weights = softmax(scores, target, num_nodes=n).to(v.dtype)
        messages = self.dropout(weights).unsqueeze(-1) * v[source]
        aggregated = torch.zeros_like(v)
        aggregated.index_add_(0, target, messages)
        return self.out(aggregated.reshape(n, self.hidden_size))


class PeriodicLineLayer(nn.Module):
    def __init__(self, hidden_size=512, heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attention = PeriodicLineAttention(hidden_size, heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, states, source, target, angle_bias):
        states = states + self.attention(
            self.norm1(states), source, target, angle_bias
        )
        return states + self.ffn(self.norm2(states))


class LocalPeriodicGraphLineTransformer(nn.Module):
    """Six-layer line-token encoder without a bond-type input channel."""

    def __init__(
        self,
        hidden_size=512,
        layers=6,
        heads=8,
        dropout=0.1,
        distance_basis=128,
        angle_basis=128,
        distance_upper=3.75,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.atom_embedding = nn.Embedding(MAX_ATOMIC_NUMBER + 2, hidden_size)
        self.distance_basis = FixedGaussianBasis(distance_basis, 0.0, distance_upper)
        self.angle_basis = FixedGaussianBasis(angle_basis, 0.0, math.pi)
        self.distance_projection = nn.Linear(distance_basis, hidden_size)
        self.angle_projection = nn.Linear(angle_basis, heads, bias=False)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.layers = nn.ModuleList([
            PeriodicLineLayer(hidden_size, heads, dropout) for _ in range(int(layers))
        ])
        self.final_norm = nn.LayerNorm(hidden_size)

    def line_inputs(
        self,
        data,
        *,
        endpoint_z_a=None,
        endpoint_z_b=None,
        observation_distances=None,
        observation_counts=None,
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
        endpoint = 0.5 * (self.atom_embedding(z_a) + self.atom_embedding(z_b))
        distance = self.distance_projection(self.distance_basis(distances, counts))
        valid = data.glt_token_valid.to(distance.dtype).unsqueeze(-1)
        return self.input_norm(endpoint + valid * distance)

    def forward(
        self,
        data,
        *,
        endpoint_z_a=None,
        endpoint_z_b=None,
        observation_distances=None,
        observation_counts=None,
    ):
        states = self.line_inputs(
            data,
            endpoint_z_a=endpoint_z_a,
            endpoint_z_b=endpoint_z_b,
            observation_distances=observation_distances,
            observation_counts=observation_counts,
        )
        source = data.glt_relation_source.long()
        target = data.glt_relation_target.long()
        angle = self.angle_basis(
            data.glt_relation_observation_angles,
            data.glt_relation_observation_count,
        )
        angle_bias = self.angle_projection(angle)
        angle_bias = angle_bias * data.glt_relation_valid.to(angle_bias.dtype).unsqueeze(-1)
        for layer in self.layers:
            states = layer(states, source, target, angle_bias)
        states = self.final_norm(states)

        graph_count = int(data.glt_geometry_valid.numel())
        graph_sum = states.new_zeros((graph_count, self.hidden_size))
        graph_count_valid = states.new_zeros((graph_count, 1))
        token_valid = data.glt_token_valid.bool()
        if bool(token_valid.any()):
            index = data.glt_token_batch[token_valid].long()
            graph_sum.index_add_(0, index, states[token_valid])
            graph_count_valid.index_add_(
                0, index, torch.ones((int(index.numel()), 1), device=states.device, dtype=states.dtype)
            )
        pooled = graph_sum / graph_count_valid.clamp_min(1.0)
        pooled = pooled * data.glt_geometry_valid.to(pooled.dtype).unsqueeze(-1)
        return pooled, states


class GLTMaskedLineHead(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.projection = nn.Linear(hidden_size, NUM_LINE_LABELS)

    def forward(self, line_states):
        return self.projection(line_states)


__all__ = [
    "FixedGaussianBasis", "GLTMaskedLineHead", "LocalPeriodicGraphLineTransformer",
    "MASK_ATOMIC_NUMBER", "NUM_LINE_LABELS", "PeriodicLineAttention",
]
