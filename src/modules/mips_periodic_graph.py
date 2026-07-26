"""Sparse MIPS-style periodic graph encoder used by ``scage_parallel``.

The attention neighborhood is selected only by periodic graph distance.
Strict-PBC Euclidean distance is an additive bias on those same sparse edges;
it never creates an attention edge by itself.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import softmax
from torch_scatter import scatter

from src.dataset.graph_data import SCAGE_ATOM_VOCABS, SCAGE_CATEGORICAL_FEATURES


class GaussianExpansion(nn.Module):
    def __init__(self, num_kernels=128, start=0.0, stop=9.0):
        super().__init__()
        centers = torch.linspace(float(start), float(stop), int(num_kernels))
        width = centers[1] - centers[0] if centers.numel() > 1 else centers.new_tensor(1.0)
        self.register_buffer("centers", centers)
        self.register_buffer("width", width)
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))

    def forward(self, values):
        values = self.scale * values.unsqueeze(-1) + self.shift
        width = self.width.clamp_min(1e-8)
        return torch.exp(-0.5 * ((values - self.centers) / width).square()) / (
            math.sqrt(2.0 * math.pi) * width
        )


class PeriodicAtomEmbedding(nn.Module):
    def __init__(self, hidden_dim=512, num_kernels=128):
        super().__init__()
        self.categorical = nn.ModuleDict({
            name: nn.Embedding(len(SCAGE_ATOM_VOCABS[name]), int(hidden_dim))
            for name in SCAGE_CATEGORICAL_FEATURES
        })
        self.continuous = nn.ModuleDict({
            name: nn.Sequential(
                GaussianExpansion(num_kernels=num_kernels),
                nn.Linear(int(num_kernels), int(hidden_dim)),
            )
            for name in ("mass", "van_der_waals_radius", "partial_charge")
        })
        self.backbone = nn.Embedding(3, int(hidden_dim))
        self.mask_token = nn.Parameter(torch.empty(int(hidden_dim)))
        self.norm = nn.LayerNorm(int(hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, data, atom_mask=None):
        embedding = None
        for name, layer in self.categorical.items():
            value = layer(getattr(data, name).long())
            embedding = value if embedding is None else embedding + value
        for name, layer in self.continuous.items():
            embedding = embedding + layer(getattr(data, name).float())
        if atom_mask is not None:
            embedding = torch.where(
                atom_mask.unsqueeze(-1), self.mask_token.unsqueeze(0), embedding
            )
        embedding = embedding + self.backbone(data.scage_backbone_role.long().clamp(0, 2))
        return self.norm(embedding)


class MIPSPathNodeBias(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8, max_path_nodes=6):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(int(hidden_dim), int(num_heads), bias=False)
            for _ in range(int(max_path_nodes))
        ])

    def forward(self, initial_nodes, path_index, path_mask):
        safe_index = path_index.clamp_min(0)
        values = []
        for position, projection in enumerate(self.projections):
            score = projection(initial_nodes[safe_index[:, position]])
            values.append(score * path_mask[:, position].unsqueeze(-1).to(score.dtype))
        denominator = path_mask.sum(dim=1, keepdim=True).clamp_min(1).to(initial_nodes.dtype)
        return torch.stack(values, dim=1).sum(dim=1) / denominator


class PBCDistanceBias(nn.Module):
    def __init__(self, num_heads=8, num_rbf=64, max_distance=12.0):
        super().__init__()
        centers = torch.linspace(0.0, float(max_distance), int(num_rbf))
        width = centers[1] - centers[0] if centers.numel() > 1 else centers.new_tensor(1.0)
        self.register_buffer("centers", centers)
        self.register_buffer("width", width)
        self.projection = nn.Linear(int(num_rbf), int(num_heads), bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, distance, valid):
        difference = (distance.float().unsqueeze(-1) - self.centers) / self.width.clamp_min(1e-8)
        bias = self.projection(torch.exp(-0.5 * difference.square()))
        return bias * valid.unsqueeze(-1).to(bias.dtype)


class MIPSPeriodicAttention(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.1, attention_dropout=0.1):
        super().__init__()
        if int(hidden_dim) % int(num_heads):
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.qkv = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_dropout = nn.Dropout(float(dropout))
        self.attention_dropout = nn.Dropout(float(attention_dropout))
        self.last_attention = None

    def forward(self, x, edge_index, attention_bias):
        residual = x
        qkv = self.qkv(self.norm(x)).view(
            x.size(0), 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=1)
        source, target = edge_index.long()
        logits = (query[target] * key[source]).sum(dim=-1) * self.scale
        logits = logits.float() + attention_bias.float()
        weights = softmax(logits, target, num_nodes=x.size(0), dim=0)
        weights = self.attention_dropout(weights).to(value.dtype)
        self.last_attention = weights.detach()
        messages = weights.unsqueeze(-1) * value[source]
        aggregated = scatter(
            messages, target, dim=0, dim_size=x.size(0), reduce="sum"
        ).reshape(x.size(0), self.hidden_dim)
        return residual + self.output_dropout(self.output(aggregated))


class MIPSPeriodicFeedForward(nn.Module):
    def __init__(self, hidden_dim=512, dropout=0.1, activation_dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.fc1 = nn.Linear(int(hidden_dim), 4 * int(hidden_dim))
        self.fc2 = nn.Linear(4 * int(hidden_dim), int(hidden_dim))
        self.activation_dropout = nn.Dropout(float(activation_dropout))
        self.output_dropout = nn.Dropout(float(dropout))

    def forward(self, x):
        update = self.fc2(self.activation_dropout(F.gelu(self.fc1(self.norm(x)))))
        return x + self.output_dropout(update)


class MIPSPeriodicLayer(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = MIPSPeriodicAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=dropout,
        )
        self.feed_forward = MIPSPeriodicFeedForward(
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation_dropout=dropout,
        )

    def forward(self, x, edge_index, attention_bias):
        return self.feed_forward(self.attention(x, edge_index, attention_bias))


class MIPSPeriodicGraphEncoder(nn.Module):
    expects_data = True
    uses_geometry = True
    architecture_name = "mips_periodic_pyg_sparse_lga"

    def __init__(
        self,
        num_layer=6,
        emb_dim=512,
        num_heads=8,
        dropout=0.1,
        max_hops=5,
        num_rbf=64,
        max_distance=12.0,
        num_kernels=128,
    ):
        super().__init__()
        if int(max_hops) != 5:
            raise ValueError("The SCAGE MIPS-PBC route is fixed to max_hops=5")
        self.num_layer = int(num_layer)
        self.emb_dim = int(emb_dim)
        self.num_heads = int(num_heads)
        self.max_hops = int(max_hops)
        self.atom_embedding = PeriodicAtomEmbedding(self.emb_dim, num_kernels)
        self.spd_embedding = nn.Embedding(self.max_hops + 1, self.num_heads)
        nn.init.zeros_(self.spd_embedding.weight)
        self.path_bias = MIPSPathNodeBias(
            self.emb_dim, self.num_heads, self.max_hops + 1
        )
        self.distance_bias = PBCDistanceBias(
            self.num_heads, num_rbf=num_rbf, max_distance=max_distance
        )
        self.layers = nn.ModuleList([
            MIPSPeriodicLayer(self.emb_dim, self.num_heads, dropout)
            for _ in range(self.num_layer)
        ])
        self.final_norm = nn.LayerNorm(self.emb_dim)

    def _validate(self, data):
        required = list(SCAGE_CATEGORICAL_FEATURES) + [
            "mass", "van_der_waals_radius", "partial_charge",
            "scage_backbone_role", "lga_edge_index", "lga_spd",
            "lga_path_index", "lga_path_mask", "lga_pbc_distance",
            "lga_geometry_valid", "graph_available", "batch",
        ]
        missing = [name for name in required if not hasattr(data, name)]
        if missing:
            raise ValueError(
                "MIPS-PBC graph input is missing fields: " + ", ".join(missing)
                + ". Rebuild the feature cache."
            )
        if data.lga_path_index.size(1) != self.max_hops + 1:
            raise ValueError("lga_path_index width does not match max_hops+1")
        if data.lga_spd.numel() and (
            int(data.lga_spd.min()) < 0 or int(data.lga_spd.max()) > self.max_hops
        ):
            raise ValueError("lga_spd is outside the configured localized range")

    def _forward_impl(self, data, atom_mask=None):
        self._validate(data)
        initial = self.atom_embedding(data, atom_mask=atom_mask)
        spd_bias = self.spd_embedding(data.lga_spd.long())
        path_bias = self.path_bias(
            initial, data.lga_path_index.long(), data.lga_path_mask.bool()
        )
        geometry_valid = data.lga_geometry_valid.bool()
        geometry_mask = getattr(data, "lga_geometry_pair_mask", None)
        if geometry_mask is not None:
            geometry_valid = geometry_valid & ~geometry_mask.bool()
        geometry_bias = self.distance_bias(
            data.lga_pbc_distance.float(), geometry_valid
        )
        attention_bias = spd_bias + path_bias + geometry_bias

        x = initial
        for layer in self.layers:
            x = layer(x, data.lga_edge_index.long(), attention_bias)
        x = self.final_norm(x)
        graph_available = data.graph_available.bool().flatten()
        node_available = graph_available[data.batch.long()]
        x = x * node_available.unsqueeze(-1).to(x.dtype)
        graph = global_mean_pool(x, data.batch.long())
        graph = graph * graph_available.unsqueeze(-1).to(graph.dtype)
        return graph, x

    def forward(self, data):
        return self._forward_impl(data)

    def forward_with_x(self, data, x_override):
        if x_override.shape != data.x.shape:
            raise ValueError("Masked x override must have the same shape as data.x")
        atom_mask = (x_override.abs().sum(dim=-1) == 0) & (
            data.x.abs().sum(dim=-1) != 0
        )
        return self._forward_impl(data, atom_mask=atom_mask)

    def forward_geometry_pretext(self, data):
        return self._forward_impl(data)

    def forward_topology_only(self, data):
        previous = getattr(data, "lga_geometry_pair_mask", None)
        data.lga_geometry_pair_mask = torch.ones_like(
            data.lga_geometry_valid, dtype=torch.bool
        )
        try:
            return self._forward_impl(data)
        finally:
            if previous is None:
                delattr(data, "lga_geometry_pair_mask")
            else:
                data.lga_geometry_pair_mask = previous
