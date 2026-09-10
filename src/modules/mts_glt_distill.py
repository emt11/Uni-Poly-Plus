"""N+1/N+2 GLT teacher and 2-D O8+MD200 student."""

from __future__ import annotations

import math
import torch
from torch import nn
from torch_geometric.utils import softmax
from torch_scatter import scatter

from .mips_local_graph import MIPSLocalGraphEncoder
from .periodic_line_glt_v3 import (
    BOND_TYPE_MASK, CONJUGATED_MASK, ELEMENT_MASK, GaussianRBF,
    PairConditionedGaussianRBF, STEREO_MASK,
)


class SourceQPreLNAttention(nn.Module):
    def __init__(self, hidden=512, heads=8, dropout=0.1):
        super().__init__()
        self.heads, self.head_dim = int(heads), int(hidden) // int(heads)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.output = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, states, edge_index, bias):
        count = states.size(0)
        q, k, v = self.qkv(states).chunk(3, dim=-1)
        q = q.view(count, self.heads, self.head_dim)
        k = k.view(count, self.heads, self.head_dim)
        v = v.view(count, self.heads, self.head_dim)
        source, target = edge_index.long()
        score = (q[source].float() * k[target].float()).sum(-1) * self.head_dim ** -0.5
        weight = softmax(score + bias.float(), target, num_nodes=count).to(v.dtype)
        message = self.attention_dropout(weight).unsqueeze(-1) * v[source]
        aggregate = torch.zeros_like(v)
        aggregate.index_add_(0, target, message)
        return self.dropout(self.output(aggregate.reshape(count, self.heads * self.head_dim)))


class SourceQPreLNLayer(nn.Module):
    def __init__(self, hidden=512, heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attention = SourceQPreLNAttention(hidden, heads, dropout)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.GELU(approximate="none"),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, hidden), nn.Dropout(dropout),
        )

    def forward(self, states, edge_index, bias):
        states = states + self.attention(self.norm1(states), edge_index, bias)
        return states + self.ffn(self.norm2(states))


class PreLNO8Encoder(MIPSLocalGraphEncoder):
    """Canonical O8 with source-Q/target-K Pre-LN updates."""

    def __init__(self, dropout=0.1):
        super().__init__(
            core="paper_corrected", num_layer=6, emb_dim=512, num_heads=8,
            dropout=dropout, max_hops=2, use_descriptors=True,
            graph_geometry_mode="trimer_scage_mcl", use_star_rbf=False,
            use_mcl=False, topology_attention_variant="o8",
        )
        self.layers = nn.ModuleList([SourceQPreLNLayer(512, 8, dropout) for _ in range(6)])
        self.final_norm = nn.Identity()
        self.md_residual = nn.Identity()
        for parameter in self.star_distance_bias.parameters():
            parameter.requires_grad = False

    def canonical(self, data, *, atom_mask=None):
        graph, nodes = self._forward_impl(data, atom_mask=atom_mask, use_star=False, use_md=False)
        _, canonical = self._canonical_pool(nodes, data)
        return graph, nodes, canonical


class AtomicConditionedMD200(nn.Module):
    def __init__(self, hidden=512, md_hidden=64, dropout=0.1):
        super().__init__()
        self.md = nn.Sequential(nn.LayerNorm(200), nn.Linear(200, md_hidden), nn.GELU())
        self.atom_norm = nn.LayerNorm(hidden)
        self.query = nn.Linear(hidden, md_hidden, bias=False)
        self.key = nn.Linear(md_hidden, md_hidden, bias=False)
        self.value = nn.Linear(md_hidden, hidden, bias=False)
        self.bias = nn.Parameter(torch.tensor(math.log(0.05 / 0.95)))
        nn.init.zeros_(self.query.weight)

    def forward(self, canonical, data, md200=None):
        values = data.mips_md if md200 is None else md200
        valid_graph = data.mips_md_valid.bool().flatten()
        graph_id = data.canonical_graph_index.long()
        if bool((~torch.isfinite(values[valid_graph])).any()):
            raise ValueError("valid MD200 row contains NaN/Inf")
        # Invalid descriptor rows never enter LayerNorm/MLP.  Their encoded
        # value and therefore their atom update are exactly zero.
        md = canonical.new_zeros((values.size(0), 64), dtype=torch.float32)
        if bool(valid_graph.any()):
            md[valid_graph] = self.md(values[valid_graph].float()).float()
        q = self.query(self.atom_norm(canonical))
        k = self.key(md)[graph_id]
        gate = torch.sigmoid((q.float() * k.float()).sum(-1) / math.sqrt(64.0) + self.bias.float())
        valid = valid_graph[graph_id].to(canonical.dtype)
        update = self.value(md)[graph_id].to(canonical.dtype)
        return canonical + valid.unsqueeze(-1) * gate.to(canonical.dtype).unsqueeze(-1) * update


class DistillLineAttention(nn.Module):
    def __init__(self, hidden=512, heads=8, dropout=0.1):
        super().__init__()
        self.heads, self.head_dim = heads, hidden // heads
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.output = nn.Linear(hidden, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states, source, target, bias):
        count = states.size(0)
        q, k, v = self.qkv(states).chunk(3, dim=-1)
        q = q.view(count, self.heads, self.head_dim)
        k = k.view(count, self.heads, self.head_dim)
        v = v.view(count, self.heads, self.head_dim)
        score = (q[target].float() * k[source].float()).sum(-1) * self.head_dim ** -0.5
        weight = softmax(score + bias.float(), target, num_nodes=count).to(v.dtype)
        message = self.dropout(weight).unsqueeze(-1) * v[source]
        aggregate = torch.zeros_like(v)
        aggregate.index_add_(0, target, message)
        return self.output(aggregate.reshape(count, self.heads * self.head_dim))


class DistillLineLayer(nn.Module):
    def __init__(self, hidden=512, heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attention = DistillLineAttention(hidden, heads, dropout)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden), nn.Dropout(dropout),
        )

    def forward(self, states, source, target, bias):
        states = states + self.attention(self.norm1(states), source, target, bias)
        return states + self.ffn(self.norm2(states))


class NPlusGLTTeacher(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.element = nn.Embedding(ELEMENT_MASK + 1, 256)
        self.bond = nn.Embedding(BOND_TYPE_MASK + 1, 256)
        self.stereo = nn.Embedding(STEREO_MASK + 1, 256)
        self.conjugated = nn.Embedding(CONJUGATED_MASK + 1, 256)
        self.distance_basis = PairConditionedGaussianRBF(256, 0.0, 3.75)
        self.distance_projection = nn.Linear(256, 256)
        self.distance_mask = nn.Parameter(torch.zeros(256))
        self.input_norm = nn.LayerNorm(512)
        self.angle_basis = GaussianRBF(128, 0.0, math.pi)
        self.angle_projection = nn.Linear(128, 8, bias=False)
        self.self_bias = nn.Parameter(torch.zeros(8))
        self.layers = nn.ModuleList([DistillLineLayer(512, 8, dropout) for _ in range(6)])
        self.teacher_projection = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 256)
        )

    def forward(self, data, *, token_mask=None, angle_mask=None):
        count = data.glt3_token_atom_a.numel()
        device = data.glt3_token_atom_a.device
        mask = torch.zeros(count, dtype=torch.bool, device=device) if token_mask is None else token_mask.bool()
        z_a = data.glt3_token_endpoint_z_a.long().masked_fill(mask, ELEMENT_MASK)
        z_b = data.glt3_token_endpoint_z_b.long().masked_fill(mask, ELEMENT_MASK)
        bond = data.glt3_token_bond_type.long().masked_fill(mask, BOND_TYPE_MASK)
        stereo = data.glt3_token_stereo.long().masked_fill(mask, STEREO_MASK)
        conj = data.glt3_token_conjugated.long().masked_fill(mask, CONJUGATED_MASK)
        chemical = self.element(z_a) + self.element(z_b) + self.bond(bond) + self.stereo(stereo) + self.conjugated(conj)
        radial = self.distance_projection(self.distance_basis(data.glt3_token_distance, z_a, z_b))
        radial = torch.where(mask.unsqueeze(-1), self.distance_mask.unsqueeze(0).to(radial.dtype), radial)
        states = self.input_norm(torch.cat([chemical, radial.to(chemical.dtype)], dim=-1))
        valid = data.glt3_relation_valid.bool()
        source = data.glt3_relation_source.long()[valid]
        target = data.glt3_relation_target.long()[valid]
        angle = self.angle_projection(self.angle_basis(data.glt3_relation_angle[valid])).to(states.dtype)
        if angle_mask is not None:
            angle = angle.masked_fill(angle_mask.bool()[valid].unsqueeze(-1), 0.0)
        identity = torch.arange(count, device=device)
        source = torch.cat([source, identity])
        target = torch.cat([target, identity])
        bias = torch.cat([angle, self.self_bias.to(states.dtype).expand(count, -1)])
        for layer in self.layers:
            states = layer(states, source, target, bias)
        projected = self.teacher_projection(states)
        center = data.glt3_token_center_internal.bool() & data.glt3_token_valid.bool()
        return {
            "line_states": states, "projected_lines": projected,
            "center_mask": center, "center_projected": projected[center],
            "center_batch": data.glt3_token_batch.long()[center],
        }


class DistillStudent(nn.Module):
    architecture_name = "MTS-GLT-v2-Distill-Student"

    def __init__(self, dropout=0.1):
        super().__init__()
        self.o8 = PreLNO8Encoder(dropout)
        self.md_residual = AtomicConditionedMD200(512, 64, dropout)

    @staticmethod
    def pool(data, canonical):
        count = int(data.graph_available.numel())
        graph = scatter(canonical, data.canonical_graph_index.long(), dim=0, dim_size=count, reduce="mean")
        return graph * data.graph_available.to(graph.dtype).unsqueeze(-1)

    def encode(self, data, *, atom_mask=None, md200=None):
        _, _, canonical = self.o8.canonical(data, atom_mask=atom_mask)
        fused = self.md_residual(canonical, data, md200=md200)
        return canonical, fused

    def forward(self, data):
        _, fused = self.encode(data)
        return self.pool(data, fused)


class AtomPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(512, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 101))

    def forward(self, states):
        return self.net(states)


class StudentLineProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 512), nn.GELU(), nn.Linear(512, 256))

    def forward(self, data, canonical):
        mask = data.glt3_token_center_internal.bool() & data.glt3_token_valid.bool()
        a = canonical[data.glt3_token_atom_a.long()[mask]]
        b = canonical[data.glt3_token_atom_b.long()[mask]]
        return self.net(torch.cat([a + b, (a - b).abs()], dim=-1)), data.glt3_token_batch.long()[mask]


def fixed_rbf(values, size, lower, upper):
    centers = torch.linspace(lower, upper, size, device=values.device, dtype=torch.float32)
    width = (upper - lower) / max(1, size - 1)
    return torch.exp(-0.5 * ((values.float().unsqueeze(-1) - centers) / width) ** 2)


__all__ = [
    "AtomPredictor", "AtomicConditionedMD200", "DistillStudent",
    "NPlusGLTTeacher", "PreLNO8Encoder", "SourceQPreLNAttention",
    "StudentLineProjection", "fixed_rbf",
]
