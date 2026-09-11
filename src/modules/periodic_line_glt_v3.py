"""Galformer-style center-image Graph Line Transformer for MTS-GLT-v3."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import softmax
from torch_scatter import scatter

from src.dataset.periodic_line_glt_complete import BOND_FEATURE_DIM
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


class CompleteTrimerGLTEncoder(nn.Module):
    """Six-layer GLT over every physical bond in a frozen three-RU Trimer.

    This is deliberately separate from :class:`LocalPeriodicGraphLineTransformerV3`.
    The legacy image/N+1/N+2 model keeps its categorical embeddings and sidecar
    contract; this encoder consumes the complete-Trimer ``[M,14]`` bond feature
    field and encodes endpoint elements through a shared 101-to-256 projection.
    """

    architecture_name = "MTS-GLT-v2-Complete-Trimer-Galformer-BondFeatures"

    def __init__(self, hidden=512, layers=6, heads=8, dropout=0.1):
        super().__init__()
        if (hidden, layers, heads) != (512, 6, 8):
            raise ValueError("complete-Trimer GLT fixes hidden/layers/heads to 512/6/8")
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.endpoint_projection = nn.Linear(101, 256)
        self.bond_feature_projection = nn.Linear(BOND_FEATURE_DIM, 256)
        self.distance_basis = PairConditionedGaussianRBF(256, 0.0, 3.75)
        self.distance_projection = nn.Linear(256, 256)
        self.input_norm = nn.LayerNorm(hidden)
        self.mask_token = nn.Parameter(torch.zeros(hidden))
        self.angle_basis = GaussianRBF(128, 0.0, math.pi)
        self.angle_projection = nn.Linear(128, heads, bias=False)
        self.identity_self_bias = nn.Parameter(torch.zeros(heads))
        self.layers = nn.ModuleList([
            ImageLineLayer(hidden, heads, dropout) for _ in range(layers)
        ])
        self.final_norm = nn.LayerNorm(hidden)

    @staticmethod
    def _endpoint_one_hot(z):
        """Map raw Z values to Z=1..100 plus a final unknown category."""

        z = z.long()
        index = torch.where((z >= 1) & (z <= 100), z - 1, z.new_full(z.shape, 100))
        return F.one_hot(index, num_classes=101).to(dtype=torch.float32)

    def _line_inputs(self, data, token_mask=None):
        z_a = data.glt3_token_endpoint_z_a.long()
        z_b = data.glt3_token_endpoint_z_b.long()
        features = getattr(data, "glt3_token_bond_features", None)
        if features is None:
            raise ValueError(
                "complete-Trimer GLT requires glt3_token_bond_features [M,14]"
            )
        if features.ndim != 2 or tuple(features.shape) != (z_a.numel(), BOND_FEATURE_DIM):
            raise ValueError("glt3_token_bond_features must have shape [M,14]")
        features = features.float()
        if not bool(torch.isfinite(features).all()):
            raise ValueError("glt3_token_bond_features contain NaN/Inf")
        endpoint = self.endpoint_projection(self._endpoint_one_hot(z_a))
        endpoint = endpoint + self.endpoint_projection(self._endpoint_one_hot(z_b))
        chemical = endpoint + self.bond_feature_projection(features)
        distance = data.glt3_token_distance.float()
        if not bool(torch.isfinite(distance).all()):
            raise ValueError("complete-Trimer token distances contain NaN/Inf")
        radial = self.distance_projection(self.distance_basis(
            distance, z_a.clamp(0, ELEMENT_INPUT_VOCAB - 1),
            z_b.clamp(0, ELEMENT_INPUT_VOCAB - 1),
        ))
        states = self.input_norm(torch.cat([chemical, radial.to(chemical.dtype)], dim=-1))
        if token_mask is not None:
            mask = token_mask.bool().reshape(-1)
            if mask.numel() != states.size(0):
                raise ValueError("complete-Trimer token_mask length mismatch")
            states = torch.where(
                mask.unsqueeze(-1), self.mask_token.to(states.dtype).unsqueeze(0), states,
            )
        return states

    def _relations(self, data, dtype, angle_mask=None):
        valid = data.glt3_relation_valid.bool().reshape(-1)
        source_all = data.glt3_relation_source.long().reshape(-1)
        target_all = data.glt3_relation_target.long().reshape(-1)
        source = source_all[valid]
        target = target_all[valid]
        angles = data.glt3_relation_angle.float()[valid]
        if not bool(torch.isfinite(angles).all()):
            raise ValueError("complete-Trimer relation angles contain NaN/Inf")
        bias = self.angle_projection(self.angle_basis(angles)).to(dtype)
        if angle_mask is not None:
            mask = angle_mask.bool().reshape(-1)
            if mask.numel() == data.glt3_token_atom_a.numel():
                relation_mask = mask[source] | mask[target]
            elif mask.numel() == int(valid.sum()):
                relation_mask = mask
            else:
                raise ValueError("complete-Trimer angle_mask length mismatch")
            bias = bias.masked_fill(relation_mask.unsqueeze(-1), 0.0)
        count = int(data.glt3_token_atom_a.numel())
        identity = torch.arange(count, device=source.device, dtype=torch.long)
        return (
            torch.cat([source, identity]), torch.cat([target, identity]),
            torch.cat([bias, self.identity_self_bias.to(dtype).expand(count, -1)]),
        )

    def _center_readout(self, data, lines):
        count = int(data.glt3_geometry_valid.numel())
        center = data.glt3_token_center_internal.bool() & data.glt3_token_valid.bool()
        line_batch = data.glt3_token_batch.long()
        if line_batch.numel() != lines.size(0):
            raise ValueError("complete-Trimer token batch length mismatch")
        selected = lines[center]
        selected_batch = line_batch[center]
        if selected.numel():
            sums = scatter(selected, selected_batch, dim=0, dim_size=count, reduce="sum")
            numbers = scatter(
                torch.ones((selected.size(0), 1), device=selected.device, dtype=selected.dtype),
                selected_batch, dim=0, dim_size=count, reduce="sum",
            )
        else:
            sums = lines.new_zeros((count, lines.size(-1)))
            numbers = lines.new_zeros((count, 1))
        valid_graph = numbers.squeeze(-1) > 0
        graph_geometry = sums / numbers.clamp_min(1.0)
        graph_geometry = graph_geometry * valid_graph.to(graph_geometry.dtype).unsqueeze(-1)
        graph_geometry = graph_geometry * data.glt3_geometry_valid.to(graph_geometry.dtype).unsqueeze(-1)
        return graph_geometry, center, selected_batch, valid_graph

    def forward(self, data, *, token_mask=None, angle_mask=None):
        states = self._line_inputs(data, token_mask=token_mask)
        source, target, bias = self._relations(data, states.dtype, angle_mask=angle_mask)
        for layer in self.layers:
            states = layer(states, source, target, bias)
        states = self.final_norm(states)
        graph_geometry, center, center_batch, center_graph_valid = self._center_readout(data, states)
        degree = torch.bincount(target, minlength=states.size(0))
        return {
            "line_states": states,
            "center_line_states": states[center],
            "center_mask": center,
            "center_batch": center_batch,
            "graph_geometry": graph_geometry,
            "z_line": graph_geometry,
            "graph_geometry_valid": center_graph_valid,
            "relation_source": source,
            "relation_target": target,
            "relation_bias": bias,
            "token_count": states.new_tensor(states.size(0), dtype=torch.long),
            "relation_count": states.new_tensor(source.size(0), dtype=torch.long),
            "degree": degree,
        }


def canonical_atom_mean(canonical_states, canonical_graph_index, graph_count=None):
    """Mean-pool O8 canonical atom states for the complete-Trimer fusion API."""

    if canonical_states.ndim != 2 or canonical_graph_index.ndim != 1:
        raise ValueError("canonical states/index must be [N,512]/[N]")
    if canonical_states.size(0) != canonical_graph_index.numel():
        raise ValueError("canonical state/index length mismatch")
    if graph_count is None:
        graph_count = int(canonical_graph_index.max().item()) + 1 if canonical_graph_index.numel() else 0
    sums = scatter(canonical_states, canonical_graph_index.long(), dim=0, dim_size=int(graph_count), reduce="sum")
    counts = scatter(
        torch.ones((canonical_states.size(0), 1), device=canonical_states.device, dtype=canonical_states.dtype),
        canonical_graph_index.long(), dim=0, dim_size=int(graph_count), reduce="sum",
    )
    return sums / counts.clamp_min(1.0)


class CompleteTrimerGLTFusionRegressor(nn.Module):
    """Downstream ``LN(GAP(O8)) || masked-LN(GAP(center GLT))`` regressor."""

    def __init__(self, dropout=0.1):
        super().__init__()
        self.o8_norm = nn.LayerNorm(512)
        self.glt_norm = nn.LayerNorm(512)
        self.predictor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(approximate="none"),
            nn.Dropout(float(dropout)),
            nn.Linear(512, 1),
        )

    def forward(self, o8_graph, glt_graph, *, glt_valid=None):
        if o8_graph.ndim != 2 or glt_graph.ndim != 2:
            raise ValueError("O8 and GLT graph states must be [B,512]")
        if o8_graph.shape != glt_graph.shape or o8_graph.size(1) != 512:
            raise ValueError("O8/GLT graph state shape mismatch")
        o8 = self.o8_norm(o8_graph)
        glt = self.glt_norm(glt_graph)
        if glt_valid is not None:
            valid = glt_valid.bool().reshape(-1)
            if valid.numel() != glt.size(0):
                raise ValueError("GLT validity length mismatch")
            glt = glt * valid.to(glt.dtype).unsqueeze(-1)
        return self.predictor(torch.cat([o8, glt], dim=-1))


CompleteTrimerGLTRegressor = CompleteTrimerGLTFusionRegressor


__all__ = [
    "ELEMENT_MASK", "ATOM_PAIR_TARGETS", "CONJUGATED_MASK",
    "atom_pair_label", "LocalPeriodicGraphLineTransformerV3",
    "CompleteTrimerGLTEncoder", "CompleteTrimerGLTFusionRegressor",
    "CompleteTrimerGLTRegressor", "canonical_atom_mean",
]
