"""MIPS-aligned local periodic 3-D Graph Line Transformer v2.

The v2 encoder deliberately reuses the frozen v1 sidecar.  It derives
Gaussian observation moments, identity self-relations and line-to-atom
incidence at runtime, so no cached geometry is rewritten.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn.models.dimenet import SphericalBasisLayer
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


class TorsionRelationBiasV2(nn.Module):
    """Reflection-invariant torsion moments on existing directed relations."""

    def __init__(self, basis_size=128, heads=8, max_count=15, mode="full"):
        super().__init__()
        if mode not in {"count", "full"}:
            raise ValueError("torsion mode must be count or full")
        self.mode = str(mode)
        centers = torch.linspace(-1.0, 1.0, int(basis_size))
        spacing = float(centers[1] - centers[0])
        self.centers = nn.Parameter(centers)
        self.raw_width = nn.Parameter(torch.full(
            (int(basis_size),), _inverse_softplus(spacing)
        ))
        self.mean_projection = nn.Linear(int(basis_size), int(heads), bias=False)
        self.variance_projection = nn.Linear(int(basis_size), int(heads), bias=False)
        self.count_embedding = nn.Embedding(int(max_count) + 1, int(heads))
        nn.init.zeros_(self.mean_projection.weight)
        nn.init.zeros_(self.variance_projection.weight)
        nn.init.zeros_(self.count_embedding.weight)

    def forward(self, data, relation_count):
        values = data.glt_torsion_observation_value.to(self.centers.dtype)
        indices = data.glt_torsion_observation_relation.long()
        counts = data.glt_relation_torsion_count.long()
        if int(counts.numel()) != int(relation_count):
            raise ValueError("torsion count/relation length mismatch")
        width = F.softplus(self.raw_width).clamp_min(1e-6)
        encoded = torch.exp(
            -0.5 * ((values.unsqueeze(-1) - self.centers) / width) ** 2
        )
        basis_size = int(self.centers.numel())
        summed = scatter(
            encoded, indices, dim=0, dim_size=relation_count, reduce="sum"
        ) if values.numel() else self.centers.new_zeros((relation_count, basis_size))
        denominator = counts.clamp_min(1).to(summed.dtype).unsqueeze(-1)
        mean = summed / denominator
        if values.numel():
            centered = encoded - mean[indices]
            variance = scatter(
                centered.square(), indices, dim=0,
                dim_size=relation_count, reduce="sum",
            ) / denominator
        else:
            variance = torch.zeros_like(mean)
        covered = counts > 0
        if self.mode == "count":
            mean = torch.zeros_like(mean)
            variance = torch.zeros_like(variance)
        bias = (
            self.mean_projection(mean)
            + self.variance_projection(variance)
            + self.count_embedding(counts.clamp(0, self.count_embedding.num_embeddings - 1))
        )
        bias = bias * covered.to(bias.dtype).unsqueeze(-1)
        return bias, {
            "torsion_bias": bias,
            "torsion_count": counts,
            "torsion_covered": covered,
            "torsion_source_cross_ru": data.glt_relation_torsion_source_cross_ru.bool(),
        }


class JointRadialAngularRelationBiasV2(nn.Module):
    """Official DimeNet SBF over matched source-line distance and angle."""

    def __init__(
        self, heads=8, mode="radial", num_spherical=7, num_radial=6,
        cutoff=3.75, envelope_exponent=5, reference_distance=1.407050,
    ):
        super().__init__()
        if str(mode) not in {"control", "radial"}:
            raise ValueError("joint basis mode must be control or radial")
        self.mode = str(mode)
        self.num_spherical = int(num_spherical)
        self.num_radial = int(num_radial)
        self.dimension = self.num_spherical * self.num_radial
        self.cutoff = float(cutoff)
        self.reference_distance = float(reference_distance)
        self.basis = SphericalBasisLayer(
            self.num_spherical, self.num_radial, self.cutoff,
            envelope_exponent=int(envelope_exponent),
        )
        self.mean_projection = nn.Linear(self.dimension, int(heads), bias=False)
        self.variance_projection = nn.Linear(self.dimension, int(heads), bias=False)
        nn.init.zeros_(self.mean_projection.weight)
        nn.init.zeros_(self.variance_projection.weight)

    def observation_features(self, distances, angles):
        distances = torch.as_tensor(distances).float().reshape(-1)
        angles = torch.as_tensor(
            angles, device=distances.device, dtype=distances.dtype
        ).reshape(-1)
        if distances.numel() != angles.numel():
            raise ValueError("joint basis observation length mismatch")
        if self.mode == "control":
            distances = torch.full_like(distances, self.reference_distance)
        if distances.numel() == 0:
            return distances.new_zeros((0, self.dimension))
        identity = torch.arange(distances.numel(), device=distances.device)
        return self.basis(distances, angles, identity)

    def forward(self, data, relation_count):
        distances = data.glt_relation_source_distances.float().reshape(-1, 3)
        angles = data.glt_relation_observation_angles.float().reshape(-1, 3)
        distance_valid = data.glt_relation_source_distance_valid.bool().reshape(-1, 3)
        counts = data.glt_relation_observation_count.long().reshape(-1)
        relation_valid = data.glt_relation_valid.bool().reshape(-1)
        fallback = data.glt_relation_is_fallback.bool().reshape(-1)
        if not (
            distances.shape == angles.shape == distance_valid.shape
            and distances.size(0) == int(relation_count)
            and counts.numel() == relation_valid.numel() == fallback.numel()
            == int(relation_count)
        ):
            raise ValueError("joint basis relation tensor mismatch")
        slots = torch.arange(3, device=distances.device).unsqueeze(0)
        mask = (
            (slots < counts.unsqueeze(1)) & distance_valid
            & relation_valid.unsqueeze(1) & (~fallback).unsqueeze(1)
            & torch.isfinite(distances) & torch.isfinite(angles)
            & (distances > 0.0)
        )
        observation_relation = mask.nonzero(as_tuple=False)[:, 0]
        selected_distances = distances[mask]
        selected_angles = angles[mask]
        features = self.observation_features(selected_distances, selected_angles)
        basis_sum = features.new_zeros((int(relation_count), self.dimension))
        square_sum = torch.zeros_like(basis_sum)
        if features.numel():
            basis_sum = scatter(
                features, observation_relation, dim=0,
                dim_size=int(relation_count), reduce="sum",
            )
            square_sum = scatter(
                features.square(), observation_relation, dim=0,
                dim_size=int(relation_count), reduce="sum",
            )
        observed_counts = mask.sum(dim=1)
        denominator = observed_counts.clamp_min(1).to(features.dtype).unsqueeze(-1)
        mean = basis_sum / denominator
        variance = (square_sum / denominator - mean.square()).clamp_min(0.0)
        active = (observed_counts > 0) & relation_valid & (~fallback)
        mean = mean * active.to(mean.dtype).unsqueeze(-1)
        variance = variance * active.to(variance.dtype).unsqueeze(-1)
        bias = self.mean_projection(mean) + self.variance_projection(variance)
        bias = bias * active.to(bias.dtype).unsqueeze(-1)
        return bias, {
            "joint_bias": bias,
            "basis_mean": mean,
            "basis_variance": variance,
            "observation_features": features,
            "observation_relation": observation_relation,
            "observation_distance": selected_distances,
            "observation_angle": selected_angles,
            "observation_count": observed_counts,
            "source_cross_ru": data.glt_relation_source_cross_ru.bool(),
            "active_relation": active,
        }


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

    def attention_weights(self, states, source, target, relation_bias, qk_states=None):
        node_count = int(states.size(0))
        q, k, v = self.qkv(states).chunk(3, dim=-1)
        if qk_states is not None:
            conditioned_q, conditioned_k, _ = self.qkv(qk_states).chunk(3, dim=-1)
            q, k = conditioned_q, conditioned_k
        q = q.view(node_count, self.heads, self.head_dim)
        k = k.view(node_count, self.heads, self.head_dim)
        v = v.view(node_count, self.heads, self.head_dim)
        if self.variant == "mips":
            left, right = q[target], k[source]
        else:
            left, right = q[source], k[target]
        scores = (left * right).sum(dim=-1) * self.scale + relation_bias
        weights = softmax(scores, target, num_nodes=node_count).to(v.dtype)
        return weights, v

    def forward(self, states, source, target, relation_bias, qk_states=None):
        node_count = int(states.size(0))
        weights, v = self.attention_weights(
            states, source, target, relation_bias, qk_states=qk_states
        )
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

    def forward(self, states, source, target, relation_bias, qk_modulation=None):
        normalized = self.norm1(states)
        qk_states = (
            None if qk_modulation is None
            else (1.0 + qk_modulation) * normalized
        )
        states = states + self.attention(
            normalized, source, target, relation_bias, qk_states=qk_states
        )
        return states + self.ffn(self.norm2(states))


class LocalPeriodicGraphLineTransformerV2(nn.Module):
    """Local periodic line encoder with canonical atom incidence readout."""

    def __init__(
        self,
        hidden_size=512,
        layers=12,
        heads=8,
        dropout=0.1,
        attention_variant="mips",
        distance_basis=256,
        angle_basis=128,
        distance_upper=3.75,
        line_conditioning_mode="none",
        attention_conditioning_mode="none",
        torsion_mode="none",
        joint_basis_mode="none",
        metadata_mode="full",
    ):
        super().__init__()
        if str(metadata_mode) not in {"full", "dedup"}:
            raise ValueError("metadata mode must be full or dedup")
        self.hidden_size = int(hidden_size)
        self.layers_count = int(layers)
        self.attention_variant = str(attention_variant)
        self.metadata_mode = str(metadata_mode)
        self.atom_embedding = nn.Embedding(MAX_ATOMIC_NUMBER + 2, hidden_size)
        self.distance_basis = LearnedGaussianMoments(
            distance_basis, 0.0, distance_upper, pair_conditioned=True
        )
        self.angle_basis = LearnedGaussianMoments(angle_basis, 0.0, math.pi)
        self.distance_mean_projection = nn.Linear(distance_basis, hidden_size)
        self.distance_variance_projection = nn.Linear(distance_basis, hidden_size)
        self.angle_mean_projection = nn.Linear(angle_basis, heads, bias=False)
        self.angle_variance_projection = nn.Linear(angle_basis, heads, bias=False)
        # ``full`` retains the historical distance-observation-count and
        # angle-multiplicity additive features.  ``dedup`` deliberately
        # removes only those two redundant metadata contributions; all
        # observation moments, shift/count fields that carry geometry, and
        # relation topology remain unchanged.  The parameters are omitted in
        # dedup rather than retained as unused dummy tensors, so checkpoint
        # accounting reflects the actual model.
        self.token_count_embedding = (
            nn.Embedding(4, hidden_size)
            if self.metadata_mode == "full" else None
        )
        self.token_shift_embedding = nn.Embedding(3, hidden_size)
        self.relation_count_embedding = nn.Embedding(4, heads)
        self.relation_multiplicity_embedding = (
            nn.Embedding(4, heads)
            if self.metadata_mode == "full" else None
        )
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
        if line_conditioning_mode not in {"none", "self", "x2l"}:
            raise ValueError("invalid GLT-v2 line conditioning mode")
        self.line_conditioning_mode = str(line_conditioning_mode)
        self.line_conditioning_projection = (
            nn.Linear(hidden_size, hidden_size)
            if self.line_conditioning_mode != "none" else None
        )
        if self.line_conditioning_projection is not None:
            nn.init.zeros_(self.line_conditioning_projection.weight)
            nn.init.zeros_(self.line_conditioning_projection.bias)
        if attention_conditioning_mode not in {"none", "self", "x2a"}:
            raise ValueError("invalid GLT-v2 attention conditioning mode")
        self.attention_conditioning_mode = str(attention_conditioning_mode)
        self.attention_conditioning_projection = (
            nn.Linear(hidden_size, hidden_size)
            if self.attention_conditioning_mode != "none" else None
        )
        if self.attention_conditioning_projection is not None:
            nn.init.zeros_(self.attention_conditioning_projection.weight)
            nn.init.zeros_(self.attention_conditioning_projection.bias)
        if (
            self.line_conditioning_projection is not None
            and self.attention_conditioning_projection is not None
        ):
            raise ValueError("line and attention conditioning are mutually exclusive")
        if torsion_mode not in {"none", "count", "full"}:
            raise ValueError("invalid GLT-v2 torsion mode")
        self.torsion_mode = str(torsion_mode)
        self.torsion_bias = (
            TorsionRelationBiasV2(angle_basis, heads, mode=self.torsion_mode)
            if self.torsion_mode != "none" else None
        )
        if joint_basis_mode not in {"none", "control", "radial"}:
            raise ValueError("invalid GLT-v2 joint basis mode")
        self.joint_basis_mode = str(joint_basis_mode)
        self.joint_basis_bias = (
            JointRadialAngularRelationBiasV2(
                heads=heads, mode=self.joint_basis_mode
            )
            if self.joint_basis_mode != "none" else None
        )

    def _condition_line_inputs(self, data, states, canonical_atom_states=None):
        if self.line_conditioning_projection is None:
            return states, None
        if self.line_conditioning_mode == "self":
            source = states.detach()
        else:
            if canonical_atom_states is None:
                raise ValueError("X2L requires canonical O8 atom states")
            source = 0.5 * (
                canonical_atom_states[data.glt_token_atom_a.long()].detach()
                + canonical_atom_states[data.glt_token_atom_b.long()].detach()
            )
        normalized = F.layer_norm(source, (self.hidden_size,))
        modulation = torch.tanh(
            self.line_conditioning_projection(normalized)
        )
        valid = data.glt_token_valid.bool()
        modulation = modulation * valid.to(modulation.dtype).unsqueeze(-1)
        conditioned = (1.0 + modulation) * states
        return conditioned, {
            "line_input": states,
            "conditioned_line_input": conditioned,
            "line_modulation": modulation,
            "line_valid": valid,
            "line_shift": data.glt_token_shift.long(),
        }

    def _attention_modulation(self, data, states, canonical_atom_states=None):
        if self.attention_conditioning_projection is None:
            return None
        if self.attention_conditioning_mode == "self":
            source = states.detach()
        else:
            if canonical_atom_states is None:
                raise ValueError("X2A requires canonical O8 atom states")
            source = 0.5 * (
                canonical_atom_states[data.glt_token_atom_a.long()].detach()
                + canonical_atom_states[data.glt_token_atom_b.long()].detach()
            )
        normalized = F.layer_norm(source, (self.hidden_size,))
        gamma = torch.tanh(self.attention_conditioning_projection(normalized))
        return gamma * data.glt_token_valid.to(gamma.dtype).unsqueeze(-1)

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
        if self.token_count_embedding is None:
            count_feature = torch.zeros_like(geometry)
        else:
            count_feature = self.token_count_embedding(counts.long().clamp(0, 3))
        shift_class = shifts.long().abs().clamp(0, MASK_SHIFT_CLASS)
        shift_feature = self.token_shift_embedding(shift_class)
        valid = data.glt_token_valid.to(geometry.dtype).unsqueeze(-1)
        return self.input_norm(endpoint + valid * (geometry + count_feature + shift_feature))

    def _relations(
        self, data, dtype, return_torsion=False, return_joint_basis=False
    ):
        real = ~data.glt_relation_is_fallback.bool()
        source = data.glt_relation_source.long()[real]
        target = data.glt_relation_target.long()[real]
        angles = data.glt_relation_observation_angles[real]
        counts = data.glt_relation_observation_count.long()[real]
        multiplicity = data.glt_relation_multiplicity.long()[real]
        valid = data.glt_relation_valid.bool()[real]
        angle_mean, angle_variance = self.angle_basis(angles, counts)
        angle_bias = (
            self.angle_mean_projection(angle_mean)
            + self.angle_variance_projection(angle_variance)
            + self.relation_count_embedding(counts.clamp(0, 3))
        )
        if self.relation_multiplicity_embedding is not None:
            angle_bias = angle_bias + self.relation_multiplicity_embedding(
                multiplicity.clamp(0, 3)
            )
        angle_bias = angle_bias * valid.to(angle_bias.dtype).unsqueeze(-1)
        torsion_views = None
        if self.torsion_bias is not None:
            torsion_all, torsion_views = self.torsion_bias(
                data, int(data.glt_relation_source.numel())
            )
            torsion_real = torsion_all[real]
        else:
            torsion_real = torch.zeros_like(angle_bias)
        bias = angle_bias + torsion_real
        joint_views = None
        if self.joint_basis_bias is not None:
            joint_all, joint_views = self.joint_basis_bias(
                data, int(data.glt_relation_source.numel())
            )
            joint_real = joint_all[real]
            bias = bias + joint_real
        else:
            joint_real = torch.zeros_like(angle_bias)
        token_count = int(data.glt_token_atom_a.numel())
        identity = torch.arange(token_count, device=source.device)
        source = torch.cat([source, identity])
        target = torch.cat([target, identity])
        self_bias = self.identity_self_bias.to(dtype).unsqueeze(0).expand(token_count, -1)
        bias = torch.cat([bias.to(dtype), self_bias], dim=0)
        if torsion_views is not None:
            torsion_views = {
                **torsion_views,
                "angle_bias": angle_bias,
                "torsion_bias_real": torsion_real,
                "real_relation_mask": real,
                "source": source[:int(angle_bias.size(0))],
                "target": target[:int(angle_bias.size(0))],
            }
        if joint_views is not None:
            joint_views = {
                **joint_views,
                "angle_bias": angle_bias,
                "joint_bias_real": joint_real,
                "real_relation_mask": real,
                "source": source[:int(angle_bias.size(0))],
                "target": target[:int(angle_bias.size(0))],
            }
        if return_joint_basis:
            auxiliary = {}
            if return_torsion and torsion_views is not None:
                auxiliary["torsion"] = torsion_views
            if joint_views is not None:
                auxiliary["joint_basis"] = joint_views
            views = auxiliary
        else:
            views = torsion_views if return_torsion else None
        return source, target, bias, views

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
        canonical_atom_states=None,
        return_attention_conditioning=False,
        return_torsion=False,
        return_joint_basis=False,
    ):
        states = self.line_inputs(
            data,
            endpoint_z_a=endpoint_z_a,
            endpoint_z_b=endpoint_z_b,
            observation_distances=observation_distances,
            observation_counts=observation_counts,
            token_shifts=token_shifts,
        )
        states, conditioning = self._condition_line_inputs(
            data, states, canonical_atom_states
        )
        attention_modulation = self._attention_modulation(
            data, states, canonical_atom_states
        )
        source, target, relation_bias, auxiliary = self._relations(
            data, states.dtype, return_torsion=return_torsion,
            return_joint_basis=return_joint_basis,
        )
        routing_diagnostics = None
        if return_attention_conditioning and attention_modulation is not None:
            normalized = self.layers[0].norm1(states)
            baseline_weights, baseline_v = self.layers[0].attention.attention_weights(
                normalized, source, target, relation_bias
            )
            conditioned_weights, conditioned_v = self.layers[0].attention.attention_weights(
                normalized, source, target, relation_bias,
                qk_states=(1.0 + attention_modulation) * normalized,
            )
            routing_diagnostics = {
                "gamma": attention_modulation,
                "baseline_attention": baseline_weights,
                "conditioned_attention": conditioned_weights,
                "baseline_value": baseline_v,
                "conditioned_value": conditioned_v,
                "source": source, "target": target,
                "token_valid": data.glt_token_valid.bool(),
                "token_shift": data.glt_token_shift.long(),
            }
        for index, layer in enumerate(self.layers):
            states = layer(
                states, source, target, relation_bias,
                qk_modulation=attention_modulation if index == 0 else None,
            )
        states = self.final_norm(states)
        atom_states, atom_valid = self._atom_incidence(data, states)
        graph = self._graph_pool(data, atom_states, atom_valid)
        result = {
            "graph_geometry": graph,
            "line_states": states,
            "atom_geometry_states": atom_states,
            "atom_geometry_valid": atom_valid,
            "relation_source": source,
            "relation_target": target,
        }
        if conditioning is not None:
            result["line_conditioning"] = conditioning
        if routing_diagnostics is not None:
            result["attention_conditioning"] = routing_diagnostics
        if return_joint_basis:
            if return_torsion and "torsion" in auxiliary:
                result["torsion"] = auxiliary["torsion"]
            if "joint_basis" in auxiliary:
                result["joint_basis"] = auxiliary["joint_basis"]
        elif return_torsion and auxiliary is not None:
            result["torsion"] = auxiliary
        return result


class GLTMaskedLineHeadV2(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.projection = nn.Linear(hidden_size, NUM_LINE_LABELS)

    def forward(self, line_states):
        return self.projection(line_states)


__all__ = [
    "GLTMaskedLineHeadV2",
    "LearnedGaussianMoments",
    "JointRadialAngularRelationBiasV2",
    "LocalPeriodicGraphLineTransformerV2",
    "MASK_ATOMIC_NUMBER",
    "MASK_SHIFT_CLASS",
    "NUM_LINE_LABELS",
    "PeriodicLineAttentionV2",
    "TorsionRelationBiasV2",
]
