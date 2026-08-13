"""Fixed graph core for the ``MIPS-Trimer-SCAGE`` (MTS) route."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import softmax
from torch_scatter import scatter

from src.dataset.graph_data import MIPS_ATOM_FEATURE_DIM
from src.dataset.mips_trimer_contract import (
    ROUTE_NAME,
    FEATURE_SCHEMA,
    LEGACY_FEATURE_SCHEMA,
    EXPLICIT_FEATURE_SCHEMA,
    EXPLICIT_LGA_SCHEMA_VERSION,
    CANONICAL_LGA_SCHEMA_VERSION,
    TOPOLOGY_CANONICAL,
    TOPOLOGY_EXPLICIT,
)
from .trimer_mcl import TrimerSCAGEMCLResidual


MSTA_LAYER_INDICES = (4, 5)
MSTA_LOCAL_SPD = (0, 1)
MSTA_CONTEXT_SPD = (0, 1, 2)


def topology_attention_identity(variant="o8"):
    """Return the strict, serializable model identity for T0 or T1."""

    variant = str(variant)
    if variant not in {"o8", "msta_last2"}:
        raise ValueError(f"unsupported topology attention variant: {variant!r}")
    return {
        "model_identity": "T1" if variant == "msta_last2" else "T0",
        "topology_attention_variant": variant,
        "msta_layer_indices": list(MSTA_LAYER_INDICES),
        "msta_local_spd": list(MSTA_LOCAL_SPD),
        "msta_context_spd": list(MSTA_CONTEXT_SPD),
        "msta_share_relation_dropout": True,
        "msta_local_output_bias": False,
        "msta_local_output_init": "zero",
    }


def checkpoint_topology_attention_variant(meta):
    """Read a checkpoint's variant while preserving legacy T0 metadata."""

    return str((meta or {}).get("topology_attention_variant", "o8"))


def add_function_preserving_t1_parameters(state_dict, *, prefix=""):
    """Add only the zero local projections needed by a T1 state dict.

    The caller still performs a strict ``load_state_dict`` after this
    explicit conversion.  No generic ``strict=False`` loading is involved.
    """

    converted = dict(state_dict)
    for index in MSTA_LAYER_INDICES:
        output_key = (
            f"{prefix}layers.{index}.attention.output.weight"
            if prefix else f"layers.{index}.attention.output.weight"
        )
        local_key = (
            f"{prefix}layers.{index}.attention.local_output.weight"
            if prefix else f"layers.{index}.attention.local_output.weight"
        )
        if output_key not in converted:
            raise KeyError(
                f"T0 state dict is missing the final-layer output: {output_key}"
            )
        if local_key in converted:
            raise ValueError(f"state dict already contains T1 parameter: {local_key}")
        converted[local_key] = torch.zeros_like(converted[output_key])
    return converted


class MIPSLocalAtomEmbedding(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.projection = nn.Linear(MIPS_ATOM_FEATURE_DIM, int(hidden_dim))
        self.backbone = nn.Embedding(2, int(hidden_dim), padding_idx=0)

    def forward(self, data, atom_mask=None):
        features = data.mips_x.float()
        if atom_mask is not None:
            features = features.masked_fill(atom_mask.unsqueeze(-1), 0.0)
        return (
            self.projection(features)
            + self.backbone(data.mips_backbone_mask.long().clamp(0, 1))
        )


class MIPSSinglePathNodeBias(nn.Module):
    def __init__(self, hidden_dim, num_heads, max_hops):
        super().__init__()
        self.num_heads = int(num_heads)
        self.projections = nn.ModuleList(
            nn.Linear(int(hidden_dim), self.num_heads, bias=False)
            for _ in range(int(max_hops) + 1)
        )

    def forward(self, initial, data):
        path = data.lga_path_index.long()
        mask = data.lga_path_mask.bool()
        output = initial.new_zeros((path.size(0), self.num_heads))
        denominator = mask.sum(dim=1).clamp_min(1).to(initial.dtype)
        for position, projection in enumerate(self.projections):
            if position >= path.size(1):
                break
            selected = mask[:, position] & (path[:, position] >= 0)
            if bool(selected.any()):
                output[selected] += (
                    projection(initial[path[selected, position]])
                    / denominator[selected].unsqueeze(-1)
                )
        return output


class SymmetricStarDistanceBias(nn.Module):
    """Zero-initialized per-head RBF bias for direct virtual Star edges."""

    def __init__(self, num_heads=8, num_rbf=32, lower=0.0, upper=3.0):
        super().__init__()
        centers = torch.linspace(float(lower), float(upper), int(num_rbf))
        self.register_buffer("centers", centers)
        spacing = float(upper - lower) / max(1, int(num_rbf) - 1)
        self.gamma = 0.5 / max(spacing * spacing, 1e-12)
        self.projection = nn.Linear(
            int(num_rbf), int(num_heads), bias=False
        )
        nn.init.zeros_(self.projection.weight)

    def forward(self, data, dtype):
        edge_count = int(data.lga_edge_index.size(1))
        output = data.lga_spd.new_zeros(
            (edge_count, self.projection.out_features), dtype=dtype
        )
        star_edges = data.lga_star_edge_mask.bool()
        if not bool(star_edges.any()):
            return output
        _, target = data.lga_edge_index.long()
        graph_id = data.batch[target[star_edges]].long()
        valid = data.star_3d_valid.bool().flatten()[graph_id]
        if not bool(valid.any()):
            return output + self.projection.weight.reshape(-1)[0] * 0.0
        distance = data.star_3d_distance.float().flatten()[graph_id[valid]]
        # Build the RBF in float32 for stable exponentials, then feed the
        # projection with the parameter dtype.  Under CUDA autocast the
        # Linear may return BF16 while ``output`` can still be float32 (or
        # vice versa), so cast the projected result explicitly before the
        # indexed write below.  Without this, the joint-pretraining BF16
        # parity gate can fail with a Float/BFloat16 mismatch.
        rbf = torch.exp(
            -self.gamma * (distance.unsqueeze(-1) - self.centers.float()) ** 2
        )
        projection_input = rbf.to(dtype=self.projection.weight.dtype)
        selected_edges = torch.nonzero(
            star_edges, as_tuple=False
        ).flatten()[valid]
        projected = self.projection(projection_input)
        output[selected_edges] = projected.to(dtype=output.dtype)
        return output


class MTSRelationGeometryBias(nn.Module):
    """Encode frozen SPD=2 relation/path geometry into per-head bias.

    The final projection is zero initialized.  Thus G1/G2/G3 have exactly the
    G0 forward at their shared step-0, while the encoder and projection remain
    trainable for subsequent updates.
    """

    def __init__(self, dim=512, num_heads=8, mode="g1", num_distance_rbf=16):
        super().__init__()
        mode = str(mode).lower()
        if mode not in {"g0", "g1", "g2", "g3"}:
            raise ValueError("relation geometry bias requires G0/G1/G2/G3")
        self.mode = mode
        self.num_heads = int(num_heads)
        self.num_distance_rbf = int(num_distance_rbf)
        # All arms keep one identical geometry-only parameter layout.  G1 and
        # G0 feed an explicit zero distance channel, so G1 cannot learn from
        # endpoint distance while checkpoints remain shape-compatible.
        input_dim = 1 + self.num_distance_rbf
        hidden = max(32, min(128, int(dim) // 4))
        self.path_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.relation_projection = nn.Linear(hidden, self.num_heads, bias=False)
        nn.init.zeros_(self.relation_projection.weight)
        centers = torch.linspace(0.0, 8.0, self.num_distance_rbf)
        self.register_buffer("distance_centers", centers)
        spacing = 8.0 / max(1, self.num_distance_rbf - 1)
        self.distance_gamma = 0.5 / max(spacing * spacing, 1e-12)

    def _distance_rbf(self, distance):
        delta = distance.unsqueeze(-1) - self.distance_centers.to(distance.device)
        return torch.exp(-self.distance_gamma * delta.square())

    def forward(self, data, *, edge_count, dtype):
        output = data.lga_spd.new_zeros((int(edge_count), self.num_heads), dtype=dtype)
        if self.mode == "g0":
            return output
        if not all(hasattr(data, name) for name in (
            "mts_relation_geometry_relation_row",
            "mts_relation_geometry_valid",
            "mts_relation_geometry_path_offsets",
            "mts_relation_geometry_path_valid",
            "mts_relation_geometry_path_cos_angle",
            "mts_relation_geometry_endpoint_distance",
        )):
            raise ValueError(f"{self.mode.upper()} requires a strict relation-geometry sidecar")
        rows = data.mts_relation_geometry_relation_row.long().reshape(-1)
        relation_valid = data.mts_relation_geometry_valid.bool().reshape(-1)
        path_offsets = data.mts_relation_geometry_path_offsets.long().reshape(-1)
        path_valid = data.mts_relation_geometry_path_valid.bool().reshape(-1)
        cosine = data.mts_relation_geometry_path_cos_angle.float().reshape(-1)
        endpoint = data.mts_relation_geometry_endpoint_distance.float().reshape(-1)
        if path_offsets.numel() != rows.numel() + 1:
            raise ValueError("relation-geometry path offsets are not relation-aligned")
        if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= int(edge_count)):
            raise ValueError("relation-geometry row is outside the MSTA edge table")
        if endpoint.numel() != rows.numel():
            raise ValueError("relation-geometry endpoint distance is not relation-aligned")
        # The original implementation launched one tiny MLP per relation.
        # A PI1M batch contains tens of thousands of relations, so that Python
        # loop turns the sidecar arm into a GPU-kernel launch bottleneck (and
        # can reduce throughput by two orders of magnitude).  Flatten all
        # paths once, run the MLP as one batch, then reduce back to relation
        # rows.  The masks and per-relation denominator are unchanged.
        relation_count = int(rows.numel())
        path_lengths = path_offsets[1:] - path_offsets[:-1]
        if relation_count and cosine.numel():
            relation_ids = torch.repeat_interleave(
                torch.arange(relation_count, device=rows.device, dtype=torch.long),
                path_lengths,
            )
            features = [cosine.unsqueeze(-1)]
            if self.mode in {"g2", "g3"}:
                endpoint_per_path = torch.repeat_interleave(endpoint, path_lengths)
                features.append(self._distance_rbf(endpoint_per_path))
            else:
                features.append(cosine.new_zeros((cosine.numel(), self.num_distance_rbf)))
            path_embedding = self.path_mlp(
                torch.cat(features, dim=-1).to(dtype=self.path_mlp[0].weight.dtype)
            )
            path_mask = path_valid.to(path_embedding.dtype)
            relation_embeddings = scatter(
                path_embedding * path_mask.unsqueeze(-1),
                relation_ids,
                dim=0,
                dim_size=relation_count,
                reduce="sum",
            )
            path_counts = scatter(
                path_mask,
                relation_ids,
                dim=0,
                dim_size=relation_count,
                reduce="sum",
            ).clamp_min(1.0).unsqueeze(-1)
            relation_embeddings = relation_embeddings / path_counts
            valid_relations = relation_valid & (path_counts.squeeze(-1) > 0)
            if bool(valid_relations.any()):
                projected = self.relation_projection(relation_embeddings).to(dtype=dtype)
                output.index_copy_(0, rows[valid_relations], projected[valid_relations])
        return output


class MIPSLocalAttention(nn.Module):
    """K_source·Q_target attention with incoming-edge normalization."""

    def __init__(self, dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.dim, 3 * self.dim)
        self.output = nn.Linear(self.dim, self.dim)
        self.dropout = nn.Dropout(float(dropout))
        self.attention_dropout = nn.Dropout(float(dropout))
        self.output_norm = nn.LayerNorm(self.dim)
        self.debug_attention = os.environ.get("MIPS_DEBUG_ATTENTION", "0") == "1"
        self.last_attention = None

    def forward(self, x, edge_index, attention_bias, **_kwargs):
        qkv = self.qkv(x).view(x.size(0), 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=1)
        source, target = edge_index.long()
        logits = (
            (query[target] * key[source]).sum(dim=-1) * self.scale
            + attention_bias
        )
        weights = softmax(
            logits.float(), target, num_nodes=x.size(0), dim=0
        ).to(value.dtype)
        # Production training keeps the tensor disabled to avoid retaining a
        # large graph per layer.  Evaluation/debug inspection remains
        # available by default, which preserves the lightweight architecture
        # parity tests without affecting training memory.
        self.last_attention = (
            weights.detach() if (self.debug_attention or not self.training)
            else None
        )
        messages = (
            self.attention_dropout(weights).unsqueeze(-1) * value[source]
        )
        aggregated = scatter(
            messages, target, dim=0, dim_size=x.size(0), reduce="sum"
        ).reshape(x.size(0), self.dim)
        return self.output_norm(
            x + self.dropout(self.output(aggregated))
        )


class MSTAMIPSLocalAttention(nn.Module):
    """Final-layer multi-scale topology attention used by the T1 variant.

    The module deliberately keeps the T0 ``qkv`` and ``output`` parameter
    names.  ``output`` consumes the context (Z2) branch and the new
    zero-initialized ``local_output`` consumes the local (Z1) branch.  This
    makes the function-preserving T0 -> T1 conversion explicit while leaving
    the ordinary T0 state-dict contract unchanged.
    """

    def __init__(
        self,
        dim=512,
        num_heads=8,
        dropout=0.1,
        local_spd=(0, 1),
        context_spd=(0, 1, 2),
        share_relation_dropout=True,
        local_output_bias=False,
        local_output_init="zero",
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        if self.dim % self.num_heads:
            raise ValueError("MSTA hidden dimension must divide num_heads")
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.dim, 3 * self.dim)
        self.output = nn.Linear(self.dim, self.dim)
        self.local_output = nn.Linear(
            self.dim, self.dim, bias=bool(local_output_bias)
        )
        if str(local_output_init) != "zero":
            raise ValueError("MSTA local_output_init must be 'zero'")
        nn.init.zeros_(self.local_output.weight)
        if self.local_output.bias is not None:
            nn.init.zeros_(self.local_output.bias)
        self.dropout = nn.Dropout(float(dropout))
        self.attention_dropout = nn.Dropout(float(dropout))
        self.output_norm = nn.LayerNorm(self.dim)
        self.local_spd = tuple(sorted({int(value) for value in local_spd}))
        self.context_spd = tuple(sorted({int(value) for value in context_spd}))
        self.share_relation_dropout = bool(share_relation_dropout)
        if self.local_spd != (0, 1):
            raise ValueError("MSTA local SPD support must be exactly (0, 1)")
        if self.context_spd != (0, 1, 2):
            raise ValueError("MSTA context SPD support must be exactly (0, 1, 2)")
        if not self.share_relation_dropout:
            raise ValueError("MSTA requires shared relation dropout")
        self.debug_attention = os.environ.get("MIPS_DEBUG_ATTENTION", "0") == "1"
        self.last_attention = None
        self.last_attention_branches = None
        # Optional scalar-only branch diagnostics.  This is toggled by the
        # pretraining loop only at declared milestone steps; it never retains
        # the autograd graph or changes the forward values.
        self.diagnostic_capture = False
        self.diagnostic_local_off = False
        self.last_diagnostic = None

    @staticmethod
    def _branch_mask(spd, allowed, target, relation_mask):
        allowed_tensor = torch.zeros_like(spd, dtype=torch.bool)
        for value in allowed:
            allowed_tensor |= spd == int(value)
        if relation_mask is not None:
            allowed_tensor &= ~relation_mask.bool().reshape(-1)
        if allowed_tensor.numel() != target.numel():
            raise ValueError("MSTA relation mask and edge count mismatch")
        # Both branches include SPD=0, but retain an explicit check so a
        # malformed padded batch cannot silently create an all -inf softmax.
        if target.numel():
            counts = scatter(
                allowed_tensor.to(torch.long), target.long(), dim=0,
                dim_size=int(target.max().item()) + 1, reduce="sum",
            )
            if bool((counts == 0).any()):
                raise ValueError(
                    "MSTA branch has a target without a valid self relation"
                )
        return allowed_tensor

    @staticmethod
    def _normalize(logits, target, mask, num_nodes):
        masked = logits.float().masked_fill(~mask.unsqueeze(-1), float("-inf"))
        return softmax(masked, target.long(), num_nodes=num_nodes, dim=0)

    def forward(
        self, x, edge_index, attention_bias, spd, relation_mask=None,
        geometry_bias=None,
    ):
        qkv = self.qkv(x).view(x.size(0), 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=1)
        source, target = edge_index.long()
        logits = (
            (query[target] * key[source]).sum(dim=-1) * self.scale
            + attention_bias
        )
        local_mask = self._branch_mask(
            spd.long().reshape(-1), self.local_spd, target, relation_mask
        )
        context_mask = self._branch_mask(
            spd.long().reshape(-1), self.context_spd, target, relation_mask
        )
        local_weights = self._normalize(
            logits, target, local_mask, x.size(0)
        ).to(value.dtype)
        context_logits = logits
        if geometry_bias is not None:
            if geometry_bias.shape != logits.shape:
                raise ValueError("MSTA geometry bias must match [edge, head]")
            context_logits = logits + geometry_bias
        context_weights = self._normalize(
            context_logits, target, context_mask, x.size(0)
        ).to(value.dtype)
        if self.training and self.attention_dropout.p:
            if self.share_relation_dropout:
                relation_keep = (
                    torch.rand_like(local_weights) >= self.attention_dropout.p
                ).to(local_weights.dtype) / (1.0 - self.attention_dropout.p)
                local_weights = local_weights * relation_keep
                context_weights = context_weights * relation_keep
            else:  # Defensive branch; constructor currently forbids this.
                local_weights = self.attention_dropout(local_weights)
                context_weights = self.attention_dropout(context_weights)
        if self.debug_attention or not self.training:
            self.last_attention = context_weights.detach()
            self.last_attention_branches = {
                "local": local_weights.detach(),
                "context": context_weights.detach(),
                "local_mask": local_mask.detach(),
                "context_mask": context_mask.detach(),
            }
        else:
            self.last_attention = None
            self.last_attention_branches = None
        local_messages = local_weights.unsqueeze(-1) * value[source]
        context_messages = context_weights.unsqueeze(-1) * value[source]
        local_aggregated = scatter(
            local_messages, target, dim=0, dim_size=x.size(0), reduce="sum"
        ).reshape(x.size(0), self.dim)
        context_aggregated = scatter(
            context_messages, target, dim=0, dim_size=x.size(0), reduce="sum"
        ).reshape(x.size(0), self.dim)
        local_projected = self.local_output(local_aggregated)
        context_projected = self.output(context_aggregated)
        projected = context_projected + (
            local_projected if not self.diagnostic_local_off
            else torch.zeros_like(local_projected)
        )
        if self.diagnostic_capture:
            def _summary(values, *, expected_range=None):
                """Summarize diagnostics without hiding invalid values.

                The old implementation filtered NaNs before deciding whether a
                row was finite.  That made an entirely invalid entropy vector
                look like a valid row with ``count=0``.  Keep the finite and
                range checks alongside the filtered statistics so the caller
                can distinguish an empty/incomplete diagnostic from a finite
                one.
                """

                raw = values.detach().float().reshape(-1)
                finite_mask = torch.isfinite(raw)
                finite_values = raw[finite_mask]
                finite = bool(finite_mask.all())
                range_valid = None
                if expected_range is not None:
                    lower, upper = expected_range
                    range_valid = bool(
                        finite_values.numel()
                        and bool((finite_values >= lower - 1e-6).all())
                        and bool((finite_values <= upper + 1e-6).all())
                    )
                summary = {
                    "mean": None,
                    "median": None,
                    "p10": None,
                    "p90": None,
                    "count": int(finite_values.numel()),
                    "finite": finite,
                }
                if expected_range is not None:
                    summary["range_valid"] = range_valid
                if finite_values.numel():
                    summary.update({
                        "mean": float(finite_values.mean().item()),
                        "median": float(finite_values.median().item()),
                        "p10": float(torch.quantile(finite_values, 0.10).item()),
                        "p90": float(torch.quantile(finite_values, 0.90).item()),
                    })
                return summary

            local_norm = local_aggregated.float().norm(dim=-1)
            context_norm = context_aggregated.float().norm(dim=-1)
            local_projected_norm = local_projected.float().norm(dim=-1)
            context_projected_norm = context_projected.float().norm(dim=-1)
            eps = torch.finfo(torch.float32).eps
            raw_ratio = local_norm / context_norm.clamp_min(eps)
            effective_ratio = local_projected_norm / context_projected_norm.clamp_min(eps)
            raw_cos = F.cosine_similarity(
                local_aggregated.float(), context_aggregated.float(), dim=-1
            )
            effective_cos = F.cosine_similarity(
                local_projected.float(), context_projected.float(), dim=-1
            )

            def _entropy(weights, mask):
                """Return per-target/per-head ``H(p) / log(n)``.

                Masked relation rows must never participate in ``p*log(p)``.
                Multiplying a zero probability by ``log(0)`` creates NaNs even
                though the row is meant to be excluded.  We also keep heads
                separate: summing entropy over heads changes the scale and can
                produce values larger than one.
                """

                valid = mask.bool().unsqueeze(-1)
                raw = weights.float()
                # Relation dropout can make a whole target/head empty.  For
                # the diagnostic, renormalize the surviving rows so entropy is
                # always computed on a probability distribution.
                mass = scatter(
                    torch.where(valid, raw, torch.zeros_like(raw)),
                    target.long(), dim=0, dim_size=x.size(0), reduce="sum",
                )
                denominator = mass[target.long()].clamp_min(eps)
                probabilities = torch.where(
                    valid,
                    raw / denominator,
                    torch.zeros_like(raw),
                )
                safe_probabilities = torch.where(
                    valid,
                    probabilities.clamp_min(eps),
                    torch.ones_like(probabilities),
                )
                target_count = scatter(
                    mask.to(torch.long), target.long(), dim=0,
                    dim_size=x.size(0), reduce="sum",
                ).float()
                entropy_terms = torch.where(
                    valid,
                    -(safe_probabilities * safe_probabilities.log()),
                    torch.zeros_like(safe_probabilities),
                )
                entropy = scatter(
                    entropy_terms, target.long(), dim=0,
                    dim_size=x.size(0), reduce="sum",
                )
                normalized = torch.where(
                    target_count.unsqueeze(-1) > 1,
                    entropy / target_count.unsqueeze(-1).clamp_min(2).log(),
                    torch.zeros_like(entropy),
                )
                return normalized, target_count

            local_entropy, local_degree = _entropy(local_weights, local_mask)
            context_entropy, context_degree = _entropy(context_weights, context_mask)
            self.last_diagnostic = {
                "rho_raw": _summary(raw_ratio),
                "rho_effective": _summary(effective_ratio),
                "cosine_raw": _summary(raw_cos),
                "cosine_effective": _summary(effective_cos),
                "local_attention_entropy": _summary(
                    local_entropy, expected_range=(0.0, 1.0)
                ),
                "context_attention_entropy": _summary(
                    context_entropy, expected_range=(0.0, 1.0)
                ),
                "local_effective_incoming_degree": _summary(local_degree),
                "context_effective_incoming_degree": _summary(context_degree),
                "finite": bool(
                    torch.isfinite(local_aggregated).all()
                    and torch.isfinite(context_aggregated).all()
                    and torch.isfinite(local_projected).all()
                    and torch.isfinite(context_projected).all()
                ),
            }
            # Do this after constructing the summaries so non-finite branch
            # statistics cannot be hidden by the tensor-only check above.
            required_summaries = (
                self.last_diagnostic["local_attention_entropy"],
                self.last_diagnostic["context_attention_entropy"],
            )
            self.last_diagnostic["finite"] = bool(
                self.last_diagnostic["finite"]
                and all(bool(item.get("finite")) for item in required_summaries)
                and all(bool(item.get("range_valid")) for item in required_summaries)
                and all(int(item.get("count", 0)) > 0 for item in required_summaries)
            )
        else:
            self.last_diagnostic = None
        return self.output_norm(x + self.dropout(projected))


class MIPSLocalLayer(nn.Module):
    def __init__(self, dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = MIPSLocalAttention(dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(int(dim), 4 * int(dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(4 * int(dim), int(dim)),
        )
        self.dropout = nn.Dropout(float(dropout))
        self.output_norm = nn.LayerNorm(int(dim))

    def forward(self, x, edge_index, attention_bias):
        x = self.attention(x, edge_index, attention_bias)
        return self.output_norm(x + self.dropout(self.ffn(x)))


class MSTAMIPSLocalLayer(nn.Module):
    """T1 replacement for one O8 layer with two independent SPD branches."""

    def __init__(
        self,
        dim=512,
        num_heads=8,
        dropout=0.1,
        local_spd=(0, 1),
        context_spd=(0, 1, 2),
        share_relation_dropout=True,
        local_output_bias=False,
        local_output_init="zero",
    ):
        super().__init__()
        self.attention = MSTAMIPSLocalAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            local_spd=local_spd,
            context_spd=context_spd,
            share_relation_dropout=share_relation_dropout,
            local_output_bias=local_output_bias,
            local_output_init=local_output_init,
        )
        self.ffn = nn.Sequential(
            nn.Linear(int(dim), 4 * int(dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(4 * int(dim), int(dim)),
        )
        self.dropout = nn.Dropout(float(dropout))
        self.output_norm = nn.LayerNorm(int(dim))

    def forward(
        self, x, edge_index, attention_bias, spd, relation_mask=None,
        geometry_bias=None,
    ):
        x = self.attention(
            x, edge_index, attention_bias, spd=spd,
            relation_mask=relation_mask, geometry_bias=geometry_bias,
        )
        return self.output_norm(x + self.dropout(self.ffn(x)))


class MD200GraphResidual(nn.Module):
    """Low-capacity graph-only residual; invalid MD is an exact zero."""

    def __init__(self, dim=512, dropout=0.10):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(200),
            nn.Linear(200, 64),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(64, int(dim), bias=False),
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, graph, data):
        valid = data.mips_md_valid.bool().flatten()
        residual = graph.new_zeros(graph.shape)
        if bool(valid.any()):
            # ``mips_md`` stays in float32 for stable LayerNorm/input
            # handling, while CUDA autocast may produce a BF16 adapter
            # output.  Indexed assignment does not promote the RHS, so make
            # the conversion explicit before writing into the graph-typed
            # residual buffer.
            adapter_output = self.adapter(data.mips_md.float()[valid])
            residual[valid] = adapter_output.to(dtype=residual.dtype)
        else:
            residual = residual + self.gate * 0.0
        gate = torch.tanh(self.gate).to(dtype=graph.dtype)
        return graph + residual * gate


class MIPSLocalGraphEncoder(nn.Module):
    """O8/MSTA topology + symmetric Star RBF + Trimer MCL + MD200."""

    expects_data = True
    uses_geometry = True
    architecture_name = ROUTE_NAME

    def __init__(
        self,
        core="paper_corrected",
        num_layer=6,
        emb_dim=512,
        num_heads=8,
        dropout=0.1,
        max_hops=2,
        feature_mode="mips137",
        use_descriptors=True,
        spatial_mode="trimer_scage",
        graph_geometry_mode="trimer_scage_mcl",
        mcl_distance_percentiles=(0.20, 0.50),
        trimer_num_candidates=4,
        trimer_max_heavy_atoms=384,
        variant="O8",
        atom_feature_mode="mips137",
        attention_scale="head_dim",
        norm_mode="post",
        activation="relu",
        spd_bias_mode="per_head",
        path_bias_mode="per_head_single_path_node",
        multi_scale_hop_gate=False,
        semantics="paper_semantic",
        descriptor_fusion_mode="graph_md_residual",
        descriptor_components="md200",
        descriptor_disturbance=0.0,
        mask_mode="zero",
        backbone_mode="independent",
        mask_policy="canonical_exact",
        masked_loss_reduction="atom_mean",
        input_norm=False,
        qk_direction="paper",
        use_star_rbf=True,
        use_mcl=True,
        mcl_mask_mode="real",
        # Low-level constructor keeps the historical T0 default for explicit
        # Python callers; new training entry points resolve T1 in config and
        # UniEncoder defaults.
        topology_attention_variant="o8",
        msta_layer_indices=(4, 5),
        msta_local_spd=(0, 1),
        msta_context_spd=(0, 1, 2),
        msta_share_relation_dropout=True,
        msta_local_output_bias=False,
        msta_local_output_init="zero",
        g_family_arm=None,
        relation_geometry_sidecar=None,
        g3_permutation_sidecar=None,
        **retired,
    ):
        super().__init__()
        geometry_aliases = {
            "trimer_scage_mcl": "current_mcl",
            "current_mcl": "current_mcl",
            "mcl_rbf": "mcl_rbf",
            "disabled": "disabled",
            "coordinate_shuffled": "coordinate_shuffled",
            "mcl_rbf_coordinate_shuffled": "mcl_rbf_coordinate_shuffled",
            "g0": "g0",
            "g1": "g1",
            "g2": "g2",
            "g3": "g3",
        }
        requested_geometry = str(graph_geometry_mode)
        if requested_geometry not in geometry_aliases:
            raise ValueError(f"unsupported MTS geometry mode: {requested_geometry}")
        resolved_geometry = geometry_aliases[requested_geometry]
        resolved_g_arm = str(g_family_arm).lower() if g_family_arm is not None else None
        if resolved_g_arm is None and requested_geometry in {"g0", "g1", "g2", "g3"}:
            resolved_g_arm = requested_geometry
        if resolved_g_arm is not None and resolved_g_arm not in {"g0", "g1", "g2", "g3"}:
            raise ValueError("g_family_arm must be one of g0/g1/g2/g3")
        if resolved_g_arm is not None and requested_geometry not in {resolved_g_arm, "trimer_scage_mcl", "current_mcl"}:
            raise ValueError("g_family_arm and graph_geometry_mode are inconsistent")
        if resolved_g_arm is not None:
            requested_geometry = resolved_g_arm
            resolved_geometry = resolved_g_arm
        contract = {
            "core": (str(core), "paper_corrected"),
            "num_layer": (int(num_layer), 6),
            "emb_dim": (int(emb_dim), 512),
            "num_heads": (int(num_heads), 8),
            "max_hops": (int(max_hops), 2),
            "feature_mode": (str(atom_feature_mode or feature_mode), "mips137"),
            "spatial_mode": (str(spatial_mode), "trimer_scage"),
            "variant": (str(variant), "O8"),
            "attention_scale": (str(attention_scale), "head_dim"),
            "norm_mode": (str(norm_mode), "post"),
            "activation": (str(activation), "relu"),
            "spd_bias_mode": (str(spd_bias_mode), "per_head"),
            "path_bias_mode": (
                str(path_bias_mode), "per_head_single_path_node"
            ),
            "descriptor_fusion": (
                str(descriptor_fusion_mode), "graph_md_residual"
            ),
            "descriptor_components": (
                str(descriptor_components), "md200"
            ),
        }
        mismatches = [
            f"{name}={actual!r} (required {expected!r})"
            for name, (actual, expected) in contract.items()
            if actual != expected
        ]
        if retired:
            mismatches.append("retired fields=" + ",".join(sorted(retired)))
        if mismatches:
            raise ValueError(
                f"{ROUTE_NAME} contract violation: "
                + "; ".join(mismatches)
            )
        if not bool(use_descriptors):
            raise ValueError(f"{ROUTE_NAME} requires graph-level MD200")
        if bool(multi_scale_hop_gate) or bool(input_norm):
            raise ValueError("retired O8 topology options are not supported")
        if abs(float(dropout) - 0.10) > 1e-12:
            raise ValueError("dropout must be 0.10")
        if tuple(float(x) for x in mcl_distance_percentiles) != (0.20, 0.50):
            raise ValueError("MCL percentiles must be 0.20/0.50")
        if int(trimer_num_candidates) != 4 or int(trimer_max_heavy_atoms) != 384:
            raise ValueError("Trimer contract requires 4 candidates/384 atoms")
        topology_attention_variant = str(topology_attention_variant)
        if topology_attention_variant not in {"o8", "msta_last2"}:
            raise ValueError(
                "topology_attention_variant must be 'o8' or 'msta_last2'"
            )
        normalized_layers = tuple(int(value) for value in msta_layer_indices)
        normalized_local_spd = tuple(sorted({int(value) for value in msta_local_spd}))
        normalized_context_spd = tuple(sorted({int(value) for value in msta_context_spd}))
        if topology_attention_variant == "msta_last2" and normalized_layers != (4, 5):
            raise ValueError("MSTA currently supports only layer indices (4, 5)")
        if topology_attention_variant == "o8" and normalized_layers != (4, 5):
            raise ValueError("T0 msta_layer_indices must remain the default (4, 5)")
        if normalized_local_spd != (0, 1):
            raise ValueError("MSTA local SPD support must be exactly (0, 1)")
        if normalized_context_spd != (0, 1, 2):
            raise ValueError("MSTA context SPD support must be exactly (0, 1, 2)")
        if not bool(msta_share_relation_dropout):
            raise ValueError("MSTA requires shared relation dropout")
        if bool(msta_local_output_bias):
            raise ValueError("MSTA local_output must be bias-free")
        if str(msta_local_output_init) != "zero":
            raise ValueError("MSTA local_output_init must be 'zero'")

        self.core = "paper_corrected"
        self.variant = "O8"
        self.emb_dim = 512
        self.num_heads = 8
        self.max_hops = 2
        self.feature_mode = "mips137"
        self.spatial_mode = "trimer_scage"
        self.graph_geometry_mode = requested_geometry
        self.geometry_mode = resolved_geometry
        self.g_family_arm = resolved_g_arm
        self.relation_geometry_sidecar = relation_geometry_sidecar
        self.g3_permutation_sidecar = g3_permutation_sidecar
        self.topology_attention_variant = topology_attention_variant
        self.model_identity = (
            "T1" if topology_attention_variant == "msta_last2" else "T0"
        )
        self.msta_layer_indices = normalized_layers
        self.msta_local_spd = normalized_local_spd
        self.msta_context_spd = normalized_context_spd
        self.msta_share_relation_dropout = bool(msta_share_relation_dropout)
        self.msta_local_output_bias = bool(msta_local_output_bias)
        self.msta_local_output_init = str(msta_local_output_init)
        if self.g_family_arm is not None:
            self.relation_geometry_bias = MTSRelationGeometryBias(
                self.emb_dim, self.num_heads, mode=self.g_family_arm
            )
        else:
            self.relation_geometry_bias = None
        self.use_descriptors = True
        self.descriptor_components = "md200"
        # Orthogonal causal-ablation axes (Plan mts_geometry_injection_ablation
        # A0-A4): star-RBF and MCL can be independently bypassed in forward and
        # excluded from the optimizer, while their parameters stay in the module
        # so a shared pretrained checkpoint still loads with strict=True.
        self.use_star_rbf = False if self.g_family_arm is not None else bool(use_star_rbf)
        self.use_mcl = False if self.g_family_arm is not None else bool(use_mcl)
        self.mcl_mask_mode = str(mcl_mask_mode)
        if self.mcl_mask_mode not in ("real", "count_matched_random"):
            raise ValueError(
                f"unsupported MCL mask mode: {self.mcl_mask_mode!r}"
            )
        if self.mcl_mask_mode == "count_matched_random" and not self.use_mcl:
            raise ValueError("random MCL masking requires use_mcl=True")
        if str(mask_policy) != "canonical_exact":
            raise ValueError(
                "all canonical-equivalent O8 copies must be masked together"
            )
        self.mask_policy = "canonical_exact"
        self.masked_atom_target = "mips101"
        self.masked_atom_classes = MIPS_ATOM_FEATURE_DIM - 36
        self.masked_loss_reduction = "atom_mean"

        self.atom_embedding = MIPSLocalAtomEmbedding(self.emb_dim)
        self.spd_embedding = nn.Embedding(3, self.num_heads)
        nn.init.zeros_(self.spd_embedding.weight)
        self.path_bias = MIPSSinglePathNodeBias(
            self.emb_dim, self.num_heads, self.max_hops
        )
        self.star_distance_bias = SymmetricStarDistanceBias(self.num_heads)
        layers = []
        for index in range(6):
            if (
                self.topology_attention_variant == "msta_last2"
                and index in self.msta_layer_indices
            ):
                layers.append(
                    MSTAMIPSLocalLayer(
                        self.emb_dim,
                        self.num_heads,
                        dropout,
                        local_spd=self.msta_local_spd,
                        context_spd=self.msta_context_spd,
                        share_relation_dropout=self.msta_share_relation_dropout,
                        local_output_bias=self.msta_local_output_bias,
                        local_output_init=self.msta_local_output_init,
                    )
                )
            else:
                layers.append(MIPSLocalLayer(self.emb_dim, self.num_heads, dropout))
        self.layers = nn.ModuleList(layers)
        self.final_norm = nn.Identity()
        self.trimer_mcl = TrimerSCAGEMCLResidual(
            dim=self.emb_dim, num_heads=self.num_heads,
            percentiles=(0.20, 0.50), dropout=dropout,
            use_distance_bias=("mcl_rbf" in resolved_geometry),
            coordinate_shuffle=("coordinate_shuffled" in resolved_geometry),
            mask_mode=self.mcl_mask_mode,
        )
        self.md_residual = MD200GraphResidual(self.emb_dim, dropout)
        # Bypassed branches stay visible to strict state-dict loading but are
        # frozen; the downstream optimizer only collects requires_grad params.
        if not self.use_star_rbf:
            for _parameter in self.star_distance_bias.parameters():
                _parameter.requires_grad_(False)
        if not self.use_mcl:
            for _parameter in self.trimer_mcl.parameters():
                _parameter.requires_grad_(False)
        if self.g_family_arm == "g0" and self.relation_geometry_bias is not None:
            for _parameter in self.relation_geometry_bias.parameters():
                _parameter.requires_grad_(False)

    def _validate(self, data, *, require_geometry, require_md):
        if getattr(data, "feature_schema", None) == LEGACY_FEATURE_SCHEMA:
            raise ValueError(
                "legacy explicit MTS topology is rejected by the production "
                "encoder; use the canonical periodic topology"
            )
        canonical_periodic = bool(
            getattr(data, "mts_canonical_periodic", False)
            or getattr(data, "mips_local_lga_schema_version", 0)
            == CANONICAL_LGA_SCHEMA_VERSION
        )
        representation = str(getattr(
            data, "topology_representation",
            TOPOLOGY_CANONICAL if canonical_periodic else "",
        ))
        if representation not in {TOPOLOGY_CANONICAL, TOPOLOGY_EXPLICIT}:
            raise ValueError("MTS batch has an unknown topology representation")
        expected_feature = (
            FEATURE_SCHEMA
            if representation == TOPOLOGY_CANONICAL else EXPLICIT_FEATURE_SCHEMA
        )
        if getattr(data, "feature_schema", None) != expected_feature:
            raise ValueError("MTS topology representation/feature schema mismatch")
        expected_lga = (
            CANONICAL_LGA_SCHEMA_VERSION
            if representation == TOPOLOGY_CANONICAL else EXPLICIT_LGA_SCHEMA_VERSION
        )
        if int(getattr(data, "mips_local_lga_schema_version", -1)) != expected_lga:
            raise ValueError("MTS topology representation/LGA schema mismatch")
        required = (
            "mips_x", "mips_backbone_mask", "lga_edge_index", "lga_spd",
            "lga_path_index", "lga_path_mask", "lga_star_edge_mask",
            "graph_available", "batch",
        )
        if canonical_periodic:
            required += (
                "lga_source_image_shift", "lga_path_shift",
                "polymer_link_mask",
            )
        else:
            required += (
                "canonical_ru_atom_index", "ru_copy_index",
                "canonical_pair_index",
            )
        if require_geometry:
            if self.use_star_rbf:
                required += (
                    "star_3d_distance", "star_3d_asymmetry", "star_3d_valid",
                )
            if self.use_mcl:
                required += (
                    "trimer_pos",
                    "trimer_central_ru_mask", "mips_to_trimer_central_index",
                    "trimer_geometry_valid", "trimer_geometry_is_3d",
                    "trimer_2d_fallback", "trimer_batch",
                )
        if require_geometry and self.g_family_arm in {"g1", "g2", "g3"}:
            required += (
                "mts_relation_geometry_relation_row",
                "mts_relation_geometry_valid",
                "mts_relation_geometry_path_offsets",
                "mts_relation_geometry_path_valid",
                "mts_relation_geometry_path_cos_angle",
                "mts_relation_geometry_endpoint_distance",
            )
        if require_md:
            required += ("mips_md", "mips_md_valid")
        missing = [name for name in required if not hasattr(data, name)]
        if require_geometry and self.use_mcl and not (
            hasattr(data, "trimer_base_ru_atom_index")
            or hasattr(data, "trimer_base_ru_atom_id")
        ):
            missing.append("trimer_base_ru_atom_index/trimer_base_ru_atom_id")
        if missing:
            raise ValueError(
                f"{ROUTE_NAME} cache is missing " + ", ".join(missing)
            )
        if data.lga_spd.numel() and (
            int(data.lga_spd.min()) < 0 or int(data.lga_spd.max()) > 2
        ):
            raise ValueError("O8 only permits 0/1/2-hop attention")
        if data.lga_star_edge_mask.numel() != data.lga_edge_index.size(1):
            raise ValueError("Star-edge mask length mismatch")
        if canonical_periodic:
            if int(data.mips_x.size(0)) != int(data.mips_backbone_mask.numel()):
                raise ValueError("canonical MIPS137/backbone length mismatch")
            if int(data.lga_source_image_shift.numel()) != int(
                data.lga_edge_index.size(1)
            ):
                raise ValueError("canonical relation shift length mismatch")
            if not torch.equal(
                data.lga_star_edge_mask.bool(), data.polymer_link_mask.bool()
            ):
                raise ValueError(
                    "canonical topology requires one symmetric polymer-link mask"
                )
        available = data.graph_available.bool().flatten()
        boundary = torch.as_tensor(
            data.mips_boundary_distance, device=available.device
        ).flatten()
        condition = torch.as_tensor(
            data.mips_condition_valid, device=available.device
        ).bool().flatten()
        if not canonical_periodic and bool((available & ((boundary <= 5) | ~condition)).any()):
            raise ValueError("available O8 graph violates boundary distance >5")

    @staticmethod
    def _canonical_pool(nodes, data):
        """Pool node copies to canonical atoms, then canonical atoms to graphs."""

        canonical_index = data.canonical_ru_atom_index.long()
        canonical_graph = data.canonical_graph_index.long()
        canonical_count = int(canonical_graph.numel())
        if canonical_index.numel() != nodes.size(0):
            raise ValueError("canonical node identity length mismatch")
        canonical_nodes = scatter(
            nodes, canonical_index, dim=0,
            dim_size=canonical_count, reduce="mean",
        )
        graph_count = int(data.graph_available.numel())
        graph = scatter(
            canonical_nodes, canonical_graph, dim=0,
            dim_size=graph_count, reduce="mean",
        )
        return graph, canonical_nodes

    def _forward_impl(
        self, data, atom_mask=None, *,
        use_star=True, use_geometry=True, use_md=True,
    ):
        g_geometry = self.g_family_arm in {"g1", "g2", "g3"}
        self._validate(
            data, require_geometry=(use_star or use_geometry or g_geometry),
            require_md=use_md,
        )
        initial = self.atom_embedding(data, atom_mask=atom_mask)
        spd_bias = self.spd_embedding(data.lga_spd.long())
        path_bias = self.path_bias(initial, data)
        star_bias = (
            self.star_distance_bias(data, initial.dtype)
            if use_star else torch.zeros_like(spd_bias)
        )
        if "coordinate_shuffled" in self.geometry_mode:
            # The negative control must destroy every learned coordinate
            # channel.  Cached d_star cannot be consistently atom-permuted
            # because the sidecar intentionally stores only its symmetric
            # scalar, so disable that bias while MCL receives permuted
            # coordinate identities.
            star_bias = torch.zeros_like(spd_bias) + (
                self.star_distance_bias.projection.weight.reshape(-1)[0] * 0.0
            )
        relation_mask = getattr(data, "lga_relation_mask", None)
        if relation_mask is not None:
            keep = (~relation_mask.bool()).unsqueeze(-1).to(initial.dtype)
            spd_bias = spd_bias * keep
            path_bias = path_bias * keep
            star_bias = star_bias * keep
        attention_bias = spd_bias + path_bias + star_bias
        geometry_bias = (
            self.relation_geometry_bias(
                data,
                edge_count=int(data.lga_edge_index.size(1)),
                dtype=initial.dtype,
            )
            if g_geometry else None
        )

        x = initial
        relation_mask = getattr(data, "lga_relation_mask", None)
        for layer in self.layers:
            if isinstance(layer, MSTAMIPSLocalLayer):
                x = layer(
                    x, data.lga_edge_index.long(), attention_bias,
                    spd=data.lga_spd, relation_mask=relation_mask,
                    geometry_bias=geometry_bias,
                )
            else:
                x = layer(x, data.lga_edge_index.long(), attention_bias)
        if use_geometry and self.geometry_mode != "disabled":
            x = x + self.trimer_mcl(x, data)

        graph_available = data.graph_available.bool().flatten()
        x = x * graph_available[data.batch.long()].unsqueeze(-1).to(x.dtype)
        graph, _ = self._canonical_pool(x, data)
        graph = graph * graph_available.unsqueeze(-1).to(graph.dtype)
        if use_md:
            graph = self.md_residual(graph, data)
            graph = graph * graph_available.unsqueeze(-1).to(graph.dtype)
        return graph, x

    @staticmethod
    def _zero_parameter_anchor(reference, *modules):
        """Keep intentionally disabled branches visible to DDP.

        Stage 1 is topology-only and Stage 2 disables MD200.  The fixed
        three-rank run uses ``find_unused_parameters=False``; attaching a
        numerical-zero anchor for every parameter therefore avoids reducer
        hangs without changing a single forward value or gradient.
        """
        anchor = reference.new_zeros(())
        for module in modules:
            for parameter in module.parameters():
                if parameter.requires_grad:
                    anchor = anchor + parameter.reshape(-1).sum() * 0.0
        return anchor

    def forward(self, data):
        return self._forward_impl(
            data,
            use_star=self.use_star_rbf,
            use_geometry=self.use_mcl,
        )

    def forward_joint_pretrain(self, data, canonical_atom_mask):
        """One O8+Star-RBF+Trimer-MCL forward for both pretext tasks.

        The angle head consumes the raw final Trimer memory returned by MCL;
        the masked-atom head consumes the copy-broadcast graph node states.
        MD200 is intentionally absent from this path.
        """
        if canonical_atom_mask is None:
            raise ValueError("joint pretraining requires canonical_atom_mask")
        canonical_atom_mask = torch.as_tensor(
            canonical_atom_mask, dtype=torch.bool, device=data.mips_x.device
        ).reshape(-1)
        if canonical_atom_mask.numel() != data.mips_x.size(0):
            raise ValueError("canonical_atom_mask length mismatch")
        g_geometry = self.g_family_arm in {"g1", "g2", "g3"}
        self._validate(data, require_geometry=(self.use_star_rbf or self.use_mcl or g_geometry), require_md=False)
        initial = self.atom_embedding(data, atom_mask=canonical_atom_mask)
        spd_bias = self.spd_embedding(data.lga_spd.long())
        path_bias = self.path_bias(initial, data)
        star_bias = (
            self.star_distance_bias(data, initial.dtype)
            if self.use_star_rbf else torch.zeros_like(spd_bias)
        )
        relation_mask = getattr(data, "lga_relation_mask", None)
        if relation_mask is not None:
            keep = (~relation_mask.bool()).unsqueeze(-1).to(initial.dtype)
            spd_bias = spd_bias * keep
            path_bias = path_bias * keep
            star_bias = star_bias * keep
        attention_bias = spd_bias + path_bias + star_bias
        geometry_bias = (
            self.relation_geometry_bias(
                data,
                edge_count=int(data.lga_edge_index.size(1)),
                dtype=initial.dtype,
            )
            if g_geometry else None
        )
        topology_nodes = initial
        for layer in self.layers:
            if isinstance(layer, MSTAMIPSLocalLayer):
                topology_nodes = layer(
                    topology_nodes, data.lga_edge_index.long(), attention_bias,
                    spd=data.lga_spd, relation_mask=relation_mask,
                    geometry_bias=geometry_bias,
                )
            else:
                topology_nodes = layer(
                    topology_nodes, data.lga_edge_index.long(), attention_bias
                )
        if self.use_mcl:
            geometry_delta, trimer_states, mcl_valid = (
                self.trimer_mcl.forward_with_aux(topology_nodes, data)
            )
        else:
            geometry_delta = torch.zeros_like(topology_nodes)
            trimer_states = topology_nodes.new_zeros((0, topology_nodes.size(-1)))
            mcl_valid = torch.zeros(
                (int(data.graph_available.numel()),), dtype=torch.bool,
                device=topology_nodes.device,
            )
        nodes = topology_nodes + geometry_delta
        graph_available = data.graph_available.bool().flatten()
        nodes = nodes * graph_available[data.batch.long()].unsqueeze(-1).to(nodes.dtype)
        graph, canonical_nodes = self._canonical_pool(nodes, data)
        graph = graph * graph_available.unsqueeze(-1).to(graph.dtype)
        return graph, nodes, {
            "final_trimer_states": trimer_states,
            "canonical_node_states": canonical_nodes,
            "mcl_valid_graph_mask": mcl_valid,
            "angle_valid_graph_mask": getattr(
                data, "trimer_angle_valid", torch.zeros_like(mcl_valid)
            ).bool().flatten(),
        }

    def _atom_mask_from_override(self, data, x_override):
        if x_override.shape != data.x.shape:
            raise ValueError("masked x override must match data.x")
        return (
            (x_override.abs().sum(dim=-1) == 0)
            & (data.x.abs().sum(dim=-1) != 0)
        )

    def forward_with_x(self, data, x_override):
        """Stage 1: topology only; never read Trimer or MD fields."""
        graph, nodes = self._forward_impl(
            data, atom_mask=self._atom_mask_from_override(data, x_override),
            use_star=False, use_geometry=False, use_md=False,
        )
        anchor = self._zero_parameter_anchor(
            nodes, self.star_distance_bias, self.trimer_mcl, self.md_residual
        )
        nodes = nodes + anchor
        graph = graph + anchor
        return graph, nodes

    def forward_geometry_with_x(self, data, x_override):
        """Stage 2: Star/MCL active, MD disabled."""
        graph, nodes = self._forward_impl(
            data, atom_mask=self._atom_mask_from_override(data, x_override),
            use_star=True, use_geometry=True, use_md=False,
        )
        anchor = self._zero_parameter_anchor(nodes, self.md_residual)
        nodes = nodes + anchor
        graph = graph + anchor
        return graph, nodes

    def forward_relation_pretext(self, data):
        graph, nodes = self._forward_impl(
            data, use_star=False, use_geometry=False, use_md=False
        )
        anchor = self._zero_parameter_anchor(
            nodes, self.star_distance_bias, self.trimer_mcl, self.md_residual
        )
        return graph + anchor, nodes + anchor
