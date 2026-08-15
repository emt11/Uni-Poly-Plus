"""Fixed graph core for the ``MIPS-Trimer-SCAGE`` (MTS) route."""

import os
import torch
import torch.nn as nn
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
    """Zero-initialized per-head RBF bias for periodic Trimer relations."""

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

    def forward_periodic_relation_v2(
        self, data, dtype, pair_observation_distances=None,
        pair_observation_mask=None,
    ):
        """Encode unique periodic pairs, then gather one shared value per inverse relation."""
        edge_count = int(data.lga_edge_index.size(1))
        output = data.lga_spd.new_zeros(
            (edge_count, self.projection.out_features), dtype=dtype
        )
        rows = data.mts_star_v2_relation_row.long().reshape(-1)
        pair_index = data.mts_star_v2_relation_pair_index.long().reshape(-1)
        distances = (
            data.mts_star_v2_pair_observation_distances
            if pair_observation_distances is None
            else pair_observation_distances
        ).float().reshape(-1, 2)
        if pair_observation_mask is None:
            counts = data.mts_star_v2_pair_observation_count.long().reshape(-1)
        else:
            mask = torch.as_tensor(
                pair_observation_mask, device=distances.device, dtype=torch.bool
            ).reshape(-1, 2)
            counts = mask.sum(dim=1).long()
        valid = data.mts_star_v2_pair_valid.bool().reshape(-1)
        sources = data.mts_star_v2_pair_geometry_source.long().reshape(-1)
        pair_count = int(valid.numel())
        if distances.size(0) != pair_count or counts.numel() != pair_count:
            raise ValueError("Star-RBF v2 pair tensor length mismatch")
        if rows.numel() != pair_index.numel() or (
            rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= edge_count)
        ):
            raise ValueError("Star-RBF v2 relation mapping mismatch")
        if pair_index.numel() and (
            int(pair_index.min()) < 0 or int(pair_index.max()) >= pair_count
        ):
            raise ValueError("Star-RBF v2 pair mapping out of bounds")
        # Source code 1 is the audited true self relation: it deliberately
        # stays zero instead of encoding RBF(0).
        encode = valid & (sources != 1) & (counts > 0)
        pair_bias = output.new_zeros((pair_count, self.projection.out_features))
        if bool(encode.any()):
            selected = distances[encode]
            rbf = torch.exp(
                -self.gamma
                * (selected.unsqueeze(-1) - self.centers.float().reshape(1, 1, -1)) ** 2
            )
            if pair_observation_mask is None:
                observation_mask = (
                    torch.arange(2, device=counts.device).reshape(1, 2)
                    < counts[encode].reshape(-1, 1)
                ).unsqueeze(-1)
            else:
                observation_mask = torch.as_tensor(
                    pair_observation_mask, device=distances.device,
                    dtype=torch.bool,
                ).reshape(-1, 2)[encode].unsqueeze(-1)
            rbf = (rbf * observation_mask).sum(dim=1) / counts[encode].clamp_min(1).unsqueeze(-1)
            projected = self.projection(rbf.to(self.projection.weight.dtype))
            pair_bias[encode] = projected.to(pair_bias.dtype)
        if rows.numel():
            output[rows] = pair_bias[pair_index]
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
    """Final-layer multi-scale topology attention over local/context SPD."""

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
        projected = context_projected + local_projected
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
    """Multi-scale replacement for one O8 layer with two SPD branches."""

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
    ):
        x = self.attention(
            x, edge_index, attention_bias, spd=spd,
            relation_mask=relation_mask,
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
        star_rbf_upper=3.0,
        use_mcl=True,
        topology_attention_variant="o8",
        msta_layer_indices=(4, 5),
        msta_local_spd=(0, 1),
        msta_context_spd=(0, 1, 2),
        msta_share_relation_dropout=True,
        msta_local_output_bias=False,
        msta_local_output_init="zero",
        **retired,
    ):
        super().__init__()
        requested_geometry = str(graph_geometry_mode)
        if requested_geometry not in {"none", "trimer_scage_mcl"}:
            raise ValueError(
                f"unsupported MTS geometry mode: {requested_geometry}; "
                "retired experiment modes are unavailable"
            )
        resolved_geometry = requested_geometry
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
            raise ValueError("o8 msta_layer_indices must remain the default (4, 5)")
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
        self.topology_attention_variant = topology_attention_variant
        self.msta_layer_indices = normalized_layers
        self.msta_local_spd = normalized_local_spd
        self.msta_context_spd = normalized_context_spd
        self.msta_share_relation_dropout = bool(msta_share_relation_dropout)
        self.msta_local_output_bias = bool(msta_local_output_bias)
        self.msta_local_output_init = str(msta_local_output_init)
        self.use_descriptors = True
        self.descriptor_components = "md200"
        self.use_star_rbf = bool(use_star_rbf)
        self.use_mcl = bool(use_mcl)
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
        self.star_distance_bias = SymmetricStarDistanceBias(
            self.num_heads, upper=float(star_rbf_upper)
        )
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
                    "mts_star_v2_relation_row",
                    "mts_star_v2_relation_pair_index",
                    "mts_star_v2_pair_observation_distances",
                    "mts_star_v2_pair_observation_count",
                    "mts_star_v2_pair_valid",
                    "mts_star_v2_pair_geometry_source",
                )
            if self.use_mcl:
                required += (
                    "trimer_pos",
                    "trimer_central_ru_mask", "mips_to_trimer_central_index",
                    "trimer_geometry_valid", "trimer_geometry_is_3d",
                    "trimer_2d_fallback", "trimer_batch",
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
        self._validate(
            data, require_geometry=(use_star or use_geometry),
            require_md=use_md,
        )
        initial = self.atom_embedding(data, atom_mask=atom_mask)
        spd_bias = self.spd_embedding(data.lga_spd.long())
        path_bias = self.path_bias(initial, data)
        star_bias = torch.zeros_like(spd_bias)
        if use_star:
            star_bias = self.star_distance_bias.forward_periodic_relation_v2(
                data, initial.dtype
            )
        relation_mask = getattr(data, "lga_relation_mask", None)
        if relation_mask is not None:
            keep = (~relation_mask.bool()).unsqueeze(-1).to(initial.dtype)
            spd_bias = spd_bias * keep
            path_bias = path_bias * keep
            star_bias = star_bias * keep
        attention_bias = spd_bias + path_bias + star_bias
        x = initial
        relation_mask = getattr(data, "lga_relation_mask", None)
        for layer in self.layers:
            if isinstance(layer, MSTAMIPSLocalLayer):
                x = layer(
                    x, data.lga_edge_index.long(), attention_bias,
                    spd=data.lga_spd, relation_mask=relation_mask,
                )
            else:
                x = layer(x, data.lga_edge_index.long(), attention_bias)
        if use_geometry:
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
        self._validate(data, require_geometry=(self.use_star_rbf or self.use_mcl), require_md=False)
        initial = self.atom_embedding(data, atom_mask=canonical_atom_mask)
        spd_bias = self.spd_embedding(data.lga_spd.long())
        path_bias = self.path_bias(initial, data)
        star_bias = torch.zeros_like(spd_bias)
        if self.use_star_rbf:
            star_bias = self.star_distance_bias.forward_periodic_relation_v2(
                data, initial.dtype
            )
        relation_mask = getattr(data, "lga_relation_mask", None)
        if relation_mask is not None:
            keep = (~relation_mask.bool()).unsqueeze(-1).to(initial.dtype)
            spd_bias = spd_bias * keep
            path_bias = path_bias * keep
            star_bias = star_bias * keep
        attention_bias = spd_bias + path_bias + star_bias
        topology_nodes = initial
        for layer in self.layers:
            if isinstance(layer, MSTAMIPSLocalLayer):
                topology_nodes = layer(
                    topology_nodes, data.lga_edge_index.long(), attention_bias,
                    spd=data.lga_spd, relation_mask=relation_mask,
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

    def forward_b0_pretrain(
        self, data, canonical_atom_mask,
        noisy_pair_observation_distances=None,
        noisy_pair_observation_mask=None,
    ):
        """Run the B0 masked/dynamic Star-RBF O8 backbone.

        B0 intentionally omits Trimer-MCL and MD200 from the pretext path.
        Dynamic observation tensors are transient batch values; the frozen
        sidecar attached to ``data`` remains the clean fallback and is never
        modified.
        """
        mask = torch.as_tensor(
            canonical_atom_mask, dtype=torch.bool, device=data.mips_x.device
        ).reshape(-1)
        if mask.numel() != data.mips_x.size(0):
            raise ValueError("B0 canonical_atom_mask length mismatch")
        self._validate(data, require_geometry=self.use_star_rbf, require_md=False)
        initial = self.atom_embedding(data, atom_mask=mask)
        spd_bias = self.spd_embedding(data.lga_spd.long())
        path_bias = self.path_bias(initial, data)
        star_bias = torch.zeros_like(spd_bias)
        if self.use_star_rbf:
            star_bias = self.star_distance_bias.forward_periodic_relation_v2(
                data, initial.dtype,
                pair_observation_distances=noisy_pair_observation_distances,
                pair_observation_mask=noisy_pair_observation_mask,
            )
        relation_mask = getattr(data, "lga_relation_mask", None)
        if relation_mask is not None:
            keep = (~relation_mask.bool()).unsqueeze(-1).to(initial.dtype)
            spd_bias, path_bias, star_bias = (
                value * keep for value in (spd_bias, path_bias, star_bias)
            )
        attention_bias = spd_bias + path_bias + star_bias
        nodes = initial
        for layer in self.layers:
            if isinstance(layer, MSTAMIPSLocalLayer):
                nodes = layer(
                    nodes, data.lga_edge_index.long(), attention_bias,
                    spd=data.lga_spd, relation_mask=relation_mask,
                )
            else:
                nodes = layer(nodes, data.lga_edge_index.long(), attention_bias)
        graph_available = data.graph_available.bool().flatten()
        nodes = nodes * graph_available[data.batch.long()].unsqueeze(-1).to(nodes.dtype)
        graph, canonical_nodes = self._canonical_pool(nodes, data)
        graph = graph * graph_available.unsqueeze(-1).to(graph.dtype)
        return graph, nodes, canonical_nodes

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
