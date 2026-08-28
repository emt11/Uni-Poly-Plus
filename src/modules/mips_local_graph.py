"""Fixed graph core for the ``MIPS-Trimer-SCAGE`` (MTS) route."""

import os
import math
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
    CANONICAL_LGA_SCHEMA_VERSION,
    TOPOLOGY_CANONICAL,
)

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
    """MTS-GLT-v2 O8 topology encoder with the MD200 residual."""

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
        use_star_rbf=False,
        star_rbf_upper=3.0,
        use_mcl=False,
        topology_attention_variant="o8",
        **retired,
    ):
        super().__init__()
        requested_geometry = str(graph_geometry_mode)
        if requested_geometry != "trimer_scage_mcl":
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
        if int(max_hops) != 2:
            raise ValueError("MTS-GLT-v2 requires max_hops=2")
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
            raise ValueError(f"{ROUTE_NAME} MTS-GLT-v2 downstream requires MD200")
        if bool(multi_scale_hop_gate) or bool(input_norm):
            raise ValueError("retired O8 topology options are not supported")
        if abs(float(dropout) - 0.10) > 1e-12:
            raise ValueError("dropout must be 0.10")
        if int(trimer_num_candidates) != 4 or int(trimer_max_heavy_atoms) != 384:
            raise ValueError("Trimer contract requires 4 candidates/384 atoms")
        topology_attention_variant = str(topology_attention_variant)
        if topology_attention_variant != "o8":
            raise ValueError("MTS-GLT-v2 requires topology_attention_variant='o8'")
        if bool(use_mcl):
            raise ValueError("MTS-GLT-v2 does not use the MCL model branch")

        self.core = "paper_corrected"
        self.variant = "O8"
        self.emb_dim = 512
        self.num_heads = 8
        self.max_hops = int(max_hops)
        self.feature_mode = "mips137"
        self.spatial_mode = "trimer_scage"
        self.graph_geometry_mode = requested_geometry
        self.geometry_mode = resolved_geometry
        self.topology_attention_variant = topology_attention_variant
        self.use_descriptors = bool(use_descriptors)
        self.descriptor_components = "md200"
        if bool(use_star_rbf):
            raise ValueError("MTS-GLT-v2 baseline fixes Star-RBF off")
        self.use_star_rbf = False
        self.use_mcl = False
        if str(mask_policy) != "canonical_exact":
            raise ValueError(
                "all canonical-equivalent O8 copies must be masked together"
            )
        self.mask_policy = "canonical_exact"
        self.masked_atom_target = "mips101"
        self.masked_atom_classes = MIPS_ATOM_FEATURE_DIM - 36
        self.masked_loss_reduction = "atom_mean"

        self.atom_embedding = MIPSLocalAtomEmbedding(self.emb_dim)
        self.spd_embedding = nn.Embedding(self.max_hops + 1, self.num_heads)
        nn.init.zeros_(self.spd_embedding.weight)
        self.path_bias = MIPSSinglePathNodeBias(
            self.emb_dim, self.num_heads, self.max_hops
        )
        self.star_distance_bias = SymmetricStarDistanceBias(
            self.num_heads, upper=float(star_rbf_upper)
        )
        self.layers = nn.ModuleList(
            MIPSLocalLayer(self.emb_dim, self.num_heads, dropout)
            for _ in range(6)
        )
        self.final_norm = nn.Identity()
        self.md_residual = MD200GraphResidual(self.emb_dim, dropout=dropout)

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
        if representation != TOPOLOGY_CANONICAL:
            raise ValueError("MTS-GLT-v2 only supports canonical_lifted topology")
        if getattr(data, "feature_schema", None) != FEATURE_SCHEMA:
            raise ValueError("MTS-GLT-v2 canonical feature schema mismatch")
        if int(getattr(data, "mips_local_lga_schema_version", -1)) != CANONICAL_LGA_SCHEMA_VERSION:
            raise ValueError("MTS-GLT-v2 canonical LGA schema mismatch")
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
        if require_md:
            required += ("mips_md", "mips_md_valid")
        missing = [name for name in required if not hasattr(data, name)]
        if missing:
            raise ValueError(
                f"{ROUTE_NAME} cache is missing " + ", ".join(missing)
            )
        if data.lga_spd.numel() and (
            int(data.lga_spd.min()) < 0 or int(data.lga_spd.max()) > self.max_hops
        ):
            raise ValueError(
                f"{ROUTE_NAME} only permits 0..{self.max_hops}-hop attention"
            )
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
        self, data, atom_mask=None, *, use_star=False, use_md=True,
    ):
        if bool(use_star):
            raise ValueError("MTS-GLT-v2 baseline fixes Star-RBF off")
        self._validate(
            data, require_geometry=use_star,
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
        for layer in self.layers:
            x = layer(x, data.lga_edge_index.long(), attention_bias)

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
            use_md=self.use_descriptors,
        )
