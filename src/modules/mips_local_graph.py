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
    CANONICAL_LGA_SCHEMA_VERSION,
)
from .trimer_mcl import TrimerSCAGEMCLResidual


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

    def forward(self, x, edge_index, attention_bias):
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
            residual[valid] = self.adapter(data.mips_md.float()[valid])
        else:
            residual = residual + self.gate * 0.0
        return graph + residual * torch.tanh(self.gate)


class MIPSLocalGraphEncoder(nn.Module):
    """O8 + symmetric Star RBF + two-layer Trimer MCL + MD200."""

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
        }
        requested_geometry = str(graph_geometry_mode)
        if requested_geometry not in geometry_aliases:
            raise ValueError(f"unsupported MTS geometry mode: {requested_geometry}")
        resolved_geometry = geometry_aliases[requested_geometry]
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

        self.core = "paper_corrected"
        self.variant = "O8"
        self.emb_dim = 512
        self.num_heads = 8
        self.max_hops = 2
        self.feature_mode = "mips137"
        self.spatial_mode = "trimer_scage"
        self.graph_geometry_mode = requested_geometry
        self.geometry_mode = resolved_geometry
        self.use_descriptors = True
        self.descriptor_components = "md200"
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
        self.layers = nn.ModuleList(
            MIPSLocalLayer(self.emb_dim, self.num_heads, dropout)
            for _ in range(6)
        )
        self.final_norm = nn.Identity()
        self.trimer_mcl = TrimerSCAGEMCLResidual(
            dim=self.emb_dim, num_heads=self.num_heads,
            percentiles=(0.20, 0.50), dropout=dropout,
            use_distance_bias=("mcl_rbf" in resolved_geometry),
            coordinate_shuffle=("coordinate_shuffled" in resolved_geometry),
        )
        self.md_residual = MD200GraphResidual(self.emb_dim, dropout)

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
            required += (
                "star_3d_distance", "star_3d_asymmetry", "star_3d_valid",
                "trimer_pos", "trimer_base_ru_atom_index",
                "trimer_central_ru_mask", "mips_to_trimer_central_index",
                "trimer_geometry_valid", "trimer_geometry_is_3d",
                "trimer_2d_fallback", "trimer_batch",
            )
        if require_md:
            required += ("mips_md", "mips_md_valid")
        missing = [name for name in required if not hasattr(data, name)]
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

        x = initial
        for layer in self.layers:
            x = layer(x, data.lga_edge_index.long(), attention_bias)
        if use_geometry and self.geometry_mode != "disabled":
            x = x + self.trimer_mcl(x, data)

        graph_available = data.graph_available.bool().flatten()
        x = x * graph_available[data.batch.long()].unsqueeze(-1).to(x.dtype)
        graph = global_mean_pool(x, data.batch.long())
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
        return self._forward_impl(data)

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
        self._validate(data, require_geometry=True, require_md=False)
        initial = self.atom_embedding(data, atom_mask=canonical_atom_mask)
        spd_bias = self.spd_embedding(data.lga_spd.long())
        path_bias = self.path_bias(initial, data)
        star_bias = self.star_distance_bias(data, initial.dtype)
        relation_mask = getattr(data, "lga_relation_mask", None)
        if relation_mask is not None:
            keep = (~relation_mask.bool()).unsqueeze(-1).to(initial.dtype)
            spd_bias = spd_bias * keep
            path_bias = path_bias * keep
            star_bias = star_bias * keep
        attention_bias = spd_bias + path_bias + star_bias
        topology_nodes = initial
        for layer in self.layers:
            topology_nodes = layer(
                topology_nodes, data.lga_edge_index.long(), attention_bias
            )
        geometry_delta, trimer_states, mcl_valid = (
            self.trimer_mcl.forward_with_aux(topology_nodes, data)
        )
        nodes = topology_nodes + geometry_delta
        graph_available = data.graph_available.bool().flatten()
        nodes = nodes * graph_available[data.batch.long()].unsqueeze(-1).to(nodes.dtype)
        graph = global_mean_pool(nodes, data.batch.long())
        graph = graph * graph_available.unsqueeze(-1).to(graph.dtype)
        return graph, nodes, {
            "final_trimer_states": trimer_states,
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
