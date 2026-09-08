"""Current-backbone downstream wrapper for the restored Original-MIPS path.

The graph branch remains the current canonical O8 implementation.  The only
Original-MIPS pieces are the MD200 supplied by the restored descriptor path
and :class:`OriginalMIPSAttentiveFusion`; Atomic-PC-v1 is consumed unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .atomic_point_encoder import AtomicPointEncoder, PackedAtomicPointCloud
from .mips_local_graph import MIPSLocalGraphEncoder
from .original_mips_knowledge_fusion import OriginalMIPSAttentiveFusion


def _get(data: Any, name: str, default: Any = None) -> Any:
    if isinstance(data, Mapping) and name in data:
        return data[name]
    return getattr(data, name, default)


class OriginalMIPSAtomicPCModel(nn.Module):
    """O8 node states -> Original KFuse -> canonical pooling -> head."""

    architecture_name = (
        "Current Backbone + Original-MIPS Knowledge Path + Atomic-PC-v1"
    )
    current_backbone_is_original_mips = False
    knowledge_names = ("md", "atomic_pc")
    output_dim = 1
    hidden_dim = 512
    joint_dim = 256

    def __init__(
        self,
        *,
        graph_encoder: MIPSLocalGraphEncoder | None = None,
        atomic_point_encoder: AtomicPointEncoder | None = None,
        fusion: OriginalMIPSAttentiveFusion | None = None,
        graph_dropout: float = 0.1,
        head_dropout: float = 0.25,
        atomic_pool_scope: str = "all",
    ) -> None:
        super().__init__()
        self.atomic_pool_scope = str(atomic_pool_scope)
        self.graph_encoder = graph_encoder or MIPSLocalGraphEncoder(
            core="paper_corrected", num_layer=6, emb_dim=512, num_heads=8,
            dropout=float(graph_dropout), max_hops=2, use_descriptors=True,
            graph_geometry_mode="trimer_scage_mcl", use_star_rbf=False,
            use_mcl=False, topology_attention_variant="o8",
        )
        self.atomic_point_encoder = atomic_point_encoder or AtomicPointEncoder(
            hidden_dim=256, output_dim=512, num_layers=4, k_neighbors=24,
            use_geometry=True, knn_backend="torch_cluster",
            pool_scope=self.atomic_pool_scope,
        )
        self.atomic_pool_scope = str(getattr(self.atomic_point_encoder, "pool_scope", self.atomic_pool_scope))
        self.fusion = fusion or OriginalMIPSAttentiveFusion(
            d_model=512, knodes=("md", "atomic_pc"),
            knowledge_dims={"md": 200, "atomic_pc": 512},
        )
        # These are the current EncoderModule graph readout/adaptation and
        # UniEncoderAttention predictor dimensions, kept outside KFuse.
        self.graph_norm = nn.LayerNorm(512)
        self.graph_projection = nn.Sequential(
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(float(head_dropout)),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(float(head_dropout)),
            nn.Linear(64, 1),
        )
        # Old graph-level MD residual is intentionally disabled on this route.
        for parameter in self.graph_encoder.md_residual.parameters():
            parameter.requires_grad = False
        # The canonical O8 downstream contract fixes Star-RBF off; keep its
        # compatibility parameters frozen just as the current finetune
        # trainability policy does.
        star_bias = getattr(self.graph_encoder, "star_distance_bias", None)
        if star_bias is not None:
            for parameter in star_bias.parameters():
                parameter.requires_grad = False
        self.last_forward_trace: dict[str, Any] = {}

    @staticmethod
    def _point_cloud(data: Any) -> Any:
        cloud = _get(data, "atomic_point_cloud")
        if cloud is not None:
            return cloud
        coords = _get(data, "atomic_point_coords", _get(data, "trimer_coords"))
        z = _get(data, "atomic_point_atomic_number", _get(data, "trimer_atomic_number"))
        offset = _get(data, "atomic_point_ru_offset", _get(data, "trimer_ru_offset"))
        batch = _get(data, "atomic_point_batch", _get(data, "trimer_batch"))
        ptr = _get(data, "atomic_point_ptr", _get(data, "trimer_ptr"))
        if coords is None or z is None or offset is None:
            raise ValueError("data is missing the explicit Atomic-PC geometry")
        if batch is None:
            batch = torch.zeros(coords.size(0), dtype=torch.long, device=coords.device)
        return PackedAtomicPointCloud(coords, z, offset, batch, ptr=ptr)

    @staticmethod
    def _knowledge_md(data: Any, graph_count: int, device: torch.device) -> torch.Tensor:
        mapping = _get(data, "knowledge")
        if mapping is None:
            mapping = _get(data, "kvecs")
        if isinstance(mapping, Mapping):
            md = mapping.get("md", mapping.get("mips_md"))
        else:
            md = _get(data, "mips_md", _get(data, "md"))
        if md is None:
            raise ValueError("data is missing restored MD200")
        md = torch.as_tensor(md, device=device)
        if md.ndim == 1:
            if graph_count != 1:
                raise ValueError("rank-1 MD200 is valid only for one graph")
            md = md.unsqueeze(0)
        if md.ndim != 2 or tuple(md.shape) != (graph_count, 200):
            raise ValueError(f"MD200 must be [{graph_count},200], got {tuple(md.shape)}")
        valid = _get(data, "mips_md_valid", _get(data, "md_valid"))
        if valid is not None:
            valid = torch.as_tensor(valid, device=device, dtype=torch.bool).reshape(-1)
            if valid.numel() != graph_count:
                raise ValueError("mips_md_valid graph count mismatch")
            md = md * valid.to(md.dtype).unsqueeze(-1)
        return md

    def forward(self, data: Any, *, return_dict: bool = False):
        if not hasattr(self.graph_encoder, "_forward_impl") or not hasattr(self.graph_encoder, "_canonical_pool"):
            raise TypeError("graph encoder must expose current _forward_impl and canonical pooling")
        self.fusion.reset_trace()
        o8_graph, node_states = self.graph_encoder._forward_impl(
            data, use_star=False, use_md=False
        )
        if node_states.ndim != 2 or int(node_states.size(1)) != self.hidden_dim:
            raise ValueError("current O8 node states must be [N,512]")
        graph_count = int(o8_graph.size(0))
        batch_index = torch.as_tensor(
            _get(data, "batch"), device=node_states.device, dtype=torch.long
        ).reshape(-1)
        if batch_index.numel() != node_states.size(0):
            raise ValueError("graph batch length does not match node states")
        cloud = self._point_cloud(data)
        atomic_pc, point_stats = self.atomic_point_encoder(cloud, return_aux=True)
        if tuple(atomic_pc.shape) != (graph_count, 512):
            raise ValueError("Atomic-PC output must match graph count and width 512")
        knowledge = {
            "md": self._knowledge_md(data, graph_count, node_states.device),
            "atomic_pc": atomic_pc,
        }
        fused_nodes = self.fusion(node_states, knowledge, batch_index)
        available = _get(data, "graph_available")
        if available is not None:
            available = torch.as_tensor(
                available, device=fused_nodes.device, dtype=torch.bool
            ).reshape(-1)
            if available.numel() != graph_count:
                raise ValueError("graph_available count mismatch")
            fused_nodes = fused_nodes * available[batch_index].to(fused_nodes.dtype).unsqueeze(-1)
        fused_graph, canonical_nodes = self.graph_encoder._canonical_pool(
            fused_nodes, data
        )
        if available is not None:
            fused_graph = fused_graph * available.to(fused_graph.dtype).unsqueeze(-1)
        joint = self.graph_projection(self.graph_norm(fused_graph))
        prediction = self.mlp(joint)
        trace = {
            "fusion_calls": int(self.fusion.fusion_call_count),
            "placement": "node_states->OriginalMIPSAttentiveFusion->canonical_pooling->predictor",
            "knowledge": {"md": tuple(knowledge["md"].shape), "atomic_pc": tuple(atomic_pc.shape)},
            "node_state_shape": tuple(node_states.shape),
            "fused_node_shape": tuple(fused_nodes.shape),
            "canonical_shape": tuple(canonical_nodes.shape),
            "graph_shape": tuple(fused_graph.shape),
            "point_stats": point_stats,
            "atomic_pool_scope": self.atomic_pool_scope,
        }
        self.last_forward_trace = trace
        if return_dict:
            return {
                "prediction": prediction,
                "embedding": joint,
                "graph": fused_graph,
                "node_states": fused_nodes,
                "o8_graph": o8_graph,
                "o8_node_states": node_states,
                "canonical_states": canonical_nodes,
                "atomic_pc": atomic_pc,
                "knowledge": knowledge,
                "fusion_calls": int(self.fusion.fusion_call_count),
                "trace": trace,
            }
        # Match the current training utility contract: prediction plus an
        # optional embedding payload.
        return prediction, joint


__all__ = ["OriginalMIPSAtomicPCModel"]
