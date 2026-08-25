"""Pure-topology O8 plus graph-query GLT channel-gated fusion."""

from __future__ import annotations

import math
import torch
from torch import nn

from .mips_local_graph import MIPSLocalGraphEncoder
from .periodic_line_glt_graphgate import LocalPeriodicGraphLineTransformerGraphGate


class MTSGraphGateModel(nn.Module):
    architecture_name = "MTS-GLT-GraphGate-v1"

    def __init__(self, hidden_size=512, layers=6, heads=8, dropout=0.1):
        super().__init__()
        self.o8_encoder = MIPSLocalGraphEncoder(
            core="paper_corrected", num_layer=6, emb_dim=hidden_size,
            num_heads=8, dropout=dropout, max_hops=2, use_descriptors=True,
            graph_geometry_mode="trimer_scage_mcl", use_star_rbf=False,
            use_mcl=False, topology_attention_variant="o8",
        )
        self.glt_line_encoder = LocalPeriodicGraphLineTransformerGraphGate(
            hidden_size, layers, heads, dropout
        )
        self.fusion_norm = nn.LayerNorm(hidden_size)
        self.fusion_projection = nn.Linear(hidden_size, hidden_size)
        self.channel_gate = nn.Parameter(torch.full((hidden_size,), math.atanh(0.05)))
        self.downstream_mode = "o8_only"
        self.glt_geometry_mode = "full"
        self.use_star_rbf = False
        self.use_mcl = False
        for parameter in self.o8_encoder.star_distance_bias.parameters():
            parameter.requires_grad = False

    @property
    def query_pool(self):
        return self.glt_line_encoder.query_pool

    def forward_pretrain(self, data, *, atom_mask, line_inputs=None):
        z_o8, atom_states = self.o8_encoder._forward_impl(
            data, atom_mask=atom_mask, use_star=False, use_md=False
        )
        glt = self.glt_line_encoder(
            data, line_inputs=line_inputs, geometry_mode=self.glt_geometry_mode
        )
        return {"z_o8": z_o8, "z_glt": glt["graph_geometry"],
                "atom_states": atom_states, **glt}

    def forward_downstream(self, data, mode="o8_only", return_audit=False):
        if mode not in {"o8_only", "o8_glt_graph"}:
            raise ValueError("invalid GraphGate downstream mode")
        z_o8, _ = self.o8_encoder._forward_impl(data, use_star=False, use_md=False)
        residual = torch.zeros_like(z_o8)
        query_valid = torch.zeros(z_o8.size(0), dtype=torch.bool, device=z_o8.device)
        if mode == "o8_glt_graph":
            glt = self.glt_line_encoder(data, geometry_mode=self.glt_geometry_mode)
            query_valid = glt["query_valid"]
            residual = (
                query_valid.to(z_o8.dtype).unsqueeze(-1)
                * torch.tanh(self.channel_gate)
                * self.fusion_projection(self.fusion_norm(glt["graph_geometry"]))
            )
        graph = self.o8_encoder.md_residual(z_o8 + residual, data)
        graph = graph * data.graph_available.bool().flatten().to(graph.dtype).unsqueeze(-1)
        if not return_audit:
            return graph
        rho = torch.linalg.vector_norm(residual, dim=-1) / torch.linalg.vector_norm(z_o8, dim=-1).clamp_min(1e-8)
        return {"graph": graph, "z_o8": z_o8, "residual": residual,
                "rho": rho, "query_valid": query_valid, "alpha": torch.tanh(self.channel_gate)}

    def encode_views(self, data):
        z_o8, _ = self.o8_encoder._forward_impl(data, use_star=False, use_md=False)
        glt = self.glt_line_encoder(data, geometry_mode=self.glt_geometry_mode)
        projected = self.fusion_projection(self.fusion_norm(glt["graph_geometry"]))
        residual = (
            glt["query_valid"].to(z_o8.dtype).unsqueeze(-1)
            * torch.tanh(self.channel_gate) * projected
        )
        return {
            "z_o8": z_o8, "z_glt": glt["graph_geometry"],
            "projected_z_glt": projected, "delta_z_3d": residual,
            "valid_3d": glt["query_valid"], "alpha": torch.tanh(self.channel_gate),
        }

    def forward(self, data, mode=None):
        return self.forward_downstream(data, self.downstream_mode if mode is None else mode)


__all__ = ["MTSGraphGateModel"]
