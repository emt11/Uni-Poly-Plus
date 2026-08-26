"""Pure-topology O8 plus graph-query GLT channel-gated fusion."""

from __future__ import annotations

import math
import torch
from torch import nn
from torch_scatter import scatter

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

    @staticmethod
    def _mean_line_readout(line_states, data, query_valid):
        token_batch = data.glt_token_batch.long()
        valid = data.glt_token_valid.bool() & query_valid[token_batch]
        graph_count = int(query_valid.numel())
        pooled = scatter(
            line_states * valid.to(line_states.dtype).unsqueeze(-1),
            token_batch, dim=0, dim_size=graph_count, reduce="sum",
        )
        count = scatter(
            valid.to(line_states.dtype), token_batch,
            dim=0, dim_size=graph_count, reduce="sum",
        ).clamp_min(1.0)
        return pooled / count.unsqueeze(-1)

    @staticmethod
    def _central_atom_readout(line_states, data, query_valid, atom_count):
        token_batch = data.glt_token_batch.long()
        valid = data.glt_token_valid.bool() & query_valid[token_batch]
        endpoints = torch.cat([
            data.glt_token_atom_a.long(), data.glt_token_atom_b.long()
        ])
        incidence_states = torch.cat([line_states, line_states], dim=0)
        incidence_valid = torch.cat([valid, valid]).to(line_states.dtype)
        summed = scatter(
            incidence_states * incidence_valid.unsqueeze(-1), endpoints,
            dim=0, dim_size=int(atom_count), reduce="sum",
        )
        count = scatter(
            incidence_valid, endpoints, dim=0,
            dim_size=int(atom_count), reduce="sum",
        )
        atom_valid = count > 0
        return summed / count.clamp_min(1.0).unsqueeze(-1), atom_valid

    @staticmethod
    def _canonical_graph_pool(atom_states, data):
        graph_count = int(data.graph_available.numel())
        return scatter(
            atom_states, data.canonical_graph_index.long(), dim=0,
            dim_size=graph_count, reduce="mean",
        )

    def _readout_views(self, data, mode):
        z_o8, node_states = self.o8_encoder._forward_impl(
            data, use_star=False, use_md=False
        )
        query_valid = torch.zeros(
            z_o8.size(0), dtype=torch.bool, device=z_o8.device
        )
        z_glt = torch.zeros_like(z_o8)
        projected = torch.zeros_like(z_o8)
        residual = torch.zeros_like(z_o8)

        if mode == "o8_glt_graph":
            glt = self.glt_line_encoder(data, geometry_mode=self.glt_geometry_mode)
            query_valid = glt["query_valid"]
            z_glt = glt["graph_geometry"]
            projected = self.fusion_projection(self.fusion_norm(z_glt))
            residual = (
                query_valid.to(z_o8.dtype).unsqueeze(-1)
                * torch.tanh(self.channel_gate) * projected
            )
            fused = z_o8 + residual
        elif mode == "o8_glt_graph_mean":
            glt = self.glt_line_encoder.encode_lines(
                data, geometry_mode=self.glt_geometry_mode
            )
            query_valid = glt["query_valid"]
            z_glt = self._mean_line_readout(
                glt["line_states"], data, query_valid
            )
            projected = self.fusion_projection(self.fusion_norm(z_glt))
            residual = (
                query_valid.to(z_o8.dtype).unsqueeze(-1)
                * torch.tanh(self.channel_gate) * projected
            )
            fused = z_o8 + residual
        elif mode == "o8_glt_atom_central":
            _, canonical_states = self.o8_encoder._canonical_pool(node_states, data)
            glt = self.glt_line_encoder.encode_lines(
                data, geometry_mode=self.glt_geometry_mode
            )
            query_valid = glt["query_valid"]
            atom_glt, atom_valid = self._central_atom_readout(
                glt["line_states"], data, query_valid,
                canonical_states.size(0),
            )
            atom_projected = self.fusion_projection(self.fusion_norm(atom_glt))
            atom_residual = (
                atom_valid.to(z_o8.dtype).unsqueeze(-1)
                * torch.tanh(self.channel_gate) * atom_projected
            )
            fused = self._canonical_graph_pool(
                canonical_states + atom_residual, data
            )
            z_glt = self._canonical_graph_pool(atom_glt, data)
            projected = self._canonical_graph_pool(
                atom_projected * atom_valid.to(z_o8.dtype).unsqueeze(-1), data
            )
            residual = fused - z_o8
        else:
            fused = z_o8
        return z_o8, z_glt, projected, residual, query_valid, fused

    def forward_downstream(self, data, mode="o8_only", return_audit=False):
        if mode not in {
            "o8_only", "o8_glt_graph", "o8_glt_graph_mean",
            "o8_glt_atom_central",
        }:
            raise ValueError("invalid GraphGate downstream mode")
        z_o8, _, _, residual, query_valid, fused = self._readout_views(data, mode)
        graph = self.o8_encoder.md_residual(fused, data)
        graph = graph * data.graph_available.bool().flatten().to(graph.dtype).unsqueeze(-1)
        if not return_audit:
            return graph
        rho = torch.linalg.vector_norm(residual, dim=-1) / torch.linalg.vector_norm(z_o8, dim=-1).clamp_min(1e-8)
        return {"graph": graph, "z_o8": z_o8, "residual": residual,
                "rho": rho, "query_valid": query_valid, "alpha": torch.tanh(self.channel_gate)}

    def encode_views(self, data):
        z_o8, z_glt, projected, residual, query_valid, _ = self._readout_views(
            data, self.downstream_mode
        )
        return {
            "z_o8": z_o8, "z_glt": z_glt,
            "projected_z_glt": projected, "delta_z_3d": residual,
            "valid_3d": query_valid, "alpha": torch.tanh(self.channel_gate),
        }

    def forward(self, data, mode=None):
        return self.forward_downstream(data, self.downstream_mode if mode is None else mode)


__all__ = ["MTSGraphGateModel"]
