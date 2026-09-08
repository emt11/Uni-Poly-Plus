"""MTS-GLT-v3 joint O8/MD200/center-image GLT model."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch_scatter import scatter

from .mips_local_graph import MIPSLocalGraphEncoder
from .periodic_line_glt_v3 import LocalPeriodicGraphLineTransformerV3


class MD200NodeResidual(nn.Module):
    def __init__(self, hidden=512, dropout=0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(200), nn.Linear(200, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, hidden, bias=False),
        )
        self.gate = nn.Parameter(torch.tensor(math.atanh(0.05)))

    def forward(self, nodes, data, md200=None):
        values = data.mips_md if md200 is None else md200
        update = self.adapter(values.float()).to(nodes.dtype)
        return nodes + torch.tanh(self.gate).to(nodes.dtype) * update[data.batch.long()]


class MTSGraphLineModelV3(nn.Module):
    architecture_name = "MTS-GLT-v3-Galformer"

    def __init__(self, *, glt_readout_mode="galformer", dropout=0.1):
        super().__init__()
        if glt_readout_mode not in {"galformer", "mips_concat"}:
            raise ValueError("glt_readout_mode must be galformer or mips_concat")
        self.glt_readout_mode = str(glt_readout_mode)
        self.o8 = MIPSLocalGraphEncoder(
            core="paper_corrected", num_layer=6, emb_dim=512, num_heads=8,
            dropout=dropout, max_hops=2, use_descriptors=True,
            graph_geometry_mode="trimer_scage_mcl", use_star_rbf=False,
            use_mcl=False, topology_attention_variant="o8",
        )
        self.glt = (
            None if self.glt_readout_mode == "galformer"
            else LocalPeriodicGraphLineTransformerV3(dropout=dropout)
        )
        self.concat_norm = nn.LayerNorm(1024) if self.glt is not None else None
        self.concat_projection = nn.Linear(1024, 512) if self.glt is not None else None
        self.md_residual = MD200NodeResidual(512, dropout)

    def instantiate_pretraining_glt(self):
        """Install GLT for joint pretraining while preserving downstream mode."""
        if self.glt is None:
            self.glt = LocalPeriodicGraphLineTransformerV3(dropout=0.1)
        return self.glt

    @staticmethod
    def _pool(data, canonical_states):
        return scatter(
            canonical_states, data.canonical_graph_index.long(), dim=0,
            dim_size=int(data.graph_available.numel()), reduce="mean",
        )

    def o8_canonical(self, data, *, atom_mask=None):
        _, nodes = self.o8._forward_impl(
            data, atom_mask=atom_mask, use_star=False, use_md=False
        )
        _, canonical = self.o8._canonical_pool(nodes, data)
        return nodes, canonical

    def forward_joint(self, data, *, atom_mask=None, md200=None, token_overrides=None):
        nodes, canonical = self.o8_canonical(data, atom_mask=atom_mask)
        fused_nodes = self.md_residual(nodes, data, md200=md200)
        _, fused_canonical = self.o8._canonical_pool(fused_nodes, data)
        glt = self.instantiate_pretraining_glt()(data, token_overrides=token_overrides)
        return {
            "o8_node_states": nodes, "o8_md_node_states": fused_nodes,
            "o8_canonical_states": fused_canonical,
            "z_o8": self._pool(data, fused_canonical), **glt,
        }

    def forward(self, data):
        nodes, canonical = self.o8_canonical(data)
        if self.glt_readout_mode == "mips_concat":
            glt = self.glt(data)
            canonical = self.concat_projection(self.concat_norm(torch.cat([
                canonical, glt["atom_geometry_states"],
            ], dim=-1)))
            # Broadcast atom-aligned fused states back onto canonical O8 nodes.
            nodes = canonical[data.canonical_ru_atom_index.long()]
        nodes = self.md_residual(nodes, data)
        _, canonical = self.o8._canonical_pool(nodes, data)
        graph = self._pool(data, canonical)
        available = data.graph_available.to(graph.dtype).unsqueeze(-1)
        return graph * available


__all__ = ["MD200NodeResidual", "MTSGraphLineModelV3"]
