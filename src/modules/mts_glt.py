"""Parallel pure-topology O8 and local periodic GLT candidate model."""

from __future__ import annotations

import torch
from torch import nn

from .mips_local_graph import MIPSLocalGraphEncoder
from .periodic_line_glt import LocalPeriodicGraphLineTransformer


class MTSGraphLineModel(nn.Module):
    """MTS-GLT-v1 encoder with strict O8-only and O8+GLT readout modes."""

    architecture_name = "MIPS-Trimer-SCAGE"

    def __init__(self, hidden_size=512, glt_layers=6, glt_heads=8, dropout=0.1):
        super().__init__()
        if int(hidden_size) != 512:
            raise ValueError("MTS-GLT-v1 fixes hidden_size=512")
        self.o8 = MIPSLocalGraphEncoder(
            core="paper_corrected",
            num_layer=6,
            emb_dim=512,
            num_heads=8,
            dropout=dropout,
            max_hops=2,
            use_descriptors=True,
            graph_geometry_mode="trimer_scage_mcl",
            use_star_rbf=False,
            use_mcl=False,
            topology_attention_variant="o8",
        )
        self.glt = LocalPeriodicGraphLineTransformer(
            hidden_size=hidden_size,
            layers=glt_layers,
            heads=glt_heads,
            dropout=dropout,
            distance_basis=128,
            angle_basis=128,
            distance_upper=3.75,
        )
        self.glt_fusion_norm = nn.LayerNorm(hidden_size)
        self.glt_fusion_projection = nn.Linear(hidden_size, hidden_size)
        self.glt_gate = nn.Parameter(torch.zeros(()))
        self.downstream_mode = "o8_only"
        self.use_star_rbf = False
        self.use_mcl = False
        for parameter in self.o8.star_distance_bias.parameters():
            parameter.requires_grad = False

    def forward_pretrain(
        self,
        data,
        *,
        atom_mask,
        line_endpoint_z_a=None,
        line_endpoint_z_b=None,
        line_observation_distances=None,
        line_observation_counts=None,
    ):
        z_o8, atom_states = self.o8._forward_impl(
            data, atom_mask=atom_mask, use_star=False, use_md=False
        )
        z_glt, line_states = self.glt(
            data,
            endpoint_z_a=line_endpoint_z_a,
            endpoint_z_b=line_endpoint_z_b,
            observation_distances=line_observation_distances,
            observation_counts=line_observation_counts,
        )
        return z_o8, z_glt, atom_states, line_states

    def forward_downstream(self, data, mode="o8_only"):
        if mode not in {"o8_only", "o8_glt"}:
            raise ValueError("MTS-GLT downstream mode must be o8_only or o8_glt")
        z_o8, _ = self.o8._forward_impl(data, use_star=False, use_md=False)
        graph = z_o8
        if mode == "o8_glt":
            z_glt, _ = self.glt(data)
            update = self.glt_fusion_projection(self.glt_fusion_norm(z_glt))
            valid = data.glt_geometry_valid.to(update.dtype).unsqueeze(-1)
            graph = graph + valid * torch.tanh(self.glt_gate) * update
        graph = self.o8.md_residual(graph, data)
        available = data.graph_available.bool().flatten().to(graph.dtype).unsqueeze(-1)
        return graph * available

    def encode_views(self, data):
        """Return the unfused O8/GLT views and the current GLT residual.

        This is an opt-in postmortem interface.  The ordinary downstream
        forward keeps exactly the same computation and return type.
        """
        z_o8, _ = self.o8._forward_impl(data, use_star=False, use_md=False)
        z_glt, _ = self.glt(data)
        projected_z_glt = self.glt_fusion_projection(
            self.glt_fusion_norm(z_glt)
        )
        valid = data.glt_geometry_valid.to(projected_z_glt.dtype).unsqueeze(-1)
        delta_z_3d = valid * torch.tanh(self.glt_gate) * projected_z_glt
        return {
            "z_o8": z_o8,
            "z_glt": z_glt,
            "projected_z_glt": projected_z_glt,
            "delta_z_3d": delta_z_3d,
            "valid_3d": data.glt_geometry_valid.bool().flatten(),
        }

    def forward(self, data, mode=None):
        return self.forward_downstream(
            data, mode=self.downstream_mode if mode is None else mode
        )


__all__ = ["MTSGraphLineModel"]
