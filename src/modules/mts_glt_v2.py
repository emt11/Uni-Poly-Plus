"""MTS-GLT-v2 with pure-topology O8 and atom-aligned 3-D fusion."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch_scatter import scatter

from .mips_local_graph import MIPSLocalGraphEncoder
from .periodic_line_glt_v2 import LocalPeriodicGraphLineTransformerV2


class CompactTrimerDescriptorResidual(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(19),
            nn.Linear(19, 64),
            nn.GELU(),
            nn.Linear(64, hidden_size),
        )
        self.gate = nn.Parameter(torch.full((hidden_size,), math.atanh(0.05)))

    def forward(self, graph, data):
        if not hasattr(data, "glt_compact19"):
            raise ValueError(
                "Compact19 descriptors are not part of the MTS-GLT-v2 baseline"
            )
        values = data.glt_compact19.to(graph.dtype)
        valid = data.glt_compact19_valid.bool()
        valid = valid.to(graph.dtype).unsqueeze(-1)
        return graph + valid * torch.tanh(self.gate) * self.network(values)


class MTSGraphLineModelV2(nn.Module):
    architecture_name = "MIPS-Trimer-GLT-v2"

    def __init__(
        self,
        hidden_size=512,
        glt_layers=6,
        glt_heads=8,
        dropout=0.1,
        glt_attention_variant="mips",
        use_compact19=False,
    ):
        super().__init__()
        if int(hidden_size) != 512:
            raise ValueError("MTS-GLT-v2 fixes hidden_size=512")
        if int(glt_layers) != 6 or int(glt_heads) != 8:
            raise ValueError("MTS-GLT-v2 fixes six GLT layers with eight heads")
        if str(glt_attention_variant) != "mips":
            raise ValueError("MTS-GLT-v2 fixes MIPS GLT attention")
        if bool(use_compact19):
            raise ValueError("MTS-GLT-v2-Base-5k does not include Compact19")
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
        self.glt = LocalPeriodicGraphLineTransformerV2(
            hidden_size=hidden_size,
            layers=glt_layers,
            heads=glt_heads,
            dropout=dropout,
            attention_variant=glt_attention_variant,
        )
        self.atom_fusion_norm = nn.LayerNorm(hidden_size)
        self.atom_fusion_projection = nn.Linear(hidden_size, hidden_size)
        self.atom_channel_gate = nn.Parameter(
            torch.full((hidden_size,), math.atanh(0.05))
        )
        self.compact19_residual = CompactTrimerDescriptorResidual(hidden_size)
        # Keep the parameter container solely so the retained pretraining
        # checkpoint remains loadable; the baseline never executes it.
        self.use_compact19 = False
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
        line_token_shifts=None,
    ):
        z_o8, atom_states = self.o8._forward_impl(
            data, atom_mask=atom_mask, use_star=False, use_md=False
        )
        glt = self.glt(
            data,
            endpoint_z_a=line_endpoint_z_a,
            endpoint_z_b=line_endpoint_z_b,
            observation_distances=line_observation_distances,
            observation_counts=line_observation_counts,
            token_shifts=line_token_shifts,
        )
        return {
            "z_o8": z_o8,
            "z_glt": glt["graph_geometry"],
            "atom_states": atom_states,
            **glt,
        }

    def _o8_canonical_states(self, data):
        z_o8, node_states = self.o8._forward_impl(data, use_star=False, use_md=False)
        _, canonical_states = self.o8._canonical_pool(node_states, data)
        return z_o8, canonical_states

    @staticmethod
    def _pool_canonical(data, canonical_states):
        graph_count = int(data.graph_available.numel())
        graph = scatter(
            canonical_states,
            data.canonical_graph_index.long(),
            dim=0,
            dim_size=graph_count,
            reduce="mean",
        )
        available = data.graph_available.bool().flatten().to(graph.dtype).unsqueeze(-1)
        return graph * available

    def forward_downstream(self, data, mode="o8_only"):
        valid_modes = {"o8_only", "o8_glt_atom"}
        if mode not in valid_modes:
            raise ValueError("MTS-GLT-v2 downstream mode is invalid")
        z_o8, canonical_states = self._o8_canonical_states(data)
        graph = z_o8
        if mode != "o8_only":
            glt = self.glt(data)
            update = self.atom_fusion_projection(
                self.atom_fusion_norm(glt["atom_geometry_states"])
            )
            valid = glt["atom_geometry_valid"].to(update.dtype).unsqueeze(-1)
            fused = canonical_states + valid * torch.tanh(self.atom_channel_gate) * update
            graph = self._pool_canonical(data, fused)
        graph = self.o8.md_residual(graph, data)
        available = data.graph_available.bool().flatten().to(graph.dtype).unsqueeze(-1)
        return graph * available

    def encode_views(self, data):
        z_o8, canonical_states = self._o8_canonical_states(data)
        glt = self.glt(data)
        update = self.atom_fusion_projection(
            self.atom_fusion_norm(glt["atom_geometry_states"])
        )
        valid = glt["atom_geometry_valid"].to(update.dtype).unsqueeze(-1)
        delta = valid * torch.tanh(self.atom_channel_gate) * update
        return {
            "z_o8": z_o8,
            "z_glt": glt["graph_geometry"],
            "atom_o8": canonical_states,
            "atom_glt": glt["atom_geometry_states"],
            "delta_atom_3d": delta,
            "valid_3d": data.glt_geometry_valid.bool().flatten(),
        }

    def forward(self, data, mode=None):
        return self.forward_downstream(
            data, mode=self.downstream_mode if mode is None else mode
        )


__all__ = ["CompactTrimerDescriptorResidual", "MTSGraphLineModelV2"]
