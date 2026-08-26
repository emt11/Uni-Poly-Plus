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
        if hasattr(data, "glt_compact19"):
            values = data.glt_compact19.to(graph.dtype)
            valid = data.glt_compact19_valid.bool()
        else:
            from .compact_trimer_descriptors import compact_trimer_descriptors
            values, valid = compact_trimer_descriptors(data)
            values = values.to(graph.dtype)
        valid = valid.to(graph.dtype).unsqueeze(-1)
        return graph + valid * torch.tanh(self.gate) * self.network(values)


class Atom3DConditionalUpdate(nn.Module):
    """Parameter-matched downstream-only refinement of canonical GLT atoms."""

    def __init__(self, hidden_size=512):
        super().__init__()
        self.condition_norm = nn.LayerNorm(hidden_size)
        self.condition_projection = nn.Linear(hidden_size, hidden_size)
        self.content_norm = nn.LayerNorm(hidden_size)
        self.content_projection = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, condition_source, geometry_state, valid):
        condition = torch.sigmoid(
            self.condition_projection(self.condition_norm(condition_source.detach()))
        )
        content = self.content_projection(self.content_norm(geometry_state))
        valid_scale = valid.to(content.dtype).unsqueeze(-1)
        delta = valid_scale * torch.tanh(self.gate) * condition * content
        return geometry_state + delta, delta, condition, content


class MTSGraphLineModelV2(nn.Module):
    architecture_name = "MIPS-Trimer-GLT-v2"

    def __init__(
        self,
        hidden_size=512,
        glt_layers=12,
        glt_heads=8,
        dropout=0.1,
        glt_attention_variant="mips",
        use_compact19=False,
        interaction_mode="none",
        line_conditioning_mode="none",
        attention_conditioning_mode="none",
        torsion_mode="none",
        o8_bond_bias_mode="none",
        joint_basis_mode="none",
        glt_metadata_mode="full",
    ):
        super().__init__()
        if int(hidden_size) != 512:
            raise ValueError("MTS-GLT-v2 fixes hidden_size=512")
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
            direct_bond_bias_mode=o8_bond_bias_mode,
        )
        self.glt = LocalPeriodicGraphLineTransformerV2(
            hidden_size=hidden_size,
            layers=glt_layers,
            heads=glt_heads,
            dropout=dropout,
            attention_variant=glt_attention_variant,
            line_conditioning_mode=line_conditioning_mode,
            attention_conditioning_mode=attention_conditioning_mode,
            torsion_mode=torsion_mode,
            joint_basis_mode=joint_basis_mode,
            metadata_mode=glt_metadata_mode,
        )
        self.atom_fusion_norm = nn.LayerNorm(hidden_size)
        self.atom_fusion_projection = nn.Linear(hidden_size, hidden_size)
        self.atom_channel_gate = nn.Parameter(
            torch.full((hidden_size,), math.atanh(0.05))
        )
        self.compact19_residual = CompactTrimerDescriptorResidual(hidden_size)
        if interaction_mode not in {"none", "self3d", "x23"}:
            raise ValueError("invalid MTS-GLT-v2 interaction mode")
        self.interaction_mode = str(interaction_mode)
        self.line_conditioning_mode = str(line_conditioning_mode)
        self.attention_conditioning_mode = str(attention_conditioning_mode)
        self.torsion_mode = str(torsion_mode)
        self.joint_basis_mode = str(joint_basis_mode)
        self.glt_metadata_mode = str(glt_metadata_mode)
        if self.interaction_mode != "none" and self.line_conditioning_mode != "none":
            raise ValueError("atom and line conditioning modes are mutually exclusive")
        if sum(mode != "none" for mode in (
            self.interaction_mode, self.line_conditioning_mode,
            self.attention_conditioning_mode, self.torsion_mode,
            self.joint_basis_mode,
        )) > 1:
            raise ValueError("downstream conditioning modes are mutually exclusive")
        self.interaction_update = (
            Atom3DConditionalUpdate(hidden_size)
            if self.interaction_mode != "none" else None
        )
        self.use_compact19 = bool(use_compact19)
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

    def _updated_geometry_states(self, canonical_states, glt):
        geometry = glt["atom_geometry_states"]
        valid = glt["atom_geometry_valid"].bool()
        if self.interaction_update is None:
            return geometry, None
        source = geometry if self.interaction_mode == "self3d" else canonical_states
        updated, delta, condition, content = self.interaction_update(
            source, geometry, valid
        )
        return updated, {
            "h3": geometry,
            "h3_new": updated,
            "delta3": delta,
            "condition": condition,
            "content": content,
            "valid": valid,
        }

    def forward_downstream(self, data, mode="o8_only"):
        valid_modes = {
            "o8_only", "o8_glt_atom", "o8_glt_atom_desc",
            "o8_glt_atom_self3d", "o8_glt_atom_x23",
            "o8_glt_atom_line_self", "o8_glt_atom_line_x2l",
            "o8_glt_atom_attn_self", "o8_glt_atom_attn_x2a",
            "o8_glt_atom_torsion_count", "o8_glt_atom_torsion",
            "o8_glt_atom_sbf_angle_control",
            "o8_glt_atom_sbf_radial_angle",
        }
        if mode not in valid_modes:
            raise ValueError("MTS-GLT-v2 downstream mode is invalid")
        z_o8, canonical_states = self._o8_canonical_states(data)
        graph = z_o8
        if mode != "o8_only":
            glt = self.glt(data, canonical_atom_states=canonical_states)
            geometry_states, _ = self._updated_geometry_states(
                canonical_states, glt
            )
            update = self.atom_fusion_projection(
                self.atom_fusion_norm(geometry_states)
            )
            valid = glt["atom_geometry_valid"].to(update.dtype).unsqueeze(-1)
            fused = canonical_states + valid * torch.tanh(self.atom_channel_gate) * update
            graph = self._pool_canonical(data, fused)
        graph = self.o8.md_residual(graph, data)
        if mode == "o8_glt_atom_desc":
            if not self.use_compact19:
                raise ValueError("compact19 mode requires descriptor branch")
            graph = self.compact19_residual(graph, data)
        available = data.graph_available.bool().flatten().to(graph.dtype).unsqueeze(-1)
        return graph * available

    def interaction_views(self, data):
        if self.interaction_update is None:
            raise ValueError("interaction diagnostics require self3d or x23")
        _, canonical_states = self._o8_canonical_states(data)
        glt = self.glt(data)
        _, views = self._updated_geometry_states(canonical_states, glt)
        return views

    def line_conditioning_views(self, data):
        if self.glt.line_conditioning_projection is None:
            raise ValueError("line conditioning diagnostics require SL or X2L")
        _, canonical_states = self._o8_canonical_states(data)
        glt = self.glt(data, canonical_atom_states=canonical_states)
        return glt["line_conditioning"]

    def attention_conditioning_views(self, data):
        if self.glt.attention_conditioning_projection is None:
            raise ValueError("attention routing diagnostics require SA or X2A")
        _, canonical_states = self._o8_canonical_states(data)
        glt = self.glt(
            data, canonical_atom_states=canonical_states,
            return_attention_conditioning=True,
        )
        return glt["attention_conditioning"]

    def torsion_views(self, data):
        if self.glt.torsion_bias is None:
            raise ValueError("torsion diagnostics require TC or TG")
        _, canonical_states = self._o8_canonical_states(data)
        glt = self.glt(
            data, canonical_atom_states=canonical_states, return_torsion=True
        )
        return glt["torsion"]

    def joint_basis_views(self, data):
        if self.glt.joint_basis_bias is None:
            raise ValueError("joint basis diagnostics require AC or RA")
        _, canonical_states = self._o8_canonical_states(data)
        glt = self.glt(
            data, canonical_atom_states=canonical_states,
            return_joint_basis=True,
        )
        return glt["joint_basis"]

    def encode_views(self, data):
        z_o8, canonical_states = self._o8_canonical_states(data)
        glt = self.glt(data, canonical_atom_states=canonical_states)
        update = self.atom_fusion_projection(
            self.atom_fusion_norm(glt["atom_geometry_states"])
        )
        valid = glt["atom_geometry_valid"].to(update.dtype).unsqueeze(-1)
        delta = valid * torch.tanh(self.atom_channel_gate) * update
        projected_graph = self._pool_canonical(data, valid * update)
        delta_graph = self._pool_canonical(data, delta)
        return {
            "z_o8": z_o8,
            "z_glt": glt["graph_geometry"],
            "projected_z_glt": projected_graph,
            "delta_z_3d": delta_graph,
            "atom_o8": canonical_states,
            "atom_glt": glt["atom_geometry_states"],
            "delta_atom_3d": delta,
            "valid_3d": data.glt_geometry_valid.bool().flatten(),
        }

    def forward(self, data, mode=None):
        return self.forward_downstream(
            data, mode=self.downstream_mode if mode is None else mode
        )


__all__ = [
    "Atom3DConditionalUpdate", "CompactTrimerDescriptorResidual",
    "MTSGraphLineModelV2",
]
