"""Multi-scale periodic non-bonded contact encoder for MTS-GLT-v2."""

from __future__ import annotations

import torch
from torch import nn
from torch_scatter import scatter

from .periodic_line_glt_v2 import LearnedGaussianMoments


class PeriodicSpatialContactEncoder(nn.Module):
    """One-layer contact message encoder independent of O8 and GLT states."""

    def __init__(self, hidden_size=512, atom_size=128, basis_size=256, dropout=0.1):
        super().__init__()
        if int(hidden_size) != 512:
            raise ValueError("MSContact fixes hidden_size=512")
        self.atom_embedding = nn.Embedding(120, atom_size)
        self.pair_atom_projection = nn.Linear(atom_size, hidden_size)
        self.distance_basis = LearnedGaussianMoments(
            basis_size, 0.0, 5.0, pair_conditioned=True
        )
        self.distance_mean_projection = nn.Linear(basis_size, hidden_size)
        self.distance_variance_projection = nn.Linear(basis_size, hidden_size)
        self.count_embedding = nn.Embedding(4, hidden_size)
        self.shift_embedding = nn.Embedding(3, hidden_size)
        self.pair_norm = nn.LayerNorm(hidden_size)
        self.message_mlp = nn.Sequential(
            nn.Linear(atom_size * 2 + hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)

    @staticmethod
    def _mean_by_atom(messages, targets, atom_count):
        if messages.numel() == 0:
            return messages.new_zeros((atom_count, messages.size(-1))), torch.zeros(
                atom_count, dtype=torch.bool, device=messages.device
            )
        aggregated = scatter(
            messages, targets, dim=0, dim_size=atom_count, reduce="mean"
        )
        degree = scatter(
            torch.ones_like(targets), targets, dim=0, dim_size=atom_count,
            reduce="sum",
        )
        return aggregated, degree > 0

    def forward(self, data, shell_mode="s4"):
        if shell_mode not in {"s4", "ms45", "c5_mixed"}:
            raise ValueError("spatial shell mode must be s4, ms45, or c5_mixed")
        atomic_numbers = data.canonical_atomic_numbers.long().clamp(0, 119)
        atom_count = int(atomic_numbers.numel())
        pair_index = data.spatial_pair_index.long()
        pair_valid = data.spatial_pair_valid.bool()
        atom_a, atom_b = pair_index[0], pair_index[1]
        z_a, z_b = atomic_numbers[atom_a], atomic_numbers[atom_b]
        observation_mask = data.spatial_obs_mask.bool()
        observation_count = observation_mask.sum(dim=1).long()
        if not torch.equal(observation_count, data.spatial_obs_count.long()):
            raise ValueError("spatial observation count/mask mismatch")
        # Pack valid observations for the shared count-based GBF implementation;
        # stable sorting preserves invariance to sidecar slot ordering.
        order = torch.argsort((~observation_mask).long(), dim=1, stable=True)
        observations = torch.gather(data.spatial_obs_distances, 1, order)
        mean, variance = self.distance_basis(
            observations,
            observation_count,
            z_a=z_a,
            z_b=z_b,
        )
        pair_state = self.pair_norm(
            self.pair_atom_projection(
                0.5 * (self.atom_embedding(z_a) + self.atom_embedding(z_b))
            )
            + self.distance_mean_projection(mean)
            + self.distance_variance_projection(variance)
            + self.count_embedding(observation_count.clamp(0, 3))
            + self.shift_embedding(data.spatial_pair_shift.long().abs().clamp(0, 2))
        )

        # A periodic self pair is represented once. Every other canonical pair
        # produces both directed messages without duplicating its pair state.
        non_self = ~data.spatial_periodic_self.bool()
        pair_rows = torch.cat(
            [torch.arange(atom_a.numel(), device=atom_a.device),
             torch.arange(atom_a.numel(), device=atom_a.device)[non_self]]
        )
        target = torch.cat([atom_a, atom_b[non_self]])
        source = torch.cat([atom_b, atom_a[non_self]])
        directed_valid = pair_valid[pair_rows]
        directed_shell = data.spatial_shell_id.long()[pair_rows]
        source_atom = self.atom_embedding(atomic_numbers[source])
        target_atom = self.atom_embedding(atomic_numbers[target])
        message = self.message_mlp(
            torch.cat(
                [target_atom, source_atom - target_atom, pair_state[pair_rows]],
                dim=-1,
            )
        )

        shell_states = []
        shell_present = []
        for shell_id in (0, 1):
            selected = directed_valid & (directed_shell == shell_id)
            aggregated, present = self._mean_by_atom(
                message[selected], target[selected], atom_count
            )
            projected = self.output_projection(self.output_norm(aggregated))
            shell_states.append(projected * present.to(projected.dtype).unsqueeze(-1))
            shell_present.append(present)

        core, outer = shell_states
        core_present, outer_present = shell_present
        if shell_mode == "s4":
            atom_state = core
            atom_present = core_present
        elif shell_mode == "ms45":
            denominator = (
                core_present.long() + outer_present.long()
            ).clamp_min(1).to(core.dtype).unsqueeze(-1)
            atom_state = (core + outer) / denominator
            atom_present = core_present | outer_present
        else:
            # C5-Mixed uses the exact same valid <5 A directed relations as
            # MS45, but performs one contact-neighbour mean before the shared
            # output norm/projection.  Shell identity has no numerical role.
            mixed_aggregated, atom_present = self._mean_by_atom(
                message[directed_valid], target[directed_valid], atom_count
            )
            atom_state = self.output_projection(self.output_norm(mixed_aggregated))
            atom_state = atom_state * atom_present.to(atom_state.dtype).unsqueeze(-1)

        graph_valid = data.spatial_graph_valid.bool().flatten()
        atom_graph_valid = graph_valid[data.canonical_graph_index.long()]
        atom_valid = atom_present & atom_graph_valid
        atom_state = atom_state * atom_valid.to(atom_state.dtype).unsqueeze(-1)
        graph_count = int(graph_valid.numel())
        graph_state = scatter(
            atom_state,
            data.canonical_graph_index.long(),
            dim=0,
            dim_size=graph_count,
            reduce="mean",
        )
        graph_state = graph_state * graph_valid.to(graph_state.dtype).unsqueeze(-1)
        return {
            "atom_spatial_states": atom_state,
            "atom_spatial_valid": atom_valid,
            "graph_spatial": graph_state,
            "spatial_core_states": core,
            "spatial_outer_states": outer,
            "spatial_core_present": core_present,
            "spatial_outer_present": outer_present,
            "spatial_distance_mean": mean,
            "spatial_distance_variance": variance,
        }


__all__ = ["PeriodicSpatialContactEncoder"]
