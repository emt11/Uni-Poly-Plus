"""Relation decoder for the B0 periodic coordinate denoising objective."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter_add


class PeriodicCoordinateDecoder(nn.Module):
    """Predict scalar relation coefficients and sum noisy displacements."""

    def __init__(self, dim=512, hidden=256):
        super().__init__()
        self.spd_embedding = nn.Embedding(3, 16)
        self.shift_embedding = nn.Embedding(3, 8)
        self.coefficient = nn.Sequential(
            nn.Linear(2 * int(dim) + 16 + 8, int(hidden)),
            nn.LayerNorm(int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), 1),
        )
        # At step 0 the decoder must be an exact noisy-coordinate baseline.
        # Keeping the final scalar projection zero-initialized preserves the
        # objective while allowing gradients to update it immediately.
        nn.init.zeros_(self.coefficient[-1].weight)
        nn.init.zeros_(self.coefficient[-1].bias)

    @staticmethod
    def _state_lookup(data):
        base = data.trimer_base_ru_atom_index.long().reshape(-1)
        offset = data.trimer_ru_offset.long().reshape(-1)
        code = base * 3 + (offset + 1).clamp(0, 2)
        order = torch.argsort(code)
        return code[order], order

    @staticmethod
    def _lookup(data, base, offset, sorted_codes, sorted_indices):
        query = base.long() * 3 + (offset.long() + 1).clamp(0, 2)
        if sorted_codes.numel() == 0:
            return torch.full_like(query, -1), torch.zeros_like(query, dtype=torch.bool)
        at = torch.searchsorted(sorted_codes, query)
        valid = at < sorted_codes.numel()
        safe = at.clamp_max(max(0, sorted_codes.numel() - 1))
        valid &= sorted_codes[safe] == query
        result = torch.full_like(query, -1)
        result[valid] = sorted_indices[safe[valid]]
        return result, valid

    def forward(self, data, node_states, noisy_positions):
        edge = data.lga_edge_index.long()
        source, target = edge[0], edge[1]
        relation_shift = data.lga_source_image_shift.long().reshape(-1)
        spd = data.lga_spd.long().reshape(-1)
        if relation_shift.numel() != source.numel():
            raise ValueError("B0 relation shift length mismatch")
        canonical_count = int(data.mips_x.size(0))
        if canonical_count == 0:
            return noisy_positions.new_zeros((0, 3)), torch.zeros(
                (source.numel(),), dtype=torch.bool, device=source.device
            )
        sorted_codes, sorted_indices = self._state_lookup(data)
        mapping = data.mips_to_trimer_central_index.long().reshape(-1)
        central_count = int(mapping.numel())
        safe_target = target.clamp(0, max(0, central_count - 1))
        safe_source = source.clamp(0, max(0, central_count - 1))
        target_trimer = mapping[safe_target]
        source_trimer = mapping[safe_source]
        central_valid = (
            (target >= 0) & (target < central_count)
            & (source >= 0) & (source < central_count)
            & (target_trimer >= 0) & (source_trimer >= 0)
        )
        safe_target_trimer = target_trimer.clamp(0, max(0, noisy_positions.size(0) - 1))
        safe_source_trimer = source_trimer.clamp(0, max(0, noisy_positions.size(0) - 1))
        target_base = data.trimer_base_ru_atom_index.long()[safe_target_trimer]
        source_base = data.trimer_base_ru_atom_index.long()[safe_source_trimer]

        # Relations are target=(i,0), source=(j,s).  For |s|=2 the open
        # Trimer only observes the outer pair, so both endpoints are shifted
        # by one RU in opposite directions.
        source_offset = relation_shift.clamp(-2, 2)
        target_offset = torch.zeros_like(source_offset)
        shift_two = relation_shift.abs() == 2
        target_offset = torch.where(
            shift_two, -relation_shift.sign(), target_offset
        )
        source_offset = torch.where(
            shift_two, relation_shift.sign(), source_offset
        )
        source_state, source_ok = self._lookup(
            data, source_base, source_offset, sorted_codes, sorted_indices
        )
        target_state, target_ok = self._lookup(
            data, target_base, target_offset, sorted_codes, sorted_indices
        )
        valid = (
            central_valid & source_ok & target_ok
            & (spd >= 0) & (spd <= 2)
            & (relation_shift >= -2) & (relation_shift <= 2)
            & ~((source == target) & (relation_shift == 0))
        )
        if hasattr(data, "trimer_geometry_valid"):
            graph_valid = (
                data.trimer_geometry_valid.bool()
                & data.trimer_geometry_is_3d.bool()
                & ~data.trimer_2d_fallback.bool()
            )
            graph_index = data.canonical_graph_index.long().reshape(-1)
            safe_graph_target = target.clamp(0, max(0, graph_index.numel() - 1))
            valid &= graph_valid[graph_index[safe_graph_target]]
        if not bool(valid.any()):
            return noisy_positions.new_zeros((canonical_count, 3)), valid
        source_safe = source_state.clamp(0, max(0, noisy_positions.size(0) - 1))
        target_safe = target_state.clamp(0, max(0, noisy_positions.size(0) - 1))
        relative = noisy_positions[source_safe] - noisy_positions[target_safe]
        valid &= torch.isfinite(relative).all(dim=-1)
        relative = torch.where(valid.unsqueeze(-1), relative, 0)
        relation_features = torch.cat([
            node_states[target], node_states[source],
            self.spd_embedding(spd.clamp(0, 2)),
            self.shift_embedding(relation_shift.abs().clamp_max(2)),
        ], dim=-1)
        coefficient = self.coefficient(relation_features).squeeze(-1)
        displacement = coefficient.unsqueeze(-1) * relative
        displacement = displacement * valid.unsqueeze(-1).to(displacement.dtype)
        return scatter_add(
            displacement, target, dim=0, dim_size=canonical_count
        ), valid


__all__ = ["PeriodicCoordinateDecoder"]
