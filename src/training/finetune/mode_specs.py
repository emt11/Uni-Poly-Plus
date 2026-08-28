"""Capability table for the retained MTS-GLT-v2 downstream modes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModeSpec:
    needs_glt: bool = False
    required_batch_fields: tuple[str, ...] = ()


_GLT_FIELDS = (
    "glt_token_atom_a", "glt_token_atom_b", "glt_token_endpoint_z_a",
    "glt_token_endpoint_z_b", "glt_token_observation_distances",
    "glt_token_observation_count", "glt_token_shift", "glt_token_valid",
    "glt_relation_source", "glt_relation_target",
    "glt_relation_observation_angles", "glt_relation_observation_count",
    "glt_relation_multiplicity", "glt_relation_is_fallback",
    "glt_relation_valid", "glt_geometry_valid", "canonical_graph_index",
)


MODE_SPECS = {
    "o8_only": ModeSpec(),
    "o8_glt_atom": ModeSpec(needs_glt=True, required_batch_fields=_GLT_FIELDS),
}


def mode_spec(name: str) -> ModeSpec:
    try:
        return MODE_SPECS[str(name)]
    except KeyError as error:
        raise ValueError(f"unsupported MTS GLT mode: {name}") from error


def missing_batch_fields(name: str, batch) -> tuple[str, ...]:
    return tuple(
        field for field in mode_spec(name).required_batch_fields
        if not hasattr(batch, field)
    )


__all__ = ["MODE_SPECS", "ModeSpec", "missing_batch_fields", "mode_spec"]
