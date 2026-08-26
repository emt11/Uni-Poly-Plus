"""Small consistency table for supported MTS GLT downstream modes.

This module deliberately does not construct models or import optional features.
It only centralises mode capabilities used by config and batch validation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModeSpec:
    needs_glt: bool = False
    needs_torsion: bool = False
    needs_joint_ra: bool = False
    needs_bond_type: bool = False
    needs_x2l: bool = False
    needs_x2a: bool = False
    required_batch_fields: tuple[str, ...] = ()
    trainable_optional_modules: tuple[str, ...] = ()


_GLT_FIELDS = (
    "glt_token_atom_a", "glt_token_atom_b", "glt_token_endpoint_z_a",
    "glt_token_endpoint_z_b", "glt_token_observation_distances",
    "glt_token_observation_count", "glt_token_shift", "glt_token_valid",
    "glt_relation_source", "glt_relation_target",
    "glt_relation_observation_angles", "glt_relation_observation_count",
    "glt_relation_multiplicity", "glt_relation_is_fallback",
    "glt_relation_valid", "glt_geometry_valid", "canonical_graph_index",
)

_TORSION_FIELDS = (
    "glt_torsion_observation_value", "glt_torsion_observation_relation",
    "glt_relation_torsion_count",
    "glt_relation_torsion_source_cross_ru",
)

_JOINT_RA_FIELDS = (
    "glt_relation_source_distances", "glt_relation_source_distance_valid",
    "glt_relation_source_cross_ru",
)

MODE_SPECS = {
    "none": ModeSpec(),
    "o8_only": ModeSpec(),
    "o8_glt": ModeSpec(needs_glt=True, required_batch_fields=_GLT_FIELDS),
    "o8_glt_atom": ModeSpec(needs_glt=True, required_batch_fields=_GLT_FIELDS),
    "o8_glt_atom_desc": ModeSpec(needs_glt=True, required_batch_fields=_GLT_FIELDS),
    "o8_glt_graph": ModeSpec(needs_glt=True, required_batch_fields=_GLT_FIELDS),
    "o8_glt_graph_mean": ModeSpec(needs_glt=True, required_batch_fields=_GLT_FIELDS),
    "o8_glt_atom_central": ModeSpec(needs_glt=True, required_batch_fields=_GLT_FIELDS),
    "o8_glt_atom_self3d": ModeSpec(
        needs_glt=True, required_batch_fields=_GLT_FIELDS,
        trainable_optional_modules=("atom_self3d",),
    ),
    "o8_glt_atom_x23": ModeSpec(
        needs_glt=True, required_batch_fields=_GLT_FIELDS,
        trainable_optional_modules=("atom_x23",),
    ),
    "o8_glt_atom_line_self": ModeSpec(
        needs_glt=True, needs_x2l=True, required_batch_fields=_GLT_FIELDS,
        trainable_optional_modules=("line_conditioning",),
    ),
    "o8_glt_atom_line_x2l": ModeSpec(
        needs_glt=True, needs_x2l=True, required_batch_fields=_GLT_FIELDS,
        trainable_optional_modules=("line_conditioning",),
    ),
    "o8_glt_atom_attn_self": ModeSpec(
        needs_glt=True, needs_x2a=True, required_batch_fields=_GLT_FIELDS,
        trainable_optional_modules=("attention_routing",),
    ),
    "o8_glt_atom_attn_x2a": ModeSpec(
        needs_glt=True, needs_x2a=True, required_batch_fields=_GLT_FIELDS,
        trainable_optional_modules=("attention_routing",),
    ),
    "o8_glt_atom_torsion_count": ModeSpec(
        needs_glt=True, needs_torsion=True,
        required_batch_fields=_GLT_FIELDS + _TORSION_FIELDS,
        trainable_optional_modules=("torsion_bias",),
    ),
    "o8_glt_atom_torsion": ModeSpec(
        needs_glt=True, needs_torsion=True,
        required_batch_fields=_GLT_FIELDS + _TORSION_FIELDS,
        trainable_optional_modules=("torsion_bias",),
    ),
    "o8_glt_atom_sbf_angle_control": ModeSpec(
        needs_glt=True, needs_joint_ra=True,
        required_batch_fields=_GLT_FIELDS + _JOINT_RA_FIELDS,
        trainable_optional_modules=("joint_radial_angular_bias",),
    ),
    "o8_glt_atom_sbf_radial_angle": ModeSpec(
        needs_glt=True, needs_joint_ra=True,
        required_batch_fields=_GLT_FIELDS + _JOINT_RA_FIELDS,
        trainable_optional_modules=("joint_radial_angular_bias",),
    ),
}


def mode_spec(name: str) -> ModeSpec:
    try:
        return MODE_SPECS[str(name)]
    except KeyError as error:
        raise ValueError(f"unsupported MTS GLT mode: {name}") from error


def missing_batch_fields(name: str, batch) -> tuple[str, ...]:
    return tuple(field for field in mode_spec(name).required_batch_fields if not hasattr(batch, field))
