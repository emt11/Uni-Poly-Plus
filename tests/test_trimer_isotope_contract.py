"""Explicit isotope-H identity contract tests (Trimer content v10).

Source/base identity is every non-dummy explicit atom of the canonical
P-SMILES, including ``[H]``/``[2H]``/``[3H]``.  Heavy is strictly ``Z > 1``.
These tests pin the decoupling of the two concepts and the heavy-only 3D
projection downstream.
"""

import copy

import pytest
import torch
from rdkit import Chem

from src.dataset.dataset import _compute_ru_base_layer, _compute_topology_layer
from src.dataset.graph_data import build_periodic_multimer_mol
from src.dataset.periodic_line_glt import _prepare_trimer
from src.dataset.periodic_line_glt_complete import build_complete_trimer_glt_sample
from src.dataset.trimer_mcl import (
    attach_finite_trimer_mcl,
)
from scripts.validate_dual_glt import renumber_frozen


BLOCKED_SMILES = (
    "*CC(*)(C)C(=O)OCC([2H])(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)"
    "C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
)
BLOCKED_KEY = "bf7f4353aad01ae9c1739cf33911d9c04924b4c5ee6049c8c41248392f317694"


def _topology(smiles):
    ru = _compute_ru_base_layer(smiles)
    top = _compute_topology_layer(smiles, ru, max_hops=2)
    top.smiles = smiles
    return top


def _attach(smiles, key, num_candidates=2, max_rounds=1):
    return attach_finite_trimer_mcl(
        _topology(smiles), smiles, num_candidates=num_candidates,
        max_rounds=max_rounds, sample_key=key,
    )


def _source_states(trimer):
    return {
        (int(base), int(offset)): (int(z), int(isotope))
        for base, offset, z, isotope in zip(
            trimer.trimer_base_ru_atom_id.tolist(),
            trimer.trimer_ru_offset.tolist(),
            trimer.trimer_atomic_number.tolist(),
            trimer.trimer_isotope.tolist(),
        )
    }


def _central_deuterium(result):
    row = torch.arange(result.trimer_atomic_number.numel())
    deuteriums = torch.nonzero(
        (result.trimer_atomic_number == 1)
        & (result.trimer_isotope == 2)
        & (row < int(result.trimer_source_atom_count)),
        as_tuple=False,
    ).flatten()
    return deuteriums, int(deuteriums[result.trimer_ru_offset[deuteriums] == 0][0])


def test_explicit_deuterium_is_source_identity_and_survives_replication():
    result = _attach("*CC([2H])C*", "isotope-contract-2h")
    assert result.trimer_geometry_valid
    deuteriums, _ = _central_deuterium(result)
    assert deuteriums.numel() == 3  # one per RU copy
    assert bool(result.is_source_atom[deuteriums].all())
    assert not bool(result.is_added_h[deuteriums].any())
    assert bool(result.is_source_explicit_h[deuteriums].all())
    assert not bool(result.trimer_heavy_mask[deuteriums].any())  # Z==1, not heavy
    parents = result.h_parent_heavy_index[deuteriums]
    assert bool(result.trimer_heavy_mask[parents].all())
    assert torch.equal(
        result.trimer_ru_offset[deuteriums], result.trimer_ru_offset[parents]
    )


def test_blocked_sample_counts_35_source_34_heavy_1_deuterium():
    molecule, metadata = build_periodic_multimer_mol(
        BLOCKED_SMILES, 3, close_periodic=False
    )
    assert metadata["base_atom_count"] == 35
    assert molecule.GetNumAtoms() == 105
    atomic = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
    isotopes = [atom.GetIsotope() for atom in molecule.GetAtoms()]
    assert sum(1 for z in atomic if z > 1) == 102
    assert sum(1 for z, iso in zip(atomic, isotopes) if z == 1 and iso == 2) == 3
    with_h = Chem.AddHs(Chem.Mol(molecule))
    assert with_h.GetNumAtoms() == 128
    added = [with_h.GetAtomWithIdx(i) for i in range(105, with_h.GetNumAtoms())]
    assert all(atom.GetAtomicNum() == 1 and atom.GetIsotope() == 0 for atom in added)


def test_o8_to_trimer_atom_targets_the_real_deuterium():
    result = _attach("*CC([2H])C*", "isotope-contract-mapping")
    deuteriums, central_d = _central_deuterium(result)
    node = torch.nonzero(
        result.o8_to_trimer_atom == central_d, as_tuple=False
    ).flatten()
    assert node.numel() == 1
    assert bool(result.o8_heavy_mask[node[0]]) is False
    assert bool(result.is_source_explicit_h[central_d])
    assert not bool(result.is_added_h[central_d])
    assert torch.equal(
        result.o8_heavy_mask, result.trimer_heavy_mask[result.o8_to_trimer_atom]
    )


def test_addhs_generated_h_is_distinguished_from_source_explicit_h():
    result = _attach("*CC([2H])C*", "isotope-contract-addhs")
    source_count = int(result.trimer_source_atom_count)
    assert source_count == int(result.is_source_atom.sum())
    added = torch.nonzero(result.is_added_h, as_tuple=False).flatten()
    assert added.numel() == result.trimer_atomic_number.numel() - source_count
    assert bool((result.trimer_atomic_number[added] == 1).all())
    assert bool((result.trimer_isotope[added] == 0).all())
    assert not bool(result.is_source_explicit_h[added].any())
    source_h = torch.nonzero(result.is_source_explicit_h, as_tuple=False).flatten()
    assert not bool(result.is_added_h[source_h].any())


def test_explicit_ordinary_h_is_canonicalized_away():
    """Plain ``[H]`` carries no isotope information; RDKit's canonical SMILES
    dissolves it into an implicit hydrogen, so the canonical source identity
    contains no explicit node for it.  Only isotope H survives as an explicit
    source atom (verified against RDKit actual behaviour)."""
    from src.dataset.trimer_mcl import _canonical_source_mol
    canonical = _canonical_source_mol("*C([H])(C)C*")
    assert not any(
        atom.GetAtomicNum() == 1 and atom.GetIsotope() == 0
        for atom in canonical.GetAtoms()
    )
    result = _attach("*C([H])(C)C*", "isotope-contract-h")
    assert result.trimer_geometry_valid
    assert int(result.is_source_explicit_h.sum()) == 0
    assert result.trimer_geometry_valid == _attach(
        "*CC(C)C*", "isotope-contract-h-equivalent"
    ).trimer_geometry_valid


def test_explicit_tritium_contract():
    result = _attach("*C([3H])(C)C*", "isotope-contract-t")
    assert result.trimer_geometry_valid
    tritiums = torch.nonzero(
        result.trimer_isotope == 3, as_tuple=False
    ).flatten()
    assert tritiums.numel() == 3
    assert bool(result.is_source_atom[tritiums].all())
    assert not bool(result.trimer_heavy_mask[tritiums].any())


def test_heavy_only_downstream_excludes_deuterium_but_keeps_canonical_mapping():
    smiles = "*CC([2H])C*"
    topology = _topology(smiles)
    result = _attach(smiles, "isotope-contract-downstream")
    assert result.trimer_geometry_valid
    row = build_complete_trimer_glt_sample(topology, result, smiles)
    assert row["geometry_valid"], row["invalid_reason"]
    tokens = row["tokens"]
    assert bool((tokens["token_endpoint_z_a"] > 1).all())
    assert bool((tokens["token_endpoint_z_b"] > 1).all())
    info, reason = _prepare_trimer(result, topology)
    assert info is not None, reason
    deuteriums, central_d = _central_deuterium(result)
    deuterium_base = int(result.trimer_base_ru_atom_id[central_d])
    assert (deuterium_base, 0) not in info["state_to_local"]
    node = torch.nonzero(
        result.o8_to_trimer_atom == central_d, as_tuple=False
    ).flatten()
    assert node.numel() == 1  # canonical mapping still targets the D atom


def test_renumbering_preserves_isotope_source_parent_and_ru_semantics():
    smiles = "*CC([2H])C*"
    topology = _topology(smiles)
    result = _attach(smiles, "isotope-contract-renumber")
    order = torch.arange(result.trimer_pos.size(0) - 1, -1, -1)
    inverse = torch.argsort(order)
    renamed = copy.copy(result)
    for name in (
        "trimer_atomic_number", "trimer_isotope", "trimer_ru_offset",
        "trimer_base_ru_atom_id", "trimer_heavy_mask", "is_source_atom",
        "is_source_explicit_h", "is_added_h",
    ):
        setattr(renamed, name, getattr(result, name)[order])
    renamed.h_parent_heavy_index = result.h_parent_heavy_index[order]
    present = renamed.h_parent_heavy_index >= 0
    renamed.h_parent_heavy_index[present] = inverse[
        renamed.h_parent_heavy_index[present]
    ]
    renamed.o8_to_trimer_atom = inverse[result.o8_to_trimer_atom]
    renamed.o8_heavy_mask = result.trimer_heavy_mask[renamed.o8_to_trimer_atom]
    assert _source_states(result) == _source_states(renamed)
    deuteriums = torch.nonzero(renamed.trimer_isotope == 2, as_tuple=False).flatten()
    parents = renamed.h_parent_heavy_index[deuteriums]
    assert bool(renamed.trimer_heavy_mask[parents].all())
    assert torch.equal(
        renamed.trimer_ru_offset[deuteriums], renamed.trimer_ru_offset[parents]
    )
    dual = renumber_frozen(topology, result, smiles)
    assert bool(dual.geometry_valid)


def test_isotope_identity_survives_canonical_roundtrip():
    from src.dataset.trimer_mcl import _canonical_source_mol
    canonical = _canonical_source_mol(BLOCKED_SMILES)
    isotopes = sorted(
        atom.GetIsotope() for atom in canonical.GetAtoms() if atom.GetIsotope() > 0
    )
    assert isotopes == [2]
