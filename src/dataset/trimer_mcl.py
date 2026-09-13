"""Deterministic finite-Trimer geometry for the selected O8 route.

The Trimer is an open, H-terminated local geometry proxy.  It is deliberately
separate from the O8 star-linked graph: O8 repeat copies are mapped to the
central repeat unit only through their canonical RU atom identity.

MMFF94 is used for a fixed-budget local coordinate relaxation only.  Converged
candidates are preferred, while a finite non-converged candidate remains an
explicitly labelled fallback.  The cached payload retains every explicit H.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from .graph_data import build_periodic_multimer_mol
from .mips_trimer_contract import (
    TRIMER_BUILDER_VERSION,
    TRIMER_CONTENT_SCHEMA,
    TRIMER_MMFF_RELAX_MAX_ITERATIONS as CONTRACT_MMFF_RELAX_MAX_ITERATIONS,
    TRIMER_PROTOCOL,
    TRIMER_SCHEMA_VERSION,
)


TRIMER_MCL_SCHEMA = TRIMER_CONTENT_SCHEMA
TRIMER_MCL_SCHEMA_VERSION = TRIMER_SCHEMA_VERSION
TRIMER_MCL_PROTOCOL = TRIMER_PROTOCOL
TRIMER_MCL_BUILDER_VERSION = TRIMER_BUILDER_VERSION
TRIMER_ETKDG_TIMEOUT_SECONDS = 60
TRIMER_ETKDG_MAX_ITERATIONS = 200
TRIMER_MMFF_RELAX_MAX_ITERATIONS = CONTRACT_MMFF_RELAX_MAX_ITERATIONS
TRIMER_CANDIDATES_PER_ROUND = 8
TRIMER_MAX_ROUNDS = 2
TRIMER_SAMPLE_TIMEOUT_SECONDS = 240.0

_ETKDG_CALL_COUNT = 0
_ETKDG_CALL_LOCK = threading.Lock()


def reset_etkdg_call_counter():
    global _ETKDG_CALL_COUNT
    with _ETKDG_CALL_LOCK:
        _ETKDG_CALL_COUNT = 0


def get_etkdg_call_count():
    with _ETKDG_CALL_LOCK:
        return int(_ETKDG_CALL_COUNT)


def _count_etkdg_call():
    global _ETKDG_CALL_COUNT
    with _ETKDG_CALL_LOCK:
        _ETKDG_CALL_COUNT += 1


def _sample_seed(smiles_or_mol) -> int:
    molecule = (
        Chem.Mol(smiles_or_mol)
        if isinstance(smiles_or_mol, Chem.Mol)
        else Chem.MolFromSmiles(str(smiles_or_mol))
    )
    identity = (
        Chem.MolToSmiles(molecule, canonical=True)
        if molecule is not None else str(smiles_or_mol)
    )
    digest = hashlib.sha256(
        f"{TRIMER_MCL_SCHEMA}:{identity}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def _canonical_source_mol(smiles_or_mol) -> Chem.Mol:
    source = (
        Chem.Mol(smiles_or_mol)
        if isinstance(smiles_or_mol, Chem.Mol)
        else Chem.MolFromSmiles(str(smiles_or_mol))
    )
    if source is None:
        raise TrimerContractError("invalid_polymer_smiles")
    canonical_smiles = Chem.MolToSmiles(source, canonical=True, isomericSmiles=True)
    canonical = Chem.MolFromSmiles(canonical_smiles)
    if canonical is None:
        raise TrimerContractError("canonical_polymer_smiles_reparse_failed")
    return canonical


def _bond_code(bond: Chem.Bond) -> int:
    value = bond.GetBondType()
    if value == Chem.rdchem.BondType.SINGLE:
        return 1
    if value == Chem.rdchem.BondType.DOUBLE:
        return 2
    if value == Chem.rdchem.BondType.TRIPLE:
        return 3
    if value == Chem.rdchem.BondType.AROMATIC:
        return 4
    return 5


def _embed_attempt(
    molecule,
    *,
    num_candidates: int,
    seed: int,
    use_random_coords: bool,
    max_iterations: int,
):
    """Run one deterministic ETKDG attempt on a fresh molecule copy."""

    _count_etkdg_call()

    candidate = Chem.Mol(molecule)
    candidate.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = 1
    params.useRandomCoords = bool(use_random_coords)
    params.enforceChirality = True
    params.maxIterations = int(max_iterations)
    params.timeout = TRIMER_ETKDG_TIMEOUT_SECONDS
    params.pruneRmsThresh = -1.0
    embedded = AllChem.EmbedMultipleConfs(
        candidate, numConfs=int(num_candidates), params=params
    )
    conformer_ids = [
        int(conf_id) for conf_id in embedded if int(conf_id) >= 0
    ]
    return candidate, conformer_ids


def _conformer_coordinates_are_finite_3d(
    mol: Chem.Mol,
    conf_id: int,
    num_atoms: int,
) -> bool:
    """Check that a conformer exists, is 3-D, and has all-finite positions."""

    try:
        conformer = mol.GetConformer(int(conf_id))
    except (ValueError, RuntimeError):
        return False
    if conformer is None or not bool(conformer.Is3D()):
        return False
    for idx in range(num_atoms):
        pos = conformer.GetAtomPosition(idx)
        if not all(
            torch.isfinite(torch.tensor(float(coord)))
            for coord in (pos.x, pos.y, pos.z)
        ):
            return False
    return True


def _calculate_mmff_energy(
    mol: Chem.Mol,
    properties,
    conf_id: int,
) -> float | None:
    """Compute current MMFF94 energy for a single conformer.

    Returns ``None`` when the force field cannot be created or energy is
    non-finite.
    """

    try:
        force_field = AllChem.MMFFGetMoleculeForceField(
            mol, properties, confId=int(conf_id)
        )
    except (ValueError, RuntimeError):
        return None
    if force_field is None:
        return None
    try:
        energy = float(force_field.CalcEnergy())
    except (ValueError, RuntimeError):
        return None
    if not bool(torch.isfinite(torch.tensor(energy))):
        return None
    return energy


# ---------------------------------------------------------------------------
#  Placeholder / unavailable
# ---------------------------------------------------------------------------

def _attach_placeholder(data, reason: str, seed: int = 0):
    node_count = int(data.num_nodes)
    data.trimer_pos = torch.empty((0, 3), dtype=torch.float)
    data.trimer_atomic_number = torch.empty((0,), dtype=torch.long)
    data.trimer_atomic_numbers = data.trimer_atomic_number
    data.trimer_atom_id = torch.empty((0,), dtype=torch.long)
    data.trimer_edge_index = torch.empty((2, 0), dtype=torch.long)
    data.trimer_bond_type = torch.empty((0,), dtype=torch.long)
    data.trimer_bond_aromatic = torch.empty((0,), dtype=torch.bool)
    data.trimer_formal_charge = torch.empty((0,), dtype=torch.long)
    data.trimer_is_aromatic = torch.empty((0,), dtype=torch.bool)
    data.trimer_chiral_tag = torch.empty((0,), dtype=torch.long)
    data.trimer_attachment_role = torch.empty((0,), dtype=torch.long)
    data.trimer_internal_degree = torch.empty((0,), dtype=torch.long)
    data.trimer_bonds = torch.empty((2, 0), dtype=torch.long)
    data.trimer_bond_types = torch.empty((0,), dtype=torch.long)
    data.trimer_base_ru_atom_id = torch.empty((0,), dtype=torch.long)
    data.trimer_base_ru_atom_index = data.trimer_base_ru_atom_id
    data.trimer_ru_offset = torch.empty((0,), dtype=torch.long)
    data.trimer_central_ru_mask = torch.empty((0,), dtype=torch.bool)
    data.trimer_central_atom_index = torch.empty((0,), dtype=torch.long)
    data.trimer_central_ru_atom_index = data.trimer_central_atom_index
    data.mips_to_trimer_central_index = torch.full(
        (node_count,), -1, dtype=torch.long
    )
    data.o8_to_trimer_atom = data.mips_to_trimer_central_index
    data.o8_heavy_mask = torch.empty((0,), dtype=torch.bool)
    data.o8_heavy_indices = torch.empty((0,), dtype=torch.long)
    data.trimer_heavy_mask = torch.empty((0,), dtype=torch.bool)
    data.trimer_heavy_indices = torch.empty((0,), dtype=torch.long)
    data.trimer_isotope = torch.empty((0,), dtype=torch.long)
    data.trimer_source_atom_count = torch.zeros((), dtype=torch.long)
    data.is_source_atom = torch.empty((0,), dtype=torch.bool)
    data.is_source_explicit_h = torch.empty((0,), dtype=torch.bool)
    data.is_added_h = torch.empty((0,), dtype=torch.bool)
    data.h_parent_heavy_index = torch.empty((0,), dtype=torch.long)
    data.trimer_geometry_valid = False
    data.trimer_geometry_is_3d = torch.tensor(False, dtype=torch.bool)
    data.trimer_2d_fallback = torch.tensor(False, dtype=torch.bool)
    data.trimer_geometry_source = "unavailable"
    data.star_3d_distance = torch.tensor(0.0, dtype=torch.float)
    data.star_3d_asymmetry = torch.tensor(float("inf"), dtype=torch.float)
    data.star_3d_valid = torch.tensor(False, dtype=torch.bool)
    data.trimer_failure_code = str(reason)[:240]
    data.trimer_conformer_energy = float("nan")
    data.trimer_conformer_seed = int(seed)
    data.trimer_conformer_method = TRIMER_MCL_PROTOCOL
    data.trimer_mcl_schema = TRIMER_MCL_SCHEMA
    data.trimer_mcl_schema_version = TRIMER_MCL_SCHEMA_VERSION
    data.trimer_conformer_round_id = -1
    data.trimer_conformer_candidate_id = -1
    data.selected_converged = False
    data.search_stop_reason = str(reason)[:240]
    data.generation_diagnostics = {
        "num_rounds": 0,
        "num_candidates_requested": 0,
        "num_candidates_embedded": 0,
        "num_pre_stereo_rejected": 0,
        "num_post_stereo_rejected": 0,
        "num_geometry_rejected": 0,
        "num_valid_candidates": 0,
        "num_mmff_attempted": 0,
        "num_mmff_converged": 0,
        "rounds": [],
        "embed_time": 0.0,
        "mmff_time": 0.0,
        "stereo_time": 0.0,
        "total_time": 0.0,
        "trimer_source_atoms": 0,
        "trimer_heavy_atoms": 0,
        "trimer_atoms_with_h": 0,
    }
    data.multi_conformer = False
    return data


def _finish_geometry_failure(data, reason: str, seed: int, diagnostics):
    """Return a shape-consistent unavailable record after a normal failure."""

    _attach_placeholder(data, reason, seed)
    data.search_stop_reason = str(reason)
    data.trimer_failure_code = str(reason)
    data.trimer_conformer_method = TRIMER_MCL_PROTOCOL
    data.generation_diagnostics = diagnostics
    return data


def attach_unavailable_trimer_mcl(data, reason: str):
    """Attach an explicit invalid Trimer payload without running RDKit 3D."""

    smiles = str(getattr(data, "smiles", ""))
    return _attach_placeholder(data, reason, _sample_seed(smiles))


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

class TrimerContractError(RuntimeError):
    """Fatal chemical identity, bond, or Stereo-reference violation."""


class _ExpectedGeometryFailure(RuntimeError):
    """Expected bounded 3-D generation failure which may produce K=0."""


@dataclass(frozen=True)
class _Candidate:
    positions: torch.Tensor
    energy: float
    round_id: int
    candidate_id: int
    mmff_status: int
    converged: bool


def _round_seed(sample_key, round_id: int) -> int:
    digest = hashlib.sha256(
        f"{TRIMER_MCL_SCHEMA}:{sample_key}:{int(round_id)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def _coordinates(mol: Chem.Mol, conf_id: int, atom_count: int) -> torch.Tensor:
    conformer = mol.GetConformer(int(conf_id))
    return torch.tensor([
        [float(conformer.GetAtomPosition(i).x), float(conformer.GetAtomPosition(i).y),
         float(conformer.GetAtomPosition(i).z)]
        for i in range(int(atom_count))
    ], dtype=torch.float64)


def audit_double_bond_stereo_coordinates(
    mol: Chem.Mol, positions, *, epsilon: float = 1e-12
) -> int:
    """Validate every explicitly retained E/Z or cis/trans bond by sign."""
    xyz = torch.as_tensor(positions, dtype=torch.float64)
    if xyz.shape != (mol.GetNumAtoms(), 3) or not bool(torch.isfinite(xyz).all()):
        raise _ExpectedGeometryFailure("INVALID_COORDINATES")
    checked = 0
    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if stereo not in {
            Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
            Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS,
        }:
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        refs = tuple(int(v) for v in bond.GetStereoAtoms())
        if len(refs) != 2:
            raise TrimerContractError("STEREO_ATOMS_MISSING")
        if (mol.GetBondBetweenAtoms(i, refs[0]) is None
                or mol.GetBondBetweenAtoms(j, refs[1]) is None):
            raise TrimerContractError("STEREO_ATOMS_NOT_NEIGHBORS")
        axis = xyz[j] - xyz[i]
        axis_norm = torch.linalg.vector_norm(axis)
        if not bool(torch.isfinite(axis_norm)) or float(axis_norm) <= epsilon:
            raise _ExpectedGeometryFailure(f"STEREO_UNDETERMINED:{i}:{j}")
        axis = axis / axis_norm
        u, v = xyz[refs[0]] - xyz[i], xyz[refs[1]] - xyz[j]
        u = u - torch.dot(u, axis) * axis
        v = v - torch.dot(v, axis) * axis
        denominator = torch.linalg.vector_norm(u) * torch.linalg.vector_norm(v)
        if not bool(torch.isfinite(denominator)) or float(denominator) <= epsilon:
            raise _ExpectedGeometryFailure(f"STEREO_UNDETERMINED:{i}:{j}")
        cosine = float(torch.dot(u, v) / denominator)
        if not torch.isfinite(torch.tensor(cosine)):
            raise _ExpectedGeometryFailure(f"STEREO_UNDETERMINED:{i}:{j}")
        expected = -1.0 if stereo in {
            Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS,
        } else 1.0
        signed_cosine = expected * cosine
        if signed_cosine == 0.0:
            raise _ExpectedGeometryFailure(f"STEREO_UNDETERMINED:{i}:{j}")
        if signed_cosine < 0.0:
            raise _ExpectedGeometryFailure(
                f"STEREO_MISMATCH:{i}:{j}:{refs[0]}:{refs[1]}"
            )
        checked += 1
    return checked


def audit_tetrahedral_stereo_coordinates(mol: Chem.Mol, positions) -> int:
    """Validate coordinates for atoms carrying an explicit RDKit tetrahedral tag.

    RDKit derives the observed tag from the conformer using the molecule's own
    neighbour ordering.  Comparing tags on the same molecular identity avoids
    assigning a handedness from raw atom ids, and remains valid after
    ``RenumberAtoms`` (which updates the tag/ordering relationship).
    """

    xyz = torch.as_tensor(positions, dtype=torch.float64)
    if xyz.shape != (mol.GetNumAtoms(), 3) or not bool(torch.isfinite(xyz).all()):
        raise _ExpectedGeometryFailure("INVALID_COORDINATES")
    tetrahedral = {
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    }
    declared = {
        int(atom.GetIdx()): atom.GetChiralTag()
        for atom in mol.GetAtoms() if atom.GetChiralTag() in tetrahedral
    }
    if not declared:
        return 0
    for atom_index in declared:
        degree = len(tuple(mol.GetAtomWithIdx(atom_index).GetNeighbors()))
        if degree not in {3, 4}:
            raise TrimerContractError(
                f"TETRAHEDRAL_REFERENCE_DEGREE:{atom_index}:{degree}"
            )
    observed = Chem.Mol(mol)
    observed.RemoveAllConformers()
    for atom in observed.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        if atom.HasProp("_CIPCode"):
            atom.ClearProp("_CIPCode")
    conformer = Chem.Conformer(observed.GetNumAtoms())
    conformer.Set3D(True)
    for atom_index, point in enumerate(xyz.tolist()):
        conformer.SetAtomPosition(atom_index, point)
    observed.AddConformer(conformer, assignId=True)
    Chem.AssignAtomChiralTagsFromStructure(
        observed, confId=0, replaceExistingTags=True
    )
    for atom_index, expected in declared.items():
        actual = observed.GetAtomWithIdx(atom_index).GetChiralTag()
        if actual == Chem.ChiralType.CHI_UNSPECIFIED:
            raise _ExpectedGeometryFailure(
                f"TETRAHEDRAL_STEREO_UNDETERMINED:{atom_index}"
            )
        if actual != expected:
            raise _ExpectedGeometryFailure(
                f"TETRAHEDRAL_STEREO_MISMATCH:{atom_index}"
            )
    return len(declared)


def _validate_trimer_contract(data, trimer, metadata):
    """Contract A (source identity) and Contract B (heavy subset), pre-AddHs.

    Source/base identity is every non-dummy explicit atom of the canonical
    P-SMILES, including explicit ``[H]``/``[2H]``/``[3H]``.  Identity is
    ``(atomic number, isotope, ...)``: ``[H] != [2H] != [3H]``.  The heavy
    subset is strictly ``Z > 1`` and is derived from, never confused with,
    the source identity.
    """
    base_count = int(metadata["base_atom_count"])
    units = metadata["unit_atoms"]
    if len(units) != 3 or any(len(unit) != base_count for unit in units):
        raise TrimerContractError("invalid_trimer_unit_mapping")
    if trimer.GetNumAtoms() != 3 * base_count:
        raise TrimerContractError("trimer_atom_count_mismatch")
    reference = [trimer.GetAtomWithIdx(i).GetAtomicNum() for i in units[0]]
    if any([trimer.GetAtomWithIdx(i).GetAtomicNum() for i in unit] != reference
           for unit in units[1:]):
        raise TrimerContractError("trimer_copy_atomic_number_mismatch")
    reference_isotopes = [trimer.GetAtomWithIdx(i).GetIsotope() for i in units[0]]
    if any([trimer.GetAtomWithIdx(i).GetIsotope() for i in unit] != reference_isotopes
           for unit in units[1:]):
        raise TrimerContractError("trimer_copy_isotope_identity_mismatch")
    unit_bonds = []
    heavy_unit_bonds = []
    for unit in units:
        reverse = {int(atom): base_id for base_id, atom in enumerate(unit)}
        unit_bonds.append({
            (min(reverse[bond.GetBeginAtomIdx()], reverse[bond.GetEndAtomIdx()]),
             max(reverse[bond.GetBeginAtomIdx()], reverse[bond.GetEndAtomIdx()]),
             _bond_code(bond))
            for bond in trimer.GetBonds()
            if bond.GetBeginAtomIdx() in reverse and bond.GetEndAtomIdx() in reverse
        })
        heavy_unit_bonds.append({
            (min(reverse[bond.GetBeginAtomIdx()], reverse[bond.GetEndAtomIdx()]),
             max(reverse[bond.GetBeginAtomIdx()], reverse[bond.GetEndAtomIdx()]),
             _bond_code(bond))
            for bond in trimer.GetBonds()
            if bond.GetBeginAtomIdx() in reverse and bond.GetEndAtomIdx() in reverse
            and trimer.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum() > 1
            and trimer.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum() > 1
        })
    if any(observed != unit_bonds[0] for observed in unit_bonds[1:]):
        raise TrimerContractError("trimer_ru_internal_bond_contract")
    if any(observed != heavy_unit_bonds[0] for observed in heavy_unit_bonds[1:]):
        raise TrimerContractError("trimer_ru_heavy_bond_contract")
    inter_edges = {tuple(sorted(map(int, edge))) for edge in metadata["inter_unit_edges"]}
    observed = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
                for b in trimer.GetBonds()}
    if len(inter_edges) != 2 or not inter_edges <= observed:
        raise TrimerContractError("trimer_inter_ru_bond_contract")
    for left, right in inter_edges:
        if (trimer.GetAtomWithIdx(left).GetAtomicNum() <= 1
                or trimer.GetAtomWithIdx(right).GetAtomicNum() <= 1):
            raise TrimerContractError("trimer_inter_ru_bond_not_heavy")
    canonical = torch.as_tensor(data.canonical_ru_atom_index, dtype=torch.long).reshape(-1)
    mapping = torch.as_tensor(getattr(data, "canonical_to_trimer_base_atom_id", canonical),
                              dtype=torch.long).reshape(-1)
    if canonical.numel() != int(data.num_nodes) or mapping.numel() != int(data.num_nodes):
        raise TrimerContractError("o8_canonical_mapping_length_mismatch")
    if mapping.numel() and (int(mapping.min()) < 0 or int(mapping.max()) + 1 != base_count):
        raise TrimerContractError("o8_trimer_canonical_count_mismatch")
    if hasattr(data, "z"):
        z = torch.as_tensor(data.z, dtype=torch.long)
        for base_id, expected in enumerate(reference):
            found = torch.unique(z[mapping == base_id])
            if found.numel() != 1 or int(found.item()) != int(expected):
                raise TrimerContractError("o8_trimer_canonical_atomic_number_mismatch")
    return base_count, units, mapping


def _attach_all_atom_topology(
    data, trimer, mol_h, metadata, base_count, units, mapping
):
    """Attach decoupled source/base and heavy identity over the all-atom layout.

    Layout invariants (Contract C): ``AddHs`` appends, so all-atom indices
    ``0..source_count-1`` are the source atoms (explicit isotope H included)
    and ``source_count..all_count-1`` are the AddHs-generated hydrogens.  The
    heavy subset is ``Z > 1`` anywhere in the all-atom layout and may therefore
    be non-contiguous when the source contains explicit isotope H.
    """
    source_count = trimer.GetNumAtoms()
    all_count = mol_h.GetNumAtoms()
    if all_count < source_count:
        raise TrimerContractError("all_atom_count_smaller_than_source_count")
    source_atomic = [atom.GetAtomicNum() for atom in trimer.GetAtoms()]
    if any(int(z) < 1 for z in source_atomic):
        raise TrimerContractError("source_identity_contains_dummy_atom")
    source_isotope = [atom.GetIsotope() for atom in trimer.GetAtoms()]
    atomic = torch.tensor(
        [atom.GetAtomicNum() for atom in mol_h.GetAtoms()], dtype=torch.long
    )
    isotope = torch.tensor(
        [atom.GetIsotope() for atom in mol_h.GetAtoms()], dtype=torch.long
    )
    if atomic[:source_count].tolist() != source_atomic:
        raise TrimerContractError("add_hs_changed_source_atom_identity")
    if isotope[:source_count].tolist() != source_isotope:
        raise TrimerContractError("add_hs_changed_source_isotope_identity")
    if all_count > source_count:
        if not bool((atomic[source_count:] == 1).all()):
            raise TrimerContractError("add_hs_appended_non_hydrogen_atom")
        if not bool((isotope[source_count:] == 0).all()):
            raise TrimerContractError("add_hs_appended_isotope_atom")

    heavy_mask = atomic > 1
    heavy_indices = torch.nonzero(heavy_mask, as_tuple=False).flatten()
    expected_heavy = [index for index, z in enumerate(source_atomic) if z > 1]
    if heavy_indices.tolist() != expected_heavy:
        raise TrimerContractError("heavy_atom_indices_are_not_identity_preserving")

    source_offsets = torch.tensor(metadata["atom_ru_index"], dtype=torch.long) - 1
    if source_offsets.numel() != source_count:
        raise TrimerContractError("source_ru_offset_length_mismatch")
    offsets = torch.empty(all_count, dtype=torch.long)
    offsets[:source_count] = source_offsets
    h_parent = torch.full((all_count,), -1, dtype=torch.long)
    for atom_index in range(source_count, all_count):
        atom = mol_h.GetAtomWithIdx(atom_index)
        neighbours = tuple(atom.GetNeighbors())
        if atom.GetAtomicNum() != 1 or len(neighbours) != 1:
            raise TrimerContractError("hydrogen_parent_contract")
        parent = int(neighbours[0].GetIdx())
        if parent >= source_count or mol_h.GetAtomWithIdx(parent).GetAtomicNum() <= 1:
            raise TrimerContractError("hydrogen_parent_is_not_heavy")
        h_parent[atom_index] = parent
        offsets[atom_index] = source_offsets[parent]
    # Source explicit H/D/T keep their own source identity; their parent-heavy
    # must still be the single real bonded heavy atom, and RU follows the parent.
    for atom_index in range(source_count):
        if int(atomic[atom_index]) != 1:
            continue
        neighbours = tuple(mol_h.GetAtomWithIdx(atom_index).GetNeighbors())
        heavy_neighbours = [
            neighbour.GetIdx() for neighbour in neighbours
            if mol_h.GetAtomWithIdx(neighbour.GetIdx()).GetAtomicNum() > 1
        ]
        if len(neighbours) != 1 or len(heavy_neighbours) != 1:
            raise TrimerContractError("source_explicit_h_parent_contract")
        parent = int(heavy_neighbours[0])
        h_parent[atom_index] = parent
        if int(offsets[atom_index]) != int(offsets[parent]):
            raise TrimerContractError("source_explicit_h_ru_contract")

    base_ids = torch.full((all_count,), -1, dtype=torch.long)
    for unit in units:
        for base_id, atom_index in enumerate(unit):
            base_ids[int(atom_index)] = int(base_id)
    if bool((base_ids[:source_count] < 0).any()):
        raise TrimerContractError("source_base_identity_incomplete")
    if bool((base_ids[source_count:] >= 0).any()):
        raise TrimerContractError("added_h_has_base_identity")

    physical_sources, physical_targets, physical_types = [], [], []
    directed_sources, directed_targets, directed_types, directed_aromatic = [], [], [], []
    for bond in mol_h.GetBonds():
        i, j, code = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), _bond_code(bond)
        physical_sources.append(i)
        physical_targets.append(j)
        physical_types.append(code)
        directed_sources.extend((i, j))
        directed_targets.extend((j, i))
        directed_types.extend((code, code))
        directed_aromatic.extend((bool(bond.GetIsAromatic()),) * 2)
    central = torch.tensor(units[1], dtype=torch.long)
    central_mapping = central[mapping]
    # The full canonical mapping may target explicit isotope H; heaviness of
    # each target is exported as o8_heavy_mask instead of being enforced here.
    o8_heavy_mask = heavy_mask[central_mapping].clone()

    data.trimer_atomic_number = atomic
    data.trimer_atomic_numbers = atomic
    data.trimer_isotope = isotope
    data.trimer_source_atom_count = torch.tensor(source_count, dtype=torch.long)
    data.is_source_atom = torch.arange(all_count, dtype=torch.long) < source_count
    data.is_source_explicit_h = data.is_source_atom & (atomic == 1)
    data.is_added_h = ~data.is_source_atom
    data.trimer_atom_id = torch.arange(all_count, dtype=torch.long)
    data.trimer_edge_index = torch.tensor(
        [directed_sources, directed_targets], dtype=torch.long
    )
    data.trimer_bond_type = torch.tensor(directed_types, dtype=torch.long)
    data.trimer_bond_aromatic = torch.tensor(directed_aromatic, dtype=torch.bool)
    data.trimer_formal_charge = torch.tensor(
        [atom.GetFormalCharge() for atom in mol_h.GetAtoms()], dtype=torch.long
    )
    data.trimer_is_aromatic = torch.tensor(
        [atom.GetIsAromatic() for atom in mol_h.GetAtoms()], dtype=torch.bool
    )
    data.trimer_chiral_tag = torch.tensor(
        [int(atom.GetChiralTag()) for atom in mol_h.GetAtoms()], dtype=torch.long
    )
    attachment_role = torch.zeros(all_count, dtype=torch.long)
    for left, right in zip(
        metadata["unit_left_boundaries"], metadata["unit_right_boundaries"]
    ):
        attachment_role[int(left)] |= 1
        attachment_role[int(right)] |= 2
    data.trimer_attachment_role = attachment_role
    internal_degree = torch.zeros(all_count, dtype=torch.long)
    for left, right in zip(physical_sources, physical_targets):
        if int(offsets[left]) == int(offsets[right]):
            internal_degree[left] += 1
            internal_degree[right] += 1
    data.trimer_internal_degree = internal_degree
    data.trimer_bonds = torch.tensor(
        [physical_sources, physical_targets], dtype=torch.long
    )
    data.trimer_bond_types = torch.tensor(physical_types, dtype=torch.long)
    data.trimer_base_ru_atom_id = base_ids
    data.trimer_base_ru_atom_index = data.trimer_base_ru_atom_id
    data.trimer_ru_offset = offsets
    data.trimer_central_ru_mask = data.trimer_ru_offset == 0
    data.trimer_central_atom_index = central
    data.trimer_central_ru_atom_index = central
    data.mips_to_trimer_central_index = central_mapping
    data.canonical_to_trimer_central_index = central_mapping
    data.o8_to_trimer_atom = central_mapping
    data.o8_heavy_mask = o8_heavy_mask
    data.o8_heavy_indices = torch.nonzero(o8_heavy_mask, as_tuple=False).flatten()
    data.trimer_heavy_mask = heavy_mask
    data.trimer_heavy_indices = heavy_indices
    data.h_parent_heavy_index = h_parent


def _audit_declared_stereo(trimer, mol_h, positions) -> tuple[int, int]:
    heavy_count = trimer.GetNumAtoms()
    double_count = audit_double_bond_stereo_coordinates(
        trimer, torch.as_tensor(positions)[:heavy_count]
    )
    tetrahedral_count = audit_tetrahedral_stereo_coordinates(mol_h, positions)
    return double_count, tetrahedral_count


def _audit_candidate_stereo_stage(
    trimer, mol_h, positions, candidate_diagnostics, stage
):
    """Run and time the two independent declared-Stereo gates for a pilot/QC row."""

    started = time.monotonic()
    double_key = f"double_bond_stereo_{stage}_pass"
    tetra_key = f"tetra_stereo_{stage}_pass"
    try:
        double_count = audit_double_bond_stereo_coordinates(
            trimer, torch.as_tensor(positions)[: trimer.GetNumAtoms()]
        )
        candidate_diagnostics[double_key] = True
        candidate_diagnostics[f"double_bond_stereo_{stage}_count"] = double_count
    except _ExpectedGeometryFailure as exc:
        candidate_diagnostics[double_key] = False
        candidate_diagnostics["stereo_error"] = str(exc)
        return False, "DOUBLE_BOND", time.monotonic() - started
    try:
        tetra_count = audit_tetrahedral_stereo_coordinates(mol_h, positions)
        candidate_diagnostics[tetra_key] = True
        candidate_diagnostics[f"tetra_stereo_{stage}_count"] = tetra_count
    except _ExpectedGeometryFailure as exc:
        candidate_diagnostics[tetra_key] = False
        candidate_diagnostics["stereo_error"] = str(exc)
        return False, "TETRA", time.monotonic() - started
    return True, "", time.monotonic() - started


def _validate_all_atom_payload(data):
    positions = torch.as_tensor(data.trimer_pos)
    atomic = torch.as_tensor(data.trimer_atomic_numbers, dtype=torch.long)
    isotope = torch.as_tensor(data.trimer_isotope, dtype=torch.long)
    atom_ids = torch.as_tensor(data.trimer_atom_id, dtype=torch.long)
    heavy_mask = torch.as_tensor(data.trimer_heavy_mask, dtype=torch.bool)
    heavy_indices = torch.as_tensor(data.trimer_heavy_indices, dtype=torch.long)
    parents = torch.as_tensor(data.h_parent_heavy_index, dtype=torch.long)
    offsets = torch.as_tensor(data.trimer_ru_offset, dtype=torch.long)
    bonds = torch.as_tensor(data.trimer_bonds, dtype=torch.long)
    bond_types = torch.as_tensor(data.trimer_bond_types, dtype=torch.long)
    source_count = int(data.trimer_source_atom_count)
    base_ids = torch.as_tensor(data.trimer_base_ru_atom_id, dtype=torch.long)
    is_source_atom = torch.as_tensor(data.is_source_atom, dtype=torch.bool)
    is_source_explicit_h = torch.as_tensor(data.is_source_explicit_h, dtype=torch.bool)
    is_added_h = torch.as_tensor(data.is_added_h, dtype=torch.bool)
    count = int(atomic.numel())
    if (
        positions.shape != (count, 3)
        or not bool(torch.isfinite(positions).all())
        or atom_ids.tolist() != list(range(count))
        or heavy_mask.numel() != count
        or parents.numel() != count
        or offsets.numel() != count
        or isotope.numel() != count
        or not 0 <= source_count <= count
        or not bool(torch.isin(offsets, torch.tensor([-1, 0, 1])).all())
    ):
        raise TrimerContractError("all_atom_identity_or_coordinate_contract")
    expected_heavy = torch.nonzero(atomic > 1, as_tuple=False).flatten()
    if not torch.equal(heavy_mask, atomic > 1) or not torch.equal(
        heavy_indices, expected_heavy
    ):
        raise TrimerContractError("heavy_mask_index_contract")
    row_index = torch.arange(count, dtype=torch.long)
    if (
        not torch.equal(is_source_atom, row_index < source_count)
        or not torch.equal(is_added_h, row_index >= source_count)
        or not torch.equal(is_source_explicit_h, is_source_atom & (atomic == 1))
    ):
        raise TrimerContractError("source_added_h_flag_contract")
    if bool((isotope < 0).any()) or bool((isotope[is_added_h] != 0).any()):
        raise TrimerContractError("added_h_isotope_contract")
    hydrogen = ~heavy_mask
    if bool((atomic[hydrogen] != 1).any()) or bool((parents[heavy_mask] != -1).any()):
        raise TrimerContractError("hydrogen_atomic_number_or_parent_sentinel")
    if hydrogen.any():
        h_parents = parents[hydrogen]
        if (
            bool((h_parents < 0).any())
            or bool((h_parents >= count).any())
            or not bool(heavy_mask[h_parents].all())
            or not torch.equal(offsets[hydrogen], offsets[h_parents])
        ):
            raise TrimerContractError("hydrogen_parent_or_ru_contract")
    if bool((base_ids[:source_count] < 0).any()) or bool(
        (base_ids[source_count:] >= 0).any()
    ):
        raise TrimerContractError("source_base_identity_completeness")
    if (
        bonds.ndim != 2 or bonds.size(0) != 2
        or bonds.size(1) != bond_types.numel()
        or (bonds.numel() and (
            int(bonds.min()) < 0 or int(bonds.max()) >= count
            or bool((bonds[0] == bonds[1]).any())
        ))
    ):
        raise TrimerContractError("all_atom_physical_bond_contract")
    physical = {tuple(sorted((int(a), int(b)))) for a, b in bonds.T.tolist()}
    directed = torch.as_tensor(data.trimer_edge_index, dtype=torch.long)
    directed_physical = {
        tuple(sorted((int(a), int(b)))) for a, b in directed.T.tolist()
    }
    if physical != directed_physical or directed.size(1) != 2 * len(physical):
        raise TrimerContractError("physical_and_directed_bond_tables_disagree")
    for hydrogen_index in torch.nonzero(hydrogen, as_tuple=False).flatten().tolist():
        if tuple(sorted((hydrogen_index, int(parents[hydrogen_index])))) not in physical:
            raise TrimerContractError("hydrogen_parent_bond_missing")
    mapping = torch.as_tensor(data.o8_to_trimer_atom, dtype=torch.long)
    o8_heavy_mask = torch.as_tensor(data.o8_heavy_mask, dtype=torch.bool)
    if mapping.numel() and (
        bool((mapping < 0).any())
        or bool((mapping >= count).any())
        or bool((mapping >= source_count).any())
        or not torch.equal(heavy_mask[mapping], o8_heavy_mask)
    ):
        raise TrimerContractError("o8_to_trimer_atom_contract")
    expected_o8_heavy = torch.nonzero(o8_heavy_mask, as_tuple=False).flatten()
    if not torch.equal(
        torch.as_tensor(data.o8_heavy_indices, dtype=torch.long), expected_o8_heavy
    ):
        raise TrimerContractError("o8_heavy_index_contract")


def _star_geometry(data, metadata):
    positions = torch.as_tensor(data.trimer_pos)
    lengths = [
        torch.linalg.vector_norm(positions[int(left)] - positions[int(right)])
        for left, right in metadata["inter_unit_edges"]
    ]
    if len(lengths) != 2:
        raise TrimerContractError("trimer_requires_two_inter_ru_bonds")
    data.star_3d_distance = torch.stack(lengths).mean().float()
    data.star_3d_asymmetry = torch.abs(lengths[0] - lengths[1]).float()
    data.star_3d_valid = torch.tensor(
        bool(
            torch.isfinite(data.star_3d_distance)
            and torch.isfinite(data.star_3d_asymmetry)
            and float(data.star_3d_asymmetry) <= 0.15
        ), dtype=torch.bool,
    )


def attach_finite_trimer_mcl(
    data, smiles, *, num_candidates: int = TRIMER_CANDIDATES_PER_ROUND,
    max_heavy_atoms=None, max_rounds: int = TRIMER_MAX_ROUNDS,
    timeout_seconds: float = TRIMER_SAMPLE_TIMEOUT_SECONDS,
    sample_key=None,
):
    """Attach one explicit-H Trimer conformer selected from at most two rounds."""
    del max_heavy_atoms  # Deliberately no atom-count rejection or 2-D fallback.
    num_candidates = int(num_candidates)
    max_rounds = int(max_rounds)
    if not 1 <= num_candidates <= TRIMER_CANDIDATES_PER_ROUND:
        raise ValueError("num_candidates must be in [1,8]")
    if not 0 <= max_rounds <= TRIMER_MAX_ROUNDS:
        raise ValueError("max_rounds must be in [0,2]")
    seed = _sample_seed(smiles)
    _attach_placeholder(data, "not_built", seed)
    if not bool(getattr(data, "graph_available", True)):
        return _attach_placeholder(data, "graph_unavailable", seed)
    started = time.monotonic()
    canonical_source = _canonical_source_mol(smiles)
    trimer, metadata = build_periodic_multimer_mol(
        canonical_source, 3, close_periodic=False
    )
    base_count, units, mapping = _validate_trimer_contract(data, trimer, metadata)
    source_count = trimer.GetNumAtoms()
    # Building must have propagated all StereoAtoms before any coordinates exist.
    for bond in trimer.GetBonds():
        if bond.GetStereo() in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
                               Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS}:
            refs = tuple(bond.GetStereoAtoms())
            if len(refs) != 2:
                raise TrimerContractError("source_to_trimer_stereo_propagation")
    mol_h_template = Chem.AddHs(Chem.Mol(trimer))
    _attach_all_atom_topology(
        data, trimer, mol_h_template, metadata, base_count, units, mapping
    )
    all_count = mol_h_template.GetNumAtoms()
    properties = AllChem.MMFFGetMoleculeProperties(mol_h_template, mmffVariant="MMFF94")
    diagnostics = dict(data.generation_diagnostics)
    diagnostics["trimer_source_atoms"] = source_count
    diagnostics["trimer_heavy_atoms"] = sum(
        1 for atom in trimer.GetAtoms() if atom.GetAtomicNum() > 1
    )
    diagnostics["trimer_atoms_with_h"] = mol_h_template.GetNumAtoms()
    diagnostics["source_explicit_h_atoms"] = sum(
        1 for atom in trimer.GetAtoms() if atom.GetAtomicNum() == 1
    )
    diagnostics["source_isotope_h_atoms"] = sum(
        1 for atom in trimer.GetAtoms()
        if atom.GetAtomicNum() == 1 and atom.GetIsotope() > 0
    )
    diagnostics["declared_double_bond_stereo_count"] = sum(
        bond.GetStereo() in {
            Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
            Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS,
        }
        for bond in trimer.GetBonds()
    )
    diagnostics["declared_tetrahedral_stereo_count"] = sum(
        atom.GetChiralTag() in {
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        }
        for atom in mol_h_template.GetAtoms()
    )
    if properties is None:
        diagnostics["total_time"] = time.monotonic() - started
        return _finish_geometry_failure(
            data, "MMFF_UNSUPPORTED", seed, diagnostics
        )
    selected = None
    stop_reason = "NO_VALID_CONFORMER"
    identity = sample_key if sample_key is not None else Chem.MolToSmiles(trimer, canonical=True)
    for round_id in range(max_rounds):
        if time.monotonic() - started >= float(timeout_seconds):
            stop_reason = "timeout"; break
        round_started = time.monotonic()
        round_seed = _round_seed(identity, round_id)
        round_diagnostics = {
            "round_id": round_id,
            "seed": round_seed,
            "requested": num_candidates,
            "embedded": 0,
            "valid": 0,
            "embed_time": 0.0,
            "mmff_time": 0.0,
            "stereo_time": 0.0,
            "total_time": 0.0,
            "candidates": [],
        }
        diagnostics["num_rounds"] += 1
        diagnostics["num_candidates_requested"] += num_candidates
        embed_started = time.monotonic()
        mol_h, ids = _embed_attempt(
            mol_h_template, num_candidates=num_candidates,
            seed=round_seed, use_random_coords=True,
            max_iterations=TRIMER_ETKDG_MAX_ITERATIONS)
        embed_elapsed = time.monotonic() - embed_started
        diagnostics["embed_time"] += embed_elapsed
        round_diagnostics["embed_time"] = embed_elapsed
        diagnostics["num_candidates_embedded"] += len(ids)
        round_diagnostics["embedded"] = len(ids)
        round_props = AllChem.MMFFGetMoleculeProperties(
            mol_h, mmffVariant="MMFF94"
        )
        if round_props is None:
            raise TrimerContractError("MMFF_parameters_changed_after_embedding")
        round_candidates = []
        for candidate_id, conf_id in enumerate(ids):
            candidate_diagnostics = {
                "candidate_id": candidate_id,
                "conformer_id": int(conf_id),
                "requested": True,
                "embedded": True,
                "finite_pre_mmff": False,
                "double_bond_stereo_pre_pass": None,
                "tetra_stereo_pre_pass": None,
                "mmff_attempted": False,
                "mmff_status": None,
                "mmff_converged": False,
                "converged": False,
                "mmff_energy_finite": False,
                "mmff_energy": None,
                "finite_energy": False,
                "finite_post_mmff": False,
                "double_bond_stereo_post_f64_pass": None,
                "tetra_stereo_post_f64_pass": None,
                "double_bond_stereo_post_f32_pass": None,
                "tetra_stereo_post_f32_pass": None,
                "final_valid": False,
                "accepted": False,
                "rejection": "",
                "stereo_error": "",
                "stereo_time": 0.0,
            }
            round_diagnostics["candidates"].append(candidate_diagnostics)
            if not _conformer_coordinates_are_finite_3d(
                    mol_h, conf_id, all_count):
                diagnostics["num_geometry_rejected"] += 1
                candidate_diagnostics["rejection"] = "PRE_MMFF_NONFINITE"
                continue
            candidate_diagnostics["finite_pre_mmff"] = True
            stereo_pass, component, stereo_elapsed = _audit_candidate_stereo_stage(
                trimer, mol_h, _coordinates(mol_h, conf_id, all_count),
                candidate_diagnostics, "pre",
            )
            candidate_diagnostics["stereo_time"] += stereo_elapsed
            diagnostics["stereo_time"] += stereo_elapsed
            round_diagnostics["stereo_time"] += stereo_elapsed
            if not stereo_pass:
                diagnostics["num_pre_stereo_rejected"] += 1
                candidate_diagnostics["rejection"] = f"{component}_STEREO_PRE"
                continue
            candidate_diagnostics["mmff_attempted"] = True
            diagnostics["num_mmff_attempted"] += 1
            mmff_started = time.monotonic()
            mmff_status = int(AllChem.MMFFOptimizeMolecule(
                mol_h, confId=int(conf_id),
                maxIters=TRIMER_MMFF_RELAX_MAX_ITERATIONS,
                mmffVariant="MMFF94",
            ))
            mmff_elapsed = time.monotonic() - mmff_started
            diagnostics["mmff_time"] += mmff_elapsed
            round_diagnostics["mmff_time"] += mmff_elapsed
            converged = mmff_status == 0
            candidate_diagnostics["mmff_status"] = mmff_status
            candidate_diagnostics["mmff_converged"] = converged
            candidate_diagnostics["converged"] = converged
            diagnostics["num_mmff_converged"] += int(converged)
            if not _conformer_coordinates_are_finite_3d(mol_h, conf_id, all_count):
                diagnostics["num_geometry_rejected"] += 1
                candidate_diagnostics["rejection"] = "POST_MMFF_NONFINITE_COORDINATES"
                continue
            candidate_diagnostics["finite_post_mmff"] = True
            energy = _calculate_mmff_energy(mol_h, round_props, conf_id)
            if energy is None:
                diagnostics["num_geometry_rejected"] += 1
                candidate_diagnostics["rejection"] = "POST_MMFF_NONFINITE_ENERGY"
                continue
            candidate_diagnostics["mmff_energy_finite"] = True
            candidate_diagnostics["mmff_energy"] = float(energy)
            candidate_diagnostics["finite_energy"] = True
            xyz64 = _coordinates(mol_h, conf_id, all_count)
            stereo_pass, component, stereo_elapsed = _audit_candidate_stereo_stage(
                trimer, mol_h, xyz64, candidate_diagnostics, "post_f64"
            )
            candidate_diagnostics["stereo_time"] += stereo_elapsed
            diagnostics["stereo_time"] += stereo_elapsed
            round_diagnostics["stereo_time"] += stereo_elapsed
            if not stereo_pass:
                diagnostics["num_post_stereo_rejected"] += 1
                candidate_diagnostics["rejection"] = f"{component}_STEREO_POST_F64"
                continue
            xyz32 = xyz64.to(torch.float32)
            stereo_pass, component, stereo_elapsed = _audit_candidate_stereo_stage(
                trimer, mol_h, xyz32, candidate_diagnostics, "post_f32"
            )
            candidate_diagnostics["stereo_time"] += stereo_elapsed
            diagnostics["stereo_time"] += stereo_elapsed
            round_diagnostics["stereo_time"] += stereo_elapsed
            if not stereo_pass:
                diagnostics["num_post_stereo_rejected"] += 1
                candidate_diagnostics["rejection"] = f"{component}_STEREO_POST_F32"
                continue
            candidate_diagnostics["final_valid"] = True
            candidate_diagnostics["accepted"] = True
            round_candidates.append(_Candidate(
                xyz32, float(energy), round_id, candidate_id,
                mmff_status, converged,
            ))
        round_diagnostics["valid"] = len(round_candidates)
        round_diagnostics["total_time"] = time.monotonic() - round_started
        diagnostics["rounds"].append(round_diagnostics)
        diagnostics["num_valid_candidates"] += len(round_candidates)
        if round_candidates:
            selected = min(
                round_candidates,
                key=lambda candidate: (
                    not candidate.converged,
                    candidate.energy,
                    candidate.candidate_id,
                ),
            )
            stop_reason = "selected"
            break
        if time.monotonic() - started >= float(timeout_seconds):
            stop_reason = "timeout"
            break
    diagnostics["total_time"] = time.monotonic() - started
    data.generation_diagnostics = diagnostics
    if selected is None:
        if stop_reason != "timeout":
            stop_reason = (
                "ETKDG_NO_VALID_CONFORMER"
                if diagnostics["num_candidates_embedded"] == 0
                else "NO_VALID_CONFORMER"
            )
        return _finish_geometry_failure(data, stop_reason, seed, diagnostics)

    data.trimer_pos = selected.positions
    data.trimer_conformer_energy = float(selected.energy)
    data.trimer_conformer_round_id = int(selected.round_id)
    data.trimer_conformer_candidate_id = int(selected.candidate_id)
    data.trimer_conformer_seed = _round_seed(identity, selected.round_id)
    data.selected_converged = bool(selected.converged)
    data.search_stop_reason = "selected"
    data.trimer_geometry_valid = True
    data.trimer_geometry_is_3d = torch.tensor(True, dtype=torch.bool)
    data.trimer_2d_fallback = torch.tensor(False, dtype=torch.bool)
    data.trimer_geometry_source = "etkdgv3_randomcoords_mmff94_allatom"
    data.trimer_failure_code = ""
    data.trimer_conformer_method = TRIMER_MCL_PROTOCOL
    _validate_all_atom_payload(data)
    _star_geometry(data, metadata)
    return data
