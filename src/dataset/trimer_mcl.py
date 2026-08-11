"""Deterministic finite-Trimer geometry for the selected O8 route.

The Trimer is an open, H-terminated local geometry proxy.  It is deliberately
separate from the O8 star-linked graph: O8 repeat copies are mapped to the
central repeat unit only through their canonical RU atom identity.

MMFF94 is used for a fixed-budget local coordinate relaxation only.  Its
convergence status is deliberately ignored: the sole acceptance criteria are
finite 3-D coordinates and finite post-relaxation MMFF energy.
"""

from __future__ import annotations

import hashlib
import threading
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
TRIMER_ETKDG_MAX_ITERATIONS = 42
TRIMER_ETKDG_TIMEOUT_SECONDS = 60
TRIMER_ETKDG_RETRY_CANDIDATES = 2
TRIMER_ETKDG_RETRY_MAX_ITERATIONS = 200
TRIMER_MMFF_RELAX_MAX_ITERATIONS = CONTRACT_MMFF_RELAX_MAX_ITERATIONS

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


def _embed_with_targeted_retry(molecule, *, num_candidates: int, seed: int):
    """Use the cheap production embedding first and retry only on failure."""

    embedded, conformer_ids = _embed_attempt(
        molecule,
        num_candidates=int(num_candidates),
        seed=int(seed),
        use_random_coords=False,
        max_iterations=TRIMER_ETKDG_MAX_ITERATIONS,
    )
    if conformer_ids:
        return embedded, conformer_ids

    retry_seed = (int(seed) ^ 0x5EED5EED) & 0x7FFFFFFF
    embedded, conformer_ids = _embed_attempt(
        molecule,
        num_candidates=TRIMER_ETKDG_RETRY_CANDIDATES,
        seed=retry_seed,
        use_random_coords=True,
        max_iterations=TRIMER_ETKDG_RETRY_MAX_ITERATIONS,
    )
    if not conformer_ids:
        raise ValueError("etkdgv3_no_conformer_after_targeted_retry")
    return embedded, conformer_ids


# ---------------------------------------------------------------------------
#  New MMFF94 relax-then-select protocol
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _MMFFConformerSelection:
    conf_id: int
    energy: float


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


def _validate_conformer_ids(mol, expected_ids):
    """Assert that the molecule contains exactly the expected conformer IDs."""

    actual = {
        int(conf.GetId()) for conf in mol.GetConformers()
    }
    expected = {int(cid) for cid in expected_ids}
    if actual != expected:
        raise ValueError(
            "mmff94_conformer_id_mapping_mismatch:"
            f"expected {sorted(expected)}, got {sorted(actual)}"
        )


def _optimize_mmff_and_select_lowest_finite(
    mol: Chem.Mol,
    conformer_ids: list[int],
    *,
    max_iterations: int = 200,
) -> _MMFFConformerSelection:
    """Relax every ETKDG conformer once with MMFF94 and select the lowest
    finite post-relaxation energy candidate.

    MMFF convergence status is deliberately discarded.  The only acceptance
    criteria are:

    * the conformer still exists and ``Is3D()`` after relaxation,
    * all coordinates are finite,
    * a post-relaxation MMFF94 energy can be computed and is finite.
    """

    if not conformer_ids:
        raise ValueError("mmff94_no_conformers_to_relax")
    num_atoms = mol.GetNumAtoms()

    properties = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    if properties is None:
        raise ValueError("mmff94_parameters_unavailable")

    _validate_conformer_ids(mol, conformer_ids)

    # Single bulk MMFF94 relaxation pass — return values are discarded on
    # purpose so no status-dependent logic can creep in.
    AllChem.MMFFOptimizeMoleculeConfs(
        mol,
        numThreads=1,
        maxIters=int(max_iterations),
        mmffVariant="MMFF94",
    )

    # Re-check that conformer IDs survived the relaxation intact.
    _validate_conformer_ids(mol, conformer_ids)

    candidates: list[_MMFFConformerSelection] = []
    for conf_id in conformer_ids:
        if not _conformer_coordinates_are_finite_3d(mol, conf_id, num_atoms):
            continue
        energy = _calculate_mmff_energy(mol, properties, conf_id)
        if energy is None:
            continue
        candidates.append(
            _MMFFConformerSelection(conf_id=int(conf_id), energy=energy)
        )

    if not candidates:
        raise ValueError("mmff94_no_finite_relaxed_3d_conformer")

    return min(
        candidates,
        key=lambda candidate: (candidate.energy, candidate.conf_id),
    )


# ---------------------------------------------------------------------------
#  Placeholder / unavailable
# ---------------------------------------------------------------------------

def _attach_placeholder(data, reason: str, seed: int = 0):
    node_count = int(data.num_nodes)
    data.trimer_pos = torch.empty((0, 3), dtype=torch.float)
    data.trimer_atomic_number = torch.empty((0,), dtype=torch.long)
    data.trimer_edge_index = torch.empty((2, 0), dtype=torch.long)
    data.trimer_bond_type = torch.empty((0,), dtype=torch.long)
    data.trimer_base_ru_atom_id = torch.empty((0,), dtype=torch.long)
    data.trimer_base_ru_atom_index = data.trimer_base_ru_atom_id
    data.trimer_ru_offset = torch.empty((0,), dtype=torch.long)
    data.trimer_central_ru_mask = torch.empty((0,), dtype=torch.bool)
    data.trimer_central_atom_index = torch.empty((0,), dtype=torch.long)
    data.trimer_central_ru_atom_index = data.trimer_central_atom_index
    data.mips_to_trimer_central_index = torch.full(
        (node_count,), -1, dtype=torch.long
    )
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
    return data


def attach_unavailable_trimer_mcl(data, reason: str):
    """Attach an explicit invalid Trimer payload without running RDKit 3D."""

    smiles = str(getattr(data, "smiles", ""))
    return _attach_placeholder(data, reason, _sample_seed(smiles))


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

def attach_finite_trimer_mcl(
    data,
    smiles,
    *,
    num_candidates: int = 4,
    max_heavy_atoms: int = 384,
):
    """Attach the lowest finite MMFF94-relaxed open-Trimer conformer.

    Geometry failures never remove the sample.  Instead a shape-safe invalid
    payload is attached so the graph encoder can fall back exactly to O8.
    """

    seed = _sample_seed(smiles)
    _attach_placeholder(data, "not_built", seed)
    if not bool(getattr(data, "graph_available", True)):
        return _attach_placeholder(data, "graph_unavailable", seed)
    try:
        trimer, metadata = build_periodic_multimer_mol(
            smiles, num_repeat_units=3, close_periodic=False
        )
        base_count = int(metadata["base_atom_count"])
        unit_atoms = metadata["unit_atoms"]
        if len(unit_atoms) != 3 or any(len(unit) != base_count for unit in unit_atoms):
            raise ValueError("invalid_trimer_unit_mapping")
        heavy_count = int(trimer.GetNumAtoms())
        if heavy_count != 3 * base_count:
            raise ValueError("trimer_atom_count_mismatch")
        use_2d_fallback = heavy_count > int(max_heavy_atoms)

        reference_atomic_numbers = [
            int(trimer.GetAtomWithIdx(idx).GetAtomicNum())
            for idx in unit_atoms[0]
        ]
        for unit in unit_atoms[1:]:
            observed = [
                int(trimer.GetAtomWithIdx(idx).GetAtomicNum()) for idx in unit
            ]
            if observed != reference_atomic_numbers:
                raise ValueError("trimer_copy_atomic_number_mismatch")

        internal_bonds = []
        for unit in unit_atoms:
            reverse = {int(atom): base for base, atom in enumerate(unit)}
            bonds = {
                (
                    min(reverse[bond.GetBeginAtomIdx()],
                        reverse[bond.GetEndAtomIdx()]),
                    max(reverse[bond.GetBeginAtomIdx()],
                        reverse[bond.GetEndAtomIdx()]),
                )
                for bond in trimer.GetBonds()
                if (
                    bond.GetBeginAtomIdx() in reverse
                    and bond.GetEndAtomIdx() in reverse
                )
            }
            internal_bonds.append(bonds)
        if any(bonds != internal_bonds[0] for bonds in internal_bonds[1:]):
            raise ValueError("trimer_central_ru_internal_connectivity_mismatch")

        inter_edges = {
            tuple(sorted((int(left), int(right))))
            for left, right in metadata["inter_unit_edges"]
        }
        if len(inter_edges) != 2:
            raise ValueError("trimer_requires_two_inter_ru_bonds")
        observed_edges = {
            tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
            for bond in trimer.GetBonds()
        }
        if not inter_edges <= observed_edges:
            raise ValueError("trimer_inter_ru_bond_missing")
        left_boundaries = metadata["unit_left_boundaries"]
        right_boundaries = metadata["unit_right_boundaries"]
        left_inter = (
            int(right_boundaries[0]), int(left_boundaries[1])
        )
        right_inter = (
            int(right_boundaries[1]), int(left_boundaries[2])
        )
        expected_bond_code = _bond_code(
            trimer.GetBondBetweenAtoms(*left_inter)
        )
        right_bond = trimer.GetBondBetweenAtoms(*right_inter)
        if right_bond is None or _bond_code(right_bond) != expected_bond_code:
            raise ValueError("trimer_inter_ru_bond_type_mismatch")

        canonical = data.canonical_ru_atom_index.long()
        canonical_to_trimer = getattr(
            data, "canonical_to_trimer_base_atom_id", canonical
        )
        canonical_to_trimer = torch.as_tensor(
            canonical_to_trimer, dtype=torch.long
        ).reshape(-1)
        if canonical.numel() != int(data.num_nodes) or canonical_to_trimer.numel() != int(data.num_nodes):
            raise ValueError("o8_canonical_mapping_length_mismatch")
        if canonical_to_trimer.numel() and (
            int(canonical_to_trimer.min()) < 0
            or int(canonical_to_trimer.max()) + 1 != base_count
        ):
            raise ValueError(
                "o8_trimer_canonical_count_mismatch:"
                f"{int(canonical_to_trimer.max()) + 1}!={base_count}"
            )
        if hasattr(data, "z"):
            o8_atomic_numbers = data.z.long()
            for canonical_id, expected_z in enumerate(
                reference_atomic_numbers
            ):
                observed_z = torch.unique(
                    o8_atomic_numbers[canonical_to_trimer == canonical_id]
                )
                if (
                    observed_z.numel() != 1
                    or int(observed_z.item()) != int(expected_z)
                ):
                    raise ValueError(
                        "o8_trimer_canonical_atomic_number_mismatch"
                    )

        if use_2d_fallback:
            coordinate_mol = Chem.Mol(trimer)
            AllChem.Compute2DCoords(coordinate_mol, canonOrient=True)
            conformer = coordinate_mol.GetConformer()
            energy = float("nan")
            geometry_is_3d = False
            geometry_source = "rdkit_2d_large_molecule_mcl_disabled"
        else:
            mol_h, conformer_ids = _embed_with_targeted_retry(
                Chem.AddHs(Chem.Mol(trimer)),
                num_candidates=int(num_candidates),
                seed=int(seed),
            )
            selection = _optimize_mmff_and_select_lowest_finite(
                mol_h,
                conformer_ids,
                max_iterations=TRIMER_MMFF_RELAX_MAX_ITERATIONS,
            )
            conformer_id = int(selection.conf_id)
            energy = float(selection.energy)
            coordinate_mol = Chem.RemoveHs(mol_h)
            if coordinate_mol.GetNumAtoms() != heavy_count:
                raise ValueError("remove_hs_heavy_atom_count_mismatch")
            conformer = coordinate_mol.GetConformer(conformer_id)
            geometry_is_3d = True
            geometry_source = "etkdgv3_mmff94_relax200"

        positions = torch.tensor(
            [
                [
                    float(conformer.GetAtomPosition(idx).x),
                    float(conformer.GetAtomPosition(idx).y),
                    float(conformer.GetAtomPosition(idx).z),
                ]
                for idx in range(heavy_count)
            ],
            dtype=torch.float,
        )
        if not bool(torch.isfinite(positions).all()):
            raise ValueError("trimer_nonfinite_coordinates")
        if geometry_is_3d:
            d_left = torch.linalg.vector_norm(
                positions[left_inter[0]] - positions[left_inter[1]]
            )
            d_right = torch.linalg.vector_norm(
                positions[right_inter[0]] - positions[right_inter[1]]
            )
            star_distance = 0.5 * (d_left + d_right)
            star_asymmetry = torch.abs(d_left - d_right)
            star_valid = bool(
                torch.isfinite(star_distance)
                and torch.isfinite(star_asymmetry)
                and float(star_asymmetry) <= 0.15
            )
        else:
            star_distance = positions.new_tensor(0.0)
            star_asymmetry = positions.new_tensor(float("inf"))
            star_valid = False

        sources, targets, bond_types = [], [], []
        for bond in trimer.GetBonds():
            left = int(bond.GetBeginAtomIdx())
            right = int(bond.GetEndAtomIdx())
            code = _bond_code(bond)
            sources.extend((left, right))
            targets.extend((right, left))
            bond_types.extend((code, code))

        central = torch.tensor(unit_atoms[1], dtype=torch.long)
        base_ids = torch.arange(base_count, dtype=torch.long).repeat(3)
        ru_offsets = torch.repeat_interleave(
            torch.tensor([-1, 0, 1], dtype=torch.long), base_count
        )
        central_mask = ru_offsets == 0
        mapping = central[canonical_to_trimer]
        if mapping.numel() != int(data.num_nodes) or bool((mapping < 0).any()):
            raise ValueError("incomplete_o8_to_trimer_mapping")

        data.trimer_pos = positions
        data.trimer_atomic_number = torch.tensor(
            [atom.GetAtomicNum() for atom in trimer.GetAtoms()],
            dtype=torch.long,
        )
        data.trimer_edge_index = torch.tensor(
            [sources, targets], dtype=torch.long
        )
        data.trimer_bond_type = torch.tensor(bond_types, dtype=torch.long)
        data.trimer_base_ru_atom_id = base_ids
        data.trimer_base_ru_atom_index = data.trimer_base_ru_atom_id
        data.trimer_ru_offset = ru_offsets
        data.trimer_central_ru_mask = central_mask
        data.trimer_central_atom_index = central
        data.trimer_central_ru_atom_index = central
        data.mips_to_trimer_central_index = mapping
        data.trimer_geometry_valid = bool(geometry_is_3d)
        data.trimer_geometry_is_3d = torch.tensor(
            geometry_is_3d, dtype=torch.bool
        )
        data.trimer_2d_fallback = torch.tensor(
            not geometry_is_3d, dtype=torch.bool
        )
        data.trimer_geometry_source = geometry_source
        data.star_3d_distance = star_distance.float()
        data.star_3d_asymmetry = star_asymmetry.float()
        data.star_3d_valid = torch.tensor(star_valid, dtype=torch.bool)
        data.trimer_failure_code = (
            "" if geometry_is_3d else "large_trimer_2d_mcl_disabled"
        )
        data.trimer_conformer_energy = float(energy)
        data.trimer_conformer_seed = int(seed)
        data.trimer_conformer_method = TRIMER_MCL_PROTOCOL
        data.trimer_mcl_schema = TRIMER_MCL_SCHEMA
        data.trimer_mcl_schema_version = TRIMER_MCL_SCHEMA_VERSION
        return data
    except Exception as exc:
        reason = f"{type(exc).__name__}:{exc}"
        return _attach_placeholder(data, reason, seed)
