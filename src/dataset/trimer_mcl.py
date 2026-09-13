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
import time
import copy
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
TRIMER_TARGET_CONFORMERS = 4
TRIMER_CANDIDATES_PER_ROUND = 8
TRIMER_MAX_ROUNDS = 4
TRIMER_RMSD_THRESHOLD = 0.3
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
    data.conformer_positions = torch.empty((0, 0, 3), dtype=torch.float32)
    data.conformer_energies = torch.empty((0,), dtype=torch.float32)
    data.conformer_round_ids = torch.empty((0,), dtype=torch.long)
    data.conformer_candidate_ids = torch.empty((0,), dtype=torch.long)
    data.num_conformers = 0
    data.target_conformers = TRIMER_TARGET_CONFORMERS
    data.target_met = False
    data.search_stop_reason = str(reason)[:240]
    data.generation_diagnostics = {
        "num_rounds": 0,
        "num_candidates_requested": 0,
        "num_candidates_embedded": 0,
        "num_pre_stereo_rejected": 0,
        "num_post_stereo_rejected": 0,
        "num_geometry_rejected": 0,
        "num_duplicate_rejected": 0,
        "num_valid_candidates": 0,
        "num_final_conformers": 0,
        "embed_time": 0.0,
        "mmff_time": 0.0,
        "total_time": 0.0,
        "trimer_heavy_atoms": 0,
        "trimer_atoms_with_h": 0,
    }
    data.multi_conformer = True
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
class _EnsembleCandidate:
    positions: torch.Tensor
    energy: float
    round_id: int
    candidate_id: int


def _round_seed(sample_key, round_id: int) -> int:
    digest = hashlib.sha256(
        f"{sample_key}:{int(round_id)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def fixed_identity_rmsd(left, right) -> float:
    """Kabsch RMSD with fixed atom identity and proper rotations only."""
    x = torch.as_tensor(left, dtype=torch.float64)
    y = torch.as_tensor(right, dtype=torch.float64)
    if x.shape != y.shape or x.ndim != 2 or x.size(1) != 3 or x.size(0) == 0:
        raise ValueError("fixed-identity RMSD requires matching [N,3] arrays")
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    u, _, vh = torch.linalg.svd(x.T @ y)
    rotation = u @ vh
    if float(torch.linalg.det(rotation)) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    delta = x @ rotation - y
    return float(torch.sqrt(torch.mean(torch.sum(delta * delta, dim=1))))


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
            raise _ExpectedGeometryFailure("STEREO_UNDETERMINED")
        axis = axis / axis_norm
        u, v = xyz[refs[0]] - xyz[i], xyz[refs[1]] - xyz[j]
        u = u - torch.dot(u, axis) * axis
        v = v - torch.dot(v, axis) * axis
        denominator = torch.linalg.vector_norm(u) * torch.linalg.vector_norm(v)
        if not bool(torch.isfinite(denominator)) or float(denominator) <= epsilon:
            raise _ExpectedGeometryFailure("STEREO_UNDETERMINED")
        cosine = float(torch.dot(u, v) / denominator)
        if not torch.isfinite(torch.tensor(cosine)):
            raise _ExpectedGeometryFailure("STEREO_UNDETERMINED")
        expected = -1.0 if stereo in {
            Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS,
        } else 1.0
        signed_cosine = expected * cosine
        if signed_cosine == 0.0:
            raise _ExpectedGeometryFailure("STEREO_UNDETERMINED")
        if signed_cosine < 0.0:
            raise _ExpectedGeometryFailure("STEREO_MISMATCH")
        checked += 1
    return checked


def _validate_trimer_contract(data, trimer, metadata):
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
    unit_bonds = []
    for unit in units:
        reverse = {int(atom): base_id for base_id, atom in enumerate(unit)}
        unit_bonds.append({
            (min(reverse[bond.GetBeginAtomIdx()], reverse[bond.GetEndAtomIdx()]),
             max(reverse[bond.GetBeginAtomIdx()], reverse[bond.GetEndAtomIdx()]),
             _bond_code(bond))
            for bond in trimer.GetBonds()
            if bond.GetBeginAtomIdx() in reverse and bond.GetEndAtomIdx() in reverse
        })
    if any(observed != unit_bonds[0] for observed in unit_bonds[1:]):
        raise TrimerContractError("trimer_ru_internal_bond_contract")
    inter_edges = {tuple(sorted(map(int, edge))) for edge in metadata["inter_unit_edges"]}
    observed = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
                for b in trimer.GetBonds()}
    if len(inter_edges) != 2 or not inter_edges <= observed:
        raise TrimerContractError("trimer_inter_ru_bond_contract")
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


def _attach_shared_topology(data, trimer, metadata, base_count, units, mapping):
    sources, targets, types = [], [], []
    for bond in trimer.GetBonds():
        i, j, code = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), _bond_code(bond)
        sources.extend((i, j)); targets.extend((j, i)); types.extend((code, code))
    central = torch.tensor(units[1], dtype=torch.long)
    data.trimer_atomic_number = torch.tensor(
        [atom.GetAtomicNum() for atom in trimer.GetAtoms()], dtype=torch.long)
    data.trimer_edge_index = torch.tensor([sources, targets], dtype=torch.long)
    data.trimer_bond_type = torch.tensor(types, dtype=torch.long)
    data.trimer_base_ru_atom_id = torch.arange(base_count, dtype=torch.long).repeat(3)
    data.trimer_base_ru_atom_index = data.trimer_base_ru_atom_id
    data.trimer_ru_offset = torch.repeat_interleave(
        torch.tensor([-1, 0, 1], dtype=torch.long), base_count)
    data.trimer_central_ru_mask = data.trimer_ru_offset == 0
    data.trimer_central_atom_index = central
    data.trimer_central_ru_atom_index = central
    data.mips_to_trimer_central_index = central[mapping]


def _deduplicate(candidates, target=4, threshold=0.3):
    kept = []
    duplicates = 0
    for candidate in sorted(candidates, key=lambda c: (c.energy, c.round_id, c.candidate_id)):
        if all(fixed_identity_rmsd(candidate.positions, old.positions) >= float(threshold)
               for old in kept):
            kept.append(candidate)
            if len(kept) == int(target):
                break
        else:
            duplicates += 1
    return kept, duplicates


def attach_finite_trimer_mcl(
    data, smiles, *, num_candidates: int = TRIMER_CANDIDATES_PER_ROUND,
    max_heavy_atoms=None, max_rounds: int = TRIMER_MAX_ROUNDS,
    target_conformers: int = TRIMER_TARGET_CONFORMERS,
    rmsd_threshold: float = TRIMER_RMSD_THRESHOLD,
    timeout_seconds: float = TRIMER_SAMPLE_TIMEOUT_SECONDS,
    sample_key=None,
):
    """Attach a bounded, deterministic ensemble; contract errors remain fatal."""
    del max_heavy_atoms  # Deliberately no atom-count rejection or 2-D fallback.
    seed = _sample_seed(smiles)
    _attach_placeholder(data, "not_built", seed)
    if not bool(getattr(data, "graph_available", True)):
        return _attach_placeholder(data, "graph_unavailable", seed)
    started = time.monotonic()
    trimer, metadata = build_periodic_multimer_mol(smiles, 3, close_periodic=False)
    base_count, units, mapping = _validate_trimer_contract(data, trimer, metadata)
    _attach_shared_topology(data, trimer, metadata, base_count, units, mapping)
    heavy_count = trimer.GetNumAtoms()
    # Building must have propagated all StereoAtoms before any coordinates exist.
    for bond in trimer.GetBonds():
        if bond.GetStereo() in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
                               Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS}:
            refs = tuple(bond.GetStereoAtoms())
            if len(refs) != 2:
                raise TrimerContractError("source_to_trimer_stereo_propagation")
    mol_h_template = Chem.AddHs(Chem.Mol(trimer))
    properties = AllChem.MMFFGetMoleculeProperties(mol_h_template, mmffVariant="MMFF94")
    diagnostics = dict(data.generation_diagnostics)
    diagnostics["trimer_heavy_atoms"] = heavy_count
    diagnostics["trimer_atoms_with_h"] = mol_h_template.GetNumAtoms()
    if properties is None:
        data.conformer_positions = torch.empty((0, heavy_count, 3), dtype=torch.float32)
        data.search_stop_reason = data.trimer_failure_code = "MMFF_UNSUPPORTED"
        data.generation_diagnostics = diagnostics
        return data
    candidates = []
    stop_reason = "max_rounds"
    identity = sample_key if sample_key is not None else Chem.MolToSmiles(trimer, canonical=True)
    for round_id in range(int(max_rounds)):
        if time.monotonic() - started >= float(timeout_seconds):
            stop_reason = "timeout"; break
        diagnostics["num_rounds"] += 1
        diagnostics["num_candidates_requested"] += int(num_candidates)
        embed_started = time.monotonic()
        mol_h, ids = _embed_attempt(
            mol_h_template, num_candidates=int(num_candidates),
            seed=_round_seed(identity, round_id), use_random_coords=False,
            max_iterations=200)
        diagnostics["embed_time"] += time.monotonic() - embed_started
        diagnostics["num_candidates_embedded"] += len(ids)
        if not ids:
            continue
        pre_valid = []
        for candidate_id, conf_id in enumerate(ids):
            if not _conformer_coordinates_are_finite_3d(
                    mol_h, conf_id, mol_h.GetNumAtoms()):
                diagnostics["num_geometry_rejected"] += 1
                continue
            try:
                audit_double_bond_stereo_coordinates(
                    trimer, _coordinates(mol_h, conf_id, heavy_count))
                pre_valid.append((candidate_id, conf_id))
            except _ExpectedGeometryFailure:
                diagnostics["num_pre_stereo_rejected"] += 1
        if not pre_valid:
            continue
        valid_conf_ids = {conf_id for _, conf_id in pre_valid}
        for conf in list(mol_h.GetConformers()):
            if conf.GetId() not in valid_conf_ids:
                mol_h.RemoveConformer(conf.GetId())
        mmff_started = time.monotonic()
        AllChem.MMFFOptimizeMoleculeConfs(
            mol_h, numThreads=1, maxIters=TRIMER_MMFF_RELAX_MAX_ITERATIONS,
            mmffVariant="MMFF94")
        diagnostics["mmff_time"] += time.monotonic() - mmff_started
        round_props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94")
        if round_props is None:
            raise TrimerContractError("MMFF_parameters_changed_after_embedding")
        heavy_mol = Chem.RemoveHs(mol_h)
        if heavy_mol.GetNumAtoms() != heavy_count:
            raise TrimerContractError("remove_hs_heavy_atom_count_mismatch")
        for candidate_id, conf_id in pre_valid:
            energy = _calculate_mmff_energy(mol_h, round_props, conf_id)
            if energy is None or not _conformer_coordinates_are_finite_3d(
                    heavy_mol, conf_id, heavy_count):
                diagnostics["num_geometry_rejected"] += 1; continue
            xyz64 = _coordinates(heavy_mol, conf_id, heavy_count)
            try:
                audit_double_bond_stereo_coordinates(trimer, xyz64)
            except _ExpectedGeometryFailure:
                diagnostics["num_post_stereo_rejected"] += 1; continue
            xyz32 = xyz64.to(torch.float32)
            try:
                audit_double_bond_stereo_coordinates(trimer, xyz32)
            except _ExpectedGeometryFailure:
                diagnostics["num_post_stereo_rejected"] += 1; continue
            candidates.append(_EnsembleCandidate(xyz32, float(energy), round_id, candidate_id))
        kept, _ = _deduplicate(candidates, target_conformers, rmsd_threshold)
        if len(kept) >= int(target_conformers):
            stop_reason = "target_met"; break
    kept, duplicate_count = _deduplicate(candidates, target_conformers, rmsd_threshold)
    diagnostics["num_duplicate_rejected"] = duplicate_count
    diagnostics["num_valid_candidates"] = len(candidates)
    diagnostics["num_final_conformers"] = len(kept)
    ordered_candidates = sorted(
        candidates, key=lambda c: (c.energy, c.round_id, c.candidate_id))
    diagnostics["valid_candidate_order"] = [
        {"energy": c.energy, "round_id": c.round_id,
         "candidate_id": c.candidate_id} for c in ordered_candidates]
    diagnostics["valid_candidate_rmsd"] = [
        [fixed_identity_rmsd(left.positions, right.positions)
         for right in ordered_candidates] for left in ordered_candidates]
    diagnostics["total_time"] = time.monotonic() - started
    k = len(kept)
    data.conformer_positions = (torch.stack([c.positions for c in kept]) if kept
                                else torch.empty((0, heavy_count, 3), dtype=torch.float32))
    data.conformer_energies = torch.tensor([c.energy for c in kept], dtype=torch.float32)
    data.conformer_round_ids = torch.tensor([c.round_id for c in kept], dtype=torch.long)
    data.conformer_candidate_ids = torch.tensor([c.candidate_id for c in kept], dtype=torch.long)
    data.num_conformers = k
    data.target_conformers = int(target_conformers)
    data.target_met = k == int(target_conformers)
    if k:
        data.search_stop_reason = stop_reason
    elif stop_reason == "timeout":
        data.search_stop_reason = "timeout"
    elif diagnostics["num_candidates_embedded"] == 0:
        data.search_stop_reason = "ETKDG_NO_VALID_CONFORMER"
    else:
        data.search_stop_reason = "NO_VALID_CONFORMER"
    data.generation_diagnostics = diagnostics
    data.trimer_geometry_valid = bool(k)
    data.trimer_geometry_is_3d = torch.tensor(bool(k), dtype=torch.bool)
    data.trimer_2d_fallback = torch.tensor(False, dtype=torch.bool)
    data.trimer_geometry_source = "etkdgv3_multiround_mmff94_ensemble" if k else "unavailable"
    data.trimer_failure_code = "" if k else data.search_stop_reason
    data.trimer_conformer_method = "etkdgv3-8x4-mmff94-fixed-identity-rmsd0.3"
    # Ensemble records intentionally have no implicit trimer_pos/conformer-0 view.
    if hasattr(data, "trimer_pos"):
        del data.trimer_pos
    return data


def _validate_ensemble_record(record):
    if not bool(getattr(record, "multi_conformer", False)):
        raise ValueError("record is not a multi-conformer Trimer cache record")
    positions = torch.as_tensor(getattr(record, "conformer_positions", None))
    if positions.ndim != 3 or positions.size(-1) != 3:
        raise ValueError("conformer_positions must have shape [K,N,3]")
    k = int(getattr(record, "num_conformers", -1))
    if k != positions.size(0):
        raise ValueError("num_conformers does not match conformer_positions")
    for name in ("conformer_energies", "conformer_round_ids", "conformer_candidate_ids"):
        if torch.as_tensor(getattr(record, name, None)).numel() != k:
            raise ValueError(f"{name} length mismatch")
    return positions, k


def select_conformer(record, index: int):
    """Return an explicit non-mutating single-conformer view."""
    positions, k = _validate_ensemble_record(record)
    index = int(index)
    if index < 0 or index >= k:
        raise IndexError(index)
    selected = copy.copy(record)
    selected.trimer_pos = positions[index].clone()
    selected.trimer_conformer_energy = float(record.conformer_energies[index])
    selected.selected_conformer_index = index
    selected.trimer_geometry_valid = True
    selected.trimer_geometry_is_3d = torch.tensor(True, dtype=torch.bool)
    edges = torch.as_tensor(selected.trimer_edge_index, dtype=torch.long)
    base = torch.as_tensor(selected.trimer_base_ru_atom_id, dtype=torch.long)
    offsets = torch.as_tensor(selected.trimer_ru_offset, dtype=torch.long)
    seam_lengths = []
    for column in range(0, edges.size(1), 2):
        i, j = int(edges[0, column]), int(edges[1, column])
        if int(offsets[i]) != int(offsets[j]):
            seam_lengths.append(torch.linalg.vector_norm(selected.trimer_pos[i] - selected.trimer_pos[j]))
    if len(seam_lengths) != 2:
        raise TrimerContractError("selected conformer does not have two inter-RU bonds")
    selected.star_3d_distance = torch.stack(seam_lengths).mean().float()
    selected.star_3d_asymmetry = torch.abs(seam_lengths[0] - seam_lengths[1]).float()
    selected.star_3d_valid = torch.tensor(
        bool(torch.isfinite(selected.star_3d_distance)
             and torch.isfinite(selected.star_3d_asymmetry)
             and float(selected.star_3d_asymmetry) <= 0.15), dtype=torch.bool)
    return selected


def iter_conformers(record):
    """Iterate explicit single-conformer views without changing sample identity."""
    _, k = _validate_ensemble_record(record)
    for index in range(k):
        yield select_conformer(record, index)
