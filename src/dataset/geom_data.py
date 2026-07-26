from collections import Counter

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import torch
from torch_geometric.data import Data

from .graph_data import build_periodic_multimer_mol


CONFORMER_3D_COUNT = 8
CONFORMER_3D_KEEP_COUNT = 4
MAX_EMBED_TRIES_MULTIPLIER = 8
CONFORMER_PROFILE = "full"
FAST_FORCE_FIELD_ATOM_LIMIT = 150
FAST_PBC_2D_ATOM_LIMIT = 120
PBC_CELL_VECTOR_MIN_NORM = 1.0
PBC_CELL_VECTOR_MAX_NORM = 100.0
PBC_T_COSINE_MIN = 0.95
PBC_T_RELATIVE_LENGTH_MAX = 0.10
PBC_ATTACHMENT_BOND_RATIO_MIN = 0.75
PBC_ATTACHMENT_BOND_RATIO_MAX = 1.25
FF_GRADIENT_RMS_MAX = 0.05
FF_GRADIENT_MAX = 0.25
FF_PROBE_STEPS = 20
FF_PROBE_ENERGY_DELTA_PER_ATOM_MAX = 5e-5
GEOM_BOND_RATIO_MIN = 0.65
GEOM_BOND_RATIO_MAX = 1.45
GEOM_HEAVY_NONBONDED_MIN = 0.8
SCREW_KABSCH_RMSD_MAX = 1.5
SCREW_ROTATION_CONSISTENCY_DEG = 30.0
SCREW_TRANSLATION_RELATIVE_MAX = 0.30
SCREW_FINAL_RMSD_MAX = 1.5
# A discrete torsion scan is the bounded analogue of the inter-unit dihedral
# sampling used by polymer chain builders such as RadonPy (npj Comput. Mater.
# 2022, 8, 206; doi:10.1038/s41524-022-00906-4).
SCREW_TORSION_ANGLES_DEG = tuple(range(0, 360, 30))
# Force-field score gates are applied to the *center RU* of an analytically
# screw-symmetric finite chain. They reject severe inter-unit strain while
# preserving the exact symmetry rather than relaxing it away.
SCREW_MAX_ENERGY_PER_ATOM = 5.0
SCREW_CENTER_GRADIENT_RMS_MAX = 10.0
SCREW_FIVE_CELL_MIN_DISTANCE = 0.8
SCREW_CONFORMER_RMSD_MIN = 0.25
POLYGEN_PERIODIC_RU_CANDIDATES = (1, 2, 3, 4, 6)
POLYGEN_TRANSVERSE_BOX = 55.0
POLYGEN_TORSION_STARTS_DEG = (-180.0, -120.0, -60.0, 0.0, 60.0, 120.0, 180.0)
POLYGEN_ADAM_STEPS = 150
POLYGEN_LBFGS_STEPS = 20
POLYGEN_BOUNDARY_BOND_MAX_ERROR = 0.15
POLYGEN_BOUNDARY_ANGLE_MAX_ERROR_DEG = 15.0
POLYGEN_BOUNDARY_TORSION_MAX_ERROR_DEG = 30.0
POLYGEN_MIN_NONBONDED_DISTANCE = 0.8
POLYGEN_TOTAL_LOSS_MAX = 2.5
# Embedding an (m+2)-RU context is cubic-to-worse in practice for flexible
# chains. Above this size, initialize the same periodic optimization from the
# open m-RU cell and derive boundary targets from local terminal geometry.
POLYGEN_DIRECT_SEED_MIN_ATOMS = 24


def set_conformer_generation_config(
    conformer_3d_count=None,
    conformer_keep_count=None,
    embed_tries_multiplier=None,
    conformer_profile=None,
):
    global CONFORMER_3D_COUNT, CONFORMER_3D_KEEP_COUNT, MAX_EMBED_TRIES_MULTIPLIER, CONFORMER_PROFILE
    if conformer_profile is not None:
        conformer_profile = str(conformer_profile).lower()
        if conformer_profile not in {"fast", "full", "quality"}:
            raise ValueError("conformer_profile must be 'fast', 'full', or 'quality'")
        CONFORMER_PROFILE = conformer_profile
    if conformer_3d_count is not None:
        conformer_3d_count = int(conformer_3d_count)
        if conformer_3d_count < 1:
            raise ValueError("conformer_3d_count must be >= 1")
        CONFORMER_3D_COUNT = conformer_3d_count
    if conformer_keep_count is not None:
        conformer_keep_count = int(conformer_keep_count)
        if conformer_keep_count < 1:
            raise ValueError("conformer_keep_count must be >= 1")
        CONFORMER_3D_KEEP_COUNT = conformer_keep_count
    if embed_tries_multiplier is not None:
        embed_tries_multiplier = int(embed_tries_multiplier)
        if embed_tries_multiplier < 1:
            raise ValueError("embed_tries_multiplier must be >= 1")
        MAX_EMBED_TRIES_MULTIPLIER = embed_tries_multiplier


def set_screw_quality_config(
    kabsch_rmsd_max=None,
    rotation_consistency_deg=None,
    translation_relative_max=None,
    final_rmsd_max=None,
    gradient_rms_max=None,
    gradient_max=None,
    probe_steps=None,
    probe_energy_delta_per_atom_max=None,
    screw_energy_per_atom_max=None,
    screw_center_gradient_rms_max=None,
):
    global SCREW_KABSCH_RMSD_MAX, SCREW_ROTATION_CONSISTENCY_DEG
    global SCREW_TRANSLATION_RELATIVE_MAX, SCREW_FINAL_RMSD_MAX
    global FF_GRADIENT_RMS_MAX, FF_GRADIENT_MAX, FF_PROBE_STEPS
    global FF_PROBE_ENERGY_DELTA_PER_ATOM_MAX
    global SCREW_MAX_ENERGY_PER_ATOM, SCREW_CENTER_GRADIENT_RMS_MAX
    if kabsch_rmsd_max is not None:
        SCREW_KABSCH_RMSD_MAX = float(kabsch_rmsd_max)
    if rotation_consistency_deg is not None:
        SCREW_ROTATION_CONSISTENCY_DEG = float(rotation_consistency_deg)
    if translation_relative_max is not None:
        SCREW_TRANSLATION_RELATIVE_MAX = float(translation_relative_max)
    if final_rmsd_max is not None:
        SCREW_FINAL_RMSD_MAX = float(final_rmsd_max)
    if gradient_rms_max is not None:
        FF_GRADIENT_RMS_MAX = float(gradient_rms_max)
    if gradient_max is not None:
        FF_GRADIENT_MAX = float(gradient_max)
    if probe_steps is not None:
        FF_PROBE_STEPS = int(probe_steps)
    if probe_energy_delta_per_atom_max is not None:
        FF_PROBE_ENERGY_DELTA_PER_ATOM_MAX = float(probe_energy_delta_per_atom_max)
    if screw_energy_per_atom_max is not None:
        SCREW_MAX_ENERGY_PER_ATOM = float(screw_energy_per_atom_max)
    if screw_center_gradient_rms_max is not None:
        SCREW_CENTER_GRADIENT_RMS_MAX = float(screw_center_gradient_rms_max)


def mol2_2Dcoords(mol):
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    assert len(mol.GetAtoms()) == len(coordinates), f"2D coordinates shape is not aligned with {Chem.MolToSmiles(mol)}"
    return coordinates


def _profile_search_config(cnt):
    """Return bounded ETKDG and force-field budgets for the selected profile."""
    if CONFORMER_PROFILE == "fast":
        # 3 candidates usually finish after 3 successful attempts. Difficult
        # structures still get a bounded random-coordinate recovery budget.
        max_tries = min(8, max(int(cnt), int(cnt) * 3))
        normal_budget = max(1, max_tries // 2)
        return {
            "max_tries": max_tries,
            "normal_budget": normal_budget,
            "normal_iterations": 100,
            "random_iterations": 300,
            "force_field_iterations": 50,
            "embedding_timeout": 5,
        }

    if CONFORMER_PROFILE == "quality":
        return {
            "max_tries": 16,
            "normal_budget": 8,
            "normal_iterations": 500,
            "random_iterations": 2000,
            "force_field_iterations": 500,
            "embedding_timeout": 15,
        }

    max_tries = min(64, max(int(cnt), int(cnt) * int(MAX_EMBED_TRIES_MULTIPLIER)))
    return {
        "max_tries": max_tries,
        "normal_budget": min(32, max_tries),
        "normal_iterations": 500,
        "random_iterations": 2000,
        "force_field_iterations": 500,
        "embedding_timeout": 15,
    }


def _force_field_gradient_metrics(ff, atom_count, atom_indices=None):
    gradient = np.asarray(ff.CalcGrad(), dtype=np.float64).reshape(int(atom_count), 3)
    if atom_indices is not None:
        atom_indices = np.asarray(atom_indices, dtype=np.int64)
        if atom_indices.size:
            gradient = gradient[atom_indices]
    atom_norms = np.linalg.norm(gradient, axis=1)
    return float(np.sqrt(np.mean(atom_norms ** 2))), float(atom_norms.max(initial=0.0))


def _optimize_conformer_and_energy(mol, optimizer, max_iterations):
    """Return force-field metadata for conformer zero, preferring MMFF."""
    if mol.GetNumAtoms() > FAST_FORCE_FIELD_ATOM_LIMIT:
        # Force-field minimization scales poorly for large periodic contexts
        # and cannot be interrupted reliably while RDKit is inside C++. ETKDG
        # supplies the initialization; the periodic optimizer's strict final
        # quality gate remains authoritative for every profile.
        return {
            "optimizer": "etkdg", "converged": False, "force_quality_accepted": False,
            "energy": float(mol.GetNumAtoms()), "gradient_rms": None, "gradient_max": None,
            "probe_energy_delta_per_atom": None,
            "error": "large_molecule_skip_force_field",
        }
    modes = ["uff"] if str(optimizer).lower() == "uff" else ["mmff", "uff"]
    errors = []
    for mode in modes:
        try:
            if mode == "mmff":
                if not AllChem.MMFFHasAllMoleculeParams(mol):
                    continue
                ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
            else:
                if not AllChem.UFFHasAllMoleculeParams(mol):
                    continue
                ff = AllChem.UFFGetMoleculeForceField(mol)
            if ff is None:
                continue
            status = int(ff.Minimize(maxIts=int(max_iterations)))
            energy_before_probe = float(ff.CalcEnergy())
            if status != 0:
                ff.Minimize(maxIts=FF_PROBE_STEPS)
            energy = float(ff.CalcEnergy())
            gradient_rms, gradient_max = _force_field_gradient_metrics(ff, mol.GetNumAtoms())
            probe_delta = (
                abs(energy - energy_before_probe) / max(1, mol.GetNumAtoms())
                if status != 0 else 0.0
            )
            if np.isfinite(energy):
                gradient_accepted = (
                    status != 0
                    and gradient_rms <= FF_GRADIENT_RMS_MAX
                    and gradient_max <= FF_GRADIENT_MAX
                    and probe_delta <= FF_PROBE_ENERGY_DELTA_PER_ATOM_MAX
                )
                return {
                    "optimizer": mode,
                    "converged": status == 0,
                    "force_quality_accepted": status == 0 or gradient_accepted,
                    "energy": energy,
                    "gradient_rms": gradient_rms,
                    "gradient_max": gradient_max,
                    "probe_energy_delta_per_atom": probe_delta,
                    "error": "",
                }
        except Exception as exc:
            errors.append(f"{mode}:{str(exc)[:120]}")
    return {
        "optimizer": "none", "converged": False, "force_quality_accepted": False,
        "energy": None, "gradient_rms": None, "gradient_max": None,
        "probe_energy_delta_per_atom": None, "error": "; ".join(errors)[:240],
    }


def _embed_candidate(
    mol, seed, use_random_coords, max_iterations, optimizer,
    force_field_iterations, embedding_timeout,
):
    candidate = Chem.Mol(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(42 + seed * 42)
    params.maxIterations = int(max_iterations)
    params.timeout = int(embedding_timeout)
    params.useRandomCoords = bool(use_random_coords)
    if AllChem.EmbedMolecule(candidate, params) != 0:
        return None, "embedding_failed"
    force_meta = _optimize_conformer_and_energy(
        candidate, optimizer, force_field_iterations
    )
    if force_meta["energy"] is None:
        # ETKDG still provides a finite 3D initialization when MMFF/UFF does
        # not parameterize an unusual PI1M chemistry (for example hypervalent
        # S/Si). Rejecting it here repeats the same impossible force-field
        # search for every conformer and every candidate unit-cell size.
        # Periodic builders still apply their strict geometry quality gate.
        return {
            "coordinates": candidate.GetConformer().GetPositions().astype(np.float32),
            "optimizer": "etkdg",
            "converged": False,
            "force_quality_accepted": False,
            "energy": float(candidate.GetNumAtoms()),
            "gradient_rms": None,
            "gradient_max": None,
            "probe_energy_delta_per_atom": None,
        }, force_meta["error"] or "force_field_unavailable_etkdg_seed"
    return {
        "coordinates": candidate.GetConformer().GetPositions().astype(np.float32),
        **{key: value for key, value in force_meta.items() if key != "error"},
    }, ""


def mol2_3Dcoords(mol, cnt, optimizer="auto", retain_count=None):
    """Generate bounded low-energy ETKDG conformers for the active profile."""
    coordinates_2d = mol2_2Dcoords(Chem.Mol(mol)).astype(np.float32)
    candidates = []
    failed_reason = ""
    search = _profile_search_config(cnt)
    max_tries = search["max_tries"]
    normal_budget = search["normal_budget"]
    for seed in range(normal_budget):
        candidate, reason = _embed_candidate(
            mol, seed, False, search["normal_iterations"], optimizer,
            search["force_field_iterations"], search["embedding_timeout"],
        )
        if candidate is not None:
            candidates.append(candidate)
        elif reason:
            failed_reason = reason
        if len(candidates) >= cnt:
            break
    if len(candidates) < cnt:
        for seed in range(normal_budget, max_tries):
            candidate, reason = _embed_candidate(
                mol, seed, True, search["random_iterations"], optimizer,
                search["force_field_iterations"], search["embedding_timeout"],
            )
            if candidate is not None:
                candidates.append(candidate)
            elif reason:
                failed_reason = reason
            if len(candidates) >= cnt:
                break
    if not candidates:
        return [coordinates_2d.copy()], {
            "optimizer_counts": {"2d": 1}, "converged_count": 0,
            "candidate_count": 0, "energies": [], "used_2d": True,
            "converged": [],
            "force_quality_accepted": [], "gradient_rms": [], "gradient_max": [],
            "probe_energy_delta_per_atom": [],
            "failed_reason": failed_reason or "3d_embedding_failed",
        }
    # A converged force-field conformer is required for periodic PBC. Keep it
    # ahead of a lower-energy but unconverged candidate so the quality gate can
    # make that decision from actual 3D candidates rather than an arbitrary
    # energy-only top-k truncation.
    candidates.sort(key=lambda item: (not item["force_quality_accepted"], not item["converged"], item["energy"]))
    requested_keep = CONFORMER_3D_KEEP_COUNT if retain_count is None else int(retain_count)
    kept = candidates[:min(max(1, requested_keep), len(candidates))]
    optimizer_counts = {}
    for item in kept:
        optimizer_counts[item["optimizer"]] = optimizer_counts.get(item["optimizer"], 0) + 1
    return [item["coordinates"] for item in kept], {
        "optimizer_counts": optimizer_counts,
        "converged_count": sum(int(item["converged"]) for item in kept),
        "candidate_count": len(candidates),
        "energies": [float(item["energy"]) for item in kept],
        "converged": [bool(item["converged"]) for item in kept],
        "force_quality_accepted": [bool(item["force_quality_accepted"]) for item in kept],
        "gradient_rms": [item["gradient_rms"] for item in kept],
        "gradient_max": [item["gradient_max"] for item in kept],
        "probe_energy_delta_per_atom": [item["probe_energy_delta_per_atom"] for item in kept],
        "used_2d": False,
        "failed_reason": failed_reason,
    }


def mol2coords(mol, process_stars=True, optimizer="auto", force_2d=False):
    geom_input = "raw"
    geom_build_ok = True
    geom_failed_reason = ""
    if process_stars:
        mol, geom_input, geom_build_ok, geom_failed_reason = process_star_atoms(mol)
    else:
        mol = Chem.Mol(mol)
        Chem.SanitizeMol(mol)
        mol = Chem.AddHs(mol)

    cnt = CONFORMER_3D_COUNT
    if force_2d or len(mol.GetAtoms()) > 400:
        coordinates = mol2_2Dcoords(mol).astype(np.float32)
        coordinate_list = [coordinates]
        conformer_meta = {
            "optimizer_counts": {"2d": 1}, "converged_count": 0,
            "candidate_count": 0, "energies": [], "used_2d": True,
            "converged": [],
            "force_quality_accepted": [], "gradient_rms": [], "gradient_max": [],
            "probe_energy_delta_per_atom": [],
            "failed_reason": "forced_2d" if force_2d else "atom_count_gt_400",
        }
        geom_build_ok = False
        geom_failed_reason = "forced_2d" if force_2d else "atom_count_gt_400"
        print("Atom count > 400, using 2D coordinates")
    else:
        coordinate_list, conformer_meta = mol2_3Dcoords(mol, cnt, optimizer=optimizer)
        if conformer_meta["used_2d"]:
            geom_build_ok = False
            geom_failed_reason = geom_failed_reason or conformer_meta["failed_reason"]

    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    positions = coordinate_list[0]
    positions_all = np.stack(coordinate_list, axis=0)

    data = Data(
        z=torch.tensor(atomic_numbers, dtype=torch.long),
        pos=torch.tensor(positions, dtype=torch.float),
        pos_confs=torch.tensor(positions_all, dtype=torch.float),
    )
    data.geom_optimizer = str(optimizer).lower()
    data.geom_optimizer_used = next(iter(conformer_meta["optimizer_counts"]), "2d")
    data.geom_input = geom_input
    data.geom_build_ok = bool(geom_build_ok)
    data.geom_coordinate_ok = bool(geom_build_ok and not conformer_meta["used_2d"])
    data.geom_failed_reason = geom_failed_reason
    data.geom_num_confs = int(positions_all.shape[0])
    data.geom_conformer_energies = torch.tensor(conformer_meta["energies"], dtype=torch.float)
    data.geom_conformer_candidate_count = int(conformer_meta["candidate_count"])
    data.geom_conformer_converged_count = int(conformer_meta["converged_count"])
    data.geom_optimizer_counts = conformer_meta["optimizer_counts"]
    data.geom_force_quality_accepted = torch.tensor(
        conformer_meta.get("force_quality_accepted", []), dtype=torch.bool
    )
    data.geom_gradient_rms = torch.tensor(conformer_meta.get("gradient_rms", []), dtype=torch.float)
    data.geom_gradient_max = torch.tensor(conformer_meta.get("gradient_max", []), dtype=torch.float)
    data.geom_probe_energy_delta_per_atom = torch.tensor(
        conformer_meta.get("probe_energy_delta_per_atom", []), dtype=torch.float
    )
    data.geom_context_id = 2
    return data


def _periodic_fallback(mol, optimizer, reason, force_2d=False):
    """Return a non-periodic repeat-unit fallback; never retain a fake PBC cell."""
    fallback = mol2coords(mol, process_stars=True, optimizer=optimizer, force_2d=force_2d)
    coordinate_ok = bool(getattr(fallback, "geom_coordinate_ok", False))
    fallback.geom_input = "repeat_unit_fallback"
    fallback.geom_context = "periodic_pbc_2d_rejected" if force_2d else "periodic_pbc_fallback"
    fallback.geom_build_ok = False
    fallback.geom_coordinate_ok = coordinate_ok
    fallback.geom_failed_reason = str(reason)[:240]
    fallback.geom_context_id = 1
    fallback.pbc = torch.tensor([False, False, False], dtype=torch.bool)
    fallback.cell = torch.zeros((3, 3), dtype=torch.float)
    fallback.geom_t_method = "not_available"
    reason_text = str(reason)
    if force_2d:
        fallback.geom_pbc_status = "pbc_2d_rejected"
    elif "pbc_3d_unconverged" in reason_text:
        fallback.geom_pbc_status = "pbc_3d_unconverged"
    else:
        fallback.geom_pbc_status = "repeat_unit_fallback"
    # Star-linking removes the two dummy atoms, while star-substitution keeps
    # them as terminal caps. Map graph atoms to the unchanged non-dummy atom
    # order so SCAGE may use a valid Euclidean fallback conformer.
    non_dummy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 0]
    if non_dummy_indices and max(non_dummy_indices) < fallback.z.numel():
        fallback.graph_to_geom_index = torch.tensor(non_dummy_indices, dtype=torch.long)
    return fallback


def _attachment_bond_quality(periodic_mol, coordinates, left_boundary, center_left_boundary,
                             center_right_boundary, right_boundary):
    periodic_table = Chem.GetPeriodicTable()
    pairs = ((left_boundary, center_left_boundary), (center_right_boundary, right_boundary))
    lengths, ratios = [], []
    for first, second in pairs:
        length = float(np.linalg.norm(coordinates[int(first)] - coordinates[int(second)]))
        z_first = periodic_mol.GetAtomWithIdx(int(first)).GetAtomicNum()
        z_second = periodic_mol.GetAtomWithIdx(int(second)).GetAtomicNum()
        expected = float(periodic_table.GetRcovalent(z_first) + periodic_table.GetRcovalent(z_second))
        ratio = length / expected if expected > 1e-8 else float("inf")
        lengths.append(length)
        ratios.append(ratio)
    passed = all(PBC_ATTACHMENT_BOND_RATIO_MIN <= ratio <= PBC_ATTACHMENT_BOND_RATIO_MAX for ratio in ratios)
    return lengths, ratios, passed


def _geometry_quality(mol, coordinates):
    if not np.isfinite(coordinates).all():
        return False, "nonfinite_coordinates", None
    periodic_table = Chem.GetPeriodicTable()
    bond_ratios = []
    bonded_pairs = set()
    for bond in mol.GetBonds():
        first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonded_pairs.add((min(first, second), max(first, second)))
        z_first = mol.GetAtomWithIdx(first).GetAtomicNum()
        z_second = mol.GetAtomWithIdx(second).GetAtomicNum()
        expected = periodic_table.GetRcovalent(z_first) + periodic_table.GetRcovalent(z_second)
        length = float(np.linalg.norm(coordinates[first] - coordinates[second]))
        ratio = length / expected if expected > 1e-8 else float("inf")
        bond_ratios.append(ratio)
    if any(ratio < GEOM_BOND_RATIO_MIN or ratio > GEOM_BOND_RATIO_MAX for ratio in bond_ratios):
        return False, "covalent_bond_length_out_of_range", None

    heavy = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    minimum_nonbonded = float("inf")
    for offset, first in enumerate(heavy):
        for second in heavy[offset + 1:]:
            if (min(first, second), max(first, second)) in bonded_pairs:
                continue
            minimum_nonbonded = min(
                minimum_nonbonded,
                float(np.linalg.norm(coordinates[first] - coordinates[second])),
            )
    if minimum_nonbonded < GEOM_HEAVY_NONBONDED_MIN:
        return False, "heavy_atom_collision", minimum_nonbonded
    return True, "", minimum_nonbonded if np.isfinite(minimum_nonbonded) else None


def _weighted_kabsch(source, target, weights):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("kabsch_coordinate_shape_mismatch")
    if source.shape[0] < 2 or weights.shape != (source.shape[0],):
        raise ValueError("kabsch_requires_two_weighted_points")
    weights = weights / weights.sum()
    source_center = np.sum(source * weights[:, None], axis=0)
    target_center = np.sum(target * weights[:, None], axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    if source.shape[0] == 2:
        source_axis = source[1] - source[0]
        target_axis = target[1] - target[0]
        source_norm = float(np.linalg.norm(source_axis))
        target_norm = float(np.linalg.norm(target_axis))
        if source_norm <= 1e-6 or target_norm <= 1e-6:
            raise ValueError("kabsch_two_point_axis_degenerate")
        source_unit = source_axis / source_norm
        target_unit = target_axis / target_norm
        cross = np.cross(source_unit, target_unit)
        sine = float(np.linalg.norm(cross))
        cosine = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))
        if sine <= 1e-8:
            if cosine > 0.0:
                rotation = np.eye(3)
            else:
                helper = np.array([1.0, 0.0, 0.0])
                if abs(float(np.dot(helper, source_unit))) > 0.9:
                    helper = np.array([0.0, 1.0, 0.0])
                axis = np.cross(source_unit, helper)
                axis /= np.linalg.norm(axis)
                rotation = 2.0 * np.outer(axis, axis) - np.eye(3)
        else:
            axis = cross / sine
            skew = np.array([
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ])
            rotation = np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)
        translation = target_center - rotation @ source_center
        predicted = source @ rotation.T + translation
        rmsd = float(np.sqrt(np.sum(weights * np.sum((predicted - target) ** 2, axis=1))))
        return rotation.astype(np.float32), translation.astype(np.float32), rmsd
    weighted_source = source_zero * np.sqrt(weights[:, None])
    singular_values = np.linalg.svd(weighted_source, compute_uv=False)
    # Two-point repeat units only determine the transport axis, not a unique
    # rotation around that axis.  The SVD still provides a deterministic
    # minimum-rotation solution; the independent left/right fit and final RMSD
    # gates below decide whether that underdetermined fit is usable.
    if source.shape[0] >= 3 and (singular_values.size < 2 or singular_values[1] <= 1e-3):
        raise ValueError("kabsch_points_are_collinear")
    covariance = source_zero.T @ (weights[:, None] * target_zero)
    u_matrix, _, vh_matrix = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(vh_matrix.T @ u_matrix.T))
    rotation = vh_matrix.T @ correction @ u_matrix.T
    translation = target_center - rotation @ source_center
    predicted = source @ rotation.T + translation
    rmsd = float(np.sqrt(np.sum(weights * np.sum((predicted - target) ** 2, axis=1))))
    return rotation.astype(np.float32), translation.astype(np.float32), rmsd


def _project_rotation(matrix):
    u_matrix, _, vh_matrix = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u_matrix @ vh_matrix))
    return (u_matrix @ correction @ vh_matrix).astype(np.float32)


def _rotation_to_quaternion(rotation):
    """Convert a proper rotation matrix to a sign-normalized wxyz quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array([
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ])
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(max(1e-12, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
            quaternion = np.array([
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            ])
        elif axis == 1:
            scale = np.sqrt(max(1e-12, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
            quaternion = np.array([
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            ])
        else:
            scale = np.sqrt(max(1e-12, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
            quaternion = np.array([
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            ])
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("rotation_quaternion_is_degenerate")
    quaternion /= norm
    return quaternion if quaternion[0] >= 0.0 else -quaternion


def _quaternion_to_rotation(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
    w_coord, x_coord, y_coord, z_coord = quaternion
    return np.array([
        [1 - 2 * (y_coord * y_coord + z_coord * z_coord), 2 * (x_coord * y_coord - z_coord * w_coord), 2 * (x_coord * z_coord + y_coord * w_coord)],
        [2 * (x_coord * y_coord + z_coord * w_coord), 1 - 2 * (x_coord * x_coord + z_coord * z_coord), 2 * (y_coord * z_coord - x_coord * w_coord)],
        [2 * (x_coord * z_coord - y_coord * w_coord), 2 * (y_coord * z_coord + x_coord * w_coord), 1 - 2 * (x_coord * x_coord + y_coord * y_coord)],
    ], dtype=np.float32)


def _rotation_midpoint(first, second):
    """Return the geodesic midpoint of two consistent proper rotations."""
    first_quaternion = _rotation_to_quaternion(first)
    second_quaternion = _rotation_to_quaternion(second)
    if float(np.dot(first_quaternion, second_quaternion)) < 0.0:
        second_quaternion = -second_quaternion
    midpoint = first_quaternion + second_quaternion
    if float(np.linalg.norm(midpoint)) <= 1e-12:
        raise ValueError("rotation_midpoint_is_degenerate")
    return _quaternion_to_rotation(midpoint)


def _rotation_difference_degrees(first, second):
    relative = np.asarray(second, dtype=np.float64) @ np.asarray(first, dtype=np.float64).T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _rotation_axis_angle(rotation, translation):
    rotation = np.asarray(rotation, dtype=np.float64)
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle <= 1e-6:
        norm = float(np.linalg.norm(translation))
        axis = np.asarray(translation, dtype=np.float64) / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0])
    else:
        axis = np.array([
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]) / (2.0 * np.sin(angle))
        axis /= max(float(np.linalg.norm(axis)), 1e-8)
    axial_rise = float(np.dot(np.asarray(translation, dtype=np.float64), axis))
    return axis.astype(np.float32), float(np.degrees(angle)), axial_rise


def _axis_angle_rotation(axis, angle_radians):
    """Return the proper rotation around ``axis`` by ``angle_radians``."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-8:
        raise ValueError("screw_attachment_axis_is_degenerate")
    axis = axis / norm
    x_coord, y_coord, z_coord = axis
    cosine, sine = float(np.cos(angle_radians)), float(np.sin(angle_radians))
    one_minus_cosine = 1.0 - cosine
    return np.array([
        [cosine + x_coord * x_coord * one_minus_cosine,
         x_coord * y_coord * one_minus_cosine - z_coord * sine,
         x_coord * z_coord * one_minus_cosine + y_coord * sine],
        [y_coord * x_coord * one_minus_cosine + z_coord * sine,
         cosine + y_coord * y_coord * one_minus_cosine,
         y_coord * z_coord * one_minus_cosine - x_coord * sine],
        [z_coord * x_coord * one_minus_cosine - y_coord * sine,
         z_coord * y_coord * one_minus_cosine + x_coord * sine,
         cosine + z_coord * z_coord * one_minus_cosine],
    ], dtype=np.float64)


def _align_vectors_rotation(source, target):
    """Return a proper rotation mapping one nonzero vector to another."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if source_norm <= 1e-8 or target_norm <= 1e-8:
        raise ValueError("screw_attachment_axis_is_degenerate")
    source = source / source_norm
    target = target / target_norm
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine <= 1e-8:
        if cosine >= 0.0:
            return np.eye(3, dtype=np.float64)
        reference = np.array([1.0, 0.0, 0.0]) if abs(source[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        return _axis_angle_rotation(np.cross(source, reference), np.pi)
    skew = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


def _screw_apply_steps(coordinates, rotation, translation, steps):
    """Apply an exact screw operation an integer number of times to row vectors."""
    result = np.asarray(coordinates, dtype=np.float64).copy()
    if steps >= 0:
        for _ in range(int(steps)):
            result = result @ rotation.T + translation
    else:
        for _ in range(-int(steps)):
            result = (result - translation) @ rotation
    return result


def _evaluate_force_field_energy(mol, optimizer="auto", gradient_atom_indices=None):
    """Evaluate, but never relax, a symmetry-constructed conformer.

    Free force-field minimization would destroy the exact screw relation. The
    optimized capped RU supplies intramolecular relaxation; this energy only
    ranks inter-unit torsion candidates that remain symmetry constrained.
    """
    modes = ["uff"] if str(optimizer).lower() == "uff" else ["mmff", "uff"]
    for mode in modes:
        try:
            if mode == "mmff":
                if not AllChem.MMFFHasAllMoleculeParams(mol):
                    continue
                properties = AllChem.MMFFGetMoleculeProperties(mol)
                force_field = AllChem.MMFFGetMoleculeForceField(mol, properties)
            else:
                if not AllChem.UFFHasAllMoleculeParams(mol):
                    continue
                force_field = AllChem.UFFGetMoleculeForceField(mol)
            if force_field is None:
                continue
            energy = float(force_field.CalcEnergy())
            if np.isfinite(energy):
                gradient_rms, gradient_max = _force_field_gradient_metrics(
                    force_field, mol.GetNumAtoms(), atom_indices=gradient_atom_indices
                )
                return mode, energy, gradient_rms, gradient_max
        except Exception:
            continue
    return "none", None, None, None


def _add_screw_coordinates_to_oligomer(oligomer_info, raw_atom_count, capped_coordinates, rotation, translation):
    """Build a physical trimer whose three units obey the same ``(R, T)``.

    The returned molecule is an explicit finite chain for force-field energy
    evaluation, but every non-terminal atom is placed by the analytical screw
    operation. It is therefore not a free trimer subsequently fitted by
    Kabsch.
    """
    bare = Chem.Mol(oligomer_info["mol"])
    conformer = Chem.Conformer(bare.GetNumAtoms())
    for atom in bare.GetAtoms():
        if not atom.HasProp("_smer_source"):
            raise ValueError("screw_source_mapping_lost")
        source_id = atom.GetIntProp("_smer_source")
        cell_id, source_atom = divmod(int(source_id), int(raw_atom_count))
        coordinates = _screw_apply_steps(
            capped_coordinates[int(source_atom)], rotation, translation, int(cell_id) - 1
        )
        conformer.SetAtomPosition(
            atom.GetIdx(), Point3D(float(coordinates[0]), float(coordinates[1]), float(coordinates[2]))
        )
    bare.RemoveAllConformers()
    bare.AddConformer(conformer, assignId=True)
    # ``addCoords`` places implicit terminal hydrogens from the symmetry-set
    # heavy-atom geometry without moving the latter.
    return Chem.AddHs(bare, addCoords=True)


def _center_atoms_with_hydrogens(mol, center_heavy):
    center_heavy = list(center_heavy)
    center_heavy_set = set(center_heavy)
    center_atoms = list(center_heavy)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1 and len(atom.GetNeighbors()) == 1:
            if atom.GetNeighbors()[0].GetIdx() in center_heavy_set:
                center_atoms.append(atom.GetIdx())
    return center_atoms


def _screw_fallback(mol, optimizer, reason):
    fallback = _periodic_fallback(mol, optimizer, reason, force_2d=False)
    fallback.geom_context = "screw_periodic_fallback"
    fallback.geom_pbc_status = "screw_periodic_fallback"
    fallback.screw_rotation = torch.eye(3, dtype=torch.float)
    fallback.screw_translation = torch.zeros(3, dtype=torch.float)
    fallback.screw_rotation_confs = torch.eye(3, dtype=torch.float).unsqueeze(0)
    fallback.screw_translation_confs = torch.zeros((1, 3), dtype=torch.float)
    fallback.screw_valid = False
    fallback.geom_periodic_mode = "euclidean_fallback" if fallback.geom_coordinate_ok else "topology_fallback"
    fallback.geom_screw_source = fallback.geom_periodic_mode
    fallback.geom_screw_source_id = 1 if fallback.geom_coordinate_ok else 0
    fallback.geometry_source_id = fallback.geom_screw_source_id
    fallback.geom_five_cell_minimum_distance = torch.tensor([float("nan")], dtype=torch.float)
    return fallback


def _smer_fallback(mol, optimizer, reason):
    """Return a clearly labelled non-periodic fallback for s-mer context."""
    fallback = _periodic_fallback(mol, optimizer, reason, force_2d=False)
    fallback.geom_context = "smer_context_fallback"
    fallback.geom_pbc_status = "smer_context_fallback"
    fallback.geom_periodic_mode = "euclidean_fallback" if fallback.geom_coordinate_ok else "topology_fallback"
    fallback.smer_valid = False
    return fallback


def build_capped_linear_oligomer(mol, num_units=3, add_hs=True, terminal_mode="hydrogen"):
    """Build a finite linear oligomer while retaining aromatic valence context.

    Unlike removing ``*`` from an isolated RU first, this connects copies while
    their attachment dummies are still present. Internal dummies are then
    deleted and only the two physical chain ends are hydrogen capped. This
    avoids transient under-valent aromatic atoms such as ``n(*)``.
    """
    if int(num_units) < 3 or int(num_units) % 2 == 0:
        raise ValueError("smer_context_requires_odd_num_units_at_least_three")
    if terminal_mode not in {"hydrogen", "cross_substitution"}:
        raise ValueError("terminal_mode must be 'hydrogen' or 'cross_substitution'")
    source = Chem.Mol(mol)
    star_indices = [atom.GetIdx() for atom in source.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(star_indices) != 2:
        raise ValueError("smer_context_requires_two_attachment_points")
    neighbors, bond_types = [], []
    for star_idx in star_indices:
        star = source.GetAtomWithIdx(int(star_idx))
        star_neighbors = [neighbor.GetIdx() for neighbor in star.GetNeighbors() if neighbor.GetAtomicNum() != 0]
        if len(star_neighbors) != 1:
            raise ValueError("smer_context_attachment_point_must_have_one_neighbor")
        neighbor_idx = star_neighbors[0]
        bond = source.GetBondBetweenAtoms(int(star_idx), int(neighbor_idx))
        if bond is None:
            raise ValueError("smer_context_attachment_bond_missing")
        neighbors.append(neighbor_idx)
        bond_types.append(bond.GetBondType())
    if neighbors[0] == neighbors[1]:
        raise ValueError("smer_context_requires_two_distinct_boundary_atoms")
    if bond_types[0] != bond_types[1]:
        raise ValueError("smer_context_attachment_bond_types_must_match")

    base_atoms = source.GetNumAtoms()
    central_cell = int(num_units) // 2
    rw_mol = Chem.RWMol()
    for cell_idx in range(int(num_units)):
        for atom in source.GetAtoms():
            copied = Chem.Atom(atom)
            # This private source id survives atom deletion, sanitization and
            # AddHs; it lets us recover the center RU without positional
            # assumptions after internal dummy atoms are removed.
            copied.SetIntProp("_smer_source", cell_idx * base_atoms + atom.GetIdx())
            copied.SetAtomMapNum(0)
            rw_mol.AddAtom(copied)
        for bond in source.GetBonds():
            rw_mol.AddBond(
                cell_idx * base_atoms + bond.GetBeginAtomIdx(),
                cell_idx * base_atoms + bond.GetEndAtomIdx(),
                bond.GetBondType(),
            )

    remove_indices = []
    # star_indices[0] is the left endpoint and star_indices[1] the right
    # endpoint for a directed P-SMILES. Reversing both labels is equivalent.
    for cell_idx in range(int(num_units) - 1):
        rw_mol.AddBond(
            cell_idx * base_atoms + neighbors[1],
            (cell_idx + 1) * base_atoms + neighbors[0],
            bond_types[1],
        )
        remove_indices.extend([
            cell_idx * base_atoms + star_indices[1],
            (cell_idx + 1) * base_atoms + star_indices[0],
        ])
    for atom_idx in sorted(set(remove_indices), reverse=True):
        rw_mol.RemoveAtom(int(atom_idx))

    # Only the two outer dummy atoms remain. Hydrogen capping creates a finite
    # molecule accepted by RDKit force fields without claiming it is PBC. A
    # terminal double/triple-bond dummy cannot be hydrogenated without an
    # impossible H valence, so use the opposite attachment-neighbor element
    # there, matching the project's repeat-unit star-substitution convention.
    terminal_replacements = {
        star_indices[0]: source.GetAtomWithIdx(neighbors[1]).GetAtomicNum(),
        (int(num_units) - 1) * base_atoms + star_indices[1]: source.GetAtomWithIdx(neighbors[0]).GetAtomicNum(),
    }
    for atom in rw_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            source_id = atom.GetIntProp("_smer_source") if atom.HasProp("_smer_source") else -1
            local_source_idx = source_id % base_atoms if source_id >= 0 else -1
            attachment_bond = bond_types[star_indices.index(local_source_idx)] if local_source_idx in star_indices else Chem.BondType.SINGLE
            use_cross_cap = terminal_mode == "cross_substitution" or attachment_bond != Chem.BondType.SINGLE
            replacement = terminal_replacements.get(source_id, 1) if use_cross_cap else 1
            atom.SetAtomicNum(int(replacement))
            atom.SetFormalCharge(0)
            atom.SetIsotope(0)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(replacement == 1)

    oligomer = rw_mol.GetMol()
    Chem.SanitizeMol(oligomer)
    if add_hs:
        oligomer = Chem.AddHs(oligomer)

    def mapped_index(cell_idx, source_idx):
        expected = int(cell_idx) * base_atoms + int(source_idx)
        for atom in oligomer.GetAtoms():
            if atom.HasProp("_smer_source") and atom.GetIntProp("_smer_source") == expected:
                return atom.GetIdx()
        raise ValueError("smer_context_source_mapping_lost")

    heavy_source_indices = [
        source_idx for source_idx in range(base_atoms)
        if source_idx not in set(star_indices)
        and source.GetAtomWithIdx(source_idx).GetAtomicNum() > 1
    ]
    left_heavy = [mapped_index(central_cell - 1, source_idx) for source_idx in heavy_source_indices]
    center_heavy = [mapped_index(central_cell, source_idx) for source_idx in heavy_source_indices]
    right_heavy = [mapped_index(central_cell + 1, source_idx) for source_idx in heavy_source_indices]
    center_heavy_set = set(center_heavy)
    center_atoms = list(center_heavy)
    for atom in oligomer.GetAtoms():
        if atom.GetAtomicNum() == 1 and len(atom.GetNeighbors()) == 1:
            if atom.GetNeighbors()[0].GetIdx() in center_heavy_set:
                center_atoms.append(atom.GetIdx())

    return {
        "mol": oligomer,
        "center_atoms": center_atoms,
        "left_heavy": left_heavy,
        "center_heavy": center_heavy,
        "right_heavy": right_heavy,
        "left_boundary": mapped_index(central_cell - 1, neighbors[1]),
        "center_left_boundary": mapped_index(central_cell, neighbors[0]),
        "center_right_boundary": mapped_index(central_cell, neighbors[1]),
        "right_boundary": mapped_index(central_cell + 1, neighbors[0]),
        "num_units": int(num_units),
    }


def mol2smer_context_coords(mol, optimizer="auto"):
    """Generate a center-RU geometry conditioned on an explicit trimer.

    This is intentionally not periodic boundary condition construction. A
    valid result means that an optimized finite oligomer supplied a physical
    left/center/right environment and that the returned coordinates are those
    of its center RU. The SCAGE branch can therefore use an environment-
    conditioned center geometry without assuming a global screw symmetry.
    """
    try:
        oligomer_info = build_capped_linear_oligomer(mol, num_units=3)
        oligomer = oligomer_info["mol"]
        coordinate_list, conformer_meta = mol2_3Dcoords(
            oligomer,
            CONFORMER_3D_COUNT,
            optimizer=optimizer,
            retain_count=CONFORMER_3D_COUNT,
        )
        if conformer_meta["used_2d"]:
            raise ValueError(conformer_meta["failed_reason"] or "smer_context_3d_embedding_failed")

        accepted = []
        for conf_idx, coordinates in enumerate(coordinate_list):
            geometry_ok, geometry_reason, minimum_nonbonded = _geometry_quality(oligomer, coordinates)
            if not geometry_ok:
                continue
            lengths, ratios, attachment_ok = _attachment_bond_quality(
                oligomer,
                coordinates,
                oligomer_info["left_boundary"], oligomer_info["center_left_boundary"],
                oligomer_info["center_right_boundary"], oligomer_info["right_boundary"],
            )
            if not attachment_ok:
                continue
            center_atoms = np.asarray(oligomer_info["center_atoms"], dtype=np.int64)
            center_pos = coordinates[center_atoms].astype(np.float32)
            center_shift = center_pos.mean(axis=0, keepdims=True)
            center_pos = center_pos - center_shift
            image_coordinates = np.stack([
                coordinates[np.asarray(oligomer_info["left_heavy"], dtype=np.int64)] - center_shift,
                coordinates[np.asarray(oligomer_info["center_heavy"], dtype=np.int64)] - center_shift,
                coordinates[np.asarray(oligomer_info["right_heavy"], dtype=np.int64)] - center_shift,
            ]).astype(np.float32)
            accepted.append({
                "coordinates": center_pos,
                "image_coordinates": image_coordinates,
                "energy": float(conformer_meta["energies"][conf_idx]),
                "force_quality_accepted": bool(conformer_meta["force_quality_accepted"][conf_idx]),
                "converged": bool(conformer_meta["converged"][conf_idx]),
                "gradient_rms": conformer_meta["gradient_rms"][conf_idx],
                "gradient_max": conformer_meta["gradient_max"][conf_idx],
                "probe_delta": conformer_meta["probe_energy_delta_per_atom"][conf_idx],
                "attachment_lengths": lengths,
                "attachment_ratios": ratios,
                "minimum_nonbonded": minimum_nonbonded,
            })
        if not accepted:
            raise ValueError("smer_context_quality_gate_failed")

        # Prefer a force-quality accepted candidate, but a finite optimized
        # geometry with valid bonds is still a valid finite-chain context.
        accepted.sort(key=lambda item: (not item["force_quality_accepted"], not item["converged"], item["energy"]))
        accepted = accepted[:min(CONFORMER_3D_KEEP_COUNT, len(accepted))]
        positions = np.stack([item["coordinates"] for item in accepted])
        center_atoms = oligomer_info["center_atoms"]
        center_heavy = oligomer_info["center_heavy"]
        atomic_numbers = [oligomer.GetAtomWithIdx(int(idx)).GetAtomicNum() for idx in center_atoms]
        data = Data(
            z=torch.tensor(atomic_numbers, dtype=torch.long),
            pos=torch.tensor(positions[0], dtype=torch.float),
            pos_confs=torch.tensor(positions, dtype=torch.float),
            cell=torch.zeros((3, 3), dtype=torch.float),
            cell_confs=torch.zeros((len(accepted), 3, 3), dtype=torch.float),
            pbc=torch.tensor([False, False, False], dtype=torch.bool),
            geom_pool_mask=torch.ones(len(center_atoms), dtype=torch.bool),
        )
        data.smer_valid = True
        data.smer_image_pos_confs = torch.tensor(
            np.stack([item["image_coordinates"] for item in accepted]), dtype=torch.float
        )
        data.smer_image_pos = data.smer_image_pos_confs[0]
        data.geom_input = "smer_context"
        data.geom_context = "smer_trimer_center_ru"
        data.geom_periodic_mode = "smer_context"
        data.geom_pbc_status = "smer_3d_valid"
        data.geom_build_ok = True
        data.geom_coordinate_ok = True
        data.geom_failed_reason = ""
        data.geom_context_id = 3
        data.geom_screw_source = "smer_context"
        data.geom_screw_source_id = 2
        data.geometry_source_id = 2
        data.geom_optimizer = str(optimizer).lower()
        data.geom_optimizer_used = next(iter(conformer_meta["optimizer_counts"]), "unknown")
        data.geom_num_confs = len(accepted)
        data.geom_conformer_energies = torch.tensor([item["energy"] for item in accepted], dtype=torch.float)
        data.geom_conformer_candidate_count = int(conformer_meta["candidate_count"])
        data.geom_conformer_converged_count = int(conformer_meta["converged_count"])
        data.geom_optimizer_counts = conformer_meta["optimizer_counts"]
        data.geom_force_quality_accepted = torch.tensor(
            [item["force_quality_accepted"] for item in accepted], dtype=torch.bool
        )
        data.geom_gradient_rms = torch.tensor([item["gradient_rms"] for item in accepted], dtype=torch.float)
        data.geom_gradient_max = torch.tensor([item["gradient_max"] for item in accepted], dtype=torch.float)
        data.geom_probe_energy_delta_per_atom = torch.tensor([item["probe_delta"] for item in accepted], dtype=torch.float)
        data.geom_attachment_bond_lengths = torch.tensor(
            [item["attachment_lengths"] for item in accepted], dtype=torch.float
        )
        data.geom_attachment_bond_ratios = torch.tensor(
            [item["attachment_ratios"] for item in accepted], dtype=torch.float
        )
        data.geom_minimum_nonbonded_distance = torch.tensor(
            [item["minimum_nonbonded"] or float("inf") for item in accepted], dtype=torch.float
        )
        data.graph_to_geom_index = torch.arange(len(center_heavy), dtype=torch.long)
        return data
    except Exception as exc:
        return _smer_fallback(mol, optimizer, f"smer_context_failed: {str(exc)[:180]}")


def _five_cell_minimum_distance(coordinates, rotation, translation, atomic_numbers=None):
    """Return the closest heavy-atom contact between five screw images."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if atomic_numbers is not None:
        heavy_mask = np.asarray(atomic_numbers, dtype=np.int64) > 1
        if heavy_mask.any():
            coordinates = coordinates[heavy_mask]
    images = {0: coordinates}
    for shift in (1, 2):
        images[shift] = images[shift - 1] @ rotation.T + translation
    inverse_rotation = rotation.T
    for shift in (-1, -2):
        images[shift] = (images[shift + 1] - translation) @ inverse_rotation.T
    minimum = float("inf")
    for left_shift in range(-2, 3):
        for right_shift in range(left_shift + 1, 3):
            delta = images[left_shift][:, None, :] - images[right_shift][None, :, :]
            minimum = min(minimum, float(np.linalg.norm(delta, axis=-1).min()))
    return minimum


def _energy_diverse_conformers(candidates, keep_count):
    """Keep low-energy conformers while removing near-identical center RUs."""
    selected = []
    for candidate in sorted(candidates, key=lambda item: item["energy"]):
        coordinates = candidate["coordinates"]
        if any(
            float(np.sqrt(np.mean(np.sum((coordinates - item["coordinates"]) ** 2, axis=1))))
            < SCREW_CONFORMER_RMSD_MIN
            for item in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= int(keep_count):
            break
    if not selected and candidates:
        selected.append(min(candidates, key=lambda item: item["energy"]))
    return selected


def _mol2screw_periodic_coords_trimer_fit(mol, optimizer="auto"):
    """Fit a one-step screw operation from an optimized connected trimer."""
    try:
        trimer = build_periodic_trimer(mol)
        periodic_mol = trimer["mol"]
        coordinate_list, conformer_meta = mol2_3Dcoords(
            periodic_mol,
            CONFORMER_3D_COUNT,
            optimizer=optimizer,
            retain_count=CONFORMER_3D_COUNT,
        )
        if conformer_meta["used_2d"]:
            raise ValueError(conformer_meta["failed_reason"] or "screw_3d_embedding_failed")

        fit_weights = np.asarray(trimer["fit_weights"], dtype=np.float64)
        left_fit = np.asarray(trimer["left_fit"], dtype=np.int64)
        center_fit = np.asarray(trimer["center_fit"], dtype=np.int64)
        right_fit = np.asarray(trimer["right_fit"], dtype=np.int64)
        center_atoms = np.asarray(trimer["center_atoms"], dtype=np.int64)
        center_heavy = np.asarray(trimer["center_heavy"], dtype=np.int64)
        atomic_numbers = [periodic_mol.GetAtomWithIdx(int(idx)).GetAtomicNum() for idx in center_atoms]

        accepted = []
        failure_counts = Counter()
        force_accepted = []
        force_statuses = []
        for converged, gradient_rms, gradient_max, probe_delta in zip(
            conformer_meta.get("converged", []),
            conformer_meta.get("gradient_rms", []),
            conformer_meta.get("gradient_max", []),
            conformer_meta.get("probe_energy_delta_per_atom", []),
        ):
            gradient_ok = (
                gradient_rms is not None
                and gradient_max is not None
                and np.isfinite(float(gradient_rms))
                and np.isfinite(float(gradient_max))
                and float(gradient_rms) <= FF_GRADIENT_RMS_MAX
                and float(gradient_max) <= FF_GRADIENT_MAX
            )
            probe_stable = (
                probe_delta is not None
                and gradient_rms is not None
                and gradient_max is not None
                and np.isfinite(float(probe_delta))
                and np.isfinite(float(gradient_rms))
                and np.isfinite(float(gradient_max))
                and float(probe_delta) <= FF_PROBE_ENERGY_DELTA_PER_ATOM_MAX
                and float(gradient_rms) <= 4.0 * FF_GRADIENT_RMS_MAX
                and float(gradient_max) <= 4.0 * FF_GRADIENT_MAX
            )
            accepted_force = bool(converged) or gradient_ok or probe_stable
            force_accepted.append(accepted_force)
            force_statuses.append(
                "ff_converged" if converged else (
                    "ff_gradient_accepted" if gradient_ok else (
                        "ff_probe_stable" if probe_stable else "ff_rejected"
                    )
                )
            )
        for conf_idx, coordinates in enumerate(coordinate_list):
            if conf_idx >= len(force_accepted) or not force_accepted[conf_idx]:
                failure_counts["force_field_quality_rejected"] += 1
                continue
            geometry_ok, geometry_reason, minimum_nonbonded = _geometry_quality(periodic_mol, coordinates)
            if not geometry_ok:
                failure_counts[geometry_reason] += 1
                continue
            lengths, ratios, attachment_ok = _attachment_bond_quality(
                periodic_mol, coordinates,
                trimer["left_boundary"], trimer["center_left_boundary"],
                trimer["center_right_boundary"], trimer["right_boundary"],
            )
            if not attachment_ok:
                failure_counts["attachment_bond_length_out_of_range"] += 1
                continue
            try:
                rotation_left, translation_left, rmsd_left = _weighted_kabsch(
                    coordinates[left_fit], coordinates[center_fit], fit_weights
                )
                rotation_right, translation_right, rmsd_right = _weighted_kabsch(
                    coordinates[center_fit], coordinates[right_fit], fit_weights
                )
            except ValueError as exc:
                failure_counts[str(exc)] += 1
                continue
            rotation_difference = _rotation_difference_degrees(rotation_left, rotation_right)
            left_norm = float(np.linalg.norm(translation_left))
            right_norm = float(np.linalg.norm(translation_right))
            translation_difference = float(np.linalg.norm(translation_left - translation_right)) / max(
                (left_norm + right_norm) / 2.0, 1e-8
            )
            if rmsd_left > SCREW_KABSCH_RMSD_MAX or rmsd_right > SCREW_KABSCH_RMSD_MAX:
                failure_counts["kabsch_rmsd_out_of_range"] += 1
                continue
            # Fit one operation to both transitions at once.  Independent
            # left/right fits remain diagnostics; rejecting on their rotations
            # before solving the joint problem incorrectly discards flexible
            # but well-approximated chains because torsion around a two-point
            # attachment axis is underdetermined.
            joint_source = np.concatenate(
                [coordinates[left_fit], coordinates[center_fit]], axis=0
            )
            joint_target = np.concatenate(
                [coordinates[center_fit], coordinates[right_fit]], axis=0
            )
            joint_weights = np.tile(fit_weights, 2)
            rotation, translation, joint_rmsd = _weighted_kabsch(
                joint_source, joint_target, joint_weights
            )
            predicted_center = coordinates[left_fit] @ rotation.T + translation
            predicted_right = coordinates[center_fit] @ rotation.T + translation
            final_rmsd_left = float(np.sqrt(np.sum(fit_weights * np.sum((predicted_center - coordinates[center_fit]) ** 2, axis=1)) / fit_weights.sum()))
            final_rmsd_right = float(np.sqrt(np.sum(fit_weights * np.sum((predicted_right - coordinates[right_fit]) ** 2, axis=1)) / fit_weights.sum()))
            if max(final_rmsd_left, final_rmsd_right) > SCREW_FINAL_RMSD_MAX:
                failure_counts["final_screw_rmsd_out_of_range"] += 1
                continue

            center_pos = coordinates[center_atoms].astype(np.float32)
            center_shift = center_pos.mean(axis=0)
            center_pos = center_pos - center_shift
            centered_translation = rotation @ center_shift + translation - center_shift
            axis, angle_deg, axial_rise = _rotation_axis_angle(rotation, centered_translation)
            five_cell_minimum = _five_cell_minimum_distance(
                center_pos, rotation, centered_translation, atomic_numbers
            )
            if five_cell_minimum < SCREW_FIVE_CELL_MIN_DISTANCE:
                failure_counts["five_cell_collision"] += 1
                continue
            accepted.append({
                "coordinates": center_pos,
                "rotation": rotation,
                "translation": centered_translation.astype(np.float32),
                "energy": float(conformer_meta["energies"][conf_idx]),
                "force_status": force_statuses[conf_idx],
                "gradient_rms": conformer_meta["gradient_rms"][conf_idx],
                "gradient_max": conformer_meta["gradient_max"][conf_idx],
                "probe_delta": conformer_meta["probe_energy_delta_per_atom"][conf_idx],
                "rmsd_left": rmsd_left, "rmsd_right": rmsd_right,
                "final_rmsd_left": final_rmsd_left, "final_rmsd_right": final_rmsd_right,
                "rotation_difference": rotation_difference,
                "translation_difference": translation_difference,
                "joint_rmsd": joint_rmsd,
                "axis": axis, "angle_deg": angle_deg, "axial_rise": axial_rise,
                "attachment_lengths": lengths, "attachment_ratios": ratios,
                "minimum_nonbonded": minimum_nonbonded,
                "five_cell_minimum": five_cell_minimum,
                "fit_point_count": int(fit_weights.size),
            })

        if not accepted:
            reason = failure_counts.most_common(1)[0][0] if failure_counts else "unknown"
            raise ValueError(f"screw_quality_gate_failed:{reason}")
        accepted = _energy_diverse_conformers(accepted, CONFORMER_3D_KEEP_COUNT)
        positions = np.stack([item["coordinates"] for item in accepted])
        rotations = np.stack([item["rotation"] for item in accepted])
        translations = np.stack([item["translation"] for item in accepted])
        data = Data(
            z=torch.tensor(atomic_numbers, dtype=torch.long),
            pos=torch.tensor(positions[0], dtype=torch.float),
            pos_confs=torch.tensor(positions, dtype=torch.float),
            cell=torch.zeros((3, 3), dtype=torch.float),
            cell_confs=torch.zeros((len(accepted), 3, 3), dtype=torch.float),
            pbc=torch.tensor([False, False, False], dtype=torch.bool),
            geom_pool_mask=torch.ones(len(atomic_numbers), dtype=torch.bool),
            screw_rotation=torch.tensor(rotations[0], dtype=torch.float),
            screw_translation=torch.tensor(translations[0], dtype=torch.float),
            screw_rotation_confs=torch.tensor(rotations, dtype=torch.float),
            screw_translation_confs=torch.tensor(translations, dtype=torch.float),
        )
        data.screw_valid = True
        data.geom_input = "screw_periodic"
        data.geom_context = "screw_periodic_1d_quality"
        data.geom_periodic_mode = "screw"
        data.geom_pbc_status = "screw_periodic_converged"
        data.geom_build_ok = True
        data.geom_coordinate_ok = True
        data.geom_failed_reason = ""
        data.geom_context_id = 0
        data.geom_num_confs = len(accepted)
        data.geom_conformer_energies = torch.tensor([item["energy"] for item in accepted])
        data.geom_conformer_candidate_count = int(conformer_meta["candidate_count"])
        data.geom_conformer_converged_count = int(conformer_meta["converged_count"])
        data.geom_optimizer_counts = conformer_meta["optimizer_counts"]
        data.geom_force_quality_status = [item["force_status"] for item in accepted]
        data.geom_gradient_rms = torch.tensor([item["gradient_rms"] for item in accepted])
        data.geom_gradient_max = torch.tensor([item["gradient_max"] for item in accepted])
        data.geom_probe_energy_delta_per_atom = torch.tensor([item["probe_delta"] for item in accepted])
        data.geom_kabsch_rmsd_left = torch.tensor([item["rmsd_left"] for item in accepted])
        data.geom_kabsch_rmsd_right = torch.tensor([item["rmsd_right"] for item in accepted])
        data.geom_final_screw_rmsd_left = torch.tensor([item["final_rmsd_left"] for item in accepted])
        data.geom_final_screw_rmsd_right = torch.tensor([item["final_rmsd_right"] for item in accepted])
        data.geom_rotation_consistency_deg = torch.tensor([item["rotation_difference"] for item in accepted])
        data.geom_translation_relative_difference = torch.tensor([item["translation_difference"] for item in accepted])
        data.geom_joint_screw_rmsd = torch.tensor([item["joint_rmsd"] for item in accepted])
        data.geom_screw_axis = torch.tensor(np.stack([item["axis"] for item in accepted]))
        data.geom_screw_angle = torch.tensor([item["angle_deg"] for item in accepted])
        data.geom_screw_axial_rise = torch.tensor([item["axial_rise"] for item in accepted])
        data.geom_minimum_nonbonded_distance = torch.tensor([item["minimum_nonbonded"] or float("inf") for item in accepted])
        data.geom_five_cell_minimum_distance = torch.tensor(
            [item["five_cell_minimum"] for item in accepted], dtype=torch.float
        )
        data.geom_screw_source = "trimer_fit"
        data.geom_screw_source_id = 3
        data.geometry_source_id = 3
        data.geom_screw_fit_point_count = torch.tensor(
            [item["fit_point_count"] for item in accepted], dtype=torch.long
        )
        data.graph_to_geom_index = torch.arange(len(center_heavy), dtype=torch.long)
        return data
    except Exception as exc:
        return _screw_fallback(mol, optimizer, f"screw_periodic_failed: {str(exc)[:180]}")


def _mol2screw_periodic_coords_forward(mol, optimizer="auto"):
    """Construct an exact one-step screw-periodic polymer geometry for SCAGE.

    This follows the forward construction used by polymer chain builders: an
    optimized capped repeat unit supplies attachment vectors, the vectors are
    aligned coaxially, and the inter-unit dihedral is sampled.  Every accepted
    left/center/right atom is then generated from one analytical rigid screw
    operation, instead of fitting a symmetry to an unconstrained trimer after
    optimization.  This is the finite force-field analogue of treating the
    helical twist and rise as structural variables in infinite helical polymer
    theory (Hirata et al., J. Phys. Chem. B 2023, 127, 3556-3583,
    doi:10.1021/acs.jpcb.3c00620).
    """
    try:
        raw_mol = Chem.Mol(mol)
        star_indices = [atom.GetIdx() for atom in raw_mol.GetAtoms() if atom.GetAtomicNum() == 0]
        if len(star_indices) != 2:
            raise ValueError("screw_periodic_requires_two_attachment_points")
        attachment_neighbors = []
        for star_idx in star_indices:
            atom = raw_mol.GetAtomWithIdx(int(star_idx))
            neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() != 0]
            if len(neighbors) != 1:
                raise ValueError("screw_attachment_point_must_have_one_neighbor")
            attachment_neighbors.append(neighbors[0])
        if attachment_neighbors[0] == attachment_neighbors[1]:
            raise ValueError("screw_periodic_requires_two_distinct_boundary_atoms")

        # ``process_star_atoms`` performs cross-substitution at the two ends.
        # Its two cap atoms represent the neighboring boundary atoms required
        # to define the inter-unit bond vectors before the caps are removed.
        capped_mol, _, capped_ok, capped_reason = process_star_atoms(raw_mol)
        if not capped_ok:
            raise ValueError(capped_reason or "screw_cap_construction_failed")
        coordinate_list, conformer_meta = mol2_3Dcoords(
            capped_mol,
            CONFORMER_3D_COUNT,
            optimizer=optimizer,
            retain_count=CONFORMER_3D_COUNT,
        )
        if conformer_meta["used_2d"]:
            raise ValueError(conformer_meta["failed_reason"] or "screw_3d_embedding_failed")

        # Cross-substituted terminal atoms make the explicit force-field
        # trimer geometrically consistent with the capped RU coordinates.
        oligomer_info = build_capped_linear_oligomer(
            raw_mol, num_units=3, add_hs=False, terminal_mode="cross_substitution"
        )
        left_boundary, right_boundary = attachment_neighbors
        left_cap, right_cap = star_indices
        accepted = []
        failure_counts = Counter()

        for conformer_idx, capped_coordinates in enumerate(coordinate_list):
            # RadonPy's chain builder aligns head/tail connection vectors and
            # samples the new-bond dihedral. Here the base alignment maps the
            # left incoming attachment vector to the right outgoing vector;
            # each torsion angle then creates a different exact screw.
            source_axis = capped_coordinates[left_boundary] - capped_coordinates[left_cap]
            target_axis = capped_coordinates[right_cap] - capped_coordinates[right_boundary]
            try:
                base_rotation = _align_vectors_rotation(source_axis, target_axis)
            except ValueError as exc:
                failure_counts[str(exc)] += 1
                continue

            for torsion_degrees in SCREW_TORSION_ANGLES_DEG:
                torsion_rotation = _axis_angle_rotation(target_axis, np.deg2rad(float(torsion_degrees)))
                rotation = (torsion_rotation @ base_rotation).astype(np.float64)
                translation = (
                    capped_coordinates[right_cap].astype(np.float64)
                    - capped_coordinates[left_boundary].astype(np.float64) @ rotation.T
                )
                try:
                    periodic_mol = _add_screw_coordinates_to_oligomer(
                        oligomer_info,
                        raw_mol.GetNumAtoms(),
                        capped_coordinates,
                        rotation,
                        translation,
                    )
                    coordinates = periodic_mol.GetConformer().GetPositions().astype(np.float64)
                    geometry_ok, geometry_reason, minimum_nonbonded = _geometry_quality(periodic_mol, coordinates)
                    if not geometry_ok:
                        failure_counts[geometry_reason] += 1
                        continue
                    lengths, ratios, attachment_ok = _attachment_bond_quality(
                        periodic_mol,
                        coordinates,
                        oligomer_info["left_boundary"], oligomer_info["center_left_boundary"],
                        oligomer_info["center_right_boundary"], oligomer_info["right_boundary"],
                    )
                    if not attachment_ok:
                        failure_counts["attachment_bond_length_out_of_range"] += 1
                        continue
                    ff_name, energy, gradient_rms, gradient_max = _evaluate_force_field_energy(
                        periodic_mol,
                        optimizer,
                        gradient_atom_indices=oligomer_info["center_heavy"],
                    )
                    if energy is None:
                        failure_counts["screw_force_field_energy_unavailable"] += 1
                        continue
                    energy_per_atom = float(energy / max(1, periodic_mol.GetNumAtoms()))
                    if energy_per_atom > SCREW_MAX_ENERGY_PER_ATOM:
                        failure_counts["screw_energy_per_atom_out_of_range"] += 1
                        continue
                    if gradient_rms is None or gradient_rms > SCREW_CENTER_GRADIENT_RMS_MAX:
                        failure_counts["screw_center_gradient_out_of_range"] += 1
                        continue
                except Exception as exc:
                    failure_counts[f"construction:{type(exc).__name__}"] += 1
                    continue

                center_heavy = list(oligomer_info["center_heavy"])
                center_atoms = _center_atoms_with_hydrogens(periodic_mol, center_heavy)
                center_positions = coordinates[np.asarray(center_atoms, dtype=np.int64)].astype(np.float32)
                center_shift = center_positions.mean(axis=0)
                center_positions = center_positions - center_shift
                centered_translation = rotation @ center_shift + translation - center_shift
                axis, angle_degrees, axial_rise = _rotation_axis_angle(rotation, centered_translation)
                # Exact construction makes the symmetry residual numerical
                # roundoff; retain it as an auditable invariant.
                center_heavy_positions = coordinates[np.asarray(center_heavy, dtype=np.int64)]
                mapped_right = _screw_apply_steps(center_heavy_positions, rotation, translation, 1)
                right_positions = np.asarray([
                    coordinates[
                        next(
                            atom.GetIdx() for atom in periodic_mol.GetAtoms()
                            if atom.HasProp("_smer_source")
                            and atom.GetIntProp("_smer_source") == 2 * raw_mol.GetNumAtoms() + source_idx
                        )
                    ]
                    for source_idx in [
                        atom.GetIntProp("_smer_source") % raw_mol.GetNumAtoms()
                        for atom in periodic_mol.GetAtoms()
                        if atom.GetIdx() in center_heavy
                    ]
                ])
                symmetry_rmsd = float(np.sqrt(np.mean(np.sum((mapped_right - right_positions) ** 2, axis=1))))
                accepted.append({
                    "coordinates": center_positions,
                    "rotation": rotation.astype(np.float32),
                    "translation": centered_translation.astype(np.float32),
                    "energy": float(energy),
                    "energy_per_atom": energy_per_atom,
                    "force_name": ff_name,
                    "gradient_rms": gradient_rms,
                    "gradient_max": gradient_max,
                    "torsion_degrees": float(torsion_degrees),
                    "axis": axis,
                    "angle_degrees": angle_degrees,
                    "axial_rise": axial_rise,
                    "symmetry_rmsd": symmetry_rmsd,
                    "attachment_lengths": lengths,
                    "attachment_ratios": ratios,
                    "minimum_nonbonded": minimum_nonbonded,
                    "center_atoms": center_atoms,
                    "center_heavy": center_heavy,
                })

        if not accepted:
            reason = failure_counts.most_common(1)[0][0] if failure_counts else "unknown"
            raise ValueError(f"screw_construct_quality_failed:{reason}")

        # Energy is comparable only within one polymer candidate set. Select
        # the lowest-energy symmetry-preserving torsion variants.
        accepted.sort(key=lambda item: item["energy"])
        accepted = accepted[:min(CONFORMER_3D_KEEP_COUNT, len(accepted))]
        reference = accepted[0]
        positions = np.stack([item["coordinates"] for item in accepted])
        rotations = np.stack([item["rotation"] for item in accepted])
        translations = np.stack([item["translation"] for item in accepted])
        atomic_numbers = [
            periodic_mol.GetAtomWithIdx(int(atom_idx)).GetAtomicNum()
            for atom_idx in reference["center_atoms"]
        ]
        data = Data(
            z=torch.tensor(atomic_numbers, dtype=torch.long),
            pos=torch.tensor(positions[0], dtype=torch.float),
            pos_confs=torch.tensor(positions, dtype=torch.float),
            cell=torch.zeros((3, 3), dtype=torch.float),
            cell_confs=torch.zeros((len(accepted), 3, 3), dtype=torch.float),
            pbc=torch.tensor([False, False, False], dtype=torch.bool),
            geom_pool_mask=torch.ones(len(reference["center_atoms"]), dtype=torch.bool),
            screw_rotation=torch.tensor(rotations[0], dtype=torch.float),
            screw_translation=torch.tensor(translations[0], dtype=torch.float),
            screw_rotation_confs=torch.tensor(rotations, dtype=torch.float),
            screw_translation_confs=torch.tensor(translations, dtype=torch.float),
        )
        data.screw_valid = True
        data.geom_input = "screw_periodic"
        data.geom_context = "screw_periodic_forward_constructed"
        data.geom_periodic_mode = "screw"
        data.geom_pbc_status = "screw_periodic_converged"
        data.geom_build_ok = True
        data.geom_coordinate_ok = True
        data.geom_failed_reason = ""
        data.geom_context_id = 0
        data.geom_optimizer = str(optimizer).lower()
        data.geom_optimizer_used = "screw_constrained_" + reference["force_name"]
        data.geom_num_confs = len(accepted)
        data.geom_conformer_energies = torch.tensor([item["energy"] for item in accepted], dtype=torch.float)
        data.geom_conformer_candidate_count = int(conformer_meta["candidate_count"] * len(SCREW_TORSION_ANGLES_DEG))
        data.geom_conformer_converged_count = int(conformer_meta["converged_count"])
        data.geom_optimizer_counts = {reference["force_name"]: len(accepted)}
        data.geom_force_quality_status = ["screw_constrained_energy_ranked" for _ in accepted]
        data.geom_force_quality_accepted = torch.ones(len(accepted), dtype=torch.bool)
        data.geom_gradient_rms = torch.tensor([item["gradient_rms"] for item in accepted], dtype=torch.float)
        data.geom_gradient_max = torch.tensor([item["gradient_max"] for item in accepted], dtype=torch.float)
        data.geom_probe_energy_delta_per_atom = torch.zeros(len(accepted), dtype=torch.float)
        data.geom_screw_torsion_degrees = torch.tensor([item["torsion_degrees"] for item in accepted], dtype=torch.float)
        data.geom_screw_energy_per_atom = torch.tensor([item["energy_per_atom"] for item in accepted], dtype=torch.float)
        data.geom_screw_symmetry_rmsd = torch.tensor([item["symmetry_rmsd"] for item in accepted], dtype=torch.float)
        data.geom_screw_axis = torch.tensor(np.stack([item["axis"] for item in accepted]), dtype=torch.float)
        data.geom_screw_angle = torch.tensor([item["angle_degrees"] for item in accepted], dtype=torch.float)
        data.geom_screw_axial_rise = torch.tensor([item["axial_rise"] for item in accepted], dtype=torch.float)
        data.geom_attachment_bond_lengths = torch.tensor([item["attachment_lengths"] for item in accepted], dtype=torch.float)
        data.geom_attachment_bond_ratios = torch.tensor([item["attachment_ratios"] for item in accepted], dtype=torch.float)
        data.geom_minimum_nonbonded_distance = torch.tensor(
            [item["minimum_nonbonded"] or float("inf") for item in accepted], dtype=torch.float
        )
        data.geom_five_cell_minimum_distance = torch.tensor(
            [
                _five_cell_minimum_distance(
                    item["coordinates"], item["rotation"], item["translation"], atomic_numbers
                )
                for item in accepted
            ],
            dtype=torch.float,
        )
        data.geom_screw_source = "forward_axis_fallback"
        data.geom_screw_source_id = 2
        data.graph_to_geom_index = torch.arange(len(reference["center_heavy"]), dtype=torch.long)
        return data
    except Exception as exc:
        return _screw_fallback(mol, optimizer, f"screw_periodic_failed: {str(exc)[:180]}")


def mol2screw_periodic_coords(mol, optimizer="auto"):
    """Build a quality-gated screw operation, then fall back without fake PBC."""
    fitted = _mol2screw_periodic_coords_trimer_fit(mol, optimizer=optimizer)
    if bool(getattr(fitted, "screw_valid", False)):
        return fitted
    fitted_reason = str(getattr(fitted, "geom_failed_reason", "trimer_fit_failed"))
    smer = mol2smer_context_coords(mol, optimizer=optimizer)
    if bool(getattr(smer, "smer_valid", False)):
        smer.screw_valid = False
        smer.screw_rotation = torch.eye(3, dtype=torch.float)
        smer.screw_translation = torch.zeros(3, dtype=torch.float)
        smer.screw_rotation_confs = torch.eye(3, dtype=torch.float).unsqueeze(0).repeat(
            int(getattr(smer, "geom_num_confs", 1)), 1, 1
        )
        smer.screw_translation_confs = torch.zeros(
            (int(getattr(smer, "geom_num_confs", 1)), 3), dtype=torch.float
        )
        smer.geom_primary_failed_reason = fitted_reason
        smer.geom_screw_source = "smer_context_fallback"
        smer.geom_screw_source_id = 2
        smer.geometry_source_id = 2
        return smer

    smer_reason = str(getattr(smer, "geom_failed_reason", "smer_context_failed"))
    fallback = _screw_fallback(
        mol, optimizer,
        f"screw_periodic_failed:trimer_fit={fitted_reason[:90]};smer={smer_reason[:90]}",
    )
    fallback.geom_primary_failed_reason = fitted_reason
    return fallback


def _rotation_to_z(axis):
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        raise ValueError("polygen_periodic_degenerate_chain_axis")
    source = axis / norm
    target = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine < 1e-10:
        if cosine > 0:
            return np.eye(3, dtype=np.float32)
        return np.asarray([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    skew = np.asarray([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    rotation = np.eye(3) + skew + (skew @ skew) * ((1.0 - cosine) / (sine ** 2))
    return rotation.astype(np.float32)


def _np_angle(first, center, last):
    left = np.asarray(first) - np.asarray(center)
    right = np.asarray(last) - np.asarray(center)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator < 1e-10:
        raise ValueError("polygen_periodic_degenerate_angle")
    return float(np.arccos(np.clip(np.dot(left, right) / denominator, -1.0, 1.0)))


def _np_dihedral(a, b, c, d):
    p0, p1, p2, p3 = (np.asarray(value, dtype=np.float64) for value in (a, b, c, d))
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = float(np.linalg.norm(b1))
    if b1_norm < 1e-10:
        raise ValueError("polygen_periodic_degenerate_dihedral")
    b1 = b1 / b1_norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    if np.linalg.norm(v) < 1e-10 or np.linalg.norm(w) < 1e-10:
        raise ValueError("polygen_periodic_degenerate_dihedral")
    return float(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def _torch_angles(coordinates, triples):
    if not triples:
        return coordinates.new_zeros((coordinates.size(0), 0))
    index = torch.as_tensor(triples, device=coordinates.device, dtype=torch.long)
    left = coordinates[:, index[:, 0]] - coordinates[:, index[:, 1]]
    right = coordinates[:, index[:, 2]] - coordinates[:, index[:, 1]]
    cosine = (left * right).sum(-1) / (
        left.norm(dim=-1) * right.norm(dim=-1)
    ).clamp_min(1e-8)
    return torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))


def _torch_dihedrals(coordinates, quadruples):
    if not quadruples:
        return coordinates.new_zeros((coordinates.size(0), 0))
    index = torch.as_tensor(quadruples, device=coordinates.device, dtype=torch.long)
    p0, p1 = coordinates[:, index[:, 0]], coordinates[:, index[:, 1]]
    p2, p3 = coordinates[:, index[:, 2]], coordinates[:, index[:, 3]]
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1_unit = b1 / b1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    v = b0 - (b0 * b1_unit).sum(-1, keepdim=True) * b1_unit
    w = b2 - (b2 * b1_unit).sum(-1, keepdim=True) * b1_unit
    x = (v * w).sum(-1)
    y = (torch.cross(b1_unit, v, dim=-1) * w).sum(-1)
    # ``atan2(0, 0)`` has undefined gradients for initially collinear chains.
    return torch.atan2(y, x + 1e-8)


def _periodic_internal_terms(periodic_mol, periodic_edge, seed_coordinates):
    periodic_pair = tuple(sorted(int(value) for value in periodic_edge))
    bonds = []
    adjacency = [set() for _ in range(periodic_mol.GetNumAtoms())]
    for bond in periodic_mol.GetBonds():
        begin, end = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
        if tuple(sorted((begin, end))) == periodic_pair:
            continue
        bonds.append((begin, end))
        adjacency[begin].add(end)
        adjacency[end].add(begin)
    angles = []
    for center, neighbors in enumerate(adjacency):
        ordered = sorted(neighbors)
        for first_idx in range(len(ordered)):
            for second_idx in range(first_idx + 1, len(ordered)):
                angles.append((ordered[first_idx], center, ordered[second_idx]))
    torsions = []
    for begin, end in bonds:
        for first in sorted(adjacency[begin] - {end}):
            for last in sorted(adjacency[end] - {begin}):
                torsions.append((first, begin, end, last))
    bond_targets = np.asarray([
        np.linalg.norm(seed_coordinates[first] - seed_coordinates[second])
        for first, second in bonds
    ], dtype=np.float32)
    angle_targets = np.asarray([
        _np_angle(seed_coordinates[first], seed_coordinates[center], seed_coordinates[last])
        for first, center, last in angles
    ], dtype=np.float32)
    torsion_targets = np.asarray([
        _np_dihedral(*(seed_coordinates[idx] for idx in item)) for item in torsions
    ], dtype=np.float32)
    return bonds, angles, torsions, bond_targets, angle_targets, torsion_targets


def _polygen_direct_boundary_targets(seed_coordinates, periodic_mol, periodic_metadata):
    """Derive physically local boundary targets without embedding an (m+2)-mer."""
    left_boundary = int(periodic_metadata["left_boundary"])
    right_boundary = int(periodic_metadata["right_boundary"])
    base_atoms = int(periodic_metadata["base_atom_count"])
    backbone = periodic_metadata["backbone_base"]
    repeat_units = int(periodic_metadata["num_repeat_units"])
    if len(backbone) < 2:
        raise ValueError("polygen_periodic_backbone_too_short")
    left_inner = int(backbone[1])
    right_inner = int((repeat_units - 1) * base_atoms + backbone[-2])

    table = Chem.GetPeriodicTable()
    left_z = periodic_mol.GetAtomWithIdx(left_boundary).GetAtomicNum()
    right_z = periodic_mol.GetAtomWithIdx(right_boundary).GetAtomicNum()
    target_bond = float(table.GetRcovalent(left_z) + table.GetRcovalent(right_z))
    bond_type = periodic_metadata.get("attachment_bond_type", Chem.BondType.SINGLE)
    if bond_type == Chem.BondType.DOUBLE:
        target_bond *= 0.90
    elif bond_type == Chem.BondType.TRIPLE:
        target_bond *= 0.85
    elif bond_type == Chem.BondType.AROMATIC:
        target_bond *= 0.93

    span = float(seed_coordinates[right_boundary, 2] - seed_coordinates[left_boundary, 2])
    cell_length = max(1.0, span + target_bond)
    vector_t = np.asarray([0.0, 0.0, cell_length], dtype=np.float32)
    next_left = seed_coordinates[left_boundary] + vector_t
    next_inner = seed_coordinates[left_inner] + vector_t
    return (
        target_bond,
        _np_angle(seed_coordinates[right_inner], seed_coordinates[right_boundary], next_left),
        _np_angle(seed_coordinates[right_boundary], next_left, next_inner),
        _np_dihedral(
            seed_coordinates[right_inner], seed_coordinates[right_boundary],
            next_left, next_inner,
        ),
    )


def _polygen_periodic_objective(
    raw_fractional,
    raw_log_length,
    seed_coordinates,
    initial_length,
    internal_terms,
    boundary_indices,
    boundary_targets,
    torsion_targets,
    return_components=False,
):
    box = float(POLYGEN_TRANSVERSE_BOX)
    length = torch.exp(raw_log_length).clamp(1.0, 200.0)
    fractional = raw_fractional
    coordinates = torch.stack([
        box * (fractional[..., 0] - 0.5),
        box * (fractional[..., 1] - 0.5),
        length.unsqueeze(-1) * fractional[..., 2],
    ], dim=-1)
    bonds, angles, torsions, bond_target, angle_target, torsion_target = internal_terms
    device, dtype = coordinates.device, coordinates.dtype
    batch_size = coordinates.size(0)
    zero = coordinates.new_zeros(batch_size)

    if bonds:
        indices = torch.as_tensor(bonds, device=device, dtype=torch.long)
        lengths = (coordinates[:, indices[:, 0]] - coordinates[:, indices[:, 1]]).norm(dim=-1)
        bond_loss = ((lengths - bond_target.to(device=device, dtype=dtype)) ** 2).mean(-1)
    else:
        bond_loss = zero
    if angles:
        values = _torch_angles(coordinates, angles)
        angle_loss = (1.0 - torch.cos(values - angle_target.to(device=device, dtype=dtype))).mean(-1)
    else:
        angle_loss = zero
    if torsions:
        values = _torch_dihedrals(coordinates, torsions)
        torsion_loss = (1.0 - torch.cos(values - torsion_target.to(device=device, dtype=dtype))).mean(-1)
    else:
        torsion_loss = zero

    right_inner, right_boundary, left_boundary, left_inner = boundary_indices
    vector_t = coordinates.new_zeros((batch_size, 3))
    vector_t[:, 2] = length
    next_left = coordinates[:, left_boundary] + vector_t
    next_inner = coordinates[:, left_inner] + vector_t
    boundary_length = (coordinates[:, right_boundary] - next_left).norm(dim=-1)
    boundary_bond_loss = (boundary_length - float(boundary_targets[0])) ** 2
    right_angle = _torch_angles(
        torch.stack([coordinates[:, right_inner], coordinates[:, right_boundary], next_left], dim=1),
        [(0, 1, 2)],
    ).squeeze(-1)
    left_angle = _torch_angles(
        torch.stack([coordinates[:, right_boundary], next_left, next_inner], dim=1),
        [(0, 1, 2)],
    ).squeeze(-1)
    boundary_angle_loss = 0.5 * (
        1.0 - torch.cos(right_angle - float(boundary_targets[1]))
        + 1.0 - torch.cos(left_angle - float(boundary_targets[2]))
    )
    boundary_coords = torch.stack([
        coordinates[:, right_inner], coordinates[:, right_boundary], next_left, next_inner
    ], dim=1)
    boundary_torsion = _torch_dihedrals(boundary_coords, [(0, 1, 2, 3)]).squeeze(-1)
    boundary_torsion_loss = 1.0 - torch.cos(boundary_torsion - torsion_targets)

    collision_loss = zero.clone()
    minimum_distance = coordinates.new_full((batch_size,), float("inf"))
    atom_count = coordinates.size(1)
    bonded = {tuple(sorted(item)) for item in bonds}
    for shift in (-2, -1, 0, 1, 2):
        shifted = coordinates + float(shift) * vector_t.unsqueeze(1)
        distances = torch.cdist(coordinates, shifted)
        valid = torch.ones((atom_count, atom_count), device=device, dtype=torch.bool)
        if shift == 0:
            valid.fill_diagonal_(False)
            for first, second in bonded:
                valid[first, second] = False
                valid[second, first] = False
            valid = torch.triu(valid, diagonal=1)
        elif shift == 1:
            valid[right_boundary, left_boundary] = False
        elif shift == -1:
            valid[left_boundary, right_boundary] = False
        selected = distances[:, valid]
        if selected.numel():
            # A full-pair mean dilutes one severe clash by O(N^2) harmless
            # pairs, while the quality gate correctly rejects by minimum
            # distance. Optimize the worst local contacts so the objective and
            # acceptance criterion measure the same steric failure.
            overlap = torch.relu(
                float(POLYGEN_MIN_NONBONDED_DISTANCE) - selected
            ).pow(2)
            top_count = min(16, int(overlap.size(-1)))
            collision_loss = collision_loss + overlap.topk(
                top_count, dim=-1, largest=True, sorted=False
            ).values.mean(-1)
            minimum_distance = torch.minimum(minimum_distance, selected.min(-1).values)

    seed_loss = (coordinates - seed_coordinates.to(device=device, dtype=dtype)).pow(2).mean(dim=(1, 2))
    length_loss = ((length - float(initial_length)) / max(float(initial_length), 1.0)).pow(2)
    # z is deliberately stored in an unwrapped fractional convention so
    # covalent fragments can cross the periodic boundary without breaking
    # their internal geometry. Only the non-periodic transverse coordinates
    # are softly confined to the 55-A vacuum box.
    transverse_fractional = fractional[..., :2]
    bounds_loss = (
        torch.relu(-transverse_fractional).pow(2)
        + torch.relu(transverse_fractional - 1.0).pow(2)
    ).mean(dim=(1, 2))
    total = (
        1.0 * bond_loss
        + 0.5 * angle_loss
        + 0.25 * torsion_loss
        + 3.0 * boundary_bond_loss
        + 1.5 * boundary_angle_loss
        + 0.75 * boundary_torsion_loss
        + 2.0 * collision_loss
        + 0.1 * seed_loss
        + 0.05 * length_loss
        + 10.0 * bounds_loss
    )
    if return_components:
        return total, {
            "coordinates": coordinates,
            "fractional": fractional,
            "length": length,
            "boundary_length": boundary_length,
            "right_angle": right_angle,
            "left_angle": left_angle,
            "boundary_torsion": boundary_torsion,
            "minimum_distance": minimum_distance,
        }
    return total


def _optimize_polygen_periodic_candidate(
    seed_coordinates,
    periodic_mol,
    periodic_metadata,
    boundary_targets,
):
    left_boundary = int(periodic_metadata["left_boundary"])
    right_boundary = int(periodic_metadata["right_boundary"])
    backbone_base = periodic_metadata["backbone_base"]
    base_atoms = int(periodic_metadata["base_atom_count"])
    if len(backbone_base) < 2:
        raise ValueError("polygen_periodic_backbone_too_short")
    left_inner = int(backbone_base[1])
    right_inner = int((periodic_metadata["num_repeat_units"] - 1) * base_atoms + backbone_base[-2])
    boundary_indices = (right_inner, right_boundary, left_boundary, left_inner)

    centered = np.asarray(seed_coordinates, dtype=np.float32).copy()
    centered[:, :2] -= centered[:, :2].mean(axis=0, keepdims=True)
    centered[:, 2] -= centered[:, 2].min()
    target_bond = float(boundary_targets[0])
    span = float(centered[right_boundary, 2] - centered[left_boundary, 2])
    if span < 0:
        centered[:, 2] *= -1.0
        span = -span
    initial_length = max(1.0, span + target_bond)
    fractional = np.stack([
        centered[:, 0] / POLYGEN_TRANSVERSE_BOX + 0.5,
        centered[:, 1] / POLYGEN_TRANSVERSE_BOX + 0.5,
        centered[:, 2] / initial_length,
    ], axis=-1).astype(np.float32)
    starts = len(POLYGEN_TORSION_STARTS_DEG)
    raw_fractional = torch.nn.Parameter(
        torch.tensor(np.repeat(fractional[None, :, :], starts, axis=0), dtype=torch.float)
    )
    raw_log_length = torch.nn.Parameter(
        torch.full((starts,), float(np.log(initial_length)), dtype=torch.float)
    )
    internal_np = _periodic_internal_terms(
        periodic_mol, periodic_metadata["periodic_edge"], centered
    )
    internal_terms = (
        internal_np[0], internal_np[1], internal_np[2],
        torch.tensor(internal_np[3]), torch.tensor(internal_np[4]), torch.tensor(internal_np[5]),
    )
    seed_tensor = torch.tensor(centered, dtype=torch.float).unsqueeze(0)
    torsion_targets = torch.deg2rad(torch.tensor(POLYGEN_TORSION_STARTS_DEG, dtype=torch.float))
    parameters = [raw_fractional, raw_log_length]
    adam = torch.optim.Adam(parameters, lr=0.02)
    for _ in range(POLYGEN_ADAM_STEPS):
        adam.zero_grad(set_to_none=True)
        loss = _polygen_periodic_objective(
            raw_fractional, raw_log_length, seed_tensor, initial_length,
            internal_terms, boundary_indices, boundary_targets, torsion_targets,
        ).sum()
        if not torch.isfinite(loss):
            raise ValueError("polygen_periodic_nonfinite_adam_loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        adam.step()

    lbfgs = torch.optim.LBFGS(
        parameters, max_iter=POLYGEN_LBFGS_STEPS, line_search_fn="strong_wolfe"
    )

    def closure():
        lbfgs.zero_grad(set_to_none=True)
        value = _polygen_periodic_objective(
            raw_fractional, raw_log_length, seed_tensor, initial_length,
            internal_terms, boundary_indices, boundary_targets, torsion_targets,
        ).sum()
        value.backward()
        return value

    lbfgs.step(closure)
    with torch.no_grad():
        losses, values = _polygen_periodic_objective(
            raw_fractional, raw_log_length, seed_tensor, initial_length,
            internal_terms, boundary_indices, boundary_targets, torsion_targets,
            return_components=True,
        )
    records = []
    for idx in range(starts):
        bond_error = abs(float(values["boundary_length"][idx]) - float(boundary_targets[0]))
        angle_errors = [
            abs(float(values["right_angle"][idx]) - float(boundary_targets[1])),
            abs(float(values["left_angle"][idx]) - float(boundary_targets[2])),
        ]
        torsion_error = abs(float(torch.atan2(
            torch.sin(values["boundary_torsion"][idx] - torsion_targets[idx]),
            torch.cos(values["boundary_torsion"][idx] - torsion_targets[idx]),
        )))
        finite = bool(
            torch.isfinite(losses[idx])
            and torch.isfinite(values["coordinates"][idx]).all()
            and torch.isfinite(values["length"][idx])
        )
        violations = []
        if not finite:
            violations.append("nonfinite")
        if float(losses[idx]) > POLYGEN_TOTAL_LOSS_MAX:
            violations.append("total_loss")
        if bond_error > POLYGEN_BOUNDARY_BOND_MAX_ERROR:
            violations.append("boundary_bond")
        if max(angle_errors) > np.deg2rad(POLYGEN_BOUNDARY_ANGLE_MAX_ERROR_DEG):
            violations.append("boundary_angle")
        if torsion_error > np.deg2rad(POLYGEN_BOUNDARY_TORSION_MAX_ERROR_DEG):
            violations.append("boundary_torsion")
        if float(values["minimum_distance"][idx]) < POLYGEN_MIN_NONBONDED_DISTANCE:
            violations.append("nonbonded_collision")
        accepted = not violations
        records.append({
            "accepted": accepted,
            "loss": float(losses[idx]),
            "coordinates": values["coordinates"][idx].detach().cpu().numpy().astype(np.float32),
            "fractional": values["fractional"][idx].detach().cpu().numpy().astype(np.float32),
            "length": float(values["length"][idx]),
            "boundary_bond_error": bond_error,
            "boundary_angle_error_deg": float(np.rad2deg(max(angle_errors))),
            "boundary_torsion_error_deg": float(np.rad2deg(torsion_error)),
            "minimum_nonbonded_distance": float(values["minimum_distance"][idx]),
            "torsion_start_deg": float(POLYGEN_TORSION_STARTS_DEG[idx]),
            "rejection_reasons": violations,
        })
    return records


def _polygen_periodic_fallback(mol, optimizer, reason):
    fallback = mol2smer_context_coords(mol, optimizer=optimizer)
    fallback.geom_input = "polygen_periodic_fallback"
    fallback.geom_context = "finite_context_fallback"
    fallback.geom_build_ok = False
    fallback.geom_failed_reason = str(reason)[:240]
    fallback.geom_pbc_status = "polygen_periodic_fallback"
    fallback.pbc = torch.tensor([False, False, False], dtype=torch.bool)
    fallback.cell = torch.zeros((3, 3), dtype=torch.float)
    fallback.cell_confs = torch.zeros((int(fallback.pos_confs.size(0)), 3, 3), dtype=torch.float)
    fallback.polygen_periodic_valid = False
    fallback.periodic_valid = False
    fallback.periodic_closure_error = float("inf")
    fallback.periodic_ru_count = 1
    fallback.periodic_cell_length = 0.0
    fallback.geometry_source_id = 2 if bool(getattr(fallback, "smer_valid", False)) else 1
    return fallback


def mol2polygen_periodic_coords(mol, optimizer="auto"):
    """Construct an exact translational m-RU cell by constrained fractional optimization."""
    failures = Counter()
    total_candidates = 0
    for repeat_units in POLYGEN_PERIODIC_RU_CANDIDATES:
        accepted_records = []
        try:
            periodic_mol, periodic_meta = build_periodic_multimer_mol(
                mol, repeat_units, close_periodic=True
            )
            use_direct_seed = (
                int(periodic_meta["base_atom_count"]) >= POLYGEN_DIRECT_SEED_MIN_ATOMS
            )
            if use_direct_seed:
                source_mol = periodic_mol
                source_with_h = Chem.AddHs(source_mol)
                source_indices = list(range(source_mol.GetNumAtoms()))
                axis_left = int(periodic_meta["left_boundary"])
                axis_right = int(periodic_meta["right_boundary"])
            else:
                context_mol, context_meta = build_periodic_multimer_mol(
                    mol, repeat_units + 2, close_periodic=False
                )
                source_with_h = Chem.AddHs(context_mol)
                source_indices = [
                    atom_idx
                    for unit_idx in range(1, repeat_units + 1)
                    for atom_idx in context_meta["unit_atoms"][unit_idx]
                ]
                base_backbone = context_meta["backbone_base"]
                if len(base_backbone) < 2:
                    raise ValueError("polygen_periodic_backbone_too_short")
                right_unit = repeat_units
                next_unit = repeat_units + 1
                right_boundary_context = context_meta["unit_right_boundaries"][right_unit]
                left_boundary_context = context_meta["unit_left_boundaries"][next_unit]
                right_inner_context = context_meta["unit_atoms"][right_unit][base_backbone[-2]]
                left_inner_context = context_meta["unit_atoms"][next_unit][base_backbone[1]]
                axis_left = int(context_meta["unit_left_boundaries"][1])
                axis_right = int(context_meta["unit_right_boundaries"][repeat_units])

            # Progressive search: the first finite source is enough for most
            # repeat units. Only retry the full configured conformer budget
            # when every torsion candidate fails the strict periodic gate.
            source_budgets = [1]
            if CONFORMER_3D_COUNT > 1:
                source_budgets.append(CONFORMER_3D_COUNT)
            conformer_meta = None
            for source_budget in source_budgets:
                coordinate_list, current_meta = mol2_3Dcoords(
                    source_with_h,
                    source_budget,
                    optimizer=optimizer,
                    retain_count=min(CONFORMER_3D_KEEP_COUNT, source_budget),
                )
                conformer_meta = current_meta
                if current_meta["used_2d"]:
                    failures[
                        f"m{repeat_units}:source_budget{source_budget}:embedding_failed"
                    ] += 1
                    continue

                for conf_idx, coordinates in enumerate(coordinate_list):
                    if not current_meta.get("force_quality_accepted", [False])[conf_idx]:
                        failures[f"m{repeat_units}:unconverged_source_used"] += 1
                    axis = coordinates[axis_right] - coordinates[axis_left]
                    rotation = _rotation_to_z(axis)
                    rotated = coordinates @ rotation.T
                    seed = rotated[np.asarray(source_indices, dtype=np.int64)]
                    if use_direct_seed:
                        boundary_targets = _polygen_direct_boundary_targets(
                            seed, periodic_mol, periodic_meta
                        )
                    else:
                        boundary_targets = (
                            float(np.linalg.norm(
                                rotated[right_boundary_context] - rotated[left_boundary_context]
                            )),
                            _np_angle(
                                rotated[right_inner_context], rotated[right_boundary_context],
                                rotated[left_boundary_context],
                            ),
                            _np_angle(
                                rotated[right_boundary_context], rotated[left_boundary_context],
                                rotated[left_inner_context],
                            ),
                            _np_dihedral(
                                rotated[right_inner_context], rotated[right_boundary_context],
                                rotated[left_boundary_context], rotated[left_inner_context],
                            ),
                        )
                    records = _optimize_polygen_periodic_candidate(
                        seed, periodic_mol, periodic_meta, boundary_targets
                    )
                    total_candidates += len(records)
                    energy = float(current_meta["energies"][conf_idx])
                    for record in records:
                        record["source_energy"] = energy
                        record["score"] = record["loss"] + 0.02 * (repeat_units - 1)
                        if record["accepted"]:
                            accepted_records.append(record)
                        else:
                            for gate_reason in record.get(
                                "rejection_reasons", ["unknown"]
                            ):
                                failures[f"m{repeat_units}:gate:{gate_reason}"] += 1
                    if accepted_records:
                        break
                if accepted_records:
                    break

            if conformer_meta is None or conformer_meta["used_2d"]:
                raise ValueError("polygen_periodic_3d_embedding_failed")
            if not accepted_records:
                failures[f"m{repeat_units}:quality_gate_failed"] += 1
                continue

            min_energy = min(record["source_energy"] for record in accepted_records)
            for record in accepted_records:
                record["score"] += 0.1 * max(0.0, record["source_energy"] - min_energy) / max(
                    1.0, abs(min_energy)
                )
            accepted_records.sort(key=lambda item: item["score"])
            kept = accepted_records[:min(CONFORMER_3D_KEEP_COUNT, len(accepted_records))]
            positions = np.stack([record["coordinates"] for record in kept], axis=0)
            fractional = np.stack([record["fractional"] for record in kept], axis=0)
            cells = np.zeros((len(kept), 3, 3), dtype=np.float32)
            for idx, record in enumerate(kept):
                cells[idx, 0, 0] = POLYGEN_TRANSVERSE_BOX
                cells[idx, 1, 1] = POLYGEN_TRANSVERSE_BOX
                cells[idx, 2, 2] = record["length"]
            atomic_numbers = [atom.GetAtomicNum() for atom in periodic_mol.GetAtoms()]
            data = Data(
                z=torch.tensor(atomic_numbers, dtype=torch.long),
                pos=torch.tensor(positions[0], dtype=torch.float),
                pos_confs=torch.tensor(positions, dtype=torch.float),
                cell=torch.tensor(cells[0], dtype=torch.float),
                cell_confs=torch.tensor(cells, dtype=torch.float),
                pbc=torch.tensor([False, False, True], dtype=torch.bool),
                geom_pool_mask=torch.ones(len(atomic_numbers), dtype=torch.bool),
                graph_to_geom_index=torch.arange(len(atomic_numbers), dtype=torch.long),
            )
            data.geom_input = "polygen_periodic"
            data.geom_context = "polygen_fractional_periodic_1d"
            data.geom_build_ok = True
            data.geom_coordinate_ok = True
            data.geom_failed_reason = ""
            data.geom_pbc_status = "polygen_periodic_valid"
            data.geom_periodic_mode = "translation_fractional"
            data.geom_polygen_seed_mode = (
                "direct_periodic_cell" if use_direct_seed else "finite_context"
            )
            data.geom_context_id = 0
            data.geom_num_confs = len(kept)
            data.geom_optimizer = str(optimizer).lower()
            data.geom_optimizer_used = next(iter(conformer_meta["optimizer_counts"]), "unknown")
            data.geom_conformer_energies = torch.tensor(
                [record["source_energy"] for record in kept], dtype=torch.float
            )
            data.geom_conformer_candidate_count = int(total_candidates)
            data.geom_conformer_converged_count = int(conformer_meta["converged_count"])
            data.geom_optimizer_counts = conformer_meta["optimizer_counts"]
            data.polygen_periodic_valid = True
            data.periodic_valid = True
            data.periodic_closure_error = 0.0
            data.periodic_ru_count = int(repeat_units)
            data.periodic_cell_length = float(kept[0]["length"])
            data.periodic_fractional_pos = torch.tensor(fractional[0], dtype=torch.float)
            data.periodic_fractional_pos_confs = torch.tensor(fractional, dtype=torch.float)
            data.periodic_optimization_loss = torch.tensor(
                [record["loss"] for record in kept], dtype=torch.float
            )
            data.periodic_boundary_bond_error = torch.tensor(
                [record["boundary_bond_error"] for record in kept], dtype=torch.float
            )
            data.periodic_boundary_angle_error_deg = torch.tensor(
                [record["boundary_angle_error_deg"] for record in kept], dtype=torch.float
            )
            data.periodic_boundary_torsion_error_deg = torch.tensor(
                [record["boundary_torsion_error_deg"] for record in kept], dtype=torch.float
            )
            data.periodic_minimum_nonbonded_distance = torch.tensor(
                [record["minimum_nonbonded_distance"] for record in kept], dtype=torch.float
            )
            data.periodic_torsion_start_deg = torch.tensor(
                [record["torsion_start_deg"] for record in kept], dtype=torch.float
            )
            data.periodic_candidate_count = int(total_candidates)
            data.periodic_failure_counts = dict(failures)
            data.geometry_source_id = 4
            data.screw_valid = False
            data.smer_valid = False
            return data
        except Exception as exc:
            failures[f"m{repeat_units}:{str(exc)[:120]}"] += 1
    reason = failures.most_common(1)[0][0] if failures else "unknown"
    fallback = _polygen_periodic_fallback(
        mol, optimizer, f"polygen_periodic_failed:{reason}"
    )
    fallback.periodic_candidate_count = int(total_candidates)
    fallback.periodic_failure_counts = dict(failures)
    return fallback


def mol2periodic_pbc_coords(mol, optimizer="auto"):
    """Build a quality-gated 1D PBC repeat-unit geometry input."""
    try:
        trimer = build_periodic_trimer(mol)
        periodic_mol = trimer["mol"]
        if CONFORMER_PROFILE == "fast" and periodic_mol.GetNumAtoms() > FAST_PBC_2D_ATOM_LIMIT:
            return _periodic_fallback(
                mol, optimizer, "fast_large_periodic_trimer_2d_rejected", force_2d=True
            )

        coordinate_list, conformer_meta = mol2_3Dcoords(
            periodic_mol,
            CONFORMER_3D_COUNT,
            optimizer=optimizer,
            # PBC validity is not monotonically related to conformer energy.
            # Evaluate every generated candidate before energy top-k selection.
            retain_count=CONFORMER_3D_COUNT,
        )
        if conformer_meta["used_2d"]:
            raise ValueError(conformer_meta["failed_reason"] or "periodic_pbc_3d_embedding_failed")

        converged = conformer_meta.get("converged", [])
        force_accepted = conformer_meta.get("force_quality_accepted", converged)
        if not any(force_accepted):
            raise ValueError("pbc_3d_unconverged")

        center_heavy = np.asarray(trimer["center_heavy"], dtype=np.int64)
        center_atoms = np.asarray(trimer["center_atoms"], dtype=np.int64)
        left_backbone = np.asarray(trimer["left_backbone"], dtype=np.int64)
        center_backbone = np.asarray(trimer["center_backbone"], dtype=np.int64)
        right_backbone = np.asarray(trimer["right_backbone"], dtype=np.int64)
        atomic_numbers = [periodic_mol.GetAtomWithIdx(int(idx)).GetAtomicNum() for idx in center_atoms]

        unit_positions, unit_cells, quality_records, selected_energies = [], [], [], []
        quality_failures = Counter()
        for conf_idx, coordinates in enumerate(coordinate_list):
            if not force_accepted[conf_idx]:
                quality_failures["force_field_unconverged"] += 1
                continue
            if not (len(left_backbone) >= 2 and len(left_backbone) == len(center_backbone) == len(right_backbone)):
                quality_failures["backbone_path_unavailable"] += 1
                continue
            t_left = np.mean(coordinates[center_backbone] - coordinates[left_backbone], axis=0).astype(np.float32)
            t_right = np.mean(coordinates[right_backbone] - coordinates[center_backbone], axis=0).astype(np.float32)
            left_norm, right_norm = float(np.linalg.norm(t_left)), float(np.linalg.norm(t_right))
            if not (np.isfinite(t_left).all() and np.isfinite(t_right).all()):
                quality_failures["nonfinite_periodic_vector"] += 1
                continue
            if not (PBC_CELL_VECTOR_MIN_NORM < left_norm < PBC_CELL_VECTOR_MAX_NORM and
                    PBC_CELL_VECTOR_MIN_NORM < right_norm < PBC_CELL_VECTOR_MAX_NORM):
                quality_failures["periodic_vector_norm_out_of_range"] += 1
                continue
            cosine = float(np.dot(t_left, t_right) / max(left_norm * right_norm, 1e-8))
            relative_delta = abs(left_norm - right_norm) / max((left_norm + right_norm) / 2.0, 1e-8)
            lengths, ratios, bonds_ok = _attachment_bond_quality(
                periodic_mol,
                coordinates,
                trimer["left_boundary"],
                trimer["center_left_boundary"],
                trimer["center_right_boundary"],
                trimer["right_boundary"],
            )
            if cosine < PBC_T_COSINE_MIN:
                quality_failures["t_direction_inconsistent"] += 1
                continue
            if relative_delta > PBC_T_RELATIVE_LENGTH_MAX:
                quality_failures["t_length_inconsistent"] += 1
                continue
            if not bonds_ok:
                quality_failures["attachment_bond_length_out_of_range"] += 1
                continue
            vector_t = ((t_left + t_right) / 2.0).astype(np.float32)
            center_pos = coordinates[center_atoms].astype(np.float32)
            center_pos = center_pos - center_pos.mean(axis=0, keepdims=True)
            cell = np.zeros((3, 3), dtype=np.float32)
            cell[0] = vector_t
            unit_positions.append(center_pos)
            unit_cells.append(cell)
            selected_energies.append(float(conformer_meta["energies"][conf_idx]))
            quality_records.append({
                "t_left_norm": left_norm,
                "t_right_norm": right_norm,
                "t_norm": float(np.linalg.norm(vector_t)),
                "t_cosine_similarity": cosine,
                "t_relative_length_difference": relative_delta,
                "attachment_bond_lengths": lengths,
                "attachment_bond_ratios": ratios,
            })

        if not unit_positions:
            reason = quality_failures.most_common(1)[0][0] if quality_failures else "unknown"
            raise ValueError(f"pbc_quality_gate_failed:{reason}")

        # The source candidates are already sorted by convergence and energy.
        # Keep the lowest-energy conformers only after the periodic quality gate.
        keep_count = min(CONFORMER_3D_KEEP_COUNT, len(unit_positions))
        unit_positions = unit_positions[:keep_count]
        unit_cells = unit_cells[:keep_count]
        selected_energies = selected_energies[:keep_count]
        quality_records = quality_records[:keep_count]

        positions_all = np.stack(unit_positions, axis=0)
        cells_all = np.stack(unit_cells, axis=0)
        data = Data(
            z=torch.tensor(atomic_numbers, dtype=torch.long),
            pos=torch.tensor(positions_all[0], dtype=torch.float),
            pos_confs=torch.tensor(positions_all, dtype=torch.float),
            cell=torch.tensor(cells_all[0], dtype=torch.float),
            cell_confs=torch.tensor(cells_all, dtype=torch.float),
            pbc=torch.tensor([True, False, False], dtype=torch.bool),
            geom_pool_mask=torch.ones(len(atomic_numbers), dtype=torch.bool),
        )
        data.geom_optimizer = str(optimizer).lower()
        data.geom_optimizer_used = next(iter(conformer_meta["optimizer_counts"]), "unknown")
        data.geom_input = "periodic_pbc"
        data.geom_context = "periodic_pbc_1d_quality"
        data.geom_t_method = "left_right_backbone_mean"
        data.geom_pbc_status = "pbc_3d_converged"
        data.geom_build_ok = True
        data.geom_coordinate_ok = True
        data.geom_failed_reason = ""
        data.geom_num_confs = int(positions_all.shape[0])
        data.geom_conformer_energies = torch.tensor(selected_energies, dtype=torch.float)
        data.geom_conformer_candidate_count = int(conformer_meta["candidate_count"])
        data.geom_conformer_converged_count = int(conformer_meta["converged_count"])
        data.geom_force_quality_accepted = True
        data.geom_gradient_rms = torch.tensor(
            [conformer_meta["gradient_rms"][idx] for idx in range(keep_count)], dtype=torch.float
        )
        data.geom_gradient_max = torch.tensor(
            [conformer_meta["gradient_max"][idx] for idx in range(keep_count)], dtype=torch.float
        )
        data.geom_probe_energy_delta_per_atom = torch.tensor(
            [conformer_meta["probe_energy_delta_per_atom"][idx] for idx in range(keep_count)], dtype=torch.float
        )
        data.geom_optimizer_counts = conformer_meta["optimizer_counts"]
        data.geom_context_id = 0
        data.geom_t_left_norm = torch.tensor([item["t_left_norm"] for item in quality_records], dtype=torch.float)
        data.geom_t_right_norm = torch.tensor([item["t_right_norm"] for item in quality_records], dtype=torch.float)
        data.geom_t_cosine_similarity = torch.tensor([item["t_cosine_similarity"] for item in quality_records], dtype=torch.float)
        data.geom_t_relative_length_difference = torch.tensor(
            [item["t_relative_length_difference"] for item in quality_records], dtype=torch.float
        )
        data.geom_attachment_bond_lengths = torch.tensor(
            [item["attachment_bond_lengths"] for item in quality_records], dtype=torch.float
        )
        data.geom_attachment_bond_ratios = torch.tensor(
            [item["attachment_bond_ratios"] for item in quality_records], dtype=torch.float
        )
        data.graph_to_geom_index = torch.arange(len(center_heavy), dtype=torch.long)
        return data
    except Exception as exc:
        return _periodic_fallback(
            mol,
            optimizer,
            f"periodic_pbc_failed: {str(exc)[:180]}",
            # A normal fallback may use 2D coordinates internally when RDKit
            # cannot embed it, but it is always marked non-periodic.
            force_2d=False,
        )


def build_periodic_trimer(mol):
    base_mol, left_idx, right_idx, bond_type = _remove_attachment_stars(mol)
    base_atoms = base_mol.GetNumAtoms()
    try:
        backbone_path = list(Chem.GetShortestPath(base_mol, int(left_idx), int(right_idx)))
    except Exception:
        backbone_path = []
    backbone_set = set(backbone_path)
    neighbor_set = set()
    for atom_idx in backbone_path:
        atom = base_mol.GetAtomWithIdx(int(atom_idx))
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor.GetAtomicNum() > 1 and neighbor_idx not in backbone_set:
                neighbor_set.add(neighbor_idx)
    fit_base = list(backbone_path) + sorted(neighbor_set)
    fit_weights = [
        3.0 if atom_idx in {left_idx, right_idx} else 1.0
        for atom_idx in backbone_path
    ] + [0.25] * len(neighbor_set)
    trimer = Chem.RWMol()

    for _ in range(3):
        for atom in base_mol.GetAtoms():
            trimer.AddAtom(Chem.Atom(atom))

    for cell in range(3):
        offset = cell * base_atoms
        for bond in base_mol.GetBonds():
            trimer.AddBond(
                offset + bond.GetBeginAtomIdx(),
                offset + bond.GetEndAtomIdx(),
                bond.GetBondType(),
            )

    trimer.AddBond(0 * base_atoms + right_idx, 1 * base_atoms + left_idx, bond_type)
    trimer.AddBond(1 * base_atoms + right_idx, 2 * base_atoms + left_idx, bond_type)

    trimer_mol = trimer.GetMol()
    Chem.SanitizeMol(trimer_mol)
    trimer_mol = Chem.AddHs(trimer_mol)

    left_heavy = [idx for idx in range(base_atoms)]
    center_heavy = [base_atoms + idx for idx in range(base_atoms)]
    right_heavy = [2 * base_atoms + idx for idx in range(base_atoms)]
    left_backbone = [idx for idx in backbone_path]
    center_backbone = [base_atoms + idx for idx in backbone_path]
    right_backbone = [2 * base_atoms + idx for idx in backbone_path]
    center_heavy_set = set(center_heavy)
    center_atoms = list(center_heavy)
    for atom in trimer_mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        neighbors = atom.GetNeighbors()
        if len(neighbors) == 1 and neighbors[0].GetIdx() in center_heavy_set:
            center_atoms.append(atom.GetIdx())
    return {
        "mol": trimer_mol,
        "left_heavy": left_heavy,
        "center_heavy": center_heavy,
        "right_heavy": right_heavy,
        "left_backbone": left_backbone,
        "center_backbone": center_backbone,
        "right_backbone": right_backbone,
        "left_fit": [idx for idx in fit_base],
        "center_fit": [base_atoms + idx for idx in fit_base],
        "right_fit": [2 * base_atoms + idx for idx in fit_base],
        "fit_weights": fit_weights,
        "center_atoms": center_atoms,
        # The explicit inter-unit bonds are left.right -> center.left and
        # center.right -> right.left.
        "left_boundary": int(right_idx),
        "center_left_boundary": int(base_atoms + left_idx),
        "center_right_boundary": int(base_atoms + right_idx),
        "right_boundary": int(2 * base_atoms + left_idx),
    }


def _remove_attachment_stars(mol):
    rw_mol = Chem.RWMol(mol)
    star_idx = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(star_idx) != 2:
        raise ValueError("periodic_pbc_requires_two_attachment_points")

    star_set = set(star_idx)
    neighbors = []
    bond_types = []
    for idx in star_idx:
        atom = rw_mol.GetAtomWithIdx(idx)
        non_star_neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() != 0]
        if len(non_star_neighbors) != 1:
            raise ValueError("attachment_point_must_have_one_neighbor")
        neighbor_idx = non_star_neighbors[0]
        neighbors.append(neighbor_idx)
        bond = rw_mol.GetBondBetweenAtoms(idx, neighbor_idx)
        if bond is None:
            raise ValueError("attachment_bond_missing")
        bond_types.append(bond.GetBondType())

    if bond_types[0] != bond_types[1]:
        raise ValueError("attachment_bond_types_must_match")
    if neighbors[0] == neighbors[1]:
        raise ValueError("periodic_pbc_requires_two_distinct_boundary_atoms")

    old_to_new = {}
    next_idx = 0
    for atom in rw_mol.GetAtoms():
        if atom.GetIdx() in star_set:
            continue
        old_to_new[atom.GetIdx()] = next_idx
        next_idx += 1

    for idx in sorted(star_idx, reverse=True):
        rw_mol.RemoveAtom(idx)

    base_mol = rw_mol.GetMol()
    Chem.SanitizeMol(base_mol)
    return base_mol, old_to_new[neighbors[0]], old_to_new[neighbors[1]], bond_types[0]


def process_star_atoms(mol):
    rw_mol = Chem.RWMol(mol)
    star_idx = [
        atom.GetIdx()
        for atom in rw_mol.GetAtoms()
        if atom.GetAtomicNum() == 0
    ]

    if len(star_idx) == 0:
        sanitized = rw_mol.GetMol()
        Chem.SanitizeMol(sanitized)
        return Chem.AddHs(sanitized), "raw", True, ""

    if len(star_idx) != 2:
        return replace_star_atoms_with_hydrogen(
            rw_mol, star_idx, "star_substitution_requires_two_attachment_points"
        )

    neighbor_idx = []
    for idx in star_idx:
        atom = rw_mol.GetAtomWithIdx(idx)
        neighbors = [
            neighbor.GetIdx()
            for neighbor in atom.GetNeighbors()
            if neighbor.GetAtomicNum() != 0
        ]
        if len(neighbors) != 1:
            return replace_star_atoms_with_hydrogen(
                rw_mol, star_idx, "attachment_point_must_have_one_neighbor"
            )
        neighbor_idx.append(neighbors[0])

    replacement_atomic_nums = [
        rw_mol.GetAtomWithIdx(neighbor_idx[1]).GetAtomicNum(),
        rw_mol.GetAtomWithIdx(neighbor_idx[0]).GetAtomicNum(),
    ]

    for idx, atomic_num in zip(star_idx, replacement_atomic_nums):
        atom = rw_mol.GetAtomWithIdx(idx)
        atom.SetAtomicNum(atomic_num)
        atom.SetFormalCharge(0)
        atom.SetIsotope(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)

    try:
        substituted_mol = rw_mol.GetMol()
        Chem.SanitizeMol(substituted_mol)
        return Chem.AddHs(substituted_mol), "star_substitution", True, ""
    except Exception as exc:
        return replace_star_atoms_with_hydrogen(
            Chem.RWMol(mol), star_idx, f"star_substitution_failed: {str(exc)[:160]}"
        )


def replace_star_atoms_with_hydrogen(rw_mol, star_idx, reason="star_substitution_fallback"):
    """Fallback geometry preparation for non-linear or malformed repeat units."""
    for idx in star_idx:
        atom = rw_mol.GetAtomWithIdx(idx)
        atom.SetAtomicNum(1)
        atom.SetFormalCharge(0)
        atom.SetIsotope(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)
    substituted_mol = rw_mol.GetMol()
    Chem.SanitizeMol(substituted_mol)
    return Chem.AddHs(substituted_mol), "hydrogen_fallback", False, reason
