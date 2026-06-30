import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data


CONFORMER_3D_COUNT = 4
MAX_EMBED_TRIES_MULTIPLIER = 2


def mol2_2Dcoords(mol):
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    assert len(mol.GetAtoms()) == len(coordinates), f"2D coordinates shape is not aligned with {Chem.MolToSmiles(mol)}"
    return coordinates


def mol2_3Dcoords(mol, cnt, optimizer="auto"):
    coordinate_list = []
    coordinates_2d = mol2_2Dcoords(Chem.Mol(mol)).astype(np.float32)
    optimizer_used = "2d"
    failed_reason = ""

    max_tries = max(cnt, cnt * MAX_EMBED_TRIES_MULTIPLIER)
    for seed in range(max_tries):
        if len(coordinate_list) >= cnt:
            break
        mol_tmp = Chem.Mol(mol)
        try:
            params = AllChem.ETKDGv3()
            params.randomSeed = 42 + seed * 42
            params.maxIterations = 42
            res = AllChem.EmbedMolecule(mol_tmp, params)
            if res == 0:
                try:
                    optimizer_mode = str(optimizer).lower()
                    if optimizer_mode == "uff":
                        AllChem.UFFOptimizeMolecule(mol_tmp, maxIters=200)
                        optimizer_used = "uff"
                    elif optimizer_mode == "auto":
                        if AllChem.MMFFHasAllMoleculeParams(mol_tmp):
                            AllChem.MMFFOptimizeMolecule(mol_tmp, maxIters=200)
                            optimizer_used = "mmff"
                        else:
                            AllChem.UFFOptimizeMolecule(mol_tmp, maxIters=200)
                            optimizer_used = "uff"
                    else:
                        raise ValueError("optimizer must be 'auto' or 'uff'")
                except Exception as exc:
                    failed_reason = str(exc)[:200]
                coordinate_list.append(mol_tmp.GetConformer().GetPositions().astype(np.float32))
        except Exception as exc:
            failed_reason = str(exc)[:200]

    if not coordinate_list:
        coordinate_list = [coordinates_2d.copy()]
        optimizer_used = "2d"

    if len(coordinate_list) < cnt:
        coordinate_list.extend([coordinate_list[-1].copy() for _ in range(cnt - len(coordinate_list))])

    return coordinate_list, optimizer_used, failed_reason


def mol2coords(mol, process_stars=True, optimizer="auto"):
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
    if len(mol.GetAtoms()) > 400:
        coordinates = mol2_2Dcoords(mol).astype(np.float32)
        coordinate_list = [coordinates] * cnt
        optimizer_used = "2d"
        geom_build_ok = False
        geom_failed_reason = "atom_count_gt_400"
        print("Atom count > 400, using 2D coordinates")
    else:
        coordinate_list, optimizer_used, embed_failed_reason = mol2_3Dcoords(mol, cnt, optimizer=optimizer)
        if optimizer_used == "2d":
            geom_build_ok = False
            geom_failed_reason = geom_failed_reason or embed_failed_reason or "3d_embedding_failed"

    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    positions = coordinate_list[0]
    positions_all = np.stack(coordinate_list, axis=0)

    data = Data(
        z=torch.tensor(atomic_numbers, dtype=torch.long),
        pos=torch.tensor(positions, dtype=torch.float),
        pos_confs=torch.tensor(positions_all, dtype=torch.float),
    )
    data.geom_optimizer = str(optimizer).lower()
    data.geom_optimizer_used = optimizer_used
    data.geom_input = geom_input
    data.geom_build_ok = bool(geom_build_ok)
    data.geom_failed_reason = geom_failed_reason
    data.geom_num_confs = int(positions_all.shape[0])
    return data


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
