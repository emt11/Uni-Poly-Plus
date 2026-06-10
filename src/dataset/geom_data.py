import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data


CONFORMER_3D_COUNT = 1


def mol2_2Dcoords(mol):
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    assert len(mol.GetAtoms()) == len(coordinates), f"2D coordinates shape is not aligned with {Chem.MolToSmiles(mol)}"
    return coordinates


def mol2_3Dcoords(mol, cnt):
    coordinate_list = []
    coordinates_2d = mol2_2Dcoords(Chem.Mol(mol)).astype(np.float32)

    for seed in range(cnt):
        mol_tmp = Chem.Mol(mol)
        coordinates = coordinates_2d.copy()
        try:
            params = AllChem.ETKDGv3()
            params.randomSeed = seed
            res = AllChem.EmbedMolecule(mol_tmp, params)
            if res == 0:
                try:
                    if AllChem.MMFFHasAllMoleculeParams(mol_tmp):
                        AllChem.MMFFOptimizeMolecule(mol_tmp, maxIters=200)
                    else:
                        AllChem.UFFOptimizeMolecule(mol_tmp, maxIters=200)
                except Exception:
                    pass
                coordinates = mol_tmp.GetConformer().GetPositions().astype(np.float32)
        except Exception:
            pass

        coordinate_list.append(coordinates)

    return coordinate_list


def mol2coords(mol):
    mol = process_star_atoms(mol)
    cnt = CONFORMER_3D_COUNT
    if len(mol.GetAtoms()) > 400:
        coordinates = mol2_2Dcoords(mol).astype(np.float32)
        coordinate_list = [coordinates] * (cnt + 1)
        print("Atom count > 400, using 2D coordinates")
    else:
        coordinate_list = mol2_3Dcoords(mol, cnt)
        coordinate_list.append(mol2_2Dcoords(Chem.Mol(mol)).astype(np.float32))

    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    positions = coordinate_list[0]
    positions_all = np.stack(coordinate_list, axis=0)

    data = Data(
        z=torch.tensor(atomic_numbers, dtype=torch.long),
        pos=torch.tensor(positions, dtype=torch.float),
        pos_confs=torch.tensor(positions_all, dtype=torch.float),
    )
    return data


def process_star_atoms(mol):
    rw_mol = Chem.RWMol(mol)
    star_idx = [
        atom.GetIdx()
        for atom in rw_mol.GetAtoms()
        if atom.GetAtomicNum() == 0
    ]
    if len(star_idx) != 2:
        raise ValueError(f"Star Substitution expects exactly two '*' atoms, got {len(star_idx)}")

    neighbor_idx = []
    for idx in star_idx:
        atom = rw_mol.GetAtomWithIdx(idx)
        neighbors = [
            neighbor.GetIdx()
            for neighbor in atom.GetNeighbors()
            if neighbor.GetAtomicNum() != 0
        ]
        if len(neighbors) != 1:
            raise ValueError(
                f"Star atom {idx} must have exactly one non-star neighbor, got {len(neighbors)}"
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

    substituted_mol = rw_mol.GetMol()
    Chem.SanitizeMol(substituted_mol)
    return Chem.AddHs(substituted_mol)
