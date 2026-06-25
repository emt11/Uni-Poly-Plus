import torch
import numpy as np
import rdkit.Chem as Chem
from torch_geometric.data import Data

allowable_features = {
    'possible_atom_symbols' : [
        "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na",
        "Ca", "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb",
        "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H",
        "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr",
        "Cr", "Pt", "Hg", "Pb", "Unknown", "*"
    ],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_stereo_list' : [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS
    ]
}

NUM_ATOM_FEATURES = (
    len(allowable_features['possible_atom_symbols']) +
    4 +
    len(allowable_features['possible_hybridization_list']) +
    len(allowable_features['possible_chirality_list']) +
    2
)
NUM_BOND_FEATURES = (
    len(allowable_features['possible_bonds']) +
    len(allowable_features['possible_bond_stereo_list']) +
    2
)


def one_of_k_encoding(x, allowable_set):
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def mol_to_graph_data_obj_simple(mol):
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_symbol = '*' if atom.GetAtomicNum() == 0 else atom.GetSymbol()
        atom_feature = (
            one_of_k_encoding_unk(atom_symbol, allowable_features['possible_atom_symbols']) +
            [
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetTotalNumHs(),
                atom.GetNumRadicalElectrons()
            ] +
            one_of_k_encoding_unk(atom.GetHybridization(), allowable_features['possible_hybridization_list']) +
            one_of_k_encoding_unk(atom.GetChiralTag(), allowable_features['possible_chirality_list']) +
            [
                int(atom.GetIsAromatic()),
                int(atom.IsInRing())
            ]
        )
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.float)

    # bonds
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = (
                one_of_k_encoding_unk(bond.GetBondType(), allowable_features['possible_bonds']) +
                one_of_k_encoding_unk(bond.GetStereo(), allowable_features['possible_bond_stereo_list']) +
                [
                    int(bond.GetIsConjugated()),
                    int(bond.IsInRing())
                ]
            )
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.float)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, NUM_BOND_FEATURES), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def _get_dummy_atoms_and_neighbors(mol):
    dummy_atoms = []
    neighbors = []
    bond_types = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            dummy_atoms.append(atom.GetIdx())
            atom_neighbors = list(atom.GetNeighbors())
            if len(atom_neighbors) != 1:
                raise ValueError("dummy atom must have exactly one neighbor")
            neighbor_idx = atom_neighbors[0].GetIdx()
            neighbors.append(neighbor_idx)
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor_idx)
            bond_types.append(bond.GetBondType())
    return dummy_atoms, neighbors, bond_types


def count_attachment_points(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)


def _sanitize_to_smiles(mol):
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol, canonical=True)


def generate_nmer_smiles(smiles, n, replace_terminal_dummy_atoms=True):
    """Generate a linear n-mer from a two-attachment repeat-unit SMILES."""
    n = int(n)
    if n <= 1:
        return smiles

    monomer = Chem.MolFromSmiles(smiles)
    if monomer is None:
        raise ValueError("invalid smiles")
    dummy_atoms, neighbors, bond_types = _get_dummy_atoms_and_neighbors(monomer)
    if len(dummy_atoms) != 2:
        raise ValueError("n-mer generation requires exactly two dummy atoms")
    if bond_types[0] != bond_types[1]:
        raise ValueError("attachment bond types are not consistent")

    num_atoms = monomer.GetNumAtoms()
    oligomer = monomer
    for _ in range(n - 1):
        oligomer = Chem.CombineMols(oligomer, monomer)

    repeat = np.zeros((n, 2), dtype=np.int64)
    connect = np.zeros((n, 2), dtype=np.int64)
    for i in range(n):
        repeat[i] = np.array(dummy_atoms, dtype=np.int64) + i * num_atoms
        connect[i] = np.array(neighbors, dtype=np.int64) + i * num_atoms

    editable = Chem.EditableMol(oligomer)
    remove_atoms = []
    for i in range(n - 1):
        editable.AddBond(int(connect[i, 1]), int(connect[i + 1, 0]), order=bond_types[0])
        remove_atoms.extend([int(repeat[i, 1]), int(repeat[i + 1, 0])])

    if replace_terminal_dummy_atoms:
        editable.ReplaceAtom(int(repeat[0, 0]), Chem.Atom(1))
        editable.ReplaceAtom(int(repeat[n - 1, 1]), Chem.Atom(1))

    for atom_idx in sorted(remove_atoms, reverse=True):
        editable.RemoveAtom(atom_idx)

    mol = editable.GetMol()
    try:
        mol = Chem.RemoveHs(mol)
    except Exception:
        pass
    return _sanitize_to_smiles(mol)





def build_structure_for_input(smiles, graph_input="repeat_unit"):
    """Build the molecular structure shared by graph and geometry inputs.

    ``repeat_unit`` returns the original repeat-unit molecule. ``dimer`` attempts
    to build a legal two-repeat-unit local segment; when this is not possible,
    it falls back to the repeat-unit molecule and records why.
    """
    graph_input = str(graph_input).lower()
    if graph_input not in {"repeat_unit", "dimer"}:
        raise ValueError("graph_input must be 'repeat_unit' or 'dimer'")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("invalid smiles")

    attachment_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)
    base = {
        "requested_input": graph_input,
        "attachment_count": int(attachment_count),
    }

    if graph_input == "repeat_unit":
        return {
            **base,
            "structure_smiles": smiles,
            "structure_mol": mol,
            "structure_input": "repeat_unit",
            "dimer_build_ok": False,
            "dimer_failed_reason": "",
        }

    if attachment_count != 2:
        return {
            **base,
            "structure_smiles": smiles,
            "structure_mol": mol,
            "structure_input": "repeat_unit",
            "dimer_build_ok": False,
            "dimer_failed_reason": "attachment_count_not_2",
        }

    try:
        dimer_smiles = generate_nmer_smiles(smiles, 2, replace_terminal_dummy_atoms=True)
        dimer_mol = Chem.MolFromSmiles(dimer_smiles)
        if dimer_mol is None:
            raise ValueError("invalid_dimer_smiles")
        return {
            **base,
            "structure_smiles": dimer_smiles,
            "structure_mol": dimer_mol,
            "structure_input": "dimer",
            "dimer_build_ok": True,
            "dimer_failed_reason": "",
        }
    except Exception as exc:
        return {
            **base,
            "structure_smiles": smiles,
            "structure_mol": mol,
            "structure_input": "repeat_unit",
            "dimer_build_ok": False,
            "dimer_failed_reason": str(exc)[:200],
        }


def annotate_structure_fields(data, structure, prefix="graph"):
    """Attach auditable structure metadata to a PyG Data object."""
    setattr(data, f"{prefix}_smiles", structure["structure_smiles"])
    setattr(data, f"{prefix}_input", structure["structure_input"])
    data.requested_graph_input = structure["requested_input"]
    data.structure_smiles = structure["structure_smiles"]
    data.structure_input = structure["structure_input"]
    data.dimer_build_ok = bool(structure["dimer_build_ok"])
    data.dimer_failed_reason = structure["dimer_failed_reason"]
    data.attachment_count = int(structure["attachment_count"])
    return data


def build_graph_for_input(smiles, graph_input="repeat_unit"):
    """Build the graph used by the default GIN backend.

    Graph construction now uses the same resolved structure that geometry can
    reuse, so ``graph_input=dimer`` can feed both graph and GEOM with the same
    dimer molecule.
    """
    structure = build_structure_for_input(smiles, graph_input)
    data = mol_to_graph_data_obj_simple(structure["structure_mol"])
    return annotate_structure_fields(data, structure, prefix="graph")
