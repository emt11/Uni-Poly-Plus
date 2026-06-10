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
