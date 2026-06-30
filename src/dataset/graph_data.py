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

GRAPH_EXTRA_ATOM_FEATURES = 3  # is_backbone, is_attachment_neighbor, is_side_chain
GRAPH_EXTRA_BOND_FEATURES = 1  # is_star_linking_edge

NUM_ATOM_FEATURES = (
    len(allowable_features['possible_atom_symbols']) +
    4 +
    len(allowable_features['possible_hybridization_list']) +
    len(allowable_features['possible_chirality_list']) +
    2 +
    GRAPH_EXTRA_ATOM_FEATURES
)
NUM_BOND_FEATURES = (
    len(allowable_features['possible_bonds']) +
    len(allowable_features['possible_bond_stereo_list']) +
    2 +
    GRAPH_EXTRA_BOND_FEATURES
)


def one_of_k_encoding(x, allowable_set):
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def _graph_backbone_annotations(mol, original_neighbors=None, star_link_edge=None):
    n_atoms = mol.GetNumAtoms()
    backbone = set()
    attachment_neighbors = set(int(i) for i in (original_neighbors or []))
    side_chain = set(range(n_atoms))

    if original_neighbors is not None and len(original_neighbors) == 2:
        try:
            path = Chem.rdmolops.GetShortestPath(mol, int(original_neighbors[0]), int(original_neighbors[1]))
            backbone.update(int(i) for i in path)
        except Exception:
            backbone.update(int(i) for i in original_neighbors if int(i) < n_atoms)

    side_chain = side_chain - backbone
    star_link_edge_set = set()
    if star_link_edge is not None:
        i, j = int(star_link_edge[0]), int(star_link_edge[1])
        star_link_edge_set.add((i, j))
        star_link_edge_set.add((j, i))

    return backbone, attachment_neighbors, side_chain, star_link_edge_set


def mol_to_graph_data_obj_simple(mol, backbone_info=None):
    backbone_info = backbone_info or {}
    backbone, attachment_neighbors, side_chain, star_link_edge_set = _graph_backbone_annotations(
        mol,
        original_neighbors=backbone_info.get("neighbors"),
        star_link_edge=backbone_info.get("star_link_edge"),
    )

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
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
                int(atom.IsInRing()),
                int(atom_idx in backbone),
                int(atom_idx in attachment_neighbors),
                int(atom_idx in side_chain),
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
                    int(bond.IsInRing()),
                    int((i, j) in star_link_edge_set),
                ]
            )
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, NUM_BOND_FEATURES), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.has_backbone_features = True
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



def build_star_linking_mol(smiles):
    """Build the induced star-linking graph molecule from a two-attachment P-SMILES.

    The two dummy atoms are removed and their neighboring boundary atoms are
    connected with the original attachment bond type. This is a graph
    construction strategy for topology input, not a physical 3D conformer.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("invalid smiles")

    dummy_atoms, neighbors, bond_types = _get_dummy_atoms_and_neighbors(mol)
    if len(dummy_atoms) != 2:
        raise ValueError("star_linking requires exactly two dummy atoms")
    if len(set(neighbors)) != 2:
        raise ValueError("star_linking requires two distinct boundary atoms")
    if bond_types[0] != bond_types[1]:
        raise ValueError("attachment bond types are not consistent")
    editable = Chem.EditableMol(mol)
    if mol.GetBondBetweenAtoms(int(neighbors[0]), int(neighbors[1])) is None:
        editable.AddBond(int(neighbors[0]), int(neighbors[1]), order=bond_types[0])
    for atom_idx in sorted(dummy_atoms, reverse=True):
        editable.RemoveAtom(int(atom_idx))

    linked_mol = editable.GetMol()
    Chem.SanitizeMol(linked_mol)
    return linked_mol


def build_structure_for_input(smiles, graph_input="repeat_unit"):
    """Resolve the molecular graph used by the GIN backend.

    ``repeat_unit`` keeps the original repeat-unit graph. ``star_linking``
    removes two attachment dummy atoms and connects their boundary atoms. The
    star-linked graph is only used for the graph modality; geometry generation
    stays on the original repeat-unit molecule in the dataset layer.
    """
    graph_input = str(graph_input).lower()
    if graph_input not in {"repeat_unit", "star_linking"}:
        raise ValueError("graph_input must be 'repeat_unit' or 'star_linking'")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("invalid smiles")

    attachment_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)
    backbone_info = {}
    try:
        dummy_atoms, boundary_neighbors, _ = _get_dummy_atoms_and_neighbors(mol)
        if len(boundary_neighbors) == 2:
            backbone_info["neighbors"] = [int(boundary_neighbors[0]), int(boundary_neighbors[1])]
    except Exception:
        backbone_info = {}

    base = {
        "requested_input": graph_input,
        "attachment_count": int(attachment_count),
        "backbone_info": backbone_info,
    }

    if graph_input == "repeat_unit":
        return {
            **base,
            "structure_smiles": smiles,
            "structure_mol": mol,
            "structure_input": "repeat_unit",
            "graph_build_ok": True,
            "graph_failed_reason": "",
        }

    try:
        linked_mol = build_star_linking_mol(smiles)
        linked_info = dict(backbone_info)
        if "neighbors" in linked_info:
            linked_info["star_link_edge"] = linked_info["neighbors"]
        return {
            **base,
            "backbone_info": linked_info,
            "structure_smiles": Chem.MolToSmiles(linked_mol, canonical=True),
            "structure_mol": linked_mol,
            "structure_input": "star_linking",
            "graph_build_ok": True,
            "graph_failed_reason": "",
        }
    except Exception as exc:
        return {
            **base,
            "structure_smiles": smiles,
            "structure_mol": mol,
            "structure_input": "repeat_unit",
            "graph_build_ok": False,
            "graph_failed_reason": str(exc)[:200],
        }


def annotate_structure_fields(data, structure, prefix="graph"):
    """Attach auditable structure metadata to a PyG Data object."""
    setattr(data, f"{prefix}_smiles", structure["structure_smiles"])
    setattr(data, f"{prefix}_input", structure["structure_input"])
    data.requested_graph_input = structure["requested_input"]
    data.structure_smiles = structure["structure_smiles"]
    data.structure_input = structure["structure_input"]
    data.graph_build_ok = bool(structure["graph_build_ok"])
    data.graph_failed_reason = structure["graph_failed_reason"]
    data.attachment_count = int(structure["attachment_count"])
    data.has_backbone_features = True
    return data


def build_graph_for_input(smiles, graph_input="repeat_unit"):
    """Build the graph used by the default GIN backend."""
    structure = build_structure_for_input(smiles, graph_input)
    data = mol_to_graph_data_obj_simple(
        structure["structure_mol"], backbone_info=structure.get("backbone_info")
    )
    return annotate_structure_fields(data, structure, prefix="graph")
