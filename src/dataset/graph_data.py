import torch
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import rdPartialCharges
import random
import math
from dataclasses import dataclass
from collections import deque
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


SCAGE_ATOM_VOCABS = {
    "atomic_num": list(range(1, 119)) + ["misc"],
    "chiral_tag": list(Chem.rdchem.ChiralType.values.values()),
    "degree": list(range(0, 11)) + ["misc"],
    "explicit_valence": list(range(0, 13)) + ["misc"],
    "formal_charge": list(range(-5, 11)) + ["misc"],
    "hybridization": list(Chem.rdchem.HybridizationType.values.values()),
    "is_aromatic": [0, 1],
    "total_numHs": list(range(0, 9)) + ["misc"],
    "atom_is_in_ring": [0, 1],
}
SCAGE_CATEGORICAL_FEATURES = tuple(SCAGE_ATOM_VOCABS)
SCAGE_CONTINUOUS_FEATURES = ("mass", "van_der_waals_radius", "partial_charge")
SCAGE_INPUT_SCHEMA_VERSION = 2
SCAGE_TOPOLOGY_SCHEMA_VERSION = 2
SCAGE_PATH_MAX_LENGTH = 5
SCAGE_SPD_MAX_DISTANCE = 20
MIPS_ATOM_FEATURE_DIM = 137
MIPS_ATOM_CLASSES = 101
MIPS_MAX_PATH_NODES = 3
PERIODIC_LGA_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MIPSPeriodicConfig:
    max_hops: int = 5
    max_repeat_rounds: int = 5
    max_model_atoms: int = 384

    @property
    def distance_threshold(self):
        return int(self.max_hops) + 1

    @property
    def required_boundary_distance(self):
        return 2 * self.distance_threshold - 1


def _scage_safe_index(name, value):
    vocabulary = SCAGE_ATOM_VOCABS[name]
    try:
        return vocabulary.index(value)
    except ValueError:
        return len(vocabulary) - 1


def _attach_scage_atom_features(data, mol):
    """Attach the atom fields consumed by the original SCAGE AtomEmbedding."""
    charge_mol = Chem.Mol(mol)
    charges = [0.0] * charge_mol.GetNumAtoms()
    try:
        rdPartialCharges.ComputeGasteigerCharges(charge_mol)
        for atom in charge_mol.GetAtoms():
            charge = atom.GetDoubleProp("_GasteigerCharge")
            if math.isnan(charge):
                charge = 0.0
            elif math.isinf(charge):
                charge = 10.0 if charge > 0 else -10.0
            charges[atom.GetIdx()] = float(charge)
    except Exception:
        charges = [0.0] * charge_mol.GetNumAtoms()

    periodic_table = Chem.GetPeriodicTable()
    categorical = {name: [] for name in SCAGE_CATEGORICAL_FEATURES}
    mass = []
    van_der_waals_radius = []
    for atom in mol.GetAtoms():
        categorical["atomic_num"].append(_scage_safe_index("atomic_num", atom.GetAtomicNum()))
        categorical["chiral_tag"].append(_scage_safe_index("chiral_tag", atom.GetChiralTag()))
        categorical["degree"].append(_scage_safe_index("degree", atom.GetTotalDegree()))
        explicit_valence = (
            atom.GetValence(Chem.ValenceType.EXPLICIT)
            if hasattr(atom, "GetValence")
            else atom.GetExplicitValence()
        )
        categorical["explicit_valence"].append(
            _scage_safe_index("explicit_valence", explicit_valence)
        )
        categorical["formal_charge"].append(_scage_safe_index("formal_charge", atom.GetFormalCharge()))
        categorical["hybridization"].append(
            _scage_safe_index("hybridization", atom.GetHybridization())
        )
        categorical["is_aromatic"].append(_scage_safe_index("is_aromatic", int(atom.GetIsAromatic())))
        categorical["total_numHs"].append(_scage_safe_index("total_numHs", atom.GetTotalNumHs()))
        categorical["atom_is_in_ring"].append(
            _scage_safe_index("atom_is_in_ring", int(atom.IsInRing()))
        )
        mass.append(float(atom.GetMass()))
        atomic_num = int(atom.GetAtomicNum())
        radius = float(periodic_table.GetRvdw(atomic_num)) if atomic_num > 0 else 0.0
        van_der_waals_radius.append(radius if math.isfinite(radius) else 0.0)

    for name, values in categorical.items():
        setattr(data, name, torch.tensor(values, dtype=torch.long))
    data.mass = torch.tensor(mass, dtype=torch.float)
    data.van_der_waals_radius = torch.tensor(van_der_waals_radius, dtype=torch.float)
    data.partial_charge = torch.tensor(charges, dtype=torch.float)
    data.scage_input_schema_version = SCAGE_INPUT_SCHEMA_VERSION
    return data


def _one_hot_with_unknown(value, choices):
    values = list(choices)
    return [float(value == item) for item in values] + [float(value not in values)]


def _attach_mips_atom_features(data, mol, backbone):
    """Attach MIPS atom features and the paper's separate backbone indicator."""
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    rows = []
    for atom in mol.GetAtoms():
        cip = atom.GetProp('_CIPCode') if atom.HasProp('_CIPCode') else None
        row = (
            _one_hot_with_unknown(atom.GetAtomicNum(), range(1, 101))
            + _one_hot_with_unknown(atom.GetDegree(), range(0, 11))
            + [float(atom.GetFormalCharge())]
            + _one_hot_with_unknown(atom.GetNumRadicalElectrons(), range(0, 5))
            + _one_hot_with_unknown(atom.GetHybridization(), hybridizations)
            + [float(atom.GetIsAromatic())]
            + _one_hot_with_unknown(atom.GetTotalNumHs(), range(0, 5))
            + [float(cip is not None), float(cip == 'R'), float(cip == 'S')]
            + [float(atom.GetMass()) * 0.01]
        )
        if len(row) != MIPS_ATOM_FEATURE_DIM:
            raise RuntimeError(f"MIPS atom feature width is {len(row)}, expected 137")
        rows.append(row)
    data.mips_x = torch.tensor(rows, dtype=torch.float)
    data.mips_backbone_mask = torch.zeros(mol.GetNumAtoms(), dtype=torch.long)
    if backbone:
        data.mips_backbone_mask[torch.tensor(sorted(backbone), dtype=torch.long)] = 1
    data.mips_input_schema_version = 2
    return data


def _apply_virtual_edge_atom_features(data, virtual_edges):
    """Replace terminal implicit-H semantics with periodic covalent edges."""
    incidence = {}
    bond_order_sum = {}
    for begin, end, bond_type in virtual_edges:
        order = int(round(float(bond_type))) if bond_type != Chem.rdchem.BondType.AROMATIC else 1
        for atom_idx in (int(begin), int(end)):
            incidence[atom_idx] = incidence.get(atom_idx, 0) + 1
            bond_order_sum[atom_idx] = bond_order_sum.get(atom_idx, 0) + order
    explicit_vocab = SCAGE_ATOM_VOCABS["explicit_valence"]
    hydrogen_vocab = SCAGE_ATOM_VOCABS["total_numHs"]
    for atom_idx, edge_count in incidence.items():
        explicit_value = explicit_vocab[int(data.explicit_valence[atom_idx])]
        hydrogen_value = hydrogen_vocab[int(data.total_numHs[atom_idx])]
        if explicit_value != "misc":
            data.explicit_valence[atom_idx] = _scage_safe_index(
                "explicit_valence", int(explicit_value) + bond_order_sum[atom_idx]
            )
        if hydrogen_value != "misc":
            data.total_numHs[atom_idx] = _scage_safe_index(
                "total_numHs", max(0, int(hydrogen_value) - edge_count)
            )
    return data


def one_of_k_encoding(x, allowable_set):
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def _graph_backbone_annotations(
    mol, original_neighbors=None, star_link_edge=None, ordered_backbone_path=None
):
    n_atoms = mol.GetNumAtoms()
    backbone = set()
    attachment_neighbors = set(int(i) for i in (original_neighbors or []))
    side_chain = set(range(n_atoms))

    if original_neighbors is not None and len(original_neighbors) == 2:
        left = int(original_neighbors[0])
        right = int(original_neighbors[1])
        if left == right:
            if left < n_atoms:
                backbone.add(left)
        else:
            try:
                path = tuple(int(idx) for idx in (ordered_backbone_path or []))
                if not path:
                    path_mol = Chem.RWMol(mol)
                    if star_link_edge is not None:
                        star_left, star_right = (int(value) for value in star_link_edge)
                        if path_mol.GetBondBetweenAtoms(star_left, star_right) is not None:
                            path_mol.RemoveBond(star_left, star_right)
                    path = Chem.rdmolops.GetShortestPath(path_mol.GetMol(), left, right)
                backbone.update(int(i) for i in path)
                path_bonds = {
                    int(mol.GetBondBetweenAtoms(int(i), int(j)).GetIdx())
                    for i, j in zip(path[:-1], path[1:])
                    if mol.GetBondBetweenAtoms(int(i), int(j)) is not None
                }
                # MIPS treats a ring as backbone only when the attachment path
                # actually traverses one of its bonds. Merely touching a ring
                # atom is insufficient.
                ring_info = mol.GetRingInfo()
                for atom_ring, bond_ring in zip(ring_info.AtomRings(), ring_info.BondRings()):
                    if path_bonds.intersection(int(idx) for idx in bond_ring):
                        backbone.update(int(idx) for idx in atom_ring)
            except Exception:
                backbone.update(int(i) for i in (left, right) if int(i) < n_atoms)

    side_chain = side_chain - backbone
    star_link_edge_set = set()
    if star_link_edge is not None:
        i, j = int(star_link_edge[0]), int(star_link_edge[1])
        star_link_edge_set.add((i, j))
        star_link_edge_set.add((j, i))

    return backbone, attachment_neighbors, side_chain, star_link_edge_set


def _bond_path_codes(bond, is_star_linking):
    """Return compact non-zero categorical codes for one path bond."""
    bond_type = allowable_features['possible_bonds'].index(bond.GetBondType()) + 1
    try:
        stereo = allowable_features['possible_bond_stereo_list'].index(bond.GetStereo()) + 1
    except ValueError:
        stereo = 1
    return (
        bond_type,
        stereo,
        int(bond.GetIsConjugated()) + 1,
        int(bond.IsInRing()) + 1,
        int(bool(is_star_linking)) + 1,
    )


def _attach_scage_topology_features(data, mol, star_link_edge_set, virtual_edges=()):
    """Cache Graphormer/MIPS shortest-path and compact path-bond fields."""
    num_atoms = mol.GetNumAtoms()
    unreachable = SCAGE_SPD_MAX_DISTANCE + 1
    spd = torch.full((num_atoms, num_atoms), unreachable, dtype=torch.uint8)
    path_fields = torch.zeros(
        (num_atoms, num_atoms, SCAGE_PATH_MAX_LENGTH, 5), dtype=torch.uint8
    )
    mips_path_nodes = torch.full(
        (num_atoms, num_atoms, MIPS_MAX_PATH_NODES), -1, dtype=torch.long
    )
    adjacency = [[] for _ in range(num_atoms)]
    for bond in mol.GetBonds():
        begin, end = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
        is_star = (begin, end) in star_link_edge_set
        codes = _bond_path_codes(bond, is_star)
        adjacency[begin].append((end, codes))
        adjacency[end].append((begin, codes))
    for begin, end, bond_type in virtual_edges:
        bond_code = allowable_features['possible_bonds'].index(bond_type) + 1
        codes = (bond_code, 1, 1, 1, 2)
        adjacency[int(begin)].append((int(end), codes))
        adjacency[int(end)].append((int(begin), codes))
    for neighbors in adjacency:
        neighbors.sort(key=lambda item: item[0])

    for source in range(num_atoms):
        spd[source, source] = 0
        queue = [source]
        predecessor = {source: (-1, None)}
        cursor = 0
        while cursor < len(queue):
            node = queue[cursor]
            cursor += 1
            for neighbor, codes in adjacency[node]:
                if neighbor in predecessor:
                    continue
                predecessor[neighbor] = (node, codes)
                queue.append(neighbor)
        for target in range(num_atoms):
            if target not in predecessor:
                continue
            node_path = [target]
            node = target
            while node != source:
                node = predecessor[node][0]
                node_path.append(node)
            node_path.reverse()
            if len(node_path) <= MIPS_MAX_PATH_NODES:
                mips_path_nodes[source, target, :len(node_path)] = torch.tensor(
                    node_path, dtype=torch.long
                )
            if target == source:
                continue
            reversed_codes = []
            node = target
            while node != source:
                previous, codes = predecessor[node]
                reversed_codes.append(codes)
                node = previous
            codes_on_path = list(reversed(reversed_codes))
            spd[source, target] = min(len(codes_on_path), unreachable)
            for hop, codes in enumerate(codes_on_path[:SCAGE_PATH_MAX_LENGTH]):
                path_fields[source, target, hop] = torch.tensor(codes, dtype=torch.uint8)

    data.scage_spd = spd
    data.scage_path_bond_fields = path_fields
    data.mips_path_nodes = mips_path_nodes
    data.scage_topology_schema_version = SCAGE_TOPOLOGY_SCHEMA_VERSION
    return data


def mol_to_graph_data_obj_simple(mol, backbone_info=None):
    backbone_info = backbone_info or {}
    backbone, attachment_neighbors, side_chain, star_link_edge_set = _graph_backbone_annotations(
        mol,
        original_neighbors=backbone_info.get("neighbors"),
        star_link_edge=backbone_info.get("star_link_edge"),
        ordered_backbone_path=backbone_info.get("ordered_backbone_path"),
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

    virtual_edges = list(backbone_info.get("virtual_edges", []))
    # bonds
    edges_list = []
    edge_features_list = []
    if len(mol.GetBonds()) > 0: # mol has bonds
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

    for i, j, bond_type in virtual_edges:
        edge_feature = (
            one_of_k_encoding_unk(bond_type, allowable_features['possible_bonds']) +
            one_of_k_encoding_unk(
                Chem.rdchem.BondStereo.STEREONONE,
                allowable_features['possible_bond_stereo_list'],
            ) + [0, 0, 1]
        )
        edges_list.extend([(int(i), int(j)), (int(j), int(i))])
        edge_features_list.extend([edge_feature, edge_feature])
    if edges_list:
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, NUM_BOND_FEATURES), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    _attach_mips_atom_features(data, mol, backbone)
    _attach_scage_atom_features(data, mol)
    _apply_virtual_edge_atom_features(data, virtual_edges)
    _attach_scage_topology_features(data, mol, star_link_edge_set, virtual_edges)
    backbone_role = torch.zeros(mol.GetNumAtoms(), dtype=torch.long)
    if backbone:
        backbone_role[torch.tensor(sorted(backbone), dtype=torch.long)] = 1
    if attachment_neighbors:
        backbone_role[torch.tensor(sorted(attachment_neighbors), dtype=torch.long)] = 2
    data.scage_backbone_role = backbone_role
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


def build_periodic_multimer_mol(smiles_or_mol, num_repeat_units, close_periodic=True):
    """Build an ordered m-RU chain or its translational periodic quotient graph.

    The atom order is deterministic: all non-dummy atoms of RU 0, followed by
    RU 1, and so on. Geometry generation and SCAGE graph construction share
    this function so ``graph_to_geom_index`` is an identity mapping.
    """
    num_repeat_units = int(num_repeat_units)
    if num_repeat_units < 1:
        raise ValueError("num_repeat_units must be >= 1")
    source = (
        Chem.Mol(smiles_or_mol)
        if isinstance(smiles_or_mol, Chem.Mol)
        else Chem.MolFromSmiles(str(smiles_or_mol))
    )
    if source is None:
        raise ValueError("invalid smiles")
    dummy_atoms, neighbors, bond_types = _get_dummy_atoms_and_neighbors(source)
    if len(dummy_atoms) != 2:
        raise ValueError("periodic multimer requires exactly two dummy atoms")
    if len(set(neighbors)) != 2:
        raise ValueError("periodic multimer requires two distinct boundary atoms")
    if bond_types[0] != bond_types[1]:
        raise ValueError("attachment bond types are not consistent")
    connection_bond_type = (
        Chem.rdchem.BondType.SINGLE
        if bond_types[0] == Chem.rdchem.BondType.AROMATIC
        else bond_types[0]
    )

    dummy_set = set(int(idx) for idx in dummy_atoms)
    copy_source = Chem.Mol(source)
    try:
        # Removing attachment dummies can leave an aromatic valence pattern
        # that RDKit cannot kekulize. Copying a concrete Kekule form preserves
        # the same chemistry while allowing the open oligomer to be sanitized.
        Chem.Kekulize(copy_source, clearAromaticFlags=True)
    except Exception:
        copy_source = Chem.Mol(source)
    base = Chem.RWMol()
    old_to_base = {}
    for atom in copy_source.GetAtoms():
        if atom.GetIdx() in dummy_set:
            continue
        old_to_base[atom.GetIdx()] = base.AddAtom(Chem.Atom(atom))
    for bond in copy_source.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin in dummy_set or end in dummy_set:
            continue
        base.AddBond(old_to_base[begin], old_to_base[end], bond.GetBondType())
    base_mol = base.GetMol()
    # Do not sanitize the isolated base RU yet: doing so assigns implicit H to
    # attachment boundaries before inter-unit bonds are added. The completed
    # finite chain is sanitized once below.
    base_mol.UpdatePropertyCache(strict=False)
    left_base = int(old_to_base[int(neighbors[0])])
    right_base = int(old_to_base[int(neighbors[1])])
    if left_base == right_base:
        raise ValueError("periodic multimer requires two distinct mapped boundaries")
    backbone_base = list(Chem.GetShortestPath(base_mol, left_base, right_base))
    base_atoms = int(base_mol.GetNumAtoms())

    combined = Chem.RWMol()
    atom_maps = []
    for _ in range(num_repeat_units):
        atom_map = {}
        for atom in base_mol.GetAtoms():
            atom_map[atom.GetIdx()] = combined.AddAtom(Chem.Atom(atom))
        for bond in base_mol.GetBonds():
            combined.AddBond(
                atom_map[bond.GetBeginAtomIdx()],
                atom_map[bond.GetEndAtomIdx()],
                bond.GetBondType(),
            )
        atom_maps.append(atom_map)

    inter_unit_edges = []
    for unit_idx in range(num_repeat_units - 1):
        right = int(atom_maps[unit_idx][right_base])
        left = int(atom_maps[unit_idx + 1][left_base])
        combined.AddBond(right, left, connection_bond_type)
        inter_unit_edges.append((right, left))
    periodic_edge = None
    if close_periodic:
        right = int(atom_maps[-1][right_base])
        left = int(atom_maps[0][left_base])
        # The closing edge crosses a periodic image. Adding it to the RDKit
        # molecule would create an artificial finite ring and can invalidate
        # aromaticity/kekulization. Keep it as an explicit virtual graph edge.
        periodic_edge = (right, left)

    result = combined.GetMol()
    Chem.SanitizeMol(result)
    unit_atoms = [
        [int(atom_maps[unit_idx][atom_idx]) for atom_idx in range(base_atoms)]
        for unit_idx in range(num_repeat_units)
    ]
    ordered_backbone = []
    for unit_idx in range(num_repeat_units):
        ordered_backbone.extend(
            int(atom_maps[unit_idx][atom_idx]) for atom_idx in backbone_base
        )
    atom_ru_index = [0] * result.GetNumAtoms()
    for unit_idx, indices in enumerate(unit_atoms):
        for atom_idx in indices:
            atom_ru_index[atom_idx] = unit_idx
    return result, {
        "num_repeat_units": num_repeat_units,
        "base_atom_count": base_atoms,
        "unit_atoms": unit_atoms,
        "atom_ru_index": atom_ru_index,
        "left_boundary": int(atom_maps[0][left_base]),
        "right_boundary": int(atom_maps[-1][right_base]),
        "unit_left_boundaries": [int(mapping[left_base]) for mapping in atom_maps],
        "unit_right_boundaries": [int(mapping[right_base]) for mapping in atom_maps],
        "backbone_base": [int(idx) for idx in backbone_base],
        "ordered_backbone_path": ordered_backbone,
        "inter_unit_edges": inter_unit_edges,
        "periodic_edge": periodic_edge,
        "attachment_bond_type": connection_bond_type,
    }


def build_polygen_periodic_structure(smiles, num_repeat_units):
    """Return the SCAGE structure for one exact translational periodic cell."""
    mol, metadata = build_periodic_multimer_mol(
        smiles, num_repeat_units=num_repeat_units, close_periodic=True
    )
    pair = [metadata["left_boundary"], metadata["right_boundary"]]
    periodic_edge = metadata["periodic_edge"]
    backbone_info = {
        "neighbors": pair,
        "star_link_edge": list(periodic_edge) if periodic_edge is not None else [],
        "ordered_backbone_path": metadata["ordered_backbone_path"],
        "virtual_edges": [
            (int(periodic_edge[0]), int(periodic_edge[1]), metadata["attachment_bond_type"])
        ] if periodic_edge is not None else [],
    }
    return {
        "requested_input": "star_linking",
        "attachment_count": 2,
        "backbone_info": backbone_info,
        "structure_smiles": Chem.MolToSmiles(mol, canonical=False),
        "structure_mol": mol,
        "structure_input": "polygen_periodic",
        "graph_build_ok": True,
        "graph_failed_reason": "",
        "periodic_metadata": metadata,
    }


def _periodic_scage_atom_template(smiles):
    """Create one cut-invariant RU template for the periodic node inputs."""
    motif, metadata = build_periodic_multimer_mol(
        smiles, num_repeat_units=1, close_periodic=True
    )
    template = Data()
    _attach_scage_atom_features(template, motif)
    periodic_edge = metadata.get("periodic_edge")
    if periodic_edge is not None:
        _apply_virtual_edge_atom_features(template, [(
            int(periodic_edge[0]),
            int(periodic_edge[1]),
            metadata["attachment_bond_type"],
        )])
    return {
        name: getattr(template, name).clone()
        for name in SCAGE_CATEGORICAL_FEATURES + SCAGE_CONTINUOUS_FEATURES
    }


def build_mips_periodic_structure(
    smiles,
    geometry_num_ru=1,
    geometry_valid=False,
    config=None,
):
    """Build the alias-free MIPS supercell used by ``scage_parallel``.

    A strict PBC primitive starts from its optimized ``geometry_num_ru`` cell.
    Topology-only samples start from one RU.  The open motif is doubled until
    its two outer boundaries satisfy the MIPS locality condition, then the
    final boundary edge is represented as a virtual periodic edge.
    """
    config = config or MIPSPeriodicConfig()
    geometry_num_ru = max(1, int(geometry_num_ru))
    base_num_ru = geometry_num_ru if bool(geometry_valid) else 1
    last_reason = ""
    for repeat_round in range(int(config.max_repeat_rounds)):
        repeat_factor = 1 << repeat_round
        model_num_ru = base_num_ru * repeat_factor
        try:
            open_mol, open_meta = build_periodic_multimer_mol(
                smiles, model_num_ru, close_periodic=False
            )
            atom_count = int(open_mol.GetNumAtoms())
            boundary_distance = len(open_meta["ordered_backbone_path"]) - 1
            if atom_count > int(config.max_model_atoms):
                last_reason = f"model_atoms={atom_count}>{config.max_model_atoms}"
                break
            if boundary_distance <= int(config.required_boundary_distance):
                last_reason = (
                    f"boundary_distance={boundary_distance}<="
                    f"{config.required_boundary_distance}"
                )
                continue
            periodic_mol, metadata = build_periodic_multimer_mol(
                smiles, model_num_ru, close_periodic=True
            )
            periodic_edge = metadata["periodic_edge"]
            backbone_info = {
                "neighbors": [metadata["left_boundary"], metadata["right_boundary"]],
                "star_link_edge": list(periodic_edge),
                "ordered_backbone_path": metadata["ordered_backbone_path"],
                "virtual_edges": [(
                    int(periodic_edge[0]),
                    int(periodic_edge[1]),
                    metadata["attachment_bond_type"],
                )],
            }
            return {
                "requested_input": "star_linking",
                "attachment_count": 2,
                "backbone_info": backbone_info,
                "structure_smiles": Chem.MolToSmiles(periodic_mol, canonical=False),
                "structure_mol": periodic_mol,
                "structure_input": "mips_periodic_supercell",
                "graph_build_ok": True,
                "graph_failed_reason": "",
                "graph_available": True,
                "periodic_metadata": metadata,
                "geometry_period_ru": geometry_num_ru if geometry_valid else 0,
                "mips_repeat_factor": repeat_factor,
                "model_cell_ru": model_num_ru,
                "mips_boundary_distance": boundary_distance,
                "mips_distance_threshold": config.distance_threshold,
                "mips_condition_valid": True,
                "periodic_atom_template": _periodic_scage_atom_template(smiles),
            }
        except Exception as exc:
            last_reason = str(exc)[:200]

    # Keep the other modalities usable.  The placeholder graph is never
    # exposed to fusion and receives self-only LGA edges below.
    placeholder_mol, metadata = build_periodic_multimer_mol(
        smiles, 1, close_periodic=False
    )
    pair = [metadata["left_boundary"], metadata["right_boundary"]]
    backbone_info = {
        "neighbors": pair,
        "star_link_edge": [],
        "ordered_backbone_path": metadata["ordered_backbone_path"],
        "virtual_edges": [],
    }
    return {
        "requested_input": "star_linking",
        "attachment_count": 2,
        "backbone_info": backbone_info,
        "structure_smiles": Chem.MolToSmiles(placeholder_mol, canonical=False),
        "structure_mol": placeholder_mol,
        "structure_input": "mips_periodic_unavailable",
        "graph_build_ok": False,
        "graph_failed_reason": f"mips_periodic_condition_failed:{last_reason}"[:240],
        "graph_available": False,
        "periodic_metadata": metadata,
        "geometry_period_ru": geometry_num_ru if geometry_valid else 0,
        "mips_repeat_factor": 0,
        "model_cell_ru": 1,
        "mips_boundary_distance": len(metadata["ordered_backbone_path"]) - 1,
        "mips_distance_threshold": config.distance_threshold,
        "mips_condition_valid": False,
        "periodic_atom_template": _periodic_scage_atom_template(smiles),
    }


def _mips_periodic_adjacency(mol, periodic_edge):
    adjacency = [[] for _ in range(mol.GetNumAtoms())]
    for bond in mol.GetBonds():
        begin, end = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
        adjacency[begin].append((end, 0))
        adjacency[end].append((begin, 0))
    if periodic_edge is not None:
        tail, head = (int(value) for value in periodic_edge)
        adjacency[tail].append((head, 1))
        adjacency[head].append((tail, -1))
    for neighbors in adjacency:
        neighbors.sort(key=lambda value: (value[0], value[1]))
    return adjacency


def _periodic_local_states(query, adjacency, max_hops):
    start = (int(query), 0)
    distance = {start: 0}
    predecessor = {start: None}
    queue = deque([start])
    while queue:
        atom, image = queue.popleft()
        current_distance = distance[(atom, image)]
        if current_distance >= int(max_hops):
            continue
        for neighbor, edge_shift in adjacency[atom]:
            state = (int(neighbor), int(image + edge_shift))
            if state in distance:
                continue
            distance[state] = current_distance + 1
            predecessor[state] = (atom, image)
            queue.append(state)
    return distance, predecessor


def attach_periodic_lga_topology(data, structure, config=None):
    """Attach sparse source->query LGA records and validate alias freedom."""
    config = config or MIPSPeriodicConfig()
    template = structure.get("periodic_atom_template") or {}
    unit_atoms = (structure.get("periodic_metadata") or {}).get("unit_atoms", [])
    if template and unit_atoms:
        for name in SCAGE_CATEGORICAL_FEATURES + SCAGE_CONTINUOUS_FEATURES:
            values = getattr(data, name).clone()
            motif_values = template[name].to(dtype=values.dtype)
            for atom_indices in unit_atoms:
                if len(atom_indices) != motif_values.size(0):
                    raise RuntimeError("periodic atom-template mapping is inconsistent")
                values[torch.tensor(atom_indices, dtype=torch.long)] = motif_values
            setattr(data, name, values)
    # The legacy SCAGE role 2 marked the arbitrary periodic cut endpoints.
    # This route keeps only the cut-invariant backbone/non-backbone role.
    data.scage_backbone_role = (data.scage_backbone_role > 0).long()
    graph_available = bool(structure.get("graph_available", False))
    num_atoms = int(data.x.size(0))
    metadata = structure.get("periodic_metadata") or {}
    periodic_edge = metadata.get("periodic_edge") if graph_available else None
    adjacency = _mips_periodic_adjacency(structure["structure_mol"], periodic_edge)
    sources, targets, distances, shifts, paths = [], [], [], [], []
    alias_free = True

    for query in range(num_atoms):
        if graph_available:
            state_distances, predecessor = _periodic_local_states(
                query, adjacency, config.max_hops
            )
            atom_images = {}
            for atom, image in state_distances:
                previous = atom_images.get(atom)
                if previous is not None and previous != image:
                    alias_free = False
                    break
                atom_images[atom] = image
            if not alias_free:
                break
            ordered_states = sorted(
                state_distances,
                key=lambda state: (state_distances[state], state[0], state[1]),
            )
        else:
            state_distances = {(query, 0): 0}
            predecessor = {(query, 0): None}
            ordered_states = [(query, 0)]

        for state in ordered_states:
            atom, image = state
            path_states = []
            current = state
            while current is not None:
                path_states.append(current)
                current = predecessor[current]
            path_states.reverse()
            path_atoms = [int(item[0]) for item in path_states]
            if len(path_atoms) > int(config.max_hops) + 1:
                raise RuntimeError("periodic LGA path exceeds configured maximum")
            sources.append(int(atom))
            targets.append(int(query))
            distances.append(int(state_distances[state]))
            shifts.append(int(image))
            paths.append(path_atoms)

    if graph_available and not alias_free:
        # A condition-valid cell must be injective in every local query. Treat
        # failure as unavailable instead of introducing image multi-edges.
        graph_available = False
        sources = list(range(num_atoms))
        targets = list(range(num_atoms))
        distances = [0] * num_atoms
        shifts = [0] * num_atoms
        paths = [[idx] for idx in range(num_atoms)]

    edge_count = len(sources)
    max_path_nodes = int(config.max_hops) + 1
    path_index = torch.full((edge_count, max_path_nodes), -1, dtype=torch.long)
    path_mask = torch.zeros((edge_count, max_path_nodes), dtype=torch.bool)
    for edge_idx, path in enumerate(paths):
        path_index[edge_idx, :len(path)] = torch.tensor(path, dtype=torch.long)
        path_mask[edge_idx, :len(path)] = True

    data.lga_edge_index = torch.tensor([sources, targets], dtype=torch.long)
    data.lga_spd = torch.tensor(distances, dtype=torch.long)
    data.lga_path_index = path_index
    data.lga_path_mask = path_mask
    data.lga_image_shift = torch.tensor(shifts, dtype=torch.long)
    data.lga_geometry_valid = torch.zeros(edge_count, dtype=torch.bool)
    data.lga_pbc_distance_confs = torch.zeros((1, edge_count), dtype=torch.float)
    data.graph_available = bool(graph_available)
    data.periodic_topology_valid = bool(graph_available)
    data.mips_alias_free = bool(alias_free and graph_available)
    data.periodic_lga_schema_version = PERIODIC_LGA_SCHEMA_VERSION
    return data


def _sanitize_to_smiles(mol):
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol, canonical=True)




def generate_multimer_smiles(num_repeat_units, smiles, replace_dummy_atoms=False):
    """Generate an N-MRU P-SMILES while preserving terminal dummy atoms by default.

    This follows the PerioGT periodic augmentation utility: adjacent repeat
    units are connected through their attachment-neighbor atoms, internal dummy
    atoms are removed, and the two terminal dummy atoms are retained unless
    explicitly replaced.
    """
    num_repeat_units = int(num_repeat_units)
    if num_repeat_units < 1:
        raise ValueError("num_repeat_units must be >= 1")
    if num_repeat_units == 1:
        return smiles

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("invalid smiles")
    dummy_atoms, neighbors, bond_types = _get_dummy_atoms_and_neighbors(mol)
    if len(dummy_atoms) != 2:
        raise ValueError("generate_multimer_smiles requires exactly two dummy atoms")
    if len(set(neighbors)) != 2:
        raise ValueError("generate_multimer_smiles requires two distinct attachment neighbors")
    if bond_types[0] != bond_types[1]:
        raise ValueError("attachment bond types are not consistent")

    combined = Chem.RWMol()
    atom_maps = []
    for _ in range(num_repeat_units):
        atom_map = {}
        for atom in mol.GetAtoms():
            new_atom = Chem.Atom(atom)
            atom_map[atom.GetIdx()] = combined.AddAtom(new_atom)
        for bond in mol.GetBonds():
            combined.AddBond(
                atom_map[bond.GetBeginAtomIdx()],
                atom_map[bond.GetEndAtomIdx()],
                bond.GetBondType(),
            )
        atom_maps.append(atom_map)

    for unit_idx in range(num_repeat_units - 1):
        right_neighbor = atom_maps[unit_idx][neighbors[1]]
        left_neighbor = atom_maps[unit_idx + 1][neighbors[0]]
        if combined.GetBondBetweenAtoms(int(right_neighbor), int(left_neighbor)) is None:
            combined.AddBond(int(right_neighbor), int(left_neighbor), bond_types[0])

    atoms_to_remove = []
    for unit_idx in range(num_repeat_units):
        left_dummy = atom_maps[unit_idx][dummy_atoms[0]]
        right_dummy = atom_maps[unit_idx][dummy_atoms[1]]
        if unit_idx > 0:
            atoms_to_remove.append(left_dummy)
        elif replace_dummy_atoms:
            combined.GetAtomWithIdx(int(left_dummy)).SetAtomicNum(1)
        if unit_idx < num_repeat_units - 1:
            atoms_to_remove.append(right_dummy)
        elif replace_dummy_atoms:
            combined.GetAtomWithIdx(int(right_dummy)).SetAtomicNum(1)

    for atom_idx in sorted(set(int(i) for i in atoms_to_remove), reverse=True):
        combined.RemoveAtom(atom_idx)

    multimer = combined.GetMol()
    Chem.SanitizeMol(multimer)
    multimer = Chem.RemoveHs(multimer)
    return Chem.MolToSmiles(multimer, canonical=True)


def periodicity_augment_smiles(
    smiles, max_mrus=3, return_n=False, rng=None, return_metadata=False
):
    """PerioGT-style periodicity augmentation for P-SMILES.

    A non-ring bond on the path between the two attachment dummy atoms is cut,
    the original attachment boundary is closed, and new dummy atoms are inserted
    at the cut. If no valid cut exists, the original SMILES is returned with
    n=1, matching the PerioGT fallback behavior.
    """
    rng = rng or random
    max_mrus = max(1, int(max_mrus))
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("invalid smiles")

    dummy_atoms, neighbors, bond_types = _get_dummy_atoms_and_neighbors(mol)
    if len(dummy_atoms) != 2:
        raise ValueError("periodicity_augment_smiles requires exactly two dummy atoms")
    if len(set(neighbors)) != 2:
        raise ValueError("periodicity_augment_smiles requires two distinct attachment neighbors")
    if bond_types[0] != bond_types[1]:
        raise ValueError("attachment bond types are not consistent")

    path = list(Chem.GetShortestPath(mol, int(dummy_atoms[0]), int(dummy_atoms[1])))
    candidate_bonds = []
    for i in range(1, len(path) - 2):
        begin_idx = int(path[i])
        end_idx = int(path[i + 1])
        bond = mol.GetBondBetweenAtoms(begin_idx, end_idx)
        if bond is None or bond.IsInRing():
            continue
        candidate_bonds.append((begin_idx, end_idx, bond.GetBondType()))

    if not candidate_bonds:
        metadata = {"cut_identity": None, "num_candidates": 0, "changed": False}
        if return_metadata:
            return (smiles, 1, metadata) if return_n else (smiles, metadata)
        if return_n:
            return smiles, 1
        return smiles

    cut_begin, cut_end, cut_bond_type = rng.choice(candidate_bonds)
    editable = Chem.RWMol(mol)
    editable.RemoveBond(cut_begin, cut_end)

    new_dummy_1 = editable.AddAtom(Chem.Atom(0))
    editable.AddBond(cut_begin, new_dummy_1, cut_bond_type)
    new_dummy_2 = editable.AddAtom(Chem.Atom(0))
    editable.AddBond(cut_end, new_dummy_2, cut_bond_type)

    if editable.GetBondBetweenAtoms(int(neighbors[0]), int(neighbors[1])) is None:
        editable.AddBond(int(neighbors[0]), int(neighbors[1]), bond_types[0])

    for atom_idx in sorted((int(i) for i in dummy_atoms), reverse=True):
        editable.RemoveAtom(atom_idx)

    augmented_mol = editable.GetMol()
    Chem.SanitizeMol(augmented_mol)
    augmented_smiles = Chem.MolToSmiles(augmented_mol, canonical=True)

    num_repeat_units = rng.randint(1, max_mrus)
    multimer_smiles = generate_multimer_smiles(
        num_repeat_units=num_repeat_units,
        smiles=augmented_smiles,
        replace_dummy_atoms=False,
    )
    metadata = {
        "cut_identity": tuple(sorted((int(cut_begin), int(cut_end)))),
        "num_candidates": len(candidate_bonds),
        "changed": str(multimer_smiles) != str(smiles),
    }
    if return_metadata:
        return (multimer_smiles, num_repeat_units, metadata) if return_n else (multimer_smiles, metadata)
    if return_n:
        return multimer_smiles, num_repeat_units
    return multimer_smiles


def build_star_linking_mol(smiles, return_mapping=False):
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
    remaining = [idx for idx in range(mol.GetNumAtoms()) if idx not in set(dummy_atoms)]
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(remaining)}
    mapped_neighbors = [old_to_new[int(idx)] for idx in neighbors]
    raw_backbone = list(Chem.GetShortestPath(mol, int(neighbors[0]), int(neighbors[1])))
    mapped_backbone = [old_to_new[int(idx)] for idx in raw_backbone if int(idx) in old_to_new]
    mapping = {
        "old_to_new": old_to_new,
        "attachment_pair": mapped_neighbors,
        "star_link_edge": mapped_neighbors,
        "ordered_backbone_path": mapped_backbone,
    }
    return (linked_mol, mapping) if return_mapping else linked_mol


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
        linked_mol, linked_mapping = build_star_linking_mol(smiles, return_mapping=True)
        linked_info = {
            "neighbors": list(linked_mapping["attachment_pair"]),
            "star_link_edge": list(linked_mapping["star_link_edge"]),
            "ordered_backbone_path": list(linked_mapping["ordered_backbone_path"]),
        }
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


def build_mips_paper_structure(smiles, distance_threshold=3, max_repeat_units=16):
    """Build the finite MIPS graph required by the paper's locality theorem.

    For a local attention radius ``distance_threshold``, the two outer
    attachment boundaries must be farther apart than ``2*d-1``. Short repeat
    units are therefore repeated until the condition is satisfied, then the
    two remaining attachment atoms are star-linked to represent periodic
    topology.
    """
    distance_threshold = int(distance_threshold)
    required_distance = 2 * distance_threshold - 1
    if distance_threshold < 1:
        raise ValueError("MIPS distance_threshold must be positive")

    original = Chem.MolFromSmiles(smiles)
    if original is None:
        raise ValueError("invalid smiles")
    dummy_atoms, neighbors, bond_types = _get_dummy_atoms_and_neighbors(original)
    if len(dummy_atoms) != 2 or len(set(neighbors)) != 2:
        raise ValueError("MIPS requires exactly two distinct attachment boundaries")
    if bond_types[0] != bond_types[1]:
        raise ValueError("attachment bond types are not consistent")

    last_error = None
    for repeat_units in range(1, int(max_repeat_units) + 1):
        try:
            expanded_smiles = generate_multimer_smiles(
                repeat_units, smiles, replace_dummy_atoms=False
            )
            linked_mol, mapping = build_star_linking_mol(
                expanded_smiles, return_mapping=True
            )
            boundary_distance = len(mapping["ordered_backbone_path"]) - 1
            if boundary_distance <= required_distance:
                continue
            linked_info = {
                "neighbors": list(mapping["attachment_pair"]),
                "star_link_edge": list(mapping["star_link_edge"]),
                "ordered_backbone_path": list(mapping["ordered_backbone_path"]),
            }
            return {
                "requested_input": "star_linking",
                "attachment_count": 2,
                "backbone_info": linked_info,
                "structure_smiles": Chem.MolToSmiles(linked_mol, canonical=True),
                "structure_mol": linked_mol,
                "structure_input": "mips_star_linking",
                "graph_build_ok": True,
                "graph_failed_reason": "",
                "mips_repeat_units": repeat_units,
                "mips_boundary_distance": boundary_distance,
                "mips_distance_threshold": distance_threshold,
            }
        except Exception as exc:
            last_error = exc
    detail = f": {last_error}" if last_error is not None else ""
    raise ValueError(
        f"MIPS short-RU expansion did not reach boundary distance > "
        f"{required_distance} within {max_repeat_units} repeat units{detail}"
    )


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
    backbone_info = structure.get("backbone_info") or {}
    pair = backbone_info.get("neighbors", [])
    star_edge = backbone_info.get("star_link_edge", [])
    path = backbone_info.get("ordered_backbone_path", [])
    data.attachment_pair = torch.tensor(pair if len(pair) == 2 else [-1, -1], dtype=torch.long)
    data.star_link_edge = torch.tensor(
        star_edge if len(star_edge) == 2 else [-1, -1], dtype=torch.long
    )
    data.ordered_backbone_path = torch.tensor(path, dtype=torch.long)
    data.star_link_metadata_valid = bool(
        data.graph_build_ok
        and data.structure_input in {
            "star_linking", "mips_star_linking", "polygen_periodic",
            "mips_periodic_supercell",
        }
        and len(pair) == 2
        and len(star_edge) == 2
        and len(path) >= 2
    )
    periodic_metadata = structure.get("periodic_metadata") or {}
    ru_index = periodic_metadata.get("atom_ru_index")
    if ru_index is None:
        ru_index = [0] * int(data.x.size(0))
    data.scage_ru_index = torch.tensor(ru_index, dtype=torch.long)
    data.periodic_ru_count = int(periodic_metadata.get("num_repeat_units", 1))
    data.mips_repeat_units = int(structure.get("mips_repeat_units", 1))
    data.mips_boundary_distance = int(structure.get("mips_boundary_distance", -1))
    data.mips_distance_threshold = int(structure.get("mips_distance_threshold", 3))
    data.graph_available = bool(structure.get("graph_available", data.graph_build_ok))
    data.geometry_period_ru = int(structure.get("geometry_period_ru", 0))
    data.mips_repeat_factor = int(structure.get("mips_repeat_factor", 1))
    data.model_cell_ru = int(structure.get("model_cell_ru", data.periodic_ru_count))
    data.mips_condition_valid = bool(
        structure.get("mips_condition_valid", data.graph_available)
    )
    return data


def build_graph_for_input(smiles, graph_input="repeat_unit"):
    """Build the graph used by the default GIN backend."""
    structure = build_structure_for_input(smiles, graph_input)
    data = mol_to_graph_data_obj_simple(
        structure["structure_mol"], backbone_info=structure.get("backbone_info")
    )
    return annotate_structure_fields(data, structure, prefix="graph")
