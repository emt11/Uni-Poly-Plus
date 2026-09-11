import torch
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import rdPartialCharges
import random
import math
from dataclasses import dataclass
from collections import deque
from torch_geometric.data import Data
from .mips_trimer_contract import FEATURE_SCHEMA, LEGACY_FEATURE_SCHEMA

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
MIPS_EXPERIMENT_FEATURE_SCHEMA = FEATURE_SCHEMA
# Canonical periodic MTS topology is LGA schema 2.  The explicit finite-copy
# helper below is retained only as a private/test reference and is labelled
# with the legacy v1 value so production validation cannot accept it.
MIPS_LOCAL_LGA_SCHEMA_VERSION = 2
LEGACY_MIPS_LOCAL_LGA_SCHEMA_VERSION = 1
CANONICAL_PERIODIC_LGA_SCHEMA_VERSION = MIPS_LOCAL_LGA_SCHEMA_VERSION
MIPS_MULTIMER_BUILDER_VERSION = 2


@dataclass(frozen=True)
class MIPSLocalConfig:
    """Configuration for the non-PBC localized MIPS graph.

    ``max_hops`` is the largest graph distance visible to attention.  The MIPS
    paper uses two hops, while topology-plus uses five.  No coordinate, cell or
    image information is part of this contract.
    """

    max_hops: int = 2
    max_repeat_units: int = 16
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


def _mips_atom_feature_rows(mol):
    """Return the public-MIPS 137-d atom rows for one RDKit molecule."""
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
    return torch.tensor(rows, dtype=torch.float)


def _attach_mips_atom_features(data, mol, backbone):
    """Attach MIPS atom features and the paper's separate backbone indicator."""
    data.mips_x = _mips_atom_feature_rows(mol)
    data.mips_backbone_mask = torch.zeros(mol.GetNumAtoms(), dtype=torch.long)
    if backbone:
        data.mips_backbone_mask[torch.tensor(sorted(backbone), dtype=torch.long)] = 1
    data.mips_input_schema_version = 3
    return data


def attach_polymerized_mips_atom_features(data, smiles):
    """Use an open topology-only Trimer to define polymerized atom features.

    O8 contains a virtual closing edge, so atom degree/H/hybridization must
    describe an interior polymer atom rather than the finite repeated graph's
    artificial termini.  The central unit of an open Trimer has both real
    inter-RU bonds.  Its rows are indexed by canonical RU atom ID and then
    broadcast to every O8 repeat copy.
    """
    trimer, metadata = build_periodic_multimer_mol(
        smiles, num_repeat_units=3, close_periodic=False
    )
    unit_atoms = metadata.get("unit_atoms") or []
    if len(unit_atoms) != 3:
        raise ValueError("polymerized MIPS features require a three-unit chain")
    central = torch.tensor(unit_atoms[1], dtype=torch.long)
    canonical = data.canonical_ru_atom_index.long()
    if canonical.numel() != int(data.num_nodes):
        raise ValueError("canonical MIPS mapping is incomplete")
    if canonical.numel() and int(canonical.max()) + 1 != central.numel():
        raise ValueError("Trimer/O8 canonical atom counts differ")
    rows = _mips_atom_feature_rows(trimer)[central]
    data.mips_x = rows[canonical].clone()
    data.mips_input_schema_version = 4
    data.mips_atom_feature_source = "topology_only_trimer_central_ru"
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


def build_mips_data_object(mol, backbone_info=None):
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
    attachment_bond_mismatch = bond_types[0] != bond_types[1]
    if attachment_bond_mismatch:
        # Public MIPS creates repeat/Star connections as single bonds.  Use
        # that direction-invariant convention only when the two P-SMILES
        # attachment bonds disagree; otherwise preserve the encoded bond.
        connection_bond_type = Chem.rdchem.BondType.SINGLE
        connection_bond_policy = "mismatch_single"
    elif bond_types[0] == Chem.rdchem.BondType.AROMATIC:
        connection_bond_type = Chem.rdchem.BondType.SINGLE
        connection_bond_policy = "aromatic_single"
    else:
        connection_bond_type = bond_types[0]
        connection_bond_policy = "matching_attachment_type"

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
    missing_attachment_neighbors = [
        int(index) for index in neighbors if int(index) not in old_to_base
    ]
    if missing_attachment_neighbors:
        raise ValueError(
            "periodic multimer attachment neighbor is a dummy atom or "
            "otherwise absent from the base RU: "
            + ",".join(str(index) for index in missing_attachment_neighbors)
        )
    left_base = int(old_to_base[int(neighbors[0])])
    right_base = int(old_to_base[int(neighbors[1])])
    shared_boundary = left_base == right_base
    backbone_base = (
        [left_base]
        if shared_boundary
        else list(Chem.GetShortestPath(base_mol, left_base, right_base))
    )
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

    # Restore bond metadata from the ORIGINAL bonds only after all real seam
    # neighbors exist. AddBond copies the order, not direction/stereo atoms.
    # Keep the records until after sanitization: SanitizeMol is allowed to
    # update bond caches/aromaticity, but it must not be the operation that
    # silently decides whether a stereo reference survived.
    bond_restores = []
    stereo_restore_failures = []
    terminal_stereo_unset = []
    defined_stereo = {
        Chem.BondStereo.STEREOE,
        Chem.BondStereo.STEREOZ,
        Chem.BondStereo.STEREOCIS,
        Chem.BondStereo.STEREOTRANS,
    }
    for unit_idx, atom_map in enumerate(atom_maps):
        mapping = {old: atom_map[base_id] for old, base_id in old_to_base.items()}
        reference_map = dict(mapping)
        if unit_idx > 0:
            reference_map[int(dummy_atoms[0])] = atom_maps[unit_idx - 1][right_base]
        if unit_idx + 1 < num_repeat_units:
            reference_map[int(dummy_atoms[1])] = atom_maps[unit_idx + 1][left_base]
        # Stereo and directional metadata come from the pristine input, not
        # from the temporary Kekule representation used to copy bond orders.
        for original in source.GetBonds():
            begin, end = original.GetBeginAtomIdx(), original.GetEndAtomIdx()
            if begin not in mapping or end not in mapping:
                continue
            copied = combined.GetBondBetweenAtoms(mapping[begin], mapping[end])
            if copied is None:
                stereo_restore_failures.append({
                    "unit": int(unit_idx),
                    "bond": [int(begin), int(end)],
                    "reason": "copied bond is missing",
                })
                continue
            stereo = original.GetStereo()
            references = list(original.GetStereoAtoms())
            resolved = []
            unresolved = False
            if references:
                if len(references) != 2:
                    if stereo in defined_stereo:
                        raise ValueError(
                            "defined stereo bond has an invalid reference count"
                        )
                    unresolved = True
                else:
                    for endpoint, other, reference in zip(
                        (begin, end), (end, begin), references
                    ):
                        if reference in reference_map:
                            resolved.append(reference_map[reference])
                            continue
                        if reference not in dummy_atoms:
                            raise ValueError(
                                "stereo reference atom is not present in the source graph"
                            )
                        # A terminal dummy has no physical neighbour in the
                        # finite open chain.  It is an implicit-H/provenance
                        # reference, not an invitation to choose another
                        # explicit substituent.  Assigning that alternative
                        # and flipping E/Z changes the chemical meaning of the
                        # source bond, so the terminal copy is intentionally
                        # left unspecified.
                        unresolved = True
                        break
            elif stereo in defined_stereo:
                raise ValueError(
                    "defined stereo bond is missing its two reference atoms"
                )
            if unresolved or (references and len(resolved) != 2):
                terminal_stereo_unset.append({
                    "unit": int(unit_idx),
                    "bond": [int(begin), int(end)],
                    "source_stereo": str(stereo),
                    "reason": "terminal or incomplete stereo reference",
                })
                resolved = []
                restore_stereo = Chem.BondStereo.STEREONONE
            else:
                restore_stereo = stereo
            bond_restores.append({
                "unit": int(unit_idx),
                "begin": int(mapping[begin]),
                "end": int(mapping[end]),
                "bond_dir": original.GetBondDir(),
                "stereo": restore_stereo,
                "references": tuple(int(value) for value in resolved),
                "source_stereo": stereo,
            })

    result = combined.GetMol()
    Chem.SanitizeMol(result)
    # SetStereoAtoms must precede SetStereo according to RDKit's bond API.  Do
    # this after sanitization so the references are checked against the final
    # atom/bond graph and cannot be erased by a later sanitize pass.
    for restore in bond_restores:
        copied = result.GetBondBetweenAtoms(restore["begin"], restore["end"])
        if copied is None:
            stereo_restore_failures.append({
                "unit": restore["unit"],
                "bond": [restore["begin"], restore["end"]],
                "reason": "bond disappeared during sanitization",
            })
            continue
        copied.SetBondDir(restore["bond_dir"])
        references = restore["references"]
        if len(references) == 2:
            copied.SetStereoAtoms(*references)
        copied.SetStereo(restore["stereo"])
    if stereo_restore_failures:
        raise ValueError(
            "stereo metadata restoration failed: "
            + str(stereo_restore_failures[:3])
        )
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
        "source_atom_to_base": [
            int(old_to_base.get(idx, -1)) for idx in range(source.GetNumAtoms())
        ],
        "base_atom_to_source": [
            int(idx) for idx in sorted(old_to_base, key=old_to_base.get)
        ],
        "source_attachment_neighbors": [int(idx) for idx in neighbors],
        "stereo_restore_failures": stereo_restore_failures,
        "terminal_stereo_unset": terminal_stereo_unset,
        "inter_unit_edges": inter_unit_edges,
        "periodic_edge": periodic_edge,
        "attachment_bond_type": connection_bond_type,
        "attachment_bond_type_left": str(bond_types[0]),
        "attachment_bond_type_right": str(bond_types[1]),
        "attachment_bond_mismatch": bool(attachment_bond_mismatch),
        "connection_bond_policy": connection_bond_policy,
        "shared_boundary": bool(shared_boundary),
        "multimer_builder_version": MIPS_MULTIMER_BUILDER_VERSION,
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
        "topology_failure_code": "",
        "periodic_metadata": metadata,
    }




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


def repeat_cut_augment_smiles(
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
        raise ValueError("repeat_cut_augment_smiles requires exactly two dummy atoms")
    if len(set(neighbors)) != 2:
        raise ValueError("repeat_cut_augment_smiles requires two distinct attachment neighbors")
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
    existing_bond = mol.GetBondBetweenAtoms(
        int(neighbors[0]), int(neighbors[1])
    )
    added_link = existing_bond is None
    editable = Chem.EditableMol(mol)
    if added_link:
        editable.AddBond(int(neighbors[0]), int(neighbors[1]), order=bond_types[0])
    for atom_idx in sorted(dummy_atoms, reverse=True):
        editable.RemoveAtom(int(atom_idx))

    linked_mol = editable.GetMol()
    Chem.SanitizeMol(linked_mol)
    remaining = [idx for idx in range(mol.GetNumAtoms()) if idx not in set(dummy_atoms)]
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(remaining)}
    mapped_neighbors = [old_to_new[int(idx)] for idx in neighbors]
    actual_link = linked_mol.GetBondBetweenAtoms(*mapped_neighbors)
    if actual_link is None:
        raise ValueError("Star-Linking edge disappeared during sanitization")
    raw_backbone = list(Chem.GetShortestPath(mol, int(neighbors[0]), int(neighbors[1])))
    mapped_backbone = [old_to_new[int(idx)] for idx in raw_backbone if int(idx) in old_to_new]
    mapping = {
        "old_to_new": old_to_new,
        "attachment_pair": mapped_neighbors,
        "star_link_edge": mapped_neighbors,
        "ordered_backbone_path": mapped_backbone,
        # Preserve the declared policy and the actual physical edge
        # separately.  If the boundary atoms were already bonded, Star-Linking
        # does not add a second edge; its actual attributes come from that
        # existing bond rather than from the attachment declaration.
        "connection_policy": (
            "matching_attachment_type"
            if bond_types[0] == bond_types[1]
            else "mismatch_single"
        ),
        "declared_attachment_bond_type_left": str(bond_types[0]),
        "declared_attachment_bond_type_right": str(bond_types[1]),
        "actual_link_added": bool(added_link),
        "actual_link_bond_type": str(
            actual_link.GetBondType()
        ),
        "actual_link_stereo": str(
            actual_link.GetStereo()
        ),
        "actual_link_conjugated": bool(
            actual_link.GetIsConjugated()
        ),
        "actual_link_ring": bool(
            actual_link.IsInRing()
        ),
    }
    return (linked_mol, mapping) if return_mapping else linked_mol


def build_structure_for_input(smiles, graph_input="repeat_unit"):
    """Resolve the MTS repeat-unit or Star-Linking topology graph.

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
            "topology_failure_code": "",
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
            "topology_failure_code": "",
        }
    except Exception as exc:
        return {
            **base,
            "structure_smiles": smiles,
            "structure_mol": mol,
            "structure_input": "repeat_unit",
            "graph_build_ok": False,
            "topology_failure_code": str(exc)[:200],
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
    if len(dummy_atoms) != 2:
        raise ValueError("MIPS requires exactly two attachment sites")

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
                "topology_failure_code": "",
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


def build_mips_local_structure(smiles_or_mol, config=None):
    """Build the finite, non-PBC repeated graph used by the second route.

    The terminal connection is a virtual topological edge.  It is deliberately
    not represented as an RDKit ring and carries no cell, translation or image
    shift.  Repetition stops at the smallest integer RU count satisfying the
    MIPS boundary condition.
    """

    config = config or MIPSLocalConfig()
    last_reason = ""
    candidates = {}

    def evaluate(repeat_units):
        nonlocal last_reason
        if repeat_units in candidates:
            return candidates[repeat_units]
        try:
            open_mol, open_meta = build_periodic_multimer_mol(
                smiles_or_mol, repeat_units, close_periodic=False
            )
            atom_count = int(open_mol.GetNumAtoms())
            if atom_count > int(config.max_model_atoms):
                last_reason = (
                    f"model_atoms={atom_count}>{int(config.max_model_atoms)}"
                )
                result = None
                candidates[repeat_units] = result
                return result
            left_boundary = int(open_meta["left_boundary"])
            right_boundary = int(open_meta["right_boundary"])
            boundary_path = (
                [left_boundary]
                if left_boundary == right_boundary
                else list(Chem.GetShortestPath(
                    open_mol, left_boundary, right_boundary
                ))
            )
            boundary_distance = len(boundary_path) - 1
            open_meta["ordered_backbone_path"] = boundary_path
            if boundary_distance <= int(config.required_boundary_distance):
                last_reason = (
                    f"boundary_distance={boundary_distance}<="
                    f"{int(config.required_boundary_distance)}"
                )
                result = False
            else:
                result = (open_mol, open_meta, boundary_distance)
            candidates[repeat_units] = result
            return result
        except Exception as exc:
            last_reason = str(exc)[:200]
            candidates[repeat_units] = False
            return False

    upper = None
    repeat_units = 1
    previous = 0
    while repeat_units <= int(config.max_repeat_units):
        result = evaluate(repeat_units)
        if result is None:
            break
        if result:
            upper = repeat_units
            break
        previous = repeat_units
        repeat_units *= 2
    if upper is not None:
        # The boundary distance is monotone with RU count. Search only inside
        # the final doubling bracket to recover the smallest legal integer.
        selected = None
        for repeat_units in range(previous + 1, upper + 1):
            result = evaluate(repeat_units)
            if result:
                selected = (repeat_units, *result)
                break
        if selected is None:
            raise RuntimeError("MIPS doubling bracket lost its valid endpoint")
        repeat_units, open_mol, open_meta, boundary_distance = selected
        tail = int(open_meta["right_boundary"])
        head = int(open_meta["left_boundary"])
        if tail == head:
            raise RuntimeError(
                "MIPS boundary condition produced a Star self-loop"
            )
        bond_type = open_meta["attachment_bond_type"]
        backbone_info = {
            "neighbors": [head, tail],
            "star_link_edge": [tail, head],
            "ordered_backbone_path": list(open_meta["ordered_backbone_path"]),
            "virtual_edges": [(tail, head, bond_type)],
        }
        return {
            "requested_input": "star_linking",
            "attachment_count": 2,
            "backbone_info": backbone_info,
            "structure_smiles": Chem.MolToSmiles(open_mol, canonical=False),
            "structure_mol": open_mol,
            "structure_input": "mips_local_star_linking",
            "graph_build_ok": True,
            "topology_failure_code": "",
            "graph_available": True,
            "repeat_metadata": open_meta,
            "mips_repeat_units": int(repeat_units),
            "mips_boundary_distance": int(boundary_distance),
            "mips_distance_threshold": int(config.distance_threshold),
            "mips_condition_valid": True,
        }

    # Preserve the other modalities with a single-RU placeholder.  It receives
    # self-only attention and is masked by graph_available at fusion time.
    try:
        placeholder, metadata = build_periodic_multimer_mol(
            smiles_or_mol, 1, close_periodic=False
        )
    except Exception:
        original = (
            Chem.Mol(smiles_or_mol)
            if isinstance(smiles_or_mol, Chem.Mol)
            else Chem.MolFromSmiles(str(smiles_or_mol))
        )
        dummy_atoms, neighbors, bond_types = _get_dummy_atoms_and_neighbors(
            original
        )
        if len(dummy_atoms) != 2 or len(neighbors) != 2:
            raise
        keep = [
            atom_idx for atom_idx in range(original.GetNumAtoms())
            if atom_idx not in set(dummy_atoms)
        ]
        old_to_new = {
            atom_idx: new_idx for new_idx, atom_idx in enumerate(keep)
        }
        editable = Chem.RWMol(original)
        for atom_idx in sorted(dummy_atoms, reverse=True):
            editable.RemoveAtom(int(atom_idx))
        placeholder = editable.GetMol()
        Chem.SanitizeMol(placeholder)
        left_boundary = old_to_new[int(neighbors[0])]
        right_boundary = old_to_new[int(neighbors[1])]
        metadata = {
            "left_boundary": left_boundary,
            "right_boundary": right_boundary,
            "unit_left_boundaries": [left_boundary],
            "unit_right_boundaries": [right_boundary],
            "unit_atoms": [list(range(placeholder.GetNumAtoms()))],
            "base_atom_count": int(placeholder.GetNumAtoms()),
            "ordered_backbone_path": [left_boundary],
            "attachment_bond_type": bond_types[0],
        }
    pair = [int(metadata["left_boundary"]), int(metadata["right_boundary"])]
    return {
        "requested_input": "star_linking",
        "attachment_count": 2,
        "backbone_info": {
            "neighbors": pair,
            "star_link_edge": pair,
            "ordered_backbone_path": list(metadata["ordered_backbone_path"]),
            "virtual_edges": [],
        },
        "structure_smiles": Chem.MolToSmiles(placeholder, canonical=False),
        "structure_mol": placeholder,
        "structure_input": "mips_local_unavailable",
        "graph_build_ok": False,
        "topology_failure_code": f"mips_condition_failed:{last_reason}"[:240],
        "graph_available": False,
        "repeat_metadata": metadata,
        "mips_repeat_units": 1,
        "mips_boundary_distance": len(metadata["ordered_backbone_path"]) - 1,
        "mips_distance_threshold": int(config.distance_threshold),
        "mips_condition_valid": False,
    }


def _mips_local_adjacency(mol, virtual_edges):
    adjacency = [[] for _ in range(mol.GetNumAtoms())]
    for bond in mol.GetBonds():
        begin, end = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
        codes = _bond_path_codes(bond, False)
        adjacency[begin].append((end, codes))
        adjacency[end].append((begin, codes))
    for begin, end, bond_type in virtual_edges:
        bond_code = allowable_features["possible_bonds"].index(bond_type) + 1
        codes = (bond_code, 1, 1, 1, 2)
        adjacency[int(begin)].append((int(end), codes))
        adjacency[int(end)].append((int(begin), codes))
    for neighbors in adjacency:
        neighbors.sort(key=lambda item: item[0])
    return adjacency


def _bounded_distances(start, adjacency, max_hops):
    distances = {int(start): 0}
    queue = deque([int(start)])
    while queue:
        node = queue.popleft()
        if distances[node] >= int(max_hops):
            continue
        for neighbor, _ in adjacency[node]:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def attach_mips_local_lga(data, structure, config=None):
    """Attach fixed-O8 sparse edges and one deterministic shortest path."""

    config = config or MIPSLocalConfig()
    graph_available = bool(structure.get("graph_available", False))
    mol = structure["structure_mol"]
    virtual_edges = (structure.get("backbone_info") or {}).get("virtual_edges", [])
    adjacency = _mips_local_adjacency(mol, virtual_edges)
    num_atoms = int(mol.GetNumAtoms())

    # Explicit canonical/orbit mapping prevents repeated copies leaking masked
    # atom targets and prevents short RUs receiving a larger loss weight.
    metadata = structure.get("repeat_metadata") or {}
    unit_atoms = metadata.get("unit_atoms") or [list(range(num_atoms))]
    canonical = torch.full((num_atoms,), -1, dtype=torch.long)
    ru_copy = torch.zeros(num_atoms, dtype=torch.long)
    for copy_idx, atom_indices in enumerate(unit_atoms):
        for canonical_idx, atom_idx in enumerate(atom_indices):
            canonical[int(atom_idx)] = int(canonical_idx)
            ru_copy[int(atom_idx)] = int(copy_idx)
    if bool((canonical < 0).any()):
        raise RuntimeError("MIPS canonical RU mapping is incomplete")

    sources, targets, spd_values = [], [], []
    representative_paths = []
    bond_histograms = []
    canonical_pairs = []
    adjacency_set = {
        (int(left), int(right))
        for left, neighbors in enumerate(adjacency)
        for right, _ in neighbors
    }

    def _representative_path(source, distance, source_distances, target_distances):
        """Rebuild one continuous shortest path from source to target.

        The per-position member sets can contain nodes that are not adjacent
        across positions (multiple shortest paths in cyclic topologies).  The
        legacy ``members[0]`` choice then produced a non-contiguous path and a
        ``StopIteration`` on the edge lookup.  Fall back to a greedy
        continuous reconstruction; if none exists, return an empty path so the
        caller records an all-masked path instead of crashing.
        """
        path = [source]
        current = source
        for position in range(1, int(distance) + 1):
            candidates = sorted(
                node for node, source_distance in source_distances.items()
                if source_distance == position
                and node in target_distances
                and source_distance + target_distances[node] == distance
            )
            chosen = next(
                (
                    node for node in candidates
                    if (current, node) in adjacency_set
                    or (node, current) in adjacency_set
                ),
                None,
            )
            if chosen is None:
                return []
            path.append(chosen)
            current = chosen
        return path

    for target in range(num_atoms):
        if graph_available:
            target_distances = _bounded_distances(
                target, adjacency, config.max_hops
            )
            source_nodes = sorted(
                target_distances,
                key=lambda node: (target_distances[node], node),
            )
        else:
            target_distances = {target: 0}
            source_nodes = [target]

        for source in source_nodes:
            distance = int(target_distances[source])
            sources.append(int(source))
            targets.append(int(target))
            spd_values.append(distance)
            canonical_pairs.append(
                int(canonical[source]) * max(1, int(canonical.max()) + 1)
                + int(canonical[target])
            )

            source_distances = _bounded_distances(
                source, adjacency, config.max_hops
            )
            representative_path = _representative_path(
                source, distance, source_distances, target_distances
            )
            representative_paths.append(representative_path)

            histogram = torch.zeros(
                int(config.max_hops), 6, dtype=torch.float
            )
            if representative_path:
                for position in range(distance):
                    left = representative_path[position]
                    right = representative_path[position + 1]
                    codes = next(
                        edge_codes
                        for neighbor, edge_codes in adjacency[left]
                        if neighbor == right
                    )
                    # Six categories: four bond types, other, and virtual star.
                    category = (
                        5 if int(codes[4]) == 2
                        else min(4, int(codes[0]) - 1)
                    )
                    histogram[position, category] = 1.0
            bond_histograms.append(histogram)

    edge_count = len(sources)
    path_width = int(config.max_hops) + 1
    path_index = torch.full((edge_count, path_width), -1, dtype=torch.long)
    path_mask = torch.zeros((edge_count, path_width), dtype=torch.bool)
    for edge_idx, path in enumerate(representative_paths):
        path_index[edge_idx, :len(path)] = torch.tensor(path, dtype=torch.long)
        path_mask[edge_idx, :len(path)] = True

    data.lga_edge_index = torch.tensor([sources, targets], dtype=torch.long)
    data.lga_spd = torch.tensor(spd_values, dtype=torch.long)
    data.lga_path_index = path_index
    data.lga_path_mask = path_mask
    data.lga_path_bond_hist = (
        torch.stack(bond_histograms, dim=0)
        if bond_histograms else torch.zeros((0, int(config.max_hops), 6))
    )
    star_edge = (
        (structure.get("backbone_info") or {}).get("star_link_edge") or []
    )
    if len(star_edge) == 2:
        left, right = (int(star_edge[0]), int(star_edge[1]))
        data.lga_star_edge_mask = torch.tensor(
            [
                (source == left and target == right)
                or (source == right and target == left)
                for source, target in zip(sources, targets)
            ],
            dtype=torch.bool,
        )
    else:
        data.lga_star_edge_mask = torch.zeros(edge_count, dtype=torch.bool)
    data.canonical_ru_atom_index = canonical
    data.ru_copy_index = ru_copy
    data.canonical_pair_index = torch.tensor(canonical_pairs, dtype=torch.long)
    data.graph_available = bool(graph_available)
    data.mips_condition_valid = bool(
        structure.get("mips_condition_valid", graph_available)
    )
    data.mips_alias_free = bool(graph_available)
    data.mips_local_lga_schema_version = LEGACY_MIPS_LOCAL_LGA_SCHEMA_VERSION
    # This helper is the retired explicit finite-copy reference.  Label it
    # with the legacy identity so production cache validation cannot mistake
    # it for canonical periodic topology.
    data.feature_schema = LEGACY_FEATURE_SCHEMA
    return data


def annotate_structure_fields(data, structure, prefix="graph"):
    """Attach auditable structure metadata to a PyG Data object."""
    setattr(data, f"{prefix}_smiles", structure["structure_smiles"])
    setattr(data, f"{prefix}_input", structure["structure_input"])
    data.requested_graph_input = structure["requested_input"]
    data.structure_smiles = structure["structure_smiles"]
    data.structure_input = structure["structure_input"]
    data.graph_build_ok = bool(structure["graph_build_ok"])
    data.topology_failure_code = structure["topology_failure_code"]
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
            "mips_periodic_supercell", "mips_local_star_linking",
        }
        and len(pair) == 2
        and len(star_edge) == 2
        and len(path) >= 2
    )
    repeat_metadata = (
        structure.get("repeat_metadata")
        or structure.get("periodic_metadata")
        or {}
    )
    ru_index = repeat_metadata.get("atom_ru_index")
    if ru_index is None:
        ru_index = [0] * int(data.x.size(0))
    data.scage_ru_index = torch.tensor(ru_index, dtype=torch.long)
    data.mips_repeat_units = int(structure.get("mips_repeat_units", 1))
    data.mips_boundary_distance = int(structure.get("mips_boundary_distance", -1))
    data.mips_distance_threshold = int(structure.get("mips_distance_threshold", 3))
    data.attachment_bond_type_left = str(
        repeat_metadata.get("attachment_bond_type_left", "")
    )
    data.attachment_bond_type_right = str(
        repeat_metadata.get("attachment_bond_type_right", "")
    )
    data.attachment_bond_mismatch = bool(
        repeat_metadata.get("attachment_bond_mismatch", False)
    )
    data.connection_bond_policy = str(
        repeat_metadata.get("connection_bond_policy", "")
    )
    data.shared_attachment_boundary = bool(
        repeat_metadata.get("shared_boundary", False)
    )
    data.multimer_builder_version = int(
        repeat_metadata.get("multimer_builder_version", 0)
    )
    data.graph_available = bool(structure.get("graph_available", data.graph_build_ok))
    if data.structure_input.startswith("mips_local_"):
        data.mips_repeat_factor = data.mips_repeat_units
    else:
        data.periodic_ru_count = int(
            repeat_metadata.get("num_repeat_units", 1)
        )
        data.geometry_period_ru = int(structure.get("geometry_period_ru", 0))
        data.mips_repeat_factor = int(structure.get("mips_repeat_factor", 1))
        data.model_cell_ru = int(
            structure.get("model_cell_ru", data.periodic_ru_count)
        )
    data.mips_condition_valid = bool(
        structure.get("mips_condition_valid", data.graph_available)
    )
    return data


def build_mips_graph_for_input(smiles, graph_input="star_linking"):
    """Build the sparse O8 MTS graph for a P-SMILES string."""
    structure = build_structure_for_input(smiles, graph_input)
    data = build_mips_data_object(
        structure["structure_mol"], backbone_info=structure.get("backbone_info")
    )
    return annotate_structure_fields(data, structure, prefix="graph")


def build_canonical_periodic_topology(smiles_or_mol, max_hops=2):
    """Native single-canonical-RU MTS topology builder.

    The implementation lives in :mod:`canonical_periodic` to keep the legacy
    explicit reference construction isolated.  This lazy wrapper avoids a
    graph_data/canonical_periodic import cycle and gives callers one natural
    builder entry point.
    """

    from .canonical_periodic import build_canonical_periodic_topology as _build
    return _build(smiles_or_mol, max_hops=max_hops)


def build_lifted_periodic_relations(smiles_or_mol, max_hops=2):
    """Compatibility wrapper for the canonical lifted-relation builder."""

    return build_canonical_periodic_topology(smiles_or_mol, max_hops=max_hops)
