"""Test-only corrected explicit ``k``-RU reference.

The production MTS path contains one canonical RU state and lifted relation
rows.  This helper is intentionally isolated under ``tests/`` so parity tests
can compare that path with a finite translation-symmetric construction without
making the retired explicit representation available to Dataset/model code.
"""

from collections.abc import Mapping

from rdkit import Chem
import torch
from torch_geometric.data import Data

from src.dataset.graph_data import build_periodic_multimer_mol
from src.dataset.mips_trimer_contract import (
    CHECKPOINT_SCHEMA,
    EXPLICIT_FEATURE_SCHEMA,
    EXPLICIT_LGA_SCHEMA_VERSION,
    EXPLICIT_TOPOLOGY_LMDB_SCHEMA,
    TOPOLOGY_EXPLICIT,
)


def _build_corrected_explicit_k_ru_reference_legacy(canonical_topology, repeat_factor):
    """Construct the finite translation-symmetric graph used only in tests."""

    k = int(repeat_factor)
    if k < 1:
        raise ValueError("repeat_factor must be >= 1")
    relation = torch.as_tensor(canonical_topology.lga_edge_index).long()
    shifts = torch.as_tensor(
        getattr(
            canonical_topology,
            "lga_source_image_shift",
            torch.zeros(relation.size(1)),
        ),
    ).long().reshape(-1)
    if shifts.numel() != relation.size(1):
        raise ValueError("canonical relation shift length mismatch")
    n = int(canonical_topology.mips_x.size(0))
    source_rows, target_rows, shift_rows = [], [], []
    spd_rows, polymer_rows = [], []
    path_rows, path_shift_rows = [], []
    canonical_polymer = torch.as_tensor(
        getattr(
            canonical_topology,
            "polymer_link_mask",
            torch.zeros(relation.size(1)),
        ),
    ).bool().reshape(-1)
    canonical_spd = torch.as_tensor(canonical_topology.lga_spd).long().reshape(-1)
    canonical_path = torch.as_tensor(canonical_topology.lga_path_index).long()
    canonical_path_shift = torch.as_tensor(
        getattr(
            canonical_topology,
            "lga_path_shift",
            torch.zeros_like(canonical_path),
        )
    ).long()
    for copy_index in range(k):
        for row in range(relation.size(1)):
            source_atom, target_atom = map(int, relation[:, row])
            source_copy = (copy_index + int(shifts[row])) % k
            source_rows.append(source_atom + source_copy * n)
            target_rows.append(target_atom + copy_index * n)
            shift_rows.append(int(shifts[row]))
            spd_rows.append(int(canonical_spd[row]))
            polymer_rows.append(bool(canonical_polymer[row]))
            path = canonical_path[row].clone()
            path_shift = canonical_path_shift[row].clone()
            valid = path >= 0
            lifted_path = path.clone()
            lifted_path[valid] += (copy_index + path_shift[valid]) % k * n
            path_rows.append(lifted_path)
            path_shift_rows.append(path_shift)

    output = Data()
    output.mips_x = torch.as_tensor(canonical_topology.mips_x).float().repeat(k, 1)
    output.x = output.mips_x.clone()
    output.mips_backbone_mask = torch.as_tensor(
        canonical_topology.mips_backbone_mask
    ).long().repeat(k)
    atomic = torch.as_tensor(
        getattr(canonical_topology, "atomic_numbers", canonical_topology.z)
    ).long()
    output.atomic_numbers = atomic.repeat(k)
    output.atomic_number = output.atomic_numbers
    output.z = output.atomic_numbers
    output.lga_edge_index = torch.tensor(
        [source_rows, target_rows], dtype=torch.long
    )
    output.canonical_lga_edge_index = output.lga_edge_index.clone()
    output.lga_spd = torch.tensor(spd_rows, dtype=torch.long)
    output.lga_path_index = torch.stack(path_rows, dim=0)
    output.lga_path_shift = torch.stack(path_shift_rows, dim=0)
    output.lga_path_shifts = output.lga_path_shift
    output.lga_path_mask = output.lga_path_index >= 0
    output.lga_path_bond_hist = torch.zeros(
        (len(path_rows), max(0, output.lga_path_index.size(1) - 1)),
        dtype=torch.float,
    ).unsqueeze(-1).expand(-1, -1, 6).clone()
    output.lga_source_image_shift = torch.tensor(shift_rows, dtype=torch.long)
    output.lga_relation_shift = output.lga_source_image_shift
    output.polymer_link_mask = torch.tensor(polymer_rows, dtype=torch.bool)
    output.lga_polymer_link_mask = output.polymer_link_mask
    output.lga_star_edge_mask = output.polymer_link_mask
    output.canonical_ru_atom_index = torch.arange(n, dtype=torch.long).repeat(k)
    output.ru_copy_index = torch.arange(k, dtype=torch.long).repeat_interleave(n)
    output.canonical_pair_index = torch.zeros(
        output.lga_edge_index.size(1), dtype=torch.long
    )
    output.graph_available = torch.tensor(True, dtype=torch.bool)
    output.mips_condition_valid = torch.tensor(True, dtype=torch.bool)
    output.mips_boundary_distance = torch.full((k,), 6, dtype=torch.long)
    for name in ("mips_md", "mips_md_valid"):
        if not hasattr(canonical_topology, name):
            continue
        value = getattr(canonical_topology, name)
        if name == "mips_md":
            value = torch.as_tensor(value).float()
            if value.ndim == 1:
                value = value.unsqueeze(0)
        elif name == "mips_md_valid":
            value = torch.as_tensor(value).bool().reshape(-1)
        elif torch.is_tensor(value):
            value = value.clone()
        setattr(output, name, value)
    output.batch = torch.zeros(k * n, dtype=torch.long)
    output.num_nodes = k * n
    if hasattr(canonical_topology, "trimer_pos"):
        for name in (
            "trimer_pos", "trimer_atomic_number", "trimer_edge_index",
            "trimer_bond_type", "trimer_base_ru_atom_id",
            "trimer_base_ru_atom_index", "trimer_ru_offset",
            "trimer_central_ru_mask", "trimer_central_atom_index",
            "trimer_geometry_valid", "trimer_geometry_is_3d",
            "trimer_2d_fallback", "trimer_geometry_source",
            "trimer_failure_code", "trimer_conformer_energy",
            "star_3d_distance", "star_3d_asymmetry", "star_3d_valid",
            "trimer_conformer_seed", "trimer_conformer_method",
            "trimer_mcl_schema", "trimer_mcl_schema_version",
            "trimer_mcl_thresholds", "mcl_valid", "trimer_angle_index",
            "trimer_angle_bins", "trimer_angle_cos", "trimer_angle_valid",
        ):
            if hasattr(canonical_topology, name):
                value = getattr(canonical_topology, name)
                if torch.is_tensor(value):
                    value = value.clone()
                setattr(output, name, value)
        mapping = getattr(canonical_topology, "mips_to_trimer_central_index", None)
        if mapping is not None:
            output.mips_to_trimer_central_index = torch.as_tensor(mapping).long().repeat(k)
        output.trimer_batch = torch.zeros(
            int(output.trimer_pos.size(0)), dtype=torch.long
        )
        if hasattr(output, "trimer_base_ru_atom_id") and not hasattr(
            output, "trimer_base_ru_atom_index"
        ):
            output.trimer_base_ru_atom_index = output.trimer_base_ru_atom_id
        for name in (
            "trimer_geometry_valid", "trimer_geometry_is_3d",
            "trimer_2d_fallback", "star_3d_distance", "star_3d_asymmetry",
            "star_3d_valid", "mcl_valid", "trimer_mcl_thresholds",
            "trimer_angle_valid",
        ):
            if hasattr(output, name) and not torch.is_tensor(getattr(output, name)):
                setattr(output, name, torch.as_tensor(getattr(output, name)).reshape(-1))
    output.repeat_factor = k
    output.canonical_node_count = n
    output.translation_symmetric_reference = True
    output.mts_canonical_periodic = False
    output.topology_representation = TOPOLOGY_EXPLICIT
    output.mts_topology_representation = TOPOLOGY_EXPLICIT
    output.feature_schema = EXPLICIT_FEATURE_SCHEMA
    output.explicit_k_ru_topology_schema = EXPLICIT_TOPOLOGY_LMDB_SCHEMA
    output.mips_local_lga_schema_version = EXPLICIT_LGA_SCHEMA_VERSION
    output.canonical_graph_index = torch.zeros(n, dtype=torch.long)
    output.canonical_local_index = torch.arange(n, dtype=torch.long)
    output.canonical_first_node_index = torch.arange(n, dtype=torch.long)
    return output


def _bond_code(bond):
    if bond is None:
        return 0
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


def build_corrected_explicit_k_ru_reference(canonical_topology, repeat_factor):
    """Build an independent finite periodic reference from P-SMILES/RDKit.

    No canonical relation tensor is read here.  The reference constructs a
    closed translation-symmetric k-RU molecule, runs its own bounded BFS and
    derives SPD/path/shift/polymer-link rows from that graph.  Tests choose
    ``k >= 2 * max_hops + 1`` so a shortest path cannot alias around the ring.
    """

    k = int(repeat_factor)
    if k < 1:
        raise ValueError("repeat_factor must be >= 1")
    max_hops = int(getattr(canonical_topology, "canonical_lga_max_hops", 2))
    if k < 2 * max_hops + 1:
        raise ValueError("repeat_factor is too small for bounded periodic reference")
    text = getattr(canonical_topology, "normalized_canonical_smiles", None)
    text = text or getattr(canonical_topology, "smiles", None)
    molecule = Chem.MolFromSmiles(str(text)) if text is not None else None
    if molecule is None:
        raise ValueError("canonical topology has no parseable normalized P-SMILES")
    finite, metadata = build_periodic_multimer_mol(
        molecule, num_repeat_units=k, close_periodic=True
    )
    units = metadata.get("unit_atoms") or []
    if len(units) != k or not units or len({len(unit) for unit in units}) != 1:
        raise ValueError("independent explicit reference has invalid unit mapping")
    n = len(units[0])
    atom_to_unit = {}
    atom_to_base = {}
    for unit_index, unit in enumerate(units):
        for base_index, atom_index in enumerate(unit):
            atom_to_unit[int(atom_index)] = int(unit_index)
            atom_to_base[int(atom_index)] = int(base_index)
    adjacency = {int(atom.GetIdx()): [] for atom in finite.GetAtoms()}
    for bond in finite.GetBonds():
        left = int(bond.GetBeginAtomIdx()); right = int(bond.GetEndAtomIdx())
        code = _bond_code(bond)
        polymer = atom_to_unit[left] != atom_to_unit[right]
        adjacency[left].append((right, code, polymer))
        adjacency[right].append((left, code, polymer))
    # ``build_periodic_multimer_mol`` exposes the open chain even when the
    # close-periodic flag is requested; add the final translation edge in the
    # test reference explicitly.  Its bond follows the historical connection
    # policy (single unless the endpoints already carry a real bond).
    if k > 1:
        left_boundary = int((metadata.get("unit_left_boundaries") or [units[0][0]])[0])
        right_boundary = int((metadata.get("unit_right_boundaries") or [units[-1][-1]])[-1])
        if not any(int(neighbour) == left_boundary for neighbour, _code, _polymer in adjacency[right_boundary]):
            adjacency[right_boundary].append((left_boundary, 1, True))
            adjacency[left_boundary].append((right_boundary, 1, True))

    def relative_shift(source_unit, target_unit):
        delta = int(source_unit) - int(target_unit)
        if delta > k // 2:
            delta -= k
        elif delta < -(k // 2):
            delta += k
        return delta

    source_rows, target_rows, shifts = [], [], []
    spd_rows, polymer_rows, path_rows, path_shift_rows = [], [], [], []
    width = max_hops + 1
    for target_atom in range(finite.GetNumAtoms()):
        distances = {target_atom: 0}
        predecessor = {target_atom: None}
        queue = [target_atom]
        while queue:
            current = queue.pop(0)
            distance = int(distances[current])
            if distance >= max_hops:
                continue
            for neighbour, _code, _polymer in sorted(adjacency[current], key=lambda value: value[0]):
                if neighbour in distances:
                    continue
                distances[neighbour] = distance + 1
                predecessor[neighbour] = current
                queue.append(neighbour)
        target_unit = atom_to_unit[target_atom]
        ordered = sorted(
            distances.items(),
            key=lambda value: (
                int(value[1]),
                int(atom_to_base[value[0]]),
                int(relative_shift(atom_to_unit[value[0]], target_unit)),
            ),
        )
        for source_atom, distance in ordered:
            source_unit = atom_to_unit[source_atom]
            source_rows.append(int(source_atom))
            target_rows.append(int(target_atom))
            shifts.append(relative_shift(source_unit, target_unit))
            spd_rows.append(int(distance))
            polymer_rows.append(bool(distance == 1 and source_unit != target_unit))
            path = []
            current = int(source_atom)
            while current is not None:
                path.append(current)
                if current == target_atom:
                    break
                current = predecessor[current]
            path_rows.append([int(atom_to_base[node]) for node in path])
            path_shift_rows.append([
                relative_shift(atom_to_unit[node], target_unit) for node in path
            ])

    path_index = torch.full((len(path_rows), width), -1, dtype=torch.long)
    path_shift = torch.zeros((len(path_rows), width), dtype=torch.long)
    path_mask = torch.zeros((len(path_rows), width), dtype=torch.bool)
    for row, (nodes, row_shifts) in enumerate(zip(path_rows, path_shift_rows)):
        length = min(width, len(nodes))
        path_index[row, :length] = torch.tensor(nodes[:length], dtype=torch.long)
        path_shift[row, :length] = torch.tensor(row_shifts[:length], dtype=torch.long)
        path_mask[row, :length] = True

    output = Data()
    output.mips_x = torch.as_tensor(canonical_topology.mips_x).float().repeat(k, 1)
    output.x = output.mips_x.clone()
    output.mips_backbone_mask = torch.as_tensor(
        getattr(canonical_topology, "mips_backbone_mask", torch.zeros(n))
    ).long().repeat(k)
    atomic = torch.tensor(
        [int(finite.GetAtomWithIdx(idx).GetAtomicNum()) for idx in range(finite.GetNumAtoms())],
        dtype=torch.long,
    )
    output.atomic_numbers = atomic
    output.atomic_number = atomic
    output.z = atomic
    output.lga_edge_index = torch.tensor([source_rows, target_rows], dtype=torch.long)
    output.canonical_lga_edge_index = output.lga_edge_index.clone()
    output.lga_spd = torch.tensor(spd_rows, dtype=torch.long)
    output.spd = output.lga_spd.clone()
    output.lga_path_index = path_index
    output.lifted_single_path_index = path_index.clone()
    output.lga_path_shift = path_shift
    output.lga_path_shifts = path_shift.clone()
    output.lifted_single_path_shift = path_shift.clone()
    output.lga_path_mask = path_mask
    output.lifted_single_path_mask = path_mask.clone()
    output.lga_path_bond_hist = torch.zeros(
        (len(path_rows), max_hops, 6), dtype=torch.float
    )
    output.lga_source_image_shift = torch.tensor(shifts, dtype=torch.long)
    output.lga_relation_shift = output.lga_source_image_shift.clone()
    output.polymer_link_mask = torch.tensor(polymer_rows, dtype=torch.bool)
    output.lga_polymer_link_mask = output.polymer_link_mask.clone()
    output.lga_star_edge_mask = output.polymer_link_mask.clone()
    output.canonical_ru_atom_index = torch.arange(n, dtype=torch.long).repeat(k)
    output.ru_copy_index = torch.arange(k, dtype=torch.long).repeat_interleave(n)
    output.canonical_pair_index = torch.zeros(len(source_rows), dtype=torch.long)
    output.graph_available = torch.tensor(True, dtype=torch.bool)
    output.mips_condition_valid = torch.tensor(True, dtype=torch.bool)
    output.mips_boundary_distance = torch.full((k,), 6, dtype=torch.long)
    output.batch = torch.zeros(k * n, dtype=torch.long)
    output.num_nodes = k * n
    for name in ("mips_md", "mips_md_valid"):
        if hasattr(canonical_topology, name):
            value = getattr(canonical_topology, name)
            setattr(output, name, torch.as_tensor(value).clone())
    if hasattr(canonical_topology, "trimer_pos"):
        for name in (
            "trimer_pos", "trimer_atomic_number", "trimer_edge_index",
            "trimer_bond_type", "trimer_base_ru_atom_id", "trimer_base_ru_atom_index",
            "trimer_ru_offset", "trimer_central_ru_mask", "trimer_central_atom_index",
            "trimer_central_ru_atom_index", "canonical_to_trimer_central_index",
            "mips_to_trimer_central_index", "trimer_geometry_valid", "trimer_geometry_is_3d",
            "trimer_2d_fallback", "trimer_geometry_source", "trimer_failure_code",
            "trimer_conformer_energy", "star_3d_distance", "star_3d_asymmetry",
            "star_3d_valid", "trimer_conformer_seed", "trimer_conformer_method",
            "trimer_mcl_schema", "trimer_mcl_schema_version",
        ):
            if hasattr(canonical_topology, name):
                value = getattr(canonical_topology, name)
                setattr(output, name, value.clone() if torch.is_tensor(value) else value)
        mapping = getattr(canonical_topology, "mips_to_trimer_central_index", None)
        if mapping is not None:
            output.mips_to_trimer_central_index = torch.as_tensor(mapping).long().repeat(k)
        output.trimer_batch = torch.zeros(int(output.trimer_pos.size(0)), dtype=torch.long)
        for name in (
            "trimer_geometry_valid", "trimer_geometry_is_3d", "trimer_2d_fallback",
            "star_3d_valid", "star_3d_distance", "star_3d_asymmetry",
        ):
            if hasattr(output, name) and not torch.is_tensor(getattr(output, name)):
                setattr(output, name, torch.as_tensor(getattr(output, name)).reshape(-1))
    output.repeat_factor = k
    output.canonical_node_count = n
    output.translation_symmetric_reference = True
    output.mts_canonical_periodic = False
    output.topology_representation = TOPOLOGY_EXPLICIT
    output.mts_topology_representation = TOPOLOGY_EXPLICIT
    output.feature_schema = EXPLICIT_FEATURE_SCHEMA
    output.explicit_k_ru_topology_schema = EXPLICIT_TOPOLOGY_LMDB_SCHEMA
    output.mips_local_lga_schema_version = EXPLICIT_LGA_SCHEMA_VERSION
    output.canonical_graph_index = torch.zeros(n, dtype=torch.long)
    output.canonical_local_index = torch.arange(n, dtype=torch.long)
    output.canonical_first_node_index = torch.arange(n, dtype=torch.long)
    return output


def load_test_only_equivalence_checkpoint(model, checkpoint):
    payload = checkpoint
    if isinstance(checkpoint, (str, bytes)) or hasattr(checkpoint, "__fspath__"):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("meta"), Mapping):
        raise ValueError("equivalence checkpoint must contain metadata")
    if payload["meta"].get("schema") not in {"mts-model-v2", CHECKPOINT_SCHEMA}:
        raise ValueError("unsupported checkpoint for test-only equivalence")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("equivalence checkpoint has no state_dict")
    model.load_state_dict(state, strict=True)
    return model


__all__ = [
    "build_corrected_explicit_k_ru_reference",
    "load_test_only_equivalence_checkpoint",
]
