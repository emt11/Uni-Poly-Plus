"""Canonical periodic MTS topology and translation-symmetric references.

The original MTS cache materialised a finite ``k``-RU molecule and therefore
made the number of RU copies part of the learned node state.  This module
stores one state per RU atom and represents periodicity in relation rows.  A
relation source is ``(canonical_atom_id, relative_ru_shift)`` and its target
is ``(canonical_atom_id, 0)``.  Relative shifts are structural metadata; they
are deliberately never embedded by the model.

The implementation is intentionally dependency-light and uses the existing
``build_periodic_multimer_mol``/MIPS137 feature helpers.  It is also useful in
tests and in the one-record migration utility, so all construction functions
return ordinary PyG ``Data`` objects and avoid a cache-wide migration.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import torch
from torch_geometric.data import Data
from rdkit import Chem

from .graph_data import (
    MIPS_ATOM_FEATURE_DIM,
    _mips_atom_feature_rows,
    build_periodic_multimer_mol,
)


# New immutable identities.  Keep these in one module as well as the public
# contract so small cache/build tools can import them without constructing the
# full Dataset.
CANONICAL_CONFIG_SCHEMA = "mts-config-v3"
CANONICAL_FEATURE_SCHEMA = "mts-canonical-periodic-feature-v1"
CANONICAL_TOPOLOGY_SCHEMA = "mts-canonical-periodic-topology-lmdb-v1"
CANONICAL_LGA_SCHEMA_VERSION = 2
CANONICAL_CHECKPOINT_SCHEMA = "mts-model-v3"
CANONICAL_CACHE_BUNDLE_SCHEMA = "mips-trimer-scage-cache-bundle-v2"


def _as_mol(smiles_or_mol) -> Chem.Mol:
    if isinstance(smiles_or_mol, Chem.Mol):
        molecule = Chem.Mol(smiles_or_mol)
    else:
        molecule = Chem.MolFromSmiles(str(smiles_or_mol))
    if molecule is None:
        raise ValueError("invalid P-SMILES")
    return molecule


def _lift_neighbors(
    atom_count: int,
    internal_edges: Sequence[Tuple[int, int]],
    left: int,
    right: int,
) -> Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]]:
    """Build directed neighbours of the infinite lifted RU graph.

    ``right,q <-> left,q+1`` is the only inter-RU relation.  If the two
    boundaries are the same atom this naturally becomes a non-zero-shift
    canonical self relation, which is important for shared-boundary P-SMILES.
    """

    adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for atom in range(int(atom_count)):
        adjacency[(atom, 0)] = []
    # Neighbours are generated lazily for arbitrary q by translating the
    # canonical relation set; callers only use states reached within two hops.
    for begin, end in internal_edges:
        begin, end = int(begin), int(end)
        adjacency.setdefault((begin, 0), []).append((end, 0))
        adjacency.setdefault((end, 0), []).append((begin, 0))
    # Store the canonical edge templates under a private key.  This avoids
    # allocating a large finite copy graph while keeping the traversal simple.
    adjacency[(-1, -1)] = []
    for begin, end in internal_edges:
        adjacency[(-1, -1)].append((int(begin), int(end)))
    adjacency[(-2, -2)] = [(int(left), int(right))]
    return {key: tuple(value) for key, value in adjacency.items()}


def _neighbors(
    state: Tuple[int, int],
    internal_edges: Sequence[Tuple[int, int]],
    left: int,
    right: int,
) -> Iterable[Tuple[int, int]]:
    atom, shift = int(state[0]), int(state[1])
    for begin, end in internal_edges:
        begin, end = int(begin), int(end)
        if atom == begin:
            yield (end, shift)
        elif atom == end:
            yield (begin, shift)
    # right,q <-> left,q+1.  The inverse expression is written explicitly so
    # the shared-boundary case emits both +/- one-shift self relations.
    if atom == right:
        yield (left, shift + 1)
    if atom == left:
        yield (right, shift - 1)


def _lifted_bfs(
    target: Tuple[int, int],
    internal_edges: Sequence[Tuple[int, int]],
    left: int,
    right: int,
    max_hops: int,
):
    distances = {tuple(target): 0}
    predecessor: Dict[Tuple[int, int], Tuple[int, int] | None] = {
        tuple(target): None
    }
    queue = deque([tuple(target)])
    while queue:
        current = queue.popleft()
        distance = int(distances[current])
        if distance >= int(max_hops):
            continue
        for neighbour in _neighbors(current, internal_edges, left, right):
            if neighbour in distances:
                continue
            distances[neighbour] = distance + 1
            predecessor[neighbour] = current
            queue.append(neighbour)
    return distances, predecessor


def _direct_polymer_relation(source, target, left, right):
    """Whether one lifted relation is a direct symmetric polymer link."""

    source_atom, source_shift = map(int, source)
    target_atom, target_shift = map(int, target)
    # Normalise target to q=0.  The two directed templates are:
    # (right,-1)->(left,0), (left,+1)->(right,0).
    if target_shift != 0:
        return False
    return (
        (source_atom == int(right) and source_shift == -1 and target_atom == int(left))
        or (source_atom == int(left) and source_shift == 1 and target_atom == int(right))
    )


def _path_from_source_to_target(
    source: Tuple[int, int],
    target: Tuple[int, int],
    predecessor: Mapping[Tuple[int, int], Tuple[int, int] | None],
):
    # The predecessor table was generated by BFS from target.  Walk source to
    # target, then reverse to expose the usual source -> ... -> target path.
    path = [tuple(source)]
    current = tuple(source)
    while current != tuple(target):
        current = predecessor[current]
        if current is None:
            raise RuntimeError("lifted BFS predecessor chain is incomplete")
        path.append(tuple(current))
    path.reverse()
    return path


def _canonical_feature_rows(molecule, metadata):
    """Return topology-only central-RU MIPS137 rows and backbone indicator."""

    # The existing helper computes polymerized degree/H/hybridisation from an
    # open three-RU chain.  It is intentionally reused so migration and native
    # construction have exactly the same frozen feature semantics.
    trimer, trimer_meta = build_periodic_multimer_mol(
        molecule, num_repeat_units=3, close_periodic=False
    )
    units = trimer_meta.get("unit_atoms") or []
    if len(units) != 3:
        raise ValueError("canonical topology requires a three-RU feature source")
    central = torch.tensor(units[1], dtype=torch.long)
    rows = _mips_atom_feature_rows(trimer)[central]
    if rows.ndim != 2 or rows.size(1) != MIPS_ATOM_FEATURE_DIM:
        raise ValueError("MIPS137 feature row width mismatch")
    backbone = torch.zeros(rows.size(0), dtype=torch.long)
    backbone_ids = metadata.get("backbone_base") or []
    if backbone_ids:
        backbone[torch.as_tensor(backbone_ids, dtype=torch.long)] = 1
    return rows.float(), backbone


def _empty_canonical_placeholder() -> Data:
    """Shape-safe unavailable record used for malformed samples."""

    data = Data()
    data.x = torch.zeros((2, MIPS_ATOM_FEATURE_DIM), dtype=torch.float)
    data.mips_x = data.x.clone()
    data.mips_atom_feature_source = "topology_only_trimer_central_ru"
    data.mips_backbone_mask = torch.zeros(2, dtype=torch.long)
    data.atomic_numbers = torch.tensor([6, 6], dtype=torch.long)
    data.atomic_number = data.atomic_numbers
    data.z = data.atomic_numbers
    data.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    data.edge_attr = torch.zeros((2, 1), dtype=torch.float)
    data.lga_edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    data.canonical_lga_edge_index = data.lga_edge_index.clone()
    data.lga_spd = torch.zeros(2, dtype=torch.long)
    data.lga_path_index = torch.tensor([[0], [1]], dtype=torch.long)
    data.lga_path_shift = torch.zeros((2, 1), dtype=torch.long)
    data.lga_path_mask = torch.ones((2, 1), dtype=torch.bool)
    data.lga_source_image_shift = torch.zeros(2, dtype=torch.long)
    data.lga_relation_shift = data.lga_source_image_shift
    data.polymer_link_mask = torch.zeros(2, dtype=torch.bool)
    data.lga_polymer_link_mask = data.polymer_link_mask
    data.lga_star_edge_mask = data.polymer_link_mask
    data.canonical_atom_id = torch.arange(2, dtype=torch.long)
    data.canonical_ru_atom_index = data.canonical_atom_id.clone()
    data.mips_to_trimer_central_index = torch.full((2,), -1, dtype=torch.long)
    data.graph_available = False
    data.mips_condition_valid = False
    data.mips_alias_free = False
    data.topology_failure_code = "canonical_topology_unavailable"
    data.mts_canonical_periodic = True
    data.canonical_periodic_topology_schema = CANONICAL_TOPOLOGY_SCHEMA
    data.mips_local_lga_schema_version = CANONICAL_LGA_SCHEMA_VERSION
    data.feature_schema = CANONICAL_FEATURE_SCHEMA
    return data


def build_canonical_periodic_topology(smiles_or_mol, max_hops: int = 2) -> Data:
    """Build one canonical RU node state plus lifted max-hop relation rows.

    The returned ``Data`` has no ``ru_copy_index`` and does not materialise a
    finite k-copy graph.  ``lga_source_image_shift`` and path shifts are kept
    as integer structural diagnostics only.
    """

    molecule = _as_mol(smiles_or_mol)
    if int(max_hops) < 0:
        raise ValueError("max_hops must be non-negative")
    ru, metadata = build_periodic_multimer_mol(
        molecule, num_repeat_units=1, close_periodic=False
    )
    atom_count = int(ru.GetNumAtoms())
    left = int(metadata["left_boundary"])
    right = int(metadata["right_boundary"])
    internal_edges: List[Tuple[int, int]] = []
    edge_sources, edge_targets, edge_attrs = [], [], []
    for bond in ru.GetBonds():
        begin, end = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
        internal_edges.append((begin, end))
        edge_sources.extend((begin, end))
        edge_targets.extend((end, begin))
        code = int(round(float(bond.GetBondTypeAsDouble())))
        edge_attrs.extend((code, code))

    # Use the original two-attachment P-SMILES for the feature source.  ``ru``
    # has already had its dummies removed and therefore cannot be expanded by
    # ``build_periodic_multimer_mol`` a second time.
    mips_x, backbone = _canonical_feature_rows(molecule, metadata)
    atomic_numbers = torch.tensor(
        [int(atom.GetAtomicNum()) for atom in ru.GetAtoms()], dtype=torch.long
    )

    sources: List[int] = []
    targets: List[int] = []
    shifts: List[int] = []
    spd: List[int] = []
    polymer_mask: List[bool] = []
    paths: List[List[int]] = []
    path_shifts: List[List[int]] = []
    # Every target is the canonical state (i,0).  Sorting by distance then
    # source/shift makes serialization deterministic while retaining duplicate
    # canonical source/destination pairs at distinct shifts.
    for target_atom in range(atom_count):
        target_state = (int(target_atom), 0)
        distances, predecessor = _lifted_bfs(
            target_state, internal_edges, left, right, int(max_hops)
        )
        ordered = sorted(
            distances.items(), key=lambda item: (int(item[1]), int(item[0][0]), int(item[0][1]))
        )
        for source_state, distance in ordered:
            source_atom, source_shift = map(int, source_state)
            sources.append(source_atom)
            targets.append(int(target_atom))
            shifts.append(source_shift)
            spd.append(int(distance))
            polymer_mask.append(
                _direct_polymer_relation(
                    source_state, target_state, left, right
                )
            )
            lifted_path = _path_from_source_to_target(
                source_state, target_state, predecessor
            )
            paths.append([int(atom) for atom, _ in lifted_path])
            path_shifts.append([int(shift) for _, shift in lifted_path])

    edge_count = len(sources)
    width = int(max_hops) + 1
    path_index = torch.full((edge_count, width), -1, dtype=torch.long)
    path_shift = torch.zeros((edge_count, width), dtype=torch.long)
    path_mask = torch.zeros((edge_count, width), dtype=torch.bool)
    for row, (nodes, row_shifts) in enumerate(zip(paths, path_shifts)):
        length = min(width, len(nodes))
        if length:
            path_index[row, :length] = torch.tensor(nodes[:length], dtype=torch.long)
            path_shift[row, :length] = torch.tensor(row_shifts[:length], dtype=torch.long)
            path_mask[row, :length] = True

    data = Data()
    data.x = mips_x.clone()
    data.edge_index = torch.tensor(
        [edge_sources, edge_targets], dtype=torch.long
    ) if edge_sources else torch.empty((2, 0), dtype=torch.long)
    data.edge_attr = torch.tensor(edge_attrs, dtype=torch.float).reshape(-1, 1)
    data.mips_x = mips_x
    data.mips_atom_feature_source = "topology_only_trimer_central_ru"
    data.mips_backbone_mask = backbone
    data.atomic_numbers = atomic_numbers
    data.atomic_number = atomic_numbers
    data.canonical_atomic_numbers = atomic_numbers
    data.backbone_mask = backbone
    data.z = atomic_numbers
    relation_index = torch.tensor([sources, targets], dtype=torch.long)
    data.lga_edge_index = relation_index
    data.canonical_lga_edge_index = relation_index.clone()
    data.lga_spd = torch.tensor(spd, dtype=torch.long)
    data.lga_path_index = path_index
    data.lifted_single_path_index = path_index
    data.lga_path_shift = path_shift
    data.lga_path_shifts = path_shift
    data.lifted_single_path_shift = path_shift
    data.lga_path_mask = path_mask
    data.lifted_single_path_mask = path_mask
    # Bond-path prediction is disabled in the canonical production objective,
    # but keep a compact shape-compatible diagnostic field for shared tooling.
    data.lga_path_bond_hist = torch.zeros(
        (edge_count, int(max_hops), 6), dtype=torch.float
    )
    data.lga_source_image_shift = torch.tensor(shifts, dtype=torch.long)
    data.lga_relation_shift = data.lga_source_image_shift
    data.canonical_lga_source_image_shift = data.lga_source_image_shift
    data.source_image_shift = data.lga_source_image_shift
    data.spd = torch.tensor(spd, dtype=torch.long)
    data.polymer_link_mask = torch.tensor(polymer_mask, dtype=torch.bool)
    data.lga_polymer_link_mask = data.polymer_link_mask
    # Existing model code calls this field Star mask.  In canonical topology
    # it is exactly the one symmetric polymer-link mask, with no seam special
    # case or directional RU embedding.
    data.lga_star_edge_mask = data.polymer_link_mask
    data.canonical_atom_id = torch.arange(atom_count, dtype=torch.long)
    # This alias is retained only as an atom-identity table for Trimer mapping;
    # it is not a copy index and does not imply explicit O8 nodes.
    data.canonical_ru_atom_index = data.canonical_atom_id.clone()
    data.canonical_atom_count = atom_count
    data.canonical_lga_max_hops = int(max_hops)
    # Minimal repeat factor/boundary distance are retained strictly as
    # diagnostics and reference-construction metadata.  No repeated nodes are
    # materialised in the production topology.  For an RU boundary path of
    # length ``d`` the old finite-chain diagnostic is ``k*(d+1)-1``.
    base_distance = (
        0 if int(left) == int(right)
        else len(Chem.GetShortestPath(ru, left, right)) - 1
    )
    required_boundary = 2 * (int(max_hops) + 1) - 1
    repeat_units = max(
        1,
        (int(required_boundary + 1) // int(base_distance + 1)) + 1,
    )
    boundary_distance = int(repeat_units * (base_distance + 1) - 1)
    data.mips_repeat_factor = int(repeat_units)
    data.mips_repeat_units = int(repeat_units)
    data.mips_boundary_distance = boundary_distance
    data.mips_distance_threshold = int(max_hops) + 1
    data.mips_condition_valid = True
    data.graph_available = True
    data.mips_alias_free = True
    data.mts_canonical_periodic = True
    data.canonical_periodic_topology_schema = CANONICAL_TOPOLOGY_SCHEMA
    data.mips_local_lga_schema_version = CANONICAL_LGA_SCHEMA_VERSION
    data.feature_schema = CANONICAL_FEATURE_SCHEMA
    data.canonical_topology_builder_version = 1
    data.ru_left_boundary = left
    data.ru_right_boundary = right
    data.ru_backbone = torch.as_tensor(
        metadata.get("backbone_base") or [], dtype=torch.long
    )
    data.ru_atomic_number = atomic_numbers.clone()
    data.ru_edge_index = data.edge_index.clone()
    data.ru_bond_type = torch.tensor(
        [int(round(float(bond.GetBondTypeAsDouble()))) for bond in ru.GetBonds()
         for _ in (0, 1)], dtype=torch.long
    )
    data.attachment_bond_mismatch = bool(metadata.get("attachment_bond_mismatch", False))
    data.connection_bond_policy = str(metadata.get("connection_bond_policy", ""))
    data.shared_attachment_boundary = bool(metadata.get("shared_boundary", False))
    data.repeat_metadata = dict(metadata)
    if not isinstance(smiles_or_mol, Chem.Mol):
        data.smiles = str(smiles_or_mol)
    return data


def migrate_explicit_topology_to_canonical(
    old_topology,
    ru_base=None,
    *,
    max_hops: int = 2,
    verify_features: bool = True,
) -> Data:
    """Migrate one old explicit record without rebuilding a cache.

    Exactly one old row per canonical atom is copied after verifying all old
    explicit copies agree.  Lifted relations are always rebuilt from the RU
    base, so old seam rows cannot leak into the canonical representation.
    """

    old = old_topology
    if not hasattr(old, "canonical_ru_atom_index"):
        raise ValueError("old topology has no canonical atom identity table")
    canonical = torch.as_tensor(old.canonical_ru_atom_index).long().reshape(-1)
    if canonical.numel() == 0 or bool((canonical < 0).any()):
        raise ValueError("old topology canonical mapping is invalid")
    count = int(canonical.max().item()) + 1
    if not hasattr(old, "mips_x"):
        raise ValueError("old topology has no MIPS137 rows")
    old_rows = torch.as_tensor(old.mips_x).float()
    if old_rows.ndim != 2 or old_rows.size(1) != MIPS_ATOM_FEATURE_DIM:
        raise ValueError("old topology MIPS137 width mismatch")
    copied = torch.empty((count, old_rows.size(1)), dtype=old_rows.dtype)
    for atom_id in range(count):
        rows = old_rows[canonical == atom_id]
        if rows.numel() == 0:
            raise ValueError("old topology mapping omits a canonical atom")
        if verify_features and not bool(torch.equal(rows, rows[:1].expand_as(rows))):
            raise ValueError("old topology copies disagree for a canonical atom")
        copied[atom_id] = rows[0]
    if ru_base is None:
        smiles = getattr(old, "smiles", None)
        if smiles is None:
            raise ValueError("migration requires ru_base or source smiles")
        migrated = build_canonical_periodic_topology(smiles, max_hops=max_hops)
    else:
        if hasattr(ru_base, "ru_mol_binary"):
            molecule = Chem.Mol(bytes(ru_base.ru_mol_binary))
        else:
            molecule = _as_mol(ru_base)
        migrated = build_canonical_periodic_topology(molecule, max_hops=max_hops)
    if int(migrated.mips_x.size(0)) != count:
        raise ValueError("RU base and old topology atom counts differ")
    migrated.mips_x = copied.clone()
    migrated.x = copied.clone()
    if hasattr(old, "mips_backbone_mask"):
        old_backbone = torch.as_tensor(old.mips_backbone_mask).long().reshape(-1)
        backbone = torch.empty(count, dtype=torch.long)
        for atom_id in range(count):
            rows = old_backbone[canonical == atom_id]
            if rows.numel() == 0:
                raise ValueError("old topology backbone mapping is incomplete")
            if verify_features and not bool(torch.equal(rows, rows[:1].expand_as(rows))):
                raise ValueError("old topology copies disagree in backbone mask")
            backbone[atom_id] = rows[0]
        migrated.mips_backbone_mask = backbone
    migrated.migration_source_schema = str(
        getattr(old, "feature_schema", "legacy-explicit-topology")
    )
    migrated.migration_verified_identical_rows = bool(verify_features)
    return migrated


def build_corrected_explicit_k_ru_reference(canonical_topology, repeat_factor: int):
    """Lift canonical rows to a corrected translation-symmetric explicit graph.

    For each canonical relation ``(source, shift) -> target`` and target copy
    ``c``, the source copy is ``(c + shift) mod k``.  There is no seam branch;
    duplicate canonical source/destination rows remain distinct relations.
    """

    k = int(repeat_factor)
    if k < 1:
        raise ValueError("repeat_factor must be >= 1")
    relation = torch.as_tensor(canonical_topology.lga_edge_index).long()
    shifts = torch.as_tensor(
        getattr(canonical_topology, "lga_source_image_shift", torch.zeros(relation.size(1))),
    ).long().reshape(-1)
    if shifts.numel() != relation.size(1):
        raise ValueError("canonical relation shift length mismatch")
    n = int(canonical_topology.mips_x.size(0))
    source_rows, target_rows, shift_rows = [], [], []
    spd_rows, polymer_rows = [], []
    path_rows, path_shift_rows = [], []
    canonical_polymer = torch.as_tensor(
        getattr(canonical_topology, "polymer_link_mask", torch.zeros(relation.size(1))),
    ).bool().reshape(-1)
    canonical_spd = torch.as_tensor(canonical_topology.lga_spd).long().reshape(-1)
    canonical_path = torch.as_tensor(canonical_topology.lga_path_index).long()
    canonical_path_shift = torch.as_tensor(
        getattr(
            canonical_topology, "lga_path_shift",
            torch.zeros_like(canonical_path),
        )
    ).long()
    for copy in range(k):
        for row in range(relation.size(1)):
            source_atom, target_atom = map(int, relation[:, row])
            source_copy = (copy + int(shifts[row])) % k
            source_rows.append(source_atom + source_copy * n)
            target_rows.append(target_atom + copy * n)
            shift_rows.append(int(shifts[row]))
            spd_rows.append(int(canonical_spd[row]))
            polymer_rows.append(bool(canonical_polymer[row]))
            path = canonical_path[row].clone()
            path_shift = canonical_path_shift[row].clone()
            valid = path >= 0
            lifted_path = path.clone()
            lifted_path[valid] += (
                (copy + path_shift[valid]) % k
            ) * n
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
    output.lga_edge_index = torch.tensor([source_rows, target_rows], dtype=torch.long)
    output.canonical_lga_edge_index = output.lga_edge_index.clone()
    output.lga_spd = torch.tensor(spd_rows, dtype=torch.long)
    output.lga_path_index = torch.stack(path_rows, dim=0)
    output.lga_path_shift = torch.stack(path_shift_rows, dim=0)
    output.lga_path_shifts = output.lga_path_shift
    output.lga_path_mask = output.lga_path_index >= 0
    output.lga_path_bond_hist = torch.zeros(
        (len(path_rows), max(0, output.lga_path_index.size(1) - 1), 6),
        dtype=torch.float,
    )
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
        if hasattr(canonical_topology, name):
            value = getattr(canonical_topology, name)
            if name == "mips_md":
                # A direct, uncollated record stores MD200 as [200], whereas
                # the graph encoder consumes one row per graph.  Preserve an
                # already-batched [B,200] sidecar and lift a scalar record to
                # [1,200] so strict canonical/explicit prediction parity can
                # include the MD residual as well.
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
    # If a canonical record already carries a frozen Trimer, reuse that same
    # geometry for the explicit equivalence reference.  Only the O8-to-central
    # mapping is lifted; the Trimer itself remains the shared three-RU object.
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
            output.mips_to_trimer_central_index = torch.as_tensor(
                mapping
            ).long().repeat(k)
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
                value = getattr(output, name)
                setattr(output, name, torch.as_tensor(value).reshape(-1))
    output.repeat_factor = k
    output.canonical_node_count = n
    output.translation_symmetric_reference = True
    output.canonical_periodic_topology_schema = CANONICAL_TOPOLOGY_SCHEMA
    output.mts_canonical_periodic = False
    return output


def load_test_only_equivalence_checkpoint(model, checkpoint):
    """Load frozen legacy/current weights for strict equivalence tests only.

    Production training validates ``mts-model-v3`` in the CLI.  This helper
    intentionally accepts the previous metadata identity because the model
    parameter layout is unchanged and a fixed old checkpoint is useful for
    proving translation symmetry without retraining.
    """

    payload = checkpoint
    if isinstance(checkpoint, (str, bytes)) or hasattr(checkpoint, "__fspath__"):
        import torch as _torch
        payload = _torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("meta"), Mapping):
        raise ValueError("equivalence checkpoint must contain metadata")
    schema = payload["meta"].get("schema")
    if schema not in {"mts-model-v2", CANONICAL_CHECKPOINT_SCHEMA}:
        raise ValueError("unsupported checkpoint for test-only equivalence")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("equivalence checkpoint has no state_dict")
    model.load_state_dict(state, strict=True)
    return model


# Friendly aliases used by focused equivalence tests and migration scripts.
lift_canonical_topology_to_explicit = build_corrected_explicit_k_ru_reference
build_explicit_translation_symmetric_reference = build_corrected_explicit_k_ru_reference
build_corrected_translation_symmetric_explicit_reference = (
    build_corrected_explicit_k_ru_reference
)
build_lifted_periodic_relations = build_canonical_periodic_topology
build_mts_canonical_periodic_topology = build_canonical_periodic_topology
build_canonical_periodic_lga = build_canonical_periodic_topology
attach_canonical_periodic_lga = build_canonical_periodic_topology


__all__ = [
    "CANONICAL_CONFIG_SCHEMA", "CANONICAL_FEATURE_SCHEMA",
    "CANONICAL_TOPOLOGY_SCHEMA", "CANONICAL_LGA_SCHEMA_VERSION",
    "CANONICAL_CHECKPOINT_SCHEMA", "CANONICAL_CACHE_BUNDLE_SCHEMA",
    "build_canonical_periodic_topology", "build_mts_canonical_periodic_topology",
    "build_canonical_periodic_lga", "migrate_explicit_topology_to_canonical",
    "attach_canonical_periodic_lga",
    "build_corrected_explicit_k_ru_reference", "lift_canonical_topology_to_explicit",
    "build_explicit_translation_symmetric_reference",
    "build_corrected_translation_symmetric_explicit_reference",
    "build_lifted_periodic_relations",
    "load_test_only_equivalence_checkpoint",
]
