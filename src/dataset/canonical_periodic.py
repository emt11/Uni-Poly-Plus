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
    _graph_backbone_annotations,
    _mips_atom_feature_rows,
    build_periodic_multimer_mol,
)
from .mips_trimer_contract import (
    CACHE_BUNDLE_SCHEMA as CANONICAL_CACHE_BUNDLE_SCHEMA,
    CANONICAL_LGA_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA as CANONICAL_CHECKPOINT_SCHEMA,
    CONFIG_SCHEMA as CANONICAL_CONFIG_SCHEMA,
    FEATURE_SCHEMA as CANONICAL_FEATURE_SCHEMA,
    TOPOLOGY_LMDB_SCHEMA as CANONICAL_TOPOLOGY_SCHEMA,
)


def _as_mol(smiles_or_mol) -> Chem.Mol:
    if isinstance(smiles_or_mol, Chem.Mol):
        molecule = Chem.Mol(smiles_or_mol)
    else:
        molecule = Chem.MolFromSmiles(str(smiles_or_mol))
    if molecule is None:
        raise ValueError("invalid P-SMILES")
    return molecule


def _atom_signature(atom):
    """Return the identity fields used by migration graph matching.

    RDKit's canonical SMILES atom order is not an identity contract.  The
    migration path therefore compares chemical graph fields explicitly and
    only uses the atom index as a deterministic tie-breaker after all
    candidates have been validated.
    """

    return (
        int(atom.GetAtomicNum()),
        int(atom.GetFormalCharge()),
        bool(atom.GetIsAromatic()),
        int(atom.GetIsotope()),
        int(atom.GetNumRadicalElectrons()),
        int(atom.GetChiralTag()),
    )


def _bond_signature(bond):
    return (
        bool(bond.GetIsAromatic()),
        str(bond.GetBondType()),
        bool(bond.GetIsConjugated()),
        bool(bond.IsInRing()),
        int(bond.GetStereo()),
    )


def _validate_full_atom_mapping(source, target, match):
    if source.GetNumAtoms() != target.GetNumAtoms():
        return False
    if len(match) != source.GetNumAtoms() or len(set(match)) != len(match):
        return False
    for source_idx, target_idx in enumerate(match):
        source_atom = source.GetAtomWithIdx(int(source_idx))
        target_atom = target.GetAtomWithIdx(int(target_idx))
        if _atom_signature(source_atom) != _atom_signature(target_atom):
            return False
        if source_atom.GetDegree() != target_atom.GetDegree():
            return False
    for source_bond in source.GetBonds():
        begin = int(match[source_bond.GetBeginAtomIdx()])
        end = int(match[source_bond.GetEndAtomIdx()])
        target_bond = target.GetBondBetweenAtoms(begin, end)
        if target_bond is None or _bond_signature(source_bond) != _bond_signature(target_bond):
            return False
    # Attachment semantics are part of the identity table.  A legal
    # automorphism may swap the two dummy sites, but each dummy must still map
    # to a dummy with exactly one corresponding boundary neighbour.
    source_attachments = []
    target_attachments = []
    for atom in source.GetAtoms():
        if atom.GetAtomicNum() == 0:
            neighbors = tuple(sorted(int(n.GetIdx()) for n in atom.GetNeighbors()))
            if len(neighbors) != 1:
                return False
            source_attachments.append((int(atom.GetIdx()), neighbors[0]))
    for atom in target.GetAtoms():
        if atom.GetAtomicNum() == 0:
            neighbors = tuple(sorted(int(n.GetIdx()) for n in atom.GetNeighbors()))
            if len(neighbors) != 1:
                return False
            target_attachments.append((int(atom.GetIdx()), neighbors[0]))
    if len(source_attachments) != 2 or len(target_attachments) != 2:
        return False
    mapped = sorted(
        (int(match[dummy]), int(match[neighbor]))
        for dummy, neighbor in source_attachments
    )
    if mapped != sorted(target_attachments):
        return False
    return True


def find_atom_graph_mapping(source, target):
    """Return a deterministic ``source_atom_id -> target_atom_id`` mapping.

    The function intentionally does not rely on canonical-SMILES order.  It
    enumerates RDKit graph-isomorphism candidates, validates atom/bond and
    attachment fields, and chooses the lexicographically smallest valid
    permutation.  Ambiguous automorphisms are therefore stable while a
    genuinely incompatible graph fails loudly instead of guessing.
    """

    source = _as_mol(source)
    target = _as_mol(target)
    if source.GetNumAtoms() != target.GetNumAtoms():
        raise ValueError("atom mapping failed: atom counts differ")
    # ``target.GetSubstructMatches(source)`` returns one target index for each
    # query atom in ``source`` (the direction needed by the callers below).
    # RDKit's ``uniquify=False`` can expose an enormous automorphism group for
    # long fluorinated chains and symmetric aromatic substituents.  Asking for
    # every match in those cases is both unnecessary (all such atoms have the
    # same graph identity) and can make a migration appear hung.  Keep the
    # historical exhaustive/lexicographic behaviour for genuinely small RUs;
    # use a deterministic bounded prefix for larger graphs.  RDKit enumerates
    # these matches deterministically for a fixed source/target atom order,
    # and every returned candidate still passes the complete identity check
    # below.  A failure to find one is therefore reported rather than guessed.
    atom_count = int(source.GetNumAtoms())
    max_matches = 100000 if atom_count <= 24 else 512
    candidates = target.GetSubstructMatches(
        source, uniquify=False, useChirality=True, maxMatches=max_matches
    )
    valid = [
        tuple(int(value) for value in match)
        for match in candidates
        if _validate_full_atom_mapping(source, target, match)
    ]
    if not valid:
        raise ValueError("atom mapping failed: no validated graph isomorphism")
    return torch.tensor(min(valid), dtype=torch.long)


def _base_atom_order(molecule):
    return [
        int(atom.GetIdx())
        for atom in molecule.GetAtoms()
        if int(atom.GetAtomicNum()) != 0
    ]


def find_base_atom_mapping(source, target):
    """Map non-dummy RU atom ranks between two equivalent P-SMILES graphs."""

    source = _as_mol(source)
    target = _as_mol(target)
    full = find_atom_graph_mapping(source, target)
    source_order = _base_atom_order(source)
    target_order = _base_atom_order(target)
    target_rank = {atom_idx: rank for rank, atom_idx in enumerate(target_order)}
    mapped = [target_rank[int(full[source_idx])] for source_idx in source_order]
    if len(mapped) != len(target_order) or sorted(mapped) != list(range(len(mapped))):
        raise ValueError("atom mapping failed: non-dummy RU mapping is incomplete")
    return torch.tensor(mapped, dtype=torch.long)


def remap_rows_by_base_atom(rows, source_to_target, *, target_count=None):
    """Reorder per-RU rows from source base order into target base order."""

    rows = torch.as_tensor(rows)
    mapping = torch.as_tensor(source_to_target, dtype=torch.long).reshape(-1)
    target_count = int(target_count if target_count is not None else mapping.numel())
    if rows.ndim < 1 or rows.size(0) != mapping.numel():
        raise ValueError("row/mapping lengths differ")
    output = rows.new_empty((target_count,) + tuple(rows.shape[1:]))
    seen = torch.zeros(target_count, dtype=torch.bool)
    for source_idx, target_idx in enumerate(mapping.tolist()):
        if target_idx < 0 or target_idx >= target_count or bool(seen[target_idx]):
            raise ValueError("row/mapping is not a permutation")
        output[target_idx] = rows[source_idx]
        seen[target_idx] = True
    if not bool(seen.all()):
        raise ValueError("row/mapping omits a target atom")
    return output


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
    # The predecessor table was generated by BFS from target.  Walking from
    # source through predecessors already yields the usual source -> ... ->
    # target path; do not reverse it (the source/target direction is part of
    # the single-path-node bias contract).
    path = [tuple(source)]
    current = tuple(source)
    while current != tuple(target):
        current = predecessor[current]
        if current is None:
            raise RuntimeError("lifted BFS predecessor chain is incomplete")
        path.append(tuple(current))
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
    # Match the frozen MIPS backbone-role semantics, including ring atoms
    # touched by a boundary path.  Merely copying ``backbone_base`` would omit
    # those ring members and make old explicit backbone rows fail migration
    # validation even though their atom features are otherwise identical.
    base_ru, base_metadata = build_periodic_multimer_mol(
        molecule, num_repeat_units=1, close_periodic=False
    )
    base_backbone, _, _, _ = _graph_backbone_annotations(
        base_ru,
        original_neighbors=[
            int(base_metadata["left_boundary"]),
            int(base_metadata["right_boundary"]),
        ],
        ordered_backbone_path=base_metadata.get("backbone_base") or [],
    )
    backbone = torch.zeros(rows.size(0), dtype=torch.long)
    if base_backbone:
        backbone[torch.as_tensor(sorted(base_backbone), dtype=torch.long)] = 1
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
    data.canonical_to_trimer_base_atom_id = data.canonical_atom_id.clone()
    data.canonical_to_trimer_base_atom_index = (
        data.canonical_to_trimer_base_atom_id
    )
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
    # The native topology is built from the normalized P-SMILES molecule, so
    # its canonical and Trimer base atom ranks coincide.  Keeping this table
    # explicit lets migration and non-canonical input paths replace it with a
    # validated graph-isomorphism permutation rather than assuming RDKit order.
    data.canonical_to_trimer_base_atom_id = data.canonical_atom_id.clone()
    data.canonical_to_trimer_base_atom_index = (
        data.canonical_to_trimer_base_atom_id
    )
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
    source_count = int(canonical.max().item()) + 1
    if not hasattr(old, "mips_x"):
        raise ValueError("old topology has no MIPS137 rows")
    old_rows = torch.as_tensor(old.mips_x).float()
    if old_rows.ndim != 2 or old_rows.size(1) != MIPS_ATOM_FEATURE_DIM:
        raise ValueError("old topology MIPS137 width mismatch")
    if ru_base is not None and hasattr(ru_base, "ru_mol_binary"):
        source_molecule = Chem.Mol(bytes(ru_base.ru_mol_binary))
    else:
        source_smiles = getattr(old, "smiles", None)
        if source_smiles is None:
            raise ValueError("migration requires ru_base or source smiles")
        source_molecule = _as_mol(source_smiles)
    normalized_text = getattr(ru_base, "normalized_polymer_smiles", None)
    if normalized_text is None:
        normalized_text = Chem.MolToSmiles(source_molecule, canonical=True)
    target_molecule = _as_mol(normalized_text)
    source_to_target = find_base_atom_mapping(source_molecule, target_molecule)
    target_count = int(target_molecule.GetNumAtoms()) - sum(
        int(atom.GetAtomicNum() == 0) for atom in target_molecule.GetAtoms()
    )
    if source_count != int(source_to_target.numel()) or target_count != source_count:
        raise ValueError("old topology and normalized RU atom counts differ")
    old_atomic = getattr(old, "atomic_numbers", getattr(old, "z", None))
    if old_atomic is not None:
        old_atomic = torch.as_tensor(old_atomic).long().reshape(-1)
        if old_atomic.numel() != canonical.numel():
            raise ValueError("old topology atomic-number mapping is incomplete")
        source_order = _base_atom_order(source_molecule)
        expected_atomic = torch.tensor(
            [int(source_molecule.GetAtomWithIdx(idx).GetAtomicNum()) for idx in source_order],
            dtype=torch.long,
        )
        for source_atom in range(source_count):
            observed = torch.unique(old_atomic[canonical == source_atom])
            if observed.numel() != 1 or int(observed.item()) != int(expected_atomic[source_atom]):
                raise ValueError("old topology atomic-number validation failed")
    copied = torch.empty((target_count, old_rows.size(1)), dtype=old_rows.dtype)
    for target_atom in range(target_count):
        source_atom = int(torch.nonzero(
            source_to_target == target_atom, as_tuple=False
        ).flatten()[0])
        rows = old_rows[canonical == source_atom]
        if rows.numel() == 0:
            raise ValueError("old topology mapping omits a canonical atom")
        if verify_features and not bool(torch.equal(rows, rows[:1].expand_as(rows))):
            raise ValueError("old topology copies disagree for a canonical atom")
        copied[target_atom] = rows[0]
    migrated = build_canonical_periodic_topology(
        target_molecule, max_hops=max_hops
    )
    if int(migrated.mips_x.size(0)) != target_count:
        raise ValueError("RU base and old topology atom counts differ")
    if verify_features and not torch.equal(copied, migrated.mips_x.to(copied.dtype)):
        raise ValueError(
            "old topology MIPS137 rows disagree with native canonical features"
        )
    migrated.mips_x = copied.clone()
    migrated.x = copied.clone()
    if hasattr(old, "mips_backbone_mask"):
        old_backbone = torch.as_tensor(old.mips_backbone_mask).long().reshape(-1)
        backbone = torch.empty(target_count, dtype=torch.long)
        for target_atom in range(target_count):
            source_atom = int(torch.nonzero(
                source_to_target == target_atom, as_tuple=False
            ).flatten()[0])
            rows = old_backbone[canonical == source_atom]
            if rows.numel() == 0:
                raise ValueError("old topology backbone mapping is incomplete")
            if verify_features and not bool(torch.equal(rows, rows[:1].expand_as(rows))):
                raise ValueError("old topology copies disagree in backbone mask")
            backbone[target_atom] = rows[0]
        if verify_features and not torch.equal(
            backbone, torch.as_tensor(migrated.mips_backbone_mask).long()
        ):
            raise ValueError(
                "old topology backbone mask disagrees with native canonical features"
            )
        migrated.mips_backbone_mask = backbone
    migrated.source_to_normalized_canonical_atom_id = source_to_target
    migrated.normalized_canonical_smiles = str(normalized_text)
    migrated.canonical_to_trimer_base_atom_id = torch.arange(
        target_count, dtype=torch.long
    )
    migrated.canonical_to_trimer_base_atom_index = (
        migrated.canonical_to_trimer_base_atom_id
    )
    migrated.migration_atom_mapping_verified = True
    migrated.migration_source_schema = str(
        getattr(old, "feature_schema", "legacy-explicit-topology")
    )
    migrated.migration_verified_identical_rows = bool(verify_features)
    return migrated


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
    "find_atom_graph_mapping", "find_base_atom_mapping",
    "remap_rows_by_base_atom",
    "attach_canonical_periodic_lga",
    "build_lifted_periodic_relations",
]
