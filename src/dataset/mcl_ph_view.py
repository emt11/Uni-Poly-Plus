"""MCL-PH multiscale distance-expert view over the frozen open Trimer.

Plan ``MCL-PH-20260921-01/r1``.  This module is the single read-only adapter that
turns one frozen ``(topology, trimer, smiles)`` record into the declared inputs of
the new route:

* section 3.1/3.3 -- heavy-atom Trimer graph, canonical centre mapping and the
  three nested ``2/3/4 A`` expert relations;
* section 5.1     -- the ``[31, 5]`` topology trajectory (Randic, revised
  normalized Wiener, efficiency, Betti-0, Betti-1) on the same noisy view;
* section 6       -- cross-RU synchronous chemical masking;
* section 7       -- the local-geometry and per-scale non-bond targets.

Nothing here generates coordinates, writes a cache or mutates a frozen record.
Every random draw comes from a labelled sub-stream of
``sample_generator(seed, key, position)`` so that the pre-existing motif/coordinate
streams of the dual route keep their positions.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch

from .glt_dual import build_dual_sample
from .glt_dual_pretrain import motif_mask, sample_generator
from .glt_dual_static import materialize_dual_geometry

# ---------------------------------------------------------------------------
# Section 3/5.1 fixed constants (all frozen by the contract, never tuned here)
# ---------------------------------------------------------------------------

EXPERT_CUTOFFS = (2.0, 3.0, 4.0)
ROUTER_RADII = tuple(1.5 + 0.1 * index for index in range(31))
ROUTER_RADII_ARRAY = np.asarray(ROUTER_RADII, dtype=np.float64)
VR_MAX_EDGE = 4.5
DESCRIPTOR_COLUMNS = 5

RBF_BINS = 64
RBF_MAX = 4.0
RBF_WIDTH = RBF_MAX / (RBF_BINS - 1)
RBF_CENTERS = np.asarray([RBF_MAX * index / (RBF_BINS - 1) for index in range(RBF_BINS)],
                         dtype=np.float64)

# Real chemical bond codes (``trimer_bond_type``) occupy 0..4, then the two
# declared synthetic categories.  "A spatial edge is not a chemical bond".
BOND_CATEGORY_COUNT = 7
NONBONDED_CATEGORY = 5
UNKNOWN_CATEGORY = 6

# Node vocabulary (section 3.2).  MASK and UNKNOWN are strictly distinct.
ELEMENT_UNKNOWN = 118
ELEMENT_MASK = 119
ELEMENT_VOCABULARY = 120
CHARGE_OTHER = 7
CHARGE_MASK = 8
CHARGE_VOCABULARY = 9
AROMATIC_MASK = 2
AROMATIC_VOCABULARY = 3

# Section 7.3 sampling budget.
NONBOND_BINS = ((0.0, 2.0), (2.0, 3.0), (3.0, 4.0))
NONBOND_MAX_PAIRS = 32

# Section 7.3 normalization: the standard deviation floor.
NORMALIZATION_STD_FLOOR = 1e-6

# Fixed sub-stream labels.  Only the pre-existing motif/noise draws stay on the
# unlabelled stream; every new draw gets its own label so that adding the new
# route cannot shift a training stream that an earlier arm already consumed.
PAIR_SUBSTREAM = 'mcl-ph-pairs'
REFERENCE_SUBSTREAM = 'mcl-ph-reference'


# ---------------------------------------------------------------------------
# Section 5.1 -- persistent homology and the five descriptors
# ---------------------------------------------------------------------------

def _as_point_cloud(points):
    cloud = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError('PH point cloud must be [n,3]')
    if not np.isfinite(cloud).all():
        raise ValueError('PH point cloud contains a non-finite coordinate')
    return cloud


def distance_matrix(points):
    """Euclidean distances in float64; the contract requires float64 here."""
    cloud = _as_point_cloud(points)
    difference = cloud[:, None, :] - cloud[None, :, :]
    return np.sqrt(np.einsum('ijk,ijk->ij', difference, difference))


def persistence_pairs(points, *, max_edge=VR_MAX_EDGE):
    """Filtered Vietoris-Rips persistence pairs over F2, up to H1.

    The simplex tree is built to dimension two so that a filling triangle can
    kill an H1 class; ``E - V + C`` on the radius graph is explicitly not a
    substitute.  Intervals that never die keep ``death = inf`` (right
    censoring); zero-length intervals are dropped.

    ``persistence_dim_max`` is required.  gudhi ignores the homology of the
    maximal dimension of the complex by default, which silently deletes every
    right-censored H1 class whenever the sample has no filling triangle at all.
    """
    import gudhi

    cloud = _as_point_cloud(points)
    pairs = {0: [], 1: []}
    if int(cloud.shape[0]) == 0:
        return pairs
    rips = gudhi.RipsComplex(points=cloud.tolist(), max_edge_length=float(max_edge))
    tree = rips.create_simplex_tree(max_dimension=2)
    for dimension, (birth, death) in tree.persistence(persistence_dim_max=True):
        if dimension not in pairs:
            continue
        birth, death = float(birth), float(death)
        if not birth < death:
            continue
        pairs[dimension].append((birth, death))
    return pairs


def _active_counts(pairs, radii):
    """Number of intervals alive at each radius (``birth <= r < death``)."""
    radii = np.asarray(radii, dtype=np.float64)
    curve = np.zeros((2, radii.size), dtype=np.float64)
    for dimension in (0, 1):
        for birth, death in pairs[dimension]:
            alive = radii >= birth
            if np.isfinite(death):
                alive = alive & (radii < death)
            curve[dimension] += alive
    return curve


def _radius_graph(distance, radius):
    """Boolean adjacency of the declared radius graph (no self relation)."""
    adjacency = distance <= float(radius)
    np.fill_diagonal(adjacency, False)
    return adjacency


def _components(adjacency):
    """Connected components of a symmetric boolean radius graph, as index groups."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    _, labels = connected_components(csr_matrix(adjacency), directed=False)
    order = np.argsort(labels, kind='stable')
    boundaries = np.flatnonzero(np.diff(labels[order])) + 1
    return [group for group in np.split(order, boundaries) if group.size]


def _hop_distances(adjacency):
    """All-pairs hop counts; unreachable pairs stay ``inf``.

    The declared object is the hop-count metric of the radius graph, so this is
    the C shortest-path implementation of that same metric; it is not an
    approximation of it.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    return shortest_path(csr_matrix(adjacency), directed=False, unweighted=True)


def _randic(adjacency):
    """Section 5.1 column 0: ``2/n * sum_edges 1/sqrt(deg(u) deg(v))``.

    The ``2/n`` factor is part of the declared definition; without it the column
    is an unnormalised degree sum, which contradicts the declared column range
    and the descriptor scaling the router relies on.  A graph with no edge is 0.
    """
    count = int(adjacency.shape[0])
    if count <= 0:
        return 0.0
    degree = adjacency.sum(axis=1).astype(np.float64)
    left, right = np.nonzero(np.triu(adjacency, 1))
    if left.size == 0:
        return 0.0
    return float(2.0 * np.sum(1.0 / np.sqrt(degree[left] * degree[right])) / float(count))


def _normalized_wiener(adjacency, distance, members):
    """Revised component-wise Wiener index, weighted by ``m(m-1)/2``."""
    numerator = 0.0
    denominator = 0.0
    for component in members:
        size = int(component.size)
        weight = size * (size - 1) / 2.0
        denominator += weight
        if size < 3:
            continue
        block = distance[np.ix_(component, component)]
        within = np.sum(np.triu(block, 1))
        span = (size ** 3 - size) / 6.0 - weight
        numerator += weight * ((within - weight) / span)
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _efficiency(distance):
    count = int(distance.shape[0])
    if count <= 1:
        return 0.0
    usable = np.isfinite(distance) & (distance > 0)
    inverted = np.zeros_like(distance)
    inverted[usable] = 1.0 / distance[usable]
    return float(inverted.sum() / (count * (count - 1)))


def five_descriptors(points, *, radii=ROUTER_RADII_ARRAY, max_edge=VR_MAX_EDGE):
    """``[31, 5]`` topology trajectory in the fixed column order of section 5.1.

    Column 0 Randic, 1 revised normalized Wiener, 2 efficiency, 3 Betti-0,
    4 Betti-1.  An empty or single-point cloud is a geometry failure upstream;
    this function still returns the honest all-zero row for ``n <= 1``.
    """
    cloud = _as_point_cloud(points)
    count = int(cloud.shape[0])
    radii = np.asarray(radii, dtype=np.float64)
    trajectory = np.zeros((radii.size, DESCRIPTOR_COLUMNS), dtype=np.float64)
    if count == 0:
        raise ValueError('an empty point cloud is not a valid topology input')
    distance = distance_matrix(cloud)
    pairs = persistence_pairs(cloud, max_edge=max_edge)
    betti = _active_counts(pairs, radii)
    betti_one_normalizer = float(max(1, len(pairs[1])))
    for index, radius in enumerate(radii.tolist()):
        adjacency = _radius_graph(distance, radius)
        hop = _hop_distances(adjacency)
        trajectory[index, 0] = _randic(adjacency)
        trajectory[index, 1] = _normalized_wiener(adjacency, hop, _components(adjacency))
        trajectory[index, 2] = _efficiency(hop)
        trajectory[index, 3] = betti[0, index] / float(count)
        trajectory[index, 4] = betti[1, index] / betti_one_normalizer
    if not np.isfinite(trajectory).all():
        raise ValueError('non-finite topology trajectory')
    return trajectory


def descriptor_channel_variance(trajectories):
    """Per-column variance of a stack of trajectories, for the P0 audit."""
    stack = np.asarray(trajectories, dtype=np.float64)
    if stack.ndim != 3 or stack.shape[1:] != (len(ROUTER_RADII), DESCRIPTOR_COLUMNS):
        raise ValueError('descriptor stack must be [N,31,5]')
    if int(stack.shape[0]) == 0:
        return np.zeros((len(ROUTER_RADII), DESCRIPTOR_COLUMNS), dtype=np.float64)
    return stack.var(axis=0)


# ---------------------------------------------------------------------------
# Section 3 -- heavy-atom view, centre mapping and physical bonds
# ---------------------------------------------------------------------------

class TrimerView:
    """The heavy-atom 3D view of one frozen Trimer record."""

    __slots__ = ('indices', 'of_all', 'z', 'charge', 'aromatic', 'base_id',
                 'offset', 'positions', 'echo', 'bond_ends', 'bond_type')

    def __init__(self, indices, of_all, z, charge, aromatic, base_id, offset,
                 positions, echo, bond_ends, bond_type):
        self.indices = indices
        self.of_all = of_all
        self.z = z
        self.charge = charge
        self.aromatic = aromatic
        self.base_id = base_id
        self.offset = offset
        self.positions = positions
        self.echo = echo
        self.bond_ends = bond_ends
        self.bond_type = bond_type

    @property
    def count(self):
        return int(self.indices.numel())


def build_trimer_view(trimer):
    """Heavy-atom enumeration plus the unique undirected covalent bonds.

    ``A`` is the number of *heavy* atoms of the frozen Trimer: no hydrogen is
    added and no capping atom is guessed.  ``echo`` counts how many heavy
    copies of each normalized base identity exist, which is what the
    cross-RU synchronous mask uses.
    """
    positions = torch.as_tensor(trimer.trimer_pos)
    geometry_valid = bool(getattr(trimer, 'trimer_geometry_valid', False))
    numbers = torch.as_tensor(trimer.trimer_atomic_number).long().reshape(-1)
    charge = torch.as_tensor(trimer.trimer_formal_charge).long().reshape(-1)
    aromatic = torch.as_tensor(trimer.trimer_is_aromatic).bool().reshape(-1)
    base_id = torch.as_tensor(trimer.trimer_base_ru_atom_id).long().reshape(-1)
    offset = torch.as_tensor(trimer.trimer_ru_offset).long().reshape(-1)
    if positions.ndim != 2 or positions.size(1) != 3:
        raise ValueError('frozen Trimer coordinates are malformed')
    atom_count = int(numbers.numel())
    if geometry_valid:
        if int(positions.size(0)) != atom_count:
            raise ValueError('frozen Trimer atomic_number length disagrees with coordinates')
    else:
        # The frozen fallback carries structural identity but may have no
        # coordinates.  These zero rows only keep tensor shapes aligned; the
        # geometry_valid gate below excludes all geometric relations/targets.
        positions = torch.zeros((atom_count, 3), dtype=torch.float32)
    for name, value in (('atomic_number', numbers), ('formal_charge', charge),
                        ('is_aromatic', aromatic), ('base_ru_atom_id', base_id),
                        ('ru_offset', offset)):
        if int(value.numel()) != atom_count:
            raise ValueError(f'frozen Trimer {name} length disagrees with atomic_number')
    heavy = getattr(trimer, 'trimer_heavy_indices', None)
    if heavy is None:
        indices = torch.arange(positions.size(0), dtype=torch.long)[numbers > 1]
    else:
        indices = torch.as_tensor(heavy).long().reshape(-1)
        if indices.numel() == 0 or int(indices.min()) < 0 or int(indices.max()) >= positions.size(0):
            raise ValueError('frozen Trimer heavy identity is malformed')
        if int(torch.unique(indices).numel()) != int(indices.numel()):
            raise ValueError('frozen Trimer heavy identity repeats an atom')
    if not bool((numbers[indices] > 1).all()):
        raise ValueError('frozen Trimer heavy projection contains a non-heavy atom')
    of_all = torch.full((positions.size(0),), -1, dtype=torch.long)
    of_all[indices] = torch.arange(indices.numel(), dtype=torch.long)
    # ``trimer_edge_index`` is directed; physical bonds are the undirected
    # quotient, so each pair is stored once with a single agreed code.
    edge = torch.as_tensor(trimer.trimer_edge_index).long()
    code = torch.as_tensor(trimer.trimer_bond_type).long().reshape(-1)
    if edge.ndim != 2 or edge.size(0) != 2 or int(code.numel()) != int(edge.size(1)):
        raise ValueError('frozen Trimer bond table is malformed')
    pairs = {}
    for column in range(int(edge.size(1))):
        left, right = int(edge[0, column]), int(edge[1, column])
        if left == right:
            raise ValueError('frozen Trimer bond table contains a self relation')
        mapped = (int(of_all[left]), int(of_all[right]))
        if mapped[0] < 0 or mapped[1] < 0:
            continue                       # an added capping hydrogen is not in A
        key = (min(mapped), max(mapped))
        value = int(code[column])
        if key in pairs and pairs[key] != value:
            raise ValueError('frozen Trimer bond type conflicts between directions')
        pairs[key] = value
    ordered = sorted(pairs)
    bond_ends = (torch.tensor(ordered, dtype=torch.long).reshape(-1, 2)
                 if ordered else torch.zeros((0, 2), dtype=torch.long))
    bond_type = torch.tensor([pairs[key] for key in ordered], dtype=torch.long)
    heavy_base = base_id[indices]
    echo = {}
    for value in heavy_base.tolist():
        echo[value] = echo.get(value, 0) + 1
    return TrimerView(indices, of_all, numbers[indices].clone(), charge[indices].clone(),
                      aromatic[indices].clone(), heavy_base.clone(), offset[indices].clone(),
                      positions[indices].clone(), echo, bond_ends, bond_type)


def centre_mapping(topology, trimer, view):
    """Canonical O8 atom -> heavy Trimer atom, and its inverse.

    ``central_atom_index[c]`` is ``-1`` when the canonical atom has no heavy
    Trimer counterpart (a frozen record whose O8 atom is an explicit isotope
    hydrogen); the mapping itself must still be injective and complete.
    """
    canonical_to_base = torch.as_tensor(
        topology.canonical_to_trimer_base_atom_id, dtype=torch.long).reshape(-1)
    count = int(topology.mips_x.size(0))
    if int(canonical_to_base.numel()) != count:
        raise ValueError('canonical-to-normalized mapping length differs from the O8 atom count')
    if sorted(canonical_to_base.tolist()) != list(range(int(canonical_to_base.numel()))):
        raise ValueError('canonical-to-normalized mapping is not a permutation')
    raw = torch.as_tensor(trimer.mips_to_trimer_central_index, dtype=torch.long).reshape(-1)
    all_atoms = int(torch.as_tensor(trimer.trimer_atomic_number).numel())
    if int(raw.numel()) != count:
        raise ValueError('canonical-to-Trimer mapping length differs from the O8 atom count')
    if int(torch.unique(raw).numel()) != int(raw.numel()):
        raise ValueError('canonical-to-Trimer mapping repeats a Trimer atom')
    if int(raw.min()) < 0 or int(raw.max()) >= all_atoms:
        raise ValueError('canonical-to-Trimer mapping leaves the frozen coordinate table')
    central = torch.as_tensor(
        getattr(trimer, 'trimer_central_ru_mask', torch.empty(0)), dtype=torch.bool).reshape(-1)
    if int(central.numel()) == all_atoms:
        if not bool(central[raw].all()):
            raise ValueError('canonical-to-Trimer mapping leaves the central repeat unit')
    elements = torch.as_tensor(trimer.trimer_atomic_number).long().reshape(-1)
    topology_z = torch.as_tensor(topology.z).long().reshape(-1)
    if not torch.equal(elements[raw], topology_z):
        raise ValueError('canonical-to-Trimer mapping disagrees on the element table')
    central_index = view.of_all[raw].clone()
    image = torch.full((view.count,), -1, dtype=torch.long)
    heavy = central_index >= 0
    if bool(heavy.any()):
        image[central_index[heavy]] = torch.nonzero(heavy, as_tuple=False).flatten()
    # ``raw`` is a complete injection onto the central repeat unit, so the
    # mapping is never missing.  A canonical atom whose counterpart is an
    # explicit isotope hydrogen simply has no heavy representative: that is a
    # readout limitation of the graph, not a broken mapping.
    return central_index, image, count - int(heavy.sum())


def connectivity_echo(topology, trimer, view, central_index):
    """Every canonical O8 atom must have exactly one heavy Trimer copy.

    Returns the per-canonical count of heavy copies so that the P0 audit can
    report unmatched atoms instead of silently treating them as ordinary
    invalid geometry.
    """
    return torch.tensor(
        [view.echo.get(int(value), 0) for value in
         torch.as_tensor(topology.canonical_to_trimer_base_atom_id).long().reshape(-1).tolist()],
        dtype=torch.long)


def physical_bonds(static, view):
    """Unique covalent bonds in the frozen GLT bond-row order.

    The order is the one already used by ``dual_static_v1`` token rows, so the
    cached centre/angle target tables index straight into it.
    """
    token_a = torch.as_tensor(np.array(static['token_pos_index_a'], dtype=np.int64, copy=True)).reshape(-1)
    token_b = torch.as_tensor(np.array(static['token_pos_index_b'], dtype=np.int64, copy=True)).reshape(-1)
    token_type = torch.as_tensor(np.array(static['token_bond_type'], dtype=np.int64, copy=True)).reshape(-1)
    center = torch.as_tensor(np.array(static['token_center_mask'], dtype=bool, copy=True))
    if token_a.numel() != token_b.numel() or token_a.numel() != token_type.numel():
        raise ValueError('frozen token bond table is malformed')
    if token_a.numel() != int(center.numel()):
        raise ValueError('frozen token centre mask length disagrees with the bond table')
    ends = view.of_all[torch.stack([token_a, token_b])]
    if bool((ends < 0).any()):
        raise ValueError('a frozen GLT bond endpoint is not a heavy Trimer atom')
    return torch.stack([torch.minimum(ends[0], ends[1]),
                        torch.maximum(ends[0], ends[1])], dim=1), \
        token_type.clone(), center.clone()


def _bond_category(ends, bond_ends, bond_type):
    """Chemical category of every physical pair; spatial-only pairs are NONBONDED.

    ``dual_static_v1`` stores ``bond_type_index`` (0..4), which is exactly the
    declared chemical part of the edge vocabulary, so the frozen code is used
    as-is and never re-encoded.
    """
    category = torch.full((int(ends.size(0)),), NONBONDED_CATEGORY, dtype=torch.long)
    if not int(bond_ends.size(0)) or not int(ends.size(0)):
        return category
    count = int(max(int(ends.max()), int(bond_ends.max()))) + 1
    keys = torch.zeros(count * count, dtype=torch.bool)
    keys[bond_ends[:, 0] * count + bond_ends[:, 1]] = True
    keys[bond_ends[:, 1] * count + bond_ends[:, 0]] = True
    found = keys[ends[:, 0] * count + ends[:, 1]]
    if bool(found.any()):
        lookup = {}
        for left, right, value in zip(bond_ends[:, 0].tolist(), bond_ends[:, 1].tolist(),
                                      bond_type.tolist()):
            lookup[int(left) * count + int(right)] = int(value)
            lookup[int(right) * count + int(left)] = int(value)
        category[found] = torch.tensor(
            [lookup[int(key)] for key in
             (ends[found, 0] * count + ends[found, 1]).tolist()], dtype=torch.long)
        if bool((category[found] < 0).any()) or bool((category[found] > 4).any()):
            raise ValueError('a frozen chemical bond code is outside the declared vocabulary')
    return category


def expert_relations(positions, cutoffs=EXPERT_CUTOFFS):
    """Nested ``d <= c_k`` relations, stored in both directions per scale."""
    cloud = torch.as_tensor(positions).float()
    count = int(cloud.size(0))
    if count == 0:
        return (torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.long),
                torch.zeros(0), torch.zeros(0, dtype=torch.long))
    if not bool(torch.isfinite(cloud).all()):
        raise ValueError('expert relations require finite coordinates')
    distance = torch.cdist(cloud, cloud)
    rows, columns = torch.triu_indices(count, count, offset=1)
    pair_distance = distance[rows, columns]
    index, scale, value = [], [], []
    for slot, cutoff in enumerate(cutoffs):
        inside = pair_distance <= float(cutoff)
        left, right = rows[inside], columns[inside]
        index.append(torch.stack([torch.cat([left, right]), torch.cat([right, left])]))
        scale.append(torch.full((2 * int(left.numel()),), slot, dtype=torch.long))
        value.append(pair_distance[inside].repeat(2))
    edges = torch.cat(index, dim=1) if index else torch.zeros((2, 0), dtype=torch.long)
    return edges, torch.cat(scale), torch.cat(value)


def rbf_features(distance):
    """Fixed 64-bin Gaussian basis with centres ``4k/63`` and width ``4/63``."""
    value = torch.as_tensor(distance).float().reshape(-1, 1)
    centres = torch.as_tensor(RBF_CENTERS, dtype=torch.float32, device=value.device).reshape(1, -1)
    return torch.exp(-0.5 * ((value - centres) / float(RBF_WIDTH)) ** 2)


# ---------------------------------------------------------------------------
# Section 3.2/6 -- node categories and the synchronous mask
# ---------------------------------------------------------------------------

def element_categories(z):
    value = torch.as_tensor(z).long().reshape(-1)
    return torch.where((value >= 1) & (value <= 118), value - 1,
                       torch.full_like(value, ELEMENT_UNKNOWN))


def charge_categories(charge):
    value = torch.as_tensor(charge).long().reshape(-1)
    return torch.where((value >= -3) & (value <= 3), value + 3,
                       torch.full_like(value, CHARGE_OTHER))


def aromatic_categories(aromatic):
    return torch.as_tensor(aromatic).bool().reshape(-1).long()


def synchronised_mask(topology, view, canonical_mask):
    """Set the heavy copies of every masked canonical atom to MASK.

    A canonical atom's normalized base identity is its copy label, so every
    heavy Trimer atom sharing that base id is a provable copy of the same
    chemical identity.  Capping atoms have no base identity and are outside the
    heavy enumeration altogether.
    """
    canonical_to_base = torch.as_tensor(
        topology.canonical_to_trimer_base_atom_id, dtype=torch.long).reshape(-1)
    flag = torch.as_tensor(canonical_mask).bool().reshape(-1)
    if int(flag.numel()) != int(canonical_to_base.numel()):
        raise ValueError('canonical mask length disagrees with the canonical mapping')
    masked_bases = {int(value) for value in canonical_to_base[flag].tolist()}
    heavy = torch.tensor([int(value) in masked_bases for value in view.base_id.tolist()],
                         dtype=torch.bool)
    z = element_categories(view.z).clone()
    charge = charge_categories(view.charge).clone()
    aromatic = aromatic_categories(view.aromatic).clone()
    z[heavy] = ELEMENT_MASK
    charge[heavy] = CHARGE_MASK
    aromatic[heavy] = AROMATIC_MASK
    return z, charge, aromatic, heavy


# ---------------------------------------------------------------------------
# Section 7 -- targets
# ---------------------------------------------------------------------------

def _target_geometry(static, positions):
    materialised = materialize_dual_geometry(static, positions)
    if not materialised['geometry_valid']:
        return None
    return materialised['bond_distance'], materialised['line_angle']


def centre_angle_rows(static):
    """Rows of the line table that define one genuine centre one-hop angle."""
    path = torch.as_tensor(np.array(static['line_path'], dtype=np.int64, copy=True)).reshape(-1, 3)
    mask = torch.as_tensor(np.array(static['line_path_mask'], dtype=bool, copy=True)).reshape(-1, 2)
    is_self = torch.as_tensor(np.array(static['line_is_self'], dtype=bool, copy=True)).reshape(-1)
    center = torch.as_tensor(np.array(static['token_center_mask'], dtype=bool, copy=True)).reshape(-1)
    if path.size(0) != int(mask.size(0)) or path.size(0) != int(is_self.numel()):
        raise ValueError('frozen line table is malformed')
    left = path[:, 0].clamp_min(0)
    right = path[:, 1].clamp_min(0)
    valid = (mask.sum(dim=1) == 1) & (~is_self) & (path[:, 0] >= 0) & (path[:, 1] >= 0)
    valid &= path[:, 0] < path[:, 1]
    valid &= center[left] & center[right]
    rows = torch.nonzero(valid, as_tuple=False).flatten()
    pairs = path[rows, :2].reshape(-1, 2)
    recorded = torch.as_tensor(np.array(static['angle_pairs'], dtype=np.int64, copy=True)).reshape(-1, 2)
    if not torch.equal(pairs, recorded):
        raise ValueError('frozen angle table disagrees with the line table')
    return rows, pairs


def local_targets(static, positions, bond_ends):
    """Centre-internal bond lengths and the angle cosines of section 7.2.

    Returns ``(length_pairs, length_target, angle_pairs, angle_cosine)``.  A
    graph with no centre-internal bond contributes no length and no angle
    target; a graph with bonds but no one-hop angle contributes lengths only.
    """
    geometry = _target_geometry(static, positions)
    if geometry is None:
        return (torch.zeros((0, 2), dtype=torch.long), torch.zeros(0),
                torch.zeros((0, 2), dtype=torch.long), torch.zeros(0))
    bond_distance, line_angle = geometry
    center = torch.as_tensor(np.array(static['token_center_mask'], dtype=bool, copy=True))
    length_rows = torch.nonzero(center, as_tuple=False).flatten()
    length_pairs = (bond_ends[length_rows].clone() if int(length_rows.numel())
                    else torch.zeros((0, 2), dtype=torch.long))
    angle_rows, angle_pairs = centre_angle_rows(static)
    cosines = (torch.cos(line_angle[angle_rows, 0]).clone() if int(angle_rows.numel())
               else torch.zeros(0))
    return length_pairs, bond_distance[length_rows].clone(), angle_pairs, cosines


def nonbond_candidates(positions, view, central_index, bond_ends):
    """Physical unordered pairs eligible for the per-scale non-bond objective.

    At least one endpoint is a centre heavy atom, the pair is not a direct
    covalent bond and the *view* distance lies in ``(0, 4]``.  Pairs are keyed by
    physical identity only, never by a canonical representative.
    """
    cloud = torch.as_tensor(positions).float()
    count = int(cloud.size(0))
    if count < 2:
        return torch.zeros((0, 2), dtype=torch.long), torch.zeros(0)
    is_center = torch.zeros(count, dtype=torch.bool)
    mapped = central_index[central_index >= 0]
    is_center[mapped] = True
    rows, columns = torch.triu_indices(count, count, offset=1)
    distance = torch.linalg.vector_norm(cloud[rows] - cloud[columns], dim=-1)
    keep = (is_center[rows] | is_center[columns]) & (distance > 0) & (distance <= EXPERT_CUTOFFS[-1])
    if int(bond_ends.size(0)):
        count_key = count
        keys = bond_ends[:, 0] * count_key + bond_ends[:, 1]
        pair_keys = rows * count_key + columns
        keep &= ~torch.isin(pair_keys, keys)
    pairs = torch.stack([rows[keep], columns[keep]], dim=1)
    return pairs, distance[keep]


def clean_nonbond_distances(trimer, heavy_indices, pairs):
    """Measure heavy-view pairs in the same compact index space as the experts."""
    clean = torch.as_tensor(trimer.trimer_pos)[torch.as_tensor(heavy_indices).long()]
    return torch.linalg.vector_norm(clean[pairs[:, 0]] - clean[pairs[:, 1]], dim=-1)


def sample_nonbond_pairs(pairs, distance, generator, *, bins=NONBOND_BINS,
                         maximum=NONBOND_MAX_PAIRS):
    """Up to ``maximum`` uniformly drawn pairs per distance bin, without replacement."""
    chosen, slots = [], []
    for slot, (lower, upper) in enumerate(bins):
        inside = torch.nonzero((distance > float(lower)) & (distance <= float(upper)),
                               as_tuple=False).flatten()
        if not int(inside.numel()):
            continue
        if int(inside.numel()) > int(maximum):
            order = torch.randperm(int(inside.numel()), generator=generator)[:int(maximum)]
            inside = inside[order.sort().values]
        chosen.append(pairs[inside])
        slots.append(torch.full((int(inside.numel()),), slot, dtype=torch.long))
    if not chosen:
        return torch.zeros((0, 2), dtype=torch.long), torch.zeros(0, dtype=torch.long)
    return torch.cat(chosen), torch.cat(slots)


def normalise_geometric(value, statistics):
    """``(log1p(x) - mu) / sigma`` with the fixed floor and a stat provenance check."""
    mean, scale = float(statistics['mu']), float(statistics['sigma'])
    if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('geometric normalization requires a finite mu and a positive sigma')
    return (torch.log1p(torch.as_tensor(value).float()) - mean) / scale


def geometric_statistics(values, *, minimum_std=NORMALIZATION_STD_FLOOR):
    """Fixed reference statistics: one mu/sigma pair per target family."""
    sample = torch.as_tensor(values).float().reshape(-1)
    if int(sample.numel()) == 0:
        raise ValueError('normalization statistics require at least one target')
    if not bool(torch.isfinite(sample).all()):
        raise ValueError('normalization targets contain a non-finite value')
    transformed = torch.log1p(sample)
    mean = float(transformed.mean())
    std = float(transformed.std(unbiased=False)) if int(transformed.numel()) > 1 else 0.0
    return {'mu': mean, 'sigma': max(std, float(minimum_stdev(minimum_std))),
            'count': int(transformed.numel()), 'log1p_min': float(transformed.min()),
            'log1p_max': float(transformed.max())}


def minimum_stdev(floor=NORMALIZATION_STD_FLOOR):
    return float(floor)


# ---------------------------------------------------------------------------
# Section 5.3 -- the shared noisy view and the fixed reference view
# ---------------------------------------------------------------------------

def _perturb(trimer, sigma, generator):
    changed = copy.copy(trimer)
    changed.trimer_pos = torch.as_tensor(trimer.trimer_pos).float().clone()
    if float(sigma) > 0:
        changed.trimer_pos += float(sigma) * torch.randn(changed.trimer_pos.shape,
                                                         generator=generator)
    return changed


def view_generator(seed, key, position, label=None):
    """A labelled sub-stream; ``label=None`` is the pre-existing shared stream."""
    if label is None:
        return sample_generator(seed, key, position)
    return sample_generator(seed, f'{key}:{label}', position)


def reference_view(trimer, key, *, sigma, seed=42):
    """The fixed reference view used for P_train statistics and router constants.

    It is a pure function of ``(trimer, key, sigma, seed)`` and never reads the
    training position, so building statistics cannot move the training stream.
    """
    return _perturb(trimer, sigma, view_generator(seed, key, 0, REFERENCE_SUBSTREAM))


def validate_geometry_carrier(topology, trimer):
    """The coordinate-free identity check ``build_dual_sample`` runs on a fallback row.

    Exposed so that an offline consumer of the same presentation validates
    exactly the carrier the online path validates, instead of re-implementing
    the check and drifting from it.  A structurally complete row is accepted.
    """
    if not bool(getattr(trimer, 'trimer_geometry_valid', False)):
        from .glt_dual import _validate_geometry_fallback_carrier

        _validate_geometry_fallback_carrier(topology, trimer)


class MCLPHView:
    """Everything random about one presentation, drawn once, in stream order.

    The motif mask and the coordinate noise come from the *same* generator, and
    the mask is drawn first, so the noisy coordinates of a presentation are not a
    function of ``(seed, key, position)`` alone.  Any consumer that needs those
    coordinates -- the training stream, the audit passes and the offline
    trajectory cache alike -- has to come through :func:`build_mcl_ph_view`, so
    that no second implementation of the draw order can exist.
    """

    __slots__ = ('view', 'generator', 'mask', 'fallback', 'identity', 'base_view',
                 'changed', 'field_view')

    def __init__(self, view, generator, mask, fallback, identity, base_view, changed,
                 field_view):
        self.view = str(view)
        self.generator = generator
        self.mask = mask
        self.fallback = fallback
        self.identity = identity
        self.base_view = base_view
        self.changed = changed
        self.field_view = field_view


def build_mcl_ph_view(topology, trimer, smiles, *, seed, key, position, sigma=0.03,
                      ratio=None, identity=None, view='noisy'):
    """Draw the mask and the perturbation that define one presentation's 3D view.

    ``view='noisy'`` is the pre-training configuration: the shared stream is
    consumed as ``motif mask`` then ``coordinate noise``, in that order, so the
    noisy coordinates depend on ``ratio`` as well as on ``sigma``.  A consumer
    that skipped the mask draw would silently see a different molecule.

    ``view='clean'`` keeps the frozen coordinates and draws nothing, which is the
    fine-tuning and deployment configuration of section 5.3.
    """
    if sigma < 0 or not math.isfinite(float(sigma)):
        raise ValueError('noise sigma must be finite and non-negative')
    if ratio is not None and not 0 < float(ratio) < 1:
        raise ValueError('the masking ratio must lie strictly inside (0,1)')
    if str(view) not in {'noisy', 'clean'}:
        raise ValueError('view must be noisy or clean')
    validate_geometry_carrier(topology, trimer)
    generator = sample_generator(seed, key, position)
    mask, fallback = None, False
    if ratio is not None:
        from .canonical_periodic import resolve_normalized_identity
        from .glt_dual_pretrain import chemical_groups
        if identity is None:
            identity = resolve_normalized_identity(topology, smiles, require_fields=True)
        mask, fallback = motif_mask(
            topology, chemical_groups(identity['normalized_smiles']), generator, float(ratio))
    shared = view == 'noisy'
    changed = _perturb(trimer, sigma, generator) if shared else trimer
    base_view = build_trimer_view(trimer)
    field_view = build_trimer_view(changed) if shared else base_view
    return MCLPHView(view, generator, mask, fallback, identity, base_view, changed,
                     field_view)


def cached_topology_trajectory(trajectory):
    """Validate one cached ``[31,5]`` trajectory; the dtype is never converted.

    A cache that is offered to the training path either matches the declared
    online product exactly or is refused here: there is no code path that mixes
    part of a cached presentation with part of a recomputed one.
    """
    value = np.asarray(trajectory)
    shape = (len(ROUTER_RADII), DESCRIPTOR_COLUMNS)
    if value.shape != shape:
        raise ValueError(f'cached topology trajectory shape {value.shape} is not {shape}')
    if value.dtype != np.float32:
        raise ValueError(f'cached topology trajectory dtype {value.dtype} is not float32')
    if not np.isfinite(value).all():
        raise ValueError('cached topology trajectory contains NaN/Inf')
    return value


# ---------------------------------------------------------------------------
# Sample construction and collation
# ---------------------------------------------------------------------------

def prepare_mcl_ph_sample(topology, trimer, smiles, *, seed, key, position,
                          sigma=0.03, ratio=None, static=None, identity=None,
                          statistics=None, view='noisy', trajectory_override=None):
    """Build one MCL-PH sample and its independent supervision targets.

    ``view='noisy'`` is the pre-training configuration: the experts, the RBF,
    the topology trajectory, the router descriptors and the per-scale non-bond
    candidates all come from the *same* perturbation of the frozen coordinates,
    while every target is measured on the clean coordinates.  ``view='clean'``
    keeps the frozen coordinates and disables the perturbation, which is the
    fine-tuning / deployment configuration of section 5.3.

    ``ratio=None`` disables the chemical mask (fine-tuning); otherwise the
    pre-existing 30% motif mask is drawn first from the shared stream, exactly
    as the dual route draws it, so the shared stream keeps its position.

    ``trajectory_override`` replaces one and only one thing: the ``[31,5]``
    topology trajectory that would otherwise be recomputed by
    :func:`five_descriptors`.  Everything else -- the noisy coordinates, the
    expert relations, the mask and every target -- is still built by this
    function from the same stream.  An override that is not exactly the declared
    product (shape, ``float32``, finite) is refused rather than repaired, so a
    cached presentation can never be silently half-used.
    """
    if static is None:
        raise ValueError('the MCL-PH route requires the frozen dual-static rows')
    shared = build_mcl_ph_view(topology, trimer, smiles, seed=seed, key=key,
                               position=position, sigma=sigma, ratio=ratio,
                               identity=identity, view=view)
    mask, fallback, changed = shared.mask, shared.fallback, shared.changed
    identity = shared.identity
    base_view, field_view = shared.base_view, shared.field_view
    clean = build_dual_sample(topology, trimer, smiles, identity=identity, static=static)
    result = clean
    if str(view) == 'noisy':
        result = build_dual_sample(topology, changed, smiles, identity=identity, static=static)
        for field in ('line_source', 'line_target', 'line_path', 'line_path_group',
                      'bond_center', 'bond_type'):
            if not torch.equal(getattr(clean, field), getattr(result, field)):
                raise ValueError('coordinate perturbation changed physical topology')
    central_index, image_to_canonical, non_heavy_canonical = centre_mapping(
        topology, trimer, base_view)
    canonical_to_base = torch.as_tensor(
        topology.canonical_to_trimer_base_atom_id, dtype=torch.long).reshape(-1)
    if sorted(canonical_to_base.tolist()) != list(range(int(canonical_to_base.numel()))):
        raise ValueError('canonical-to-normalized mapping is not a permutation')
    bond_index, bond_type, bond_center = physical_bonds(static, base_view)
    geometry_valid = bool(result.geometry_valid) and bool(static['geometry_valid'])
    readout_valid = bool(geometry_valid) and int(non_heavy_canonical) == 0
    if mask is None:
        element = element_categories(base_view.z)
        charge = charge_categories(base_view.charge)
        aromatic = aromatic_categories(base_view.aromatic)
        masked_atoms = torch.zeros(base_view.count, dtype=torch.bool)
    else:
        element, charge, aromatic, masked_atoms = synchronised_mask(topology, base_view, mask)
    trajectory = np.zeros((len(ROUTER_RADII), DESCRIPTOR_COLUMNS), dtype=np.float32)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_scale = torch.zeros(0, dtype=torch.long)
    edge_distance = torch.zeros(0)
    edge_type = torch.zeros(0, dtype=torch.long)
    if geometry_valid:
        trajectory = (five_descriptors(field_view.positions.numpy()).astype(np.float32)
                      if trajectory_override is None
                      else cached_topology_trajectory(trajectory_override))
        edge_index, edge_scale, edge_distance = expert_relations(field_view.positions)
        if int(edge_index.size(1)):
            edge_type = _bond_category(edge_index.t().contiguous(), bond_index, bond_type)
    elif trajectory_override is not None:
        # The online path stores the declared all-zero row for a sample without a
        # usable geometry.  A cached row that says anything else is a mismatch
        # between the cache and this run, not a richer trajectory.
        if cached_topology_trajectory(trajectory_override).any():
            raise ValueError('cached topology trajectory disagrees with an invalid geometry')
    result.mcl_z = element
    result.mcl_charge = charge
    result.mcl_aromatic = aromatic
    result.mcl_masked = masked_atoms
    result.mcl_pos = field_view.positions.float().clone()
    result.mcl_central_index = central_index
    result.mcl_image_to_canonical = image_to_canonical
    result.mcl_bond_index = bond_index
    result.mcl_bond_type = bond_type
    result.mcl_bond_center = bond_center
    result.mcl_edge_index = edge_index
    result.mcl_edge_scale = edge_scale
    result.mcl_edge_distance = edge_distance.float()
    result.mcl_edge_type = edge_type
    result.mcl_trajectory = torch.from_numpy(trajectory.copy())
    result.mcl_geometry_valid = torch.tensor(bool(geometry_valid))
    result.mcl_readout_valid = torch.tensor(bool(readout_valid))
    labels = _sample_targets(topology, trimer, static, field_view, central_index,
                             bond_index, bond_center, mask, fallback,
                             geometry_valid, seed, key, position, statistics)
    labels['sample_key'] = str(key)
    labels['geometry_valid'] = bool(geometry_valid)
    labels['readout_valid'] = bool(readout_valid)
    labels['non_heavy_canonical'] = int(non_heavy_canonical)
    return result, labels


def _sample_targets(topology, trimer, static, field_view, central_index, bond_index,
                    bond_center, mask, fallback, geometry_valid, seed, key, position,
                    statistics):
    """Atom, local-geometry and per-scale non-bond targets of one sample."""
    count = int(topology.mips_x.size(0))
    atom_mask = (mask.clone() if mask is not None else torch.zeros(count, dtype=torch.bool))
    atom_label = topology.mips_x[:, :101].argmax(-1).long()
    clean_positions = torch.as_tensor(trimer.trimer_pos)
    if geometry_valid:
        length_pairs, length_raw, angle_pairs, angle_cosine = local_targets(
            static, clean_positions, bond_index)
        # Candidates come from the same view the experts see; a graph without a
        # valid geometry has no view, so it must not produce a non-bond target
        # either.  Fabricating one would put a fake sample in a denominator.
        pairs, view_distance = nonbond_candidates(field_view.positions, field_view,
                                                  central_index, bond_index)
    else:
        length_pairs = torch.zeros((0, 2), dtype=torch.long)
        length_raw = torch.zeros(0)
        angle_pairs = torch.zeros((0, 2), dtype=torch.long)
        angle_cosine = torch.zeros(0)
        pairs = torch.zeros((0, 2), dtype=torch.long)
        view_distance = torch.zeros(0)
    sampled, slots = sample_nonbond_pairs(
        pairs, view_distance, view_generator(seed, key, position, PAIR_SUBSTREAM))
    clean_distance = (clean_nonbond_distances(trimer, field_view.indices, sampled)
                      if int(sampled.size(0)) else torch.zeros(0))
    applied = statistics is not None
    length_target = (normalise_geometric(length_raw, statistics['length']) if applied
                     else length_raw.clone())
    nonbond_target = (normalise_geometric(clean_distance, statistics['nonbond']) if applied
                      else clean_distance.clone())
    return {
        'atom_mask': atom_mask, 'atom_label': atom_label,
        'mcl_length_pair': length_pairs, 'mcl_length_target': length_target,
        'mcl_length_raw': length_raw,
        'mcl_angle_pair': angle_pairs, 'mcl_angle_target': angle_cosine,
        'mcl_nonbond_pair': sampled, 'mcl_nonbond_slot': slots,
        'mcl_nonbond_target': nonbond_target, 'mcl_nonbond_raw': clean_distance,
        'mcl_local_valid': torch.tensor(bool(int(length_pairs.numel()))),
        'mcl_nonbond_valid': torch.tensor(bool(int(sampled.size(0)))),
        'mcl_center_bonds': torch.tensor(int(bond_center.sum())),
        'mcl_fallback': bool(fallback), 'mcl_statistics_applied': bool(applied),
    }


_COLLATE_ATOM_FIELDS = ('mcl_z', 'mcl_charge', 'mcl_aromatic', 'mcl_masked', 'mcl_pos',
                        'mcl_bond_type', 'mcl_edge_scale', 'mcl_edge_distance',
                        'mcl_edge_type')

_COLLATE_TARGET_FIELDS = ('atom_mask', 'atom_label', 'mcl_length_target',
                          'mcl_length_raw', 'mcl_angle_target', 'mcl_nonbond_target',
                          'mcl_nonbond_raw', 'mcl_nonbond_slot')

# Fields the shared dual collator produces but the MCL-PH architecture never
# reads.  ``bond_z_a``/``bond_z_b`` are the only ones that could carry a masked
# atom's element identity, so the new route drops the whole unused block instead
# of shipping a latent identity copy that no current consumer happens to read.
UNUSED_BY_MCL_PH = ('bond_z_a', 'bond_z_b', 'bond_distance', 'bond_type', 'bond_center',
                    'bond_batch', 'line_angle', 'line_mask', 'line_source', 'line_target',
                    'line_path', 'line_path_group', 'line_is_self')


def mcl_ph_collate(records):
    """Pack MCL-PH samples: O8 fields plus the declared 3D view and targets.

    Two distinct index spaces are offset independently: the O8 canonical atoms
    and the Trimer heavy atoms.  A single shared offset would silently shift
    ``image_to_canonical`` away from ``central_atom_index``.
    """
    from .glt_dual import dual_glt_collate

    inputs, targets = zip(*records)
    batch = dual_glt_collate(list(inputs))
    for name in UNUSED_BY_MCL_PH:
        if hasattr(batch, name):
            delattr(batch, name)
    packed = {name: [] for name in _COLLATE_ATOM_FIELDS}
    packed.update(mcl_central_index=[], mcl_image_to_canonical=[], mcl_atom_batch=[],
                  mcl_bond_index=[], mcl_edge_index=[])
    length_pair, angle_pair, nonbond_pair = [], [], []
    length_graph, angle_graph, nonbond_graph = [], [], []
    heavy_offset = canonical_offset = 0
    for graph, item in enumerate(inputs):
        heavy = int(item.mcl_z.numel())
        canonical = int(item.mcl_central_index.numel())
        for name in _COLLATE_ATOM_FIELDS:
            packed[name].append(getattr(item, name))
        packed['mcl_atom_batch'].append(torch.full((heavy,), graph, dtype=torch.long))
        packed['mcl_central_index'].append(item.mcl_central_index + heavy_offset)
        packed['mcl_image_to_canonical'].append(item.mcl_image_to_canonical + canonical_offset)
        packed['mcl_bond_index'].append(item.mcl_bond_index + heavy_offset)
        packed['mcl_edge_index'].append(item.mcl_edge_index + heavy_offset)
        length_pair.append(targets[graph]['mcl_length_pair'] + heavy_offset)
        angle_pair.append(targets[graph]['mcl_angle_pair'] + heavy_offset)
        nonbond_pair.append(targets[graph]['mcl_nonbond_pair'] + heavy_offset)
        length_graph.append(torch.full((int(length_pair[-1].size(0)),), graph, dtype=torch.long))
        angle_graph.append(torch.full((int(angle_pair[-1].size(0)),), graph, dtype=torch.long))
        nonbond_graph.append(torch.full((int(nonbond_pair[-1].size(0)),), graph, dtype=torch.long))
        heavy_offset += heavy
        canonical_offset += canonical
    for name, values in packed.items():
        setattr(batch, name, torch.cat(values, dim=1 if name == 'mcl_edge_index' else 0))
    batch.mcl_trajectory = torch.stack([item.mcl_trajectory for item in inputs])
    # The per-graph validity flags are what the architecture branches on, so
    # they travel on the batch and not only in the supervision dictionary.
    batch.mcl_geometry_valid = torch.tensor(
        [bool(target['geometry_valid']) for target in targets], dtype=torch.bool)
    batch.mcl_readout_valid = torch.tensor(
        [bool(target['readout_valid']) for target in targets], dtype=torch.bool)
    labels = {name: torch.cat([target[name] for target in targets])
              for name in _COLLATE_TARGET_FIELDS}
    labels['mcl_length_pair'] = _pairs(length_pair)
    labels['mcl_angle_pair'] = _pairs(angle_pair)
    labels['mcl_nonbond_pair'] = _pairs(nonbond_pair)
    labels['mcl_length_graph'] = torch.cat(length_graph)
    labels['mcl_angle_graph'] = torch.cat(angle_graph)
    labels['mcl_nonbond_graph'] = torch.cat(nonbond_graph)
    for name, key in (('mcl_local_valid', 'mcl_local_valid'),
                      ('mcl_nonbond_valid', 'mcl_nonbond_valid')):
        labels[name] = torch.stack([target[key] for target in targets])
    labels['mcl_center_bonds'] = torch.stack([target['mcl_center_bonds'] for target in targets])
    labels['mcl_geometry_valid'] = torch.tensor(
        [bool(target['geometry_valid']) for target in targets], dtype=torch.bool)
    labels['mcl_readout_valid'] = torch.tensor(
        [bool(target['readout_valid']) for target in targets], dtype=torch.bool)
    labels['sample_key'] = [target['sample_key'] for target in targets]
    labels['mcl_fallback_count'] = sum(bool(target['mcl_fallback']) for target in targets)
    labels['mcl_statistics_applied'] = all(
        bool(target['mcl_statistics_applied']) for target in targets)
    return batch, labels


def _pairs(blocks):
    if not blocks:
        return torch.zeros((0, 2), dtype=torch.long)
    return torch.cat(blocks).reshape(-1, 2)


# ---------------------------------------------------------------------------
# Shared normalization statistics (produced once by the P0 pass)
# ---------------------------------------------------------------------------

STATISTICS_FIELDS = ('router_mean', 'router_std', 'length_mu', 'length_sigma',
                     'nonbond_mu', 'nonbond_sigma', 'samples', 'seed', 'sigma',
                     'ordered_key_sha256')


def load_geometric_statistics(path):
    """Load the single shared P_train statistics artifact.

    The same file feeds every arm, so no arm can fit its own normalization on
    the downstream or validation rows.
    """
    with np.load(path, allow_pickle=False) as payload:
        missing = [name for name in STATISTICS_FIELDS if name not in payload.files]
        if missing:
            raise ValueError('MCL-PH statistics artifact is missing: ' + ','.join(missing))
        mean = np.asarray(payload['length_mu'], dtype=np.float64).reshape(-1)
        scale = np.asarray(payload['length_sigma'], dtype=np.float64).reshape(-1)
        nonbond_mean = np.asarray(payload['nonbond_mu'], dtype=np.float64).reshape(-1)
        nonbond_scale = np.asarray(payload['nonbond_sigma'], dtype=np.float64).reshape(-1)
        router_mean = np.asarray(payload['router_mean'], dtype=np.float32)
        router_std = np.asarray(payload['router_std'], dtype=np.float32)
        record = {
            'length': {'mu': float(mean[0] if mean.size else 0.0),
                       'sigma': float(scale[0] if scale.size else 0.0)},
            'nonbond': {'mu': float(nonbond_mean[0] if nonbond_mean.size else 0.0),
                        'sigma': float(nonbond_scale[0] if nonbond_scale.size else 0.0)},
            'router_mean': router_mean,
            'router_std': router_std,
            'samples': int(np.asarray(payload['samples']).reshape(-1)[0]),
            'seed': int(np.asarray(payload['seed']).reshape(-1)[0]),
            'sigma_noise': float(np.asarray(payload['sigma']).reshape(-1)[0]),
            'ordered_key_sha256': str(np.asarray(payload['ordered_key_sha256']).reshape(-1)[0]),
            'path': str(path),
        }
    for family in ('length', 'nonbond'):
        value = record[family]
        if not math.isfinite(value['mu']) or not math.isfinite(value['sigma']) \
                or value['sigma'] <= 0:
            raise ValueError(f'{family} normalization statistics are not usable')
    if record['router_mean'].shape != (len(ROUTER_RADII), DESCRIPTOR_COLUMNS):
        raise ValueError('router constant profile must be [31,5]')
    return record


class MCLPHMicrobatchStream:
    """Per-rank microbatch stream identical whether it is iterated inline or
    from DataLoader workers, because every draw is a pure function of
    ``(seed, key, position)``."""

    def __init__(self, source, *, seed, world, rank, microbatch, accumulation,
                 start_step, max_steps, sigma, ratio, statistics, order=None,
                 view='noisy', trajectory_cache=None):
        from src.training.glt_dual_runtime import OrderedSampleStream

        self.source = source
        self.stream = OrderedSampleStream(len(source), int(seed))
        self.seed = int(seed)
        self.world = int(world)
        self.rank = int(rank)
        self.microbatch = int(microbatch)
        self.accumulation = int(accumulation)
        self.batch_size = self.microbatch * self.world * self.accumulation
        self.start_step = int(start_step)
        self.steps = max(0, int(max_steps) - self.start_step)
        self.sigma = float(sigma)
        self.ratio = ratio
        self.statistics = statistics
        self.order = order
        self.view = str(view)
        # One reader per process: the cache opens its shards lazily, so a worker
        # that never touches a shard never pays for it, and the 3 GB is never
        # pickled into a worker.
        self.trajectory_cache = trajectory_cache

    def __len__(self):
        return self.steps * self.accumulation

    def __getitem__(self, item):
        step = self.start_step + item // self.accumulation
        offset = item % self.accumulation
        rows = []
        for local in range(self.microbatch):
            position = (step * self.batch_size + offset * self.world * self.microbatch
                        + self.rank * self.microbatch + local)
            index = self.stream.index_at(position)
            if self.order is not None:
                index = int(self.order[index])
            key = self.source.samples[index][0].hex()
            rows.append(prepare_mcl_ph_sample(
                *self.source[index], seed=self.seed, key=key, position=position,
                sigma=self.sigma, ratio=self.ratio,
                static=self.source.static_for(index), statistics=self.statistics,
                view=self.view,
                trajectory_override=(None if self.trajectory_cache is None
                                     else self.trajectory_cache[position])))
        return mcl_ph_collate(rows)





__all__ = [
    'AROMATIC_MASK', 'AROMATIC_VOCABULARY', 'BOND_CATEGORY_COUNT', 'CHARGE_MASK',
    'CHARGE_OTHER', 'CHARGE_VOCABULARY', 'DESCRIPTOR_COLUMNS', 'ELEMENT_MASK',
    'ELEMENT_UNKNOWN', 'ELEMENT_VOCABULARY', 'EXPERT_CUTOFFS', 'NONBOND_BINS',
    'NONBOND_MAX_PAIRS', 'NONBONDED_CATEGORY', 'NORMALIZATION_STD_FLOOR',
    'PAIR_SUBSTREAM', 'RBF_BINS', 'RBF_CENTERS', 'RBF_WIDTH', 'REFERENCE_SUBSTREAM',
    'ROUTER_RADII', 'ROUTER_RADII_ARRAY', 'UNKNOWN_CATEGORY', 'VR_MAX_EDGE',
    'TrimerView', 'aromatic_categories', 'build_mcl_ph_view', 'build_trimer_view',
    'cached_topology_trajectory', 'centre_angle_rows',
    'centre_mapping', 'charge_categories', 'connectivity_echo', 'descriptor_channel_variance',
    'distance_matrix', 'element_categories', 'expert_relations', 'five_descriptors',
    'MCLPHMicrobatchStream', 'MCLPHView', 'geometric_statistics', 'load_geometric_statistics',
    'local_targets', 'mcl_ph_collate', 'normalise_geometric',
    'nonbond_candidates', 'persistence_pairs', 'physical_bonds',
    'prepare_mcl_ph_sample', 'rbf_features', 'reference_view',
    'sample_nonbond_pairs', 'synchronised_mask', 'validate_geometry_carrier',
    'view_generator',
]
