"""Family-B spatial inputs over the frozen all-atom Trimer (read-only).

Nothing here builds a cache, generates coordinates or writes to a historical
artifact.  The module reads the frozen Trimer, derives the three declared
nested spatial graphs, the STAT descriptors and the arm-specific conditional
profile, and exposes the reader/collate pair used by the fusion runners.

Three facts drive the design:

* the GLT bond rows are a *heavy-atom only* projection of the frozen Trimer
  (``trimer_heavy_indices``), so the spatial graph is defined over all atoms
  while the message write-back still lands on those heavy-atom bond rows;
* the existing 32-byte sample keys are raw hashes, so every comparison here is
  on bytes (a NUL byte would truncate any string comparison);
* the conditional input is the *only* thing the three arms differ in, so the
  arm is resolved into one ``profile_input`` tensor plus its validity flag.
"""
import numpy as np
import torch

from .glt_galformer_ph import prepare_galformer_sample
from .glt_ph import PH_CHANNELS, PH_BINS, radius_grid

SPATIAL_RADII = (2.5, 4.0, 6.0)
RBF_BINS = 32
RBF_MAX = 6.0
RBF_BONDED_DIM = 1
EDGE_FEATURE_DIM = RBF_BINS + RBF_BONDED_DIM      # 33
ARMS = ('CONST', 'STAT', 'PH')


def rbf_centers():
    """32 equally spaced RBF centres on [0, 6] A; width is the centre spacing."""
    return np.linspace(0.0, RBF_MAX, RBF_BINS)


def rbf_width():
    centers = rbf_centers()
    return float(centers[1] - centers[0])


def edge_rbf(distances, bonded, centers=None, width=None):
    """[R] distances + [R] bonded bits -> [R, 33] (32 Gaussian RBF + bonded bit).

    The basis lives on the *input's* device: a batch of edges that reaches the
    GPU must not be combined with a CPU constant.
    """
    centers = rbf_centers() if centers is None else centers
    width = rbf_width() if width is None else width
    distance = torch.as_tensor(distances, dtype=torch.float32).reshape(-1, 1)
    device = distance.device
    center = torch.as_tensor(centers, dtype=torch.float32,
                             device=device).reshape(1, -1)
    gaussian = torch.exp(-((distance - center) / float(width)) ** 2)
    bit = torch.as_tensor(bonded, dtype=torch.float32,
                          device=device).reshape(-1, 1)
    return torch.cat([gaussian, bit], -1)


def heavy_projection(trimer):
    """All-atom coordinates plus the cache's own heavy-atom projection."""
    positions = torch.as_tensor(trimer.trimer_pos).float()
    numbers = torch.as_tensor(trimer.trimer_atomic_number).long().reshape(-1)
    edge = torch.as_tensor(trimer.trimer_edge_index).long()
    codes = torch.as_tensor(trimer.trimer_bond_type).long().reshape(-1)
    if positions.ndim != 2 or positions.size(1) != 3 or numbers.numel() != positions.size(0):
        raise ValueError('trimer all-atom identity is malformed')
    if edge.ndim != 2 or edge.size(0) != 2 or codes.numel() != edge.size(1):
        raise ValueError('trimer bond table is malformed')
    heavy = getattr(trimer, 'trimer_heavy_indices', None)
    if heavy is None:
        # Legacy records are already heavy-only; the identity is explicit.
        keep = torch.arange(positions.size(0), dtype=torch.long)
    else:
        keep = torch.as_tensor(heavy).long().reshape(-1)
        if (keep.numel() == 0 or int(keep.min()) < 0
                or int(keep.max()) >= positions.size(0)
                or torch.unique(keep).numel() != keep.numel()):
            raise ValueError('trimer heavy-atom identity is malformed')
    if not bool((numbers[keep] > 1).all()):
        raise ValueError('trimer heavy projection contains a non-heavy atom')
    inverse = torch.full((positions.size(0),), -1, dtype=torch.long)
    inverse[keep] = torch.arange(keep.numel(), dtype=torch.long)
    mapped = inverse[edge]
    alive = (mapped[0] >= 0) & (mapped[1] >= 0)
    return {
        'positions': positions,
        'numbers': numbers,
        'heavy_indices': keep,
        'heavy_of_all': inverse,
        'edge_heavy': mapped[:, alive].contiguous(),
        'code_heavy': codes[alive].contiguous(),
    }


def physical_bond_table(trimer, topology):
    """One row per unique undirected physical bond, in the frozen GLT row order.

    The row order is the complete-Trimer route's own ordering rule -- centre
    bonds first, then ``(atom_a, atom_b, q_a, q_b, local_a, local_b)`` on the
    heavy-projected indices, with each row oriented so that
    ``(atom_a, q_a, local_a) <= (atom_b, q_b, local_b)``.  Reproducing it here
    is what lets a spatial message land on the right bond token.
    """
    projection = heavy_projection(trimer)
    heavy = projection['heavy_indices']
    numbers = projection['numbers'][heavy]
    base = torch.as_tensor(trimer.trimer_base_ru_atom_id).long().reshape(-1)[heavy]
    offsets = torch.as_tensor(trimer.trimer_ru_offset).long().reshape(-1)[heavy]
    mapping = torch.as_tensor(topology.canonical_to_trimer_base_atom_id).long().reshape(-1)
    base_to_canonical = {}
    for canonical, value in enumerate(mapping.tolist()):
        if int(value) in base_to_canonical:
            raise ValueError('canonical base identity is duplicated')
        base_to_canonical[int(value)] = canonical
    pairs = {}
    edge, codes = projection['edge_heavy'], projection['code_heavy']
    for column in range(int(edge.size(1))):
        left, right = int(edge[0, column]), int(edge[1, column])
        if left == right:
            raise ValueError('trimer physical bond table has a self loop')
        key = (min(left, right), max(left, right))
        code = int(codes[column])
        if key in pairs and pairs[key] != code:
            raise ValueError('trimer physical bond type conflict')
        pairs[key] = code
    records = []
    for (local_a, local_b) in pairs:
        atom_a, atom_b = base_to_canonical.get(int(base[local_a])), \
            base_to_canonical.get(int(base[local_b]))
        if atom_a is None or atom_b is None:
            raise ValueError('physical bond endpoint has no canonical identity')
        q_a, q_b = int(offsets[local_a]), int(offsets[local_b])
        if (atom_b, q_b, local_b) < (atom_a, q_a, local_a):
            local_a, local_b = local_b, local_a
            atom_a, atom_b = atom_b, atom_a
            q_a, q_b = q_b, q_a
        records.append(dict(local_a=local_a, local_b=local_b, atom_a=atom_a,
                            atom_b=atom_b, q_a=q_a, q_b=q_b,
                            center=(q_a == 0 and q_b == 0)))
    records.sort(key=lambda row: (not row['center'], row['atom_a'], row['atom_b'],
                                  row['q_a'], row['q_b'], row['local_a'], row['local_b']))
    index = (torch.tensor([[row['local_a'], row['local_b']] for row in records],
                          dtype=torch.long).t().contiguous() if records
             else torch.zeros((2, 0), dtype=torch.long))
    # ``index`` lives in the heavy enumeration the GLT rows use; ``atom_index``
    # is the same bond expressed in the all-atom coordinate space the spatial
    # graph is built on.  The two differ whenever explicit hydrogens are
    # interleaved rather than trailing.
    return {'index': index, 'atom_index': projection['heavy_indices'][index],
            'code': torch.tensor([pairs[(min(row['local_a'], row['local_b']),
                                        max(row['local_a'], row['local_b']))]
                                  for row in records], dtype=torch.long),
            'center': torch.tensor([row['center'] for row in records], dtype=torch.bool),
            'projection': projection}


def bond_row_alignment(table, bond_z_a, bond_z_b, bond_distance, atol=1e-3, detail=False):
    """Check that the GLT bond rows are the physical bonds of this Trimer."""
    projection = table['projection']
    heavy_z = projection['numbers'][projection['heavy_indices']]
    index = table['index']
    issues = []
    evidence = {}
    pair_count = int(index.size(1))
    if pair_count != int(bond_z_a.numel()):
        issues.append(f'bond_row_count:{pair_count}!={int(bond_z_a.numel())}')
        return (issues, evidence) if detail else issues
    for name, observed, expected in (
            ('z_a', bond_z_a.long(), heavy_z[index[0]]),
            ('z_b', bond_z_b.long(), heavy_z[index[1]])):
        if not torch.equal(observed, expected):
            issues.append(f'bond_row_element_{name}')
            if detail:
                wrong = torch.nonzero(observed != expected).flatten()
                evidence[f'{name}_first_rows'] = [int(value) for value in wrong[:5]]
    positions = projection['positions']
    atom_index = table['atom_index']
    distance = (positions[atom_index[0]] - positions[atom_index[1]]).norm(dim=-1)
    deviation = (distance - bond_distance.float()).abs()
    if pair_count and float(deviation.max()) > float(atol):
        issues.append('bond_row_distance')
        if detail:
            worst = int(torch.argmax(deviation))
            evidence['distance'] = {
                'rows_over_tolerance': int((deviation > float(atol)).sum()),
                'first_rows': [int(value) for value in
                               torch.nonzero(deviation > float(atol)).flatten()[:5]],
                'worst_row': worst, 'worst_stored': float(bond_distance[worst]),
                'worst_recomputed': float(distance[worst]),
                'worst_z': [int(bond_z_a[worst]), int(bond_z_b[worst])],
                'worst_local': [int(index[0][worst]), int(index[1][worst])]}
    return (issues, evidence) if detail else issues


def spatial_edges(positions, radii=SPATIAL_RADII):
    """Nested d<=r graphs over all non-self atom pairs, stored in both directions."""
    count = int(positions.size(0))
    if count == 0:
        return [torch.zeros((2, 0), dtype=torch.long) for _ in radii]
    distance = torch.cdist(positions.float(), positions.float())
    rows, columns = torch.triu_indices(count, count, offset=1)
    pair_distance = distance[rows, columns]
    graphs = []
    for radius in radii:
        inside = pair_distance <= float(radius)
        pair_rows, pair_columns = rows[inside], columns[inside]
        graphs.append(torch.stack([
            torch.cat([pair_rows, pair_columns]),
            torch.cat([pair_columns, pair_rows])]))
    return graphs


def spatial_edge_count(positions, radii=SPATIAL_RADII):
    """Directed edge count per scale without materializing the graphs."""
    count = int(positions.size(0))
    if count == 0:
        return [0 for _ in radii]
    distance = torch.cdist(positions.float(), positions.float())
    rows, columns = torch.triu_indices(count, count, offset=1)
    pair_distance = distance[rows, columns]
    return [int((pair_distance <= float(radius)).sum()) * 2 for radius in radii]


def _pair_keys(pair, count):
    """One int64 key per undirected pair (``count`` must bound both endpoints)."""
    left = torch.minimum(pair[0], pair[1]).long()
    right = torch.maximum(pair[0], pair[1]).long()
    if int(right.max()) >= int(count) if right.numel() else False:
        raise ValueError('pair key index is outside the atom table')
    return left * int(count) + right


def stat_profile(positions, numbers, grid=None):
    """[3,32] STAT descriptor on the same radii as the PH profile.

    channel 0: heavy-atom unordered-pair distance CDF
    channel 1: all-atom unordered-pair distance CDF
    channel 2: heavy-atom cutoff-graph degree variance / max(1, n_heavy-1)^2

    The CDF denominator is the number of actual unordered pairs (no self
    pairs).  A channel with fewer than two points is exactly zero.
    """
    grid = radius_grid() if grid is None else np.asarray(grid, dtype=np.float64)
    positions = torch.as_tensor(positions, dtype=torch.float32)
    numbers = torch.as_tensor(numbers, dtype=torch.long).reshape(-1)
    profile = np.zeros((PH_CHANNELS, PH_BINS), dtype=np.float32)

    def pair_distance(mask):
        points = positions[mask]
        if int(points.size(0)) < 2:
            return None
        distance = torch.cdist(points, points)
        rows, columns = torch.triu_indices(points.size(0), points.size(0), offset=1)
        return distance[rows, columns]

    heavy, all_atom = numbers > 1, numbers >= 1
    for channel, mask in ((0, heavy), (1, all_atom)):
        pairs = pair_distance(mask)
        if pairs is None:
            continue
        ordered = pairs.sort().values.numpy().astype(np.float64)
        # searchsorted counts the pairs with d <= r directly.
        profile[channel] = (np.searchsorted(ordered, grid, side='right')
                            / float(ordered.size)).astype(np.float32)
    if int(heavy.sum()) >= 2:
        points = positions[heavy]
        distance = torch.cdist(points, points)
        radii = torch.as_tensor(grid, dtype=torch.float32).reshape(1, 1, -1)
        inside = (distance.unsqueeze(-1) <= radii)
        degree = inside.sum(1).to(torch.float64) - 1.0   # the point itself is always inside
        variance = ((degree - degree.mean(0)) ** 2).mean(0)
        profile[2] = (variance / max(1, int(heavy.sum()) - 1) ** 2).to(torch.float32).numpy()
    return profile


def standardize(profile, mean, std):
    """Per-element P_train standardization (same transform for all three arms)."""
    value = torch.as_tensor(profile, dtype=torch.float32)
    center = torch.as_tensor(mean, dtype=torch.float32)
    scale = torch.as_tensor(std, dtype=torch.float32)
    if center.shape != (PH_CHANNELS, PH_BINS) or scale.shape != (PH_CHANNELS, PH_BINS):
        raise ValueError('PH standardization statistics must be [3,32]')
    if float(scale.min()) <= 0:
        raise ValueError('PH standardization std must be positive')
    return (value - center) / scale


def load_ph_stats(path):
    """Load the standardization statistics plus their provenance record."""
    payload = np.load(path, allow_pickle=False)
    mean = np.asarray(payload['mean'], dtype=np.float32)
    std = np.asarray(payload['std'], dtype=np.float32)
    if mean.shape != (PH_CHANNELS, PH_BINS) or std.shape != (PH_CHANNELS, PH_BINS):
        raise ValueError('PH statistics file must hold [3,32] mean and std')
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError('PH statistics file contains non-finite values')
    if float(std.min()) <= 0:
        raise ValueError('PH statistics file has a non-positive std')
    return {'mean': mean, 'std': std,
            'source': str(payload['source']) if 'source' in payload.files else None,
            'scope': str(payload['scope']) if 'scope' in payload.files else None}


def save_ph_stats(path, mean, std, *, source, scope, samples):
    """Write statistics with their scope recorded inside the file itself."""
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != (PH_CHANNELS, PH_BINS) or std.shape != (PH_CHANNELS, PH_BINS):
        raise ValueError('PH statistics must be [3,32]')
    np.savez(path, mean=mean, std=std, source=str(source), scope=str(scope),
             samples=int(samples))


def conditional_input(positions, numbers, profile, valid, arm, const_profile):
    """The declared arm input plus its validity; the only thing arms differ in."""
    if arm not in ARMS:
        raise ValueError(f'arm must be one of {ARMS}')
    if const_profile is None:
        raise ValueError('the fixed P_train mean profile is required')
    constant = torch.as_tensor(const_profile, dtype=torch.float32).clone()
    if arm == 'CONST':
        return constant, True, 'p_train_mean'
    if arm == 'STAT':
        return (torch.as_tensor(stat_profile(positions, numbers), dtype=torch.float32),
                True, 'sample_spatial_stat')
    tensor = torch.as_tensor(np.asarray(profile, dtype=np.float32).copy())
    if not torch.isfinite(tensor).all():
        raise ValueError('non-finite PH profile')
    if not bool(valid):
        # An invalid PH row keeps the training sample and falls back to CONST,
        # exactly as the plan's PH-invalid rule requires.
        return constant, False, 'p_train_mean_fallback'
    return tensor, True, 'sample_own_frozen'


def attach_spatial_fields(data, table, radii=SPATIAL_RADII):
    """Attach the declared spatial relations of one sample (no coordinates)."""
    if not bool(data.geometry_valid):
        data.spatial_edge_index = torch.zeros((2, 0), dtype=torch.long)
        data.spatial_distance = torch.zeros(0)
        data.spatial_bonded = torch.zeros(0, dtype=torch.bool)
        data.spatial_scale = torch.zeros(0, dtype=torch.long)
        data.bond_atom_index = torch.zeros((2, 0), dtype=torch.long)
        data.atom_count = torch.tensor(int(table['projection']['positions'].size(0)))
        data.bond_count = torch.tensor(int(table['index'].size(1)))
        return data
    graphs = spatial_edges(table['projection']['positions'], radii)
    edge_index = (torch.cat(graphs, dim=1) if graphs
                  else torch.zeros((2, 0), dtype=torch.long))
    scale = torch.cat([torch.full((graph.size(1),), index, dtype=torch.long)
                       for index, graph in enumerate(graphs)])
    positions = table['projection']['positions']
    distance = (positions[edge_index[0]] - positions[edge_index[1]]).norm(dim=-1)
    count = int(positions.size(0))
    bonded = torch.zeros(edge_index.size(1), dtype=torch.bool)
    if table['atom_index'].size(1) and edge_index.size(1):
        bonded = torch.isin(_pair_keys(edge_index, count),
                            _pair_keys(table['atom_index'], count))
    data.spatial_edge_index = edge_index
    data.spatial_distance = distance
    data.spatial_bonded = bonded
    data.spatial_scale = scale
    data.bond_atom_index = table['atom_index'].clone()
    data.atom_count = torch.tensor(count)
    data.bond_count = torch.tensor(int(table['index'].size(1)))
    return data


def prepare_fusion_sample(topology, trimer, smiles, *, static, seed, key, position,
                          ph_reader, arm, const_profile, radii=SPATIAL_RADII):
    """``prepare_galformer_sample`` plus the declared spatial and conditional fields."""
    data, labels = prepare_galformer_sample(
        topology, trimer, smiles, static=static, seed=seed, key=key,
        position=position, ph_reader=(ph_reader if arm == 'PH' else None))
    table = physical_bond_table(trimer, topology)
    if bool(data.geometry_valid):
        issues = bond_row_alignment(table, data.bond_z_a, data.bond_z_b,
                                    data.bond_distance)
        if issues:
            raise ValueError('fusion bond-row alignment failed: ' + ','.join(issues))
    attach_spatial_fields(data, table, radii)
    profile, valid, source = conditional_input(
        table['projection']['positions'], table['projection']['numbers'],
        getattr(data, 'ph_profile', None), getattr(data, 'ph_valid', False),
        arm, const_profile)
    data.profile_input = profile
    data.profile_valid = torch.tensor(bool(valid))
    data.profile_source = source
    labels['profile_source'] = source
    return data, labels


def _pack_spatial(batch, samples):
    """Pack the per-graph spatial and conditional fields into one batch."""
    atom_offset = 0
    atom_batch, bond_atom_index = [], []
    edge_index, distance, bonded, scale = [], [], [], []
    profiles, valids, sources = [], [], []
    for data in samples:
        atoms = int(data.atom_count)
        graph = len(atom_batch)
        atom_batch.append(torch.full((atoms,), graph, dtype=torch.long))
        bond_atom_index.append(data.bond_atom_index + atom_offset)
        edge_index.append(data.spatial_edge_index + atom_offset)
        distance.append(data.spatial_distance)
        bonded.append(data.spatial_bonded)
        scale.append(data.spatial_scale)
        profiles.append(data.profile_input)
        valids.append(data.profile_valid)
        sources.append(data.profile_source)
        atom_offset += atoms
    batch.atom_batch = torch.cat(atom_batch)
    batch.bond_atom_index = torch.cat(bond_atom_index, dim=1)
    batch.spatial_edge_index = torch.cat(edge_index, dim=1)
    batch.spatial_distance = torch.cat(distance)
    batch.spatial_bonded = torch.cat(bonded)
    batch.spatial_scale = torch.cat(scale)
    batch.profile_input = torch.stack(profiles, 0)
    batch.profile_valid = torch.stack(valids, 0)
    batch.profile_source = sources
    return batch


def fusion_collate(records):
    """``galformer_collate`` plus the packed spatial and conditional fields."""
    from .glt_galformer_ph import galformer_collate

    batch, labels = galformer_collate(records)
    return _pack_spatial(batch, [data for data, _ in records]), labels


class FusionDataset:
    """Read-only downstream wrapper attaching one arm's declared inputs."""

    def __init__(self, source, targets, *, arm, ph_reader=None, const_profile=None,
                 key_rows=None, radii=SPATIAL_RADII):
        from ..training.glt_dual_runtime import CleanLabeledDataset

        if arm not in ARMS:
            raise ValueError(f'arm must be one of {ARMS}')
        if const_profile is None:
            raise ValueError('the fixed P_train mean profile is required')
        self.inner = CleanLabeledDataset(source, targets)
        self.source = source
        self.arm = arm
        self.ph_reader = ph_reader
        self.key_rows = dict(key_rows or {})
        self.const_profile = np.asarray(const_profile, dtype=np.float32)
        self.radii = tuple(radii)
        self.sources = []

    def __len__(self):
        return len(self.inner)

    def set_target_override(self, targets):
        self.inner.set_target_override(targets)

    @property
    def raw_targets(self):
        return self.inner.raw_targets

    @property
    def targets(self):
        return self.inner.targets

    def cache_stats(self):
        return self.inner.cache_stats()

    def __getitem__(self, index):
        index = int(index)
        data = self.inner[index]
        topology, trimer, _ = self.source[index]
        table = physical_bond_table(trimer, topology)
        if bool(data.geometry_valid):
            issues = bond_row_alignment(table, data.bond_z_a, data.bond_z_b,
                                        data.bond_distance)
            if issues:
                raise ValueError('fusion bond-row alignment failed: ' + ','.join(issues))
        attach_spatial_fields(data, table, self.radii)
        profile, valid = None, True
        if self.arm == 'PH':
            row = self.key_rows.get(self.source.samples[index][0].hex(), -1)
            profile, valid = self.ph_reader.get(row, self.source.samples[index][0].hex())
        value, valid, source = conditional_input(
            table['projection']['positions'], table['projection']['numbers'],
            profile, valid, self.arm, self.const_profile)
        data.profile_input = value
        data.profile_valid = torch.tensor(bool(valid))
        data.profile_source = source
        data.sample_key = self.source.samples[index][0].hex()
        return data


def fusion_downstream_collate(records):
    """Spatial-aware collate for the downstream adapter (no masks, no PH target)."""
    from .glt_dual import dual_glt_collate

    return _pack_spatial(dual_glt_collate(records), records)
