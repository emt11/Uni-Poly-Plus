"""Isolated PH r2 structure contracts. No target-bearing JSONL in prepare."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

SCHEMA = 'ua3d-ph-sidecar-r2'
RADII = np.linspace(1.5, 4.5, 31, dtype=np.float64)
CUTOFFS = (2., 3., 4.)


def undirected(edges, n):
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if e.size and (e.min() < 0 or e.max() >= n):
        raise ValueError('graph endpoint outside node table')
    if any(a == b for a, b in e):
        raise ValueError('chemical self edge')
    return np.asarray(sorted({tuple(sorted((int(a), int(b)))) for a, b in e}), dtype=np.int64).reshape(-1, 2)


def lifted_graph(path, shift, mask, spd, atoms):
    path, shift, mask, spd = (np.asarray(x) for x in (path, shift, mask, spd))
    if path.shape != shift.shape or path.shape != mask.shape or path.ndim != 2 or path.shape[1] != 3 or len(spd) != len(path):
        raise ValueError('lifted path schema')
    edges = []
    for p, q, m, hop in zip(path, shift, mask, spd):
        if hop != 1:
            continue
        if not (m[0] and m[1]) or m[2] or min(p[:2]) < 0 or max(p[:2]) >= atoms:
            raise ValueError('one-hop chemical path malformed')
        # Simultaneously translate both endpoints, including paths stored away from copy 0.
        for left in (-1, 0, 1):
            right = left + int(q[1]) - int(q[0])
            if right in (-1, 0, 1):
                edges.append((int(p[0]) + (left + 1) * atoms, int(p[1]) + (right + 1) * atoms))
    e = undirected(edges, 3 * atoms)
    return dict(nodes=3 * atoms, canonical_atoms=atoms, edges=e, center=np.arange(atoms, 2 * atoms),
                cross_ru=sum(int(a // atoms != b // atoms) for a, b in e), available=True)


def random_walk_pe(n, edges):
    a = np.zeros((n, n), np.float64)
    for u, v in undirected(edges, n):
        a[u, v] = a[v, u] = 1
    deg = a.sum(1)
    a[deg == 0, deg == 0] = 1
    p = a / a.sum(1)[:, None] if n else a
    state = p.copy(); result = np.zeros((n, 16), np.float64)
    for k in range(16):
        result[:, k] = state.diagonal()
        if k < 15:
            state = state @ p
    if not np.isfinite(result).all():
        raise FloatingPointError('RWPE')
    return result.astype(np.float32)


def graph_events(filtration, edges):
    """Discrete decisions only; tuples retain references to original node values.

    Each event holds birth/time reference sets, recipient nodes and multiplicity.
    Nonlinear encoding must precede distribution to recipients.
    """
    f = np.asarray(filtration, dtype=np.float64); n = len(f)
    if f.shape != (n,) or not np.isfinite(f).all() or np.any((f < 0) | (f > 1)):
        raise ValueError('filtration outside [0,1]')
    e = undirected(edges, n)
    parent = list(range(n)); members = {i: [i] for i in range(n)}
    births = {i: [i] for i in range(n)}; h0 = []; h1 = []
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    times = np.maximum(f[e[:, 0]], f[e[:, 1]]) if len(e) else np.zeros(0)
    for threshold in np.unique(times):
        pairs = [(find(int(a)), find(int(b))) for a, b in e[times == threshold]]
        adj = {}
        for a, b in pairs:
            adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
        visited = set()
        for start in sorted(adj):
            if start in visited:
                continue
            stack = [start]; cluster = []
            while stack:
                r = stack.pop()
                if r in visited:
                    continue
                visited.add(r); cluster.append(r); stack.extend(sorted(adj[r] - visited))
            cluster.sort(); roots = set(cluster)
            nodes = sorted(i for r in cluster for i in members[r])
            tref = [i for i in nodes if f[i] == threshold]
            if not tref:
                raise ValueError('threshold has no differentiable representative')
            groups = {}
            for r in cluster:
                groups.setdefault(float(f[births[r][0]]), []).append(r)
            minimum = min(groups)
            for b, rs in sorted(groups.items()):
                count = len(rs) - int(b == minimum)
                if count:
                    h0.append(dict(birth=sorted(i for r in rs for i in births[r]), death=tref,
                                   nodes=sorted(i for r in rs for i in members[r]), count=count, censored=False))
            added = sum(a in roots and b in roots for a, b in pairs)
            born = added - len(cluster) + 1
            if born < 0:
                raise ValueError('negative cycle increment')
            if born:
                h1.append(dict(birth=tref, death=[], nodes=nodes, count=born, censored=True))
            keep = cluster[0]
            minimum_refs = sorted(i for r in groups[minimum] for i in births[r])
            for r in cluster:
                parent[r] = keep
            members[keep] = nodes; births[keep] = minimum_refs
    for r in range(n):
        if find(r) == r:
            h0.append(dict(birth=births[r], death=[], nodes=sorted(members[r]), count=1, censored=True))
    if sum(x['count'] for x in h0) != n:
        raise ValueError('H0 must contain exactly one bar per vertex, including zero lifetimes')
    return (h0, h1)


def event_tuples(f, events):
    values = []
    for event in events:
        b = f[event['birth']].mean()
        death = f.new_tensor(1.) if event['censored'] else f[event['death']].mean()
        values.append(torch.stack((b, death, death - b, f.new_tensor(float(event['censored'])))))
    return torch.stack(values) if values else f.new_zeros((0, 4))


def distribute(encoded, events, nodes):
    # Dense membership multiplication avoids scatter atomics in the new PH branch.
    w = encoded.new_zeros((nodes, len(events)))
    for col, event in enumerate(events):
        w[event['nodes'], col] = event['count'] / len(event['nodes'])
    return w @ encoded


def neighbor_mean(x, edges):
    n = len(x); w = x.new_zeros((n, n))
    if len(edges):
        e = torch.as_tensor(edges, dtype=torch.long, device=x.device)
        w[e[:, 0], e[:, 1]] = 1; w[e[:, 1], e[:, 0]] = 1
    return (w / w.sum(1, keepdim=True).clamp_min(1)) @ x


def local_tuples(f, edges):
    neighbors = [[] for _ in range(len(f))]
    for u, v in np.asarray(edges).reshape(-1, 2):
        neighbors[int(u)].append(int(v)); neighbors[int(v)].append(int(u))
    h0, h1 = [], []
    for i, ids in enumerate(neighbors):
        q = f[ids]; zero = f.new_tensor(0.); one = f.new_tensor(1.)
        h0.append(torch.stack((f[i], q.mean() if ids else zero, f.new_tensor(len(ids) / (len(ids) + 1)), one)))
        h1.append(torch.stack((f[i], q.min() if ids else zero, q.max() if ids else zero, one)))
    empty = f.new_zeros((0, 4))
    return (torch.stack(h0), torch.stack(h1)) if len(f) else (empty, empty)


def vr_betti(points):
    import gudhi
    if gudhi.__version__ != '3.8.0':
        raise ValueError('GUDHI version must be 3.8.0')
    points = np.asarray(points, np.float64).reshape(-1, 3)
    if not np.isfinite(points).all():
        raise FloatingPointError('coordinates')
    curves = np.zeros((31, 2), np.float32); rows = []
    if len(points):
        st = gudhi.RipsComplex(points=points, max_edge_length=4.5).create_simplex_tree(max_dimension=2)
        st.compute_persistence(homology_coeff_field=2, persistence_dim_max=True, min_persistence=-1)
        # Reading intervals does not recompute persistence with default coefficients.
        for dim in (0, 1):
            for birth, death in st.persistence_intervals_in_dimension(dim):
                censored = not math.isfinite(float(death))
                rows.append([dim, float(birth), float(death), int(censored)])
                curves[:, dim] += ((birth <= RADII) & (RADII < death)).astype(np.float32)
        curves /= len(points)
    return dict(curves=curves, intervals=np.asarray(rows, np.float64).reshape(-1, 4))


def non_ph_curves(points):
    from scipy.sparse.csgraph import shortest_path
    points = np.asarray(points, np.float64); n = len(points); result = np.zeros((31, 3), np.float32)
    if n <= 1:
        return result
    distance = np.linalg.norm(points[:, None] - points[None], axis=-1)
    for k, tau in enumerate(RADII):
        adj = (distance > 1e-8) & (distance <= tau); hops = shortest_path(adj, directed=False, unweighted=True)
        ids = np.triu(np.isfinite(hops), 1); positive = (hops > 0) & np.isfinite(hops)
        result[k, 0] = adj.sum() / (n * (n - 1))
        result[k, 1] = hops[ids].mean() / (n - 1) if ids.any() else 0.
        result[k, 2] = (1 / hops[positive]).sum() / (n * (n - 1))
    return result


def structure_fields(record):
    """Explicit allowlist: never access target/y/label properties of a record."""
    return SimpleNamespace(topology=record.topology, trimer=record.trimer, static=record.static,
                           normalized_smiles=record.normalized_smiles, final_geometry_status=record.final_geometry_status)


class StructureReader:
    def __init__(self, root):
        import lmdb
        self.root = Path(root); manifest = json.loads((self.root / 'direct_source.json').read_text())
        if manifest.get('conclusion') != 'PASS' or manifest.get('coordinate_generation_calls') != 0 or not (self.root / '.frozen').is_file():
            raise ValueError('frozen structure source')
        self.env = lmdb.open(str(self.root / 'data.lmdb'), readonly=True, lock=False, readahead=True)
    def get(self, key):
        from src.dataset.cache_lifecycle import deserialize_record
        with self.env.begin() as txn:
            payload = txn.get(bytes.fromhex(key))
        if payload is None:
            raise ValueError('structure missing; no fallback')
        return structure_fields(deserialize_record(bytes(payload), bytes.fromhex(key)))
    def close(self):
        self.env.close()


def fit_router(values, train_indices):
    x = np.asarray(values, np.float64); ids = np.asarray(train_indices, np.int64)
    if x.ndim != 3 or x.shape[1:] != (31, 5) or not len(ids) or not np.isfinite(x).all():
        raise ValueError('router train-only input')
    mean = x[ids].mean(0); scale = x[ids].std(0); scale[scale == 0] = 1.
    return dict(mean=mean.tolist(), scale=scale.tolist(), train_indices=ids.tolist(), fitted_on='train_only', ddof=0)


class PHBatch:
    def __init__(self, base, sides):
        self.base = base; self.sides = sides
    def __getattr__(self, name):
        return getattr(self.base, name)
    def to(self, device, non_blocking=False):
        def move(x):
            if torch.is_tensor(x):
                return x.to(device, non_blocking=non_blocking)
            if isinstance(x, dict):
                return {k: move(v) for k, v in x.items()}
            if isinstance(x, list):
                return [move(v) for v in x]
            return x
        return PHBatch(self.base.to(device, non_blocking=non_blocking), move(self.sides))


def collate(samples):
    from src.dataset.ua3m_paperfuse import collate as original_collate
    return PHBatch(original_collate(samples), [x['ph_side'] for x in samples])


class Data(torch.utils.data.Dataset):
    def __init__(self, task, fold, formal, source, split, cache, sidecar):
        from src.dataset.ua3m_paperfuse import Data as OriginalData
        self.base = OriginalData(task, fold, formal, source, split, cache)
        self.sidecar = Path(sidecar); self.task = task; self.fold = fold; self.memo = {}
        self.stats = json.loads((self.sidecar / 'scalers' / task / f'fold{fold}.json').read_text())
    def __getattr__(self, name):
        return getattr(self.base, name)
    @property
    def targets(self):
        return self.base.targets
    @targets.setter
    def targets(self, values):
        self.base.targets = values
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        row = self.indices[i]; key = str(self.keys[i]); token = (row, key)
        if token not in self.memo:
            side = torch.load(self.sidecar / self.task / f'row{row}.pt', map_location='cpu', weights_only=False)
            if side['task'] != self.task or side['row'] != row or side['key'] != key:
                raise ValueError('sidecar row/key identity')
            side['router'] = torch.as_tensor((np.asarray(side['router_raw'], np.float64) - np.asarray(self.stats['mean'])) / np.asarray(self.stats['scale']), dtype=torch.float32)
            self.memo[token] = side
        x = self.base[i]; x['ph_side'] = self.memo[token]; return x
    def close(self):
        self.base.close()


def loader(ds, ids, shuffle=False, seed=42):
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds, ids), batch_size=24, shuffle=shuffle,
        num_workers=0, drop_last=False, collate_fn=collate,
        generator=torch.Generator().manual_seed(seed) if shuffle else None)
