"""Finite physical-bond exact shells and clean spatial PH; no cache mutation."""
from collections import defaultdict, deque
import numpy as np
import torch

from .glt_dual import build_dual_sample, dual_glt_collate
from .mcl_ph_view import five_descriptors, PH_SEMANTICS

SCHEMA = 'ua-hop-ph-input-r1'
SEMANTICS = dict(schema=SCHEMA, hops=[1, 2, 3, 4], graph='physical_heavy_bond_line',
    shells='exact_shortest_distance_no_self', view='clean_open_trimer',
    rbf_centers=[4*j/15 for j in range(16)], rbf_sigma=4/15,
    rbf_formula='exp(-(r-c)^2/(2*sigma^2));unclipped',
    reduction='path_mean;all_shortest_paths_mean;bias_mlp',
    incidence='central_heavy_physical_endpoints', mask='after_W_out_including_bias',
    ph=dict(PH_SEMANTICS))


def exact_shells(ends, positions):
    """Return directed shells and exact averages using a shortest-path DAG.

    Endpoint rows are original physical coordinate rows, never canonical IDs.
    Dynamic sums count every shortest path, not every predecessor equally.
    """
    ends = np.asarray(ends, dtype=np.int64).reshape(-1, 2)
    pos = np.asarray(positions, dtype=np.float64)
    n = len(ends)
    if n == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty((0, 18))
    if pos.ndim != 2 or pos.shape[1] != 3 or not np.isfinite(pos).all():
        raise ValueError('nonfinite or malformed physical coordinates')
    if ends.min() < 0 or ends.max() >= len(pos) or (ends[:, 0] == ends[:, 1]).any():
        raise ValueError('invalid physical bond endpoint')
    if len({tuple(sorted(e)) for e in ends.tolist()}) != n:
        raise ValueError('duplicated physical bond token')
    at = defaultdict(list)
    for b, (a, c) in enumerate(ends):
        at[int(a)].append(b); at[int(c)].append(b)
    adj = [set() for _ in range(n)]
    angles = {}
    for center, bonds in at.items():
        for s in bonds:
            for t in bonds:
                if s == t:
                    continue
                outer_s = int(ends[s, 1] if ends[s, 0] == center else ends[s, 0])
                outer_t = int(ends[t, 1] if ends[t, 0] == center else ends[t, 0])
                a, b = pos[outer_s]-pos[center], pos[outer_t]-pos[center]
                denom = np.linalg.norm(a)*np.linalg.norm(b)
                if not np.isfinite(denom) or denom <= 0:
                    raise ValueError('degenerate physical angle')
                c = float(np.clip(np.dot(a, b)/denom, -1, 1))
                if (s, t) in angles:
                    raise ValueError('physical bonds share more than one endpoint')
                adj[s].add(t); angles[s, t] = np.array([c, 2*c*c-1])
    length = np.linalg.norm(pos[ends[:, 0]]-pos[ends[:, 1]], axis=1)
    if not np.isfinite(length).all() or (length <= 0).any():
        raise ValueError('invalid physical bond distance')
    centers = np.arange(16, dtype=np.float64)*4/15
    rbf = np.exp(-0.5*((length[:, None]-centers)/(4/15))**2)
    edges, hops, features = [], [], []
    for root in range(n):
        dist = {root: 0}; levels = [[root]]
        for depth in range(1, 5):
            nxt = sorted({t for s in levels[-1] for t in adj[s] if t not in dist})
            for t in nxt: dist[t] = depth
            levels.append(nxt)
        ways = {root: 1}; radial = {root: rbf[root]}; angular = {root: np.zeros(2)}
        for depth in range(1, 5):
            for t in levels[depth]:
                pred = sorted(s for s in adj[t] if dist.get(s) == depth-1)
                ways[t] = sum(ways[s] for s in pred)
                radial[t] = sum((radial[s]+ways[s]*rbf[t] for s in pred), np.zeros(16))
                angular[t] = sum((angular[s]+ways[s]*angles[s, t] for s in pred), np.zeros(2))
                feature = np.r_[radial[t]/ways[t]/(depth+1), angular[t]/ways[t]/depth]
                edges.append((root, t)); hops.append(depth); features.append(feature)
    feat = np.asarray(features, dtype=np.float32).reshape(-1, 18)
    if not np.isfinite(feat).all(): raise ValueError('nonfinite path geometry')
    return (torch.tensor(edges, dtype=torch.long).reshape(-1, 2).t().contiguous(),
            torch.tensor(hops, dtype=torch.long), torch.from_numpy(feat))


def periodic_coverage(physical_keys, roots, templates):
    """Topology-only infinite lift, bounded by four line steps; no coordinates."""
    finite = set(physical_keys)
    atom_adj = defaultdict(set)
    for a, b, shift in templates:
        atom_adj[a].add((b, shift)); atom_adj[b].add((a, -shift))
    complete = [0]*4
    detail = {}
    for root in sorted(set(roots)):
        key = physical_keys[root]
        seen = {key}; frontier = {key}; okay = True; flags = []
        for depth in range(4):
            nxt = set()
            for bond in frontier:
                for atom, image in bond:
                    for other, shift in atom_adj[atom]:
                        edge = tuple(sorted(((atom, image), (other, image+shift))))
                        if edge not in seen: nxt.add(edge)
            if not nxt.issubset(finite): okay = False
            seen.update(nxt); frontier = nxt; flags.append(okay)
            complete[depth] += int(okay)
        detail[root] = flags
    return complete, detail


def prepare_sample(topology, trimer, smiles, static):
    data = build_dual_sample(topology, trimer, smiles, static=static, geometry_mode='finite_incident')
    data.hop_input_schema = SCHEMA
    coverage = dict(geometry_valid=bool(data.geometry_valid), roots=0, complete=[0]*4,
                    empty_shells=[0]*4, truncated_roots=[0]*4, affected_atoms=[0]*4,
                    cross_image_relations=[0]*4, image_range=None)
    if data.geometry_valid:
        a = np.asarray(static['token_pos_index_a'], dtype=np.int64)
        b = np.asarray(static['token_pos_index_b'], dtype=np.int64)
        ends = np.stack([a, b], 1)
        edge, hop, feature = exact_shells(ends, trimer.trimer_pos)
        mapping = torch.as_tensor(topology.canonical_to_trimer_base_atom_id).long().tolist()
        inverse = {base: c for c, base in enumerate(mapping)}
        base = torch.as_tensor(trimer.trimer_base_ru_atom_id).long().tolist()
        image = torch.as_tensor(trimer.trimer_ru_offset).long().tolist()
        keys = [tuple(sorted(((inverse[base[x]], image[x]), (inverse[base[y]], image[y])))) for x, y in ends]
        z = torch.as_tensor(topology.z).long()
        # Frozen lifted *atomic* SPD=1 paths carry actual relative image shifts.
        # They are distinct from the mixed physical-bond line relations.
        paths = torch.as_tensor(topology.lga_path_index).long()
        shifts = torch.as_tensor(topology.lga_path_shift).long()
        masks = torch.as_tensor(topology.lga_path_mask).bool()
        spd = torch.as_tensor(topology.lga_spd).long()
        templates = set()
        for path, shift, mask in zip(paths[spd == 1], shifts[spd == 1], masks[spd == 1]):
            if not (mask[0] and mask[1]) or int(mask.sum()) != 2:
                raise ValueError('periodic atomic SPD1 path has invalid endpoints')
            x, y = int(path[0]), int(path[1])
            if z[x] > 1 and z[y] > 1:
                templates.add((x, y, int(shift[1]-shift[0])))
        # Every observed physical bond must be an instance of a periodic template.
        normalized = lambda u, v, q: min((u, v, q), (v, u, -q))
        template_set = {normalized(x, y, s) for x, y, s in templates}
        for (x, q), (y, r) in keys:
            if normalized(x, y, r-q) not in template_set:
                raise ValueError('physical bond absent from periodic chemical templates')
        roots = data.incidence_index[0].unique().tolist()
        complete, detail = periodic_coverage(keys, roots, templates)
        coverage.update(roots=len(roots), complete=complete,
                        truncated_roots=[len(roots)-v for v in complete],
                        image_range=[min(image), max(image)])
        for k in range(1, 5):
            chosen = edge[:, hop == k]
            targets = set(chosen[1].tolist())
            coverage['empty_shells'][k-1] = sum(r not in targets for r in roots)
            affected = {int(atom) for r, atom in data.incidence_index.t().tolist() if not detail[r][k-1]}
            coverage['affected_atoms'][k-1] = len(affected)
            coverage['cross_image_relations'][k-1] = sum(
                (keys[s][0][1], keys[s][1][1]) != (keys[t][0][1], keys[t][1][1])
                for s, t in chosen.t().tolist())
        heavy = torch.as_tensor(trimer.trimer_atomic_number).long() > 1
        cloud = torch.as_tensor(trimer.trimer_pos)[heavy].numpy()
        ph = torch.from_numpy(five_descriptors(cloud).astype(np.float32))
    else:
        edge, hop, feature = exact_shells([], np.empty((0, 3)))
        ph = torch.zeros(31, 5)
    data.hop_edge_index, data.hop_distance, data.hop_features = edge, hop, feature
    data.hop_ph = ph
    if 'y' in data: del data.y  # task labels live only in task/fold readers
    return data, coverage


def collate(samples):
    if not samples or any(s.hop_input_schema != SCHEMA for s in samples):
        raise ValueError('mixed or missing hop input schema')
    batch = dual_glt_collate(samples)
    offsets = np.cumsum([0]+[s.bond_distance.numel() for s in samples[:-1]])
    batch.hop_edge_index = torch.cat([s.hop_edge_index+int(o) for s, o in zip(samples, offsets)], 1)
    batch.hop_distance = torch.cat([s.hop_distance for s in samples])
    batch.hop_features = torch.cat([s.hop_features for s in samples])
    batch.hop_ph = torch.stack([s.hop_ph for s in samples])
    batch.hop_input_schema = SCHEMA
    if all('y' in s for s in samples): batch.y = torch.cat([s.y for s in samples])
    if not torch.isfinite(batch.hop_features).all() or not torch.isfinite(batch.hop_ph).all():
        raise ValueError('nonfinite hop/PH batch')
    return batch
