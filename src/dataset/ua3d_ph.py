"""Read-only structure sidecar for the fixed PH cycle.

The original modal cache, labels, and coordinates stay untouched. This module
materializes only topology and geometry fields and can reuse a structure when
its canonical mapping, physical mapping, and frozen coordinates agree.
"""
from pathlib import Path
import json
import math
import numpy as np
import torch

SCHEMA = "ua3d-ph-sidecar-r1"
A_FILTERS = (2, 4, 8, 16)
RADII = np.linspace(1.5, 4.5, 31, dtype=np.float64)
CUTOFFS = (2.0, 3.0, 4.0)


def _finite(value, name):
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise FloatingPointError(f"nonfinite {name}")
    return array


def lifted_graph(path_index, path_shift, path_mask, available):
    """Three-copy chemical graph recovered only from existing one-hop paths."""
    path = torch.as_tensor(path_index).long()
    shift = torch.as_tensor(path_shift).long()
    mask = torch.as_tensor(path_mask).bool()
    if path.ndim != 2 or path.shape[1] != 3 or path.shape != shift.shape or path.shape != mask.shape:
        raise ValueError("lifted path schema mismatch")
    if not bool(available):
        empty = np.zeros((0, 2), dtype=np.int64)
        return dict(nodes=0, edges=empty, center=np.zeros(0, dtype=np.int64), cross_ru=0, available=False)
    atoms = int(path.max()) + 1 if path.numel() else 0
    if atoms <= 0 or int(path[mask].min()) < 0:
        raise ValueError("invalid canonical atom in lifted path")
    edges = {}
    cross = 0
    for row in range(path.size(0)):
        if not bool(mask[row, 0] and mask[row, 1]) or bool(mask[row, 2]):
            continue  # exactly one chemical hop; two-hop attention is not a bond
        a, b = int(path[row, 0]), int(path[row, 1])
        sa, sb = int(shift[row, 0]), int(shift[row, 1])
        if a == b and sa == sb:
            raise ValueError("self loop in chemical graph")
        for offset in range(-1, 2):
            left, right = sa + offset, sb + offset
            if left not in (-1, 0, 1) or right not in (-1, 0, 1):
                continue
            u, v = a + (left + 1) * atoms, b + (right + 1) * atoms
            key = (min(u, v), max(u, v))
            # Opposite directed records describe one undirected physical edge.
            edges.setdefault(key, (a, b, sa, sb))
            if left != right:
                cross += 1
    ordered = sorted(edges)
    center = np.arange(atoms, 2 * atoms, dtype=np.int64)
    return dict(nodes=3 * atoms, canonical_atoms=atoms, edges=np.asarray(ordered, dtype=np.int64).reshape(-1, 2),
                center=center, cross_ru=cross, available=True)


def random_walk_pe(nodes, edges, steps=16):
    """CPU float64 return probabilities. Isolated nodes receive a PE-only self loop."""
    if nodes < 0 or steps != 16:
        raise ValueError("invalid RWPE shape")
    result = np.zeros((nodes, steps), dtype=np.float64)
    if nodes == 0:
        return result.astype(np.float32)
    adjacency = np.zeros((nodes, nodes), dtype=np.float64)
    if len(edges):
        adjacency[edges[:, 0], edges[:, 1]] = 1
        adjacency[edges[:, 1], edges[:, 0]] = 1
    degree = adjacency.sum(1)
    isolated = degree == 0
    adjacency[isolated, isolated] = 1
    degree = adjacency.sum(1)
    transition = adjacency / degree[:, None]
    state = transition.copy()
    for k in range(steps):
        result[:, k] = np.diag(state)
        if k + 1 < steps:
            state = state @ transition
    return _finite(result, "RWPE").astype(np.float32)


def _parent(birth):
    parent = np.arange(len(birth), dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    return find, parent


def graph_persistence(nodes, edges, filtration):
    values=np.asarray(filtration,dtype=np.float64)
    if values.shape!=(nodes,) or not np.isfinite(values).all(): raise ValueError('invalid node filtration')
    h0n=np.zeros((nodes,4)); h1n=np.zeros((nodes,4)); h0=[]; h1=[]
    if nodes==0:
        z=np.zeros((0,4),np.float32); return dict(h0=z,h1=z,h0_node=h0n.astype(np.float32),h1_node=h1n.astype(np.float32))
    parent=np.arange(nodes); members={i:[i] for i in range(nodes)}; birth={i:float(values[i]) for i in range(nodes)}
    cycles={i:0 for i in range(nodes)}; existing={i:0 for i in range(nodes)}
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return int(x)
    def consume(roots, added, time):
        roots=sorted(set(find(r) for r in roots))
        groups={}
        for root in roots: groups.setdefault(birth[root],[]).append(root)
        minimum=min(groups)
        for value,group in groups.items():
            bars=len(group)-(1 if value==minimum else 0)
            if not bars: continue
            participants=[node for root in group for node in members[root]]
            code=np.array([value,time,time-value,0.])
            h0n[participants]+=code*bars/len(participants); h0.extend([code]*bars)
        merged=[node for root in roots for node in members[root]]
        born=sum(existing[r] for r in roots)+added-len(merged)+1-sum(cycles[r] for r in roots)
        if born<0: raise RuntimeError('negative graph cycle count')
        if born:
            code=np.array([time,1.,1.-time,1.]); h1n[merged]+=code*born/len(merged); h1.extend([code]*int(born))
        keep=min(roots,key=lambda r:(birth[r],r))
        for root in roots:
            if root!=keep: parent[root]=keep
        members[keep]=merged; birth[keep]=minimum; cycles[keep]=sum(cycles[r] for r in roots)+born; existing[keep]=sum(existing[r] for r in roots)+added
    if len(edges):
        times=np.maximum(values[edges[:,0]],values[edges[:,1]]); order=np.argsort(times,kind='mergesort'); cursor=0
        while cursor<len(order):
            end=cursor+1
            while end<len(order) and times[order[end]]==times[order[cursor]]: end+=1
            pairs=[]; seen=set()
            for index in order[cursor:end]:
                a,b=int(edges[index,0]),int(edges[index,1]); key=(min(a,b),max(a,b))
                if key[0]!=key[1] and key not in seen: seen.add(key); pairs.append((find(a),find(b)))
            adjacency={}
            for a,b in pairs:
                adjacency.setdefault(a,set()).add(b); adjacency.setdefault(b,set()).add(a)
            visited=set()
            for start in list(adjacency):
                if start in visited: continue
                stack=[start]; cluster=[]
                while stack:
                    node=stack.pop()
                    if node in visited: continue
                    visited.add(node); cluster.append(node); stack.extend(adjacency.get(node,()))
                roots=set(cluster); added=sum(a in roots and b in roots for a,b in pairs)
                consume(cluster,added,float(times[order[cursor]]))
            cursor=end
    for root,nodes_here in list(members.items()):
        if find(root)!=root: continue
        code=np.array([birth[root],1.,1.-birth[root],1.]); h0n[nodes_here]+=code/len(nodes_here); h0.append(code)
    pack=lambda xs: np.asarray(xs,np.float32).reshape(-1,4) if xs else np.zeros((0,4),np.float32)
    return dict(h0=pack(h0),h1=pack(h1),h0_node=h0n.astype(np.float32),h1_node=h1n.astype(np.float32))

def vr_betti(points):
    """H0/H1 Betti curves of the 2-skeleton VR complex. H2 is intentionally discarded."""
    coords = np.asarray(points, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.isfinite(coords).all():
        raise ValueError("invalid physical coordinates")
    n = len(coords)
    curves = np.zeros((len(RADII), 2), dtype=np.float32)
    if n <= 1:
        return dict(curves=curves, intervals=np.zeros((0, 3), dtype=np.float32), censored=0)
    import gudhi
    if gudhi.__version__ != "3.8.0":
        raise RuntimeError("GUDHI 3.8.0 required")
    complex_ = gudhi.RipsComplex(points=coords, max_edge_length=4.5).create_simplex_tree(max_dimension=2)
    complex_.compute_persistence(homology_coeff_field=2, persistence_dim_max=True, min_persistence=-1)
    intervals = []
    censored = 0
    for dim, (birth, death) in complex_.persistence():
        if dim > 1:
            continue
        finite_death = math.isfinite(float(death))
        recorded_death = float(death) if finite_death else np.inf
        if not finite_death:
            censored += 1
        intervals.append((dim, float(birth), recorded_death, 0 if finite_death else 1))
        for i, radius in enumerate(RADII):
            if float(birth) <= radius < recorded_death:
                curves[i, dim] += 1
    curves /= max(n, 1)
    return dict(curves=curves, intervals=np.asarray(intervals, dtype=np.float32).reshape(-1, 4), censored=censored)


def non_ph_curves(points):
    """Edge density, normalized mean shortest path, and global efficiency at 31 radii."""
    coords = np.asarray(points, dtype=np.float64)
    n = len(coords)
    result = np.zeros((len(RADII), 3), dtype=np.float32)
    if n <= 1:
        return result
    distance = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    np.fill_diagonal(distance, np.inf)
    possible = n * (n - 1) / 2
    for i, radius in enumerate(RADII):
        adjacency = distance <= radius
        edges = int(np.count_nonzero(np.triu(adjacency, 1)))
        result[i, 0] = edges / possible
        hops = np.full((n, n), np.inf)
        np.fill_diagonal(hops, 0)
        hops[adjacency] = 1
        for k in range(n):
            hops = np.minimum(hops, hops[:, k:k + 1] + hops[k:k + 1])
        finite = np.triu(np.isfinite(hops), 1)
        if finite.any():
            result[i, 1] = np.mean(hops[finite]) / max(n - 1, 1)
        inverse = np.zeros_like(hops)
        positive = hops > 0
        inverse[positive & np.isfinite(hops)] = 1 / hops[positive & np.isfinite(hops)]
        result[i, 2] = inverse.sum() / (n * (n - 1))
    return result


def validate_coordinate_identity(cache_geometry, direct_record):
    """Require canonical, physical endpoint, and frozen coordinate agreement before reuse."""
    if cache_geometry is None or direct_record.final_geometry_status != "PASS":
        return False
    geometry = cache_geometry
    if isinstance(geometry, dict):
        valid = bool(geometry.get("geometry_valid"))
        distance = np.asarray(geometry["bond_distance"], dtype=np.float64)
        ends = np.stack([np.asarray(geometry["bond_canonical_a"]), np.asarray(geometry["bond_canonical_b"])], 0)
    else:
        valid = bool(geometry.geometry_valid)
        distance = np.asarray(geometry.bond_distance, dtype=np.float64)
        ends = np.stack([np.asarray(geometry.bond_canonical_a), np.asarray(geometry.bond_canonical_b)], 0)
    if not valid:
        return False
    static = direct_record.static
    if not bool(static["geometry_valid"]):
        raise ValueError("PASS status disagrees with frozen static geometry")
    token_a = np.asarray(static["token_pos_index_a"])
    token_b = np.asarray(static["token_pos_index_b"])
    if len(token_a) != len(distance):
        raise ValueError("physical bond count disagrees with cached geometry")
    positions = np.asarray(direct_record.trimer.trimer_pos, dtype=np.float64)
    if not np.isfinite(positions).all():
        raise FloatingPointError("nonfinite frozen coordinates")
    actual = np.linalg.norm(positions[token_a] - positions[token_b], axis=1)
    if not np.allclose(actual, distance, rtol=0, atol=1e-5):
        raise ValueError("frozen bond length disagrees with cached geometry")
    topology = direct_record.topology
    canonical = np.asarray(topology.canonical_to_trimer_base_atom_id)
    mapped = np.asarray(direct_record.trimer.mips_to_trimer_central_index)
    if len(canonical) != len(mapped) or len(np.unique(mapped)) != len(mapped):
        raise ValueError("canonical/physical mapping is not complete and injective")
    elements = np.asarray(direct_record.trimer.trimer_atomic_number)
    if not np.array_equal(elements[mapped], np.asarray(topology.z)):
        raise ValueError("canonical element mapping mismatch")
    if np.any((ends < 0) | (ends >= len(canonical))):
        raise ValueError("cached bond endpoint leaves canonical atoms")
    return True
