"""On-demand atom/geometry and optional FGR/ALIGN targets; no FP labels."""
import copy
import hashlib
import math
from collections import deque
from functools import lru_cache

import numpy as np

import torch
from rdkit import Chem
from rdkit.Chem import BRICS

from .graph_data import build_periodic_multimer_mol
from .glt_bond_chemistry import bond_type_index
from .glt_dual import bond_paths, build_dual_sample, dual_glt_collate
from .canonical_periodic import resolve_normalized_identity


def sample_generator(seed, key, position):
    payload = f'{seed}:{key}:{position}'.encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % (2**63 - 1)
    return torch.Generator().manual_seed(value)


@lru_cache(maxsize=512)
def chemical_groups(smiles):
    """Build the deterministic 3-RU BRICS partition for atom masking."""
    source = Chem.MolFromSmiles(str(smiles))
    if source is None:
        raise ValueError("invalid P-SMILES for chemical groups")
    # BRICS groups are model-side identities. The
    # input spelling is a provenance view and may use a different RDKit atom
    # order, so canonicalize before extracting any integer atom ids.
    normalized = Chem.MolToSmiles(source, canonical=True)
    mol, meta = build_periodic_multimer_mol(
        normalized, 3, close_periodic=False
    )
    size = meta['base_atom_count']
    cut = {frozenset(pair) for pair, _ in BRICS.FindBRICSBonds(mol)}
    unseen, groups = set(range(mol.GetNumAtoms())), []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        queue, component = [root], []
        while queue:
            atom = queue.pop()
            component.append(atom)
            for neighbor in mol.GetAtomWithIdx(atom).GetNeighbors():
                other = neighbor.GetIdx()
                if other in unseen and frozenset((atom, other)) not in cut:
                    unseen.remove(other)
                    queue.append(other)
        center = sorted(i - size for i in component if size <= i < 2 * size)
        if center:
            groups.append(tuple(center))
    return tuple(sorted(groups))


def _fgr_canonical_mapping(topology, trimer, identity, molecule, metadata):
    """Validate normalized graph, canonical atoms and central coordinates."""
    base_count = int(metadata.get("base_atom_count", -1))
    units = metadata.get("unit_atoms") or []
    if len(units) != 3 or any(len(unit) != base_count for unit in units):
        raise ValueError("FGR true-Trimer unit mapping is incomplete")
    canonical_to_base = torch.as_tensor(
        identity.get("canonical_to_normalized_base"), dtype=torch.long
    ).reshape(-1)
    canonical_count = int(topology.mips_x.size(0))
    if (canonical_to_base.numel() != canonical_count
            or sorted(canonical_to_base.tolist()) != list(range(base_count))):
        raise ValueError("FGR canonical-to-normalized mapping is not a permutation")
    topology_mapping = torch.as_tensor(
        getattr(topology, "canonical_to_trimer_base_atom_id", canonical_to_base),
        dtype=torch.long,
    ).reshape(-1)
    if not torch.equal(topology_mapping, canonical_to_base):
        raise ValueError("FGR cached canonical-to-Trimer mapping disagrees with normalized identity")
    topology_z = torch.as_tensor(
        getattr(topology, "atomic_numbers", getattr(topology, "z", torch.empty(0))),
        dtype=torch.long,
    ).reshape(-1)
    if topology_z.numel() != canonical_count:
        raise ValueError("FGR canonical element table length mismatch")
    central_atoms = torch.as_tensor(units[1], dtype=torch.long)
    normalized_z = torch.tensor(
        [int(molecule.GetAtomWithIdx(int(atom)).GetAtomicNum()) for atom in central_atoms],
        dtype=torch.long,
    )
    if not torch.equal(normalized_z[canonical_to_base], topology_z):
        raise ValueError("FGR canonical/normalized atomic-number mapping mismatch")
    central_mapping = torch.as_tensor(
        getattr(trimer, "mips_to_trimer_central_index", torch.empty(0)),
        dtype=torch.long,
    ).reshape(-1)
    positions = torch.as_tensor(
        getattr(trimer, "trimer_pos", torch.empty((0, 3))), dtype=torch.float32
    )
    if (central_mapping.numel() != canonical_count
            or central_mapping.unique().numel() != canonical_count
            or positions.ndim != 2 or positions.size(1) != 3
            or (central_mapping.numel() and (
                int(central_mapping.min()) < 0
                or int(central_mapping.max()) >= positions.size(0)))):
        raise ValueError("FGR canonical-to-central coordinate mapping is invalid")
    central_mask = getattr(trimer, "trimer_central_ru_mask", None)
    if central_mask is not None:
        central_mask = torch.as_tensor(central_mask, dtype=torch.bool).reshape(-1)
        if (central_mask.numel() != positions.size(0)
                or not bool(central_mask[central_mapping].all())):
            raise ValueError("FGR canonical mapping leaves the central RU")
    trimer_z = torch.as_tensor(
        getattr(trimer, "trimer_atomic_number", torch.empty(0)), dtype=torch.long
    ).reshape(-1)
    if (trimer_z.numel() != positions.size(0)
            or not torch.equal(trimer_z[central_mapping], topology_z)):
        raise ValueError("FGR central coordinate atomic-number mapping mismatch")
    inverse = torch.full((base_count,), -1, dtype=torch.long)
    inverse[canonical_to_base] = torch.arange(canonical_count, dtype=torch.long)
    if bool((inverse < 0).any()):
        raise ValueError("FGR normalized-to-canonical mapping is incomplete")
    return central_atoms, inverse, central_mapping, positions


def fgr_true_trimer_candidates(topology, trimer, smiles=None, *, identity=None):
    """Return every canonical SPD2/SPD3 candidate and its clean distance.

    Candidate identity is defined only by RDKit shortest paths in the open
    chemical Trimer.  Sampling and normalization remain in :func:`fgr_pairs`;
    this pure helper is used by the full P_train statistics pass so that the
    fitted moments cannot depend on a random 32-pair subsample.
    """
    if not bool(getattr(trimer, 'trimer_geometry_valid', False)):
        return (torch.empty((0, 2), dtype=torch.long),
                torch.empty(0), torch.empty(0, dtype=torch.long))
    if identity is None:
        if smiles is None:
            raise ValueError('FGR requires normalized identity provenance')
        identity = resolve_normalized_identity(topology, smiles, require_fields=True)
    molecule, metadata = build_periodic_multimer_mol(
        str(identity["normalized_smiles"]), num_repeat_units=3, close_periodic=False
    )
    central_atoms, inverse, mapping, positions = _fgr_canonical_mapping(
        topology, trimer, identity, molecule, metadata
    )
    base_index = {int(atom): index for index, atom in enumerate(central_atoms.tolist())}
    heavy_atoms = [
        int(atom) for atom in central_atoms.tolist()
        if int(molecule.GetAtomWithIdx(int(atom)).GetAtomicNum()) > 1
    ]
    candidates = {}
    for left_offset, left in enumerate(heavy_atoms):
        for right in heavy_atoms[left_offset + 1:]:
            path = tuple(int(value) for value in Chem.GetShortestPath(molecule, left, right))
            hop = len(path) - 1
            if hop not in (2, 3):
                continue
            left_canonical = int(inverse[base_index[left]])
            right_canonical = int(inverse[base_index[right]])
            if left_canonical == right_canonical:
                raise ValueError('FGR true-Trimer candidate produced a self pair')
            pair = (min(left_canonical, right_canonical), max(left_canonical, right_canonical))
            if pair in candidates and candidates[pair] != hop:
                raise ValueError('FGR shortest-path identity is ambiguous')
            candidates[pair] = int(hop)
    ordered = sorted(candidates, key=lambda pair: (candidates[pair], pair[0], pair[1]))
    if not ordered:
        return (torch.empty((0, 2), dtype=torch.long),
                torch.empty(0), torch.empty(0, dtype=torch.long))
    pair_tensor = torch.as_tensor(ordered, dtype=torch.long)
    mapped = mapping[pair_tensor]
    if int(mapped.max()) >= positions.size(0) or int(mapped.min()) < 0:
        raise ValueError('FGR mapped coordinate index is out of range')
    distances = torch.linalg.vector_norm(positions[mapped[:, 0]] - positions[mapped[:, 1]], dim=-1)
    if not bool(torch.isfinite(distances).all()) or bool((distances <= 0).any()):
        raise ValueError('FGR clean pair distance is invalid')
    return pair_tensor, distances, torch.as_tensor(
        [candidates[pair] for pair in ordered], dtype=torch.long)


def fgr_pairs(topology, trimer, smiles=None, *, identity=None, seed, key, position,
              mu=0.0, sigma=1.0, max_pairs=32, generator=None):
    """Return deterministic sampled, normalized centre-RU SPD2/3 pairs."""
    if not math.isfinite(float(mu)) or not math.isfinite(float(sigma)) or sigma <= 0:
        raise ValueError('FGR normalization requires finite mu and positive sigma')
    if int(max_pairs) <= 0:
        raise ValueError('FGR max_pairs must be positive')
    pair_tensor, distances, spd = fgr_true_trimer_candidates(
        topology, trimer, smiles, identity=identity)
    if not pair_tensor.numel():
        return pair_tensor, distances, spd
    if generator is None:
        generator = sample_generator(seed, f'{key}:fgr', position)
    chosen_indices = []
    remaining = int(max_pairs)
    for hop in (2, 3):
        if remaining <= 0:
            break
        class_indices = torch.where(spd == hop)[0]
        cap = min(16, remaining, int(class_indices.numel()))
        if int(class_indices.numel()) > cap:
            selected = torch.randperm(int(class_indices.numel()), generator=generator)[:cap]
            class_indices = class_indices[selected.sort().values]
        chosen_indices.extend(class_indices.tolist())
        remaining -= int(class_indices.numel())
    chosen_indices = sorted(chosen_indices, key=lambda index: (
        int(spd[index]), int(pair_tensor[index, 0]), int(pair_tensor[index, 1])))
    selected = torch.as_tensor(chosen_indices, dtype=torch.long)
    pair_tensor = pair_tensor[selected]
    distances = distances[selected]
    spd = spd[selected]
    normalized = (torch.log1p(distances) - float(mu)) / float(sigma)
    return pair_tensor, normalized, spd


def motif_mask(topology, groups, generator, ratio=0.3):
    count = topology.mips_x.size(0)
    mapping = torch.as_tensor(
        topology.canonical_to_trimer_base_atom_id, dtype=torch.long
    ).reshape(-1).tolist()
    if len(mapping) != count or sorted(mapping) != list(range(count)):
        raise ValueError("canonical-to-Trimer atom mapping is not a permutation")
    mask = torch.zeros(count, dtype=torch.bool)
    if count <= 1:
        return mask, False
    inverse = {base: canonical for canonical, base in enumerate(mapping)}
    canonical = [[inverse[i] for i in group] for group in groups]
    if sorted(i for group in canonical for i in group) != list(range(count)):
        raise ValueError('BRICS center groups must partition canonical atoms')
    target = min(math.ceil(ratio * count), count - 1)
    fallback = len(canonical) == 1
    if not fallback:
        for group_id in torch.randperm(len(canonical), generator=generator).tolist():
            group = canonical[group_id]
            if int(mask.sum()) + len(group) >= count:
                continue
            mask[group] = True
            if int(mask.sum()) >= target:
                break
        return mask, False
    adjacency = [set() for _ in range(count)]
    for relation in torch.where(topology.lga_spd == 1)[0].tolist():
        a, b = topology.lga_edge_index[:, relation].tolist()
        if a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)
    root = int(torch.randint(count, (1,), generator=generator))
    queue, seen = deque([root]), {root}
    while queue and int(mask.sum()) < target:
        atom = queue.popleft()
        mask[atom] = True
        neighbors = sorted(adjacency[atom] - seen)
        for k in torch.randperm(len(neighbors), generator=generator).tolist():
            other = neighbors[k]
            seen.add(other)
            queue.append(other)
    if int(mask.sum()) != target:
        raise ValueError('canonical topology cannot supply connected fallback mask')
    return mask, True


def _topology_z(topology):
    """Canonical element table of one frozen topology record."""
    z = getattr(topology, 'atomic_numbers', None)
    if z is None:
        z = getattr(topology, 'z', None)
    if z is None:
        raise ValueError('frozen topology is missing the canonical element table')
    return torch.as_tensor(z, dtype=torch.long).reshape(-1)


def periodic_bond_table(normalized_smiles, canonical_to_base):
    """Canonical periodic chemical bonds of one sample.

    One entry ``(a, b, bond_type, shift)`` per distinct periodic edge of the
    quotient graph: center-internal bonds (shift 0) and both seam directions
    (shift -1 / +1).  Endpoints are canonical atom indices; the table is built
    from the frozen normalized SMILES only, never from atom array positions.
    """
    molecule, metadata = build_periodic_multimer_mol(
        str(normalized_smiles), num_repeat_units=3, close_periodic=False
    )
    size = int(metadata['base_atom_count'])
    mapping = torch.as_tensor(canonical_to_base, dtype=torch.long).reshape(-1)
    if mapping.numel() != size or sorted(mapping.tolist()) != list(range(size)):
        raise ValueError('canonical-to-base mapping is not a base permutation')
    inverse = torch.full((size,), -1, dtype=torch.long)
    inverse[mapping] = torch.arange(mapping.numel(), dtype=torch.long)
    table = set()
    for bond in molecule.GetBonds():
        a, b = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
        qa, qb = a // size, b // size
        if qa == qb and qa != 1:
            continue
        if qa == qb == 1:
            shift = 0
        elif {qa, qb} == {0, 1}:
            shift = -1
        elif {qa, qb} == {1, 2}:
            shift = +1
        else:
            raise ValueError('periodic bond leaves the three-unit chain')
        ca, cb = int(inverse[a % size]), int(inverse[b % size])
        if ca > cb:
            ca, cb, shift = cb, ca, -shift
        table.add((ca, cb, int(bond_type_index(bond)), shift))
    return sorted(table)


def environment_atom_labels(bond_table, atomic_numbers, vocab):
    """Periodic rooted radius-1 atom-environment class per canonical atom.

    The environment of atom ``i`` is its element plus the unordered multiset of
    ``(neighbor element, bond type)`` over every real periodic chemical bond of
    ``i``.  Categories missing from the frozen P_train vocabulary map to UNK.
    """
    z = torch.as_tensor(atomic_numbers, dtype=torch.long).reshape(-1).tolist()
    neighbors = {}
    for ca, cb, bond_type, _ in bond_table:
        neighbors.setdefault(ca, []).append((z[cb], bond_type))
        neighbors.setdefault(cb, []).append((z[ca], bond_type))
    unk = vocab.get('<unk>')
    if unk is None:
        raise ValueError('environment vocabulary is missing the <unk> entry')
    labels = torch.full((len(z),), int(unk), dtype=torch.long)
    for atom, pairs in neighbors.items():
        category = f'{z[atom]}|' + ';'.join(f'{element}:{bond}'
                                            for element, bond in sorted(pairs))
        labels[atom] = int(vocab.get(category, unk))
    return labels


def torsion_quadruplets(bond_table, atomic_numbers):
    """Every heavy-atom torsion a-b-c-d on center-RU internal bonds.

    All four atoms are distinct center-RU heavy atoms and the three relations
    are real chemical bonds; the middle bond is identified by its canonical
    (a, b) pair.  A path and its reverse describe one physical torsion, so only
    one orientation is recorded per (middle bond, endpoint pair).
    """
    z = torch.as_tensor(atomic_numbers, dtype=torch.long).reshape(-1).tolist()
    adjacency = {}
    for ca, cb, bond_type, shift in bond_table:
        if shift != 0:
            continue
        adjacency.setdefault(ca, []).append((cb, bond_type))
        adjacency.setdefault(cb, []).append((ca, bond_type))
    quadruplets = []
    for bond in sorted(key for key in bond_table if key[3] == 0):
        b, c = bond[0], bond[1]
        if z[b] <= 1 or z[c] <= 1:
            continue
        left = [a for a, _ in adjacency.get(b, []) if a != c and z[a] > 1]
        right = [d for d, _ in adjacency.get(c, []) if d != b and z[d] > 1]
        for a in left:
            for d in right:
                if a == d:
                    continue
                quadruplets.append((a, b, c, d, (b, c)))
    return quadruplets


def torsion_features(positions, mapping, quadruplets, *, zero=False):
    """[cos(phi), cos(2*phi)] from the given (noisy or clean) coordinates.

    Degenerate (near-collinear) quadruplets are masked instead of being forced
    through an unstable inverse cosine.
    """
    positions = torch.as_tensor(positions, dtype=torch.float32)
    mapping = torch.as_tensor(mapping, dtype=torch.long).reshape(-1)
    values = torch.zeros((len(quadruplets), 2), dtype=torch.float32)
    mask = torch.zeros((len(quadruplets),), dtype=torch.bool)
    for index, (a, b, c, d, _) in enumerate(quadruplets):
        p0, p1 = positions[mapping[a]], positions[mapping[b]]
        p2, p3 = positions[mapping[c]], positions[mapping[d]]
        b0 = p1 - p0
        b1 = p2 - p1
        b2 = p3 - p2
        norm_b1 = torch.linalg.vector_norm(b1)
        if not bool(torch.isfinite(norm_b1)) or float(norm_b1) <= 0:
            continue
        b1 = b1 / norm_b1
        v = b0 - torch.dot(b0, b1) * b1
        w = b2 - torch.dot(b2, b1) * b1
        if (not bool(torch.isfinite(v).all()) or not bool(torch.isfinite(w).all())):
            continue
        if float(torch.linalg.vector_norm(v)) < 1e-3 or float(torch.linalg.vector_norm(w)) < 1e-3:
            continue
        y = torch.dot(torch.cross(b1, v, dim=0), w)
        x = torch.dot(v, w)
        phi = torch.atan2(y, x)
        cos = torch.cos(phi)
        values[index, 0] = cos
        values[index, 1] = 2.0 * cos * cos - 1.0
        mask[index] = True
    if zero:
        values = torch.zeros_like(values)
    return values, mask


def _torsion_fields(static, atomic_numbers, positions, mapping, mode, bond_table):
    """Shared data path for S4_TOR_ON (real values) and S4_TOR_OFF (zeros).

    Quadruplet identity comes from the canonical periodic bond table; middle
    bonds are mapped to frozen 3D bond-token rows through the stored center
    row index, with an element/bond-type cross-check against the static row.
    """
    center_table = sorted((key[0], key[1], key[2]) for key in bond_table
                          if key[3] == 0 and atomic_numbers[key[0]] > 1
                          and atomic_numbers[key[1]] > 1)
    center_rows = np.asarray(static['distance_token_index'], dtype=np.int64)
    if len(center_table) != len(center_rows):
        raise ValueError('center bond table disagrees with frozen token rows')
    lookup = {}
    for position, (ca, cb, bond_type) in enumerate(center_table):
        row = int(center_rows[position])
        z = atomic_numbers
        if z[ca] != int(static['token_z_a'][row]) or z[cb] != int(static['token_z_b'][row]) \
                or bond_type != int(static['token_bond_type'][row]):
            raise ValueError('torsion middle bond does not match the frozen token row')
        lookup[(ca, cb)] = row
    quadruplets = torsion_quadruplets(bond_table, atomic_numbers)
    index = torch.as_tensor([lookup[middle] for _, _, _, _, middle in quadruplets],
                            dtype=torch.long)
    features = [(a, b, c, d, lookup[middle]) for a, b, c, d, middle in quadruplets]
    values, mask = torsion_features(positions, mapping, features,
                                    zero=bool(mode == 'off'))
    return index, values, mask


def physical_torsion_fields(static, trimer, positions):
    """Deterministic paths over frozen physical bonds; no canonical/image merge."""
    if static is None:raise ValueError('physical torsion requires frozen static')
    positions=torch.as_tensor(positions,dtype=torch.float32)
    z=torch.as_tensor(trimer.trimer_atomic_number,dtype=torch.long).reshape(-1)
    if positions.shape!=(len(z),3):raise ValueError('physical torsion position shape mismatch')
    pa=np.asarray(static['token_pos_index_a']);pb=np.asarray(static['token_pos_index_b'])
    za=np.asarray(static['token_z_a']);zb=np.asarray(static['token_z_b'])
    if any(x.ndim!=1 or x.dtype.kind not in 'iu' for x in (pa,pb,za,zb)) or not(len(pa)==len(pb)==len(za)==len(zb)):
        raise ValueError('physical torsion endpoint schema mismatch')
    adjacency={};bonds={}
    for row,(a,b) in enumerate(zip(pa.tolist(),pb.tolist())):
        if a==b or min(a,b)<0 or max(a,b)>=len(z):raise ValueError('physical torsion endpoint bounds')
        if int(z[a])!=int(za[row]) or int(z[b])!=int(zb[row]):raise ValueError('physical torsion element identity mismatch')
        key=tuple(sorted((a,b)))
        if key in bonds:raise ValueError('duplicate frozen physical bond')
        bonds[key]=row
        if int(z[a])>1 and int(z[b])>1:
            adjacency.setdefault(a,set()).add(b);adjacency.setdefault(b,set()).add(a)
    quads=[];rows=[];seams=[]
    offset=torch.as_tensor(trimer.trimer_ru_offset).reshape(-1)
    base=torch.as_tensor(trimer.trimer_base_ru_atom_id).reshape(-1)
    if len(offset)!=len(z) or len(base)!=len(z):raise ValueError('physical torsion image identity mismatch')
    for a in adjacency:
        if int(offset[a]) not in (-1,0,1) or int(base[a])<0:raise ValueError('invalid physical torsion RU identity')
    seen=set()
    for (b,c),row in bonds.items():
        if b not in adjacency or c not in adjacency:continue
        for a in sorted(adjacency[b]-{c}):
            for d in sorted(adjacency[c]-{b}):
                quad=(a,b,c,d)
                if len(set(quad))!=4:continue
                key=min(quad,quad[::-1])
                if key in seen:continue
                seen.add(key);quads.append(quad);rows.append(row)
                seams.append(any(int(offset[x])!=int(offset[y]) for x,y in zip(quad,quad[1:])))
    index=torch.tensor(rows,dtype=torch.long)
    if not quads:
        values=torch.zeros((0,2));valid=torch.zeros(0,dtype=torch.bool)
    else:
        coords=positions[torch.tensor(quads,dtype=torch.long)]
        if not torch.isfinite(coords).all():raise ValueError('nonfinite valid physical torsion coordinates')
        b0=coords[:,0]-coords[:,1];b1=coords[:,2]-coords[:,1];b2=coords[:,3]-coords[:,2]
        length=torch.linalg.vector_norm(b1,dim=1)
        u=b1/length.clamp_min(1e-8).unsqueeze(1)
        v=b0-(b0*u).sum(1,keepdim=True)*u;w=b2-(b2*u).sum(1,keepdim=True)*u
        valid=(length>1e-8)&(torch.linalg.vector_norm(v,dim=1)>=1e-3)&(torch.linalg.vector_norm(w,dim=1)>=1e-3)
        phi=torch.atan2((torch.cross(u,v,dim=1)*w).sum(1),(v*w).sum(1))
        cosine=phi.cos();values=torch.stack((cosine,2*cosine.square()-1),dim=1)
        values=torch.where(valid.unsqueeze(1),values,torch.zeros_like(values))
    counts=torch.tensor([[len(quads),int(valid.sum()),int(valid.any()),int(not valid.any()),int((~valid).sum()),sum(seams)]],dtype=torch.long)
    return index,values,valid,counts

def attach_physical_torsion(data,static,trimer,positions):
    if data.geometry_valid:
        index,values,valid,counts=physical_torsion_fields(static,trimer,positions)
    else:
        index=torch.empty(0,dtype=torch.long);values=torch.empty((0,2));valid=torch.empty(0,dtype=torch.bool)
        counts=torch.zeros((1,6),dtype=torch.long)
    data.torsion_bond_index=index;data.torsion_values=values;data.torsion_mask=valid
    data.torsion_counts=counts;data.torsion_revision='glt_physical_torsion_r1'
    data.torsion_available=torch.tensor([bool(valid.any())])


def validate_canonical_topology(topology):
    """Accept the original tagged schema or its published direct field schema."""
    from .canonical_periodic import (CANONICAL_TOPOLOGY_SCHEMA,
        CANONICAL_FEATURE_SCHEMA, CANONICAL_LGA_SCHEMA_VERSION)
    from .graph_data import MIPS_ATOM_FEATURE_DIM
    tags = dict(canonical_periodic_topology_schema=CANONICAL_TOPOLOGY_SCHEMA,
        feature_schema=CANONICAL_FEATURE_SCHEMA,
        mips_local_lga_schema_version=CANONICAL_LGA_SCHEMA_VERSION)
    if any(getattr(topology,k,None) not in (None,v) for k,v in tags.items()):
        raise ValueError('canonical topology schema mismatch')
    # select_record_fields intentionally omits the historical schema labels.
    required = ('mips_x','mips_backbone_mask','canonical_ru_atom_index',
        'canonical_to_trimer_base_atom_id','normalized_canonical_smiles',
        'lga_edge_index','lga_spd','lga_path_index','lga_path_mask',
        'lga_path_shift','lga_source_image_shift')
    if not bool(getattr(topology,'mts_canonical_periodic',False)) or any(
            getattr(topology,k,None) is None for k in required):
        raise ValueError('canonical topology field schema mismatch')
    n = topology.mips_x.size(0)
    mapping = topology.canonical_to_trimer_base_atom_id
    if (topology.mips_x.shape != (n,MIPS_ATOM_FEATURE_DIM)
            or topology.mips_x.dtype != torch.float32 or not torch.isfinite(topology.mips_x).all()
            or mapping.dtype != torch.long or mapping.shape != (n,)
            or not torch.equal(mapping.sort().values,torch.arange(n))
            or not torch.equal(topology.canonical_ru_atom_index,torch.arange(n))):
        raise ValueError('canonical topology feature/identity schema mismatch')


def prepare_pretrain_sample(topology, trimer, smiles, *, seed, key, position,
                            sigma=0.03, ratio=0.3, static=None, target=None, identity=None,
                            source_type="legacy_uncached",
                            third_task='none', fgr_mu=0.0, fgr_sigma=1.0,
                            fgr_max_pairs=32, atom_target='element',
                            env_vocab=None, torsion_mode=None, torsion_revision=None, opt_mode=None, geometry_mode="finite_internal", nonbonded_revision=None):
    if not bool(topology.graph_available):
        raise ValueError('three-task training requires valid canonical 2D topology')
    if not math.isfinite(sigma) or sigma < 0 or not 0 < ratio < 1:
        raise ValueError('invalid noise sigma or masking ratio')
    if torsion_revision not in (None,'glt_physical_torsion_r1') or (torsion_revision and (torsion_mode!='on' or geometry_mode!='finite_incident')):
        raise ValueError('invalid physical torsion preparation mode')
    new_cycle = opt_mode == 'GLT_FUSION_R1'
    seam = geometry_mode in ('finite_incident','periodic_template')
    generator = (torch.Generator().manual_seed((int(seed) + 1000003 * int(position)) % (2**63-1))
                 if new_cycle else sample_generator(seed, key, position))
    third_task = str(third_task).lower()
    if third_task not in {'none', 'fgr', 'align'}:
        raise ValueError('unsupported third pretraining task')
    if identity is not None:
        validate_canonical_topology(topology)
        mapping = identity.get('canonical_to_normalized_base')
        if (identity.get('validated_source_smiles') != str(smiles)
                or identity.get('normalized_smiles') != topology.normalized_canonical_smiles
                or identity.get('validated_source_schema') != 'mcl-ph-direct-source-v1'
                or identity.get('validated_topology_schema') != getattr(topology,'canonical_periodic_topology_schema',None)
                or not torch.is_tensor(mapping) or mapping.dtype != torch.long
                or not torch.equal(mapping, topology.canonical_to_trimer_base_atom_id)
                or identity.get('base_count') != topology.mips_x.size(0)):
            raise ValueError('validated identity/source/topology schema mismatch')
    else:
        identity = resolve_normalized_identity(topology, smiles, require_fields=True)
    if target is not None:
        # Legacy direct records retain an obsolete packed fingerprint. Like
        # the BRICS cache reader, leave it untouched and consume only groups.
        if (not isinstance(target, dict) or 'brics_groups' not in target
                or set(target) - {'brics_groups', 'fingerprint_packed'}):
            raise ValueError('invalid pretrain target cache schema')
        raw = target['brics_groups']
        if (not isinstance(raw, (tuple, list)) or not raw
                or any(not isinstance(group, (tuple, list)) or not group for group in raw)
                or any(isinstance(atom, (bool, np.bool_)) or not isinstance(atom, (int, np.integer))
                       for group in raw for atom in group)):
            raise ValueError('invalid cached BRICS groups')
        groups = tuple(tuple(int(atom) for atom in group) for group in raw)
        count = topology.mips_x.size(0)
        if (sorted(atom for group in groups for atom in group) != list(range(count))
                or any(tuple(sorted(group)) != group for group in groups)
                or tuple(sorted(groups)) != groups):
            raise ValueError('cached BRICS groups/order do not partition normalized atoms')
    else:
        if source_type == 'direct_pi1m':
            raise ValueError('direct PI1M target cache is missing')
        groups = chemical_groups(identity['normalized_smiles'])
    mask, fallback = motif_mask(topology, groups, generator, ratio)
    # Static chemistry/connectivity/path is a pure function of
    # (topology, smiles): build it once and share it between the clean and
    # noisy views.  Coordinates stay view-local: clean coordinates produce the
    # clean length/angle targets, and only the noisy clone is re-encoded.
    if static is None:
        static_paths = bond_paths(topology, smiles, identity=identity)
        clean = build_dual_sample(
            topology, trimer, smiles, identity=identity,
            bond_path_features=static_paths, geometry_mode=geometry_mode,
        )
    else:
        clean = build_dual_sample(topology, trimer, smiles, static=static, geometry_mode=geometry_mode)
    noisy = clean
    if clean.geometry_valid:
        changed = copy.copy(trimer)
        changed.trimer_pos = trimer.trimer_pos.float().clone()
        changed.trimer_pos += sigma * torch.randn(changed.trimer_pos.shape, generator=generator)
        if static is None:
            noisy = build_dual_sample(
                topology, changed, smiles, identity=identity,
                bond_path_features=static_paths, geometry_mode=geometry_mode,
            )
        else:
            noisy = build_dual_sample(topology, changed, smiles, static=static, geometry_mode=geometry_mode)
        if not noisy.geometry_valid:
            raise ValueError(f'noise invalidated required geometry: {noisy.geometry_invalid_reason}')
        for field in ('line_source', 'line_target', 'line_path', 'line_path_group', 'bond_center'):
            if not torch.equal(getattr(clean, field), getattr(noisy, field)):
                raise ValueError('coordinate perturbation changed physical topology')
    if new_cycle or opt_mode in ('F_TOKEN', 'M_TOKEN_CE'):
        # Unknown element (0) maps to existing unknown Gaussian categories.
        za = noisy.bond_z_a.clone(); zb = noisy.bond_z_b.clone()
        za[mask[noisy.bond_canonical_a]] = 0
        zb[mask[noisy.bond_canonical_b]] = 0
        noisy.bond_z_a, noisy.bond_z_b = za, zb
    if new_cycle:
        if not hasattr(noisy, 'md200'):
            # MD-off arms retain the same projection parameters, with an
            # explicitly unavailable channel. MD-on sources require a sidecar.
            noisy.md200 = torch.zeros((1, 200), dtype=torch.float32)
            noisy.md200_valid = torch.zeros(1, dtype=torch.bool)
        md_generator = torch.Generator().manual_seed((20261007 + 2000003 * int(position)) % (2**63-1))
        noisy.md200_valid = noisy.md200_valid & (torch.rand((), generator=md_generator) >= 0.5)
    needs_bond_table = (atom_target == 'environment') or (torsion_mode is not None and torsion_revision is None)
    bond_table = None
    if needs_bond_table:
        if identity is None:
            raise ValueError('environment/torsion targets require the normalized identity')
        bond_table = periodic_bond_table(
            identity['normalized_smiles'], identity['canonical_to_normalized_base'])
    if atom_target == 'environment':
        targets_atom = environment_atom_labels(bond_table, _topology_z(topology), env_vocab)
        if targets_atom.numel() != topology.mips_x.size(0):
            raise ValueError('environment label count differs from canonical atoms')
    else:
        targets_atom = None
    # Use one path row for each genuine undirected one-hop center angle.  Keep
    # the original row order and directed relation multiplicity, but gather the
    # validity mask in one tensor operation instead of scanning Python rows.
    path = clean.line_path
    row_valid = (clean.line_mask.sum(dim=1) == 1) & (~clean.line_is_self)
    safe_a = path[:, 0].clamp_min(0)
    safe_b = path[:, 1].clamp_min(0)
    row_valid &= path[:, 0] < path[:, 1]
    if seam or opt_mode == 'G_SEAM_GEO':
        row_valid &= clean.line_center_angle
    else:
        row_valid &= clean.bond_center[safe_a] & clean.bond_center[safe_b]
    selected = torch.where(row_valid)[0]
    pairs = path[selected, :2].long()
    cosines = clean.line_angle[selected, 0].cos()
    reasons = []
    if not mask.any():
        reasons.append('chem:single_atom')
    if not clean.geometry_valid:
        reasons.append('geo:' + clean.geometry_invalid_reason)
    elif not bool((clean.bond_length_weight > 0).any() if seam else clean.bond_center.any()):
        reasons.append('geo:no_center_bonds')
    elif not pairs.numel():
        reasons.append('angle:no_center_angles')
    fgr_index, fgr_target, fgr_spd = (fgr_pairs(
        topology, trimer, smiles, identity=identity, seed=seed, key=key, position=position,
        mu=fgr_mu, sigma=fgr_sigma, max_pairs=fgr_max_pairs,
        generator=(torch.Generator().manual_seed((20261007 + 3000017 * int(position)) % (2**63-1))
                   if new_cycle else None))
        if third_task == 'fgr' else
        (torch.empty((0, 2), dtype=torch.long), torch.empty(0), torch.empty(0, dtype=torch.long)))
    targets = dict(preparation_source=source_type, brics_source="cached" if target is not None else "legacy_computed",
        atom_mask=mask,
        atom_label=(targets_atom if targets_atom is not None
                    else topology.mips_x[:, :101].argmax(-1)),
        distance=(clean.bond_distance[clean.bond_length_weight > 0] if seam or opt_mode == 'G_SEAM_GEO' else clean.bond_distance[clean.bond_center]).clone(),
        distance_weight=(clean.bond_length_weight[clean.bond_length_weight > 0] if seam or opt_mode == 'G_SEAM_GEO' else torch.ones(int(clean.bond_center.sum()))),
        angle_pairs=pairs.reshape(-1, 2),
        angle_cos=cosines if cosines.numel() else torch.empty(0),
        fallback=fallback, skip_reasons=reasons,
        fgr_pair_index=fgr_index, fgr_target=fgr_target, fgr_spd=fgr_spd,
        fgr_valid=bool(fgr_target.numel()),
        align_identity=(str(topology.normalized_canonical_smiles) if new_cycle else str(key)),
        align_valid=bool(clean.geometry_valid and ((clean.bond_length_weight>0).any() if seam else clean.bond_center.any())),
        third_task=third_task)
    if torsion_revision:
        attach_physical_torsion(noisy,static,changed if clean.geometry_valid else trimer,(changed if clean.geometry_valid else trimer).trimer_pos)
    elif torsion_mode is not None:
        mode = str(torsion_mode).lower()
        if mode not in {'on', 'off'}:
            raise ValueError('torsion mode must be on or off')
        if static is None:
            raise ValueError('torsion fields require the frozen static rows')
        if clean.geometry_valid:
            mapping = torch.as_tensor(
                getattr(trimer, 'mips_to_trimer_central_index', torch.empty(0)),
                dtype=torch.long).reshape(-1)
            index, values, valid = _torsion_fields(
                static, _topology_z(topology).tolist(), changed.trimer_pos, mapping, mode,
                bond_table)
        else:
            index = torch.zeros(0, dtype=torch.long)
            values = torch.zeros((0, 2), dtype=torch.float32)
            valid = torch.zeros(0, dtype=torch.bool)
        noisy.torsion_bond_index = index
        noisy.torsion_values = values
        noisy.torsion_mask = valid
    if nonbonded_revision is not None:
        if nonbonded_revision!='glt_nonbonded_r10_g1' or geometry_mode!='finite_incident' or torsion_mode is not None:
            raise ValueError('invalid nonbonded preparation contract')
        from .glt_r10 import attach_nonbonded
        carrier=changed if clean.geometry_valid else trimer
        attach_nonbonded(noisy,static,carrier,carrier.trimer_pos)
    return noisy, targets


def pretrain_collate(records):
    inputs, targets = zip(*records)
    data = dual_glt_collate(inputs)
    atom_offset = bond_offset = 0
    pairs, angle_graph, fgr_pairs_list, fgr_targets, fgr_graph = [], [], [], [], []
    fgr_spd, fgr_graph_valid = [], []
    for graph, (item, target) in enumerate(records):
        pairs.append(target['angle_pairs'] + bond_offset)
        angle_graph.append(torch.full((target['angle_pairs'].size(0),), graph, dtype=torch.long))
        if target['fgr_pair_index'].numel():
            fgr_pairs_list.append(target['fgr_pair_index'] + atom_offset)
            fgr_targets.append(target['fgr_target'])
            fgr_spd.append(target['fgr_spd'])
            fgr_graph.append(torch.full((target['fgr_target'].numel(),), graph, dtype=torch.long))
        fgr_graph_valid.append(bool(target['fgr_valid']))
        atom_offset += item.mips_x.size(0)
        bond_offset += item.bond_distance.numel()
    labels = {name: torch.cat([t[name] for t in targets]) for name in
              ('atom_mask', 'atom_label', 'distance', 'distance_weight', 'angle_cos')}
    labels.update(angle_pairs=torch.cat(pairs), angle_graph=torch.cat(angle_graph),
        fgr_pair_index=(torch.cat(fgr_pairs_list) if fgr_pairs_list else torch.empty((0, 2), dtype=torch.long)),
        fgr_target=(torch.cat(fgr_targets) if fgr_targets else torch.empty(0)),
        fgr_spd=(torch.cat(fgr_spd) if fgr_spd else torch.empty(0, dtype=torch.long)),
        fgr_graph=(torch.cat(fgr_graph) if fgr_graph else torch.empty(0, dtype=torch.long)),
        fgr_graph_valid=torch.tensor(fgr_graph_valid, dtype=torch.bool),
        align_identity=[t['align_identity'] for t in targets],
        align_valid=torch.tensor([bool(t['align_valid']) for t in targets], dtype=torch.bool),
        preparation_source=[t['preparation_source'] for t in targets],
        brics_source=[t['brics_source'] for t in targets],
        fallback_count=sum(t['fallback'] for t in targets),
        skip_reasons=[reason for t in targets for reason in t['skip_reasons']])
    if any('geometry_quality_weight' in t for t in targets):
        if not all('geometry_quality_weight' in t for t in targets):raise ValueError('mixed QC weight views')
        labels['geometry_quality_weight']=torch.tensor([t['geometry_quality_weight'] for t in targets],dtype=torch.float32)
    return data, labels
