"""On-demand three-task targets; no descriptor input or persistent cache writes."""
import copy
import hashlib
import math
from collections import deque
from functools import lru_cache

import numpy as np

import torch
from rdkit import Chem
from rdkit.Chem import BRICS, rdFingerprintGenerator

from .graph_data import build_periodic_multimer_mol
from .glt_dual import bond_paths, build_dual_sample, dual_glt_collate
from .canonical_periodic import resolve_normalized_identity


def sample_generator(seed, key, position):
    payload = f'{seed}:{key}:{position}'.encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % (2**63 - 1)
    return torch.Generator().manual_seed(value)


@lru_cache(maxsize=512)
def chemical_targets(smiles):
    source = Chem.MolFromSmiles(str(smiles))
    if source is None:
        raise ValueError("invalid P-SMILES for chemical targets")
    # BRICS groups and the fingerprint roots are model-side identities.  The
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
    large, info = build_periodic_multimer_mol(
        normalized, 7, close_periodic=False
    )
    roots = info['unit_atoms'][3]
    terminals = [info['unit_left_boundaries'][0], info['unit_right_boundaries'][-1]]
    distances = Chem.GetDistanceMatrix(large)
    if any(distances[root, end] < 3 for root in roots for end in terminals):
        raise ValueError('7-RU topology does not cover center Morgan radius plus boundary context')
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    fingerprint = torch.from_numpy(generator.GetFingerprintAsNumPy(large, fromAtoms=roots).copy()).float()
    return tuple(sorted(groups)), fingerprint


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


def _unpack_fingerprint(packed):
    packed = np.asarray(packed, dtype=np.uint8)
    if packed.shape != (256,):
        raise ValueError('packed Morgan fingerprint must have shape [256]')
    bits = np.unpackbits(packed, bitorder='little')
    return torch.from_numpy(bits[:2048].copy()).float()


def prepare_pretrain_sample(topology, trimer, smiles, *, seed, key, position,
                            sigma=0.03, ratio=0.3, static=None, target=None):
    if not bool(topology.graph_available):
        raise ValueError('three-task training requires valid canonical 2D topology')
    if not math.isfinite(sigma) or sigma < 0 or not 0 < ratio < 1:
        raise ValueError('invalid noise sigma or masking ratio')
    generator = sample_generator(seed, key, position)
    if target is not None:
        if 'brics_groups' not in target or 'fingerprint_packed' not in target:
            raise ValueError('pretrain target row is missing required fields')
        groups = tuple(tuple(int(atom) for atom in group)
                       for group in target['brics_groups'])
        fingerprint = _unpack_fingerprint(target['fingerprint_packed'])
        identity = None
    else:
        identity = resolve_normalized_identity(topology, smiles, require_fields=True)
        groups, fingerprint = chemical_targets(identity["normalized_smiles"])
    mask, fallback = motif_mask(topology, groups, generator, ratio)
    # Static chemistry/connectivity/path is a pure function of
    # (topology, smiles): build it once and share it between the clean and
    # noisy views.  Coordinates stay view-local: clean coordinates produce the
    # clean length/angle targets, and only the noisy clone is re-encoded.
    if static is None:
        static_paths = bond_paths(topology, smiles, identity=identity)
        clean = build_dual_sample(
            topology, trimer, smiles, identity=identity,
            bond_path_features=static_paths,
        )
    else:
        clean = build_dual_sample(topology, trimer, smiles, static=static)
    noisy = clean
    if clean.geometry_valid:
        changed = copy.copy(trimer)
        changed.trimer_pos = trimer.trimer_pos.float().clone()
        changed.trimer_pos += sigma * torch.randn(changed.trimer_pos.shape, generator=generator)
        if static is None:
            noisy = build_dual_sample(
                topology, changed, smiles, identity=identity,
                bond_path_features=static_paths,
            )
        else:
            noisy = build_dual_sample(topology, changed, smiles, static=static)
        if not noisy.geometry_valid:
            raise ValueError(f'noise invalidated required geometry: {noisy.geometry_invalid_reason}')
        for field in ('line_source', 'line_target', 'line_path', 'line_path_group', 'bond_center'):
            if not torch.equal(getattr(clean, field), getattr(noisy, field)):
                raise ValueError('coordinate perturbation changed physical topology')
    # Use one path row for each genuine undirected one-hop center angle.  Keep
    # the original row order and directed relation multiplicity, but gather the
    # validity mask in one tensor operation instead of scanning Python rows.
    path = clean.line_path
    row_valid = (clean.line_mask.sum(dim=1) == 1) & (~clean.line_is_self)
    safe_a = path[:, 0].clamp_min(0)
    safe_b = path[:, 1].clamp_min(0)
    row_valid &= path[:, 0] < path[:, 1]
    row_valid &= clean.bond_center[safe_a] & clean.bond_center[safe_b]
    selected = torch.where(row_valid)[0]
    pairs = path[selected, :2].long()
    cosines = clean.line_angle[selected, 0].cos()
    reasons = []
    if not mask.any():
        reasons.append('chem:single_atom')
    if not clean.geometry_valid:
        reasons.append('geo:' + clean.geometry_invalid_reason)
    elif not clean.bond_center.any():
        reasons.append('geo:no_center_bonds')
    elif not pairs.numel():
        reasons.append('angle:no_center_angles')
    targets = dict(atom_mask=mask, atom_label=topology.mips_x[:, :101].argmax(-1),
        distance=clean.bond_distance[clean.bond_center].clone(),
        angle_pairs=pairs.reshape(-1, 2),
        angle_cos=cosines if cosines.numel() else torch.empty(0),
        fingerprint=fingerprint.clone(), fallback=fallback, skip_reasons=reasons)
    return noisy, targets


def pretrain_collate(records):
    inputs, targets = zip(*records)
    data = dual_glt_collate(inputs)
    bond_offset, pairs, angle_graph = 0, [], []
    for graph, (item, target) in enumerate(records):
        pairs.append(target['angle_pairs'] + bond_offset)
        angle_graph.append(torch.full((target['angle_pairs'].size(0),), graph, dtype=torch.long))
        bond_offset += item.bond_distance.numel()
    labels = {name: torch.cat([t[name] for t in targets]) for name in
              ('atom_mask', 'atom_label', 'distance', 'angle_cos')}
    labels.update(angle_pairs=torch.cat(pairs), angle_graph=torch.cat(angle_graph),
        fingerprint=torch.stack([t['fingerprint'] for t in targets]),
        fallback_count=sum(t['fallback'] for t in targets),
        skip_reasons=[reason for t in targets for reason in t['skip_reasons']])
    return data, labels
