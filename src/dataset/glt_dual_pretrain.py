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


def chemical_groups(smiles):
    """Build only the common 3-RU BRICS groups.

    FP-free objectives still need the deterministic motif partition, but must
    not pay for or accidentally consume the 7-RU Morgan target.
    """
    source = Chem.MolFromSmiles(str(smiles))
    if source is None:
        raise ValueError("invalid P-SMILES for chemical groups")
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
    return tuple(sorted(groups))


@lru_cache(maxsize=512)
def chemical_targets(smiles):
    groups = chemical_groups(str(smiles))
    source = Chem.MolFromSmiles(str(smiles))
    if source is None:
        raise ValueError("invalid P-SMILES for chemical targets")
    normalized = Chem.MolToSmiles(source, canonical=True)
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
    return groups, fingerprint


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


def fgr_pairs(topology, trimer, smiles=None, *, identity=None, seed, key, position,
              mu=0.0, sigma=1.0, max_pairs=32):
    """Return deterministic centre-RU SPD2/3 pairs from the real 3-RU graph.

    Pair identity is defined only by RDKit shortest paths in the open chemical
    Trimer. Periodic lifted relations and coordinates are not consulted for
    candidate selection; coordinates are used only after strict mapping checks
    to produce the clean distance label.
    """
    if not math.isfinite(float(mu)) or not math.isfinite(float(sigma)) or sigma <= 0:
        raise ValueError('FGR normalization requires finite mu and positive sigma')
    if int(max_pairs) <= 0:
        raise ValueError('FGR max_pairs must be positive')
    if not bool(getattr(trimer, 'trimer_geometry_valid', False)):
        return torch.empty((0, 2), dtype=torch.long), torch.empty(0), torch.empty(0, dtype=torch.long)
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
    generator = sample_generator(seed, f'{key}:fgr', position)
    ordered = []
    # Keep the prescribed SPD2/SPD3 cap independent of the order of the
    # shortest-path table.  The two classes are sampled without replacement;
    # all remaining capacity is naturally used when one class has fewer than
    # its cap, while no class can exceed 16 entries.
    remaining = int(max_pairs)
    for hop in (2, 3):
        if remaining <= 0:
            break
        class_pairs = sorted(pair for pair in candidates if candidates[pair] == hop)
        cap = min(16, remaining, len(class_pairs))
        if len(class_pairs) > cap:
            chosen = torch.randperm(len(class_pairs), generator=generator)[:cap].tolist()
            class_pairs = [class_pairs[index] for index in sorted(chosen)]
        ordered.extend(class_pairs)
        remaining -= len(class_pairs)
    ordered = sorted(ordered, key=lambda pair: (candidates[pair], pair[0], pair[1]))
    if not ordered:
        return torch.empty((0, 2), dtype=torch.long), torch.empty(0), torch.empty(0, dtype=torch.long)
    pair_tensor = torch.as_tensor(ordered, dtype=torch.long)
    mapped = mapping[pair_tensor]
    if int(mapped.max()) >= positions.size(0) or int(mapped.min()) < 0:
        raise ValueError('FGR mapped coordinate index is out of range')
    distances = torch.linalg.vector_norm(positions[mapped[:, 0]] - positions[mapped[:, 1]], dim=-1)
    if not bool(torch.isfinite(distances).all()) or bool((distances <= 0).any()):
        raise ValueError('FGR clean pair distance is invalid')
    normalized = (torch.log1p(distances) - float(mu)) / float(sigma)
    return pair_tensor, normalized, torch.as_tensor([candidates[pair] for pair in ordered], dtype=torch.long)


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
                            sigma=0.03, ratio=0.3, static=None, target=None,
                            third_task='fp', fgr_mu=0.0, fgr_sigma=1.0,
                            fgr_max_pairs=32):
    if not bool(topology.graph_available):
        raise ValueError('three-task training requires valid canonical 2D topology')
    if not math.isfinite(sigma) or sigma < 0 or not 0 < ratio < 1:
        raise ValueError('invalid noise sigma or masking ratio')
    generator = sample_generator(seed, key, position)
    third_task = str(third_task).lower()
    if third_task not in {'fp', 'none', 'fgr', 'align'}:
        raise ValueError('unsupported third pretraining task')
    identity = None
    # FGR candidate identity is independent of the optional target cache, but
    # still requires the strict source→normalized→canonical mapping contract.
    if third_task == 'fgr' and identity is None:
        identity = resolve_normalized_identity(topology, smiles, require_fields=True)
    if target is not None:
        if 'brics_groups' not in target:
            raise ValueError('pretrain target row is missing BRICS groups')
        groups = tuple(tuple(int(atom) for atom in group)
                       for group in target['brics_groups'])
        if third_task == 'fp':
            if 'fingerprint_packed' not in target:
                raise ValueError('FP target row is missing fingerprint_packed')
            fingerprint = _unpack_fingerprint(target['fingerprint_packed'])
        else:
            fingerprint = torch.zeros(2048)
        identity = None
    else:
        identity = resolve_normalized_identity(topology, smiles, require_fields=True)
        if third_task == 'fp':
            groups, fingerprint = chemical_targets(identity["normalized_smiles"])
        else:
            groups = chemical_groups(identity["normalized_smiles"])
            fingerprint = torch.zeros(2048)
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
    fgr_index, fgr_target, fgr_spd = (fgr_pairs(
        topology, trimer, smiles, identity=identity, seed=seed, key=key, position=position,
        mu=fgr_mu, sigma=fgr_sigma, max_pairs=fgr_max_pairs)
        if third_task == 'fgr' else
        (torch.empty((0, 2), dtype=torch.long), torch.empty(0), torch.empty(0, dtype=torch.long)))
    targets = dict(atom_mask=mask, atom_label=topology.mips_x[:, :101].argmax(-1),
        distance=clean.bond_distance[clean.bond_center].clone(),
        angle_pairs=pairs.reshape(-1, 2),
        angle_cos=cosines if cosines.numel() else torch.empty(0),
        fingerprint=fingerprint.clone(), fallback=fallback, skip_reasons=reasons,
        fgr_pair_index=fgr_index, fgr_target=fgr_target, fgr_spd=fgr_spd,
        fgr_valid=bool(fgr_target.numel()),
        align_identity=str(key),
        align_valid=bool(clean.geometry_valid and clean.bond_center.any()),
        third_task=third_task)
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
              ('atom_mask', 'atom_label', 'distance', 'angle_cos')}
    labels.update(angle_pairs=torch.cat(pairs), angle_graph=torch.cat(angle_graph),
        fingerprint=torch.stack([t['fingerprint'] for t in targets]),
        fgr_pair_index=(torch.cat(fgr_pairs_list) if fgr_pairs_list else torch.empty((0, 2), dtype=torch.long)),
        fgr_target=(torch.cat(fgr_targets) if fgr_targets else torch.empty(0)),
        fgr_spd=(torch.cat(fgr_spd) if fgr_spd else torch.empty(0, dtype=torch.long)),
        fgr_graph=(torch.cat(fgr_graph) if fgr_graph else torch.empty(0, dtype=torch.long)),
        fgr_graph_valid=torch.tensor(fgr_graph_valid, dtype=torch.bool),
        align_identity=[t['align_identity'] for t in targets],
        align_valid=torch.tensor([bool(t['align_valid']) for t in targets], dtype=torch.bool),
        fallback_count=sum(t['fallback'] for t in targets),
        skip_reasons=[reason for t in targets for reason in t['skip_reasons']])
    return data, labels
