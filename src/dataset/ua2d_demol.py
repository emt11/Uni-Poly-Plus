"""Geometry-free periodic bond templates and exact shared-image incidence."""
from collections import defaultdict
import torch
from src.dataset.glt_dual import build_glt_2d_sample, glt_2d_collate, _periodic_key

INPUT_SCHEMA = 'ua2d-demol-periodic-incidence-r1'
SEMANTICS = dict(schema=INPUT_SCHEMA, bonds='translation_quotient_undirected_a_b_shift',
    adjacency='shared_physical_endpoint_then_translation_quotient_with_multiplicity',
    incidence='both_periodic_endpoints_mean', geometry=False)


def attach_bonds(data):
    table = {}
    for i in (data.lga_spd == 1).nonzero().flatten().tolist():
        mask = data.lga_path_mask[i].bool()
        if mask.tolist() != [True, True, False]:
            raise ValueError('unexpected one-hop path mask')
        a, b = data.lga_path_index[i, :2].tolist()
        qa, qb = data.lga_path_shift[i, :2].tolist()
        key = _periodic_key(a, qa, b, qb)
        if a == b and qa == qb: raise ValueError('physical atomic self bond')
        feature = data.bond_path_features[i, 0]
        if not bool(data.bond_path_mask[i, 0]): raise ValueError('missing bond chemistry')
        if key in table and not torch.equal(table[key], feature):
            raise ValueError('periodic bond chemistry mismatch')
        table[key] = feature.clone()
    keys = sorted(table)
    data.demol_keys = torch.tensor(keys, dtype=torch.long).reshape(-1, 3)
    data.demol_chem = torch.stack([table[k] for k in keys]) if keys else torch.empty(0, 14)
    # Each template yields an incidence at each endpoint. Translations are
    # resolved in the lifted graph BEFORE mapping a neighbor to its template.
    incident = defaultdict(list)
    for i, (a, b, shift) in enumerate(keys):
        incident[a].append((i, 0)); incident[b].append((i, shift))
    src, dst = [], []
    for target, (a, b, shift) in enumerate(keys):
        physical_neighbors = set()
        for atom, image in ((a, 0), (b, shift)):
            for source, local_image in incident[atom]:
                physical_neighbors.add((source, image-local_image))
        physical_neighbors.add((target, 0))
        for source, translation in sorted(physical_neighbors):
            src.append(source); dst.append(target)
    data.demol_line = torch.tensor([src, dst], dtype=torch.long).reshape(2, -1)
    data.demol_incidence = torch.stack((torch.arange(len(keys)).repeat_interleave(2),
                                      data.demol_keys[:, :2].reshape(-1)))
    data.demol_schema = INPUT_SCHEMA
    return data


def build(topology, smiles, *, static=None):
    return attach_bonds(build_glt_2d_sample(topology, smiles, static=static))


def collate(samples):
    if any(s.demol_schema != INPUT_SCHEMA for s in samples): raise ValueError('input schema mismatch')
    batch = glt_2d_collate(samples)
    keys, chemistry, edges, incidence = [], [], [], []
    atoms = bonds = 0
    for s in samples:
        key = s.demol_keys.clone(); key[:, :2] += atoms
        keys.append(key); chemistry.append(s.demol_chem)
        edges.append(s.demol_line+bonds)
        incidence.append(s.demol_incidence+torch.tensor([[bonds], [atoms]]))
        atoms += len(s.mips_x); bonds += len(key)
    batch.demol_keys = torch.cat(keys); batch.demol_chem = torch.cat(chemistry)
    batch.demol_line = torch.cat(edges, 1); batch.demol_incidence = torch.cat(incidence, 1)
    batch.demol_schema = INPUT_SCHEMA
    return batch
