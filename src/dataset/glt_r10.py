"""Finite nonbonded observations and selected-conformer QC, without new geometry."""
from pathlib import Path
import json
import numpy as np
import torch


def nonbonded_fields(static, trimer, positions):
    if static is None:
        raise ValueError('nonbonded input requires frozen physical endpoint identities')
    z = torch.as_tensor(trimer.trimer_atomic_number).long().reshape(-1)
    pos = torch.as_tensor(positions).float()
    offset = torch.as_tensor(trimer.trimer_ru_offset).long().reshape(-1)
    base = torch.as_tensor(trimer.trimer_base_ru_atom_id).long().reshape(-1)
    if pos.shape != (len(z), 3) or len(offset) != len(z) or len(base) != len(z):
        raise ValueError('nonbonded physical/image schema mismatch')
    if not torch.isfinite(pos).all():
        raise ValueError('nonfinite valid finite Trimer coordinates')
    adjacency = {}; bonds = {}
    pa, pb = np.asarray(static['token_pos_index_a']), np.asarray(static['token_pos_index_b'])
    za, zb = np.asarray(static['token_z_a']), np.asarray(static['token_z_b'])
    if any(a.ndim != 1 or a.dtype.kind not in 'iu' for a in (pa, pb, za, zb)) or not (len(pa) == len(pb) == len(za) == len(zb)):
        raise ValueError('nonbonded physical bond schema mismatch')
    for row, (a, b) in enumerate(zip(pa.tolist(), pb.tolist())):
        if a == b or min(a, b) < 0 or max(a, b) >= len(z) or int(z[a]) != za[row] or int(z[b]) != zb[row]:
            raise ValueError('nonbonded physical bond identity mismatch')
        key = tuple(sorted((a, b)))
        if key in bonds:
            raise ValueError('duplicate physical bond')
        bonds[key] = row
        if z[a] > 1 and z[b] > 1:
            adjacency.setdefault(a, set()).add(b); adjacency.setdefault(b, set()).add(a)
    rows, values, masks, endpoints = [], [], [], []
    seen = set()
    for (b, c), row in sorted(bonds.items()):
        if b not in adjacency or c not in adjacency:
            continue
        for a in sorted(adjacency[b] - {c}):
            for d in sorted(adjacency[c] - {b}):
                if len({a, b, c, d}) != 4 or d in adjacency[a] or adjacency[a] & adjacency[d]:
                    continue  # A three-edge path exists; reject shorter one/two-edge paths.
                key = (row, min(a, d), max(a, d))
                if key in seen:
                    continue
                seen.add(key)
                distance = torch.linalg.vector_norm(pos[a] - pos[d])
                # A zero endpoint distance is degenerate; keep the relation masked.
                valid = bool(distance > 1e-8)
                rows.append(row); values.append(float(distance) if valid else 0.0); masks.append(valid)
                endpoints.append(((int(base[a]), int(offset[a])), (int(base[d]), int(offset[d]))))
    return (torch.tensor(rows, dtype=torch.long), torch.tensor(values, dtype=torch.float32),
            torch.tensor(masks, dtype=torch.bool), endpoints)


def attach_nonbonded(data, static, trimer, positions):
    if data.geometry_valid:
        index, distance, mask, _ = nonbonded_fields(static, trimer, positions)
    else:
        index = torch.empty(0, dtype=torch.long); distance = torch.empty(0); mask = torch.empty(0, dtype=torch.bool)
    data.nonbonded_bond_index = index; data.nonbonded_distance = distance; data.nonbonded_mask = mask
    data.nonbonded_revision = 'glt_nonbonded_r10_g1'


def build_qc_sidecar(destination, ledger, records, count=959588):
    """Read existing raw selected QC once; do not use old eligible filtering."""
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    weights = np.full(count, -1, dtype=np.float32); keys = np.zeros((count, 32), dtype=np.uint8)
    # Existing records.jsonl supplies exact normalized identity, not coordinates.
    identities = {}
    with Path(records).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['pretrain_role'] != 'excluded':
                identities[row['source_index']] = (row['sample_key'], row['normalized_smiles'])
    counts = {'converged': 0, 'unconverged': 0, 'invalid': 0}
    with Path(ledger).open() as stream:
        for line in stream:
            q = json.loads(line); i = q['source_index']
            if i not in identities or weights[i] != -1 or identities.pop(i) != (q['sample_key'], q['normalized_smiles']):
                raise ValueError('selected QC identity/duplicate mismatch')
            valid = q['geometry_valid']
            if type(valid) is not bool or valid != (q['final_status'] == 'PASS'):
                raise ValueError('selected QC geometry status mismatch')
            if valid:
                selected, status = q.get('selected_converged'), q.get('mmff_status')
                if type(selected) is not bool or status not in (-1, 0, 1) or selected != (status == 0):
                    raise ValueError('missing/contradictory selected QC convergence')
                weights[i] = 1 if selected else .5
                counts['converged' if selected else 'unconverged'] += 1
            else:
                weights[i] = 0; counts['invalid'] += 1
            keys[i] = np.frombuffer(bytes.fromhex(q['sample_key']), dtype=np.uint8)
    if identities:
        raise ValueError('missing selected QC rows')
    np.save(destination / 'weights.npy', weights); np.save(destination / 'keys.npy', keys)
    (destination / 'metadata.json').write_text(json.dumps(dict(schema='glt-r10-selected-qc-v1',
        counts=counts, ledger=str(ledger), rows=sum(counts.values()), coordinate_generation=0), indent=2))


def attach_qc_source(source, root):
    if not root:
        return source
    root = Path(root)
    metadata = json.loads((root / 'metadata.json').read_text())
    if metadata['schema'] != 'glt-r10-selected-qc-v1' or source.kind != 'pi1m':
        raise ValueError('QC weight/source schema mismatch')
    weights = np.load(root / 'weights.npy', mmap_mode='r', allow_pickle=False)
    keys = np.load(root / 'keys.npy', mmap_mode='r', allow_pickle=False)
    values = []
    for row in source.entries:
        i = row['source_index']; weight = float(weights[i])
        if row['pretrain_role'] == 'excluded':
            if weight != -1:
                raise ValueError('QC contains excluded identity')
            values.append(0.0); continue
        if weight not in (0.0, .5, 1.0) or bytes(keys[i]) != bytes.fromhex(row['sample_key']):
            raise ValueError('QC weight/source identity mismatch')
        if 'geometry_status' in row and (weight > 0) != (row['geometry_status'] == 'PASS'):
            raise ValueError('QC cannot change geometry coverage')
        values.append(weight)
    source.r10_qc_weights = values
    return source
