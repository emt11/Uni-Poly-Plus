"""On-demand, read-only frozen-record adapter for the dual GLT route.

No coordinates are generated and no cache records are written. A source dataset
must return a canonical topology record with frozen Trimer fields, or a tuple
``(topology, trimer, smiles)``. The dedicated collator never handles MD200.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .graph_data import build_periodic_multimer_mol
from .glt_bond_chemistry import bond_feature_vector
from .periodic_line_glt_complete import build_complete_trimer_glt_sample
from .canonical_periodic import resolve_normalized_identity


def _periodic_key(a, qa, b, qb):
    """Translation-invariant UNDIRECTED bond identity, including seam shift."""
    return min((a, b, qb - qa), (b, a, qa - qb))


def bond_paths(topology, smiles, *, identity=None):
    """Periodic chemistry: central internal bonds and matching real seams.

    Terminal internal copies belong to a finite molecule and may legitimately
    lose/change stereo. They are not representatives of the periodic 2D cell.
    """
    identity = identity or resolve_normalized_identity(
        topology, smiles, require_fields=hasattr(topology, "mips_x")
    )
    construction_smiles = identity["normalized_smiles"]
    mol, meta = build_periodic_multimer_mol(
        construction_smiles, num_repeat_units=3, close_periodic=False
    )
    size = int(meta['base_atom_count'])
    chemistry = {}
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a // size == b // size and a // size != 1:
            continue
        key = _periodic_key(a % size, a // size, b % size, b // size)
        feature = bond_feature_vector(bond)
        if key in chemistry and not np.array_equal(chemistry[key], feature):
            raise ValueError('real seam bonds have inconsistent periodic chemistry')
        chemistry[key] = feature
    mapping = torch.as_tensor(
        identity["canonical_to_normalized_base"], dtype=torch.long
    )
    path, shift, mask = topology.lga_path_index.long(), topology.lga_path_shift.long(), topology.lga_path_mask.bool()
    if path.shape != mask.shape or path.shape != shift.shape or path.shape[1] != 3:
        raise ValueError('dual O8 requires existing max-hop=2 lifted paths of width 3')
    features = torch.zeros((path.size(0), 2, 14))
    valid = torch.zeros((path.size(0), 2), dtype=torch.bool)
    for row in range(path.size(0)):
        for k in range(2):
            if bool(mask[row, k] and mask[row, k + 1]):
                source_node, target_node = int(path[row, k]), int(path[row, k + 1])
                if (
                    source_node < 0 or target_node < 0
                    or source_node >= mapping.numel()
                    or target_node >= mapping.numel()
                ):
                    raise ValueError("lifted path contains an invalid canonical atom")
                a, b = int(mapping[source_node]), int(mapping[target_node])
                key = _periodic_key(a, int(shift[row, k]), b, int(shift[row, k + 1]))
                if key not in chemistry:
                    raise ValueError(f'no real periodic bond for path step {key}')
                features[row, k] = torch.from_numpy(chemistry[key].copy())
                valid[row, k] = True
    if not torch.equal(valid.sum(-1), topology.lga_spd.long()):
        raise ValueError('bond path length disagrees with SPD')
    return features, valid


def two_hop_paths(row):
    """All shortest paths per physical pair, grouped into one attention edge."""
    count = len(row['tokens']['token_distance'])
    rel = row['relations']
    adjacent = [set() for _ in range(count)]
    angles = {}
    for s, t, angle, valid in zip(rel['relation_source'], rel['relation_target'],
                                  rel['relation_angle'], rel['relation_valid']):
        if not valid or not np.isfinite(angle):
            raise ValueError('invalid required physical angle')
        s, t = int(s), int(t)
        if (s, t) in angles and not np.isclose(angles[s, t], angle):
            raise ValueError('ambiguous physical line relation')
        adjacent[s].add(t)
        angles[s, t] = float(angle)
    paths, values, self_flags, groups = [], [], [], []
    sources, targets = [], []
    for start in range(count):
        found = {target: [[start, target]] for target in adjacent[start] if target != start}
        for middle in adjacent[start]:
            for target in adjacent[middle]:
                if target != start and target not in adjacent[start]:
                    found.setdefault(target, []).append([start, middle, target])
        for end in sorted(found):
            if end <= start:
                continue
            for reverse in (False, True):
                group = len(sources)
                sources.append(end if reverse else start)
                targets.append(start if reverse else end)
                for forward in sorted(found[end]):
                    path = forward[::-1] if reverse else forward
                    paths.append(path)
                    values.append([angles[a, b] for a, b in zip(path, path[1:])])
                    self_flags.append(False)
                    groups.append(group)
    for token in range(count):
        groups.append(len(sources))
        sources.append(token)
        targets.append(token)
        paths.append([token, token])
        values.append([0.0])
        self_flags.append(True)
    p = torch.full((len(paths), 3), -1, dtype=torch.long)
    a = torch.zeros((len(paths), 2))
    mask = torch.zeros_like(a, dtype=torch.bool)
    for i, (path, angle) in enumerate(zip(paths, values)):
        p[i, :len(path)] = torch.tensor(path)
        a[i, :len(angle)] = torch.tensor(angle)
        mask[i, :len(angle)] = True
    return dict(path=p, angle=a, mask=mask, is_self=torch.tensor(self_flags, dtype=torch.bool),
                path_group=torch.tensor(groups, dtype=torch.long),
                source=torch.tensor(sources, dtype=torch.long), target=torch.tensor(targets, dtype=torch.long))


def _validate_geometry_fallback_carrier(topology, trimer):
    """Prove an invalid-geometry row still has complete structural identity."""

    names = (
        "trimer_atomic_number", "trimer_base_ru_atom_id", "trimer_ru_offset",
        "trimer_central_ru_mask", "trimer_edge_index", "trimer_bond_type",
        "mips_to_trimer_central_index",
    )
    missing = [name for name in names if not hasattr(trimer, name)]
    if missing:
        raise ValueError("geometry fallback identity carrier missing: " + ",".join(missing))
    atomic = torch.as_tensor(trimer.trimer_atomic_number).long().reshape(-1)
    base = torch.as_tensor(trimer.trimer_base_ru_atom_id).long().reshape(-1)
    offsets = torch.as_tensor(trimer.trimer_ru_offset).long().reshape(-1)
    central = torch.as_tensor(trimer.trimer_central_ru_mask).bool().reshape(-1)
    edge = torch.as_tensor(trimer.trimer_edge_index).long()
    bond_type = torch.as_tensor(trimer.trimer_bond_type).long().reshape(-1)
    mapping = torch.as_tensor(trimer.mips_to_trimer_central_index).long().reshape(-1)
    topology_z = torch.as_tensor(topology.z).long().reshape(-1)
    count = atomic.numel()
    if (
        count == 0 or base.numel() != count or offsets.numel() != count
        or central.numel() != count or edge.ndim != 2 or edge.size(0) != 2
        or bond_type.numel() != edge.size(1) or mapping.numel() != topology_z.numel()
        or (edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= count))
        or (mapping.numel() and (
            int(mapping.min()) < 0 or int(mapping.max()) >= count
            or not bool(central[mapping].all())
            or not torch.equal(atomic[mapping], topology_z)
        ))
    ):
        raise ValueError("geometry fallback structural identity is corrupt")


def build_dual_sample(topology, trimer, smiles, *, identity=None,
                      bond_path_features=None):
    """Build only model-required fields; never carry MD/coordinate tensors into a batch.

    ``identity`` and ``bond_path_features`` are pure functions of
    ``(topology, smiles)``: a caller that builds the clean and noisy views of
    one sample may resolve/construct them once and reuse them here instead of
    paying for a second graph-isomorphism search.
    """
    result = Data()
    required = (
        "mips_x", "mips_backbone_mask", "canonical_ru_atom_index",
        "lga_edge_index", "lga_spd", "lga_path_index", "lga_path_mask",
        "lga_path_shift", "lga_source_image_shift", "graph_available",
    )
    missing = [name for name in required if not hasattr(topology, name)]
    if missing:
        raise ValueError(
            "dual Topology is missing required fields: " + ",".join(missing)
        )
    if not torch.equal(topology.canonical_ru_atom_index.long(), torch.arange(topology.mips_x.size(0))):
        raise ValueError('dual route requires one canonical state per atom')
    if identity is None:
        identity = resolve_normalized_identity(
            topology, smiles, require_fields=True
        )
    for name in ('mips_x', 'mips_backbone_mask', 'lga_edge_index', 'lga_spd',
                 'lga_path_index', 'lga_path_mask', 'lga_path_shift',
                 'lga_source_image_shift'):
        setattr(result, name, getattr(topology, name).clone())
    result.graph_available = bool(topology.graph_available)
    if hasattr(topology, 'lga_relation_mask'):
        result.lga_relation_mask = topology.lga_relation_mask.clone()
    if bond_path_features is None:
        result.bond_path_features, result.bond_path_mask = bond_paths(
            topology, smiles, identity=identity
        )
    else:
        # Clone so clean and noisy views never alias one tensor.
        result.bond_path_features = bond_path_features[0].clone()
        result.bond_path_mask = bond_path_features[1].clone()
    if not bool(getattr(trimer, "trimer_geometry_valid", False)):
        _validate_geometry_fallback_carrier(topology, trimer)
    row = build_complete_trimer_glt_sample(
        topology, trimer, smiles, identity=identity
    )
    if row["geometry_valid"]:
        paths = two_hop_paths(row)
    else:
        # Invalid frozen geometry has no usable physical relations.  Structural
        # identity/chemistry failures are not geometry fallbacks: reject them
        # before an empty row could hide a malformed cached relation table.
        reason = str(row["invalid_reason"])
        geometry_reasons = {
            "geometry_invalid", "non_3d_geometry", "2d_fallback",
            "trimer_coordinates_invalid", "bond_distance_invalid",
        }
        if reason not in geometry_reasons:
            raise ValueError(f"complete Trimer contract failure: {reason}")
        from .periodic_line_glt_complete import empty_complete_trimer_row
        paths = two_hop_paths(empty_complete_trimer_row(row["invalid_reason"]))
    result.geometry_valid = bool(row['geometry_valid'])
    result.geometry_invalid_reason = row['invalid_reason']
    for src, dst in [('token_endpoint_z_a', 'bond_z_a'), ('token_endpoint_z_b', 'bond_z_b'),
                     ('token_distance', 'bond_distance'), ('token_bond_type', 'bond_type'),
                     ('token_center_internal', 'bond_center')]:
        setattr(result, dst, torch.from_numpy(row['tokens'][src].copy()))
    for name, value in paths.items():
        setattr(result, 'line_' + name, value)
    if getattr(topology, 'y', None) is not None:
        result.y = topology.y.clone()
    return result


class DualGLTDataset(Dataset):
    """Wrap a read-only source that MUST have MD and conformer generation disabled."""
    def __init__(self, frozen_dataset):
        self.source = frozen_dataset

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        record = self.source[index]
        if isinstance(record, tuple):
            return build_dual_sample(*record)
        return build_dual_sample(record, record, str(record.smiles))


class FrozenDualLayerSource(Dataset):
    """Read one immutable active bundle through a frozen ordered cohort.

    ``cohort`` is the value returned by :func:`load_dual_cohort`.  Original
    P-SMILES are provenance/audit input only; keys are verified against each
    record's explicit normalized canonical identity.
    """
    def __init__(self, cache_root, cohort):
        from .glt_dual_cache import DualFrozenBundle

        self.bundle = self.topology = self.trimer = None
        self.cohort = cohort
        self.entries = list(cohort["records"])
        try:
            self.bundle = DualFrozenBundle(
                cache_root,
                expected_bundle_hash=cohort["manifest"]["main_bundle_hash"],
            )
            self.topology = self.bundle.topology
            self.trimer = self.bundle.trimer
            self.samples = [
                (bytes.fromhex(row["sample_key"]), str(row["source_smiles"]))
                for row in self.entries
            ]
            # Parent-bound cohort creation proves complete coverage.  Reader
            # startup repeats deterministic boundary checks; every later miss
            # remains a hard error in ReadonlyArtifact.__getitem__.
            if self.samples:
                for index in sorted({0, len(self.samples) // 2, len(self.samples) - 1}):
                    key = self.samples[index][0]
                    if key not in self.topology or key not in self.trimer:
                        raise ValueError("frozen dual cohort boundary key is missing")
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        key, smiles = self.samples[index]
        topology, trimer = self.topology[key], self.trimer[key]
        # Validate the frozen source/canonical join at the cache boundary.  A
        # missing or contradictory identity table is a data-contract failure,
        # never a reason to substitute a geometry placeholder.
        resolve_normalized_identity(topology, smiles, require_fields=True)
        return topology, trimer, smiles

    def close(self):
        bundle = self.bundle
        self.bundle = self.topology = self.trimer = None
        if bundle is not None:
            bundle.close()


def dual_glt_collate(samples):
    if not samples:
        raise ValueError('empty dual batch')
    batch = Data()
    fields = {}
    atom_offset = bond_offset = relation_offset = 0
    for graph, item in enumerate(samples):
        n, m = item.mips_x.size(0), item.bond_distance.numel()
        for name in ('mips_x', 'mips_backbone_mask', 'lga_spd', 'lga_path_mask',
                     'lga_path_shift', 'lga_source_image_shift', 'bond_path_features',
                     'bond_path_mask', 'bond_z_a', 'bond_z_b', 'bond_distance',
                     'bond_type', 'bond_center', 'line_angle', 'line_mask', 'line_is_self'):
            fields.setdefault(name, []).append(getattr(item, name))
        fields.setdefault('lga_relation_mask', []).append(getattr(
            item, 'lga_relation_mask', torch.zeros(item.lga_spd.numel(), dtype=torch.bool)))
        fields.setdefault('lga_edge_index', []).append(item.lga_edge_index + atom_offset)
        for name, offset in [('lga_path_index', atom_offset), ('line_path', bond_offset)]:
            value = getattr(item, name)
            fields.setdefault(name, []).append(torch.where(value >= 0, value + offset, value))
        for name in ('line_source', 'line_target'):
            fields.setdefault(name, []).append(getattr(item, name) + bond_offset)
        fields.setdefault('line_path_group', []).append(item.line_path_group + relation_offset)
        fields.setdefault('canonical_graph_index', []).append(torch.full((n,), graph, dtype=torch.long))
        fields.setdefault('bond_batch', []).append(torch.full((m,), graph, dtype=torch.long))
        atom_offset += n
        bond_offset += m
        relation_offset += item.line_source.numel()
    for name, values in fields.items():
        setattr(batch, name, torch.cat(values, dim=1 if name == 'lga_edge_index' else 0))
    batch.graph_available = torch.tensor([x.graph_available for x in samples], dtype=torch.bool)
    batch.geometry_valid = torch.tensor([x.geometry_valid for x in samples], dtype=torch.bool)
    batch.geometry_invalid_reason = [x.geometry_invalid_reason for x in samples]
    if all(hasattr(x, 'y') and x.y is not None for x in samples):
        batch.y = torch.stack([x.y for x in samples])
    return batch
