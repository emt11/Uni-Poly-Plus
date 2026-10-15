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
        "trimer_edge_index", "trimer_bond_type",
        "mips_to_trimer_central_index",
    )
    missing = [name for name in names if not hasattr(trimer, name)]
    if missing:
        raise ValueError("trimer_identity_missing: geometry fallback identity carrier missing: " + ",".join(missing))
    atomic = torch.as_tensor(trimer.trimer_atomic_number).long().reshape(-1)
    base = torch.as_tensor(trimer.trimer_base_ru_atom_id).long().reshape(-1)
    offsets = torch.as_tensor(trimer.trimer_ru_offset).long().reshape(-1)
    central = torch.as_tensor(
        getattr(trimer, "trimer_central_ru_mask", offsets == 0)
    ).bool().reshape(-1)
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


def _build_dual_sample_without_md(topology, trimer, smiles, *, identity=None,
                      bond_path_features=None, static=None, torsion_mode=None, torsion_revision=None,
                      geometry_mode="finite_internal"):
    """Build only model-required fields; never carry MD/coordinate tensors into a batch.

    ``identity`` and ``bond_path_features`` are pure functions of
    ``(topology, smiles)``: a caller that builds the clean and noisy views of
    one sample may resolve/construct them once and reuse them here instead of
    paying for a second graph-isomorphism search.
    """
    if geometry_mode not in ("finite_internal", "finite_incident", "periodic_template"):
        raise ValueError("unknown geometry mode")
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
    if identity is None and static is None:
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
    if static is not None:
        # Persistent static rows are the single reference for chemistry and
        # physical line topology.  No RDKit graph or shortest-path rebuild is
        # allowed on this branch; only coordinates are materialized below.
        required_static = (
            'bond_path_features', 'bond_path_mask', 'token_z_a', 'token_z_b',
            'token_bond_type', 'token_center_mask', 'line_source',
            'line_target', 'line_path', 'line_path_mask', 'line_path_group',
            'line_is_self', 'geometry_valid', 'geometry_invalid_reason',
        )
        missing_static = [name for name in required_static if name not in static]
        if missing_static:
            raise ValueError('dual static row is missing fields: ' + ','.join(missing_static))
        result.bond_path_features = torch.from_numpy(
            np.array(static['bond_path_features'], dtype=np.float32, copy=True))
        result.bond_path_mask = torch.from_numpy(
            np.array(static['bond_path_mask'], dtype=bool, copy=True))
    elif bond_path_features is None:
        result.bond_path_features, result.bond_path_mask = bond_paths(
            topology, smiles, identity=identity
        )
    else:
        # Clone so clean and noisy views never alias.
        result.bond_path_features = bond_path_features[0].clone()
        result.bond_path_mask = bond_path_features[1].clone()
    if not bool(getattr(trimer, "trimer_geometry_valid", False)):
        _validate_geometry_fallback_carrier(topology, trimer)
    if static is None:
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
    else:
        from .glt_dual_static import materialize_dual_geometry
        result.geometry_valid = bool(static['geometry_valid'])
        result.geometry_invalid_reason = str(static['geometry_invalid_reason'])
        geometry = materialize_dual_geometry(static, trimer.trimer_pos)
        if bool(geometry['geometry_valid']) != result.geometry_valid:
            raise ValueError('dual static geometry validity disagrees with materialization')
        result.bond_z_a = torch.from_numpy(np.array(static['token_z_a'], dtype=np.int16, copy=True))
        result.bond_z_b = torch.from_numpy(np.array(static['token_z_b'], dtype=np.int16, copy=True))
        result.bond_distance = geometry['bond_distance']
        result.bond_type = torch.from_numpy(np.array(static['token_bond_type'], dtype=np.int8, copy=True))
        result.bond_center = torch.from_numpy(np.array(static['token_center_mask'], dtype=bool, copy=True))
        result.line_source = torch.from_numpy(np.array(static['line_source'], dtype=np.int64, copy=True))
        result.line_target = torch.from_numpy(np.array(static['line_target'], dtype=np.int64, copy=True))
        result.line_path = torch.from_numpy(np.array(static['line_path'], dtype=np.int64, copy=True))
        result.line_angle = geometry['line_angle']
        result.line_mask = torch.from_numpy(np.array(static['line_path_mask'], dtype=bool, copy=True))
        result.line_path_group = torch.from_numpy(np.array(static['line_path_group'], dtype=np.int64, copy=True))
        result.line_is_self = torch.from_numpy(np.array(static['line_is_self'], dtype=bool, copy=True))
    # Physical endpoint identities come from frozen coordinate rows and RU tables.
    if result.geometry_valid:
        if static is not None:
            pa = torch.as_tensor(np.array(static['token_pos_index_a'], copy=True)).long()
            pb = torch.as_tensor(np.array(static['token_pos_index_b'], copy=True)).long()
            angle_pos = torch.as_tensor(np.array(static['angle_pos_triplet'], copy=True)).long()
        else:
            from .glt_dual_static import _token_position_indices, _angle_triplets
            pa0, pb0 = _token_position_indices(row['tokens'], topology, trimer)
            pa, pb = torch.from_numpy(pa0).long(), torch.from_numpy(pb0).long()
            angle_pos = torch.from_numpy(_angle_triplets(result.line_path.numpy(), result.line_mask.numpy(), result.line_is_self.numpy(), pa0, pb0)).long()
        offset = torch.as_tensor(trimer.trimer_ru_offset).long()
        base = torch.as_tensor(trimer.trimer_base_ru_atom_id).long()
        z = torch.as_tensor(trimer.trimer_atomic_number).long()
        central = torch.as_tensor(trimer.mips_to_trimer_central_index).long()
        inverse = {int(base[pos]): canonical for canonical, pos in enumerate(central.tolist())}
        if len(inverse) != result.mips_x.size(0):raise ValueError('nonunique canonical physical identity')
        ca=torch.tensor([inverse[int(base[pos])] for pos in pa.tolist()])
        cb=torch.tensor([inverse[int(base[pos])] for pos in pb.tolist()])
        a0=(offset[pa]==0)&(z[pa]>1);b0=(offset[pb]==0)&(z[pb]>1)
        rows=torch.arange(pa.numel())
        result.incidence_index=torch.stack([torch.cat([rows[a0],rows[b0]]),torch.cat([ca[a0],cb[b0]])])
        result.bond_canonical_a=ca;result.bond_canonical_b=cb
        result.bond_length_weight=(a0.float()+b0.float())*.5
        shared=angle_pos[:,0,1]
        result.line_center_angle=(shared>=0)&(offset[shared.clamp_min(0)]==0)&(z[shared.clamp_min(0)]>1)
    else:
        result.incidence_index=torch.empty((2,0),dtype=torch.long)
        result.bond_canonical_a=result.bond_canonical_b=torch.empty(0,dtype=torch.long)
        result.bond_length_weight=torch.empty(0)
        result.line_center_angle=torch.zeros(result.line_path.size(0),dtype=torch.bool)
    if geometry_mode == 'periodic_template':
        from .glt_periodic_geometry import attach_periodic_template
        if result.geometry_valid:
            attach_periodic_template(result, offset[pa], offset[pb], angle_pos, offset)
        else:
            attach_periodic_template(result, torch.empty(0,dtype=torch.long), torch.empty(0,dtype=torch.long), torch.empty((0,2,3),dtype=torch.long), torch.empty(0,dtype=torch.long))
    result.geometry_mode = geometry_mode
    if torsion_revision not in (None,'glt_physical_torsion_r1') or (torsion_revision and (torsion_mode!='on' or geometry_mode!='finite_incident')):
        raise ValueError('invalid physical torsion FT mode')
    if torsion_revision:
        from .glt_dual_pretrain import attach_physical_torsion
        attach_physical_torsion(result,static,trimer,trimer.trimer_pos)
    elif torsion_mode is not None:
        # Deployment-side torsion fields: real values from the frozen clean
        # geometry, via the same canonical bond table and physical-bond
        # identity as the pretraining path.  A deployment that carries the
        # torsion modules is never silently fed an empty field.
        from .glt_dual_pretrain import _torsion_fields, _topology_z, periodic_bond_table
        mode = str(torsion_mode).lower()
        if mode not in {'on', 'off'}:
            raise ValueError('torsion mode must be on or off')
        normalized_text = getattr(topology, 'normalized_canonical_smiles', None)
        if normalized_text is None:
            raise ValueError('torsion fields require the frozen normalized SMILES')
        bond_table = periodic_bond_table(
            str(normalized_text), topology.canonical_to_trimer_base_atom_id)
        mapping = torch.as_tensor(
            getattr(trimer, 'mips_to_trimer_central_index', torch.empty(0)),
            dtype=torch.long).reshape(-1)
        if result.geometry_valid:
            index, values, valid = _torsion_fields(
                static, _topology_z(topology).tolist(), trimer.trimer_pos, mapping, mode,
                bond_table)
        else:
            index = torch.zeros(0, dtype=torch.long)
            values = torch.zeros((0, 2), dtype=torch.float32)
            valid = torch.zeros(0, dtype=torch.bool)
        result.torsion_bond_index = index
        result.torsion_values = values
        result.torsion_mask = valid
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
    def __init__(self, cache_root, cohort, *, static_cache=None, target_cache=None):
        from .glt_dual_cache import DualFrozenBundle

        self.bundle = self.topology = self.trimer = None
        self.static_cache = static_cache
        self.target_cache = target_cache
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

    def static_for(self, index):
        if self.static_cache is None:
            return None
        key, _ = self.samples[int(index)]
        return self.static_cache.get_by_key(key)

    def target_for(self, index):
        if self.target_cache is None:
            return None
        key, _ = self.samples[int(index)]
        return self.target_cache.get_by_key(key)

    def close(self):
        bundle = self.bundle
        self.bundle = self.topology = self.trimer = None
        static_cache = getattr(self, 'static_cache', None)
        target_cache = getattr(self, 'target_cache', None)
        self.static_cache = self.target_cache = None
        try:
            if bundle is not None:
                bundle.close()
        finally:
            if static_cache is not None:
                static_cache.close()
            if target_cache is not None:
                target_cache.close()


def build_dual_sample(topology, trimer, smiles, **kwargs):
    data = _build_dual_sample_without_md(topology, trimer, smiles, **kwargs)
    if hasattr(topology, 'geometry_quality_eligible') and not bool(topology.geometry_quality_eligible):
        data.geometry_valid=False
        data.geometry_invalid_reason='quality_ineligible'
        data.bond_length_weight.zero_()
        data.bond_center.zero_()
        data.line_center_angle.zero_()
    if hasattr(topology, 'md200'):
        data.md200 = topology.md200.clone()
        data.md200_valid = topology.md200_valid.clone()
    return data


def dual_glt_collate(samples):
    if not samples:
        raise ValueError('empty dual batch')
    batch = Data()
    fields = {}
    atom_offset = bond_offset = relation_offset = 0
    periodic_offset = periodic_relation_offset = 0
    modes = {getattr(item, "geometry_mode", "finite_internal") for item in samples}
    if len(modes) != 1: raise ValueError("mixed geometry modes")
    batch.geometry_mode = next(iter(modes))
    has_torsion = hasattr(samples[0], 'torsion_values')
    if not all(hasattr(item, 'torsion_values') == has_torsion for item in samples):
        raise ValueError('torsion fields must be present on every sample or none')
    torsion_index, torsion_values, torsion_mask = [], [], []
    for graph, item in enumerate(samples):
        n, m = item.mips_x.size(0), item.bond_distance.numel()
        for name in ('mips_x', 'mips_backbone_mask', 'lga_spd', 'lga_path_mask',
                     'lga_path_shift', 'lga_source_image_shift', 'bond_path_features',
                     'bond_path_mask', 'bond_z_a', 'bond_z_b', 'bond_distance',
                     'bond_type', 'bond_center', 'line_angle', 'line_mask', 'line_is_self',
                     'bond_length_weight', 'line_center_angle'):
            fields.setdefault(name, []).append(getattr(item, name))
        fields.setdefault('lga_relation_mask', []).append(getattr(
            item, 'lga_relation_mask', torch.zeros(item.lga_spd.numel(), dtype=torch.bool)))
        fields.setdefault('lga_edge_index', []).append(item.lga_edge_index + atom_offset)
        fields.setdefault('incidence_index', []).append(item.incidence_index + torch.tensor([[bond_offset], [atom_offset]]))
        for endpoint in ('bond_canonical_a', 'bond_canonical_b'):
            fields.setdefault(endpoint, []).append(getattr(item, endpoint) + atom_offset)
        for name, offset in [('lga_path_index', atom_offset), ('line_path', bond_offset)]:
            value = getattr(item, name)
            fields.setdefault(name, []).append(torch.where(value >= 0, value + offset, value))
        for name in ('line_source', 'line_target'):
            fields.setdefault(name, []).append(getattr(item, name) + bond_offset)
        fields.setdefault('line_path_group', []).append(item.line_path_group + relation_offset)
        fields.setdefault('canonical_graph_index', []).append(torch.full((n,), graph, dtype=torch.long))
        fields.setdefault('bond_batch', []).append(torch.full((m,), graph, dtype=torch.long))
        if has_torsion:
            torsion_index.append(item.torsion_bond_index + bond_offset)
            torsion_values.append(item.torsion_values)
            torsion_mask.append(item.torsion_mask)
        if batch.geometry_mode == 'periodic_template':
            for name in ('periodic_observation_anchor','periodic_observation_weight','periodic_line_shift','periodic_line_path_shift','periodic_line_angle','periodic_line_mask','periodic_line_self'):
                fields.setdefault(name, []).append(getattr(item,name))
            fields.setdefault('periodic_bond_index',[]).append(item.periodic_bond_index+periodic_offset)
            keys=item.periodic_keys.clone();keys[:,:2]+=atom_offset
            fields.setdefault('periodic_keys',[]).append(keys)
            fields.setdefault('periodic_incidence_index',[]).append(item.periodic_incidence_index+torch.tensor([[periodic_offset],[atom_offset]]))
            for name in ('periodic_line_source','periodic_line_target'):
                fields.setdefault(name,[]).append(getattr(item,name)+periodic_offset)
            value=item.periodic_line_path
            fields.setdefault('periodic_line_path',[]).append(torch.where(value>=0,value+periodic_offset,value))
            fields.setdefault('periodic_line_group',[]).append(item.periodic_line_group+periodic_relation_offset)
            fields.setdefault('periodic_bond_batch',[]).append(torch.full((item.periodic_keys.size(0),),graph,dtype=torch.long))
            periodic_offset+=item.periodic_keys.size(0)
            periodic_relation_offset+=item.periodic_line_source.numel()
        atom_offset += n
        bond_offset += m
        relation_offset += item.line_source.numel()
    for name, values in fields.items():
        setattr(batch, name, torch.cat(values, dim=1 if name in ('lga_edge_index', 'incidence_index', 'periodic_incidence_index') else 0))
    revisions=[getattr(item,'torsion_revision',None) for item in samples]
    if any(r!=revisions[0] for r in revisions):raise ValueError('mixed torsion revisions')
    if revisions[0]:
        batch.torsion_revision=revisions[0]
        batch.torsion_counts=torch.cat([item.torsion_counts for item in samples],dim=0)
        batch.torsion_available=torch.cat([item.torsion_available for item in samples],dim=0)
    if has_torsion:
        batch.torsion_bond_index = torch.cat(torsion_index, dim=0)
        batch.torsion_values = torch.cat(torsion_values, dim=0)
        batch.torsion_mask = torch.cat(torsion_mask, dim=0)
    batch.graph_available = torch.tensor([x.graph_available for x in samples], dtype=torch.bool)
    batch.geometry_valid = torch.tensor([x.geometry_valid for x in samples], dtype=torch.bool)
    batch.geometry_invalid_reason = [x.geometry_invalid_reason for x in samples]
    revisions=[getattr(item,'nonbonded_revision',None) for item in samples]
    if any(r!=revisions[0] for r in revisions):raise ValueError('mixed nonbonded input revisions')
    if revisions[0]:
        offset=0;indices=[]
        for item in samples:
            indices.append(item.nonbonded_bond_index+offset);offset+=item.bond_distance.numel()
        batch.nonbonded_revision=revisions[0]
        batch.nonbonded_bond_index=torch.cat(indices)
        batch.nonbonded_distance=torch.cat([item.nonbonded_distance for item in samples])
        batch.nonbonded_mask=torch.cat([item.nonbonded_mask for item in samples])
    if all(hasattr(x, 'y') and x.y is not None for x in samples):
        batch.y = torch.stack([x.y for x in samples])
    has_md = [hasattr(item, 'md200') for item in samples]
    if any(has_md):
        if not all(has_md):
            raise ValueError('MD200 must be present for every graph')
        batch.md200 = torch.cat([item.md200 for item in samples], dim=0)
        batch.md200_valid = torch.cat([item.md200_valid for item in samples], dim=0)
        if batch.md200.shape != (len(samples), 200) or batch.md200_valid.shape != (len(samples),):
            raise ValueError('MD200 collate shape mismatch')
    return batch


def build_glt_2d_sample(topology, smiles, *, static=None):
    """O8-only chemical fields; no geometry carrier or coordinate access."""
    names=('mips_x','mips_backbone_mask','lga_edge_index','lga_spd','lga_path_index',
           'lga_path_mask','lga_path_shift','lga_source_image_shift')
    if any(not hasattr(topology,k) for k in names):raise ValueError('incomplete O8 topology')
    n=topology.mips_x.size(0)
    if not torch.equal(topology.canonical_ru_atom_index.long(),torch.arange(n)):raise ValueError('noncanonical O8 topology')
    data=Data(**{k:getattr(topology,k).clone() for k in names})
    data.graph_available=bool(topology.graph_available)
    data.lga_relation_mask=getattr(topology,'lga_relation_mask',torch.zeros(data.lga_spd.numel(),dtype=torch.bool)).clone()
    if static is None:data.bond_path_features,data.bond_path_mask=bond_paths(topology,smiles)
    else:
        if any(k not in static for k in ('bond_path_features','bond_path_mask')):raise ValueError('missing 2D bond chemistry')
        data.bond_path_features=torch.from_numpy(np.array(static['bond_path_features'],dtype=np.float32,copy=True))
        data.bond_path_mask=torch.from_numpy(np.array(static['bond_path_mask'],dtype=bool,copy=True))
    if not torch.isfinite(data.mips_x).all() or data.bond_path_features.shape!=(data.lga_spd.numel(),2,14) or data.bond_path_mask.shape!=(data.lga_spd.numel(),2):raise ValueError('invalid 2D feature schema')
    for value in (data.lga_edge_index,data.lga_path_index[data.lga_path_mask.bool()]):
        if value.numel() and (value.min()<0 or value.max()>=n):raise ValueError('O8 index out of range')
    return data


def glt_2d_collate(samples):
    """Offset only canonical atoms; physical geometry is never copied."""
    if not samples:raise ValueError('empty 2D batch')
    fields={};offset=0
    for graph,item in enumerate(samples):
        n=item.mips_x.size(0)
        for k in ('mips_x','mips_backbone_mask','lga_spd','lga_path_mask','lga_path_shift',
                  'lga_source_image_shift','lga_relation_mask','bond_path_features','bond_path_mask'):
            fields.setdefault(k,[]).append(getattr(item,k))
        fields.setdefault('lga_edge_index',[]).append(item.lga_edge_index+offset)
        fields.setdefault('lga_path_index',[]).append(torch.where(item.lga_path_index>=0,item.lga_path_index+offset,item.lga_path_index))
        fields.setdefault('canonical_graph_index',[]).append(torch.full((n,),graph,dtype=torch.long))
        offset+=n
    data=Data(**{k:torch.cat(v,dim=1 if k=='lga_edge_index' else 0) for k,v in fields.items()})
    data.graph_available=torch.tensor([s.graph_available for s in samples],dtype=torch.bool)
    if all(getattr(s,'y',None) is not None for s in samples):data.y=torch.stack([s.y for s in samples])
    return data
