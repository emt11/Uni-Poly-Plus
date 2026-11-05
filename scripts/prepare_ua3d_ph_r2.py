#!/usr/bin/env python3
"""Exactly one CPU, label-free structure preparation for PH r2."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare():
    import numpy as np
    import torch
    from src.dataset import ua3d_ph_r2 as d
    from src.dataset.ua3m_paperfuse import InputCache, validate_missing_inventory
    from src.dataset.mcl_ph_view import build_trimer_view, centre_mapping, physical_bonds
    from src.training import ua3d_ph_r2 as t
    if not os.environ.get('TMUX') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('prepare requires GPU-hidden tmux')
    folder = t.OUT / 'inputs'
    folder.mkdir(exist_ok=False)
    state = dict(status='RUNNING', prepare=1, prepare_rows=0, prepare_a_reductions=0, prepare_b_reductions=0,
                 geometry_available=0, geometry_missing=0, model_forwards=0, labels_accessed=False, **t.contract())
    def record():
        t.write(folder / 'consumption.json', state)
    record(); cache = None; source = None
    try:
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        cache = InputCache(t.PARENT / 'inputs'); source = d.StructureReader(t.DOWN)
        missing = []; rows = {}; reusable = {}; all_router = {}
        for task in t.TASKS:
            rows[task] = []; all_router[task] = []
            entries = sorted(cache.rows[task], key=lambda x: x['row'])
            if [e['row'] for e in entries] != list(range(len(entries))):
                raise ValueError('sidecar row coverage')
            (folder / task).mkdir()
            for e in entries:
                t.guard_deadline(); row, key = e['row'], e['key']
                raw = cache.get(task, row, key, e['smiles']); graph = raw['two_d']; record_ = source.get(key)
                if record_.normalized_smiles != e['smiles'] or bool(raw['available']) != (record_.final_geometry_status == 'PASS'):
                    raise ValueError('frozen structure identity/availability')
                topology, trimer, static = record_.topology, record_.trimer, record_.static
                for name in ('mips_x', 'lga_path_index', 'lga_path_shift', 'lga_path_mask', 'lga_spd'):
                    if not torch.equal(getattr(graph, name), getattr(topology, name)):
                        raise ValueError('canonical topology differs: ' + name)
                if key in reusable:
                    side = dict(reusable[key])
                    if side['available'] != e['available']:
                        raise ValueError('shared structural key availability')
                else:
                    g = d.lifted_graph(graph.lga_path_index.numpy(), graph.lga_path_shift.numpy(), graph.lga_path_mask.numpy(), graph.lga_spd.numpy(), len(graph.mips_x))
                    q = d.random_walk_pe(g['nodes'], g['edges']); events = []
                    for k in (1, 3, 7, 15):
                        state['prepare_a_reductions'] += 1; record()
                        events.append(d.graph_events(q[:, k], g['edges']))
                    side = dict(schema=d.SCHEMA, available=bool(e['available']), graph_available=True,
                                rwpe=torch.from_numpy(q), edges=g['edges'], center=torch.from_numpy(g['center']).long(), events=events,
                                canonical_atoms=len(graph.mips_x), cross_ru=g['cross_ru'])
                    router = np.zeros((31, 5), np.float32)
                    if e['available']:
                        view = build_trimer_view(trimer); central, inverse, unmatched = centre_mapping(topology, trimer, view)
                        ends, codes, center_mask = physical_bonds(static, view)
                        geo = raw['three_d']; coords = view.positions.numpy().astype(np.float64)
                        actual = np.linalg.norm(coords[ends[:, 0]] - coords[ends[:, 1]], axis=1)
                        if len(actual) != len(geo.bond_distance) or not np.allclose(actual, geo.bond_distance.numpy(), rtol=0, atol=1e-5):
                            raise ValueError('physical bond length/order differs from frozen BASE')
                        # Cached canonical endpoints also describe noncentral copies. Compare
                        # base identities, rather than incorrectly collapsing their positions.
                        canonical_base = torch.as_tensor(topology.canonical_to_trimer_base_atom_id).long()
                        for idx, pair in enumerate(ends.tolist()):
                            actual_base = sorted(view.base_id[pair].tolist())
                            expected_base = sorted([int(canonical_base[geo.bond_canonical_a[idx]]), int(canonical_base[geo.bond_canonical_b[idx]])])
                            if actual_base != expected_base:
                                raise ValueError('canonical/physical endpoint mismatch')
                        state['prepare_b_reductions'] += 1; record()
                        vr = d.vr_betti(coords); router = np.concatenate((d.non_ph_curves(coords), vr['curves']), -1)
                        distances = np.linalg.norm(coords[:, None] - coords[None], axis=-1); spatial_edges = []; spatial_distances = []
                        for cutoff in d.CUTOFFS:
                            a, b = np.where((distances > 1e-8) & (distances <= cutoff))
                            spatial_edges.append(torch.from_numpy(np.stack((a, b))).long())
                            spatial_distances.append(torch.as_tensor(distances[a, b], dtype=torch.float32))
                        side.update(physical_atoms=view.count, bond_atoms=ends.T.contiguous(), incidence=geo.incidence_index.clone(),
                                    canonical_graph=torch.zeros(len(graph.mips_x), dtype=torch.long), sp_edges=spatial_edges,
                                    sp_distances=spatial_distances, intervals=vr['intervals'], physical_positions=coords,
                                    physical_z=view.z, physical_ru_offset=view.offset, physical_base_id=view.base_id,
                                    center_physical=central, unmatched_heavy=unmatched)
                    side['router_raw'] = router; reusable[key] = dict(side)
                # A repeated row is tied to the same frozen direct LMDB key and all canonical tensors
                # above have been compared. The key maps to one immutable coordinate payload.
                side.update(task=task, row=row, key=key, source_coordinate_key=key)
                torch.save(side, folder / task / f'row{row}.pt')
                rows[task].append(dict(row=row, key=key, available=e['available'], source_coordinate_key=key))
                all_router[task].append(side['router_raw']); state['prepare_rows'] += 1
                state['geometry_available' if e['available'] else 'geometry_missing'] += 1
                if not e['available']:
                    missing.append(dict(task=task, row=row, key=key, smiles=e['smiles'], reason=record_.final_geometry_status))
                if state['prepare_rows'] % 64 == 0:
                    record(); print(state['prepare_rows'], flush=True)
                if state['prepare_rows'] > 5875 or state['prepare_a_reductions'] > 23500 or state['prepare_b_reductions'] > 5860:
                    raise RuntimeError('prepare budget')
            cache.memory.clear()
        validate_missing_inventory(missing, state['prepare_rows'], state['geometry_available'])
        for task in t.TASKS:
            for fold, split in enumerate(t.read(t.SPLIT / f'{task}.json')['folds']):
                t.write(folder / 'scalers' / task / f'fold{fold}.json', d.fit_router(all_router[task], split['train_indices']))
        state['status'] = 'COMPLETE'; record()
        t.write(folder / 'rows.json', rows); t.write(folder / 'geometry_missing.json', missing)
        t.write(folder / 'coverage.json', dict(status='COMPLETE', records=5875, geometry_available=5860, geometry_missing=15))
        t.write(folder / 'metadata.json', dict(schema=d.SCHEMA, status='COMPLETE', records=5875, geometry_available=5860, geometry_missing=15,
            labels_cached=False, labels_accessed=False, model_forward=0, source=str(t.DOWN), original_cache=str(t.PARENT / 'inputs'),
            graph_ph='one-dimensional chemical complex; event encoding then distribution; essential horizon1 censored',
            vr_ph='F2 2-skeleton, H0/H1 only; right_censored infinity; birth<=tau<death', physical_identity='frozen direct LMDB key + canonical tensor/endpoint/length checks', **t.contract()))
    except BaseException:
        import traceback
        state.update(status='FAILED', error=traceback.format_exc()); record(); raise
    finally:
        if cache is not None:
            cache.close()
        if source is not None:
            source.close()


if __name__ == '__main__':
    prepare()
