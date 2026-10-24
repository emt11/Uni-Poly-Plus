#!/usr/bin/env python3
"""Build the read-only periodic-image sidecar for UA2D-3DSHIFT (one CPU pass)."""
import argparse
from collections import deque
import os
import concurrent.futures
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_SOURCE = _TEMPLATES = None


def _worker_init():
    import lmdb
    import torch
    from src.dataset.mcl_ph_direct import DirectMCLSource

    global _SOURCE, _TEMPLATES
    torch.set_num_threads(1)
    from src.training.ua2d_3dshift import CACHE, SOURCE, SIDECAR, process_identity, write
    write(SIDECAR/f'worker_{os.getpid()}.json',dict(process=process_identity(os.getpid()),cwd=str(ROOT)))
    _SOURCE = DirectMCLSource(SOURCE, task=None, selected_indices=None, split_manifest=None,
                              verify_key_digest=False, readahead=True)
    _SOURCE._open_environment()
    _TEMPLATES = lmdb.open(str(CACHE / 'templates.lmdb'), readonly=True, lock=False,
                           readahead=True, max_readers=128)


def _numpy(value):
    import numpy as np
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def covering_relations(keys):
    """Adjacency of b^0 from canonical endpoint images; never collapse images."""
    edges=set()
    for g,(a,b,shift) in enumerate(keys):
        for atom,image in ((a,0),(b,shift)):
            for h,(c,d,delta) in enumerate(keys):
                for endpoint,relative in ((c,0),(d,delta)):
                    if endpoint==atom:
                        q=image-relative
                        if (g,h,q)!=(g,g,0):edges.add((g,h,q))
    return edges


def validated_identity(identity):
    if not isinstance(identity,dict) or not isinstance(identity.get('normalized_smiles'),str):
        raise ValueError('validated normalized identity dictionary required')
    return identity['normalized_smiles']


def build_record(template, trimer, key, logical_slot, source_row, topology=None):
    """Materialize exact covering identities and existing physical observations only."""
    import numpy as np
    offsets=_numpy(trimer.trimer_ru_offset).astype(np.int64).reshape(-1)
    base=_numpy(trimer.trimer_base_ru_atom_id).astype(np.int64).reshape(-1)
    central=_numpy(trimer.mips_to_trimer_central_index).astype(np.int64).reshape(-1)
    atoms=int(_numpy(template['atom_label']).size)
    if central.shape!=(atoms,) or len(offsets)!=len(base) or len(set(base[central].tolist()))!=atoms:
        raise ValueError('canonical atom correspondence mismatch')
    inverse={int(base[pos]):i for i,pos in enumerate(central)}
    if topology is not None:
        expected=_numpy(topology.canonical_to_trimer_base_atom_id).reshape(-1)
        if not np.array_equal(base[central],expected):raise ValueError('topology canonical/base mapping mismatch')
    ca=np.asarray([inverse.get(int(x),-1) for x in base],np.int64)
    pa=_numpy(template['endpoint_a']).astype(np.int64).reshape(-1)
    pb=_numpy(template['endpoint_b']).astype(np.int64).reshape(-1)
    group=_numpy(template['group_index']).astype(np.int64).reshape(-1)
    first=_numpy(template['group_first']).astype(np.int64).reshape(-1)
    tokens=len(pa);groups=len(first)
    if not tokens or len(pb)!=tokens or len(group)!=tokens or min(pa.min(),pb.min())<0 or max(pa.max(),pb.max())>=len(base):
        raise ValueError('physical token endpoints invalid')
    if (ca[pa]<0).any() or (ca[pb]<0).any():raise ValueError('bond endpoint lacks canonical atom correspondence')
    if group.min()<0 or group.max()>=groups or not np.array_equal(group[first],np.arange(groups)):
        raise ValueError('canonical group/representative invalid')
    normalized=[];anchors=[]
    for a,b in zip(pa,pb):
        forward=(int(ca[a]),int(ca[b]),int(offsets[b]-offsets[a]))
        reverse=(forward[1],forward[0],-forward[2])
        normalized.append(min(forward,reverse));anchors.append(int(offsets[a] if forward<=reverse else offsets[b]))
    keys=[normalized[i] for i in first]
    if len(set(keys))!=groups:raise ValueError('two canonical groups share one bond key')
    if any(normalized[i]!=keys[g] for i,g in enumerate(group)):raise ValueError('frozen corruption/canonical partition mismatch')
    clean=template.get('clean')
    if clean is not None and not bool(clean.geometry_valid):raise ValueError('excluded invalid geometry in sidecar input')
    if clean is not None and (not np.array_equal(_numpy(clean.bond_canonical_a),ca[pa]) or not np.array_equal(_numpy(clean.bond_canonical_b),ca[pb])):
        raise ValueError('template canonical atom mapping mismatch')
    central_mask=np.zeros(len(base),bool);central_mask[central]=True
    heavy=_numpy(trimer.trimer_atomic_number).reshape(-1)>1
    token_weight=((central_mask[pa]&heavy[pa]).astype(float)+(central_mask[pb]&heavy[pb]).astype(float))*.5
    if clean is not None and not np.array_equal(token_weight,_numpy(clean.bond_length_weight)):
        raise ValueError('physical center-incidence weights mismatch')
    group_weight=np.bincount(group,weights=token_weight,minlength=groups).astype(np.float32)
    adjacent=covering_relations(keys)
    relations={r:[] for r in adjacent}
    relations.update({(g,g,0):[] for g in range(groups)})
    incidence={}
    for token in range(tokens):
        for position in (int(pa[token]),int(pb[token])):incidence.setdefault(position,[]).append(token)
    skipped=0
    for position,incident in incidence.items():
        if not (central_mask[position] and heavy[position]):
            continue
        for source in incident:
            left=int(pb[source] if pa[source]==position else pa[source])
            for target in incident:
                if source==target:continue
                right=int(pb[target] if pa[target]==position else pa[target])
                if left==right:skipped+=1;continue
                relation=(int(group[source]),int(group[target]),anchors[target]-anchors[source])
                if relation not in adjacent:raise ValueError('physical observation contradicts covering relation')
                relations[relation].append((source,target,left,position,right))
    if any(not relations[r] for r in adjacent):raise ValueError('covering local relation lacks an existing physical observation')
    ordered=sorted(relations);index={r:i for i,r in enumerate(ordered)}
    obs_source=[];obs_target=[];obs_pos=[];counts=[]
    for r in ordered:
        observations=sorted(set(relations[r]));counts.append(len(observations))
        for source,target,a,b,c in observations:
            obs_source.append(source);obs_target.append(target);obs_pos.append((a,b,c))
    paths={}
    by_source={g:[] for g in range(groups)}
    for r in sorted(adjacent):by_source[r[0]].append(r)
    for left in sorted(adjacent):
        for right in by_source[left[1]]:
            relation=(left[0],right[1],left[2]+right[2])
            if relation in index:continue # Exact image relation already reachable at distance0/1.
            paths.setdefault(relation,set()).add((index[left],index[right]))
    ordered2=sorted(paths);path_count=[len(paths[r]) for r in ordered2]
    all_paths=[p for r in ordered2 for p in sorted(paths[r])]
    ib=[g for g in range(groups) for _ in range(2)];ia=[atom for a,b,_ in keys for atom in (a,b)]
    return dict(schema='ua2d-3dshift-input-r1',sample_key=key,logical_slot=int(logical_slot),source_row=int(source_row),
        identity=str(template['identity']),token_count=tokens,group_count=groups,atom_count=atoms,
        rel1_count=len(ordered),rel2_count=len(ordered2),obs_count=len(obs_source),path_count=len(all_paths),
        geometry_valid=True,invalid_reason=None,skipped_parallel_angle=skipped,
        token_group=group,group_first=first,token_weight=token_weight.astype(np.float32),
        group_atom_a=np.asarray([r[0] for r in keys],np.int64),group_atom_b=np.asarray([r[1] for r in keys],np.int64),
        group_shift=np.asarray([r[2] for r in keys],np.int64),group_weight=group_weight,
        group_self=np.asarray([a==b for a,b,_ in keys],bool),
        rel1_source=np.asarray([r[0] for r in ordered],np.int64),rel1_target=np.asarray([r[1] for r in ordered],np.int64),
        rel1_shift=np.asarray([r[2] for r in ordered],np.int64),rel1_self=np.asarray([a==b and q==0 for a,b,q in ordered],bool),
        rel1_obs_count=np.asarray(counts,np.int64),obs_source=np.asarray(obs_source,np.int64),obs_target=np.asarray(obs_target,np.int64),
        obs_pos=np.asarray(obs_pos,np.int64).reshape(-1,3),rel2_source=np.asarray([r[0] for r in ordered2],np.int64),
        rel2_target=np.asarray([r[1] for r in ordered2],np.int64),rel2_shift=np.asarray([r[2] for r in ordered2],np.int64),
        rel2_path_count=np.asarray(path_count,np.int64),rel2_path=np.asarray(all_paths,np.int64).reshape(-1,2),
        inc_bond=np.asarray(ib,np.int64),inc_atom=np.asarray(ia,np.int64))


def prepare_chunk(items):
    from src.dataset.ua2d_fast_pretrain import decode_template
    from src.training.glt_dual_runtime import pretrain_source_fields
    output = []
    for position, logical_slot, key, source_row in items:
        with _TEMPLATES.begin() as txn:
            payload = txn.get(key)
        if payload is None:
            raise ValueError('prepared template missing for the saved logical slot')
        template = decode_template(payload)
        if template['sample_key'] != key:
            raise ValueError('prepared template identity mismatch')
        top, tri, text, static, _, identity = pretrain_source_fields(_SOURCE, source_row,
                                                                    include_identity=True)
        if _SOURCE.samples[source_row][0] != key:
            raise ValueError('source row/key disagrees with the saved logical mapping')
        if str(template['identity']) != validated_identity(identity):
            raise ValueError('source/template identity mismatch')
        record = build_record(template, tri, key, logical_slot, source_row, topology=top)
        output.append((key, pickle.dumps(record, protocol=4)))
    return output


def main():
    from src.training.ua2d_3dshift import REVISION
    if REVISION in ('r1.3-RECOVERY1','r1.5-ALL1-PCTRL-GPU2'):raise PermissionError('recovery1 reuses COMPLETE cache; prepare forbidden')
    import lmdb
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from src.training.glt_dual_runtime import require_tmux, load_sample_index_artifact
    from src.training.ua2d_3dshift import CACHE, PSCALE, SIDECAR, REVISION, now, write, cache_references
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    require_tmux()
    if not 1 <= args.workers <= 8:
        raise ValueError('prepare worker bound is 8')
    cache_references(require_sidecar=False)
    output = SIDECAR
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cache_meta = json.loads((CACHE / 'metadata.json').read_text())
    mapping_meta = json.loads((PSCALE / 'metadata.json').read_text())
    if cache_meta.get('train_rows') != 911391 or cache_meta.get('status') != 'COMPLETE':
        raise ValueError('prepared template cache is not complete')
    if (mapping_meta.get('plan') != 'UA2D-PTSCALE-20261022-01' or mapping_meta.get('status') != 'COMPLETE'
            or mapping_meta.get('records') != 910756 or mapping_meta.get('excluded') != 635):
        raise ValueError('logical mapping is not complete')
    eligible = np.load(PSCALE / 'eligible_logical.npy', mmap_mode='r', allow_pickle=False)
    keys = np.load(CACHE / 'keys.npy', mmap_mode='r', allow_pickle=False)
    rows = np.load(CACHE / 'source_rows.npy', mmap_mode='r', allow_pickle=False)
    if eligible.shape != (910756,) or keys.shape != (911391,) or rows.shape != (911391,):
        raise ValueError('logical mapping shape changed')
    meta = dict(schema='ua2d-3dshift-input-r1', status='RUNNING', plan='UA2D-3DSHIFT-20261022-01',
                revision=REVISION, source=cache_meta['source'], index=cache_meta['index'],
                parent_cache=str(CACHE.resolve()), mapping=str(PSCALE.resolve()), records=910756, excluded_rows=635,
                workers=args.workers, generated=0, completed=0, started=now(), elapsed_seconds=0.,
                geometry='relative_periodic_image_shift', absolute_anchor_in_model=False,
                online_fallback=False, coverage='every saved eligible logical slot; no skip',
                observation_scope='center_ru_heavy_atoms', max_inflight_chunks=args.workers,
                chunk_records=32, transaction_scope='one_chunk')
    write(output / 'metadata.json', meta)
    environment = lmdb.open(str(output / 'sidecar.lmdb'), map_size=64 * 1024 ** 3, max_readers=8,
                            writemap=False, sync=True, metasync=True)
    position = 0
    generated = 0
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                                                    mp_context=mp.get_context('fork'),
                                                    initializer=_worker_init) as pool:
            pending = deque()
            def submit_chunk():
                nonlocal position
                items = []
                for offset in range(32):
                    index = position + offset
                    if index >= len(eligible):
                        break
                    items.append((index, int(eligible[index]), keys[int(eligible[index])].tobytes(),
                                  int(rows[int(eligible[index])])))
                position += len(items)
                return pool.submit(prepare_chunk, items)
            while position < len(eligible) and len(pending) < args.workers:
                pending.append(submit_chunk())
            while pending:
                records = pending.popleft().result()
                with environment.begin(write=True) as txn:
                    for key, payload in records:
                        if txn.get(key) is not None:
                            raise ValueError('duplicate sidecar key in the logical mapping')
                        txn.put(key, payload)
                        generated += 1
                if time.monotonic()-started>=43200:raise TimeoutError('prepare 12h limit exceeded')
                if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>120*1024**3:raise RuntimeError('prepare disk cap exceeded')
                meta.update(generated=generated, completed=generated,
                            elapsed_seconds=time.monotonic() - started)
                write(output / 'metadata.json', meta)
                if position < len(eligible):
                    pending.append(submit_chunk())
        meta.update(status='COMPLETE', completed=generated,
                    elapsed_seconds=time.monotonic() - started, finished=now(),
                    count_matches_logical_mapping=generated == 910756)
        if generated != 910756:
            raise ValueError('sidecar transaction coverage is incomplete')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>120*1024**3:
            raise RuntimeError('prepare disk cap exceeded')
        environment.sync(True)
        environment.close()
        environment = None
        write(output / 'metadata.json', meta)
        write(output / 'coverage.json',
              dict(records=generated, excluded=635, unique_keys=generated,
                   identity='every saved logical slot has exactly one record', complete=True))
    except BaseException:
        write(output / 'failure.json', dict(error=traceback.format_exc(), retry=False, time=now()))
        write(output / 'metadata.json', dict(meta, status='FAILED', error=traceback.format_exc()))
        if environment is not None:
            environment.close()
        raise


if __name__ == '__main__':
    main()
