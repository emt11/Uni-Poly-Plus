#!/usr/bin/env python3
"""One formal CPU prepare, with exact-key identity comparison before deduplication."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def prepare():
    import concurrent.futures as cf
    import multiprocessing as mp
    import os,time,math
    import numpy as np
    import torch
    from rdkit import Chem
    from src.dataset import ua3d_hmr_r2 as d
    from src.modules import ua3d_hmr_r2 as m
    from src.training import ua3d_hmr_r2 as t
    from src.dataset.ua3d_ph_r2 import StructureReader
    from src.dataset.mcl_ph_view import build_trimer_view,centre_mapping
    t.cpu_contract();t.check_clock(True);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    started=time.time();rows=t.read(t.SIDE/'rows.json');original=t.read(t.INPUT/'rows.json');radius={str(z):float(Chem.GetPeriodicTable().GetRvdw(z)) for z in range(1,119)}
    t.exclusive(t.OUT/'surface/radii.json',dict(unit='angstrom',radii=radius,probe=0))
    structures=[];bykey={};mapping=[];source=StructureReader(t.DOWN);valid_count=0
    try:
        for task in t.TASKS:
            if len(rows[task])!=len(original[task]):raise ValueError('source rows length')
            for e,o in zip(rows[task],original[task]):
                if any(e[k]!=o[k] for k in ('row','key','available')):raise ValueError('original/source key identity')
                row,key=e['row'],e['key'];s=torch.load(t.SIDE/task/f'row{row}.pt',map_location='cpu',weights_only=False)
                if any(s[k]!=v for k,v in dict(schema='ua3d-ph-sidecar-r2',task=task,row=row,key=key,source_coordinate_key=key,available=e['available']).items()):raise ValueError('PH sidecar metadata')
                entry=dict(task=task,row=row,key=key,source_available=e['available'],structure_id=None)
                if e['available']:
                    valid_count+=1
                    pos=np.asarray(s['physical_positions'],dtype=np.float64);z=np.asarray(s['physical_z']);ru=np.asarray(s['physical_ru_offset']);base=np.asarray(s['physical_base_id']);center=np.asarray(s['center_physical']);d.finite(pos)
                    record=source.get(key);view=build_trimer_view(record.trimer);central,_,unmatched=centre_mapping(record.topology,record.trimer,view)
                    expected=(view.positions.numpy().astype(np.float64),view.z.numpy(),view.offset.numpy(),view.base_id.numpy(),central.numpy())
                    if record.final_geometry_status!='PASS' or record.normalized_smiles!=o['smiles'] or any(not np.array_equal(x,y) for x,y in zip((pos,z,ru,base,center),expected)) or unmatched!=s['unmatched_heavy']:raise ValueError('source coordinate/canonical/physical mapping identity')
                    d.validate_center(center,z,ru,unmatched,np.asarray(record.topology.z),base,np.asarray(record.topology.canonical_to_trimer_base_atom_id))
                    payload=dict(pos=pos,z=z,ru=ru,base=base,center=center)
                    matched=None
                    for i in bykey.get(key,[]):
                        prior=structures[i]
                        if all(np.array_equal(payload[k],prior[k]) for k in payload):matched=i;break
                    if matched is None:
                        matched=len(structures);structures.append(dict(id=f's{matched:05d}',key=key,**payload));bykey.setdefault(key,[]).append(matched)
                    entry['structure_id']=structures[matched]['id']
                mapping.append(entry)
                if len(mapping)%128==0:print('identity rows',len(mapping),flush=True);t.check_clock(True)
    finally:source.close()
    if len(mapping)!=5875 or valid_count!=5860:raise ValueError('5875/5860/15 source inventory')
    t.exclusive(t.OUT/'surface/mapping.json',mapping)
    rotations=[]
    for task,fold in [('eps',0),('egc',1)]:
        chosen=0
        for row in t.read(t.SPLIT/f'{task}.json')['folds'][fold]['train_indices']:
            e=next(x for x in mapping if x['task']==task and x['row']==row)
            if not e['source_available']:continue
            source_id=int(e['structure_id'][1:]);s=structures[source_id];angle=np.pi/6;R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
            rotations.append(dict(s,id=f'rot_{task}_{chosen}',pos=s['pos']@R.T,original_id=s['id'],task=task,row=row));chosen+=1
            if chosen==2:break
    if len(structures)!=3646 or len(rotations)!=4:raise ValueError('fixed 3646+4 queue inventory')
    parent=t.ROOT/'results/ua3d_hmr_20261103/r1'
    prior={p.parent.name:t.read(p) for p in (parent/'surface/structures').glob('*/status.json')}
    priority_ids=sorted(k for k,v in prior.items() if not v['available'])+['s00578']
    if len(priority_ids)!=120 or len(set(priority_ids))!=120:raise ValueError('fixed 119 unavailable plus s00578')
    byid={s['id']:s for s in structures};priority=[byid[k] for k in priority_ids];remaining=[s for s in structures if s['id'] not in set(priority_ids)]+rotations
    resources=t.read(t.OUT/'authorization/resources.json');workers=resources['P_geom'];allitems=priority+remaining;results={};timings=[]
    t.exclusive(t.OUT/'surface/queue.json',[{k:v for k,v in s.items() if k not in ('pos','z','ru','base','center')}|dict(atoms=len(s['z']),priority=s['id'] in priority_ids) for s in allitems]);t.exclusive(t.OUT/'surface/priority.json',dict(ids=priority_ids,count=120,not_additional_attempts=True))
    weights={k:{task:0 for task in t.TASKS} for k in byid};totals={task:0 for task in t.TASKS};bad={task:0 for task in t.TASKS}
    for e in mapping:
        if e['source_available']:weights[e['structure_id']][e['task']]+=1;totals[e['task']]+=1
    print('CPU geometry workers',workers,'unique',len(structures),'rotation',len(rotations),'priority',len(priority),flush=True)
    with cf.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:
        futures={};completed_ids=[]
        def submit(s):
            t.check_clock(True)
            f=pool.submit(d.surface_worker,{k:v for k,v in s.items() if k not in ('ru','base')},str(t.OUT/'surface/structures'),radius);futures[f]=s
        try:
            for stage,items in [('priority120',priority),('remaining',remaining)]:
                iterator=iter(items)
                for _ in range(min(workers,len(items))):submit(next(iterator))
                while futures:
                    done,_=cf.wait(futures,timeout=1,return_when=cf.FIRST_COMPLETED);t.check_clock(True)
                    for f in done:
                        s=futures.pop(f);r=f.result();results[s['id']]=r;completed_ids.append(s['id']);timings.append(dict(id=s['id'],atoms=len(s['z']),**r))
                        if not r['available'] and s['id'] in weights:
                            for task in t.TASKS:bad[task]+=weights[s['id']][task]
                        bound=dict(unavailable_rows=sum(bad.values()),maximum_overall_coverage=(5860-sum(bad.values()))/5860,tasks={task:dict(source=totals[task],known_unavailable=bad[task],maximum_coverage=(totals[task]-bad[task])/totals[task]) for task in t.TASKS},completed=len(results),stage=stage)
                        t.write(t.OUT/'surface/coverage_upper_bound.json',bound)
                        if sum(bad.values())>58 or any((totals[task]-bad[task])/totals[task]<.95 for task in t.TASKS):raise ValueError('COVERAGE_IMPOSSIBLE: original-row upper bound below 99%/95%')
                    t.write(t.OUT/'surface/progress.json',dict(completed=len(results),reserved=3650,elapsed_seconds=time.time()-started,workers=workers,stage=stage));t.write(t.OUT/'surface/timing.json',timings)
                    if len(results)%16==0:print('surfaces',len(results),'/3650',stage,flush=True)
                    if t.size(t.OUT/'surface')>48*1024**3:raise ValueError('surface bytes cap')
                    for _ in range(len(done)):
                        s=next(iterator,None)
                        if s is not None:submit(s)
                if stage=='priority120':
                    obs=[dict(id=s['id'],atoms=len(s['z']),seconds=results[s['id']]['elapsed_seconds'],peak_rss_kib=results[s['id']]['peak_rss_kib']) for s in priority];per=max(x['seconds'] for x in obs);critical=per*math.ceil(len(remaining)/workers);eta=1.5*(critical+1800+2400);clock=t.read(t.OUT/'authorization/clock.json');t.write(t.OUT/'gates/prepare_eta.json',dict(observations=obs,remaining=len(remaining),conservative_cpu_seconds=critical,conservative_remaining_with_reserves=eta,prepare_deadline=clock['prepare_deadline_unix'],total_deadline=clock['deadline_unix'],peak_rss_kib=max(x['peak_rss_kib'] for x in obs),full_GPU_matrix_pending_diagnostic_gate=True))
                    if time.time()+1.5*(critical+1800)>=clock['prepare_deadline_unix'] or time.time()+eta>=clock['deadline_unix']:raise TimeoutError('prepare ETA gate')
                    peak=max(x['peak_rss_kib'] for x in obs)*1024
                    if peak>1024**3:
                        # Reducing submission parallelism is allowed; do not redo completed items.
                        workers=max(1,min(workers,int(resources['M_pool']//peak)))
                    t.write(t.OUT/'surface/priority_complete.json',dict(ids=completed_ids.copy(),count=120,elapsed_seconds=time.time()-started))
        except BaseException:
            processes=list(pool._processes.values());stamp=time.time();t.write(t.OUT/'surface/pool_stop.json',dict(time=stamp,processes=[t.identity(p.pid) for p in processes if p.is_alive()],reason=__import__('traceback').format_exc(),no_end_semantics='interrupted; elapsed lower bound from real start to this collection; no fabricated end/exit'))
            for process in processes:
                if process.is_alive():process.terminate()
            for future in futures:future.cancel()
            raise
    t.write(t.OUT/'surface/r1_r2_comparison.json',[dict(id=s['id'],r1=prior.get(s['id'],{}).get('reason','NOT_COMPLETED'),r2=results[s['id']]['reason'],r2_available=results[s['id']]['available']) for s in structures])
    coverage={};status=[]
    for task in t.TASKS:
        entries=[x for x in mapping if x['task']==task];src=[x for x in entries if x['source_available']];ok=sum(results[x['structure_id']]['available'] for x in src);coverage[task]=dict(source=len(src),success=ok,rate=ok/len(src));arrays={k:[] for k in ('lam','C','a','A','Ac','stat','mode_mask','mask')}
        for e in entries:
            s=d.sentinel('SOURCE_MISSING_3D') if not e['source_available'] else None
            r=dict(available=False,reason='SOURCE_MISSING_3D') if s is not None else results[e['structure_id']]
            if s is None:
                with np.load(t.OUT/'surface/structures'/e['structure_id']/'moments.npz') as npz:s={k:npz[k] for k in npz.files}
            for k in arrays:arrays[k].append(float(r['available']) if k=='mask' else s[k])
            status.append(dict(e,available=r['available'],reason=r['reason']))
        dest=t.OUT/'surface/compact';dest.mkdir(exist_ok=True);np.savez(dest/f'{task}.npz',**{k:np.array(v) for k,v in arrays.items()})
    rate=sum(x['success'] for x in coverage.values())/5860;t.write(t.OUT/'surface/status.json',status);t.write(t.OUT/'surface/coverage.json',dict(tasks=coverage,overall=rate,source_valid=5860,source_missing=15,rows=5875));t.write(t.OUT/'surface/timing.json',timings)
    if rate<.99 or any(x['rate']<.95 for x in coverage.values()):raise ValueError('surface coverage gate')
    rotation=[]
    for s in rotations:
        if not results[s['id']]['available'] or not results[s['original_id']]['available']:raise ValueError('rotation requires usable surface')
        with np.load(t.OUT/'surface/structures'/s['id']/'moments.npz') as x:r={k:x[k] for k in x.files}
        with np.load(t.OUT/'surface/structures'/s['original_id']/'moments.npz') as x:o={k:x[k] for k in x.files}
        area=float(abs(r['A']-o['A'])/o['A']);rv=results[s['id']]['audit']['volume'];ov=results[s['original_id']]['audit']['volume'];volume=abs(rv-ov)/max(abs(ov),1e-12);desc=float(np.linalg.norm(d.fixed_descriptor(r)-d.fixed_descriptor(o))/max(np.linalg.norm(d.fixed_descriptor(o)),1e-12));rotation.append(dict(id=s['id'],area=area,volume=volume,descriptor_relative_L2=desc))
    t.write(t.OUT/'surface/rotation.json',rotation)
    if any(x['area']>.05 or x['volume']>.05 or x['descriptor_relative_L2']>.1 for x in rotation):raise ValueError('rotation gate')
    # Qualified compact packing: bounded I/O lanes. Source records are read exactly once.
    with cf.ThreadPoolExecutor(max_workers=resources['R']) as pool:
        fs=[pool.submit(t.pack_cache,task,fold,False) for task in t.TASKS for fold in range(5)]
        for f in cf.as_completed(fs):f.result();t.check_clock(True)
    for task in t.TASKS:
        surf=d.load_surface(t.OUT,task)
        for fold in range(5):
            cache=t.load_cache(task,fold,False);ids=t.partition_positions(cache,'train',task,fold);stats=m.fit_stats(cache,surf,ids);t.exclusive(t.OUT/'scalers'/task/f'fold{fold}.json',stats)
    t.exclusive(t.OUT/'surface/complete.json',dict(status='COMPLETE',elapsed_seconds=time.time()-started,unique_attempts=len(allitems),labels_read_by_geometry=False,encoder_forwards=0))
if __name__=='__main__':prepare()
