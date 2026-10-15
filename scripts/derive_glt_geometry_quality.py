#!/usr/bin/env python3
"""One authorized pass of frozen geometry records, without hashes or regeneration."""
import collections,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/glt_quality_fusion_20261010/r1/quality_completed'
PARTIAL=OUT.parent/'quality/pi1m_quality.jsonl'
BASE=ROOT/'data/processed/mcl_ph_r4_20260930'
ORIGINAL=ROOT/'results/mcl_ph_next_20260929/p0/direct_indices_v1/summary.json'

def classify(record):
 import torch
 tri=record.trimer
 status=record.final_geometry_status
 valid=bool(tri.trimer_geometry_valid)
 if valid != (status=='PASS') or bool(record.static['geometry_valid'])!=valid:raise ValueError('quality validity contradicts final status')
 selected=getattr(tri,'selected_converged',None)
 if selected is not None and not isinstance(selected,bool):raise ValueError('selected convergence not boolean')
 q=dict(final_status=status,geometry_valid=valid,selected_converged=selected,mmff_status=None,quality_eligible=False)
 if not valid:
  q['reason']='explicit_failure';return q
 pos=torch.as_tensor(tri.trimer_pos)
 if pos.ndim!=2 or pos.shape[1]!=3 or not torch.isfinite(pos).all():raise ValueError('successful coordinates corrupt')
 candidates=[c for rnd in getattr(tri,'generation_diagnostics',{}).get('rounds',[]) for c in rnd.get('candidates',[]) if c.get('accepted')]
 if len(candidates)>1:raise ValueError('multiple accepted conformers')
 if candidates:
  c=candidates[0]
  if c['round_id']!=int(tri.trimer_conformer_round_id) or c['candidate_id']!=int(tri.trimer_conformer_candidate_id) or not c.get('final_valid'):raise ValueError('selected conformer mismatch')
  mmff=c.get('mmff_status');q['mmff_status']=mmff
  if mmff not in (None,-1,0,1):raise ValueError('invalid MMFF status')
  if mmff is not None and selected is not None and selected!=(mmff==0):raise ValueError('selected/MMFF contradiction')
  for name in ('finite_post_mmff','mmff_energy_finite','finite_energy'):
   if c.get(name) is False:raise ValueError('accepted candidate failed existing validity checks')
  if mmff is not None:selected=mmff==0
 if selected is None:q['reason']='convergence_unverified'
 elif selected:q.update(quality_eligible=True,reason='selected_converged')
 else:q['reason']='selected_not_converged'
 return q

def selected_qc(db,key,final_status):
    """Use the coordinate-producing lane, never another candidate or failed lane."""
    row=db.execute('SELECT final_status,coordinate_lane,main_detail,retry_detail FROM rows WHERE sample_key=?',(key,)).fetchone()
    if row is None or row[0]!=final_status:raise ValueError('QC identity/final status mismatch')
    if final_status!='PASS':return None
    lane=row[1]
    if lane not in ('main','retry'):raise ValueError('successful QC lacks coordinate lane')
    payload=row[2] if lane=='main' else row[3]
    if payload is None:raise ValueError('selected QC detail missing')
    qc=json.loads(payload)
    if qc.get('selected_converged') is not None and not isinstance(qc['selected_converged'],bool):raise ValueError('QC convergence field invalid')
    candidates=[c for rnd in qc.get('diagnostics',{}).get('rounds',[]) for c in rnd.get('candidates',[]) if c.get('accepted')]
    if len(candidates)!=1:raise ValueError('selected QC candidate not unique')
    c=candidates[0]
    if c['round_id']!=qc['round_id'] or c['candidate_id']!=qc['candidate_id'] or not c.get('final_valid'):raise ValueError('QC selected candidate mismatch')
    if c.get('mmff_status') not in (-1,0,1):raise ValueError('QC selected status missing')
    if qc.get('selected_converged') is not None and qc['selected_converged']!=(c['mmff_status']==0):raise ValueError('QC convergence contradiction')
    return qc


def classify_with_qc(record,qc):
    import copy,math
    if qc is None:return classify(record)
    if not math.isfinite(float(qc['energy'])) or int(qc['atoms'])!=int(record.trimer.trimer_pos.size(0)):raise ValueError('QC atom count/energy mismatch')
    tri=copy.copy(record.trimer)
    for name in ('selected_converged','trimer_conformer_round_id','trimer_conformer_candidate_id'):
        value=qc['selected_converged'] if name=='selected_converged' else qc['round_id'] if name.endswith('round_id') else qc['candidate_id']
        if hasattr(tri,name) and getattr(tri,name)!=value:raise ValueError('cached and selected QC fields contradict')
        if value is not None:setattr(tri,name,value)
    tri.generation_diagnostics=qc['diagnostics']
    view=copy.copy(record);view.trimer=tri
    result=classify(view);result['quality_metadata_source']='selected_coordinate_lane_QC'
    return result


def complete_persisted_quality(previous,qc):
    """Reuse durable validity evidence without traversing its LMDB record again."""
    import math
    q=dict(previous)
    valid=q['geometry_valid']
    if valid != (q['final_status']=='PASS'):raise ValueError('persisted quality validity contradiction')
    if not valid:
        if qc is not None or q['reason']!='explicit_failure':raise ValueError('persisted failure contradiction')
        return q
    if q['reason']!='convergence_unverified' or q['selected_converged'] is not None or qc is None:raise ValueError('unexpected partial quality state')
    if not math.isfinite(float(qc['energy'])) or int(qc['atoms'])<=0:raise ValueError('invalid selected QC energy/atom metadata')
    c=[c for rnd in qc['diagnostics']['rounds'] for c in rnd['candidates'] if c.get('accepted')][0]
    if any(c.get(name) is False for name in ('finite_post_mmff','mmff_energy_finite','finite_energy')):raise ValueError('selected QC validity contradiction')
    converged=c['mmff_status']==0
    q.update(selected_converged=converged,mmff_status=c['mmff_status'],quality_eligible=converged,
             reason='selected_converged' if converged else 'selected_not_converged',
             quality_metadata_source='selected_coordinate_lane_QC',geometry_evidence='preserved_partial_ledger')
    return q


def main():
 import lmdb,numpy as np,sqlite3
 from src.dataset.cache_lifecycle import deserialize_record
 from src.training.glt_dual_runtime import require_tmux,load_sample_index_artifact
 require_tmux();OUT.mkdir(parents=True,exist_ok=False)
 totals={};down={};source_keys=np.zeros((959588,32),dtype=np.uint8);eligible=np.full(959588,-1,dtype=np.int8)
 try:
  reused=0
  for kind,dirname in [('pi1m','pi1m_direct_01.staging'),('downstream','downstream_direct_01.staging')]:
   root=BASE/dirname;meta=json.loads((root/'direct_source.json').read_text())
   if meta['schema']!='mcl-ph-direct-source-v1' or meta['conclusion']!='PASS' or not (root/'.frozen').exists():raise ValueError('source schema/freeze mismatch')
   env=lmdb.open(str(root/'data.lmdb'),readonly=True,lock=False,readahead=True,max_readers=4096)
   qcdb=sqlite3.connect('file:'+str(Path(meta['geometry_source'])/'rows.sqlite')+'?mode=ro',uri=True)
   counts=collections.Counter();n=0
   with env.begin() as txn,(OUT/(kind+'_quality.jsonl')).open('x') as ledger:
    if kind=='pi1m':
     previous={}
     with PARTIAL.open() as partial:
      for line in partial:
       q=json.loads(line);i=q['source_index']
       if i in previous:raise ValueError('duplicate partial identity')
       previous[i]=q
     with (root/'records.jsonl').open() as stream:
      for line in stream:
       row=json.loads(line)
       if row['pretrain_role']=='excluded':continue
       i=row['source_index'];key=bytes.fromhex(row['sample_key'])
       if i in previous:
        old=previous.pop(i)
        if old['sample_key']!=row['sample_key'] or old['normalized_smiles']!=row['normalized_smiles']:raise ValueError('partial identity mismatch')
        q=complete_persisted_quality(old,selected_qc(qcdb,key.hex(),old['final_status']));reused+=1
       else:
        payload=txn.get(key)
        if payload is None:raise ValueError('quality record missing')
        record=deserialize_record(payload,key)
        if record.normalized_smiles!=row['normalized_smiles'] or record.topology.normalized_canonical_smiles!=row['normalized_smiles']:raise ValueError('quality identity mismatch')
        q=classify_with_qc(record,selected_qc(qcdb,key.hex(),record.final_geometry_status));q.update(source_index=i,sample_key=key.hex(),normalized_smiles=row['normalized_smiles'])
       eligible[i]=int(q['quality_eligible']);source_keys[i]=np.frombuffer(key,dtype=np.uint8)
       ledger.write(json.dumps(q)+'\n');counts[q['reason']]+=1;n+=1
       if n%10000==0:print(json.dumps(dict(kind=kind,processed=n,counts=counts)),flush=True)
     if previous:raise ValueError('partial contains foreign source indices')
    else:
     for key,payload in txn.cursor():
      record=deserialize_record(payload,key)
      if record.topology.normalized_canonical_smiles!=record.normalized_smiles:raise ValueError('downstream identity mismatch')
      q=classify_with_qc(record,selected_qc(qcdb,key.hex(),record.final_geometry_status));q.update(sample_key=key.hex(),normalized_smiles=record.normalized_smiles)
      down[key.hex()]=q;ledger.write(json.dumps(q)+'\n');counts[q['reason']]+=1;n+=1
   env.close();qcdb.close()
   if n!=meta['identity_records']:raise ValueError('quality identity count mismatch')
   totals[kind]=dict(identities=n,counts=dict(counts),eligible=counts['selected_converged'],retention=counts['selected_converged']/n)
  train=load_sample_index_artifact(ORIGINAL,'train')['indices'];kept=train[eligible[train]==1]
  if not len(kept):raise ValueError('empty eligible training pool')
  if np.any(eligible[train]<0):raise ValueError('training row missing quality')
  np.save(OUT/'pi1m_eligible.npy',eligible);np.save(OUT/'pi1m_keys.npy',source_keys);np.save(OUT/'train_indices.npy',kept)
  (OUT/'downstream_quality.json').write_text(json.dumps(down))
  row_counts={}
  for task in ('eea','egb','egc','ei','eps','nc','xc','eat'):
   rows=[json.loads(x) for x in (BASE/'downstream_direct_01.staging'/f'{task}.jsonl').open()]
   flags=[]
   for row in rows:
    q=down[row['sample_key']]
    if q['normalized_smiles']!=row['normalized_smiles'] or q['final_status']!=row['geometry_status']:raise ValueError('downstream row ledger mismatch')
    flags.append(q['quality_eligible'])
   split=json.loads((ROOT/'data/splits/mips_paper5_outer'/f'{task}.json').read_text())
   row_counts[task]=dict(rows=len(rows),eligible=sum(flags),folds=[{role:dict(total=len(f[role+'_indices']),eligible=sum(flags[i] for i in f[role+'_indices'])) for role in ('train','validation','test')} for f in split['folds']])
  summary=dict(schema='glt-quality-v1',status='PASS',sources=totals,train_original=len(train),train_retained=len(kept),train_retention=len(kept)/len(train),train_presentations_per_arm=5040000,unique_training_pool=len(kept),downstream=row_counts,coordinate_generation_calls=0,persisted_geometry_records_reused=reused,partial_evidence=str(PARTIAL),reused_atom_count_crosscheck='not available in partial; original selected-QC validity and identity retained')
  (OUT/'index.json').write_text(json.dumps(dict(schema='glt-quality-index-v1',original_index=str(ORIGINAL),quality_root=str(OUT))))
  (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
 except BaseException as error:
  (OUT/'failure.json').write_text(json.dumps(dict(error=repr(error),traceback=traceback.format_exc(),retry=False)));raise
if __name__=='__main__':main()
