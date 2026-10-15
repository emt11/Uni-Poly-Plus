"""GP P0 read-only source, split, baseline and actual checkpoint coverage checks."""
import json,math
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/gp_sota_20261004'
torch.set_num_threads(1)
def read(p):return json.loads(Path(p).read_text())
def finite(x):
 if isinstance(x,torch.Tensor):assert torch.isfinite(x).all()
 elif isinstance(x,np.ndarray) and x.dtype.kind in 'fc':assert np.isfinite(x).all()
 elif isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,(list,tuple)):
  for v in x:finite(v)
 elif isinstance(x,float):assert math.isfinite(x)
common_path=ROOT/'results/mcl_ph_20260921/shared_pretrain/common_init_v1.pt'
common=torch.load(common_path,map_location='cpu',weights_only=False)['common_state_dict'];finite(common)
packages=[];missing=[]
for name in ('r3_glt_ref_fast_ra_01','r3_m_xattn_fast_01'):
 folder=ROOT/'results/mcl_ph_r4_20261002'/name
 state=torch.load(folder/'resume_05000.pt',map_location='cpu',weights_only=False)
 deploy=torch.load(folder/'deploy_05000.pt',map_location='cpu',weights_only=False)
 assert state['step']==deploy['step']==5000 and state['next_position']==5040000 and len(state['rng'])==4
 finite(state);finite(deploy)
 encoder={k[len('encoder.'):]:v for k,v in state['model'].items() if k.startswith('encoder.') and not k.startswith('encoder.predictor.')}
 assert encoder.keys()==deploy['state_dict'].keys()
 for k,v in encoder.items():assert torch.equal(v,deploy['state_dict'][k]),k
 if name.startswith('r3_glt'):
  assert common.keys()==state['direct_binding']['common_initial_state'].keys()
  for k,v in common.items():assert torch.equal(v,state['direct_binding']['common_initial_state'][k]) and v.shape==state['model'][k].shape,k
  missing=sorted(set(state['model'])-set(common))
  assert missing==['fp_head.0.bias','fp_head.0.weight','fp_head.3.bias','fp_head.3.weight']
  coverage=dict(common_tensor_count=len(common),required_common_pretrainer_tensor_count=len(state['model']),missing_common_initial_tensors=missing,
   historical_binding_has_fp_step0=False,historical_loader_policy='FP tensors initialized from seed42 outside common artifact; no saved FP step0 tensor evidence found')
 packages.append(dict(folder=str(folder),finite=True,complete_encoder_export_exact=True,step=5000,next_position=5040000))
 del state,deploy
pi=ROOT/'data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging';down=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
assert (pi/'.frozen').exists() and (down/'.frozen').exists()
idx=ROOT/'results/mcl_ph_next_20260929/p0/direct_indices_v1'
with np.load(idx/'pretrain_indices.npz',allow_pickle=False) as z:
 arrays={k:z[k].copy() for k in z.files}
train=arrays['train_source_indices'];val=arrays['validation_source_indices'];fixed=arrays['fixed_validation_source_indices']
assert len(train)==911391 and len(val)==47968 and len(fixed)==1024
assert len(set(train)&set(val))==0 and set(fixed)<=set(val)
roles=np.full(959588,'excluded',dtype='U10');roles[train]='train';roles[val]='validation'
identities={'train':set(),'validation':set(),'excluded':set()};count=0
with (pi/'records.jsonl').open() as f:
 for i,line in enumerate(f):
  row=json.loads(line);assert row['source_index']==i and row['pretrain_role']==roles[i]
  identities[roles[i]].add(row['normalized_smiles']);count+=1
assert count==959588
baseline=read(ROOT/'results/mcl_ph_r6_20261004/seven_task_results.json');rows=[];down_ids=set();total_budget=0
for original in baseline['rows']:
 task=original['task'];manifest=read(ROOT/f'data/splits/mips_paper5_outer/{task}.json')
 import pandas as pd
 raw=pd.read_csv(ROOT/f'data/raw/smi_{task}.csv');labels=raw.iloc[:,1].to_numpy(dtype=float)
 with (down/f'{task}.jsonl').open() as f:
  for i,line in enumerate(f):
   row=json.loads(line);assert row['original_row']==i and row['task']==task and row['label']==labels[i]
   down_ids.add(row['normalized_smiles'])
 metrics=[]
 for fold,p in enumerate(original['prediction_paths']):
  entry=manifest['folds'][fold];sets=[set(entry[r+'_indices']) for r in ('train','validation','test')]
  assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]) and set.union(*sets)==set(range(len(labels)))
  with np.load(p,allow_pickle=False) as z:
   y,zp=z['y_true'].reshape(-1),z['y_pred'].reshape(-1);assert np.isfinite(y).all() and np.isfinite(zp).all()
   assert z['test_indices'].tolist()==sorted(sets[2]);np.testing.assert_allclose(y,labels[sorted(sets[2])],rtol=2e-6,atol=2e-6)
   r2=float(1-np.sum((y-zp)**2)/np.sum((y-y.mean())**2));rmse=float(np.sqrt(np.mean((y-zp)**2)))
  np.testing.assert_allclose(r2,original['selected_fold_r2'][fold],rtol=0,atol=1e-12)
  glt=ROOT/f'results/mcl_ph_r4_20261002/fast_campaign_04/paper5_outer/glt_ref/{task}/fold{fold}' if task!='xc' else ROOT/f'results/mcl_ph_r5_20261004/paper5_outer/xc_standard/glt_ref/xc/fold{fold}'
  assert (glt/'head_initial.pt').exists() and (glt/'validation_predictions.npz').exists()
  m=read(glt/'metrics.json');assert m['initialization_seed']==42+fold and m['outer_test']=='RUN'
  metrics.append(dict(fold=fold,train_count=len(sets[0]),incumbent_test_r2=r2,incumbent_test_rmse=rmse,glt_reference=str(glt),glt_validation_rmse=m['best_validation_rmse']))
 nsteps=lambda f:(len(manifest['folds'][f]['train_indices'])+23)//24
 epochs=60 if task=='egc' else 70
 total_budget+=sum(epochs*nsteps(f) for f in (2,3,4))
 if task in ('ei','eps','nc','xc'):total_budget+=sum(epochs*nsteps(f)*4 for f in (0,1))
 else:total_budget+=sum(epochs*nsteps(f) for f in (0,1))
 rows.append(dict(task=task,folds=metrics,mean_r2=float(np.mean([m['incumbent_test_r2'] for m in metrics])),paper_model=original['paper_model'],paper_r2=original['paper_best_r2'],project_minus_paper=original['project_minus_paper']))
assert not down_ids & (identities['train']|identities['validation'])
assert total_budget==69930
cache=read(ROOT/'data/processed/mcl_ph_r4_20261001/trajectory_train_5040000/manifest.json')
assert cache['complete'] and cache['global_batch']==1008 and cache['max_updates']==5000 and cache['mask_ratio']==.3 and cache['noise_sigma']==.03
report=dict(status='STOPPED_AT_P0',checked_at=datetime.now(timezone.utc).isoformat(),stop_reason='Missing four common FP-head step0 tensors under plan section 8; G_REF fully matched initialization contract not established',coverage=coverage,
 packages=packages,source=dict(pi1m_rows=count,train_rows=len(train),validation_rows=len(val),fixed_validation_rows=len(fixed),downstream_overlap=0,train_validation_shared_structures=len(identities['train']&identities['validation']),seal=True,ph_cache_metadata=True),
 baseline_rows=rows,full_finetune_updates_upper_bound=total_budget,paper_protocol_equivalence='未核实',label_physical_units='未核实',
 new_model_calls=0,new_updates=0,pretraining_updates=0,finetuning_updates=0,formal_test_only_units=0,stages_not_executed=['P1','P2a','P3a','P2b/P3b','P3c','P4','P5','P6'])
(OUT/'p0_contract_01.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(status=report['status'],missing=missing,pretraining_updates=0,finetuning_updates=0),ensure_ascii=False),flush=True)
