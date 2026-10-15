#!/usr/bin/env python3
"""S5: matched two-rank F_CE without torsion, using unchanged PT/FT computations."""
import argparse,json,math,os,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from scripts import run_glt_align_2d_unit as u1
from src.modules.glt_dual import build_dual_glt_model,PHYSICAL_TORSION
from src.modules.glt_dual_pretrain import load_deployment,DualPretrainer
OUT=ROOT/'results/glt_optimization_20261011/r9/S5'
LOG=ROOT/'logs/glt_optimization_20261011/r9/S5'
BASE=ROOT/'results/glt_periodic_seam_20261009/r1'
COMMON=BASE/'initialization/F_CE_step0.pt'
REFERENCE=ROOT/'results/glt_optimization_20261011/r6/S4'
ARMS=('F_CE_2R',);TASKS=('eea','egb','egc','ei','eps','nc','xc')
PARAMETERS=39876344
REVISION='glt_matched_reference_s5'
RESOURCE=dict(resource_revision=1,expected_world_size=2,microbatch=84,accumulation=6,global_batch=1008,reference_protocol='matched_two_rank_T_FCE')
PROFILE=u1.PROFILE;DOWN=u1.DOWN;SPLIT=u1.SPLIT
read=u1.read;write=u1.write;strict_restore=u1.strict_restore;now=u1.now
MATCHED_FIELDS=('architecture','fusion_mode','fusion_revision','geometry_revision','geometry_mode','angle_value_enabled','readout',
 'use_md200','md200_root','atom_ce_input','local_enabled','third_task','fusion_gate','fusion_attention_mode','incident_bias',
 'endpoint_revision','quality_root','common_initialization','ft_profile','split','ft_seed_formula','target_transform',
 'head_initialization','prediction_path','input_dim','opt_mode','geometry_head_norm','seed','microbatch','global_batch',
 'expected_world_size','accumulation','max_optimizer_steps','save_every','lr','warmup_steps','schedule_total_steps',
 'end_lr','weight_decay','amp_dtype','atom_mask_ratio','noise_sigma','loss_weights','sample_index_artifact',
 'sample_index_split','geometry_target','element_mask','input_schema','joint_3d_drop_probability','ft_batch_size','ft_amp_dtype','ft_workers','ft_budget')

def parent(arm):
 if arm not in ARMS:raise ValueError('invalid S5 arm')
 return OUT/'pretrain'/arm/'deploy_05000.pt'
def head_path(arm,task,fold):
 if arm not in ARMS:raise ValueError('invalid S5 arm')
 return ROOT/'results/glt_fusion_20261007/r1/finetune/B0'/task/f'fold{fold}'/'head_initial.pt'
def ft_folder(arm,task,fold):return OUT/'finetune'/arm/task/f'fold{fold}'
def epochs(task):return 70
def make_config():
 reference=read(REFERENCE/'initialization/T_FCE.json')
 cfg={k:v for k,v in reference.items() if k not in PHYSICAL_TORSION}
 cfg.update(plan='GLT-OPTIMIZATION-20261011-01',revision='r9',cycle='S5',arm='F_CE_2R',cycle_root=str(OUT),
  initial_state=str(OUT/'initialization/F_CE_2R_step0.pt'),parent=str(parent('F_CE_2R')),
  reference_revision=REVISION,comparison_root=str(REFERENCE),reference_arm='T_FCE',reference_root=str(REFERENCE),
  wrapper_architecture='dual_glt_matched_reference_s5',torsion_modules=False,torsion_mode='off',torsion_revision=None,
  new_parameters=0,total_parameters=PARAMETERS,**RESOURCE)
 for key in MATCHED_FIELDS:
  if key not in cfg or cfg[key]!=reference.get(key):raise ValueError('matched S4 setting differs: '+key)
 if reference.get('torsion_revision')!='glt_physical_torsion_r1' or not reference.get('torsion_modules'):
  raise ValueError('T_FCE reference is not the completed physical torsion arm')
 if reference['ft_profile']!=PROFILE:raise ValueError('reference uniform FT recipe differs')
 return cfg
def validate_config(cfg,arm='F_CE_2R'):
 if arm not in ARMS or cfg!=make_config():raise ValueError('invalid S5 matched configuration')
 return cfg
def reference_metadata(cfg):
 validate_config(cfg)
 return {k:cfg[k] for k in ('reference_revision','plan','revision','cycle','arm','resource_revision','expected_world_size',
  'microbatch','accumulation','global_batch','reference_protocol','common_initialization','comparison_root')}
def bind_deployment(package,cfg):
 metadata=reference_metadata(cfg)
 if package.get('torsion_modules') or package.get('torsion_revision') or package.get('endpoint_revision') or package.get('align_enabled'):
  raise ValueError('S5 deployment contains a forbidden branch')
 return dict(package,**metadata,training_config=cfg)
class ReferenceArm(nn.Module):
 def __init__(self,package,arm,fold,cfg):
  super().__init__();validate_config(cfg,arm)
  if fold not in range(5):raise ValueError('invalid fold')
  if package.get('training_config')!=cfg or any(package.get(k)!=v for k,v in reference_metadata(cfg).items()):
   raise ValueError('S5 parent metadata/source mismatch')
  for k in ('architecture','fusion_mode','fusion_revision','geometry_revision','geometry_mode','angle_value_enabled','readout',
            'use_md200','atom_ce_input','third_task','align_enabled','local_enabled','torsion_modules'):
   if package.get(k)!=cfg[k]:raise ValueError('S5 parent architecture mismatch '+k)
  if package.get('torsion_revision') or package.get('endpoint_revision') or package.get('fusion_gate','none')!='none':
   raise ValueError('S5 parent contains forbidden branch')
  if any(not torch.isfinite(v).all() for v in package['state_dict'].values()):raise ValueError('S5 parent nonfinite tensor')
  self.model=build_dual_glt_model('token',fusion_revision='glt_fusion_r1',torsion=False,
    **{k:cfg[k] for k in ('geometry_revision','geometry_mode','angle_value_enabled','readout')})
  load_deployment(self.model,package,5000)
  if sum(p.numel() for p in self.parameters())!=PARAMETERS:raise ValueError('S5 parameter count differs from F_CE')
 def features(self,data):return self.model.fuse(self.model.encode(data))
 def forward(self,data):return self.model.predictor(self.features(data))
def validate_best(best,arm,task,fold,cfg):
 validate_config(best['config'],arm)
 if (best['config']!=cfg or best['task']!=task or best['fold']!=fold or best['seed']!=42+fold
     or best['parent']!=str(parent(arm)) or best.get('resource')!=RESOURCE
     or best.get('matched_reference')!=reference_metadata(cfg) or best.get('prediction_path')!='fused'
     or best.get('input_dim')!=1024 or best.get('head_initial')!=str(head_path(arm,task,fold))
     or best.get('target_transform')!='standard' or best.get('scaler_fit')!='train'):
  raise ValueError('S5 best metadata mismatch')

def run_ft(arm,task,fold,formal=False):
 import numpy as np,torch
 from torch import nn
 from torch.utils.data import DataLoader,Subset
 from sklearn.preprocessing import StandardScaler
 from src.utils import set_global_seed,scale_targets,evaluate
 import scripts.finetune_mcl_ph as old
 from src.dataset.glt_dual import dual_glt_collate
 from src.training.glt_dual_runtime import open_source,save_checkpoint
 from src.modules.glt_dual_pretrain import load_deployment
 from src.modules.glt_adaptation import AdaptationAdapter
 folder=OUT/'formal'/arm/task/f'fold{fold}' if formal else ft_folder(arm,task,fold)
 if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2','3'):raise ValueError('S5 FT/formal requires one physical GPU2 or GPU3')
 folder.mkdir(parents=True,exist_ok=False)
 updates={'attempted':0,'completed':0};history=[];source=None
 try:
  cfg=read(OUT/'initialization'/f'{arm}.json');set_global_seed(42+fold)
  package=torch.load(parent(arm),map_location='cpu',weights_only=False)
  model=ReferenceArm(package,arm,fold,cfg)
  head=torch.load(head_path(arm,task,fold),map_location='cpu',weights_only=False)
  strict_restore(old.periodic_tdl_head(model),head)
  model.to('cuda:0');old.configure_periodic_tdl_dropout(model,task,profile='uniform_u1')
  manifest=read(SPLIT/f'{task}.json');entry=manifest['folds'][fold]
  best_folder=ft_folder(arm,task,fold)
  if formal:
   selection=read(OUT/'evaluation_before_test.json')
   if selection['validation_units']!=35:raise ValueError('selection not complete')
   best=torch.load(best_folder/'best.pt',map_location='cpu',weights_only=False)
   if best['config']!=cfg or best['task']!=task or best['fold']!=fold or best.get('seed')!=42+fold or best.get('parent')!=str(parent(arm)):raise ValueError('formal model metadata mismatch')
   units=[u for u in selection['units'] if (u['arm'],u['task'],u['fold'])==(arm,task,fold)]
   if len(units)!=1 or units[0]['best']!=str(best_folder/'best.pt') or selection['configs'][arm]!=cfg or read(best_folder/'scaler.json')!=units[0]['scaler'] or best['scaler_mean']!=units[0]['scaler']['mean'] or best['scaler_scale']!=units[0]['scaler']['scale']:
    raise ValueError('formal frozen best/config/scaler mismatch')
   if best.get('prediction_path')!=cfg['prediction_path'] or best.get('input_dim')!=cfg['input_dim'] or best.get('head_initial')!=str(head_path(arm,task,fold)):raise ValueError('formal path/head mismatch')
   validate_best(best,arm,task,fold,cfg)
   strict_restore(model,best['state_dict'])
   selected=sorted(entry['test_indices']);local_val=list(range(len(selected)))
   if selected!=units[0]['test_indices'] or units[0]['parent']!=str(parent(arm)) or best['best_epoch']!=units[0]['best_epoch'] or any(units[0][k]!=cfg[k] for k in RESOURCE) or units[0]['head_initial']!=str(head_path(arm,task,fold)):raise ValueError('formal frozen indices/parent/epoch mismatch')
  else:selected,local_train,local_val=old.compact_train_validation_indices(entry['train_indices'],entry['validation_indices'])
  md_root=read(Path(cfg['md200_root'])/'metadata.json').get('downstream_sidecar') if cfg['use_md200'] else None
  if cfg['use_md200'] and not md_root:raise ValueError('MD sidecar missing downstream binding')
  source,frame=open_source(DOWN,DOWN,task=task,dual_static_root=DOWN,selected_indices=selected,split_manifest=manifest,record_index_path=DOWN/f'{task}_offsets.npy',expected_task_rows=manifest['sample_count'],verify_key_digest=False,md200_root=md_root)
  if frame['original_row'].astype(int).tolist()!=selected:raise ValueError('selected source rows differ')
  class NewCleanDataset(old.CleanDualDataset):
   def __getitem__(self,index):
    if index not in self._clean_cache:
     from src.training.glt_dual_runtime import pretrain_source_fields
     from src.dataset.glt_dual import build_dual_sample
     top,tri,text,static,_=pretrain_source_fields(self.source,index)
     self._clean_cache[index]=build_dual_sample(top,tri,text,static=static,geometry_mode=cfg['geometry_mode'])
    data=old.clone_clean_record(self._clean_cache[index])
    data.y=torch.tensor([float(self.targets[index])],dtype=torch.float32)
    if not hasattr(data,'md200'):
     if cfg['use_md200']:raise ValueError('MD-on FT sample missing sidecar')
     data.md200=torch.zeros((1,200));data.md200_valid=torch.zeros(1,dtype=torch.bool)
    return data
  ds=NewCleanDataset(source,frame['label'].to_numpy(dtype=np.float64))
  set_global_seed(42+fold)
  if formal:
   from src.utils import TargetScaler
   # Reuse the project's scalar interface without fitting on test labels.
   scaler=TargetScaler(task,StandardScaler(),transform_mode='standard');scaler.scaler.mean_=np.asarray(best['scaler_mean']);scaler.scaler.scale_=np.asarray(best['scaler_scale']);scaler.scaler.var_=scaler.scaler.scale_**2;scaler.scaler.n_features_in_=1
   ds.set_target_override(scaler.transform(frame['label'].to_numpy(dtype=np.float64)).reshape(-1))
  else:
   scaler=scale_targets(ds,task,train_indices=local_train,transform_mode='standard')
   references=[u for u in read(BASE/'selection_before_test.json')['units'] if (u['arm'],u['task'],u['fold'])==('F_CE',task,fold)]
   if len(references)!=1 or references[0]['scaler']['mean']!=scaler.scaler.mean_.tolist() or references[0]['scaler']['scale']!=scaler.scaler.scale_.tolist():raise ValueError('F_CE train-only scaler mismatch')
  collate=dual_glt_collate
  loader=DataLoader(Subset(ds,local_val),batch_size=24,shuffle=False,num_workers=0,collate_fn=collate)
  if formal:
   _,r2,truth,prediction=evaluate(AdaptationAdapter(model),loader,nn.MSELoss(),'cuda:0',scaler=scaler,amp_dtype='fp32')
   epoch=best['best_epoch'];total_epochs=0
  else:
   save_checkpoint(folder/'head_initial.pt',head)
   train=DataLoader(Subset(ds,local_train),batch_size=24,shuffle=True,num_workers=0,collate_fn=collate,generator=torch.Generator().manual_seed(42+fold))
   def observe(kind):
    updates[kind]+=1;write(folder/'updates.json',updates)
   def progress(rec):
    rec=dict(rec,prediction_path=cfg['prediction_path'])
    history.append(rec);print(json.dumps(rec),flush=True)
    write(folder/'progress.json',dict(history=history,updates=updates,outer_test='NOT_RUN'))
   result=old.run_periodic_tdl_epochs(model,train,loader,nn.MSELoss(),'cuda:0',task=task,scaler=scaler,progress=progress,profile='uniform_u1',step_observer=observe)
   if updates['attempted']!=updates['completed'] or updates['completed']!=epochs(task)*len(train):raise ValueError('FT updates mismatch')
   write(folder/'scaler.json',dict(mean=scaler.scaler.mean_.tolist(),scale=scaler.scaler.scale_.tolist(),fit='train',transform='standard'))
   truth,prediction=result['predictions'];r2=result['best_r2'];epoch=result['best_epoch'];total_epochs=len(history)
   save_checkpoint(folder/'best.pt',dict(state_dict=result['state'],config=cfg,task=task,fold=fold,seed=42+fold,parent=str(parent(arm)),best_epoch=epoch,scaler_mean=scaler.scaler.mean_.tolist(),scaler_scale=scaler.scaler.scale_.tolist(),history=history,optimizer_groups=result['optimizer_groups_by_stage'],optimizer_updates=updates['completed'],resource=dict(RESOURCE),matched_reference=reference_metadata(cfg),head_initial=str(head_path(arm,task,fold)),prediction_path=cfg['prediction_path'],input_dim=cfg['input_dim'],target_transform='standard',scaler_fit='train',outer_test='NOT_RUN'))
  truth=np.asarray(truth).reshape(-1);prediction=np.asarray(prediction).reshape(-1)
  if not np.isfinite(truth).all() or not np.isfinite(prediction).all() or not math.isfinite(float(r2)):raise FloatingPointError('nonfinite predictions')
  ids=selected if formal else entry['validation_indices']
  np.savez(folder/('test_predictions.npz' if formal else 'validation_predictions.npz'),y_true=truth,y_pred=prediction,sample_keys=np.asarray([source.samples[i][0].hex() for i in local_val]),indices=np.asarray(ids),scaler_mean=scaler.scaler.mean_,scaler_scale=scaler.scaler.scale_,best_epoch=epoch,geometry_valid=np.asarray([source.entries[i]['geometry_status']=='PASS' for i in local_val]),torsion_available=np.zeros(len(local_val),dtype=bool),outer_test='RUN' if formal else 'NOT_RUN')
  write(folder/'metrics.json',dict(status='PASS',arm=arm,task=task,fold=fold,modes=cfg,parameters=sum(p.numel() for p in model.parameters()),epochs=total_epochs,updates=updates,r2=float(r2),rmse=float(np.sqrt(np.mean((truth-prediction)**2))),outer_test='RUN' if formal else 'NOT_RUN'))
 except BaseException as error:
  write(folder/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),updates=updates,completed_epochs=len(history),time=now()))
  raise
 finally:
  if source is not None:source.close()

def initialize():
 from src.training.glt_dual_runtime import save_checkpoint
 folder=OUT/'initialization';folder.mkdir(parents=True,exist_ok=False)
 cfg=validate_config(make_config())
 common=torch.load(COMMON,map_location='cpu',weights_only=False)
 if common['step']!=0:raise ValueError('S5 requires original public step0')
 torch.manual_seed(42)
 model=DualPretrainer('token',torsion=False,fusion_revision='glt_fusion_r1',
  atom_ce_input='fused',geometry_head_norm=True,opt_mode='GLT_FUSION_R1',
  **{k:cfg[k] for k in ('geometry_revision','geometry_mode','angle_value_enabled','readout')})
 state=model.state_dict()
 if set(state)!=set(common['model']):raise ValueError('S5 must not introduce any tensor')
 for k,v in common['model'].items():
  if v.shape!=state[k].shape or v.dtype!=state[k].dtype or not torch.isfinite(v).all():raise ValueError('S5 public tensor mismatch '+k)
  state[k]=v.clone()
 strict_restore(model,state)
 save_checkpoint(folder/'F_CE_2R_step0.pt',dict(model=state,step=0,opt_mode=cfg['opt_mode'],arm='F_CE_2R',config=cfg))
 write(folder/'F_CE_2R.json',cfg)
 write(folder/'summary.json',dict(pt_parameters=sum(p.numel() for p in model.parameters()),new_parameters=0,
  ft_parameters=PARAMETERS,common=str(COMMON),reference=str(REFERENCE),public_state='exact original encoder and PT heads; no new tensors'))

if __name__=='__main__':
 from src.training.glt_dual_runtime import require_tmux
 require_tmux()
 parser=argparse.ArgumentParser();parser.add_argument('--initialize',action='store_true');parser.add_argument('--arm',choices=ARMS)
 parser.add_argument('--task',choices=TASKS);parser.add_argument('--fold',type=int,choices=range(5));parser.add_argument('--formal',action='store_true')
 args=parser.parse_args()
 if args.initialize:initialize()
 elif args.arm and args.task and args.fold is not None:run_ft(args.arm,args.task,args.fold,args.formal)
 else:raise ValueError('invalid unit arguments')
