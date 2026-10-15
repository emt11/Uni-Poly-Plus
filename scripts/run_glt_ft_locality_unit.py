#!/usr/bin/env python3
"""Fixed FT-only locality units; frozen F_CE restored before adaptation."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/glt_optimization_20261011/r2/S2'
BASE=ROOT/'results/glt_periodic_seam_20261009/r1'
PARENT=BASE/'pretrain/F_CE/deploy_05000.pt'
ARMS=('L_FT','B_FT');TASKS=('eea','egb','egc','ei','eps','nc','xc')

import torch
from torch import nn
from src.modules.glt_dual_opt import BondTokenFusion
from src.modules.glt_dual import build_dual_glt_model
from src.modules.glt_dual_pretrain import load_deployment

def strict_restore(module,state):
 current=module.state_dict()
 if set(current)!=set(state):raise ValueError('FT tensor names mismatch')
 for name,value in state.items():
  if value.shape!=current[name].shape or value.dtype!=current[name].dtype or not torch.isfinite(value).all():raise ValueError('FT tensor shape/dtype/finite mismatch '+name)
 module.load_state_dict(state,strict=True)

def connection_config(arm):
 import math
 if arm=='L_FT':return dict(fusion_attention_mode='incident_only',incident_bias=0.0)
 if arm=='B_FT':return dict(fusion_attention_mode='incident_bias',incident_bias=math.log(2.0))
 raise ValueError('invalid locality arm')

def validate_config(cfg,arm):
 required=dict(plan='GLT-OPTIMIZATION-20261011-01',revision='r2',cycle='S2',arm=arm,parent=str(PARENT),
  fusion_mode='token',fusion_revision='glt_fusion_r1',geometry_revision='glt_periodic_seam_r1',geometry_mode='finite_incident',
  angle_value_enabled=True,readout='incident',use_md200=False,md200_root=None,atom_ce_input='fused',align_enabled=False,
  local_enabled=False,third_task='none',torsion_modules=False,fusion_gate='none',joint_3d_drop_probability=0.0,
  ft_batch_size=24,ft_amp_dtype='fp32',ft_workers=0,ft_seed_formula='42+fold',split='paper5_outer',target_transform='train_only_standard',
  **connection_config(arm))
 if any(k not in cfg or type(cfg[k]) is not type(v) or cfg[k]!=v for k,v in required.items()):
  raise ValueError('S2 missing/invalid configuration')
 return cfg

class LocalityArm(nn.Module):
 def __init__(self,package,arm,fold,cfg):
  super().__init__()
  if arm not in ARMS or fold not in range(5):raise ValueError('invalid FT arm/fold')
  validate_config(cfg,arm)
  required=dict(fusion_mode='token',fusion_revision='glt_fusion_r1',use_md200=False,atom_ce_input='fused',align_enabled=False,local_enabled=False,third_task='none',geometry_revision='glt_periodic_seam_r1',geometry_mode='finite_incident',angle_value_enabled=True,readout='incident')
  if any(package.get(k)!=v for k,v in required.items()) or package.get('fusion_gate','none')!='none' or package.get('torsion_modules') is not False:raise ValueError('F_CE parent mode mismatch')
  self.model=build_dual_glt_model('token',fusion_revision='glt_fusion_r1',use_md200=False,torsion=False,**{k:required[k] for k in ('geometry_revision','geometry_mode','angle_value_enabled','readout')})
  load_deployment(self.model,package,5000)
  if any(not torch.isfinite(value).all() for value in package['state_dict'].values()):raise ValueError('nonfinite F_CE parent')
  self.model.token_fusion.set_attention_mode(cfg['fusion_attention_mode'],cfg['incident_bias'])
  for name,value in package['state_dict'].items():
   if not torch.equal(self.model.state_dict()[name],value):raise ValueError('parent parameter changed '+name)
 def features(self,data):return self.model.fuse(self.model.encode(data))
 def forward(self,data):return self.model.predictor(self.features(data))
import datetime,math,os,traceback,types
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT=ROOT/'data/splits/mips_paper5_outer'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def epochs(task):return 60 if task=='egc' else 70
def parent(arm):return PARENT
def ft_folder(arm,task,fold):return OUT/'finetune'/arm/task/f'fold{fold}'
def run_ft(arm,task,fold,formal=False):
 ft_candidate=None
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
 folder.mkdir(parents=True,exist_ok=False)
 updates={'attempted':0,'completed':0};history=[];source=None
 try:
  cfg=read(OUT/'initialization'/f'{arm}.json');set_global_seed(42+fold)
  package=torch.load(parent(arm),map_location='cpu',weights_only=False)
  model=LocalityArm(package,arm,fold,cfg)
  if cfg['parent']!=str(PARENT) or cfg['fusion_gate']!=model.model.fusion_gate or cfg['joint_3d_drop_probability']!=0.0:raise ValueError('FT config/model mismatch')
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(42+fold)
   common_head=nn.Sequential(nn.Linear(1024,512),nn.GELU(),nn.Dropout(.1),nn.Linear(512,1))
  head=torch.load(ROOT/'results/glt_fusion_20261007/r1/finetune/B0'/task/f'fold{fold}'/'head_initial.pt',map_location='cpu',weights_only=False)
  if set(head)!=set(common_head.state_dict()):raise ValueError('B0 head tensor names mismatch')
  for name,value in head.items():
   target=common_head.state_dict()[name]
   if value.shape!=target.shape or value.dtype!=target.dtype or not torch.isfinite(value).all():raise ValueError('B0 head tensor shape/dtype/value mismatch '+name)
  old.periodic_tdl_head(model).load_state_dict(head,strict=True)
  model.to('cuda:0');old.configure_periodic_tdl_dropout(model,task)
  manifest=read(SPLIT/f'{task}.json');entry=manifest['folds'][fold]
  best_folder=ft_folder(arm,task,fold)
  if formal:
   selection=read(OUT/'evaluation_before_test.json')
   if selection['validation_units']!=70:raise ValueError('selection not complete')
   best=torch.load(best_folder/'best.pt',map_location='cpu',weights_only=False)
   if best['config']!=cfg or best['task']!=task or best['fold']!=fold or best.get('seed')!=42+fold or best.get('parent')!=str(parent(arm)):raise ValueError('formal model metadata mismatch')
   units=[u for u in selection['units'] if (u['arm'],u['task'],u['fold'])==(arm,task,fold)]
   if len(units)!=1 or units[0]['best']!=str(best_folder/'best.pt') or selection['configs'][arm]!=cfg or read(best_folder/'scaler.json')!=units[0]['scaler'] or best['scaler_mean']!=units[0]['scaler']['mean'] or best['scaler_scale']!=units[0]['scaler']['scale']:
    raise ValueError('formal frozen best/config/scaler mismatch')
   strict_restore(model,best['state_dict'])
   selected=sorted(entry['test_indices']);local_val=list(range(len(selected)))
   if selected!=units[0]['test_indices'] or units[0]['parent']!=str(PARENT) or best['best_epoch']!=units[0]['best_epoch'] or units[0]['fusion_attention_mode']!=cfg['fusion_attention_mode'] or units[0]['incident_bias']!=cfg['incident_bias']:raise ValueError('formal frozen indices/parent/epoch mismatch')
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
  loader=DataLoader(Subset(ds,local_val),batch_size=24,shuffle=False,num_workers=0,collate_fn=dual_glt_collate)
  if formal:
   _,r2,truth,prediction=evaluate(AdaptationAdapter(model),loader,nn.MSELoss(),'cuda:0',scaler=scaler,amp_dtype='fp32')
   epoch=best['best_epoch'];total_epochs=0
  else:
   save_checkpoint(folder/'head_initial.pt',head)
   train=DataLoader(Subset(ds,local_train),batch_size=24,shuffle=True,num_workers=0,collate_fn=dual_glt_collate,generator=torch.Generator().manual_seed(42+fold))
   old_optimizer=old.periodic_tdl_optimizer
   def counted(*args,**kwargs):
    opt,evidence=old_optimizer(*args,**kwargs);real=opt.step
    def step(self,*a,**kw):
     updates['attempted']+=1
     write(folder/'updates.json',updates)
     result=real(*a,**kw);updates['completed']+=1
     write(folder/'updates.json',updates);return result
    opt.step=types.MethodType(step,opt);return opt,evidence
   old.periodic_tdl_optimizer=counted
   def progress(rec):
    rec=dict(rec,fusion_attention_mode=cfg['fusion_attention_mode'],incident_bias=cfg['incident_bias'])
    history.append(rec);print(json.dumps(rec),flush=True)
    write(folder/'progress.json',dict(history=history,updates=updates,outer_test='NOT_RUN'))
   result=old.run_periodic_tdl_epochs(model,train,loader,nn.MSELoss(),'cuda:0',task=task,scaler=scaler,progress=progress,ft_candidate=ft_candidate)
   if updates['attempted']!=updates['completed'] or updates['completed']!=epochs(task)*len(train):raise ValueError('FT updates mismatch')
   write(folder/'scaler.json',dict(mean=scaler.scaler.mean_.tolist(),scale=scaler.scaler.scale_.tolist(),fit='train',transform='standard'))
   truth,prediction=result['predictions'];r2=result['best_r2'];epoch=result['best_epoch'];total_epochs=len(history)
   save_checkpoint(folder/'best.pt',dict(state_dict=result['state'],config=cfg,task=task,fold=fold,seed=42+fold,parent=str(parent(arm)),best_epoch=epoch,scaler_mean=scaler.scaler.mean_.tolist(),scaler_scale=scaler.scaler.scale_.tolist(),history=history,optimizer_groups=result['optimizer_groups_by_stage'],optimizer_updates=updates['completed'],target_transform='standard',scaler_fit='train',outer_test='NOT_RUN'))
  truth=np.asarray(truth).reshape(-1);prediction=np.asarray(prediction).reshape(-1)
  if not np.isfinite(truth).all() or not np.isfinite(prediction).all() or not math.isfinite(float(r2)):raise FloatingPointError('nonfinite predictions')
  ids=selected if formal else entry['validation_indices']
  np.savez(folder/('test_predictions.npz' if formal else 'validation_predictions.npz'),y_true=truth,y_pred=prediction,sample_keys=np.asarray([source.samples[i][0].hex() for i in local_val]),indices=np.asarray(ids),scaler_mean=scaler.scaler.mean_,scaler_scale=scaler.scaler.scale_,best_epoch=epoch,geometry_valid=np.asarray([source.entries[i]['geometry_status']=='PASS' for i in local_val]),outer_test='RUN' if formal else 'NOT_RUN')
  write(folder/'metrics.json',dict(status='PASS',arm=arm,task=task,fold=fold,modes=cfg,epochs=total_epochs,updates=updates,r2=float(r2),rmse=float(np.sqrt(np.mean((truth-prediction)**2))),outer_test='RUN' if formal else 'NOT_RUN'))
 except BaseException as error:
  write(folder/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),updates=updates,completed_epochs=len(history),time=now()))
  raise
 finally:
  if source is not None:source.close()

if __name__=='__main__':
 from src.training.glt_dual_runtime import require_tmux
 require_tmux()
 p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=ARMS)
 p.add_argument('--task',required=True,choices=TASKS);p.add_argument('--fold',required=True,type=int,choices=range(5));p.add_argument('--formal',action='store_true')
 args=p.parse_args();run_ft(args.arm,args.task,args.fold,args.formal)
