#!/usr/bin/env python3
"""F1 sole candidate: graph product residual on the original frozen PT parent."""
import argparse,datetime,json,math,os,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from scripts import run_glt_align_2d_unit as u1
OUT=ROOT/'results/glt_optimization_20261011/r5/F1'
BASE=ROOT/'results/glt_periodic_seam_20261009/r1'
U1=ROOT/'results/glt_optimization_20261011/r4/U1'
ARMS=('F1_MUL',);TASKS=('eea','egb','egc','ei','eps','nc','xc')
PROFILE=u1.PROFILE
read=u1.read;write=u1.write;strict_restore=u1.strict_restore;now=u1.now
DOWN=u1.DOWN;SPLIT=u1.SPLIT
PARAMETERS=39975288
NEW_PARAMETERS=98944

def parent(arm):
 if arm not in ARMS:raise ValueError('invalid F1 arm')
 return BASE/'pretrain/F_CE/deploy_05000.pt'
def head_path(arm,task,fold):return ROOT/'results/glt_fusion_20261007/r1/finetune/B0'/task/f'fold{fold}'/'head_initial.pt'
def epochs(task):return 70
def ft_folder(arm,task,fold):return OUT/'finetune'/arm/task/f'fold{fold}'
def interaction_config():
 return dict(graph_interaction_revision='glt_graph_mul_f1',interaction_rank=64,interaction_bias=True,
  interaction_init_seed=314163,interaction_output_zero_initialized=True,interaction_mask='after_output_projection_final_geometry_valid',
  prediction_path='fused_graph_mul',input_dim=1024)
def make_config():
 reference=u1.validate_config(read(U1/'initialization/U0_FUSED.json'),'U0_FUSED')
 return dict(reference,plan='GLT-OPTIMIZATION-20261011-01',revision='r5',cycle='F1',arm='F1_MUL',
  parent=str(parent('F1_MUL')),reference_arm='U0_FUSED',reference_root=str(U1),wrapper_architecture='dual_glt_graph_mul_f1',
  new_parameters=NEW_PARAMETERS,total_parameters=PARAMETERS,ft_budget=dict(starts=35,epochs=2450,updates=56000,formal_predictions=35),**interaction_config())
def validate_config(cfg,arm):
 if arm not in ARMS:raise ValueError('invalid F1 arm')
 expected=make_config()
 if any(k not in cfg or type(cfg[k]) is not type(v) or cfg[k]!=v for k,v in expected.items()):raise ValueError('invalid F1 configuration')
 return cfg
class GraphMultiplicative(nn.Module):
 def __init__(self):
  super().__init__()
  with torch.random.fork_rng(devices=[]):
   torch.default_generator.manual_seed(314163)
   self.left=nn.Linear(512,64);self.right=nn.Linear(512,64);self.output=nn.Linear(64,512)
   nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def forward(self,features,valid):
  if features.ndim!=2 or features.size(1)!=1024 or valid.shape!=(features.size(0),) or valid.dtype!=torch.bool:raise ValueError('F1 feature/validity schema mismatch')
  left,right=features[:,:512],features[:,512:]
  update=self.output(self.left(left)*self.right(right))
  update=torch.where(valid.unsqueeze(-1),update,torch.zeros_like(update))
  return torch.cat((left+update,right),dim=-1)
class GraphInteractionArm(nn.Module):
 def __init__(self,package,arm,fold,cfg):
  super().__init__();validate_config(cfg,arm)
  reference=u1.build_arm(package,'U0_FUSED',fold,read(U1/'initialization/U0_FUSED.json'))
  self.model=reference.model
  self.interaction=GraphMultiplicative()
  if sum(p.numel() for p in self.interaction.parameters())!=NEW_PARAMETERS or sum(p.numel() for p in self.parameters())!=PARAMETERS:raise ValueError('F1 parameter budget mismatch')
 def features(self,data):
  encoded=self.model.encode(data)
  # fuse owns its short-lived dictionary cache; do not modify source or encoded.
  features=self.model.fuse(dict(encoded))
  return self.interaction(features,encoded['geometry_valid'])
 def forward(self,data):return self.model.predictor(self.features(data))
def validate_best(best,arm,task,fold,cfg):
 validate_config(best['config'],arm)
 if best['config']!=cfg or best['task']!=task or best['fold']!=fold or best['seed']!=42+fold or best['parent']!=str(parent(arm)) or best.get('graph_interaction')!=interaction_config() or best.get('prediction_path')!=cfg['prediction_path'] or best.get('input_dim')!=1024 or best.get('head_initial')!=str(head_path(arm,task,fold)):raise ValueError('F1 best metadata mismatch')
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
 folder.mkdir(parents=True,exist_ok=False)
 updates={'attempted':0,'completed':0};history=[];source=None
 try:
  cfg=read(OUT/'initialization'/f'{arm}.json');set_global_seed(42+fold)
  package=torch.load(parent(arm),map_location='cpu',weights_only=False)
  model=GraphInteractionArm(package,arm,fold,cfg)
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
   if selected!=units[0]['test_indices'] or units[0]['parent']!=str(parent(arm)) or best['best_epoch']!=units[0]['best_epoch'] or any(units[0][k]!=cfg[k] for k in interaction_config()) or units[0]['head_initial']!=str(head_path(arm,task,fold)):raise ValueError('formal frozen indices/parent/epoch mismatch')
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
   save_checkpoint(folder/'best.pt',dict(state_dict=result['state'],config=cfg,task=task,fold=fold,seed=42+fold,parent=str(parent(arm)),best_epoch=epoch,scaler_mean=scaler.scaler.mean_.tolist(),scaler_scale=scaler.scaler.scale_.tolist(),history=history,optimizer_groups=result['optimizer_groups_by_stage'],optimizer_updates=updates['completed'],graph_interaction=interaction_config(),head_initial=str(head_path(arm,task,fold)),prediction_path=cfg['prediction_path'],input_dim=cfg['input_dim'],target_transform='standard',scaler_fit='train',outer_test='NOT_RUN'))
  truth=np.asarray(truth).reshape(-1);prediction=np.asarray(prediction).reshape(-1)
  if not np.isfinite(truth).all() or not np.isfinite(prediction).all() or not math.isfinite(float(r2)):raise FloatingPointError('nonfinite predictions')
  ids=selected if formal else entry['validation_indices']
  np.savez(folder/('test_predictions.npz' if formal else 'validation_predictions.npz'),y_true=truth,y_pred=prediction,sample_keys=np.asarray([source.samples[i][0].hex() for i in local_val]),indices=np.asarray(ids),scaler_mean=scaler.scaler.mean_,scaler_scale=scaler.scaler.scale_,best_epoch=epoch,geometry_valid=np.asarray([source.entries[i]['geometry_status']=='PASS' for i in local_val]),outer_test='RUN' if formal else 'NOT_RUN')
  write(folder/'metrics.json',dict(status='PASS',arm=arm,task=task,fold=fold,modes=cfg,parameters=sum(p.numel() for p in model.parameters()),epochs=total_epochs,updates=updates,r2=float(r2),rmse=float(np.sqrt(np.mean((truth-prediction)**2))),outer_test='RUN' if formal else 'NOT_RUN'))
 except BaseException as error:
  write(folder/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),updates=updates,completed_epochs=len(history),time=now()))
  raise
 finally:
  if source is not None:source.close()

def initialize():
 folder=OUT/'initialization';folder.mkdir(parents=True,exist_ok=False)
 cfg=make_config();validate_config(cfg,'F1_MUL')
 reference=read(U1/'final_review/recovery1/seven_task_results.json')['arms']['U0_FUSED']
 if reference['macro_test_r2']!=.7868789263757971 or set(reference['tasks'])!=set(TASKS):raise ValueError('U0_FUSED reference mismatch')
 for task in TASKS:
  folds=reference['tasks'][task]['folds']
  if len(folds)!=5 or any(not Path(f['predictions']).is_file() for f in folds):raise ValueError('U0_FUSED reference matrix incomplete')
 package=torch.load(parent('F1_MUL'),map_location='cpu',weights_only=False)
 model=GraphInteractionArm(package,'F1_MUL',0,cfg)
 from src.training.glt_dual_runtime import save_checkpoint
 save_checkpoint(folder/'interaction_initial.pt',model.interaction.state_dict())
 write(folder/'F1_MUL.json',cfg)
 write(folder/'summary.json',dict(parameters=PARAMETERS,new_parameters=NEW_PARAMETERS,interaction=interaction_config(),parent=str(parent('F1_MUL')),reference=str(U1/'final_review/recovery1/seven_task_results.json')))

if __name__=='__main__':
 from src.training.glt_dual_runtime import require_tmux
 require_tmux()
 parser=argparse.ArgumentParser();parser.add_argument('--initialize',action='store_true');parser.add_argument('--arm',choices=ARMS)
 parser.add_argument('--task',choices=TASKS);parser.add_argument('--fold',type=int,choices=range(5));parser.add_argument('--formal',action='store_true')
 args=parser.parse_args()
 if args.initialize:initialize()
 elif args.arm and args.task and args.fold is not None:run_ft(args.arm,args.task,args.fold,args.formal)
 else:raise ValueError('invalid unit arguments')
