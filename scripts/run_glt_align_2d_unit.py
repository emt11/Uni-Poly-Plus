#!/usr/bin/env python3
"""U1 initialization and fixed 2D/fused FT units; no retries or task exceptions."""
import argparse,json,sys,datetime,math,os,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/glt_optimization_20261011/r4/U1'
RECOVERY=OUT/'recovery1'
REVIEW=OUT/'final_review/recovery1'
BASE=ROOT/'results/glt_periodic_seam_20261009/r1'
COMMON=BASE/'initialization/F_CE_step0.pt'
ARMS=('U0_2D','UA_2D','U0_FUSED','UA_FUSED');TASKS=('eea','egb','egc','ei','eps','nc','xc')
import torch
from torch import nn
from src.modules.glt_dual import build_dual_glt_model,mean_pool
from src.modules.glt_dual_pretrain import load_deployment,u1_alignment_options
PROFILE=dict(name='uniform_u1',batch=24,amp_dtype='fp32',workers=0,head_epochs=10,joint_epochs=60,
 encoder_dropout=0.0,head_dropout=0.10,modality_dropout=0.0,weight_decay=0.02,head_only_lr=3e-4,head_joint_lr=1e-3,
 backbone_lr=1e-4,grad_clip=5.0,head_scheduler=dict(name='CosineAnnealingLR',T_max=10,eta_min=3e-5),
 joint_scheduler=dict(name='CosineAnnealingWarmRestarts',T_0=10,T_mult=1,eta_min=1e-6))
GEOMETRY=dict(geometry_revision='glt_periodic_seam_r1',geometry_mode='finite_incident',angle_value_enabled=True,readout='incident')
def prediction_config(arm):
 if arm not in ARMS:raise ValueError('invalid U1 arm')
 pure=arm.endswith('_2D');aligned=arm.startswith('UA_')
 return dict(prediction_path='pure_2d' if pure else 'fused',input_dim=512 if pure else 1024,
  parent_align_revision='glt_align_2d_u1' if aligned else 'legacy_f_ce',align_enabled=aligned,
  align_init_seed=314161 if aligned else None,align_input='raw_graph_2d_graph_3d' if aligned else None)
def parent(arm):return OUT/'pretrain/P_ALIGN/deploy_05000.pt' if arm.startswith('UA_') else BASE/'pretrain/F_CE/deploy_05000.pt'
def validate_config(cfg,arm):
 expected=dict(plan='GLT-OPTIMIZATION-20261011-01',revision='r4',cycle='U1',arm=arm,
  architecture='O8-BondPath-GalformerTrimer-Hop2-MD200-r1',fusion_mode='token',fusion_revision='glt_fusion_r1',**GEOMETRY,
  use_md200=False,md200_root=None,atom_ce_input='fused',local_enabled=False,third_task='none',torsion_modules=False,
  fusion_gate='none',fusion_attention_mode='global',incident_bias=0.0,endpoint_revision=None,quality_root=None,
  align_temperature=.1,align_weight=.1,align_warmup_steps=1000,parent=str(parent(arm)),parent_step=5000,
  common_initialization=str(COMMON),ft_profile=PROFILE,split='paper5_outer',ft_seed_formula='42+fold',target_transform='train_only_standard',
  head_initialization='independent_cpu_seed42+fold_shared_same_path' if arm.endswith('_2D') else 'original_B0_shared_same_path',
  **prediction_config(arm))
 if any(k not in cfg or type(cfg[k]) is not type(v) or cfg[k]!=v for k,v in expected.items()):raise ValueError('invalid U1 configuration')
 return cfg
def strict_restore(module,state):
 current=module.state_dict()
 if set(current)!=set(state):raise ValueError('FT tensor names mismatch')
 for k,v in state.items():
  if v.shape!=current[k].shape or v.dtype!=current[k].dtype or not torch.isfinite(v).all():raise ValueError('FT tensor mismatch '+k)
 module.load_state_dict(state,strict=True)
class Pure2D(nn.Module):
 def __init__(self,full):
  super().__init__();self.o8=full.o8;self.norm2=full.norm2
  self.head=nn.Sequential(nn.Linear(512,512),nn.GELU(),nn.Dropout(.10),nn.Linear(512,1))
 def features(self,data):
  atoms,_=self.o8(data)
  return self.norm2(mean_pool(atoms,data.canonical_graph_index,data.graph_available.numel()))
 def forward(self,data):return self.head(self.features(data))
class Fused(nn.Module):
 def __init__(self,full):super().__init__();self.model=full
 def features(self,data):return self.model.fuse(self.model.encode(data))
 def forward(self,data):return self.model.predictor(self.features(data))
def build_arm(package,arm,fold,cfg):
 if fold not in range(5):raise ValueError('invalid fold')
 validate_config(cfg,arm);aligned=u1_alignment_options(package)
 if bool(aligned)!=arm.startswith('UA_'):raise ValueError('ALIGN parent mismatch')
 if package.get('endpoint_revision') is not None or package.get('fusion_attention_mode','global')!='global' or package.get('incident_bias',0)!=0:raise ValueError('locality parent forbidden')
 for k in ('architecture','fusion_mode','fusion_revision','geometry_revision','geometry_mode','angle_value_enabled','readout','use_md200','atom_ce_input','third_task','align_enabled','local_enabled','torsion_modules'):
  if package.get(k)!=cfg[k]:raise ValueError('parent architecture mismatch '+k)
 from src.modules.glt_dual_pretrain import restore_deployment
 from src.modules.glt_dual import restore_pure_2d
 if cfg['prediction_path']=='pure_2d':
  # The actual frozen head is loaded by run_ft; this temporary head does
  # not consume the caller's initialization RNG.
  with torch.random.fork_rng(devices=[]):
   head=nn.Sequential(nn.Linear(512,512),nn.GELU(),nn.Dropout(.10),nn.Linear(512,1))
  return restore_pure_2d(package,head.state_dict())
 return Fused(restore_deployment(package))

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def epochs(task):return 70
def ft_folder(arm,task,fold):return OUT/'finetune'/arm/task/f'fold{fold}'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT=ROOT/'data/splits/mips_paper5_outer'
def head_path(arm,task,fold):
 return OUT/'initialization/heads_2d'/task/f'fold{fold}.pt' if arm.endswith('_2D') else ROOT/'results/glt_fusion_20261007/r1/finetune/B0'/task/f'fold{fold}'/'head_initial.pt'
def run_ft(arm,task,fold,formal=False):
 import numpy as np,torch
 from torch import nn
 from torch.utils.data import DataLoader,Subset
 from sklearn.preprocessing import StandardScaler
 from src.utils import set_global_seed,scale_targets,evaluate
 import scripts.finetune_mcl_ph as old
 from src.dataset.glt_dual import dual_glt_collate,glt_2d_collate
 from src.training.glt_dual_runtime import open_source,save_checkpoint
 from src.modules.glt_dual_pretrain import load_deployment
 from src.modules.glt_adaptation import AdaptationAdapter
 folder=OUT/'formal'/arm/task/f'fold{fold}' if formal else ft_folder(arm,task,fold)
 folder.mkdir(parents=True,exist_ok=False)
 updates={'attempted':0,'completed':0};history=[];source=None
 try:
  cfg=read(OUT/'initialization'/f'{arm}.json');set_global_seed(42+fold)
  package=torch.load(parent(arm),map_location='cpu',weights_only=False)
  model=build_arm(package,arm,fold,cfg)
  head=torch.load(head_path(arm,task,fold),map_location='cpu',weights_only=False)
  strict_restore(old.periodic_tdl_head(model),head)
  model.to('cuda:0');old.configure_periodic_tdl_dropout(model,task,profile='uniform_u1')
  manifest=read(SPLIT/f'{task}.json');entry=manifest['folds'][fold]
  best_folder=ft_folder(arm,task,fold)
  if formal:
   selection=read(OUT/'evaluation_before_test.json')
   if selection['validation_units']!=140:raise ValueError('selection not complete')
   best=torch.load(best_folder/'best.pt',map_location='cpu',weights_only=False)
   if best['config']!=cfg or best['task']!=task or best['fold']!=fold or best.get('seed')!=42+fold or best.get('parent')!=str(parent(arm)):raise ValueError('formal model metadata mismatch')
   units=[u for u in selection['units'] if (u['arm'],u['task'],u['fold'])==(arm,task,fold)]
   if len(units)!=1 or units[0]['best']!=str(best_folder/'best.pt') or selection['configs'][arm]!=cfg or read(best_folder/'scaler.json')!=units[0]['scaler'] or best['scaler_mean']!=units[0]['scaler']['mean'] or best['scaler_scale']!=units[0]['scaler']['scale']:
    raise ValueError('formal frozen best/config/scaler mismatch')
   if best.get('prediction_path')!=cfg['prediction_path'] or best.get('input_dim')!=cfg['input_dim'] or best.get('head_initial')!=str(head_path(arm,task,fold)):raise ValueError('formal path/head mismatch')
   strict_restore(model,best['state_dict'])
   selected=sorted(entry['test_indices']);local_val=list(range(len(selected)))
   if selected!=units[0]['test_indices'] or units[0]['parent']!=str(parent(arm)) or best['best_epoch']!=units[0]['best_epoch'] or any(units[0][k]!=cfg[k] for k in prediction_config(arm)) or units[0]['head_initial']!=str(head_path(arm,task,fold)):raise ValueError('formal frozen indices/parent/epoch mismatch')
  else:selected,local_train,local_val=old.compact_train_validation_indices(entry['train_indices'],entry['validation_indices'])
  md_root=read(Path(cfg['md200_root'])/'metadata.json').get('downstream_sidecar') if cfg['use_md200'] else None
  if cfg['use_md200'] and not md_root:raise ValueError('MD sidecar missing downstream binding')
  source,frame=open_source(DOWN,DOWN,task=task,dual_static_root=DOWN,selected_indices=selected,split_manifest=manifest,record_index_path=DOWN/f'{task}_offsets.npy',expected_task_rows=manifest['sample_count'],verify_key_digest=False,md200_root=md_root)
  if frame['original_row'].astype(int).tolist()!=selected:raise ValueError('selected source rows differ')
  class NewCleanDataset(old.CleanDualDataset):
   def __getitem__(self,index):
    if index not in self._clean_cache:
     from src.training.glt_dual_runtime import pretrain_source_fields
     from src.dataset.glt_dual import build_dual_sample,build_glt_2d_sample
     top,tri,text,static,_=pretrain_source_fields(self.source,index)
     self._clean_cache[index]=(build_glt_2d_sample(top,text,static=static) if cfg['prediction_path']=='pure_2d' else build_dual_sample(top,tri,text,static=static,geometry_mode=cfg['geometry_mode']))
    data=old.clone_clean_record(self._clean_cache[index])
    data.y=torch.tensor([float(self.targets[index])],dtype=torch.float32)
    if cfg['prediction_path']=='fused' and not hasattr(data,'md200'):
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
  collate=glt_2d_collate if cfg['prediction_path']=='pure_2d' else dual_glt_collate
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
   save_checkpoint(folder/'best.pt',dict(state_dict=result['state'],config=cfg,task=task,fold=fold,seed=42+fold,parent=str(parent(arm)),best_epoch=epoch,scaler_mean=scaler.scaler.mean_.tolist(),scaler_scale=scaler.scaler.scale_.tolist(),history=history,optimizer_groups=result['optimizer_groups_by_stage'],optimizer_updates=updates['completed'],head_initial=str(head_path(arm,task,fold)),prediction_path=cfg['prediction_path'],input_dim=cfg['input_dim'],target_transform='standard',scaler_fit='train',outer_test='NOT_RUN'))
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

def save_head_initial(head,destination,*,resume=False):
 from src.training.glt_dual_runtime import save_checkpoint
 destination=Path(destination)
 if destination.exists():
  if not resume:raise FileExistsError('head initialization already exists: '+str(destination))
  saved=torch.load(destination,map_location='cpu',weights_only=False)
  expected=head.state_dict()
  if set(saved)!=set(expected) or any(v.shape!=expected[k].shape or v.dtype!=expected[k].dtype or not torch.equal(v,expected[k]) for k,v in saved.items()):raise ValueError('existing head initialization mismatch')
  return False
 if destination.with_suffix(destination.suffix+'.tmp').exists():raise FileExistsError('unfinished head writer: '+str(destination))
 destination.parent.mkdir(parents=True,exist_ok=True)
 save_checkpoint(destination,head.state_dict())
 return True

def initialize(*,resume=False):
 from src.modules.glt_dual_pretrain import DualPretrainer
 from src.training.glt_dual_runtime import save_checkpoint
 folder=OUT/'initialization'
 if resume:
  if not folder.is_dir() or (folder/'summary.json').exists():raise ValueError('recovery requires incomplete existing initialization')
 else:folder.mkdir(parents=True,exist_ok=False)
 common=torch.load(COMMON,map_location='cpu',weights_only=False)
 if common['step']!=0:raise ValueError('common PT step mismatch')
 base=read(BASE/'initialization/F_CE.json')
 fixed=dict(expected_world_size=4,microbatch=84,global_batch=1008,seed=42,max_optimizer_steps=5000,save_every=5000,lr=2e-4,weight_decay=0,warmup_steps=2000,schedule_total_steps=20000,end_lr=1e-9,amp_dtype='bf16',atom_mask_ratio=.3,noise_sigma=.03,loss_weights=[1.,1.,0.],sample_index_split='train',sample_index_artifact=str(ROOT/'results/mcl_ph_next_20260929/p0/direct_indices_v1/summary.json'))
 if any(base.get(k)!=v for k,v in fixed.items()):raise ValueError('original F_CE PT recipe changed')
 cfg=dict(base,plan='GLT-OPTIMIZATION-20261011-01',revision='r4',cycle='U1',arm='P_ALIGN',cycle_root=str(OUT),
  initial_state=str(folder/'P_ALIGN_step0.pt'),align_enabled=True,align_revision='glt_align_2d_u1',align_init_seed=314161,
  align_temperature=.10,align_weight=.10,align_warmup_steps=1000,fusion_gate='none',quality_root=None)
 torch.manual_seed(42)
 trainer=DualPretrainer('token',fusion_revision='glt_fusion_r1',atom_ce_input='fused',geometry_head_norm=True,opt_mode='GLT_FUSION_R1',
  align_enabled=True,align_revision='glt_align_2d_u1',align_init_seed=314161,**GEOMETRY)
 state=trainer.state_dict();new=set(state)-set(common['model'])
 expected={f'align_proj{branch}.net.{layer}.{kind}' for branch in (2,3) for layer in (0,2) for kind in ('weight','bias')}
 if new!=expected or set(common['model'])-set(state) or sum(state[k].numel() for k in new)!=328448:raise ValueError('ALIGN common initialization schema mismatch')
 if resume:
  saved=torch.load(folder/'P_ALIGN_step0.pt',map_location='cpu',weights_only=False)
  if saved.get('step')!=0 or saved.get('opt_mode')!=cfg['opt_mode'] or saved.get('arm')!='P_ALIGN' or saved.get('config')!=cfg or read(folder/'P_ALIGN.json')!=cfg:raise ValueError('saved recovery step0/config mismatch')
  strict_restore(trainer,saved['model']);state=trainer.state_dict()
 for k,v in common['model'].items():
  if v.shape!=state[k].shape or v.dtype!=state[k].dtype or not torch.isfinite(v).all():raise ValueError('common tensor mismatch '+k)
  if resume:
   if not torch.equal(state[k],v):raise ValueError('saved recovery public tensor mismatch '+k)
  else:state[k]=v.clone()
 trainer.load_state_dict(state,strict=True)
 for k,v in common['model'].items():
  if not torch.equal(v,trainer.state_dict()[k]):raise ValueError('public initialization changed '+k)
 if not resume:
  save_checkpoint(folder/'P_ALIGN_step0.pt',dict(model=state,step=0,opt_mode=cfg['opt_mode'],arm='P_ALIGN',config=cfg))
  write(folder/'P_ALIGN.json',cfg)
 for arm in ARMS:
  ft=dict(plan='GLT-OPTIMIZATION-20261011-01',revision='r4',cycle='U1',arm=arm,
   architecture=trainer.encoder.architecture_name,fusion_mode='token',fusion_revision='glt_fusion_r1',**GEOMETRY,
   use_md200=False,md200_root=None,atom_ce_input='fused',local_enabled=False,third_task='none',torsion_modules=False,
   fusion_gate='none',fusion_attention_mode='global',incident_bias=0.0,endpoint_revision=None,quality_root=None,
   align_temperature=.1,align_weight=.1,align_warmup_steps=1000,parent=str(parent(arm)),parent_step=5000,
   common_initialization=str(COMMON),ft_profile=PROFILE,split='paper5_outer',ft_seed_formula='42+fold',target_transform='train_only_standard',
   head_initialization='independent_cpu_seed42+fold_shared_same_path' if arm.endswith('_2D') else 'original_B0_shared_same_path',**prediction_config(arm))
  validate_config(ft,arm)
  if resume:
   if read(folder/f'{arm}.json')!=ft:raise ValueError('saved recovery FT config mismatch '+arm)
  else:write(folder/f'{arm}.json',ft)
 for task in TASKS:
  for fold in range(5):
   with torch.random.fork_rng(devices=[]):
    torch.default_generator.manual_seed(42+fold)
    head=nn.Sequential(nn.Linear(512,512),nn.GELU(),nn.Dropout(.10),nn.Linear(512,1))
   save_head_initial(head,head_path('U0_2D',task,fold),resume=resume)
 write(folder/'summary.json',dict(common=str(COMMON),pt_parameters=sum(p.numel() for p in trainer.parameters()),new_parameters=328448,
  alignment_seed=314161,head_2d_parameters=sum(p.numel() for p in head.parameters()),arms=ARMS,public_tensors_equal=True))

if __name__=='__main__':
 from src.training.glt_dual_runtime import require_tmux
 require_tmux()
 parser=argparse.ArgumentParser();initial=parser.add_mutually_exclusive_group();initial.add_argument('--initialize',action='store_true');initial.add_argument('--complete-initialization',action='store_true');parser.add_argument('--arm',choices=ARMS)
 parser.add_argument('--task',choices=TASKS);parser.add_argument('--fold',type=int,choices=range(5));parser.add_argument('--formal',action='store_true')
 args=parser.parse_args()
 if args.initialize:initialize()
 elif args.complete_initialization:initialize(resume=True)
 elif args.arm and args.task and args.fold is not None:run_ft(args.arm,args.task,args.fold,args.formal)
 else:raise ValueError('invalid unit arguments')
