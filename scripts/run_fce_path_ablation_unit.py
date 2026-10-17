#!/usr/bin/env python3
"""P1 same-F_CE-parent path ablation; validation-only."""
import argparse, json, math, os, sys, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import StandardScaler
import scripts.finetune_mcl_ph as old
from src.training.fce_param import recipe_for, run_periodic_tdl_epochs, configure_periodic_tdl_dropout
from src.training.glt_dual_runtime import open_source, pretrain_source_fields, save_checkpoint, require_tmux
from src.modules.glt_dual import strict_module_state, mean_pool, restore_pure_2d
from src.modules.glt_dual_pretrain import restore_deployment
from src.dataset.glt_dual import build_dual_sample, build_glt_2d_sample, dual_glt_collate, glt_2d_collate
from src.modules.glt_adaptation import AdaptationAdapter
from src.utils import set_global_seed, scale_targets

OUT=ROOT/'results/fce_path_ablation_20261015/p1'; LOG=ROOT/'logs/fce_path_ablation_20261015/p1'
BASE=ROOT/'results/glt_periodic_seam_20261009/r1'; PARENT=BASE/'pretrain/F_CE/deploy_05000.pt'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'; SPLIT=ROOT/'data/splits/mips_paper5_outer'
ARMS=('R0_RAW2D','R0_FUSED2D','R0_FUSED'); TASKS=('eea','egb','egc','ei','eps','nc','xc')
RUNTIME='fce_param_r2_cache_global_loop_v1'

def read(p): return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp')
 t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); os.replace(t,p)
def kind(arm): return {'R0_RAW2D':'raw_2d','R0_FUSED2D':'fused_2d','R0_FUSED':'fused'}[arm]
def input_dim(arm): return 1024 if arm=='R0_FUSED' else 512
def epochs(task): return 60 if task=='egc' else 70
def head_path(arm,task,fold):
 return OUT/'initialization'/'heads_512'/task/f'fold{fold}.pt' if arm!='R0_FUSED' else OUT/'initialization'/'heads_fused'/task/f'fold{fold}.pt'
def config_for(arm,task):
 return dict(plan='FCE-UA2D-AUDIT-20261015-r1',revision='p1',cycle='P1',arm=arm,task=task,
  runtime_revision=RUNTIME,parent=str(PARENT),parent_step=5000,architecture='O8-BondPath-GalformerTrimer-Hop2-MD200-r1',
  atom_ce_input='fused',fusion_mode='token',fusion_revision='glt_fusion_r1',geometry_revision='glt_periodic_seam_r1',geometry_mode='finite_incident',
  angle_value_enabled=True,readout='incident',fusion_attention_mode='global',fusion_implementation='loop',align_enabled=False,
  prediction_path=kind(arm),input_dim=input_dim(arm),head_initialization='shared_512_seed42+fold' if arm!='R0_FUSED' else 'fce_r0_reference_head',
  split=str(SPLIT),seed='42+fold',source=str(DOWN),target_transform='train_only_standard',batch=24,precision='fp32',workers=0,
  recipe=recipe_for('R0',task),formal_test='FORBIDDEN')
def validate_config(cfg,arm,task):
 if cfg!=config_for(arm,task): raise ValueError('P1 configuration mismatch')
 return cfg

class Fused2D(nn.Module):
 def __init__(self,full,head): super().__init__(); self.model=full; self.head=head
 def features(self,data):
  encoded=self.model.encode(data); self.model.fuse(encoded)
  pooled=mean_pool(encoded['fused_atom_states'],encoded['canonical_graph_index'],encoded['geometry_valid'].numel())
  return self.model.norm2(pooled)
 def forward(self,data): return self.head(self.features(data))
class Fused(nn.Module):
 def __init__(self,full): super().__init__(); self.model=full
 def features(self,data): return self.model.fuse(self.model.encode(data))
 def forward(self,data): return self.model.predictor(self.features(data))

class CleanDataset(old.CleanDualDataset):
 def __init__(self,source,targets,cfg): super().__init__(source,targets); self.cfg=cfg
 def __getitem__(self,index):
  if index not in self._clean_cache:
   top,tri,text,static,_=pretrain_source_fields(self.source,index)
   if self.cfg['prediction_path']=='raw_2d': data=build_glt_2d_sample(top,text,static=static)
   else:
    data=build_dual_sample(top,tri,text,static=static,geometry_mode='finite_incident')
    data.md200=torch.zeros((1,200)); data.md200_valid=torch.zeros(1,dtype=torch.bool)
   self._clean_cache[index]=data
  data=old.clone_clean_record(self._clean_cache[index]); data.y=torch.tensor([float(self.targets[index])],dtype=torch.float32); return data

def make_head(dim,seed):
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(seed); return nn.Sequential(nn.Linear(dim,512),nn.GELU(),nn.Dropout(.10),nn.Linear(512,1))
def build_model(package,arm,head):
 state = head.state_dict() if hasattr(head,'state_dict') else head
 if arm=='R0_RAW2D': return restore_pure_2d(package,state,expected_step=5000,dropout=.10)
 full=restore_deployment(package,expected_step=5000,dropout=.10); full.token_fusion.set_implementation('loop'); full.collect_fusion_attention=False
 if arm=='R0_FUSED2D':
  module=make_head(512,0); strict_module_state(module,state); return Fused2D(full,module)
 strict_module_state(full.predictor,state); return Fused(full)

def run_ft(arm,task,fold):
 if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2','3'): raise ValueError('single physical GPU2/3 required')
 folder=OUT/'finetune'/arm/task/f'fold{fold}'; folder.mkdir(parents=True,exist_ok=True)
 if (folder/'metrics.json').exists(): return
 cfg=validate_config(read(OUT/'initialization'/arm/(task+'.json')),arm,task); updates={'attempted':0,'completed':0}; history=[]; source=None
 try:
  set_global_seed(42+fold); package=torch.load(PARENT,map_location='cpu',weights_only=False); head=torch.load(head_path(arm,task,fold),map_location='cpu',weights_only=False)
  model=build_model(package,arm,head); del package; model.to('cuda:0'); configure_periodic_tdl_dropout(model,task)
  manifest=read(SPLIT/(task+'.json')); entry=manifest['folds'][fold]; selected,local_train,local_val=old.compact_train_validation_indices(entry['train_indices'],entry['validation_indices'])
  source,frame=open_source(DOWN,DOWN,task=task,selected_indices=selected,split_manifest=manifest,verify_key_digest=False)
  if frame['original_row'].astype(int).tolist()!=selected: raise ValueError('selected rows changed')
  ds=CleanDataset(source,frame['label'].to_numpy(dtype=np.float64),cfg); set_global_seed(42+fold); scaler=scale_targets(ds,task,train_indices=local_train,transform_mode='standard')
  ref=read(BASE/'selection_before_test.json'); old_bind=[u for u in ref['units'] if (u['arm'],u['task'],u['fold'])==('F_CE',task,fold)]
  if len(old_bind)!=1 or old_bind[0]['scaler']['mean']!=scaler.scaler.mean_.tolist() or old_bind[0]['scaler']['scale']!=scaler.scaler.scale_.tolist(): raise ValueError('FCE scaler mismatch')
  collate=glt_2d_collate if cfg['prediction_path']=='raw_2d' else dual_glt_collate
  validation=DataLoader(Subset(ds,local_val),batch_size=24,shuffle=False,num_workers=0,collate_fn=collate)
  train=DataLoader(Subset(ds,local_train),batch_size=24,shuffle=True,num_workers=0,collate_fn=collate,generator=torch.Generator().manual_seed(42+fold))
  save_checkpoint(folder/'head_initial.pt',head)
  def observe(stage):
   updates[stage]+=1; write(folder/'updates.json',updates)
   if updates['attempted']>epochs(task)*len(train): raise RuntimeError('FT update budget exceeded')
  def progress(rec):
   rec=dict(rec,prediction_path=cfg['prediction_path']); history.append(rec); print(json.dumps(rec),flush=True); write(folder/'progress.json',dict(history=history,updates=updates,outer_test='NOT_RUN'))
  model.feature_cache_context=dict(parent=str(PARENT),configuration=cfg,input_view='clean_train_validation',dtype='fp32',sample_indices=selected)
  result=run_periodic_tdl_epochs(model,train,validation,nn.MSELoss(),'cuda:0',task=task,scaler=scaler,progress=progress,recipe=cfg['recipe'],step_observer=observe)
  expected=epochs(task)*len(train)
  if updates!={'attempted':expected,'completed':expected} or len(history)!=epochs(task): raise ValueError('FT budget mismatch')
  write(folder/'scaler.json',dict(mean=scaler.scaler.mean_.tolist(),scale=scaler.scaler.scale_.tolist(),fit='train',transform='standard'))
  truth,pred=result['predictions']; truth=np.asarray(truth).reshape(-1); pred=np.asarray(pred).reshape(-1); r2=float(result['best_r2']); epoch=result['best_epoch']
  save_checkpoint(folder/'best.pt',dict(state_dict=result['state'],config=cfg,task=task,fold=fold,seed=42+fold,parent=str(PARENT),head_initial=str(head_path(arm,task,fold)),best_epoch=epoch,scaler=dict(mean=scaler.scaler.mean_.tolist(),scale=scaler.scaler.scale_.tolist()),optimizer_groups=result['optimizer_groups_by_stage'],optimizer_updates=updates['completed'],outer_test='NOT_RUN'))
  if not np.isfinite(truth).all() or not np.isfinite(pred).all() or not math.isfinite(r2): raise FloatingPointError('nonfinite validation result')
  np.savez(folder/'validation_predictions.npz',y_true=truth,y_pred=pred,indices=np.asarray(entry['validation_indices']),sample_keys=np.asarray([source.samples[i][0].hex() for i in local_val]),scaler_mean=scaler.scaler.mean_,scaler_scale=scaler.scaler.scale_,best_epoch=epoch,geometry_valid=np.asarray([source.entries[i]['geometry_status']=='PASS' for i in local_val]),outer_test='NOT_RUN')
  write(folder/'metrics.json',dict(status='PASS',arm=arm,task=task,fold=fold,config=cfg,parameters=sum(p.numel() for p in model.parameters()),epochs=len(history),updates=updates,best_epoch=epoch,r2=r2,rmse=float(np.sqrt(np.mean((truth-pred)**2))),valid_samples=len(truth),outer_test='NOT_RUN'))
 finally:
  if source is not None: source.close()

def initialize(resume=False):
 folder=OUT/'initialization'; folder.mkdir(parents=True,exist_ok=resume)
 if not resume and any(folder.iterdir()): raise FileExistsError('initialization is not empty')
 package=torch.load(PARENT,map_location='cpu',weights_only=False)
 for arm in ARMS:
  armdir=folder/arm; armdir.mkdir(exist_ok=True)
  for task in TASKS:
   write(armdir/(task+'.json'),config_for(arm,task))
   h=make_head(input_dim(arm),42+TASKS.index(task)*10)
   if arm!='R0_FUSED':
    dest=folder/'heads_512'/task
   else: dest=folder/'heads_fused'/task
   dest.mkdir(parents=True,exist_ok=True); save_checkpoint(dest/f'fold0.pt',h.state_dict())
   for fold in range(1,5):
    h=make_head(input_dim(arm),42+fold+TASKS.index(task)*10); save_checkpoint(dest/f'fold{fold}.pt',h.state_dict())
 write(folder/'summary.json',dict(status='READY',arms=list(ARMS),tasks=list(TASKS),parent=str(PARENT),split=str(SPLIT),validation_only=True,formal_test='FORBIDDEN'))

def main():
 p=argparse.ArgumentParser(); p.add_argument('--initialize',action='store_true'); p.add_argument('--arm',choices=ARMS); p.add_argument('--task',choices=TASKS); p.add_argument('--fold',type=int,choices=range(5)); p.add_argument('--formal',action='store_true')
 a=p.parse_args(); require_tmux()
 if a.formal: raise RuntimeError('P1 is validation-only; formal TEST is forbidden')
 if a.initialize: initialize(); return
 if not (a.arm and a.task is not None and a.fold is not None): p.error('unit requires --arm --task --fold')
 run_ft(a.arm,a.task,a.fold)
if __name__=='__main__':
 try: main()
 except BaseException as e:
  print(repr(e),file=sys.stderr); raise
