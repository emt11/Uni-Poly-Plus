#!/usr/bin/env python3
"""本周期固定训练/预测单元入口；复用正式两阶段配方，不启动旧控制器。"""
import argparse,datetime,json,math,os,sys,traceback,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/glt_fusion_20261007/r1'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT=ROOT/'data/splits/mips_paper5_outer'
TASKS=('eea','egb','egc','ei','eps','nc','xc')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def epochs(task):return 60 if task=='egc' else 70
def parent(arm):return OUT/'pretrain'/arm/'deploy_05000.pt'
def ft(arm,task,fold,formal=False,early_t0_a_test=False,ft_candidate=None,reference_root=None):
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
 folder=OUT/('formal' if formal else 'finetune')/arm/task/f'fold{fold}'
 folder.mkdir(parents=True,exist_ok=False)
 updates={'attempted':0,'completed':0};history=[];source=None
 try:
  cfg=read(OUT/('configs' if ft_candidate else 'initialization')/f'{arm}.json');set_global_seed(42+fold)
  if ft_candidate and (cfg.get('ft_candidate')!=ft_candidate or cfg.get('ft_parent')!=str(parent(arm))):
   raise ValueError('FT candidate/parent mismatch')
  model=old.GLTReferenceArm(fusion_mode=cfg['fusion_mode'],fusion_revision='glt_fusion_r1',use_md200=cfg['use_md200'])
  package=torch.load(parent(arm),map_location='cpu',weights_only=False)
  for key in ('fusion_mode','fusion_revision','use_md200','atom_ce_input','align_enabled','local_enabled','third_task'):
   if package.get(key)!=cfg[key]:raise ValueError('deployment mode mismatch '+key)
  load_deployment(model.model,package,5000)
  model.model.opt_mode=arm
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(42+fold)
   common_head=nn.Sequential(nn.Linear(1024,512),nn.GELU(),nn.Dropout(.1),nn.Linear(512,1))
  head=common_head.state_dict()
  if ft_candidate:
   reference_head=torch.load(Path(reference_root)/'finetune/B0'/task/f'fold{fold}/head_initial.pt',map_location='cpu',weights_only=False)
   if set(head)!=set(reference_head) or any(not torch.equal(head[k],reference_head[k]) for k in head):
    raise ValueError('candidate head initial state differs from B0')
  old.periodic_tdl_head(model).load_state_dict(head,strict=True)
  model.to('cuda');old.configure_periodic_tdl_dropout(model,task)
  manifest=read(SPLIT/f'{task}.json');entry=manifest['folds'][fold]
  best_folder=OUT/'finetune'/arm/task/f'fold{fold}'
  if formal:
   selection=read(OUT/('selection_before_T0_A_test.json' if early_t0_a_test else 'selection_before_test.json'))
   if early_t0_a_test:
    if arm!='T0_A' or selection.get('scope')!='T0_A_seven_tasks_five_folds' or selection['validation_units']!=35 or not selection.get('user_authorized_early_test'):
     raise ValueError('early formal prediction scope/authorization mismatch')
    units=[unit for unit in selection['units'] if unit['task']==task and unit['fold']==fold]
    if len(units)!=1 or units[0]['best']!=str(best_folder/'best.pt') or selection['config']!=cfg or read(best_folder/'scaler.json')!=units[0]['scaler']:
     raise ValueError('early formal selection/config/scaler mismatch')
   elif selection['validation_units']!=(105 if ft_candidate else 420):raise ValueError('selection not complete')
   best=torch.load(best_folder/'best.pt',map_location='cpu',weights_only=False)
   if best['config']!=cfg or best['task']!=task or best['fold']!=fold:raise ValueError('formal model metadata mismatch')
   if ft_candidate:
    units=[u for u in selection['units'] if (u['arm'],u['task'],u['fold'])==(arm,task,fold)]
    if len(units)!=1 or units[0]['best']!=str(best_folder/'best.pt') or selection['configs'][arm]!=cfg or read(best_folder/'scaler.json')!=units[0]['scaler'] or best['scaler_mean']!=units[0]['scaler']['mean'] or best['scaler_scale']!=units[0]['scaler']['scale']:
     raise ValueError('formal candidate frozen best/config/scaler mismatch')
   if early_t0_a_test and (best['scaler_mean']!=units[0]['scaler']['mean'] or best['scaler_scale']!=units[0]['scaler']['scale']):
    raise ValueError('early formal checkpoint scaler differs from frozen scaler')
   model.load_state_dict(best['state_dict'],strict=True)
   selected=sorted(entry['test_indices']);local_val=list(range(len(selected)))
  else:selected,local_train,local_val=old.compact_train_validation_indices(entry['train_indices'],entry['validation_indices'])
  md_root=read(Path(cfg['md200_root'])/'metadata.json').get('downstream_sidecar') if cfg['use_md200'] else None
  if cfg['use_md200'] and not md_root:raise ValueError('MD sidecar missing downstream binding')
  source,frame=open_source(DOWN,DOWN,task=task,dual_static_root=DOWN,selected_indices=selected,split_manifest=manifest,record_index_path=DOWN/f'{task}_offsets.npy',expected_task_rows=manifest['sample_count'],verify_key_digest=False,md200_root=md_root)
  if frame['original_row'].astype(int).tolist()!=selected:raise ValueError('selected source rows differ')
  class NewCleanDataset(old.CleanDualDataset):
   def __getitem__(self,index):
    data=super().__getitem__(index)
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
  else:scaler=scale_targets(ds,task,train_indices=local_train,transform_mode='standard')
  loader=DataLoader(Subset(ds,local_val),batch_size=24,shuffle=False,num_workers=0,collate_fn=dual_glt_collate)
  if formal:
   _,r2,truth,prediction=evaluate(AdaptationAdapter(model),loader,nn.MSELoss(),'cuda',scaler=scaler,amp_dtype='fp32')
   epoch=best['best_epoch'];total_epochs=0
  else:
   save_checkpoint(folder/'head_initial.pt',head)
   train=DataLoader(Subset(ds,local_train),batch_size=24,shuffle=True,num_workers=0,collate_fn=dual_glt_collate,generator=torch.Generator().manual_seed(42+fold))
   old_optimizer=old.periodic_tdl_optimizer
   def counted(*args,**kwargs):
    opt,evidence=old_optimizer(*args,**kwargs);real=opt.step
    def step(self,*a,**kw):
     updates['attempted']+=1
     result=real(*a,**kw);updates['completed']+=1;return result
    opt.step=types.MethodType(step,opt);return opt,evidence
   old.periodic_tdl_optimizer=counted
   def progress(rec):
    history.append(rec);print(json.dumps(rec),flush=True)
    write(folder/'progress.json',dict(history=history,updates=updates,outer_test='NOT_RUN'))
   result=old.run_periodic_tdl_epochs(model,train,loader,nn.MSELoss(),'cuda',task=task,scaler=scaler,progress=progress,ft_candidate=ft_candidate)
   if updates['attempted']!=updates['completed'] or updates['completed']!=epochs(task)*len(train):raise ValueError('FT updates mismatch')
   write(folder/'scaler.json',dict(mean=scaler.scaler.mean_.tolist(),scale=scaler.scaler.scale_.tolist(),fit='train',transform='standard'))
   truth,prediction=result['predictions'];r2=result['best_r2'];epoch=result['best_epoch'];total_epochs=len(history)
   save_checkpoint(folder/'best.pt',dict(state_dict=result['state'],config=cfg,task=task,fold=fold,seed=42+fold,parent=str(parent(arm)),best_epoch=epoch,scaler_mean=scaler.scaler.mean_.tolist(),scaler_scale=scaler.scaler.scale_.tolist(),history=history,optimizer_groups=result['optimizer_groups_by_stage'],optimizer_updates=updates['completed'],target_transform='standard',scaler_fit='train',outer_test='NOT_RUN'))
  truth=np.asarray(truth).reshape(-1);prediction=np.asarray(prediction).reshape(-1)
  if not np.isfinite(truth).all() or not np.isfinite(prediction).all() or not math.isfinite(float(r2)):raise FloatingPointError('nonfinite predictions')
  ids=selected if formal else entry['validation_indices']
  np.savez(folder/('test_predictions.npz' if formal else 'validation_predictions.npz'),y_true=truth,y_pred=prediction,sample_keys=np.asarray([source.samples[i][0].hex() for i in local_val]),indices=np.asarray(ids),scaler_mean=scaler.scaler.mean_,scaler_scale=scaler.scaler.scale_,best_epoch=epoch,outer_test='RUN' if formal else 'NOT_RUN')
  write(folder/'metrics.json',dict(status='PASS',arm=arm,task=task,fold=fold,modes=cfg,epochs=total_epochs,updates=updates,r2=float(r2),rmse=float(np.sqrt(np.mean((truth-prediction)**2))),outer_test='RUN' if formal else 'NOT_RUN'))
 except BaseException as error:
  write(folder/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),updates=updates,completed_epochs=len(history),time=now()))
  raise
 finally:
  if source is not None:source.close()

if __name__=='__main__':
 from src.training.glt_dual_runtime import require_tmux
 require_tmux()
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--arm',required=True);p.add_argument('--task',required=True,choices=TASKS)
 p.add_argument('--fold',required=True,type=int,choices=range(5));p.add_argument('--formal',action='store_true')
 p.add_argument('--early-t0-a-test',action='store_true')
 args=p.parse_args()
 if args.arm not in ('B0','T_RAW','T0','T0_A','K0','K_MD','K_MD_A','TK_MD','TK_MD_A','TK_MD_L','G_NORM','G_FGR'):raise ValueError('arm outside fixed matrix')
 if args.early_t0_a_test and (not args.formal or args.arm!='T0_A'):raise ValueError('early test is authorized only for formal T0_A')
 ft(args.arm,args.task,args.fold,args.formal,args.early_t0_a_test)
