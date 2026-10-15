"""Explicit GP validation training or frozen best-only prediction; never reads test during development."""
import argparse,json,sys,time,math,fcntl
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training.gp_sota_contract import *
from src.modules.glt_mcl_ph import GPEncoder
from src.dataset.glt_mcl_ph import prepare_gp_sample,gp_collate
from src.training.glt_dual_runtime import require_tmux,open_source,pretrain_source_fields,save_checkpoint
from src.training.mcl_ph_direct_contract import assert_actual_equal
from scripts.finetune_mcl_ph import periodic_tdl_optimizer as _f0_optimizer
from scripts.finetune_mcl_ph import (run_periodic_tdl_epochs,configure_periodic_tdl_dropout,periodic_tdl_head,periodic_tdl_trainability,compact_train_validation_indices,clone_clean_record)
from scripts.run_mcl_ph_r4_fast import atomic_json,timestamp
from src.utils import set_global_seed,scale_targets,evaluate
from src.modules.glt_adaptation import AdaptationAdapter

class GPArm(nn.Module):
 def __init__(self,route):super().__init__();self.model=GPEncoder(route=route)
 def forward(self,data):return self.model(data)

class CleanDataset:
 def __init__(self,source,targets):self.source=source;self.raw_targets=np.asarray(targets);self.targets=self.raw_targets.copy();self.cache={}
 def __len__(self):return len(self.source)
 def set_target_override(self,y):self.targets=np.asarray(y).reshape(-1)
 def __getitem__(self,i):
  if i not in self.cache:
   top,trimer,s,static,_,identity=pretrain_source_fields(self.source,i,include_identity=True)
   self.cache[i]=prepare_gp_sample(top,trimer,s,seed=42,key=self.source.samples[i][0].hex(),position=i,static=static,identity=identity,view='clean',ratio=None,sigma=0)
  d,t=clone_clean_record(self.cache[i]);d.y=torch.tensor([self.targets[i]],dtype=torch.float32);return d,t

def collate(records):return gp_collate(records)[0]

def optimizer_for(model,task,*,head_only,backbone_lr=None,recipe='F0'):
 from scripts.finetune_mcl_ph import periodic_tdl_optimizer
 if head_only or recipe=='F0':return _f0_optimizer(model,task,head_only=head_only)
 from scripts.finetune_mcl_ph import PERIODIC_TDL_WEIGHT_DECAY
 lr=2e-4 if task=='xc' else 1e-4;wd=PERIODIC_TDL_WEIGHT_DECAY[task];head=set(id(p) for p in model.model.predictor.parameters())
 groups=[]
 for name,rate,condition in [('public',lr*.3,lambda n:n.startswith(('model.o8.','model.glt.','model.norm2.','model.norm3.'))),('new',lr,lambda n:n.startswith(('model.experts.','model.router.','model.adaptor.'))),('head',1e-3,lambda n:n.startswith('model.predictor.'))]:
  params=[p for n,p in model.named_parameters() if condition(n)];assert params
  groups.append(dict(params=params,lr=rate,weight_decay=wd,name=name))
 assert sum(len(g['params']) for g in groups)==len(list(model.parameters()))
 evidence=[dict(name=g['name'],lr=g['lr'],weight_decay=wd,num_tensors=len(g['params']),num_parameters=sum(p.numel() for p in g['params'])) for g in groups]
 return torch.optim.AdamW(groups),evidence

def validate_promotion(old,m,saved,initial,contract,scope,key):
 if m['outer_test']!='NOT_RUN' or (old/'test_predictions.npz').exists():raise ValueError('development test was read')
 if read(old/'runtime.json').get('status')!='PASS':raise ValueError('development failed')
 assert_actual_equal(saved['contract'],contract,'promotion.contract')
 assert_actual_equal(m['contract'],contract,'promotion.metrics')
 binding=scope['unit_contracts'].get(key,{})
 expected=dict(contract=contract,development_unit=str(old),best_epoch=m['best_epoch'],scaler_mean=saved['scaler_mean'],scaler_scale=saved['scaler_scale'],scaler_train_raw=saved['scaler_train_raw'],frozen_weights=binding.get('frozen_weights'))
 if binding!=expected or not binding['frozen_weights']:raise ValueError('frozen promotion binding differs')
 snapshot=torch.load(binding['frozen_weights'],map_location='cpu',weights_only=False)
 assert_actual_equal(snapshot['state_dict'],saved['state_dict'],'promotion.frozen_weights')
 assert_actual_equal(snapshot['head_initial'],saved['head_initial'],'promotion.frozen_head')
 assert_actual_equal(snapshot['contract'],contract,'promotion.frozen_contract')
 assert_actual_equal(saved['head_initial'],initial,'promotion.head_initial')
 if not all(torch.isfinite(v).all() for v in saved['state_dict'].values()):raise FloatingPointError('promotion model nonfinite')
 best=min(m['history'],key=lambda r:r['validation_rmse'])
 if best['epoch']!=m['best_epoch'] or best['validation_rmse']!=m['best_validation_rmse']:raise ValueError('promotion best selection differs')
 if saved['best_epoch']!=m['best_epoch'] or saved['best_validation_rmse']!=m['best_validation_rmse']:raise ValueError('promotion checkpoint selection differs')
 if sum(r['training_steps'] for r in m['history'])!=m['optimizer_updates']:raise ValueError('promotion update ledger differs')


def main(*, failure_context=None):
 p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--fold',type=int,required=True);p.add_argument('--route',required=True);p.add_argument('--recipe',default='F0');p.add_argument('--stage',choices=('smoke','development','formal'),required=True);p.add_argument('--scope');p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--reuse')
 args=p.parse_args();require_tmux()
 if args.stage!='smoke' and read(OUT/'p1_acceptance.json')['status']!='PASS':raise ValueError('P1 not accepted')
 scope=read(args.scope) if args.scope else {};key=validate_ft_scope(args.stage,args.task,args.fold,args.route,args.recipe,scope)
 if args.stage=='formal' and not args.reuse:raise ValueError('GP formal must reuse its own best')
 if args.stage=='formal' and args.route=='GU':raise ValueError('GU is diagnostic only')
 recipe=ft_recipe(args.task,args.fold,args.route,args.recipe);manifest=recipe['split'];entry=manifest['folds'][args.fold]
 folder=Path(args.output).resolve();assert folder.is_relative_to(OUT)
 validate_r3_output(args.stage,args.task,args.fold,args.route,args.recipe,scope,folder)
 folder.mkdir(parents=True,exist_ok=False)
 if failure_context is not None:failure_context['owned_output']=folder
 counter={'kind':'finetune','stage':args.stage,'task':args.task,'attempted_updates':0,'completed_updates':0,'epochs':0,'attempted_epochs':0}
 if args.stage!='formal':
  cap=24 if args.stage=='smoke' else epochs(args.task)*math.ceil(len(entry['train_indices'])/24)
  counter.update(reserved_updates=cap,reserved_epochs=2 if args.stage=='smoke' else epochs(args.task),started=True)
  with (OUT/'ft_budget.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   prior=[read(b) for b in OUT.glob('**/budget.json') if read(b).get('kind')=='finetune' and read(b).get('stage')==args.stage]
   if args.stage=='development':
    assert len(prior)+1<=59 and sum(v.get('reserved_epochs',v['epochs']) for v in prior)+counter['reserved_epochs']<=4080,'FT starts/epoch reservation exceeded'
    assert sum(v.get('reserved_updates',v['attempted_updates']) for v in prior)+cap<=69930,'FT update reservation exceeded'
   else:assert not prior,'FT smoke already consumed'
   atomic_json(folder/'budget.json',counter)
 start=time.perf_counter();seed=42+args.fold;set_global_seed(seed);model=GPArm(args.route)
 initial={k:v.clone() for k,v in periodic_tdl_head(model).state_dict().items()}
 reference=(ROOT/f'results/mcl_ph_r5_20261004/paper5_outer/xc_standard/glt_ref/xc/fold{args.fold}' if args.task=='xc' else ROOT/f'results/mcl_ph_r4_20261002/fast_campaign_04/paper5_outer/glt_ref/{args.task}/fold{args.fold}')
 assert_actual_equal(initial,torch.load(reference/'head_initial.pt',map_location='cpu',weights_only=False),'head.initial')
 save_checkpoint(folder/'head_initial.pt',initial)
 pkg=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
 assert pkg['schema']=='gp-sota-deploy-v2' and pkg['route']==args.route and pkg['step']==(4 if args.stage=='smoke' else 5000)
 state=model.model.state_dict();expected=set(state)-{k for k in state if k.startswith('predictor.')}
 assert expected==set(pkg['state_dict'])
 for k,v in pkg['state_dict'].items():assert state[k].shape==v.shape and state[k].dtype==v.dtype and torch.isfinite(v).all();state[k]=v
 model.model.load_state_dict(state,strict=True);model.model.router.configure('top2',pkg['step']);configure_periodic_tdl_dropout(model,args.task)
 pretrain_state=torch.load(Path(args.checkpoint).with_name(f"resume_{pkg['step']:05d}.pt"),map_location='cpu',weights_only=False)
 assert pretrain_state['contract']['recipe']==pretrain_recipe(args.route)
 assert_actual_equal({k.removeprefix('encoder.'):v for k,v in pretrain_state['model'].items() if k.startswith('encoder.')},pkg['state_dict'],'pretrain.resume_deploy')
 contract=dict(training=recipe,pretrain_package=str(Path(args.checkpoint).resolve()),pretrain_step=pkg['step'],pretrain_recipe=pretrain_state['contract']['recipe'],pretrain_cache=pretrain_state['contract']['cache'],architecture=pkg['architecture'],source_metadata=read(DOWN/'direct_source.json'),common_schema='gp-sota-common-init-v2')
 del pretrain_state
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(device);source=None
 try:
  if args.stage=='formal':
   old=Path(args.reuse).resolve();m=read(old/'metrics.json');saved=torch.load(old/'best.pt',map_location='cpu',weights_only=False)
   validate_promotion(old,m,saved,initial,contract,scope,key)
   model.load_state_dict(saved['state_dict'],strict=True);assert_actual_equal({k:v.cpu() for k,v in model.state_dict().items()},saved['state_dict'],'promotion.weights')
   # Use the existing train-only scaler implementation through a tiny holder.
   class Holder:
    raw_targets=np.array(saved['scaler_train_raw'])
    def set_target_override(self,y):self.targets=y
   h=Holder();scaler=scale_targets(h,args.task,train_indices=list(range(len(h.raw_targets))),transform_mode='standard')
   np.testing.assert_array_equal(saved['scaler_mean'],scaler.scaler.mean_)
   np.testing.assert_array_equal(saved['scaler_scale'],scaler.scaler.scale_)
   indices=sorted(entry['test_indices']);source,frame=open_source(DOWN,DOWN,task=args.task,dual_static_root=DOWN,selected_indices=indices,record_index_path=DOWN/f'{args.task}_offsets.npy',split_manifest=manifest)
   data=CleanDataset(source,frame['label'].to_numpy(dtype=float));data.set_target_override(scaler.transform(data.raw_targets));loader=DataLoader(data,batch_size=24,shuffle=False,num_workers=0,collate_fn=collate)
   _,r2,y,z=evaluate(AdaptationAdapter(model),loader,nn.MSELoss(),device,scaler=scaler,amp_dtype='fp32');y,z=np.array(y).reshape(-1),np.array(z).reshape(-1)
   assert np.isfinite(y).all() and np.isfinite(z).all();m=dict(m,stage='formal',outer_test='RUN',test_r2=float(r2),test_rmse=float(np.sqrt(np.mean((y-z)**2))),promotion_optimizer_updates=0,reused_development_unit=str(old),frozen_selection=str(Path(args.scope).resolve()))
   np.savez(folder/'test_predictions.npz',y_true=y,y_pred=z,test_indices=indices,sample_keys=[k.hex() for k,_ in source.samples],original_row=indices,fold=args.fold,seed=seed,best_epoch=m['best_epoch'])
   save_checkpoint(folder/'best.pt',dict(saved,stage='formal',promotion_optimizer_updates=0))
  else:
   selected,train_local,val_local=compact_train_validation_indices(entry['train_indices'],entry['validation_indices'])
   source,frame=open_source(DOWN,DOWN,task=args.task,dual_static_root=DOWN,selected_indices=selected,record_index_path=DOWN/f'{args.task}_offsets.npy',split_manifest=manifest)
   dataset=CleanDataset(source,frame['label'].to_numpy(dtype=float));scaler=scale_targets(dataset,args.task,train_indices=train_local,transform_mode='standard')
   train_loader=DataLoader(Subset(dataset,train_local),batch_size=24,shuffle=True,num_workers=0,collate_fn=collate,generator=torch.Generator().manual_seed(seed))
   val_loader=DataLoader(Subset(dataset,val_local),batch_size=24,shuffle=False,num_workers=0,collate_fn=collate)
   import scripts.finetune_mcl_ph as ft
   original_optimizer=ft.periodic_tdl_optimizer;original_epoch=ft.train_epoch
   ft.periodic_tdl_optimizer=lambda model,task,**kwargs:optimizer_for(model,task,recipe=args.recipe,**kwargs)
   # Preserve the original optimizer function used by optimizer_for's F0 path.
   globals()['optimizer_for_f0']=original_optimizer
   assert cap==(24 if args.stage=='smoke' else epochs(args.task)*math.ceil(len(train_local)/24))
   epoch_gradient={}
   def counted_epoch(*a,**kw):
    counter['attempted_epochs']+=1;assert counter['attempted_epochs']<=counter['reserved_epochs']
    atomic_json(folder/'budget.json',counter);value=original_epoch(*a,**kw)
    epoch_gradient.clear()
    for name,part in dict(o8=model.model.o8,galformer=model.model.glt,experts=model.model.experts,router=model.model.router,adaptor=model.model.adaptor,head=model.model.predictor).items():
     grads=[p.grad.detach().float().square().sum() for p in part.parameters() if p.grad is not None]
     epoch_gradient[name]=float(torch.stack(grads).sum().sqrt()) if grads else 0.
     if not np.isfinite(epoch_gradient[name]):raise FloatingPointError('FT module gradient NaN/Inf: '+name)
    return value
   ft.train_epoch=counted_epoch
   original_step=torch.optim.AdamW.step
   def counted_step(opt,*a,**kw):
    assert counter['attempted_updates']<cap,'FT unit update cap reached'
    used=sum(read(b)['attempted_updates'] for b in OUT.glob('**/budget.json') if read(b).get('kind')=='finetune' and read(b).get('stage')==args.stage)
    assert used<(24 if args.stage=='smoke' else 69930),'FT global budget exhausted'
    counter['attempted_updates']+=1;atomic_json(folder/'budget.json',counter)
    value=original_step(opt,*a,**kw);counter['completed_updates']+=1;atomic_json(folder/'budget.json',counter);return value
   torch.optim.AdamW.step=counted_step
   def progress(row):
    row['gradient_summary']=dict(last_batch_after_clip=epoch_gradient.copy(),measurement='existing final-batch gradients; no extra forward/backward')
    counter['epochs']=row['epoch'];atomic_json(folder/'budget.json',counter);print(json.dumps(row),flush=True)
   result=run_periodic_tdl_epochs(model,train_loader,val_loader,nn.MSELoss(),device,task=args.task,scaler=scaler,smoke=args.stage=='smoke',progress=progress)
   torch.optim.AdamW.step=original_step;ft.periodic_tdl_optimizer=original_optimizer;ft.train_epoch=original_epoch
   assert result['optimizer_updates']==counter['completed_updates']
   y,z=result['predictions'];indices=entry['validation_indices']
   np.savez(folder/'validation_predictions.npz',y_true=y,y_pred=z,validation_indices=indices,sample_keys=[source.samples[i][0].hex() for i in val_local],original_row=indices,fold=args.fold,seed=seed,best_epoch=result['best_epoch'],outer_test='NOT_RUN')
   m=dict(status='PASS',task=args.task,fold=args.fold,route=args.route,recipe=args.recipe,stage=args.stage,outer_test='NOT_RUN',contract=contract,history=result['history'],best_epoch=result['best_epoch'],best_validation_rmse=result['best_rmse'],best_validation_r2=result['best_r2'],optimizer_updates=result['optimizer_updates'],executed_epochs=len(result['history']),optimizer_groups_by_stage=result['optimizer_groups_by_stage'],initialization_seed=seed)
   save_checkpoint(folder/'best.pt',dict(state_dict=result['state'],contract=contract,scaler_mean=scaler.scaler.mean_.tolist(),scaler_scale=scaler.scaler.scale_.tolist(),scaler_train_raw=dataset.raw_targets[train_local].tolist(),head_initial=initial,**{k:m[k] for k in ('task','fold','route','recipe','stage','best_epoch','best_validation_rmse','optimizer_updates','executed_epochs')}))
  atomic_json(folder/'metrics.json',m);atomic_json(folder/'runtime.json',dict(status='PASS',command=sys.argv,exit_code=0,started_seconds=start,wall_seconds=time.perf_counter()-start,finished_at=timestamp()))
 finally:
  if source:source.close()

def run_cli():
 failure_context={}
 try:main(failure_context=failure_context)
 except BaseException as e:
  # Rejected paths and previous attempts are never this process's output.
  folder=failure_context.get('owned_output')
  if folder is not None:
   try:atomic_json(folder/'failure.json',dict(status='FAILED',error=f'{type(e).__name__}: {e}',automatic_retry=False,failed_at=timestamp()))
   except Exception as metadata_error:print(f'failure metadata unavailable: {type(metadata_error).__name__}: {metadata_error}',file=sys.stderr,flush=True)
  raise

if __name__=='__main__':run_cli()
