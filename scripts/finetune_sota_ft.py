"""Exclusive F0/SP development: train/validation only, no test-loader branch."""
import argparse,functools,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training.sota_ft_contract import *
from src.training.sota_ft_runtime import *
from src.training.glt_dual_runtime import require_tmux,open_source
from src.modules.glt_dual_pretrain import load_deployment
from src.modules.glt_adaptation import AdaptationAdapter
from src.dataset.glt_dual import dual_glt_collate
from src.utils import set_global_seed,scale_targets,train_epoch,evaluate
from scripts.finetune_mcl_ph import (GLTReferenceArm,CleanDualDataset,compact_train_validation_indices,periodic_tdl_head,periodic_tdl_trainability,periodic_tdl_optimizer,configure_periodic_tdl_dropout)

def prediction(y,z,keys,indices,scaler):
 return dict(y_true=np.asarray(y,dtype=float).reshape(-1),y_pred=np.asarray(z,dtype=float).reshape(-1),sample_keys=np.array(keys),indices=np.array(indices),scaler_mean=np.array(scaler.scaler.mean_),scaler_scale=np.array(scaler.scaler.scale_))
def score(row):
 y,z=row['y_true'],row['y_pred'];finite(y.tolist());finite(z.tolist())
 return dict(rmse=float(np.sqrt(np.mean((y-z)**2))),r2=float(1-np.sum((y-z)**2)/np.sum((y-y.mean())**2)),mae=float(np.mean(np.abs(y-z))))
def main(context):
 require_tmux();torch.set_num_threads(1)
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=('smoke','development'),required=True);p.add_argument('--task',choices=('eps','nc'),required=True);p.add_argument('--fold',type=int,required=True);p.add_argument('--arm',choices=('F0','SP'),required=True);p.add_argument('--scope',type=Path,default=OUT/'scope.json');a=p.parse_args()
 scope=read(a.scope);optimization=scope.get('plan')=='MCL-PH-OPT-20261006-01';root=ROOT/'results/mcl_ph_opt_20261006/R1' if optimization else OUT
 k=key(a.stage,a.task,a.fold,a.arm);folder=root/k;unit=check_scope(scope,a.stage,a.task,a.fold,a.arm,folder);r=unit['recipe']
 ledger=root/'budget.json';context['ledger']=ledger;context['total_cap']=2310 if optimization else 6226
 record_budget=functools.partial(account,ledger=ledger,total_cap=context['total_cap'])
 folder.mkdir(parents=True,exist_ok=False);context['folder']=folder;(folder/'writer.claim').touch(exist_ok=False);(folder/'UNCOMMITTED').touch(exist_ok=False)
 reserve(folder,a.stage,k,unit['reserved_epochs'],unit['reserved_updates'],ledger,caps=(3,210,2310) if optimization else None,plan=r['plan'],revision=r['revision']);context['key']=k;record_budget(k,None,status='RUNNING')
 seed=r['seed'];set_global_seed(seed);pkg=torch.load(PARENT,map_location='cpu',weights_only=False);finite(pkg)
 model=GLTReferenceArm(torsion=False);load_deployment(model.model,pkg,5000)
 head=torch.load(r['head_initial'],map_location='cpu',weights_only=False);finite(head)
 generated=periodic_tdl_head(model).state_dict();initial_constructor_equal=all(torch.equal(generated[n],v) for n,v in head.items())
 periodic_tdl_head(model).load_state_dict(head,strict=True)
 assert all(torch.equal(model.model.state_dict()[n],v) for n,v in pkg['state_dict'].items())
 save_tensor(folder/'head_initial.pt',tensor_copy(head));configure_periodic_tdl_dropout(model,a.task)
 device=torch.device('cuda');model.to(device);anchor=anchors(model);anchor_initial=tensor_copy(anchor)
 manifest=read(SPLITS/f'{a.task}.json');entry=manifest['folds'][a.fold];selected,train,val=compact_train_validation_indices(entry['train_indices'],entry['validation_indices'])
 assert selected==unit['indices']['selected'] and not set(selected)&set(entry['test_indices'])
 source,frame=open_source(DOWN,DOWN,task=a.task,dual_static_root=DOWN,selected_indices=selected,record_index_path=DOWN/f'{a.task}_offsets.npy',split_manifest=manifest,expected_task_rows=manifest['sample_count'])
 try:
  assert frame['original_row'].astype(int).tolist()==selected
  data=CleanDualDataset(source,frame['label'].to_numpy(dtype=np.float64));set_global_seed(seed)
  scaler=scale_targets(data,a.task,train_indices=train,transform_mode='standard');np.testing.assert_allclose(scaler.scaler.mean_,unit['scaler_mean'],atol=1e-12,rtol=0);np.testing.assert_allclose(scaler.scaler.scale_,unit['scaler_scale'],atol=1e-12,rtol=0)
  keys=[source.samples[i][0].hex() for i in val];assert keys==unit['validation_keys']
  tr=DataLoader(Subset(data,train),batch_size=24,shuffle=True,num_workers=0,collate_fn=dual_glt_collate,generator=torch.Generator().manual_seed(seed))
  va=DataLoader(Subset(data,val),batch_size=24,shuffle=False,num_workers=0,collate_fn=dual_glt_collate)
  contract=dict(recipe=r,scope_path=str(a.scope.resolve()),unit=k,parent_path=str(PARENT),source_metadata=scope['source_metadata'],train_indices=entry['train_indices'],validation_indices=entry['validation_indices'],validation_keys=keys,scaler_mean=unit['scaler_mean'],scaler_scale=unit['scaler_scale'],head_initial_path=r['head_initial'],historical_fp_step0_exact='未核实')
  json_write(folder/'contract.json',contract)
  class HeadAdapter(AdaptationAdapter):
   def train(self,mode=True):
    super().train(mode)
    if mode:model.eval();periodic_tdl_head(model).train()
    return self
  class Proxy:
   def step(self):pass
  history=[];groups={};best=None;cycle_best=None;cycles=[];bestrow=None
  for stage,epochs in (('head',r['head_epochs']),('joint',r['joint_epochs'])):
   wrapped=HeadAdapter(model) if stage=='head' else AdaptationAdapter(model);periodic_tdl_trainability(model,head_only=stage=='head')
   opt,groups[stage]=periodic_tdl_optimizer(model,a.task,head_only=stage=='head')
   scheduler=CosineAnnealingLR(opt,T_max=10,eta_min=3e-5) if stage=='head' else CosineAnnealingWarmRestarts(opt,T_0=r['joint_scheduler']['T_0'],T_mult=1,eta_min=1e-6)
   original_step=opt.step
   @functools.wraps(original_step)
   def counted(*args,**kwargs):
    record_budget(k,'attempted_updates');v=original_step(*args,**kwargs);record_budget(k,'completed_updates');return v
   opt.step=counted
   loss=SPLoss(model,anchor,r['alpha'] if stage=='joint' else 0.)
   for se in range(1,epochs+1):
    record_budget(k,'attempted_epochs');loss.mse=[];loss.sp=[]
    result=train_epoch(wrapped,tr,loss,opt,Proxy(),device,epoch=len(history)+1,amp_dtype='fp32',max_grad_norm=5.,fail_nonfinite=True,return_timing=True)
    scheduler.step()
    # Smoke reloads the SAME current state before its sole scheduled validation.
    if a.stage=='smoke':
     exported=dict(state_dict=tensor_copy(model.state_dict()),contract=contract,scaler_mean=unit['scaler_mean'],scaler_scale=unit['scaler_scale'])
     save_tensor(folder/'smoke_reload.pt',exported);loaded=torch.load(folder/'smoke_reload.pt',map_location='cpu',weights_only=False);finite(loaded)
     assert loaded['contract']==contract;model.load_state_dict(loaded['state_dict'],strict=True)
     assert all(torch.equal(v.detach().cpu(),loaded['state_dict'][n]) for n,v in model.state_dict().items())
    vl,vr,y,z=evaluate(wrapped,va,nn.MSELoss(),device,scaler=scaler,amp_dtype='fp32')
    row=prediction(y,z,keys,entry['validation_indices'],scaler);sc=score(row);epoch=len(history)+1
    np.savez(folder/f'validation_epoch_{epoch:03d}.npz',**row,epoch=epoch,outer_test='NOT_RUN')
    grad={}
    for name,part in [('encoder',[p for n,p in model.named_parameters() if not n.startswith('model.predictor.')]),('head',list(periodic_tdl_head(model).parameters()))]:
     g=[p.grad.detach().float().square().sum() for p in part if p.grad is not None];grad[name]=float(torch.stack(g).sum().sqrt()) if g else 0.
    finite(grad)
    if stage=='head' and grad['encoder']!=0.:raise ValueError('frozen encoder gradient')
    if stage=='joint' and not all(v>0 for v in grad.values()):raise ValueError('joint gradient path')
    displacement=sum((p.detach()-anchor[n]).square().sum() for n,p in model.named_parameters() if n in anchor)
    relative=float(displacement.sqrt()/sum(v.square().sum() for v in anchor.values()).sqrt())
    record=dict(epoch=epoch,stage=stage,stage_epoch=se,train_loss=float(result[0]),train_mse=float(torch.stack(loss.mse).mean()),anchor_term=float(torch.stack(loss.sp).mean()),validation_loss=float(vl),validation_r2=float(vr),validation_rmse=sc['rmse'],learning_rates=[g['lr'] for g in opt.param_groups],training_steps=int(result[-1]['training_steps']),gradient_summary=grad,encoder_relative_displacement=relative)
    finite(record);history.append(record);record_budget(k,'completed_epochs');print(json.dumps(record),flush=True);json_write(folder/'history.json',history)
    if a.arm=='F0' and a.stage=='development' and stage=='head':
     old=read(reference(a.task,a.fold)/'metrics.json')['history'][epoch-1]
     for field in ('train_loss','validation_loss','validation_r2','validation_rmse'):
      if not np.isclose(record[field],old[field],rtol=1e-4,atol=1e-6):raise ValueError('STOP_REPLAY_DIFFERENCE head '+field)
    payload=dict(state_dict=None,contract=contract,epoch=epoch,validation_rmse=sc['rmse'],stage=stage,stage_epoch=se,scaler_mean=unit['scaler_mean'],scaler_scale=unit['scaler_scale'])
    if best is None or sc['rmse']<best['validation_rmse']:
     best=dict(payload,state_dict=tensor_copy(model.state_dict()));bestrow={n:v.copy() for n,v in row.items()};save_tensor(folder/'best.pt',best)
    if stage=='joint' and not optimization:
     if cycle_best is None or sc['rmse']<cycle_best['payload']['validation_rmse']:cycle_best=dict(payload=dict(payload,state_dict=tensor_copy(model.state_dict())),row={n:v.copy() for n,v in row.items()})
     if se%r['joint_scheduler']['T_0']==0:
      number=se//r['joint_scheduler']['T_0'];path=folder/f'cycle_{number:02d}.pt';save_tensor(path,cycle_best['payload']);np.savez(folder/f'cycle_{number:02d}_validation.npz',**cycle_best['row'],epoch=cycle_best['payload']['epoch'],cycle=number,outer_test='NOT_RUN');cycles.append(dict(cycle=number,epoch=cycle_best['payload']['epoch'],path=str(path),predictions=str(folder/f'cycle_{number:02d}_validation.npz')));cycle_best=None
  if not all(torch.equal(v.detach().cpu(),anchor_initial[n]) and not v.requires_grad for n,v in anchor.items()):raise ValueError('anchor changed')
  expected_cycles=0 if optimization else r['joint_epochs']//r['joint_scheduler']['T_0'];assert len(cycles)==expected_cycles
  chosen=cycles[-min(5,len(cycles)):];preds=[]
  for c in chosen:
   with np.load(c['predictions'],allow_pickle=False) as z:preds.append({n:z[n] for n in ('y_true','y_pred','sample_keys','indices','scaler_mean','scaler_scale')})
  np.savez(folder/'single_validation.npz',**bestrow,epoch=best['epoch'],outer_test='NOT_RUN')
  single=score(bestrow);ens=None
  if not optimization:
   ensemble=mean_predictions(preds);np.savez(folder/'E5_validation.npz',**ensemble,component_epochs=[c['epoch'] for c in chosen],outer_test='NOT_RUN');ens=score(ensemble)
  replay=None
  if a.arm=='F0' and a.stage=='development':
   old=read(reference(a.task,a.fold)/'metrics.json')['best_validation_rmse'];replay=abs(single['rmse']-old)/old
   if replay>.005:raise ValueError('STOP_REPLAY_DIFFERENCE best')
  # Strict checkpoint reload/finite/contract and saved prediction identity; no new inference.
  metrics=dict(status='PASS',task=a.task,fold=a.fold,arm=a.arm,stage=a.stage,smoke_only=a.stage=='smoke',outer_test='NOT_RUN',contract=contract,history=history,optimizer_groups_by_stage=groups,cycles=cycles,E5_components=chosen,best_epoch=best['epoch'],results=dict(single=single,E5=ens),f0_replay_relative_difference=replay,initial_constructor_head_exact=initial_constructor_equal,anchor_immutable=True,export_reload_exact='未执行；集中到最终测试',validation_passes=len(history),extra_validation_passes=0)
  json_write(folder/'metrics.json',metrics);json_write(folder/'runtime.json',dict(status='PASS',command=sys.argv,finished_at=time.time(),outer_test='NOT_RUN',SMOKE_ONLY=a.stage=='smoke'));record_budget(k,None,status='COMPLETED')
  # Last mutation: earlier errors leave this gate and the unit cannot be consumed.
  (folder/'UNCOMMITTED').unlink()
 finally:source.close()
if __name__=='__main__':
 context={}
 try:main(context)
 except BaseException as e:
  folder=context.get('folder')
  if folder is not None:
   try:json_write(folder/'failure.json',dict(status='FAILED',error=f'{type(e).__name__}: {e}',automatic_retry=False,failed_at=time.time()))
   except Exception as error:print('failure metadata unavailable: '+str(error),file=sys.stderr,flush=True)
   if 'key' in context:
    try:account(context['key'],None,ledger=context['ledger'],total_cap=context['total_cap'],status='FAILED')
    except Exception as error:print('budget failure metadata unavailable: '+str(error),file=sys.stderr,flush=True)
  raise
