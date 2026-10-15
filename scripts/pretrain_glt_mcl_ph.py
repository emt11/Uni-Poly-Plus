"""Bounded four-rank GP pretraining; one writer per package, full resume state."""
import argparse,json,os,sys,time
from datetime import timedelta
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training.gp_sota_contract import *
from src.training.glt_dual_runtime import (require_tmux,open_source,IndexedFrozenDualSource,load_sample_index_artifact,OrderedSampleStream,
 pretrain_source_fields,pretrain_loader,rng_state,restore_rng,scheduled_lr,move_labels,save_checkpoint)
from src.training.mcl_ph_direct_contract import direct_binding,assert_actual_equal
from src.dataset.glt_mcl_ph import prepare_gp_sample,gp_collate
from src.dataset.mcl_ph_view import load_geometric_statistics,validate_direct_statistics_source
from scripts.pretrain_mcl_ph import open_trajectory_cache
from src.modules.glt_mcl_ph_pretrain import GPPretrainer,deployment_package
from src.modules.glt_dual_pretrain import global_objective,alignment_local_anchor_count,alignment_loss
from scripts.run_mcl_ph_r4_fast import atomic_json,timestamp
from src.utils import set_global_seed
from src.training.gp_checkpoint_save import synchronized_save,CoordinatedCheckpointExit

class Stream:
 def __init__(self,source,cache,start,stop,rank):self.source,self.cache,self.start,self.stop,self.rank=source,cache,start,stop,rank;self.order=OrderedSampleStream(len(source),42)
 def __len__(self):return (self.stop-self.start)*3
 def __getitem__(self,item):
  step=self.start+item//3;offset=item%3;rows=[]
  for local in range(84):
   p=step*1008+offset*4*84+self.rank*84+local;i=self.order.index_at(p)
   top,trimer,smiles,static,target,identity=pretrain_source_fields(self.source,i,include_target=False,include_identity=True)
   rows.append(prepare_gp_sample(top,trimer,smiles,seed=42,key=self.source.samples[i][0].hex(),position=p,
     static=static,target=target,identity=identity,trajectory_override=self.cache[p]))
  return gp_collate(rows)

def main():
 p=argparse.ArgumentParser();p.add_argument('--route',choices=('GP','GU','GP_A'),required=True);p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--stop',type=int,default=5000);p.add_argument('--resume');p.add_argument('--workers',type=int,default=8)
 args=p.parse_args();require_tmux()
 if not args.smoke and read(OUT/'p1_acceptance.json')['status']!='PASS':raise ValueError('P1 has not released formal pretraining')
 assert 0<args.stop<=(4 if args.smoke else 5000)
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK']);assert world==4
 torch.cuda.set_device(local);device=torch.device('cuda',local);dist.init_process_group('nccl',timeout=timedelta(seconds=300));torch.set_num_threads(1)
 folder=Path(args.output).resolve();assert folder.is_relative_to(OUT);folder.mkdir(parents=True,exist_ok=True)
 if rank==0:
  marker=folder/'writer.claim';marker.touch(exist_ok=False)
  assert not list(folder.glob('resume_*.pt')) and not list(folder.glob('deploy_*.pt'))
  prior=[]
  for b in OUT.glob('**/budget.json'):
   value=read(b)
   if 'smoke' in value and b.parent!=folder:prior.append((b,value))
  used=sum(v['attempted_updates'] for b,v in prior if v['smoke']==args.smoke)
  route_used=sum(v['attempted_updates'] for b,v in prior if v['smoke']==args.smoke and args.route in b.parts)
  requested=args.stop-(torch.load(args.resume,map_location='cpu',weights_only=False)['step'] if args.resume else 0)
  assert used+requested<=(8 if args.smoke else 15000) and route_used+requested<=(4 if args.smoke else 5000), 'GP pretrain budget exhausted'
 cfg=pretrain_recipe(args.route);idx=load_sample_index_artifact(INDEX,'train');base,_=open_source(PI,PI,dual_static_root=PI,readonly_readahead=True);source=IndexedFrozenDualSource(base,idx['indices'])
 statistics=load_geometric_statistics(STAT);validate_direct_statistics_source(STAT,statistics,source,seed=42,sigma=.03)
 cache,cache_contract=open_trajectory_cache(CACHE,config=dict(seed=42,noise_sigma=.03,atom_mask_ratio=.3),batch_size=1008,sample_index=idx,source=source)
 binding=direct_binding(source,common_init=COMMON,seed=42,total_positions=5040000)
 contract=dict(recipe=cfg,cache=cache_contract,binding=binding,smoke=args.smoke)
 set_global_seed(42);model=GPPretrainer(route=args.route,common_path=COMMON).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0)
 module=DDP(model,device_ids=[local],find_unused_parameters=True);start=0;saved_rng=None
 if args.resume:
  state=torch.load(args.resume,map_location='cpu',weights_only=False);assert_actual_equal(state['contract'],contract,'resume.contract')
  model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);start=state['step'];assert state['next_position']==start*1008;saved_rng=state['rng'][rank]
  assert_actual_equal({k:v.detach().cpu() for k,v in model.state_dict().items()},state['model'],'resume.model')
  assert_actual_equal(optimizer.state_dict(),state['optimizer'],'resume.optimizer')
 assert start<args.stop
 def save(step,stop=None):
  states=[None]*world;dist.all_gather_object(states,rng_state())
  def write_resume(path):
   state=dict(schema='gp-sota-resume-v2',step=step,next_position=step*1008,model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),rng=states,
     scheduler=dict(step=step,lr=optimizer.param_groups[0]['lr']),contract=contract,router_mode=model.encoder.router.mode)
   save_checkpoint(path,state)
  synchronized_save(folder,step,write_resume,lambda path:save_checkpoint(path,deployment_package(model,step)),stop=stop)
 if not args.resume:
  set_global_seed(42+rank);save(0)
 dataset=Stream(source,cache,start,args.stop,rank);iterator=iter(pretrain_loader(dataset,workers=args.workers,seed=42,rank=rank))
 if saved_rng:restore_rng(saved_rng);assert_actual_equal(rng_state(),saved_rng,'resume.rng')
 else:set_global_seed(42+rank)
 if rank==0:
  atomic_json(folder/'runtime.json',dict(status='RUNNING',started_at=timestamp(),recipe=cfg,command=sys.argv,workers_per_rank=args.workers,start_step=start,stop_step=args.stop))
 # Synthetic distributed algebra covers repeated identities and a rank with no valid anchors, without extra model calls.
 if args.smoke:
  with torch.random.fork_rng(devices=[local]):
   torch.manual_seed(100+rank);a=torch.randn(2,128,device=device,requires_grad=True);b=torch.randn(2,128,device=device,requires_grad=True)
   ids=['duplicate_identity','other_identity_'+str(rank)];mask=torch.tensor([True,True] if rank else [False,False],device=device)
   loss,count=alignment_loss(torch.nn.functional.normalize(a,dim=-1),torch.nn.functional.normalize(b,dim=-1),ids,mask)
   loss.backward();assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
 windows=[];zero_windows=np.zeros(3,dtype=np.int64)
 for step in range(start,args.stop):
  began=time.perf_counter();model.encoder.router.configure(model.encoder.router.routing_mode_for_step(step+1),step+1)
  batches=[next(iterator) for _ in range(3)]
  counts=torch.zeros(4,device=device)
  for data,labels in batches:
   n=data.graph_available.numel();masked=torch.bincount(data.canonical_graph_index[labels['atom_mask']],minlength=n);centers=torch.bincount(data.bond_batch[data.bond_center],minlength=n)
   align_count=alignment_local_anchor_count(labels['align_identity'],labels['align_valid'],device=device) if args.route=='GP_A' else torch.tensor(0.,device=device)
   counts+=torch.stack([(masked>0).sum().to(device),((centers>0)&data.geometry_valid).sum().to(device),torch.tensor(0.,device=device),align_count])
  dist.all_reduce(counts);optimizer.zero_grad(set_to_none=True);totals=torch.zeros(4,device=device);balance_value=0.;usage=torch.zeros(3,device=device);valids=torch.zeros(1,device=device)
  lr=scheduled_lr(step,lr=2e-4,warmup_steps=2000,schedule_total_steps=20000,end_lr=1e-9)
  for g in optimizer.param_groups:g['lr']=lr
  for offset,(data,labels) in enumerate(batches):
   with (module.no_sync() if offset<2 else nullcontext()):
    data=data.to(device);labels=move_labels(labels,device)
    with torch.autocast('cuda',dtype=torch.bfloat16):
     result=module(data,labels)
     loss=global_objective(result['sums'],counts[:3],world,cfg['loss_weights'])+cfg['balance_weight']*world*result['balance']/3
     if args.route=='GP_A':loss=loss+cfg['alignment_weight']*world*result['align_sum']/counts[3].clamp_min(1)
    if not torch.isfinite(loss):raise FloatingPointError('GP loss NaN/Inf')
    loss.backward()
   totals[:3]+=result['sums'].detach();totals[3]+=result['align_sum'].detach();balance_value+=float(result['balance'].detach())/3
   flag=result['router_valid'];usage+=((result['router_weights']>0)*flag.unsqueeze(-1)).sum(0);valids+=flag.sum()
  diagnostic={}
  if (step+1)%100==0:
   for name,part in dict(o8=model.encoder.o8,galformer=model.encoder.glt,experts=model.encoder.experts,router=model.encoder.router,adaptor=model.encoder.adaptor).items():
    values=[q.grad.detach().float().square().sum() for q in part.parameters() if q.grad is not None]
    diagnostic[name]=float(torch.stack(values).sum().sqrt()) if values else 0.
    if not np.isfinite(diagnostic[name]):raise FloatingPointError('module gradient NaN/Inf: '+name)
  grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
  # One rank-zero accounting event precedes each physical update; failures are never erased.
  if rank==0:atomic_json(folder/'budget.json',dict(attempted_updates=step-start+1,completed_updates=step-start,total_step=step,smoke=args.smoke,status='UPDATING'))
  optimizer.step()
  dist.all_reduce(totals);dist.all_reduce(usage);dist.all_reduce(valids)
  rate=(usage/valids.clamp_min(1)).cpu().tolist();windows.append(rate)
  row=dict(step=step+1,rank=rank,losses=(totals/counts.clamp_min(1)).tolist(),global_counts=counts.tolist(),balance=balance_value,grad_norm=grad,router_mode=model.encoder.router.mode,selection_rates=rate,lr=lr,seconds=time.perf_counter()-began,
     peak_cuda_bytes=torch.cuda.max_memory_allocated(),module_gradient_norms=diagnostic)
  with (folder/f'rank{rank}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if rank==0:
   print(json.dumps(row),flush=True);atomic_json(folder/'budget.json',dict(attempted_updates=step-start+1,completed_updates=step-start+1,total_step=step+1,smoke=args.smoke,status='RUNNING'))
  if (step+1)%100==0:
   rates=np.mean(windows[-100:],axis=0);low=(rates<.01)&(model.encoder.router.mode=='top2')
   zero_windows=np.where(low,zero_windows+1,0)
   if (zero_windows>=3).any():
    save(step+1,stop=dict(step=step+1,window_rates=rates.tolist(),consecutive_low_windows=zero_windows.tolist(),threshold=.01))
  if (step+1)%1000==0 or step+1==args.stop:save(step+1)
 if rank==0:atomic_json(folder/'runtime.json',dict(status='PASS',finished_at=timestamp(),recipe=cfg,command=sys.argv,start_step=start,executed_updates=args.stop-start,step=args.stop,world=4,workers_per_rank=args.workers,synthetic_alignment_collectives_checked=args.smoke))
 dist.barrier();cache.close();source.close();dist.destroy_process_group()
if __name__=='__main__':
 try:main()
 except BaseException as error:
  if int(os.environ.get('RANK','0'))==0 and not isinstance(error,CoordinatedCheckpointExit):
   path=Path(sys.argv[sys.argv.index('--output')+1]);path.mkdir(parents=True,exist_ok=True)
   atomic_json(path/'failure.json',dict(status='FAILED',error=f'{type(error).__name__}: {error}',automatic_retry=False,failed_at=timestamp()))
  raise
