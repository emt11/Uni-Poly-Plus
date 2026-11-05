#!/usr/bin/env python3
"""One authorized native three-modal PT, allocated GPU/world1/10K; no resume entry."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(key,'2')
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import signal
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.training import ua3d_base_opt_r6 as t
from src.modules import ua3d_base_opt_r6 as m
from src.dataset import ua3m_paperfuse as d
from src.dataset.ua2d_fast_pretrain import decode_batch
from src.training.glt_dual_runtime import rng_state


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--physical-gpu',type=int,choices=(2,),default=2);p.add_argument('--execution-root');args=p.parse_args()
    if args.execution_root is not None and Path(args.execution_root).resolve()!=t.OUT:raise ValueError('unauthorized root')
    t.require_tmux();torch.set_num_threads(2)
    if args.physical_gpu not in t.ALLOWED_GPUS or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.physical_gpu) or os.environ.get('WORLD_SIZE','1')!='1':raise ValueError('allocated physical GPU/world1 mismatch')
    def stop(sig,frame):t.ignore_termination();raise RuntimeError('controlled PT termination '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    folder=t.OUT/'pretrain'/args.arm;folder.mkdir(parents=True,exist_ok=False)
    context=mp.get_context('spawn');prepared_counts=context.Array('q',4);iterator=None;model=None;workers=[];attempted=completed=consumed=0;exposure={};pair_totals={};weighted_align=0.;started=time.monotonic()
    def record(status):
        with prepared_counts.get_lock():read_count=int(sum(prepared_counts[:]))
        value=dict(status=status,plan=t.PLAN,revision=t.REVISION,arm=args.arm,pid=os.getpid(),process=t.process_identity(os.getpid()),workers=workers,
            physical_gpu=args.physical_gpu,gpu_uuid=os.environ.get('UA3M_PHYSICAL_GPU_UUID'),allowed_gpus=list(t.ALLOWED_GPUS),resource_revision=t.EXECUTION_REVISION,pt_updates=attempted,completed_updates=completed,pt_microbatches=consumed//168,pt_presentations=consumed,
            prefetched_presentations=read_count,prefetched_not_consumed=read_count-consumed,elapsed_seconds=time.monotonic()-started,
            stream=model.stream_state() if model is not None else None,actual_effective_positive_pairs=pair_totals,weighted_positive_exposure=exposure,total_weighted_align_loss=weighted_align)
        t.write(folder/'consumption.json',value);return value
    record('RUNNING')
    try:
        t.check_prepared();actual_device=t.gpu_contract();t.seed_all(42)
        pt=t.registry()['pt'][args.arm];model=m.BaseOptFuse(True,pt=pt)
        common=torch.load(t.INITIAL,map_location='cpu',weights_only=False)
        if (common.get('plan')!=t.PLAN or common.get('revision')!=t.REVISION or common.get('step')!=0 or common.get('config')!=m.CONFIG):raise ValueError('cycle step0 mismatch')
        state=dict(common['state_dict'])
        if pt['independent']:
            extra=torch.load(t.INITIAL.parent/'QFP_step0.pt',map_location='cpu',weights_only=False)
            if extra['source']!='common_step0.Q2':raise ValueError('QFP source mismatch')
            state.update({'QFP.'+k:v for k,v in extra['state_dict'].items()})
        m.strict_module_state(model,state)
        init=dict(common_step0=str(t.INITIAL),qfp='deepcopy loaded Q2' if pt['independent'] else None)
        del common,state
        metadata=dict(plan=t.PLAN,revision=t.REVISION,schema=m.PT_SCHEMA,arm=args.arm,config=m.CONFIG,initialization=init,fp_as_input=True,fp_prediction=False,
            world_size=1,physical_gpus=[2],job_concurrency_limit=t.resource_contract('pt')['job_concurrency_limit'],actual_device=actual_device,physical_gpu=args.physical_gpu,gpu_uuid=os.environ.get('UA3M_PHYSICAL_GPU_UUID'),allowed_gpus=list(t.ALLOWED_GPUS),resource_revision=t.EXECUTION_REVISION,microbatch=168,accumulation=6,global_batch=1008,updates=10000,seed=42,
            objective='mask2D+mask3D+direction-wise TMOL decoupled',directions=pt['directions'],align_each=pt['align_each'],pt_config=pt,runtime_contract=model.runtime_contract(),parameters=sum(p.numel() for p in model.parameters()),workers=4,worker_threads=1,prefetch_factor=2,
            lr='two continuous5K cycles; warmup500;2e-4 to1e-6; global align warmup1000 once',stream=model.stream_state(),started=t.now())
        t.write(folder/'config.json',metadata);model.to('cuda:0');t.seed_all(42)
        opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0.,betas=(.9,.999),eps=1e-8)
        ds=d.PTData(t.PT_CACHE,t.MAP_ROOT,prepared_counts)
        loader=torch.utils.data.DataLoader(ds,batch_size=None,num_workers=4,prefetch_factor=2,
            multiprocessing_context=context,worker_init_fn=d.worker_init,generator=torch.Generator().manual_seed(42))
        iterator=iter(loader);workers=[t.process_identity(w.pid) for w in iterator._workers];record('RUNNING')
        with (folder/'sampling.jsonl').open('x') as sampling,(folder/'history.jsonl').open('x') as steps:
            for update in range(10000):
                tick=time.monotonic();model.completed_updates=update;attempted+=1;record('RUNNING');weights=model.weights();totals={key:torch.zeros(()) for key in weights};batches=[]
                for micro in range(6):
                    data,labels,trace=decode_batch(next(iterator))
                    if trace['micro_index']!=update*6+micro or trace['start_position']!=(update*6+micro)*168:raise ValueError('ordered PT stream mismatch')
                    consumed+=168;sampling.write(json.dumps(dict(step=update+1,**trace))+'\n')
                    batch=d.pt_batch(data,labels);counts=model.target_counts(batch,labels)
                    for key in totals:totals[key]+=counts[key]
                    data=data.pin_memory();labels={k:v.pin_memory() if torch.is_tensor(v) else v for k,v in labels.items()};batches.append((data,labels))
                sampling.flush();record('RUNNING');totals={k:v.to('cuda:0') for k,v in totals.items()}
                lr=t.pt_learning_rate(update)
                for group in opt.param_groups:group['lr']=lr
                opt.zero_grad(set_to_none=True);raw={key:torch.zeros((),device='cuda:0') for key in weights}
                for data,labels in batches:
                    data=data.to('cuda:0',non_blocking=True);labels={k:v.to('cuda:0',non_blocking=True) if torch.is_tensor(v) else v for k,v in labels.items()}
                    batch=d.pt_batch(data,labels)
                    with torch.autocast('cuda',dtype=torch.bfloat16):result=model.pretrain_result(batch,labels)
                    with torch.autocast('cuda',enabled=False):loss=t.window_objective(result,weights,totals)
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite PT loss; negative TMOL is allowed')
                    loss.backward()
                    for key in raw:raw[key]+=result['sums'][key].detach().float()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();completed+=1
                for key in weights:
                    if key.startswith('align_'):
                        pair_totals[key]=pair_totals.get(key,0)+int(totals[key])
                        exposure[key]=exposure.get(key,0.)+weights[key]*float(totals[key])
                align_contribution=sum(weights[k]*float(raw[k]/totals[k].clamp_min(1)) for k in weights if k.startswith('align_'))
                weighted_align+=align_contribution
                rec=dict(align_contribution=align_contribution,total_align_nominal_weight=sum(v for k,v in weights.items() if k.startswith('align_')),step=completed,arm=args.arm,raw_sums={k:float(v) for k,v in raw.items()},raw_losses={k:float(v/totals[k].clamp_min(1)) for k,v in raw.items()},weights=weights,
                    counts={k:float(v) for k,v in totals.items()},lr=lr,grad_norm=float(norm),seconds=time.monotonic()-tick,elapsed_seconds=time.monotonic()-started,
                    gpu_peak_bytes=torch.cuda.max_memory_allocated(),stream=model.stream_state())
                steps.write(json.dumps(rec)+'\n');steps.flush();print(json.dumps(rec),flush=True);t.write(folder/'timing.json',rec);record('RUNNING')
        if next(iterator,None) is not None:raise ValueError('extra microbatch after fixed60K')
        iterator=None
        model.completed_updates=completed
        package=dict(schema=m.PT_SCHEMA,plan=t.PLAN,revision=t.REVISION,arm=pt['arm'],pt_id=args.arm,pt_config=pt,step=10000,config=m.CONFIG,fp_as_input=True,fp_prediction=False,
            state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},stream=model.stream_state(),execution=metadata)
        t.save_checkpoint(folder/'deploy.pt',package)
        t.save_checkpoint(folder/'resume.pt',dict(**package,optimizer=opt.state_dict(),rng=rng_state(),loader_rng=loader.generator.get_state(),next_micro_index=60000,
            next_global_position=10080000,attempted_updates=attempted,completed_updates=completed,schedule_position=completed,automatic_resume=False))
        record('COMPLETE')
    except BaseException:
        t.ignore_termination();error=traceback.format_exc()
        if iterator is not None:
            iterator._shutdown_workers();iterator=None
        record('FAILED');t.write(folder/'failure.json',dict(error=error,time=t.now(),automatic_retry=False));raise
    finally:
        t.ignore_termination()
        if iterator is not None:
            iterator._shutdown_workers();iterator=None

if __name__=='__main__':main()
