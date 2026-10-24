#!/usr/bin/env python3
"""Six-arm clean/paired-view PT; fixed 5000 updates and global-window weights."""
import argparse
from contextlib import nullcontext
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def interrupted(signum, frame):
    raise KeyboardInterrupt('pretraining stopped by signal '+str(signum))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','output','initialization','prepared-cache'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3' or int(os.environ.get('WORLD_SIZE',0))!=2
            or os.environ.get('UA2D_CLEANFP_CAMPAIGN')!='r1'):
        raise RuntimeError('requires cleanfp r1 and physical GPU2,3/world2')
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from src.modules.ua2d_cleanfp_pretrain import CleanConfig,CleanPretrainer,deployment_package,objective
    from src.dataset.ua2d_cleanfp_pretrain import CleanMicrobatches,decode_clean_batch
    from src.dataset.ua2d_fast_pretrain import worker_init,CACHE_SCHEMA
    from src.training.glt_dual_runtime import require_tmux,save_checkpoint,rng_state
    from src.training.ua2d_readout_distill import write,read,load_initialization,validate_cache,LossBalancer,PLAN,REVISION,SOURCE,INDEX
    from scripts.pretrain_ua2d_galformer import learning_rate
    require_tmux()
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    started=time.monotonic()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl')
    context=mp.get_context('spawn')
    prepared_counts=context.Array('q',4)
    completed=attempted=consumed=0;owns=False;iterator=None
    def consumption(status):
        with prepared_counts.get_lock():presentations=sum(prepared_counts[:])
        write(args.output/f'rank{rank}_consumption.json',dict(attempted_updates=attempted,
            completed_updates=completed,sample_presentations=int(presentations),
            consumed_presentations=consumed,o8_view_presentations=2*consumed,galformer_view_presentations=2*consumed,prefetched_not_consumed=int(presentations)-consumed,
            status=status,rank=rank,elapsed_seconds=time.monotonic()-started))
    try:
        failure=None
        try:
            if rank==0:args.output.mkdir(parents=True,exist_ok=False);owns=True
            values=read(args.config);config=CleanConfig(**values)
            cache=validate_cache(args.prepared_cache)
            torch.manual_seed(42);model=CleanPretrainer(config)
            load_initialization(model,args.initialization);model.to(device)
        except BaseException:failure=traceback.format_exc()
        failures=[None]*2;dist.all_gather_object(failures,failure)
        if any(failures):raise RuntimeError('collective initialization failed: '+repr(failures))
        owns=True
        metadata=dict(plan=PLAN,revision=REVISION,config=values,argv=sys.argv,cwd=str(ROOT),seed=42,
            world_size=2,microbatch=84,accumulation=6,global_batch=1008,contrast_candidates_max=168,
            initialization=str(args.initialization),source=str(SOURCE),index=str(INDEX),
            prepared_cache=str(args.prepared_cache),cache_schema=CACHE_SCHEMA,
            preprocessing_seconds=cache['preprocessing_seconds'],coordinate_unit=None,views_per_source=dict(o8=2,galformer=2),
            fp_cache_spec=cache['spec'],fp_input='clean',align_input='clean',lr_schedule=dict(warmup=500,end_update=5000,peak=2e-4,end=1e-6),
            task_weight_schedule=config.weight_policy,outer_test='NOT_RUN',
            engineering=dict(cpu_workers_per_rank=4,prefetch_per_worker=2,ordered=True,bytes_ipc=True,
                             cpu_target_counts=True,pinned_h2d=True,deadline_seconds=18000))
        if rank==0:write(args.output/'config.json',metadata)
        torch.manual_seed(42+rank);torch.cuda.manual_seed_all(42+rank)
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0.,betas=(.9,.999),eps=1e-8)
        wrapped=DDP(model,device_ids=[local],find_unused_parameters=True)
        dataset=CleanMicrobatches(args.prepared_cache,config,rank=rank,prepared_counts=prepared_counts)
        loader=torch.utils.data.DataLoader(dataset,batch_size=None,num_workers=4,
            multiprocessing_context=context,persistent_workers=True,prefetch_factor=2,
            worker_init_fn=worker_init,generator=torch.Generator().manual_seed(42))
        iterator=iter(loader)
        consumption('RUNNING')
        balancer=LossBalancer(list(model.weights),config.weight_policy)
        with (args.output/f'rank{rank}_sampling.jsonl').open('x') as sampling, (args.output/f'rank{rank}_weights.jsonl').open('x') as weightlog:
            for step in range(5000):
                tick=time.monotonic();model.completed_updates=step;attempted+=1
                consumption('RUNNING')
                window_weights=dict(balancer.weights)
                batches=[];totals={k:torch.zeros(()) for k in model.weights}
                for micro in range(6):
                    data,labels,clean,trace=decode_clean_batch(next(iterator))
                    if trace['micro_index']!=step*6+micro:raise ValueError('prefetch changed microbatch order')
                    consumed+=84
                    sampling.write(json.dumps(dict(step=step,micro=micro,**trace))+'\n')
                    counts=model.target_counts(data,labels)
                    for key in totals:totals[key]+=counts[key]
                    # Pin once and transfer only immediately before the forward.
                    data=data.pin_memory();clean=clean.pin_memory()
                    labels={k:v.pin_memory() if torch.is_tensor(v) else v for k,v in labels.items()}
                    batches.append((data,labels,clean))
                sampling.flush();consumption('RUNNING')
                names=list(totals)
                packed=torch.stack([totals[k] for k in names]).to(device,non_blocking=True)
                dist.all_reduce(packed)
                totals={k:packed[i] for i,k in enumerate(names)}
                data_seconds=time.monotonic()-tick
                rate=learning_rate(step)
                for group in optimizer.param_groups:group['lr']=rate
                optimizer.zero_grad(set_to_none=True)
                losses={k:torch.zeros((),device=device) for k in names};diagnostics=[]
                for micro,(data,labels,clean) in enumerate(batches):
                    data=data.to(device,non_blocking=True);clean=clean.to(device,non_blocking=True)
                    labels={k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in labels.items()}
                    with wrapped.no_sync() if micro<5 else nullcontext():
                        with torch.autocast('cuda',dtype=torch.bfloat16):result=wrapped(data,labels,clean)
                        with torch.autocast('cuda',enabled=False):loss=objective(result,window_weights,totals)
                        if not torch.isfinite(loss):raise FloatingPointError('nonfinite pretraining objective')
                        loss.backward()
                    for key in names:losses[key]+=result['sums'][key].detach().float()
                    diagnostics.append(result['diagnostics'])
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();completed+=1
                values_loss=torch.stack([losses[k] for k in names]);dist.all_reduce(values_loss)
                raw_loss=(values_loss/packed.clamp_min(1)).tolist()
                raw=dict(zip(names,raw_loss));global_counts=dict(zip(names,packed.tolist()))
                if rank==0:balancer.observe(raw,global_counts)
                next_weights=torch.tensor([balancer.weights[k] for k in names],dtype=torch.float64,device=device)
                dist.broadcast(next_weights,src=0)
                balancer.weights=dict(zip(names,next_weights.tolist()))
                weightlog.write(json.dumps(dict(step=completed,applied=window_weights,next=balancer.weights,raw=raw,counts=global_counts))+'\n');weightlog.flush()
                elapsed=time.monotonic()-started;seconds=time.monotonic()-tick
                consumption('RUNNING')
                if rank==0:
                    raw=dict(zip(names,raw_loss))
                    timing=dict(step_seconds=seconds,data_seconds=data_seconds,
                        compute_and_communication_seconds=seconds-data_seconds,elapsed_seconds=elapsed,
                        projected_total_seconds=elapsed/completed*5000,
                        remaining_seconds=elapsed/completed*(5000-completed),deadline_seconds=18000,
                        gpu_peak_gib=torch.cuda.max_memory_allocated()/1024**3)
                    write(args.output/'timing.json',dict(step=completed,**timing))
                    print(json.dumps(dict(step=completed,raw_losses=raw,
                        weighted_losses={k:v*window_weights[k] for k,v in raw.items()},weights=window_weights,next_weights=balancer.weights,
                        global_counts=dict(zip(names,packed.tolist())),lr=rate,grad_norm=float(norm),
                        timing=timing,fp_diagnostics_rank0=diagnostics)),flush=True)
        # Exhaust the finite iterator: no extra records or sample presentations.
        if next(iterator,None) is not None:raise ValueError('extra prepared microbatch')
        iterator._shutdown_workers();iterator=None
        consumption('SAVING')
        save_checkpoint(args.output/f'rank{rank}_rng.pt',rng_state());dist.barrier()
        if rank==0:
            package=deployment_package(model,completed);package['execution']=metadata
            package['consumption']=dict(attempted_updates=attempted,completed_updates=completed)
            save_checkpoint(args.output/'deploy.pt',package)
            save_checkpoint(args.output/'resume.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                completed_updates=completed,attempted_updates=attempted,schedule_position=completed,
                next_global_position=completed*1008,rank_rng=[str(args.output/f'rank{r}_rng.pt') for r in range(2)],
                execution=metadata,balancer=balancer.state_dict(),automatic_resume=False))
        dist.barrier();consumption('COMPLETE')
    except BaseException as error:
        if iterator is not None:iterator._shutdown_workers();iterator=None
        if owns:
            consumption('FAILED')
            write(args.output/f'rank{rank}_failure.json',dict(error=repr(error),traceback=traceback.format_exc(),retry=False))
        raise
    finally:
        if iterator is not None:iterator._shutdown_workers()
        dist.destroy_process_group()


if __name__=='__main__':main()
