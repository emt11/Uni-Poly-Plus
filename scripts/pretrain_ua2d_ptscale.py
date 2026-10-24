#!/usr/bin/env python3
"""Continuous 20000-update PTSCALE PT with cached templates and ordered CPU prefetch."""
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
    for name in ('config','output','initialization','prepared-cache','mapping'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3' or int(os.environ.get('WORLD_SIZE',0))!=2
            or os.environ.get('UA2D_PTSCALE_CAMPAIGN')!='r1'):
        raise RuntimeError('requires PTSCALE r1 and physical GPU2,3/world2')
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from src.modules.ua2d_galformer_pretrain import PretrainConfig,UA2DGalformerPretrainer,deployment_package,objective,attention_record,assert_attention_scale
    from src.dataset.ua2d_ptscale import PreparedScaleMicrobatches,decode_batch,worker_init,CACHE_SCHEMA
    from src.training.glt_dual_runtime import require_tmux,save_checkpoint,rng_state
    from src.training.ua2d_ptscale import write,read,load_pt_initialization,PLAN,REVISION,PT_SCHEMA,PT_REFERENCE,pt_learning_rate,summarize_timing,process_identity
    require_tmux()
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    started=time.monotonic()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl')
    context=mp.get_context('spawn')
    prepared_counts=context.Array('q',4)
    completed=attempted=consumed=0;owns=False;iterator=None;workers=[];all_timings=[];window_timings=[]
    rank_identity=process_identity(os.getpid())
    def consumption(status):
        with prepared_counts.get_lock():presentations=sum(prepared_counts[:])
        write(args.output/f'rank{rank}_consumption.json',dict(attempted_updates=attempted,
            completed_updates=completed,sample_presentations=int(presentations),
            consumed_presentations=consumed,prefetched_not_consumed=int(presentations)-consumed,
            status=status,rank=rank,pid=os.getpid(),process=rank_identity,workers=workers,plan=PLAN,
            elapsed_seconds=time.monotonic()-started))
    try:
        failure=None
        try:
            if rank==0:args.output.mkdir(parents=True,exist_ok=False);owns=True
            reference=read(PT_REFERENCE);FP_SPEC=reference['fp_spec'];SOURCE=reference['source'];INDEX=reference['index']
            values=read(args.config)
            if values!=reference['config']:raise ValueError('fixed original B_FP objectives differ')
            config=PretrainConfig(**values)
            cache=read(args.prepared_cache/'metadata.json')
            if (cache.get('schema')!=CACHE_SCHEMA or cache.get('status')!='COMPLETE'
                    or cache.get('source')!=SOURCE or cache.get('index')!=INDEX
                    or cache.get('spec')!=FP_SPEC or cache.get('train_rows')!=911391):
                raise ValueError('prepared source/FP/schema mismatch')
            torch.manual_seed(42);model=UA2DGalformerPretrainer(config)
            load_pt_initialization(model,args.initialization);assert_attention_scale(model);model.to(device)
        except BaseException:failure=traceback.format_exc()
        failures=[None]*2;dist.all_gather_object(failures,failure)
        if any(failures):raise RuntimeError('collective initialization failed: '+repr(failures))
        owns=True
        metadata=dict(plan=PLAN,revision=REVISION,config=values,attention=attention_record(config),argv=sys.argv,cwd=str(ROOT),seed=42,
            started=time.time(),world_size=2,microbatch=84,accumulation=6,global_batch=1008,contrast_candidates_max=168,
            initialization=str(args.initialization),source=SOURCE,index=INDEX,
            prepared_cache=str(args.prepared_cache),cache_schema=CACHE_SCHEMA,eligible_mapping=str(args.mapping),eligible_rows=910756,trajectory_updates=20000,
            mapping_semantics='eligible is logical slot, not source_row',
            preprocessing_seconds=cache['preprocessing_seconds'],coordinate_unit=config.coordinate_unit,
            fp_spec=FP_SPEC if config.fp_enabled else None,lr_schedule=dict(warmup=500,cycle_updates=5000,total_updates=20000,position='completed_updates % 5000',peak=2e-4,end=1e-6,reset_optimizer=False),
            task_weight_schedule='0.1*min(1,completed_updates/1000)',outer_test='NOT_RUN',
            engineering=dict(cpu_workers_per_rank=4,prefetch_per_worker=2,ordered=True,bytes_ipc=True,
                             cpu_target_counts=True,pinned_h2d=True,lmdb_readahead=True,deadline_seconds=43200))
        if rank==0:write(args.output/'config.json',metadata)
        torch.manual_seed(42+rank);torch.cuda.manual_seed_all(42+rank)
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0.,betas=(.9,.999),eps=1e-8)
        wrapped=DDP(model,device_ids=[local],find_unused_parameters=True)
        dataset=PreparedScaleMicrobatches(args.prepared_cache,args.mapping,config,rank=rank,prepared_counts=prepared_counts)
        loader=torch.utils.data.DataLoader(dataset,batch_size=None,num_workers=4,
            multiprocessing_context=context,persistent_workers=True,prefetch_factor=2,
            worker_init_fn=worker_init,generator=torch.Generator().manual_seed(42))
        iterator=iter(loader)
        workers=[process_identity(w.pid) for w in iterator._workers]
        consumption('RUNNING')
        def snapshot(step):
            checkpoint_tick=time.monotonic();folder=args.output/'checkpoints'/f'step{step}'
            if rank==0:folder.mkdir(parents=True,exist_ok=False)
            dist.barrier()
            save_checkpoint(folder/f'rank{rank}_rng.pt',rng_state())
            with prepared_counts.get_lock():read_count=int(sum(prepared_counts[:]))
            write(folder/f'rank{rank}_counts.json',dict(rank=rank,completed_updates=completed,attempted_updates=attempted,
                consumed_presentations=consumed,read_presentations=read_count,next_micro_index=completed*6,
                next_global_position=completed*1008,rank_process=rank_identity,workers=workers))
            dist.barrier()
            if rank==0:
                package=deployment_package(model,step)
                package.update(schema=PT_SCHEMA,plan=PLAN,revision=REVISION,eligible_rows=910756,
                    execution=metadata,trajectory_id=PLAN+'/r1/continuous-step0-20000',
                    source_checkpoint=dict(step=step,initialization=str(args.initialization),mask_head=str(args.initialization.parent/'task_heads/V1.pt')))
                save_checkpoint(folder/'deploy.pt',package)
                save_checkpoint(folder/'resume.pt',dict(schema=PT_SCHEMA,plan=PLAN,revision=REVISION,step=step,
                    model=model.state_dict(),optimizer=optimizer.state_dict(),completed_updates=completed,attempted_updates=attempted,
                    schedule_position=completed,schedule_cycle_position=completed%5000,global_weight_warmup_position=completed,
                    next_global_position=completed*1008,next_micro_index=completed*6,
                    loader_rng=loader.generator.get_state(),rank_rng=[str(folder/f'rank{r}_rng.pt') for r in range(2)],
                    rank_counts=[read(folder/f'rank{r}_counts.json') for r in range(2)],execution=metadata,automatic_resume=False))
                write(folder/'metadata.json',dict(plan=PLAN,revision=REVISION,schema=PT_SCHEMA,status='COMPLETE',step=step,
                    deploy=str(folder/'deploy.pt'),resume=str(folder/'resume.pt'),source=str(args.initialization),
                    trajectory_id=package['trajectory_id'],snapshot_seconds_rank0=time.monotonic()-checkpoint_tick,time=metadata['started']))
            dist.barrier()
        with (args.output/f'rank{rank}_sampling.jsonl').open('x') as sampling, (args.output/f'rank{rank}_timing.jsonl').open('x') as timing_log:
            for step in range(20000):
                tick=time.monotonic();model.completed_updates=step;attempted+=1
                consumption('RUNNING')
                loader_wait=decode_seconds=0.
                batches=[];totals={k:torch.zeros(()) for k in model.weights}
                for micro in range(6):
                    io_tick=time.monotonic();payload=next(iterator);loader_wait+=time.monotonic()-io_tick
                    io_tick=time.monotonic();data,labels,trace=decode_batch(payload);decode_seconds+=time.monotonic()-io_tick
                    if trace['micro_index']!=step*6+micro:raise ValueError('prefetch changed microbatch order')
                    consumed+=84
                    sampling.write(json.dumps(dict(step=step,micro=micro,**trace))+'\n')
                    counts=model.target_counts(data,labels)
                    for key in totals:totals[key]+=counts[key]
                    # Pin once and transfer only immediately before the forward.
                    data=data.pin_memory()
                    labels={k:v.pin_memory() if torch.is_tensor(v) else v for k,v in labels.items()}
                    batches.append((data,labels))
                sampling.flush();consumption('RUNNING')
                names=list(totals)
                packed=torch.stack([totals[k] for k in names]).to(device,non_blocking=True)
                dist.all_reduce(packed)
                totals={k:packed[i] for i,k in enumerate(names)}
                data_seconds=time.monotonic()-tick
                rate=pt_learning_rate(step)
                for group in optimizer.param_groups:group['lr']=rate
                optimizer.zero_grad(set_to_none=True)
                losses={k:torch.zeros((),device=device) for k in names};diagnostics=[]
                for micro,(data,labels) in enumerate(batches):
                    data=data.to(device,non_blocking=True)
                    labels={k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in labels.items()}
                    with wrapped.no_sync() if micro<5 else nullcontext():
                        with torch.autocast('cuda',dtype=torch.bfloat16):result=wrapped(data,labels)
                        with torch.autocast('cuda',enabled=False):loss=objective(result,model.weights,totals)
                        if not torch.isfinite(loss):raise FloatingPointError('nonfinite pretraining objective')
                        loss.backward()
                    for key in names:losses[key]+=result['sums'][key].detach().float()
                    diagnostics.append(result['diagnostics'])
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();completed+=1
                values_loss=torch.stack([losses[k] for k in names]);dist.all_reduce(values_loss)
                raw_loss=(values_loss/packed.clamp_min(1)).tolist()
                elapsed=time.monotonic()-started;seconds=time.monotonic()-tick
                consumption('RUNNING')
                measured=dict(step=completed,step_seconds=seconds,loader_wait_seconds=loader_wait,decode_seconds=decode_seconds,
                    data_seconds=data_seconds,graphs_per_second=1008/seconds,elapsed_seconds=elapsed)
                timing_log.write(json.dumps(measured)+'\n');timing_log.flush()
                all_timings.append(measured);window_timings.append(measured)
                if completed%500==0:
                    write(args.output/f'rank{rank}_timing_summary.json',dict(full=summarize_timing(all_timings),
                        window=summarize_timing(window_timings),elapsed_seconds=elapsed,
                        definition='data_seconds includes counts, pinning and denominator communication, not pure I/O; no extra CUDA sync'))
                    with (args.output/f'rank{rank}_windows.jsonl').open('a') as windows:windows.write(json.dumps(summarize_timing(window_timings))+'\n')
                    window_timings=[]
                if rank==0:
                    raw=dict(zip(names,raw_loss))
                    timing=dict(step_seconds=seconds,data_seconds=data_seconds,loader_wait_seconds=loader_wait,decode_seconds=decode_seconds,graphs_per_second=1008/seconds,
                        compute_and_communication_seconds=seconds-data_seconds,elapsed_seconds=elapsed,
                        projected_total_seconds=elapsed/completed*20000,
                        remaining_seconds=elapsed/completed*(20000-completed),deadline_seconds=43200,
                        gpu_peak_gib=torch.cuda.max_memory_allocated()/1024**3)
                    write(args.output/'timing.json',dict(step=completed,**timing))
                    print(json.dumps(dict(step=completed,raw_losses=raw,
                        weighted_losses={k:v*model.weights[k] for k,v in raw.items()},weights=model.weights,
                        global_counts=dict(zip(names,packed.tolist())),lr=rate,grad_norm=float(norm),
                        timing=timing,fp_diagnostics_rank0=diagnostics)),flush=True)
                if completed in (5000,10000,20000):snapshot(completed)
        # Exhaust the finite iterator: no extra records or sample presentations.
        if next(iterator,None) is not None:raise ValueError('extra prepared microbatch')
        iterator._shutdown_workers();iterator=None
        write(args.output/f'rank{rank}_final_timing.json',dict(full=summarize_timing(all_timings),total_wallclock_seconds=time.monotonic()-started,includes_snapshot_and_worker_shutdown=True))
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
