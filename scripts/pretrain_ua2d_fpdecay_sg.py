#!/usr/bin/env python3
"""One physical GPU3, one fixed5k branch; no DDP or retry/resume entrance."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',required=True,choices=('BASE','FP_DECAY'));args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3' or int(os.environ.get('WORLD_SIZE','1'))!=1:
        raise RuntimeError('requires only physical GPU3 and world_size1')
    import torch
    from src.modules.ua2d_galformer_pretrain import PretrainConfig,UA2DGalformerPretrainer,deployment_package,attention_record,assert_attention_scale
    from src.dataset.ua2d_fpdecay_sg import PreparedSingleMicrobatches,decode_batch,worker_init
    from src.training.glt_dual_runtime import require_tmux,save_checkpoint,rng_state
    from src.training.ua2d_fpdecay_sg import (OUT,PLAN,REVISION,PT_SCHEMA,PT_REFERENCE,PT_CACHE,MAP_ROOT,INITIALIZATION,
        read,write,now,seed_all,cache_references,load_pt_initialization,pt_learning_rate,pt_weights,sg_objective,
        fp_decay_multiplier,process_identity,summarize_timing)
    require_tmux();torch.set_num_threads(2)
    def interrupted(signum,frame):raise KeyboardInterrupt('authorized PT stopped by signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    torch.cuda.set_device(0);device=torch.device('cuda:0');started=time.monotonic()
    output=OUT/'pretrain'/args.arm;output.mkdir(parents=True,exist_ok=False)
    context=mp.get_context('spawn');prepared_counts=context.Array('q',4)
    completed=attempted=consumed=0;iterator=None;workers=[];process=process_identity(os.getpid());all_timings=[];window_timings=[]
    def consumption(status):
        with prepared_counts.get_lock():read_count=int(sum(prepared_counts[:]))
        state=dict(plan=PLAN,revision=REVISION,arm=args.arm,status=status,pid=os.getpid(),process=process,workers=workers,
            attempted_updates=attempted,completed_updates=completed,consumed_presentations=consumed,
            sample_presentations=read_count,prefetched_not_consumed=read_count-consumed,elapsed_seconds=time.monotonic()-started)
        write(output/'consumption.json',state);return state
    consumption('INITIALIZING')
    try:
        references=cache_references();reference=read(PT_REFERENCE);values=reference['config'];cfg=PretrainConfig(**values)
        if cfg.strategy!='mask_align' or cfg.fusion!='none' or cfg.fp_enabled is not True or cfg.attention_scale_denominator!=512:raise ValueError('fixed B_FP target/attention config mismatch')
        seed_all(42);model=UA2DGalformerPretrainer(cfg);load_pt_initialization(model,INITIALIZATION);assert_attention_scale(model);model.to(device)
        metadata=dict(plan=PLAN,revision=REVISION,arm=args.arm,config=values,attention=attention_record(cfg),started=now(),pid=os.getpid(),
            argv=sys.argv,cwd=str(ROOT),seed=42,world_size=1,physical_gpus=[3],microbatch=168,accumulation=6,global_batch=1008,
            contrast_candidates_max=168,total_updates=5000,initialization=str(INITIALIZATION),mask_head=str(INITIALIZATION.parent/'task_heads/V1.pt'),fp_seed=20261019,
            eligible_rows=910756,excluded_rows=635,source=reference['source'],index=reference['index'],fp_spec=reference['fp_spec'],
            cache_references=references,coordinate_unit=cfg.coordinate_unit,precision='BF16 model with original key FP32 loss calculations',
            objective_normalization='single GPU loss sums / all6micro target counts; no world multiplier or accumulation divisor',
            lr=dict(warmup=500,peak=2e-4,end=1e-6,total_updates=5000),
            fp_schedule=dict(base='0.1*min(1,completed_updates/1000)',update_number='t=completed_updates+1',
                multiplier='1' if args.arm=='BASE' else 't<=4000:1; otherwise0.5*(1+cos(pi*(t-4000)/1000))',decay_start=4000,decay_end=5000),
            align_schedule='unchanged original global warmup, no FP multiplier',
            engineering=dict(cpu_workers=4,prefetch_per_worker=2,ordered=True,bytes_ipc=True,pinned_h2d=True,lmdb_readahead=True,timeout_seconds=21600),outer_test='NOT_RUN')
        write(output/'config.json',metadata)
        # Both independent branches reset to the SAME initial training RNG state.
        seed_all(42);save_checkpoint(output/'initial_training_rng.pt',rng_state())
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0.,betas=(.9,.999),eps=1e-8)
        dataset=PreparedSingleMicrobatches(PT_CACHE,MAP_ROOT,cfg,prepared_counts=prepared_counts)
        loader=torch.utils.data.DataLoader(dataset,batch_size=None,num_workers=4,multiprocessing_context=context,
            persistent_workers=True,prefetch_factor=2,worker_init_fn=worker_init,generator=torch.Generator().manual_seed(42))
        iterator=iter(loader);workers=[process_identity(w.pid) for w in iterator._workers];consumption('RUNNING')
        with (output/'sampling.jsonl').open('x') as sampling,(output/'step_timing.jsonl').open('x') as timing_log:
            for u in range(5000):
                tick=time.monotonic();model.completed_updates=u;weights=pt_weights(model,args.arm,u);attempted+=1;consumption('RUNNING')
                loader_wait=decode_seconds=0.;batches=[];totals={k:torch.zeros(()) for k in weights}
                for micro in range(6):
                    io_tick=time.monotonic();payload=next(iterator);loader_wait+=time.monotonic()-io_tick
                    io_tick=time.monotonic();data,labels,trace=decode_batch(payload);decode_seconds+=time.monotonic()-io_tick
                    if trace['micro_index']!=u*6+micro or trace['start_position']!=(u*6+micro)*168:raise ValueError('ordered global position mismatch')
                    consumed+=168;sampling.write(json.dumps(dict(step=u+1,micro=micro,**trace))+'\n')
                    counts=model.target_counts(data,labels)
                    for key in totals:totals[key]+=counts[key]
                    data=data.pin_memory();labels={k:v.pin_memory() if torch.is_tensor(v) else v for k,v in labels.items()};batches.append((data,labels))
                sampling.flush();consumption('RUNNING');names=list(totals)
                packed=torch.stack([totals[k] for k in names]).to(device,non_blocking=True)
                totals={k:packed[i] for i,k in enumerate(names)};data_seconds=time.monotonic()-tick
                rate=pt_learning_rate(u)
                for group in optimizer.param_groups:group['lr']=rate
                optimizer.zero_grad(set_to_none=True);losses={k:torch.zeros((),device=device) for k in names};diagnostics=[]
                for data,labels in batches:
                    data=data.to(device,non_blocking=True);labels={k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in labels.items()}
                    with torch.autocast('cuda',dtype=torch.bfloat16):result=model(data,labels)
                    with torch.autocast('cuda',enabled=False):loss=sg_objective(result,weights,totals)
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite pretraining objective')
                    loss.backward()
                    for key in names:losses[key]+=result['sums'][key].detach().float()
                    diagnostics.append(result['diagnostics'])
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();completed+=1
                raw_loss=(torch.stack([losses[k] for k in names])/packed.clamp_min(1)).tolist()
                elapsed=time.monotonic()-started;seconds=time.monotonic()-tick;consumption('RUNNING')
                measured=dict(step=completed,step_seconds=seconds,data_seconds=data_seconds,loader_wait_seconds=loader_wait,
                    decode_seconds=decode_seconds,graphs_per_second=1008/seconds,elapsed_seconds=elapsed,
                    compute_seconds=seconds-data_seconds,projected_total_seconds=elapsed/completed*5000,
                    remaining_seconds=elapsed/completed*(5000-completed),gpu_peak_gib=torch.cuda.max_memory_allocated()/1024**3)
                timing_log.write(json.dumps(measured)+'\n');timing_log.flush();all_timings.append(measured);window_timings.append(measured)
                write(output/'timing.json',measured)
                if completed%500==0:
                    with (output/'windows.jsonl').open('a') as windows:windows.write(json.dumps(summarize_timing(window_timings))+'\n')
                    write(output/'timing_summary.json',dict(full=summarize_timing(all_timings),window=summarize_timing(window_timings),
                        data_seconds_definition='includes target counts and pinning; not pure I/O; no added CUDA synchronization'));window_timings=[]
                raw=dict(zip(names,raw_loss));print(json.dumps(dict(step=completed,arm=args.arm,raw_losses=raw,
                    weighted_losses={k:v*weights[k] for k,v in raw.items()},weights=weights,fp_multiplier=fp_decay_multiplier(args.arm,completed),
                    global_counts=dict(zip(names,packed.tolist())),lr=rate,grad_norm=float(norm),timing=measured,fp_diagnostics=diagnostics)),flush=True)
        if next(iterator,None) is not None:raise ValueError('unexpected extra microbatch')
        iterator._shutdown_workers();iterator=None;state=consumption('SAVING')
        save_checkpoint(output/'rng.pt',rng_state())
        package=deployment_package(model,5000);package.update(schema=PT_SCHEMA,plan=PLAN,revision=REVISION,arm=args.arm,eligible_rows=910756,
            execution=metadata,consumption=state,initialization_source=str(INITIALIZATION),trained_parent=False)
        save_checkpoint(output/'deploy.pt',package)
        save_checkpoint(output/'resume.pt',dict(schema=PT_SCHEMA,plan=PLAN,revision=REVISION,arm=args.arm,step=5000,model=model.state_dict(),
            optimizer=optimizer.state_dict(),completed_updates=completed,attempted_updates=attempted,schedule_position=completed,
            next_global_position=completed*1008,next_micro_index=completed*6,loader_rng=loader.generator.get_state(),rng=str(output/'rng.pt'),
            execution=metadata,consumption=state,automatic_resume=False))
        state=consumption('COMPLETE')
        write(output/'metadata.json',dict(plan=PLAN,revision=REVISION,arm=args.arm,schema=PT_SCHEMA,status='COMPLETE',step=5000,
            world_size=1,microbatch=168,global_batch=1008,deploy=str(output/'deploy.pt'),resume=str(output/'resume.pt'),
            initialization_source=str(INITIALIZATION),common_mask_head=str(INITIALIZATION.parent/'task_heads/V1.pt'),
            fp_seed=20261019,seed=42,workers=workers,consumption=state,time=now()))
        write(output/'final_timing.json',dict(full=summarize_timing(all_timings),total_wallclock_seconds=time.monotonic()-started,
            includes_initialization_checkpoint_and_worker_shutdown=True))
    except BaseException:
        if iterator is not None:iterator._shutdown_workers();iterator=None
        consumption('FAILED');write(output/'failure.json',dict(error=traceback.format_exc(),retry=False,time=now()));raise
    finally:
        if iterator is not None:iterator._shutdown_workers()

if __name__=='__main__':main()
