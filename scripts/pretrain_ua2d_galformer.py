#!/usr/bin/env python3
"""Authorized r2 two-GPU, fixed-budget UA2D objectives; no retry or TEST."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import sys
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def learning_rate(completed):
    update = completed + 1
    if update <= 500:
        return 2e-4 * update / 500
    return 1e-6 + (2e-4 - 1e-6) * .5 * (1 + math.cos(math.pi * (update - 500) / 4500))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--initialization', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--index-artifact', type=Path, required=True)
    p.add_argument('--fp-cache', type=Path)
    p.add_argument('--max-steps', type=int, default=5000)
    p.add_argument('--microbatch', type=int, default=84)
    p.add_argument('--accumulation', type=int, default=6)
    args = p.parse_args()
    if (args.max_steps,args.microbatch,args.accumulation) != (5000,84,6):
        p.error('r2 requires exactly 5000/84/6')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2,3' or int(os.environ.get('WORLD_SIZE',0)) != 2:
        raise RuntimeError('r2 requires physical GPU2,3 and world=2')
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from src.modules.ua2d_galformer_pretrain import PretrainConfig, UA2DGalformerPretrainer, deployment_package, objective
    from src.dataset.ua2d_galformer_pretrain import prepare_sample, collate_samples
    from src.training.glt_dual_runtime import (require_tmux, open_source, load_sample_index_artifact,
        IndexedFrozenDualSource, OrderedSampleStream, pretrain_source_fields, save_checkpoint, rng_state)
    from src.training.ua2d_pretrain_fp import write, FPReader, load_initialization, PLAN, REVISION, FP_SPEC
    require_tmux()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl')
    source=None;cache=None;completed=attempted=presentations=0;owns=False
    def consumption(status):
        write(args.output/f'rank{rank}_consumption.json',dict(attempted_updates=attempted,
            completed_updates=completed,sample_presentations=presentations,status=status,rank=rank))
    try:
        # All ranks report initialization errors before anyone enters DDP.
        failure=None
        try:
            if rank==0:
                args.output.mkdir(parents=True,exist_ok=False);owns=True
            values=json.loads(args.config.read_text());config=PretrainConfig(**values)
            torch.manual_seed(42)
            model=UA2DGalformerPretrainer(config)
            load_initialization(model,args.initialization)
            model.to(device)
            if json.loads(args.index_artifact.read_text()).get('schema')!='mcl-ph-next-direct-index-v1':
                raise ValueError('requires trusted direct index')
            if not (args.source_root/'direct_source.json').is_file():raise ValueError('requires direct source')
            source,_=open_source(args.source_root,args.source_root,verify_key_digest=False)
            indices=load_sample_index_artifact(args.index_artifact,'train')['indices']
            if len(indices)!=911391:raise ValueError('fixed train count changed')
            source=IndexedFrozenDualSource(source,indices)
            if config.fp_enabled:
                if args.fp_cache is None:raise ValueError('FP cache missing')
                cache=FPReader(args.fp_cache,args.source_root,args.index_artifact)
            elif args.fp_cache is not None:raise ValueError('FP-off must not read FP cache')
        except BaseException:
            failure=traceback.format_exc()
        failures=[None]*2;dist.all_gather_object(failures,failure)
        if any(failures):raise RuntimeError('collective initialization failed: '+repr(failures))
        owns=True
        metadata=dict(plan=PLAN,revision=REVISION,config=values,argv=sys.argv,cwd=str(ROOT),seed=42,
            world_size=2,microbatch=84,accumulation=6,global_batch=1008,initialization=str(args.initialization),
            index=str(args.index_artifact),source=str(args.source_root),coordinate_unit=config.coordinate_unit,
            fp_spec=FP_SPEC if config.fp_enabled else None,lr_schedule=dict(warmup=500,end_update=5000,peak=2e-4,end=1e-6),
            task_weight_schedule=('constant 1 from update1' if config.weight_warmup == 0
                                  else '0.1*min(1,completed_updates/1000)'),
            objective_policy=config.objective_policy,align_multiplier=config.alignment_multiplier,outer_test='NOT_RUN')
        if rank==0:write(args.output/'config.json',metadata)
        torch.manual_seed(42+rank);torch.cuda.manual_seed_all(42+rank)
        stream=OrderedSampleStream(len(source),42)
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0.,betas=(.9,.999),eps=1e-8)
        wrapped=DDP(model,device_ids=[local],find_unused_parameters=True)
        consumption('RUNNING')
        with (args.output/f'rank{rank}_sampling.jsonl').open('x') as sampling:
            for step in range(5000):
                model.completed_updates=step;attempted+=1;consumption('RUNNING')
                totals={k:torch.zeros((),device=device) for k in model.weights};batches=[]
                for micro in range(6):
                    records=[];rows=[];start=((step*6+micro)*2+rank)*84
                    for offset in range(84):
                        position=start+offset;i=stream.index_at(position)
                        top,tri,text,static,_,identity=pretrain_source_fields(source,i,include_identity=True)
                        row=source.source_index_at(i)
                        fp=cache.get(row,text,top) if cache else None
                        presentations+=1
                        records.append(prepare_sample(top,tri,text,config=config,seed=42,position=position,
                            static=static,identity=identity,fp_target=fp))
                        rows.append(row)
                    sampling.write(json.dumps(dict(step=step,micro=micro,start_position=start,source_rows=rows))+'\n');sampling.flush()
                    consumption('RUNNING')
                    data,labels=collate_samples(records);data=data.to(device)
                    labels={k:v.to(device) if torch.is_tensor(v) else v for k,v in labels.items()}
                    counts=model.target_counts(data,labels)
                    for key in totals:totals[key]+=counts[key]
                    batches.append((data.cpu(),{k:v.cpu() if torch.is_tensor(v) else v for k,v in labels.items()}))
                for value in totals.values():dist.all_reduce(value)
                rate=learning_rate(step)
                for group in optimizer.param_groups:group['lr']=rate
                optimizer.zero_grad(set_to_none=True)
                losses={k:torch.zeros((),device=device) for k in totals};diagnostics=[]
                for micro,(data,labels) in enumerate(batches):
                    data=data.to(device);labels={k:v.to(device) if torch.is_tensor(v) else v for k,v in labels.items()}
                    with wrapped.no_sync() if micro<5 else nullcontext():
                        with torch.autocast('cuda',dtype=torch.bfloat16):result=wrapped(data,labels)
                        with torch.autocast('cuda',enabled=False):loss=objective(result,model.weights,totals)
                        if not torch.isfinite(loss):raise FloatingPointError('nonfinite pretraining objective')
                        loss.backward()
                    for key in losses:losses[key]+=result['sums'][key].detach().float()
                    diagnostics.append(result['diagnostics'])
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();completed+=1;consumption('RUNNING')
                for value in losses.values():dist.all_reduce(value)
                if rank==0:
                    raw={k:float(v/totals[k].clamp_min(1)) for k,v in losses.items()}
                    print(json.dumps(dict(step=completed,raw_losses=raw,
                        weighted_losses={k:v*model.weights[k] for k,v in raw.items()},weights=model.weights,
                        global_counts={k:float(v) for k,v in totals.items()},lr=rate,grad_norm=float(norm),
                        fp_diagnostics_rank0=diagnostics)),flush=True)
        # Preserve every rank's RNG; these records do not authorize resume.
        save_checkpoint(args.output/f'rank{rank}_rng.pt',rng_state())
        dist.barrier()
        if rank==0:
            package=deployment_package(model,completed);package['execution']=metadata
            package['consumption']=dict(attempted_updates=attempted,completed_updates=completed)
            save_checkpoint(args.output/'deploy.pt',package)
            save_checkpoint(args.output/'resume.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                completed_updates=completed,attempted_updates=attempted,schedule_position=completed,
                next_global_position=completed*1008,rank_rng=[str(args.output/f'rank{r}_rng.pt') for r in range(2)],
                execution=metadata,automatic_resume=False))
        dist.barrier();consumption('COMPLETE')
    except BaseException as error:
        if owns:
            consumption('FAILED')
            write(args.output/f'rank{rank}_failure.json',dict(error=repr(error),traceback=traceback.format_exc(),retry=False))
        raise
    finally:
        if source:source.close()
        if cache:cache.close()
        dist.destroy_process_group()

if __name__=='__main__':main()
