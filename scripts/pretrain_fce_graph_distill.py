"""r15 two-rank full-state G05 continuation, one student pass per microbatch."""
import argparse
from contextlib import nullcontext
import datetime
import os
from pathlib import Path
import signal
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from src.training.fce_graph_distill import *
from src.modules.glt_dual_pretrain import DualPretrainer,global_objective,deployment_package,restore_deployment
from src.modules.glt_dual import locality_options
from src.training.glt_dual_runtime import (require_tmux,IndexedFrozenDualSource,pretrain_loader,
    save_checkpoint,move_labels)
from src.utils import set_global_seed


def build_student(cfg):
    keys=('geometry_head_norm','third_task','opt_mode','fusion_revision','use_md200','atom_ce_input',
          'align_enabled','local_enabled','geometry_revision','geometry_mode','angle_value_enabled','readout')
    return DualPretrainer(cfg['fusion_mode'],torsion=cfg['torsion_modules'],
        **{k:cfg[k] for k in keys},**locality_options(cfg))


def main(arm):
    require_tmux();cfg=validate_pt_config(read(OUT/'pt_config'/f'{arm}.json'))
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3':raise ValueError('PT physical GPU2/3 required')
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=2 or rank not in (0,1) or local!=rank:raise ValueError('exactly two original ranks required')
    torch.cuda.set_device(local);device=torch.device('cuda',local);dist.init_process_group('nccl')
    folder=OUT/'pretrain'/arm;source=None;attempted=completed=0;student_presented=teacher_presented=0
    started=datetime.datetime.now(datetime.timezone.utc).isoformat();rc=1
    def interrupted(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        # Exclusive shared writer and synchronization before rank-specific files.
        if rank==0:folder.mkdir(parents=True,exist_ok=False)
        dist.barrier()
        source,_=open_source(PI,PI,dual_static_root=PI,readonly_readahead=True,verify_key_digest=False)
        if read(cfg['sample_index_artifact']).get('schema')!='mcl-ph-next-direct-index-v1':raise ValueError('direct index only; no legacy digest branch')
        artifact=load_sample_index_artifact(cfg['sample_index_artifact'],'train')
        if len(source)!=959588 or len(artifact['indices'])!=911391:raise ValueError('original PI1M counts mismatch')
        source=IndexedFrozenDualSource(source,artifact['indices'])
        state=torch.load(RESUME,map_location='cpu',weights_only=False)
        set_global_seed(cfg['seed']);base=build_student(cfg).to(device)
        if len(base.state_dict())!=212 or sum(p.numel() for p in base.parameters())!=40846686:raise ValueError('original student schema mismatch')
        optimizer=torch.optim.AdamW(base.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        actual_source=dict(world_size=world,sample_count=len(source),cohort_root=str(PI),cache_root=str(PI),
            sample_index_split='train')
        rank_rng,ancestry=restore_branch(base,optimizer,state,cfg,rank=rank,world=world,
            ordered_keys=[key.hex() for key,_ in source.samples],source_identity=actual_source)
        if rank_rng['cuda'] is None or len(rank_rng['cuda'])!=2:raise ValueError('original two-device CUDA RNG required')
        package=torch.load(TEACHER,map_location='cpu',weights_only=False);validate_g05(package,'T3B_G05')
        teacher=restore_deployment(package,expected_step=5000).to(device);teacher.eval();teacher.requires_grad_(False)
        if {id(p) for p in teacher.parameters()} & {id(p) for g in optimizer.param_groups for p in g['params']}:raise ValueError('teacher optimizer overlap')
        del package
        module=DistributedDataParallel(base,device_ids=[local],find_unused_parameters=True)
        dataset=PairedStream(source,cfg,rank);loader=pretrain_loader(dataset,workers=8,seed=cfg['seed'],rank=rank);prefetch=iter(loader)
        # All construction and iterator setup precede restoration of each rank's RNG.
        restore_rng(rank_rng)
        if not actual_equal(rng_state(),rank_rng):raise ValueError('rank RNG restoration differs')
        write(folder/f'ancestry_rank{rank}.json',dict(**ancestry,restored_rng=True,optimizer='AdamW(original WD=0)',workers=8,teacher_in_optimizer=False))
        if rank==0:
            write(folder/'run.json',dict(config=cfg,identity=dict(state['identity'],config=cfg),parent_identity=state['identity'],
                start_step=5000,stop_step=10000,start_position=5040000,teacher=str(TEACHER),student_parameters=40846686,command=sys.argv))
        original_identity=state['identity'];ordered_keys=state['ordered_keys'];del state
        def ledger(status):
            write(folder/f'consumption_rank{rank}.json',dict(attempted_updates=attempted,completed_updates=completed,
                student_training_presentations=student_presented,teacher_training_presentations=teacher_presented,status=status))
            if rank==0:write(folder/'consumption.json',dict(attempted_updates=attempted,completed_updates=completed,
                student_training_presentations=student_presented*world,teacher_training_presentations=teacher_presented*world,status=status))
        base.train();ledger('RUNNING')
        for step in range(5000,10000):
            tick=time.monotonic();k=step-5000+1;attempted+=1;ledger('RUNNING')
            prepared=[next(prefetch) for _ in range(6)]
            counts=torch.zeros(3,device=device);graph_count=torch.zeros((),device=device)
            for data,labels,clean,meta in prepared:
                check_pair(data,clean)
                masked=torch.bincount(data.canonical_graph_index[labels['atom_mask']],minlength=data.graph_available.numel())
                centers=torch.bincount(data.bond_batch[data.bond_length_weight>0],minlength=data.graph_available.numel())
                counts+=torch.stack(((masked>0).sum(),((centers>0)&data.geometry_valid).sum(),torch.tensor(0))).to(device)
                graph_count+=len(meta)
            dist.all_reduce(counts);dist.all_reduce(graph_count)
            if int(graph_count)!=1008:raise ValueError('global graph count must equal1008')
            lr=lr_at(step,cfg)
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True);totals=torch.zeros(3,device=device);raw=torch.zeros(2,device=device);targets=torch.zeros(4,device=device)
            graph_total=torch.zeros((),device=device);norms=torch.zeros(2,device=device)
            for offset,(data,labels,clean,meta) in enumerate(prepared):
                data=data.to(device);clean=clean.to(device);labels=move_labels(labels,device)
                with torch.autocast('cuda',dtype=torch.bfloat16):target=teacher_target(teacher,clean)
                teacher_presented+=len(meta);del clean
                with module.no_sync() if offset<5 else nullcontext():
                    with torch.autocast('cuda',dtype=torch.bfloat16):result=module(data,labels,return_graph_feature=True)
                    student_presented+=len(meta)
                    errors,sn,tn=graph_errors(result['graph_feature'],target)
                    loss=global_objective(result['sums'],counts,world,cfg['loss_weights'])+graph_weight(arm,k)*graph_objective(errors,world,int(graph_count))
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite graph-distill PT objective')
                    loss.backward()
                totals+=result['sums'].detach();raw+=result['raw_geometry_sums'].detach();targets+=result['targets'].detach()
                graph_total+=errors.detach().sum();norms+=torch.stack((sn.sum(),tn.sum()))
                del target,result,errors,data,labels
                ledger('RUNNING')
            grad=float(torch.nn.utils.clip_grad_norm_(base.parameters(),1.,error_if_nonfinite=True));optimizer.step();completed+=1;ledger('RUNNING')
            for value in (totals,raw,targets,graph_total,norms):dist.all_reduce(value)
            record=dict(step=step+1,new_attempted=attempted,new_completed=completed,rank=rank,lr=lr,graph_weight=graph_weight(arm,k),
                local_losses=(totals/counts.clamp_min(1)).tolist(),valid_counts=counts.tolist(),target_counts=targets.tolist(),
                raw_geometry_losses=(raw/counts[1].clamp_min(1)).tolist(),graph_unweighted=float(graph_total/graph_count),
                norm_mean=dict(student=float(norms[0]/graph_count),teacher=float(norms[1]/graph_count)),graph_count=int(graph_count),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),teacher_parameter_bytes=sum(p.numel()*p.element_size() for p in teacher.parameters()),grad_preclip=grad,first_position=step*1008,last_position=(step+1)*1008-1,step_seconds=time.monotonic()-tick)
            print(json.dumps(record),flush=True)
            with (folder/f'records_rank{rank}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if step+1 in VALIDATION_STEPS:evaluate_validation(base,teacher,cfg,step+1,device,rank,world)
        if (attempted,completed,student_presented,teacher_presented)!=(5000,5000,2520000,2520000):raise ValueError('PT actual budget mismatch')
        states=[None]*world;dist.all_gather_object(states,rng_state())
        if rank==0:
            save_checkpoint(folder/'resume_10000.pt',dict(identity=dict(original_identity,config=cfg),parent_identity=original_identity,parent_resume=str(RESUME),
                ordered_keys=ordered_keys,direct_binding=None,step=10000,next_position=10080000,model=base.state_dict(),optimizer=optimizer.state_dict(),rng=states,scheduler=dict(step=10000,lr=lr_at(9999,cfg))))
            save_checkpoint(folder/'deploy_10000.pt',bind_deployment(deployment_package(base,10000),cfg))
        dist.barrier();ledger('COMPLETED');rc=0
    except BaseException as error:
        if isinstance(error,SystemExit):rc=int(error.code)
        if folder.exists():
            write(folder/f'failure_rank{rank}.json',dict(error=repr(error),traceback=traceback.format_exc(),attempted_updates=attempted,completed_updates=completed,
                student_training_presentations=student_presented,teacher_training_presentations=teacher_presented))
        raise
    finally:
        if folder.exists():write(folder/f'execution_rank{rank}.json',dict(start=started,end=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=rc,rank=rank,command=sys.argv,cwd=str(ROOT),environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}))
        if source is not None:source.close()
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);a=p.parse_args();main(a.arm)
