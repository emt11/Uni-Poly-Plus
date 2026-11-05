#!/usr/bin/env python3
"""One registered fresh10K PT trajectory. Independent process, no resume path."""
import os,sys,time,json,traceback,multiprocessing as mp
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training import ua3d_base_stabilize_r1 as t
import torch,numpy as np

def main():
    if len(sys.argv)!=2 or sys.argv[1] not in ('PT_R0','PT_R1'):raise ValueError('exact two PT identities only')
    run=sys.argv[1];t.require_tmux();t.install_stop_handler();setup=t.m.configure('DET');actual=t.gpu_contract();t.check_clock();t.seed_all(42)
    folder=t.OUT/'pretrain'/run;folder.mkdir(parents=True,exist_ok=False);clock=t.read(t.OUT/'authorization/pt_clock.json');deadline=clock['units'][run]['deadline_unix'];t.write(folder/'clock.json',dict(started_unix=time.time(),deadline_unix=deadline))
    context=mp.get_context('spawn');prepared=context.Array('q',4);model=None;iterator=None;attempted=completed=consumed=0;started=time.monotonic();durations=[];worker_io={};success=False
    def record(status):
        with prepared.get_lock():count=int(sum(prepared[:]))
        t.write(folder/'consumption.json',dict(status=status,pid=os.getpid(),process=t.process_identity(os.getpid()),attempted_updates=attempted,completed_updates=completed,pt_microbatches=consumed//168,pt_presentations=consumed,prefetched_presentations=count,prefetched_not_consumed=count-consumed,stream=model.stream_state() if model is not None else None,worker_io=worker_io,historical_tensor_reads=0,initial=str(t.INITIAL),elapsed_seconds=time.monotonic()-started))
    def pack(step,lr,loader):
        return dict(schema=t.m.PT_SCHEMA,plan=t.PLAN,revision='r1',step=step,seed=42,config=t.m.CONFIG,fp_prediction=False,historical_tensor_reads=0,initial=str(t.INITIAL),state_dict=t.m.clone_tree(model.state_dict()),ft_fusion_state=fusion,stream=model.stream_state(),optimizer=t.m.clone_tree(opt.state_dict()),scheduler=dict(kind='manual_continuous_two_5K',completed_updates=step,last_update=step-1,last_lr=lr,alignment_global_warmup=1000,optimizer_rebuilt=False),rng=t.m.rng_state(),loader_rng=loader.generator.get_state().clone())
    record('RUNNING')
    try:
        if time.time()>=deadline:raise TimeoutError('EXPIRED: PT dispatch deadline before initialization')
        model=t.m.pretrainer();initial=t.load_tensor(t.INITIAL);t.m.validate_pt(initial,0);t.m.strict_module_state(model,initial['state_dict']);initial_equal=t.m.exact_compare(model.state_dict(),initial['state_dict'])
        if not initial_equal['equal']:raise ValueError('new step0 strict loaded values differ')
        fusion=initial['ft_fusion_state'];del initial
        t.write(folder/'initial_load.json',dict(initial=str(t.INITIAL),strict_equal=initial_equal['equal'],comparison=initial_equal,schema=t.m.INITIAL_SCHEMA,historical_tensor_reads=0));model.to('cuda:0');t.seed_all(42);opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0.,betas=(.9,.999),eps=1e-8)
        ds=t.d.DecodedPT(t.PT_CACHE,t.MAP_ROOT,prepared);loader=torch.utils.data.DataLoader(ds,batch_size=None,num_workers=4,prefetch_factor=1,multiprocessing_context=context,worker_init_fn=t.d.worker_init,collate_fn=t.d.pt_collate,pin_memory=True,generator=torch.Generator().manual_seed(42));iterator=iter(loader)
        t.write(folder/'config.json',dict(plan=t.PLAN,run=run,model=t.m.CONFIG,pt=t.m.PT,initial=str(t.INITIAL),actual_device=actual,setup=setup,workers=4,prefetch=1,microbatch=168,accumulation=6,global_batch=1008,precision='BF16 forward / FP32 loss and optimizer',seed=42,optimizer='AdamW2e-4 wd0 clip1',schedule='two continuous5K, warmup500 each; ALIGN global warmup1000 once'))
        with (folder/'sampling.jsonl').open('x') as sampling,(folder/'history.jsonl').open('x') as history:
            for update in range(10000):
                t.check_clock()
                if time.time()>=deadline:raise TimeoutError('EXPIRED: PT single/combined stage deadline')
                tick=time.monotonic();attempted+=1;model.completed_updates=update;record('RUNNING');weights=model.weights();totals={k:torch.zeros(()) for k in weights};batches=[]
                for micro in range(6):
                    item=next(iterator);data,labels,trace=item.data,item.labels,item.trace
                    if trace['micro_index']!=update*6+micro or trace['start_position']!=(update*6+micro)*168:raise ValueError('PT ordered position identity')
                    consumed+=168;sampling.write(json.dumps(dict(step=update+1,**trace))+'\n');worker_io[str(trace['io']['worker'])]=trace['io'];counts=model.target_counts(t.d.pt_batch(data,labels),labels)
                    for k in totals:totals[k]+=counts[k]
                    batches.append((data,labels))
                sampling.flush();record('RUNNING');totals={k:v.to('cuda:0',non_blocking=True) for k,v in totals.items()};lr=t.pt_learning_rate(update)
                for group in opt.param_groups:group['lr']=lr
                opt.zero_grad(set_to_none=True);raw={k:torch.zeros((),device='cuda:0') for k in weights};micro_losses=[]
                for data,labels in batches:
                    data=data.to('cuda:0',non_blocking=True);labels={k:v.to('cuda:0',non_blocking=True) if torch.is_tensor(v) else v for k,v in labels.items()}
                    with torch.autocast('cuda',dtype=torch.bfloat16):result=model.pretrain_result(t.d.pt_batch(data,labels),labels)
                    with torch.autocast('cuda',enabled=False):loss=t.window_objective(result,weights,totals)
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite PT loss')
                    loss.backward();micro_losses.append(loss.detach())
                    for k in raw:raw[k]+=result['sums'][k].detach().float()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();completed+=1;model.completed_updates=completed
                rec=dict(step=completed,lr=lr,total_loss=float(torch.stack(micro_losses).sum()),grad_norm=float(norm),raw_sums={k:float(v) for k,v in raw.items()},losses={k:float(raw[k]/totals[k].clamp_min(1)) for k in raw},counts={k:float(v) for k,v in totals.items()},weights=weights,stream=model.stream_state(),seconds=time.monotonic()-tick,gpu_peak_bytes=torch.cuda.max_memory_allocated())
                durations.append(rec['seconds']);history.write(json.dumps(rec)+'\n');history.flush();t.write(folder/'timing.json',rec);record('RUNNING');print(json.dumps(rec),flush=True)
                if completed==100 or completed%500==0:
                    per=float(np.mean(durations[20:] if completed<=500 else durations[-500:]));remaining=(10000-completed)*per;other=10000*per if run=='PT_R0' else 0.;prior=t.read(t.OUT/'gates/remaining_eta.json');eta=1.5*(remaining+other+prior['downstream_serial_seconds'])+7200
                    t.write(folder/'speed.json',dict(step=completed,seconds_per_update=per,remaining_single=remaining,remaining_both=remaining+other,conservative_remaining=eta))
                    if time.time()+1.5*remaining>=deadline or time.time()+1.5*(remaining+other)>=clock['deadline_unix'] or time.time()+eta>=t.read(t.OUT/'authorization/clock.json')['deadline_unix']:raise TimeoutError('ETA_STOP: normal PT speed cannot fit full campaign')
                if completed in (5000,10000):
                    package=pack(completed,lr,loader);t.save_checkpoint(folder/f'step{completed}.pt',package)
                    if completed==10000:t.save_checkpoint(folder/'deploy.pt',package)
                    del package
        if next(iterator,None) is not None:raise ValueError('extra PT batch after60000')
        success=True;record('COMPLETE')
    except BaseException:record('FAILED');t.write(folder/'failure.json',dict(error=traceback.format_exc(),automatic_retry=False));raise
    finally:
        t.ignore_termination()
        if iterator is not None:iterator._shutdown_workers()
        record('COMPLETE' if success else 'FAILED')
if __name__=='__main__':main()
