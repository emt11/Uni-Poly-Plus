"""E1's sole bounded GPU3 comparison: 36 pairs, no optimizer or extra probes."""
import copy,json,math,os,sys,time
from pathlib import Path
from types import ModuleType
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.modules.glt_dual_opt import BondTokenFusion

def main():
    from src.training.glt_dual_runtime import require_tmux,open_source,pretrain_source_fields
    from src.dataset.glt_dual import build_dual_sample,dual_glt_collate
    require_tmux()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3':raise ValueError('E1 final comparison requires physical GPU3 only')
    torch.set_num_threads(16)
    out=ROOT/'results/glt_optimization_20261011/r10/E1';consumption=out/'gpu_consumption.json'
    counter=dict(pairs_attempted=0,pairs_completed=0,fusion_forward_backward_attempted=0,fusion_forward_backward_completed=0,optimizer_updates=0,worker_starts=0)
    consumption.write_text(json.dumps(counter,indent=2))
    path=out/'before/src/modules/glt_dual_opt.py';legacy=ModuleType('legacy_e1');exec(compile(path.read_text(),str(path),'exec'),legacy.__dict__)
    down=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
    split=json.loads((ROOT/'data/splits/mips_paper5_outer/eea.json').read_text())
    selected=sorted(split['folds'][0]['train_indices'])[:24]
    source,_=open_source(down,down,task='eea',selected_indices=selected,split_manifest=split,verify_key_digest=False)
    try:
        samples=[]
        for index in range(len(source)):
            top,tri,text,static,_=pretrain_source_fields(source,index)
            samples.append(build_dual_sample(top,tri,text,static=static,geometry_mode='finite_incident'))
    finally:source.close()
    sizes=(2,4,8,12,16,20,24,24);batches=[]
    generator=torch.Generator().manual_seed(733)
    for size in sizes:
        data=dual_glt_collate(samples[:size])
        atoms=torch.randn(data.canonical_graph_index.numel(),512,generator=generator)
        bonds=torch.randn(data.bond_batch.numel(),512,generator=generator)
        batches.append((atoms.to('cuda:0'),bonds.to('cuda:0'),data.canonical_graph_index.to('cuda:0'),data.bond_batch.to('cuda:0'),data.geometry_valid.to('cuda:0'),data.incidence_index.to('cuda:0')))
    package=torch.load(ROOT/'results/glt_periodic_seam_20261009/r1/pretrain/F_CE/deploy_05000.pt',map_location='cpu',weights_only=False)
    state={k.removeprefix('token_fusion.'):v for k,v in package['state_dict'].items() if k.startswith('token_fusion.')}
    rows=[];torch.cuda.reset_peak_memory_stats()
    for mode,beta in [('global',0),('incident_only',0),('incident_bias',math.log(2))]:
        old=legacy.BondTokenFusion(attention_mode=mode,incident_bias=beta).to('cuda:0')
        new=BondTokenFusion(attention_mode=mode,incident_bias=beta).to('cuda:0')
        old.load_state_dict(state,strict=True);new.load_state_dict(state,strict=True);new.set_implementation('bucket')
        times={'loop':[],'bucket':[]}
        for pair in range(12):
            counter['pairs_attempted']+=1;consumption.write_text(json.dumps(counter,indent=2))
            batch=batches[pair%8];outputs=[];grads=[]
            for label,module in [('loop',old),('bucket',new)]:
                a=batch[0].detach().requires_grad_();b=batch[1].detach().requires_grad_();module.zero_grad(set_to_none=True)
                counter['fusion_forward_backward_attempted']+=1;consumption.write_text(json.dumps(counter,indent=2))
                torch.cuda.synchronize();start=time.perf_counter()
                value=module(a,b,*batch[2:5],incidence=batch[5]);value.square().mean().backward()
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                outputs.append(value.detach());grads.append([a.grad,b.grad,*[p.grad for p in module.parameters()]])
                counter['fusion_forward_backward_completed']+=1;consumption.write_text(json.dumps(counter,indent=2))
                if pair>=4:times[label].append(elapsed)
            torch.testing.assert_close(outputs[0],outputs[1],atol=1e-5,rtol=1e-4)
            for a,b in zip(*grads):
                if (a is None)!=(b is None):raise ValueError('fusion gradient presence differs')
                if a is not None:torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-4)
            counter['pairs_completed']+=1;consumption.write_text(json.dumps(counter,indent=2))
        rows.append(dict(mode=mode,loop_seconds=sum(times['loop'])/8,bucket_seconds=sum(times['bucket'])/8,speedup=sum(times['loop'])/sum(times['bucket'])))
    result=dict(status='PASS',consumption=counter,modes=rows,peak_allocated_bytes=torch.cuda.max_memory_allocated(),dtype='fp32',atol=1e-5,rtol=1e-4,batches=[dict(graphs=x[4].numel(),atoms=x[0].size(0),bonds=x[1].size(0)) for x in batches],states='fixed synthetic states at existing 24 train-record graph sizes; fusion-only timing, not end-to-end training',gpu='physical3/logical0')
    with (out/'gpu_comparison.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
