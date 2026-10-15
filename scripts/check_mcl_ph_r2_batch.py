#!/usr/bin/env python3
"""One actual train microbatch: split gradients and noisy-distance copy baseline, zero updates."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.dataset.mcl_ph_view import (load_geometric_statistics, validate_direct_statistics_source,
    prepare_mcl_ph_sample, mcl_ph_collate, normalise_geometric)
from src.modules.mcl_ph_pretrain import MCLPHPretrainer, effective_graph_counts, per_graph, HUBER_BETA
from src.training.glt_dual_runtime import (open_source, IndexedFrozenDualSource, OrderedSampleStream,
    load_sample_index_artifact, apply_common_initialization, move_labels, require_tmux)
from scripts.pretrain_mcl_ph import (SHARED_INIT_EXCLUDE_NAMES, build_shared_init, apply_shared_init,
                                    open_trajectory_cache)
from src.utils import set_global_seed


def copy_baseline(data, labels, statistics):
    graphs=int(data.graph_available.numel())
    position=data.mcl_pos.float()
    predictions={}
    for family in ('length','nonbond'):
        pairs=labels[f'mcl_{family}_pair']
        distance=torch.linalg.vector_norm(position[pairs[:,0]]-position[pairs[:,1]],dim=-1)
        predictions[family]=normalise_geometric(distance,statistics[family])
    pairs=labels['mcl_angle_pair']; bonds=data.mcl_bond_index
    first=bonds[pairs[:,0]]; second=bonds[pairs[:,1]]
    shared=torch.where((first[:,0]==second[:,0])|(first[:,0]==second[:,1]),first[:,0],first[:,1])
    if not (((shared==second[:,0])|(shared==second[:,1])).all()):
        raise ValueError('copy baseline angle bonds do not share an atom')
    a=torch.where(first[:,0]==shared,first[:,1],first[:,0])
    b=torch.where(second[:,0]==shared,second[:,1],second[:,0])
    va=position[a]-position[shared]; vb=position[b]-position[shared]
    predictions['angle']=F.cosine_similarity(va,vb,dim=-1,eps=1e-12)
    terms={}
    for family,prediction in predictions.items():
        target=labels[f'mcl_{family}_target']
        loss=F.huber_loss(prediction,target,reduction='none',delta=HUBER_BETA)
        terms[family]=per_graph(loss,labels[f'mcl_{family}_graph'],graphs)
    local_count=terms['length'][1].float()+terms['angle'][1].float()
    local=(terms['length'][0]+terms['angle'][0])/local_count.clamp_min(1)
    local_valid=local_count>0
    nonbond,nonbond_valid=terms['nonbond']
    # Match the model's per-bin then per-graph averaging, rather than pooling targets.
    bin_terms=[]; bin_valid=[]
    for slot in range(3):
        selected=labels['mcl_nonbond_slot']==slot
        loss=F.huber_loss(predictions['nonbond'][selected],labels['mcl_nonbond_target'][selected],reduction='none',delta=HUBER_BETA)
        value,present=per_graph(loss,labels['mcl_nonbond_graph'][selected],graphs)
        bin_terms.append(value); bin_valid.append(present.float())
    nonbond=torch.stack(bin_terms).sum(0)/torch.stack(bin_valid).sum(0).clamp_min(1)
    valid=local_valid|nonbond_valid
    geometry=torch.where(local_valid&nonbond_valid,(local+nonbond)*.5,torch.where(local_valid,local,nonbond))
    result={name:float(value[present].mean()) if present.any() else 0. for name,(value,present) in terms.items()}
    result.update(geometry=float(geometry[valid].mean()) if valid.any() else 0., geometry_graphs=int(valid.sum()),
                  protocol='normalized Huber delta0.5; per-bin/per-graph weighting matching geometry objective')
    return result



def attention_timing(model, data, report):
    import copy
    from scripts.benchmark_mcl_ph_graph_attention import original
    graph=data.canonical_graph_index
    selected=data.mcl_readout_valid[graph]
    index=data.mcl_central_index
    tokens=report['mixed'].detach().new_zeros((len(index),128))
    present=index>=0
    tokens[present]=report['mixed'].detach()[index[present]]
    hidden=report['atom_states'].detach()[selected]
    tokens=tokens[selected];graph=graph[selected]
    if not len(hidden):return dict(scope='NOT_APPLICABLE_NO_VALID_READOUT')
    module=copy.deepcopy(model.encoder.fusion)
    results=[];reference=None
    parameters=list(module.parameters())
    for name in ('original_chunked','graph_bucket'):
        measurements=[]
        for repeat in range(4):
            x=hidden.clone().requires_grad_(); t=tokens.clone().requires_grad_()
            torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            begin.record()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=original(module,x,t,graph) if name=='original_chunked' else module(x,t,graph)
            grads=torch.autograd.grad((out-x).float().square().mean(),[x,t]+parameters)
            end.record();torch.cuda.synchronize()
            if any(not torch.isfinite(g).all() for g in grads):raise ValueError('attention timing gradient is nonfinite')
            if repeat==0:
                current=[(out-x).detach().float()]+[g.detach().float() for g in grads]
                if reference is None:reference=current
                else:
                    errors=[]
                    for a,b in zip(current,reference):
                        errors.append(float((a-b).norm()/b.norm().clamp_min(1e-12)))
                    if max(errors)>2*torch.finfo(torch.bfloat16).eps:
                        raise ValueError('real-batch attention relative L2 differs beyond BF16 gate')
            else:measurements.append(dict(milliseconds=begin.elapsed_time(end),peak_extra_bytes=torch.cuda.max_memory_allocated()-baseline))
        results.append(dict(variant=name,forward_backward_ms=sum(row['milliseconds'] for row in measurements)/3,peak_extra_bytes=max(row['peak_extra_bytes'] for row in measurements)))
    return dict(scope='fusion only, detached states from the same actual batch, BF16',warmups=1,repeats=3,results=results,bf16_relative_L2_errors=errors,selected_implementation='graph_bucket',selection_reason='retain the verified implementation; report speed/memory tradeoff without adding a model factor')


def run(args):
    require_tmux()
    if args.output.exists():raise FileExistsError(args.output)
    for logdir in (args.statistics_log,args.cache_log):
        if (logdir/'exit.code').read_text().strip()!='0' or (logdir/'audit.exit.code').read_text().strip()!='0':
            raise ValueError('R1 build/audit exit code is not zero')
        if json.loads((logdir/'audit.json').read_text())['conclusion']!='PASS':
            raise ValueError('R1 audit report is not PASS')
    stats_audit=json.loads((args.statistics_log/'audit.json').read_text())
    cache_audit=json.loads((args.cache_log/'audit.json').read_text())
    if (stats_audit.get('scope')!='full' or stats_audit.get('valid_indices_vs_final_geometry')!='exact'
            or Path(stats_audit['statistics']).resolve()!=args.statistics.resolve()):
        raise ValueError('R1 statistics audit scope/path differs')
    if (cache_audit.get('positions')!=32 or cache_audit.get('structural_finite_shards')!=51
            or cache_audit.get('total_positions')!=5040000
            or cache_audit.get('manifest_complete') is not True
            or Path(cache_audit['cache_root']).resolve()!=args.cache.resolve()):
        raise ValueError('R1 cache audit scope/path differs')
    config=json.loads(args.config.read_text())
    set_global_seed(config['seed'])
    base,_=open_source(args.source_root,args.source_root,dual_static_root=args.source_root)
    split=load_sample_index_artifact(config['sample_index_artifact'],'train')
    source=IndexedFrozenDualSource(base,split['indices'])
    statistics=load_geometric_statistics(args.statistics)
    validate_direct_statistics_source(args.statistics,statistics,source,seed=config['seed'],sigma=config['noise_sigma'],ratio=config['atom_mask_ratio'])
    cache,_=open_trajectory_cache(args.cache,config=config,batch_size=config['global_batch'],sample_index=split,source=source)
    if not cache.complete or cache.manifest.get('complete') is not True or len(cache)!=5040000:raise ValueError('formal cache must be complete at 5040000 positions')
    cache.require_positions(5040000)
    stream=OrderedSampleStream(len(source),config['seed']); rows=[]; source_rows=[]
    for position in range(config['microbatch']):
        index=stream.index_at(position); record=source.base.record(int(source.source_indices[index]))
        rows.append(prepare_mcl_ph_sample(record.topology,record.trimer,source.samples[index][1],seed=config['seed'],key=source.samples[index][0].hex(),position=position,sigma=config['noise_sigma'],ratio=config['atom_mask_ratio'],static=record.static,statistics=statistics,trajectory_override=cache[position]))
        source_rows.append(source.source_index_at(index))
    data,labels=mcl_ph_collate(rows)
    baseline=copy_baseline(data,labels,statistics)
    trajectory=data.mcl_trajectory.numpy()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'budget.json').write_text(json.dumps(dict(unique_real_batches=1,optimizer_updates=0,source_indices=source_rows),indent=2)+'\n')
    model=MCLPHPretrainer('xattn',dropout=config['dropout'],cutoffs=config['cutoffs'],router_dense_updates=config['router_dense_updates'])
    path=config['common_init_artifact']; init=torch.load(path,map_location='cpu',weights_only=False)['common_state_dict']
    apply_common_initialization(model,path,exclude=SHARED_INIT_EXCLUDE_NAMES(model,init),direct=True)
    if not args.shared_init.exists():build_shared_init(args.shared_init,dropout=config['dropout'],cutoffs=config['cutoffs'],dense_updates=config['router_dense_updates'])
    apply_shared_init(model,args.shared_init,direct=True)
    device=torch.device('cuda:0');model=model.to(device).train();data=data.to(device);labels=move_labels(labels,device)
    # Match rank-0's post-initialization dropout stream in the training entry.
    set_global_seed(config['seed'])
    counts=effective_graph_counts(data.canonical_graph_index,labels,len(rows))['counts']
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):report=model(data,labels)
    loss=model.objective(report)
    empty_report=dict(report,atom_sum=report['atom_sum']*0,geo_sum=report['geo_sum']*0,
                      atom_count=report['atom_count']*0,geo_count=report['geo_count']*0,
                      balance=report['balance']*0,balance_empty_valid=True)
    empty_loss=model.objective(empty_report)
    if not torch.isfinite(empty_loss) or float(empty_loss)!=0:
        raise ValueError('empty-denominator objective is not finite zero')
    torch.cuda.synchronize();forward_seconds=time.perf_counter()-start
    if not torch.isfinite(loss):raise ValueError('nonfinite batch loss')
    for family,field in [('atom','atom_count'),('geometry','geo_count')]:
        if int(report[field])!=counts[family]:raise ValueError('effective graph denominators differ')
    names,parameters=zip(*[(name,p) for name,p in model.named_parameters() if p.requires_grad])
    groups={'o8':'encoder.o8.','experts':'encoder.branch.experts.','router':'encoder.branch.router.','fusion':'encoder.fusion.','atom_head':'atom_head.','local_decoder':'local_decoder.','nonbond_decoder':'nonbond_decoder.'}
    terms=dict(CE=report['atom_sum']/report['atom_count'].clamp_min(1),geometry=report['geo_sum']/report['geo_count'].clamp_min(1),balance=report['balance'])
    gradients={};start=time.perf_counter()
    for family,term in terms.items():
        values=torch.autograd.grad(term,parameters,retain_graph=True,allow_unused=True) if term.requires_grad else [None]*len(parameters)
        gradients[family]={}
        for group,prefix in groups.items():
            selected=[grad.detach().float() for name,grad in zip(names,values) if name.startswith(prefix) and grad is not None]
            if any(not torch.isfinite(grad).all() for grad in selected):raise ValueError('nonfinite split gradients')
            gradients[family][group]=dict(norm=float(torch.sqrt(sum((grad.square().sum() for grad in selected),torch.tensor(0.,device=device)))),connected_tensors=len(selected))
    if any(gradients['geometry'][group]['norm']!=0 for group in ('router','fusion')):raise ValueError('unexpected geometry router/fusion gradient')
    if counts['atom'] and bool(data.mcl_readout_valid.any()):
        if any(gradients['CE'][group]['connected_tensors']==0 for group in ('o8','experts','router','fusion')):
            raise ValueError('CE expected module path is disconnected')
    loss.backward();torch.cuda.synchronize();backward_seconds=time.perf_counter()-start
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):raise ValueError('nonfinite total gradients')
    full_peak=torch.cuda.max_memory_allocated()
    attention=attention_timing(model,data,report)
    result=dict(conclusion='PASS',scope='one train microbatch, split gradients, zero updates',graphs=len(rows),source_indices=source_rows,effective_graph_counts=counts,target_counts=report['targets'].tolist(),loss=float(loss),objective=report['objective_statistics'],empty_denominator_check=dict(scope='objective scaling only on zeroed numerators/counts; no extra model forward',loss=float(empty_loss)),gradients=gradients,copy_baseline=baseline,ph_column_variance=trajectory.astype(np.float64).var(axis=(0,1)).tolist(),ph_variance_scope='this batch across graphs/radii, not population',forward_seconds=forward_seconds,split_and_total_backward_seconds=backward_seconds,peak_allocated_bytes=full_peak,attention=attention,optimizer_updates=0)
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    cache.close();source.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','source-root','statistics','cache','statistics-log','cache-log','shared-init','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
