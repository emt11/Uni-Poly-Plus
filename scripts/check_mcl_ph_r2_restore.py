#!/usr/bin/env python3
"""Verify a saved one-update DDP checkpoint on every rank, without another update."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import torch.distributed as dist
from src.modules.glt_dual_pretrain import DualPretrainer
from src.modules.mcl_ph_pretrain import MCLPHPretrainer
from src.training.glt_dual_runtime import (open_source, IndexedFrozenDualSource, load_sample_index_artifact,
    rng_state, restore_rng, require_tmux)
from src.training.mcl_ph_direct_contract import direct_binding, assert_actual_equal, verify_restored_checkpoint


def run(args):
    require_tmux()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);dist.init_process_group('nccl')
    if rank==0:args.output.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    config=json.loads(args.config.read_text())
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if state['identity']['config']!=config or state['identity']['world_size']!=world:
        raise ValueError('checkpoint recipe/world differs')
    if state['step']!=1 or state['next_position']!=1008:raise ValueError('smoke checkpoint consumed unexpected budget')
    base,_=open_source(args.source_root,args.source_root,dual_static_root=args.source_root)
    split=load_sample_index_artifact(config['sample_index_artifact'],'train')
    source=IndexedFrozenDualSource(base,split['indices'])
    if args.route=='m_xattn':
        model=MCLPHPretrainer('xattn',dropout=config['dropout'],cutoffs=config['cutoffs'],router_dense_updates=config['router_dense_updates'])
    else:model=DualPretrainer('concat',geometry_head_norm=config['geometry_head_norm'],third_task=config.get('third_task','none'))
    binding=direct_binding(source,common_init=config['common_init_artifact'],
        shared_init=args.shared_init if args.route=='m_xattn' else None,
        statistics=args.statistics if args.route=='m_xattn' else None,
        seed=config['seed'],total_positions=config['max_optimizer_steps']*config['global_batch'])
    assert_actual_equal(state.get('direct_binding'),binding)
    if state['ordered_keys']!=[key.hex() for key,_ in source.samples]:raise ValueError('ordered source keys differ')
    model=model.to(torch.device('cuda',local));model.load_state_dict(state['model'],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    optimizer.load_state_dict(state['optimizer'])
    verify_restored_checkpoint(model,optimizer,state,batch_size=config['global_batch'])
    restore_rng(state['rng'][rank]);assert_actual_equal(rng_state(),state['rng'][rank],'restored.rng')
    report=dict(conclusion='PASS',rank=rank,world=world,step=state['step'],next_position=state['next_position'],
                model='exact',optimizer='exact',rng='exact',source_sampling_statistics_initialization='actual values exact',
                model_forwards=0,optimizer_updates=0,checkpoint=str(args.checkpoint.resolve()))
    (args.output/f'rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
    gathered=[None]*world;dist.all_gather_object(gathered,report)
    if rank==0:(args.output/'report.json').write_text(json.dumps(dict(conclusion='PASS',ranks=gathered,extra_updates=0),indent=2)+'\n')
    source.close();dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route',choices=('glt_ref','m_xattn'),required=True)
    for name in ('config','checkpoint','source-root','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--shared-init',type=Path);parser.add_argument('--statistics',type=Path)
    run(parser.parse_args())
