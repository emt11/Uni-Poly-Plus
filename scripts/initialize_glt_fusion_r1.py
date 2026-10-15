#!/usr/bin/env python3
"""生成本周期公共无FP step0与固定核心配置，不执行模型前向。"""
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.modules.glt_dual_pretrain import DualPretrainer

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/glt_fusion_20261007/r1'
COMMON=ROOT/'results/mcl_ph_20260921/shared_pretrain/common_init_v1.pt'
TASKS=('eea','egb','egc','ei','eps','nc','xc')
CORE=('B0','T_RAW','T0','T0_A','K0','K_MD','K_MD_A','TK_MD','TK_MD_A','TK_MD_L')

def modes(arm):
    return dict(fusion_mode=('concat' if arm=='B0' else 'token' if arm.startswith('T0') or arm=='T_RAW' else 'tk' if arm.startswith('TK') else 'kfuse'),
        use_md200=('MD' in arm),atom_ce_input='raw' if arm in ('B0','T_RAW') else 'fused',
        align_enabled=arm.endswith('_A'),local_enabled=arm.endswith('_L'),
        third_task='none',fusion_revision='glt_fusion_r1',opt_mode='GLT_FUSION_R1',geometry_head_norm=True)

def write(path,payload):
    with path.open('x') as stream:
        json.dump(payload,stream,indent=2,allow_nan=False);stream.write('\n')

def initialize():
    folder=OUT/'initialization'; folder.mkdir(parents=True,exist_ok=False)
    common=torch.load(COMMON,map_location='cpu',weights_only=False)
    shared=common['common_state_dict']
    selected={name:value for name,value in shared.items() if not name.startswith('fp_head.')}
    torch.manual_seed(20261007)
    master=DualPretrainer('tk',fusion_revision='glt_fusion_r1',use_md200=True,
        atom_ce_input='fused',third_task='fgr',align_enabled=True,geometry_head_norm=True)
    state=master.state_dict()
    for name,value in selected.items():
        if name not in state or value.shape!=state[name].shape or value.dtype!=state[name].dtype:
            raise ValueError('common checkpoint shared tensor mismatch: '+name)
        if not bool(torch.isfinite(value).all()):
            raise ValueError('common checkpoint nonfinite: '+name)
        state[name]=value.detach().clone()
    if len(selected)!=201:
        raise ValueError('common checkpoint shared tensor count differs from planned 201: '+str(len(selected)))
    master.load_state_dict(state,strict=True)
    torch.save(dict(model=state,shared_names=sorted(selected),step=0,new_seed=20261007,common=str(COMMON),fp_task=False),folder/'common_step0.pt')
    units=[]
    for arm in CORE:
        cfg=modes(arm)
        model=DualPretrainer(**cfg)
        current=model.state_dict()
        for name,value in current.items():
            if name not in state or state[name].shape!=value.shape:
                raise ValueError('undeclared step0 tensor: '+name)
            current[name]=state[name].clone()
        model.load_state_dict(current,strict=True)
        torch.save(dict(model=current,opt_mode='GLT_FUSION_R1',step=0,arm=arm,modes=cfg),folder/f'{arm}_step0.pt')
        config=dict(cfg,arm=arm,seed=42,microbatch=84,global_batch=1008,expected_world_size=4,
            max_optimizer_steps=5000,save_every=5000,lr=2e-4,warmup_steps=2000,
            schedule_total_steps=20000,end_lr=1e-9,weight_decay=0,amp_dtype='bf16',
            atom_mask_ratio=.3,noise_sigma=.03,loss_weights=[1.,1.,0.],
            sample_index_artifact=str(ROOT/'results/mcl_ph_next_20260929/p0/direct_indices_v1/summary.json'),sample_index_split='train',
            initial_state=str(folder/f'{arm}_step0.pt'),md200_root=str(OUT/'md200/pi1m') if cfg['use_md200'] else None,
            plan='GLT-FUSION-20261007-01',revision='r1')
        write(folder/f'{arm}.json',config)
        write(folder/f'{arm}_parameters.json',dict(total=sum(p.numel() for p in model.parameters()),
            shared_tensor_names=sorted(set(current)&set(selected)),new_tensor_names=sorted(set(current)-set(selected))))
        for task in TASKS:
            split=json.loads((ROOT/f'data/splits/mips_paper5_outer/{task}.json').read_text())
            for fold,entry in enumerate(split['folds']):
                epochs=60 if task=='egc' else 70
                updates=epochs*((len(entry['train_indices'])+23)//24)
                units.append(dict(arm=arm,task=task,fold=fold,epochs=epochs,updates=updates,status='待执行'))
    if sum(unit['updates'] for unit in units)!=514500 or sum(unit['epochs'] for unit in units)!=24000:
        raise ValueError('actual split training budget differs from fixed plan')
    write(OUT/'core_units.json',units)
    write(folder/'summary.json',dict(shared_count=len(selected),arms=list(CORE),scientific_starts=0,optimizer_updates=0))
    print(json.dumps(dict(status='completed',shared_count=len(selected),arms=list(CORE))),flush=True)

if __name__=='__main__':initialize()
