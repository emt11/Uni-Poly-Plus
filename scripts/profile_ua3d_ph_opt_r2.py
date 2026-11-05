#!/usr/bin/env python3
"""Registered operator-only engineering measurement; real targets never opened."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.dataset import ua3d_ph_opt as prior_data
from src.dataset import ua3d_ph_opt_r2 as data
from src.modules import ua3d_ph_opt as prior_module
from src.modules import ua3d_ph_opt_r2 as module
from src.modules import ua3d_stability as numerical

OUT=ROOT/'results/ua3d_ph_opt_20261031/r1/repairs/auto_r2'
PARENT=ROOT/'results/ua3d_ph_opt_20261031/r1'
ROWS=[947,1468,3281,2870,1799,1993,1258,1843,438,258,2453,1066,1778,1529,67,1163,2388,285,1781,1331,1606,3057,938,2136]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--round',type=int,required=True);args=ap.parse_args()
    if not os.environ.get('TMUX') or os.environ.get('CUDA_VISIBLE_DEVICES')!='2':raise RuntimeError('GPU2 tmux required')
    if args.round not in (1,2,3):raise ValueError('engineering round budget exceeded')
    folder=OUT/'engineering'/f'round{args.round:02d}';folder.mkdir(parents=True,exist_ok=False)
    state=dict(status='RUNNING',forwards=0,backwards=0,graph_reductions=0,full_base_forwards=0,optimizer_updates=0,real_labels=0,measurements=[])
    def record():
        q=folder/'consumption.json';tmp=q.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(q)
    record()
    numerical.configure('DET')
    if str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')!='31456dd5-a421-142b-644b-7a80f482dfb9':raise RuntimeError('wrong GPU')
    try:
        sides=[torch.load(PARENT/'packing/egc'/f'row{i}.pt',map_location='cpu',weights_only=False) for i in ROWS]
        for row,side in zip(ROWS,sides):
            if side['row']!=row or side['task']!='egc':raise ValueError('training input identity')
        packed=data.move(data.batch_plan(sides,'P_LEARN'),'cuda:0')
        torch.manual_seed(2026102801);old=prior_module.GraphPHAdapter().cuda()
        new=module.GraphPHAdapter().cuda();new.load_state_dict(old.state_dict(),strict=True)
        with torch.no_grad():old.out[-1].weight.fill_(.01);new.out[-1].weight.fill_(.01)
        outputs={};grads={};prof=cProfile.Profile()
        for name,adapter in [('parent',old),('optimized',new)]:
            for repeat in range(2):
                adapter.zero_grad(set_to_none=True);before=adapter.reductions
                state['forwards']+=1;record();torch.cuda.synchronize();tick=time.perf_counter()
                if name=='optimized' and repeat==1:prof.enable()
                value,info=adapter.forward_batch(packed,'P_LEARN')
                if name=='optimized' and repeat==1:prof.disable()
                torch.cuda.synchronize();forward=time.perf_counter()-tick
                state['graph_reductions']+=adapter.reductions-before
                state['backwards']+=1;record();tick=time.perf_counter();value.square().sum().backward();torch.cuda.synchronize()
                backward=time.perf_counter()-tick
                state['measurements'].append(dict(version=name,repeat=repeat,forward=forward,backward=backward,pairing=adapter.pairing_seconds,transfer=adapter.cpu_transfer_seconds,nodes=packed['nodes']))
                outputs[name]=value.detach().cpu();grads[name]={k:p.grad.detach().cpu().clone() for k,p in adapter.named_parameters() if p.grad is not None}
                record()
        if not torch.allclose(outputs['parent'],outputs['optimized'],atol=1e-6,rtol=1e-5):raise ValueError('operator output differs')
        if set(grads['parent'])!=set(grads['optimized']):raise ValueError('gradient set differs')
        for key in grads['parent']:
            if not torch.allclose(grads['parent'][key],grads['optimized'][key],atol=1e-5,rtol=1e-4):raise ValueError('operator gradient differs: '+key)
        stream=io.StringIO();pstats.Stats(prof,stream=stream).sort_stats('cumulative').print_stats(24)
        (folder/'profile.txt').write_text(stream.getvalue())
        state.update(status='COMPLETE',equivalence=True)
    except BaseException:
        state.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        record()
        print(json.dumps(state),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
