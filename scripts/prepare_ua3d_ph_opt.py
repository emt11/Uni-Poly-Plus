#!/usr/bin/env python3
"""One no-label packing pass; existing graph/VR results are never recomputed."""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def packing():
    import torch
    from src.dataset import ua3d_ph_opt as d
    from src.training import ua3d_ph_opt as t
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or not os.environ.get('TMUX'):
        raise RuntimeError('packing must run in GPU-hidden tmux')
    folder=t.OUT/'packing'; folder.mkdir(exist_ok=False)
    u=t.Consumption(folder,'packing'); u.add('packing'); u.record()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    try:
        rows=t.read(t.SOURCE/'inputs/rows.json'); inventory={}; count=0; available=0
        for task in t.TASKS:
            (folder/task).mkdir(); inventory[task]={}
            for entry in rows[task]:
                t.guard_deadline(); t.guard_preparation(); row=entry['row']
                side=torch.load(t.SOURCE/'inputs'/task/f'row{row}.pt',map_location='cpu',weights_only=False)
                if side['task']!=task or side['row']!=row or side['key']!=entry['key'] or side['available']!=entry['available']:
                    raise ValueError('source sidecar identity')
                u.add('packing_rows'); u.record()
                packed=d.pack_side(side)
                torch.save(packed,folder/task/f'row{row}.pt')
                inventory[task][str(row)]=d.workload(side); count+=1; available+=int(side['available'])
                u.add('packing_rows',completed=True)
                if count%64==0:
                    u.record(); print(count,flush=True)
        if (count,available)!=(5875,5860):
            raise ValueError('packing coverage 5875/5860/15')
        diagnostic_inputs={}
        for label,task,fold,ids in t.BATCHES:
            records=[inventory[task][str(i)] for i in ids]
            diagnostic_inputs[label]=dict(task=task,fold=fold,rows=ids,work=d.sum_work(records),
                                         max_graph={k:max(r[k] for r in records) for k in records[0]})
        t.write(folder/'inventory.json',dict(rows=inventory,diagnostic_inputs=diagnostic_inputs,
                work_definitions='per-row counts; quadratic attention terms; explicit worst-case learned event membership bound'))
        t.write(folder/'metadata.json',dict(schema=d.SCHEMA,status='COMPLETE',records=count,geometry_available=available,geometry_missing=count-available,
                labels_accessed=0,model_forward=0,new_graph_reductions=0,new_vr_reductions=0,new_rwpe=0,source=str(t.SOURCE/'inputs'),strategy=d.STRATEGY))
        u.add('packing',completed=True); u.finish()
    except BaseException:
        u.fail(); raise


if __name__=='__main__':
    packing()
