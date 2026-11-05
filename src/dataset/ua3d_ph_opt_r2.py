"""Protected packing reader with vectorized dynamic index construction.

The parent packs, graph event algorithm and fixed P/C0/M paths are unchanged.
Only construction of the dynamic P_LEARN event tables is replaced.
"""
from itertools import chain
import numpy as np
import torch
from src.dataset.ua3d_ph_opt import (
    SCHEMA, STRATEGY, old, pack_side, batch_plan, move, Batch, collate,
    PData, make_data, loader, diagnostic_collate, workload, sum_work,
)


def padded(rows, dtype=np.int64):
    lengths = np.fromiter((len(row) for row in rows), dtype=np.int64, count=len(rows))
    width = int(lengths.max(initial=0))
    data = np.zeros((len(rows), width), dtype=dtype)
    mask = np.zeros((len(rows), width), dtype=np.bool_)
    total = int(lengths.sum())
    if total:
        row_ids = np.repeat(np.arange(len(rows)), lengths)
        starts = np.cumsum(lengths) - lengths
        columns = np.arange(total) - np.repeat(starts, lengths)
        flat = np.fromiter(chain.from_iterable(rows), dtype=dtype, count=total)
        data[row_ids, columns] = flat
        mask[row_ids, columns] = True
    return torch.from_numpy(data), torch.from_numpy(mask)


def event_plan(collections, total_nodes):
    """Exactly the parent's stable order, with bulk NumPy allocation/copy.

    No tensor is constructed in an individual node/event membership loop.
    Returning identical padded tables preserves forward and backward reductions.
    """
    births=[]; deaths=[]; censored=[]
    recipients=[[] for _ in range(4*total_nodes)]
    weights=[[] for _ in recipients]
    for k,offset,events in collections:
        for event in events:
            col=len(births)
            births.append([(i+offset)*4+k for i in event['birth']])
            deaths.append([(i+offset)*4+k for i in event['death']])
            censored.append(event['censored'])
            weight=event['count']/len(event['nodes'])
            for i in event['nodes']:
                row=k*total_nodes+offset+i
                recipients[row].append(col); weights[row].append(weight)
    bi,bm=padded(births); di,dm=padded(deaths)
    ri,rm=padded(recipients); rw,_=padded(weights,np.float32)
    return dict(birth_index=bi,birth_mask=bm,death_index=di,death_mask=dm,
                censored=torch.from_numpy(np.asarray(censored,dtype=np.bool_)),
                recipient_index=ri,recipient_mask=rm,recipient_weight=rw,
                events=len(births))
