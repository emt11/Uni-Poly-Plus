#!/usr/bin/env python3
"""仅P_train clean有效目标的总体矩；不使用validation/test，不生成构象。"""
import json
import math
from pathlib import Path
import sys
import time
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.training.glt_dual_runtime import open_source, load_sample_index_artifact, require_tmux
from src.dataset.glt_dual import build_dual_sample
from src.dataset.glt_dual_pretrain import fgr_true_trimer_candidates

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/glt_fusion_20261007/r1'
PI=ROOT/'data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging'

def main():
    require_tmux();torch.set_num_threads(4)
    path=OUT/'geometry_statistics.json'
    if path.exists():raise FileExistsError('statistics output already exists; do not overwrite')
    index=load_sample_index_artifact(ROOT/'results/mcl_ph_next_20260929/p0/direct_indices_v1/summary.json','train')
    source,_=open_source(PI,PI,selected_indices=index['indices'].tolist(),verify_key_digest=False)
    moments={name:[0,0.,0.] for name in ('length','angle','fgr_log1p')}
    started=time.time()
    try:
        for i in range(len(source)):
            row=source.entries[i]
            if row['pretrain_role']!='train':raise ValueError('statistics source is not P_train')
            record=source.record(i)
            clean=build_dual_sample(record.topology,record.trimer,row['source_smiles'],static=record.static)
            if clean.geometry_valid:
                valid=(clean.line_mask.sum(1)==1)&~clean.line_is_self
                pairs=clean.line_path[:,:2]
                valid &= pairs[:,0]<pairs[:,1]
                valid &= clean.bond_center[pairs[:,0].clamp_min(0)]&clean.bond_center[pairs[:,1].clamp_min(0)]
                _,distances,_=fgr_true_trimer_candidates(record.topology,record.trimer,identity=record.validated_normalized_identity)
                targets=dict(length=clean.bond_distance[clean.bond_center],angle=clean.line_angle[valid,0].cos(),fgr_log1p=torch.log1p(distances))
                for name,value in targets.items():
                    value=value.double()
                    if not bool(torch.isfinite(value).all()):raise FloatingPointError('nonfinite clean statistics '+name)
                    moments[name][0]+=value.numel();moments[name][1]+=float(value.sum());moments[name][2]+=float(value.square().sum())
            if (i+1)%1000==0:print(json.dumps(dict(rows=i+1,moments=moments,seconds=time.time()-started)),flush=True)
        result={}
        for name,(count,total,squared) in moments.items():
            if count==0:raise ValueError('no valid train targets '+name)
            mean=total/count;variance=squared/count-mean**2
            if not math.isfinite(variance) or variance<=0:raise ValueError('nonpositive/nonfinite train variance '+name)
            result[name]=dict(count=count,mean=mean,std=math.sqrt(variance))
        with path.open('x') as handle:json.dump(dict(source=str(PI),indices=index['path'],role='P_train',rows=len(source),ddof=0,target_weighting='all clean valid targets',statistics=result,seconds=time.time()-started),handle,indent=2)
    finally:source.close()
if __name__=='__main__':main()
