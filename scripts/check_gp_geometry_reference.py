"""Read-only geometry difficulty on the same P1 24 train rows; no model calls."""
import sys,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training.gp_sota_contract import *
from src.training.glt_dual_runtime import require_tmux,open_source,pretrain_source_fields
from src.dataset.glt_mcl_ph import prepare_gp_sample
from scripts.run_mcl_ph_r4_fast import atomic_json
require_tmux();torch.set_num_threads(1)
path=OUT/'p1_joint_probe_01/geometry_reference.json';assert not path.exists()
ids=read(OUT/'p1_joint_probe_01/input_acceptance.json')['samples'];indices=[r['source_index'] for r in ids]
source,_=open_source(PI,PI,dual_static_root=PI,selected_indices=sorted(indices))
local={source.source_index_at(i):i for i in range(len(source))};lengths=[];angles=[];types=[]
for r in ids:
 top,trimer,s,static,target,identity=pretrain_source_fields(source,local[r['source_index']],include_target=True,include_identity=True)
 d,t=prepare_gp_sample(top,trimer,s,seed=42,key=r['key'],position=r['position'],static=static,target=target,identity=identity)
 if not d.geometry_valid:continue
 labels=t['glt'];lengths.append((d.bond_distance[d.bond_center].numpy(),labels['distance'].numpy()));types.append(d.bond_type[d.bond_center].numpy())
 pair=labels['angle_pairs'];lookup={tuple(p):j for j,p in enumerate(d.line_path[:,:2].tolist())};rows=[lookup[tuple(p)] for p in pair.tolist()]
 angles.append((d.line_angle[rows,0].cos().numpy(),labels['angle_cos'].numpy()))
x,y=np.concatenate([v[0] for v in lengths]),np.concatenate([v[1] for v in lengths]);k=np.concatenate(types);means={int(t):float(y[k==t].mean()) for t in np.unique(k)};baseline=np.array([means[int(t)] for t in k])
a,b=np.concatenate([v[0] for v in angles]),np.concatenate([v[1] for v in angles]);assert len(x)>0 and len(a)>0
atomic_json(path,dict(status='PASS',sample_indices=indices,model_forward_backward_calls=0,updates=0,noise_sigma=.03,length_noisy_copy_rmse=float(np.sqrt(np.mean((x-y)**2))),angle_cos_noisy_copy_rmse=float(np.sqrt(np.mean((a-b)**2))),length_train_bond_type_mean_rmse=float(np.sqrt(np.mean((baseline-y)**2))),train_bond_type_means=means,length_targets=len(y),angle_targets=len(b),mean_fit_scope='same fixed 24 PI1M train rows only; in-sample diagnostic, not validation generalization',interpretation='Small noise reconstruction is a diagnostic; low geometric loss is not evidence of downstream property benefit'))
source.close()
