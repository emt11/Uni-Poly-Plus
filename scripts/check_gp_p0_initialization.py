"""Read-only step0 coverage check; no model forward or updates."""
import json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
paths=[ROOT/'results/mcl_ph_20260921/shared_pretrain/common_init_v1.pt',ROOT/'results/glt_pred_20260918/s3b_prep/common_init_v1.pt']
rows=[]
for path in paths:
 payload=torch.load(path,map_location='cpu',weights_only=False)
 row={'path':str(path),'keys':list(payload),'states':{}}
 for name,value in payload.items():
  if isinstance(value,dict):
   tensors={k:v for k,v in value.items() if isinstance(v,torch.Tensor)}
   row['states'][name]={'tensor_count':len(tensors),'fp_names':[k for k in tensors if 'fp_head' in k],'head_names':[k for k in tensors if 'head' in k],'all_finite':all(torch.isfinite(v).all().item() for v in tensors.values())}
 rows.append(row)
report={'status':'INSPECTED','artifacts':rows,'model_calls':0,'updates':0}
print(json.dumps(report,indent=2))
(ROOT/'results/gp_sota_20261004/p0_initialization_01.json').write_text(json.dumps(report,indent=2)+'\n')
