"""Final package audit plus exactly one fixed 1024-row diagnostic evaluation."""
import sys,argparse,json,math
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training.gp_sota_contract import *
from src.training.glt_dual_runtime import require_tmux,load_sample_index_artifact,open_source,pretrain_source_fields,pretrain_loader,move_labels
from src.training.mcl_ph_direct_contract import assert_actual_equal
from src.modules.glt_mcl_ph_pretrain import GPPretrainer
from src.dataset.glt_mcl_ph import prepare_gp_sample,gp_collate
from scripts.run_mcl_ph_r4_fast import atomic_json,timestamp

def finite(obj):
 if torch.is_tensor(obj):assert torch.isfinite(obj).all()
 elif isinstance(obj,float):assert math.isfinite(obj)
 elif isinstance(obj,(dict,list,tuple)):
  for v in (obj.values() if isinstance(obj,dict) else obj):finite(v)

class Validation:
 def __init__(self,source):self.source=source
 def __len__(self):return (len(self.source)+83)//84
 def __getitem__(self,b):
  items=[]
  for i in range(b*84,min((b+1)*84,len(self.source))):
   top,trimer,s,static,target,identity=pretrain_source_fields(self.source,i,include_target=True,include_identity=True)
   assert self.source.entries[i]['pretrain_role']=='validation'
   d,t=prepare_gp_sample(top,trimer,s,seed=42,key=self.source.samples[i][0].hex(),position=i,static=static,target=target,identity=identity)
   items.append((d,t))
  return gp_collate(items)

def main():
 p=argparse.ArgumentParser();p.add_argument('--folder',required=True);args=p.parse_args();require_tmux();torch.set_num_threads(1)
 folder=Path(args.folder);assert not (folder/'acceptance.json').exists();(folder/'validation.writer').touch(exist_ok=False)
 runtime=read(folder/'runtime.json');assert runtime['status']=='PASS' and runtime['step']==5000
 budget=read(folder/'budget.json');assert budget['attempted_updates']==budget['completed_updates']==5000
 s=torch.load(folder/'resume_05000.pt',map_location='cpu',weights_only=False);d=torch.load(folder/'deploy_05000.pt',map_location='cpu',weights_only=False)
 finite(s);finite(d);assert s['step']==5000 and s['next_position']==5040000 and len(s['rng'])==4
 assert s['contract']['recipe']==pretrain_recipe(d['route'])
 expected={k[len('encoder.'):]:v for k,v in s['model'].items() if k.startswith('encoder.')};assert_actual_equal(expected,d['state_dict'],'complete.encoder')
 common=torch.load(COMMON,map_location='cpu',weights_only=False)['common_state_dict'];initial=torch.load(folder/'resume_00000.pt',map_location='cpu',weights_only=False)
 for k,v in common.items():assert torch.equal(v,initial['model'][k]),k
 rows=[json.loads(x) for x in (folder/'rank0.jsonl').read_text().splitlines()];assert [r['step'] for r in rows]==list(range(1,5001));assert all(r['router_mode']==('dense' if r['step']<=500 else 'top2') for r in rows)
 for rank in range(1,4):assert len((folder/f'rank{rank}.jsonl').read_text().splitlines())==5000
 torch.manual_seed(42);model=GPPretrainer(route=d['route'],common_path=COMMON);model.load_state_dict(s['model'],strict=True);model.encoder.router.configure('top2',5000);model.cuda().eval()
 idx=load_sample_index_artifact(INDEX,'fixed_validation');assert len(idx['indices'])==1024
 source,_=open_source(PI,PI,dual_static_root=PI,selected_indices=idx['indices'].tolist());assert len(source)==1024
 sums=np.zeros(3);counts=np.zeros(3);length=angle=0.;usage=np.zeros(3);valid=0;processed=0;align_sum=align_count=0.
 try:
  with torch.no_grad():
   for data,labels in pretrain_loader(Validation(source),workers=8,seed=42,rank=0):
    processed+=data.graph_available.numel()
    with torch.autocast('cuda',dtype=torch.bfloat16):r=model(data.cuda(),move_labels(labels,torch.device('cuda')),collect_diagnostics=True)
    finite(r);sums+=r['sums'].cpu().numpy();counts+=r['counts'].cpu().numpy();diag=r['diagnostics']['components']
    length+=diag['length_sum'];angle+=diag['angle_sum'];flag=r['router_valid'];usage+=((r['router_weights']>0)*flag.unsqueeze(-1)).sum(0).cpu().numpy();valid+=int(flag.sum())
    align_sum+=float(r['align_sum']);align_count+=float(r['align_count'])
    print(json.dumps(dict(processed=processed,objective_means=(sums/np.maximum(counts,1)).tolist())),flush=True)
  assert processed==1024 and np.isfinite(sums).all()
  report=dict(status='PASS',checked_at=timestamp(),step=5000,source_indices=idx['indices'].tolist(),evaluated_rows=processed,evaluation_calls_per_package=1,optimizer_updates=0,
   objective_sums=sums.tolist(),objective_counts=counts.tolist(),objective_means=(sums/np.maximum(counts,1)).tolist(),length_mean=length/max(counts[1],1),angle_cos_mean=angle/max(counts[1],1),alignment_mean=align_sum/max(align_count,1),selection_rates=(usage/max(valid,1)).tolist(),complete_encoder_exact=True,step0_205_exact=True,resume_finite=True,route=d['route'],historical_fp_step0_exact='未核实',validation_total_loss_candidate_selection=False,diagnostic_microbatch=84,alignment_diagnostic_pool='single rank current diagnostic batch; not a cross-objective selection metric')
  atomic_json(folder/'acceptance.json',report)
 finally:source.close()
if __name__=='__main__':main()
