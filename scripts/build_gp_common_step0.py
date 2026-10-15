"""Write a fresh FP-free common initialization to a new output directory."""
import argparse,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.modules.glt_dual_pretrain import DualPretrainer
from src.training.glt_dual_runtime import save_checkpoint

def construct():
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(42)
  model=DualPretrainer('concat',geometry_head_norm=True,third_task='none',atom_target='element',torsion=False)
 return {k:v.clone() for k,v in model.state_dict().items()}

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args();torch.set_num_threads(1)
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
 a=construct()
 old_path=ROOT/'results/mcl_ph_20260921/shared_pretrain/common_init_v1.pt'
 old=torch.load(old_path,map_location='cpu',weights_only=False)['common_state_dict']
 if set(a)!=set(old):raise ValueError('FP-free common tensor coverage differs')
 for k,v in old.items():
  if v.shape!=a[k].shape or v.dtype!=a[k].dtype:raise ValueError('common tensor incompatible: '+k)
 payload=dict(schema='gp-common-init-no-fp-v1',seed=42,common_state_dict=old,
  constructor=dict(class_name='DualPretrainer',fusion_mode='concat',geometry_head_norm=True,third_task='none',atom_target='element',torsion=False),
  torch_version=str(torch.__version__),code_sources=['src/modules/glt_dual_pretrain.py','src/modules/glt_dual.py'],
  historical_common_path=str(old_path))
 save_checkpoint(out/'common_step0_no_fp.pt',payload)
 print('Saved FP-free common initialization: '+str(out/'common_step0_no_fp.pt'),flush=True)
if __name__=='__main__':main()
