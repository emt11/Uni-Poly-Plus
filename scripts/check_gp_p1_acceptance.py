"""Saved-artifact P1 self acceptance, without model inference."""
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training.gp_sota_contract import *
from src.training.mcl_ph_direct_contract import assert_actual_equal
from scripts.run_mcl_ph_r4_fast import atomic_json,timestamp

def finite(x):
 if isinstance(x,torch.Tensor):assert torch.isfinite(x).all()
 elif isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,(tuple,list)):
  for v in x:finite(v)
 elif isinstance(x,(float,np.floating)):assert np.isfinite(x)

def main():
 torch.set_num_threads(1)
 for name in ('p0_common_v2_01','p1_cpu_02','p1_guards_01','p1_promotion_tests_01','p1_property_tests_01','p1_probe_01','p1_ddp_GP_02','p1_ddp_GP_resume_01','p1_ddp_GP_A_01','p1_ft_smoke_01'):
  assert (LOG/name/'exit.code').read_text().strip()=='0',name
 assert read(OUT/'p1_joint_probe_01/report.json')['calls']==10
 common=torch.load(COMMON,map_location='cpu',weights_only=False)['common_state_dict']
 g0=torch.load(OUT/'smoke/GP/part1_02/resume_00000.pt',map_location='cpu',weights_only=False,mmap=True)
 a0=torch.load(OUT/'smoke/GP_A/full_01/resume_00000.pt',map_location='cpu',weights_only=False,mmap=True)
 for k,v in common.items():assert_actual_equal(g0['model'][k],v,'GP.common.'+k);assert_actual_equal(a0['model'][k],v,'GP_A.common.'+k)
 for k,v in g0['model'].items():assert_actual_equal(v,a0['model'][k],'core.'+k)
 for folder,step,start in [(OUT/'smoke/GP/part1_02',2,0),(OUT/'smoke/GP/part2_01',4,2),(OUT/'smoke/GP_A/full_01',4,0)]:
  state=torch.load(folder/f'resume_{step:05d}.pt',map_location='cpu',weights_only=False,mmap=True);pkg=torch.load(folder/f'deploy_{step:05d}.pt',map_location='cpu',weights_only=False,mmap=True)
  assert state['step']==pkg['step']==step and state['next_position']==step*1008 and len(state['rng'])==4;finite(state);finite(pkg)
  expected={k.removeprefix('encoder.'):v for k,v in state['model'].items() if k.startswith('encoder.')}
  assert_actual_equal(pkg['state_dict'],expected,'complete.encoder.export')
  assert read(folder/'budget.json')['completed_updates']==step-start
  for rank in range(4):
   rows=[json.loads(l) for l in (folder/f'rank{rank}.jsonl').read_text().splitlines()];assert [r['step'] for r in rows]==list(range(start+1,step+1))
   assert all(r['global_counts'][0]==1008 and r['global_counts'][2]==1008 for r in rows)
   if 'GP_A' in folder.parts:assert all(r['global_counts'][3]>0 and r['losses'][3]>0 for r in rows)
 smoke=OUT/'smoke/ft_xc_fold0_01';m=read(smoke/'metrics.json');s=torch.load(smoke/'best.pt',map_location='cpu',weights_only=False);finite(s);finite(m)
 assert m['optimizer_updates']==24 and len(m['history'])==2 and m['outer_test']=='NOT_RUN' and not (smoke/'test_predictions.npz').exists()
 assert read(smoke/'budget.json')['completed_updates']==24
 raw=np.array(s['scaler_train_raw']);np.testing.assert_allclose(s['scaler_mean'],[raw.mean()],rtol=0,atol=1e-12);np.testing.assert_allclose(s['scaler_scale'],[raw.std()],rtol=0,atol=1e-12)
 p=np.load(smoke/'validation_predictions.npz');y,z=p['y_true'],p['y_pred'];np.testing.assert_allclose(np.sqrt(np.mean((y-z)**2)),m['best_validation_rmse'],rtol=0,atol=1e-12)
 best=min(m['history'],key=lambda h:h['validation_rmse']);assert best['epoch']==m['best_epoch']
 atomic_json(OUT/'p1_acceptance.json',dict(status='PASS',checked_at=timestamp(),release_formal_pretraining=True,common_205_exact=True,three_arm_initialization_cpu_test=True,
  gp_gp_a_saved_complete_core_exact=True,single_noisy_view_and_legacy_glt_input_target_parity=True,online_cached_ph_exact=True,
  real_probe_calls=10,real_probe_updates=0,pretrain_smoke_updates=8,ft_smoke_epochs=2,ft_smoke_updates=24,
  ddp_global_denominators=True,synthetic_ddp_invalid_rank_duplicate_positives=True,resume_model_optimizer_rank_rng_exact=True,complete_encoder_export=True,
  scope_freeze_promotion_cpu_checks=True,history_best_raw_metric_scaler=True,extra_model_calls=0,extra_updates=0,historical_fp_step0_exact='未核实'))
 print('PASS: P1 prerequisites, no extra model calls/updates',flush=True)
if __name__=='__main__':main()
