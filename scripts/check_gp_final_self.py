"""Read saved evidence and predictions only; self-check is not independent review."""
import sys,json,subprocess,math,difflib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training.gp_sota_contract import *
from scripts.run_mcl_ph_r4_fast import atomic_json,timestamp
from scripts.finish_gp_sota import summarize
from scripts.run_gp_development_stage import audit
from src.training.mcl_ph_direct_contract import assert_actual_equal
from src.training.glt_dual_runtime import require_tmux,scheduled_lr

def finite(x):
 if torch.is_tensor(x):assert torch.isfinite(x).all()
 elif isinstance(x,(float,np.floating)):assert math.isfinite(x)
 elif isinstance(x,(dict,list,tuple)):
  for v in (x.values() if isinstance(x,dict) else x):finite(v)

def otherdiff(s):
 chunks=s.split('diff --git ');return [x for x in chunks[1:] if not x.startswith(('a/CURRENT_PLAN.md ','a/src/dataset/mcl_ph_view.py '))]

def main():
 require_tmux();torch.set_num_threads(1);folder=OUT/'final_self_check_r2_02';folder.mkdir(exist_ok=False)
 assert (LOG/'continuation_r2_02/exit.code').exists() and (LOG/'monitor_r2_02/exit.code').exists()
 assert int((LOG/'continuation_r2_03/exit.code').read_text())==1
 assert int((LOG/'p2b_GP_A_01/exit.code').read_text())==1
 olddiff=(LOG/'implementation_r2_01/before.diff').read_text();newdiff=subprocess.check_output(['git','diff'],text=True);assert otherdiff(olddiff)==otherdiff(newdiff),'unrelated tracked diff changed'
 (folder/'actual_tracked_diff.patch').write_text(newdiff)
 (folder/'mcl_ph_view_incremental.patch').write_text(''.join(difflib.unified_diff((LOG/'implementation_r2_01/mcl_ph_view.py').read_text().splitlines(True),(ROOT/'src/dataset/mcl_ph_view.py').read_text().splitlines(True),fromfile='protected_before/mcl_ph_view.py',tofile='r2/mcl_ph_view.py')))
 protected_checks=dict(unrelated_tracked_diff_unchanged=True,r1_reports_preserved=True)
 assert read(OUT/'p0_contract_01.json')['status'].startswith('STOP') or read(OUT/'p0_contract_01.json').get('release_for_training') is False
 assert read(OUT/'initialization/acceptance_01.json')['status']=='PASS' and read(OUT/'p1_acceptance.json')['status']=='PASS'
 packrows=[];stops=[];deficiencies=[]
 common=torch.load(COMMON,map_location='cpu',weights_only=False)['common_state_dict'];assert len(common)==205
 for path in sorted((OUT/'pretrain').glob('*/*/writer.claim')):
  pack=path.parent;b=read(pack/'budget.json');last=b['total_step'];route=pack.parent.name
  rows=[json.loads(x) for x in (pack/'rank0.jsonl').read_text().splitlines()];finite(rows)
  assert [r['step'] for r in rows]==list(range(1,last+1)) and b['completed_updates']==b['attempted_updates']==last<=5000
  for rank in range(1,4):
   rankrows=[json.loads(x) for x in (pack/f'rank{rank}.jsonl').read_text().splitlines()];finite(rankrows);assert [r['step'] for r in rankrows]==list(range(1,last+1))
  assert all(abs(r['lr']-scheduled_lr(r['step']-1,lr=2e-4,warmup_steps=2000,schedule_total_steps=20000,end_lr=1e-9))<1e-15 for r in rows)
  saved_step=last
  if not (pack/f'resume_{last:05d}.pt').exists() or not (pack/f'deploy_{last:05d}.pt').exists():
   assert (pack/'router_stop.json').exists(), 'missing completed-package checkpoint'
   saved_step=0
   deficiencies.append(dict(route=route,missing_stop_checkpoint_step=last,remaining_temporary_files=[dict(path=str(p),bytes=p.stat().st_size) for p in pack.glob('*.tmp')],final_weights_optimizer_rng='不可核实；仅核实step0'))
  state=torch.load(pack/f'resume_{saved_step:05d}.pt',map_location='cpu',weights_only=False,mmap=True);deploy=torch.load(pack/f'deploy_{saved_step:05d}.pt',map_location='cpu',weights_only=False,mmap=True);finite(state);finite(deploy)
  assert state['step']==saved_step and state['next_position']==saved_step*1008 and len(state['rng'])==4 and state['contract']['recipe']==pretrain_recipe(route)
  assert_actual_equal({k.removeprefix('encoder.'):v for k,v in state['model'].items() if k.startswith('encoder.')},deploy['state_dict'],'complete.encoder')
  start=torch.load(pack/'resume_00000.pt',map_location='cpu',weights_only=False,mmap=True)
  for k,v in common.items():assert torch.equal(v,start['model'][k]),k
  windows=[];streak=np.zeros(3,dtype=int)
  for step in range(100,last+1,100):
   rates=np.mean([r['selection_rates'] for r in rows[step-100:step]],axis=0);streak=np.where((rates<.01)&(step>500),streak+1,0);windows.append(dict(step=step,rates=rates.tolist(),low_streak=streak.tolist()))
  stopped=(pack/'router_stop.json').exists()
  if stopped:
   evidence=read(pack/'router_stop.json');assert evidence['step']==last and (streak>=3).any();np.testing.assert_allclose(evidence['window_rates'],windows[-1]['rates'],rtol=0,atol=0)
   stops.append(dict(route=route,reason='router low-selection guard',evidence=evidence,weights_preserved=saved_step==last))
  else:assert last==5000 and read(pack/'acceptance.json')['status']=='PASS'
  packrows.append(dict(route=route,attempted_updates=last,completed_updates=last,next_position=last*1008,checked_checkpoint_step=saved_step,complete_encoder_exact_at_checked_step=True,step0_205_exact=True,optimizer_rng_finite_at_checked_step=True,windows=windows,stopped=stopped))
  del state,deploy,start
 ftrows=[]
 for path in sorted(OUT.glob('development/**/metrics.json')):
  unit=path.parent;m=read(path);c=torch.load(unit/'best.pt',map_location='cpu',weights_only=False,mmap=True);finite(c);finite(m)
  audit(unit,dict(task=m['task'],fold=m['fold'],route=m['route'],recipe=m['recipe']))
  raw=pd.read_csv(ROOT/f"data/raw/smi_{m['task']}.csv").iloc[:,1].to_numpy(dtype=float);entry=read(SPLITS/f"{m['task']}.json")['folds'][m['fold']]
  np.testing.assert_array_equal(np.array(c['scaler_train_raw']),raw[entry['train_indices']]);np.testing.assert_allclose(c['scaler_mean'],[raw[entry['train_indices']].mean()],rtol=0,atol=1e-12);np.testing.assert_allclose(c['scaler_scale'],[raw[entry['train_indices']].std()],rtol=0,atol=1e-12)
  with np.load(unit/'validation_predictions.npz',allow_pickle=False) as z:
   assert z['validation_indices'].tolist()==entry['validation_indices'];np.testing.assert_allclose(z['y_true'],raw[entry['validation_indices']],rtol=2e-6,atol=2e-6);value=float(np.sqrt(np.mean((z['y_true']-z['y_pred'])**2)));assert abs(value-m['best_validation_rmse'])<1e-9
  assert m['outer_test']=='NOT_RUN' and not (unit/'test_predictions.npz').exists();h=min(m['history'],key=lambda v:v['validation_rmse']);assert h['epoch']==m['best_epoch']==c['best_epoch']
  ftrows.append(dict(folder=str(unit),train_only_scaler_exact=True,raw_validation_metric_exact=True,best_epoch=m['best_epoch'],outer_test='NOT_RUN'))
 if not (OUT/'seven_task_results_r2.json').exists():
  assert stops,'non-router failure requires manual evidence review before result synthesis';summarize({}, {},'P2b GP_A在800updates触发路由低选择率守卫；GP/GU完成，未进行候选正式预测')
 results=read(OUT/'seven_task_results_r2.json');paper={}
 if deficiencies:
  results['status']='REUSED_PREDICTIONS_RECOMPUTED_A_INCOMPLETE'
  results['self_check_deficiencies']=deficiencies
  atomic_json(OUT/'seven_task_results_r2.json',results)
 for line in (ROOT/'SOTA_RESULT.md').read_text().splitlines():
  c=[v.strip() for v in line.split('|')]
  if len(c)==5 and c[1].lower() in TASKS:paper[c[1].lower()]=(c[2],float(c[3].strip('*').split(' ± ')[0]))
 assert set(paper)==set(TASKS)
 for row in results['rows']:
  task=row['task'];assert (row['paper_model'],row['paper_best_r2'])==paper[task];assert abs(row['project_minus_paper']-(row['mean_r2']-paper[task][1]))<1e-12
 ledger=read(LOG/'status.json');finite(ledger['budgets'])
 report=dict(status='SELF_CHECK_INCOMPLETE_MISSING_STOP_CHECKPOINT' if deficiencies else 'SELF_CHECK_PASS_PENDING_INDEPENDENT_REVIEW',deficiencies=deficiencies,checked_at=timestamp(),protected_checks=protected_checks,pretrain_packages=packrows,stops=stops,development_checks=ftrows,budgets=ledger['budgets'],seven_task_rows=results['rows'],extra_model_calls=0,extra_updates=0,independent_review='未执行',project_history_archived=False,historical_fp_step0_exact='未核实')
 atomic_json(folder/'report.json',report);print(json.dumps(dict(status=report['status'],stops=stops,budgets=ledger['budgets']),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
