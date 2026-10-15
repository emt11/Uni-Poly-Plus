"""A bounded explicit unit list, four tmux GPU workers, fail-stop validation audit."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training.gp_sota_contract import *
from src.training.glt_dual_runtime import require_tmux
from scripts.run_mcl_ph_r4_fast import write_shell,start_window,atomic_json,timestamp

def rmse(p):
 with np.load(p,allow_pickle=False) as z:
  y,zp=z['y_true'],z['y_pred'];assert np.isfinite(y).all() and np.isfinite(zp).all();return float(np.sqrt(np.mean((y-zp)**2)))

def reference(task,fold,incumbent=False):
 if task=='xc':return ROOT/f'results/mcl_ph_r5_20261004/paper5_outer/xc_standard/glt_ref/xc/fold{fold}'
 arm='m_xattn' if incumbent and task=='eea' else 'glt_ref'
 return ROOT/f'results/mcl_ph_r4_20261002/fast_campaign_04/paper5_outer/{arm}/{task}/fold{fold}'

def audit(folder,unit):
 m=read(folder/'metrics.json');b=read(folder/'budget.json');r=read(folder/'runtime.json');c=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
 assert r['status']=='PASS' and m['outer_test']=='NOT_RUN' and not (folder/'test_predictions.npz').exists()
 task,fold,route,recipe=unit['task'],unit['fold'],unit['route'],unit['recipe']
 assert m['contract']['training']==ft_recipe(task,fold,route,recipe)
 assert len(m['history'])==epochs(task) and m['executed_epochs']==epochs(task)
 assert [v['stage'] for v in m['history']]==['head']*10+['joint']*(epochs(task)-10)
 assert b['attempted_updates']==b['completed_updates']==m['optimizer_updates']==sum(v['training_steps'] for v in m['history'])
 assert b['epochs']==b['attempted_epochs']==epochs(task)
 h=min(m['history'],key=lambda x:x['validation_rmse']);assert h['epoch']==m['best_epoch']==c['best_epoch'] and h['validation_rmse']==m['best_validation_rmse']==c['best_validation_rmse']
 assert abs(rmse(folder/'validation_predictions.npz')-m['best_validation_rmse'])<1e-8
 raw=np.array(c['scaler_train_raw']);assert abs(raw.mean()-c['scaler_mean'][0])<1e-10 and abs(raw.std()-c['scaler_scale'][0])<1e-10
 initial=torch.load(folder/'head_initial.pt',map_location='cpu',weights_only=False);old=torch.load(reference(task,fold)/'head_initial.pt',map_location='cpu',weights_only=False)
 assert set(initial)==set(old)==set(c['head_initial'])
 for k in initial:assert torch.equal(initial[k],old[k]) and torch.equal(initial[k],c['head_initial'][k])
 assert all(torch.isfinite(v).all() for v in c['state_dict'].values())
 for kind in ('glt','incumbent'):
  ref=reference(task,fold,incumbent=kind=='incumbent');oldm=read(ref/'metrics.json');value=rmse(ref/'validation_predictions.npz');assert abs(value-oldm['best_validation_rmse'])<1e-8
  m[f'{kind}_reference_rmse']=value;m[f'{kind}_relative_gain']=(value-m['best_validation_rmse'])/value
 return dict(unit=unit,folder=str(folder),best_epoch=m['best_epoch'],best_validation_rmse=m['best_validation_rmse'],glt_reference_rmse=m['glt_reference_rmse'],glt_relative_gain=m['glt_relative_gain'],incumbent_reference_rmse=m['incumbent_reference_rmse'],incumbent_relative_gain=m['incumbent_relative_gain'],optimizer_updates=m['optimizer_updates'],executed_epochs=m['executed_epochs'],outer_test='NOT_RUN',status='PASS')

def main():
 p=argparse.ArgumentParser();p.add_argument('--scope',required=True);args=p.parse_args();require_tmux();torch.set_num_threads(1)
 scope=read(args.scope);name=scope['name'];stageout=OUT/'development'/name;stageout.mkdir(parents=True,exist_ok=False)
 assert len(scope['units'])==len(scope['allowed_units']);active={};pending=list(scope['units']);completed=[]
 while pending or active:
  for gpu in range(4):
   if gpu in active or not pending:continue
   unit=pending.pop(0);task,fold,route,recipe=[unit[k] for k in ('task','fold','route','recipe')];key=f'{recipe}/{route}/{task}/fold{fold}/seed{42+fold}';assert key in scope['allowed_units']
   checkpoint=Path(scope['packages'][route]);assert read(checkpoint.parent/'acceptance.json')['status']=='PASS'
   folder=stageout/route/task/f'fold{fold}';log=LOG/f'{name}_{route}_{task}_f{fold}_01'
   command=['python','scripts/finetune_glt_mcl_ph.py','--task',task,'--fold',str(fold),'--route',route,'--recipe',recipe,'--stage','development','--scope',str(Path(args.scope).resolve()),'--checkpoint',str(checkpoint),'--output',str(folder)]
   write_shell(log,command,gpu=gpu);start_window(f'gp_{name}_{task}_{fold}',log);active[gpu]=dict(unit=unit,folder=folder,log=log)
   atomic_json(stageout/'status.json',dict(status='RUNNING',active=[dict(gpu=g,unit=v['unit'],folder=str(v['folder']),log=str(v['log'])) for g,v in active.items()],completed=completed,pending=pending))
  for gpu,entry in list(active.items()):
   if not (entry['log']/'exit.code').exists():continue
   code=int((entry['log']/'exit.code').read_text())
   if code!=0:
    atomic_json(stageout/'status.json',dict(status='FAILED',affected=entry['unit'],exit_code=code,pending_not_started=pending,other_active=[str(v['log']) for g,v in active.items() if g!=gpu],automatic_retry=False));raise RuntimeError('development failure '+str(entry['log']))
   completed.append(audit(entry['folder'],entry['unit']));del active[gpu]
  print(json.dumps(dict(stage=name,completed=len(completed),active=len(active),pending=len(pending))),flush=True);time.sleep(10)
 atomic_json(stageout/'report.json',dict(status='PASS',finished_at=timestamp(),rows=completed))
 atomic_json(stageout/'status.json',dict(status='PASS',completed=completed,active=[],pending=[]))
if __name__=='__main__':main()
