"""Single rolling campaign ledger writer; actual per-unit budgets are authoritative."""
import json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training.gp_sota_contract import *
from scripts.run_mcl_ph_r4_fast import atomic_json,timestamp
from src.training.glt_dual_runtime import require_tmux
require_tmux()
while True:
 state=read(LOG/'status.json');budget=state['budgets'];pre=[(p,read(p)) for p in OUT.glob('**/budget.json') if 'smoke' in read(p)];ft=[(p,read(p)) for p in OUT.glob('**/budget.json') if read(p).get('kind')=='finetune']
 budget.update(formal_pretrain_packages=len(list((OUT/'pretrain').glob('*/*/writer.claim'))),formal_pretrain_updates=sum(v['attempted_updates'] for p,v in pre if not v['smoke']),pretrain_smoke_updates=sum(v['attempted_updates'] for p,v in pre if v['smoke']),complete_finetune_starts=sum(v['stage']=='development' for p,v in ft),complete_finetune_epochs=sum(v.get('attempted_epochs',v['epochs']) for p,v in ft if v['stage']=='development'),complete_finetune_completed_epochs=sum(v['epochs'] for p,v in ft if v['stage']=='development'),complete_finetune_updates=sum(v['attempted_updates'] for p,v in ft if v['stage']=='development'),finetune_smoke_epochs=sum(v.get('attempted_epochs',v['epochs']) for p,v in ft if v['stage']=='smoke'),finetune_smoke_updates=sum(v['attempted_updates'] for p,v in ft if v['stage']=='smoke'),formal_test_only_units=len(list((OUT/'formal').glob('*/*/*/fold*/head_initial.pt'))))
 for k,cap in dict(formal_pretrain_packages=3,formal_pretrain_updates=15000,pretrain_smoke_updates=8,probe_forward_backward_calls=10,complete_finetune_starts=59,complete_finetune_epochs=4080,complete_finetune_updates=69930,finetune_smoke_epochs=2,finetune_smoke_updates=24,formal_test_only_units=35).items():assert budget[k]<=cap,(k,budget[k],cap)
 running=[];unitlogs=[]
 for d in sorted(LOG.iterdir()):
  if not d.is_dir() or not (d/'command.txt').exists():continue
  code=int((d/'exit.code').read_text()) if (d/'exit.code').exists() else None
  unitlogs.append(dict(attempt=d.name,status='RUNNING' if code is None else 'EXIT0' if code==0 else 'FAILED_PRESERVED',exit_code=code,command=(d/'command.txt').read_text().strip(),cwd=(d/'cwd.txt').read_text().strip(),log=str(d)))
  if code is None:running.append(str(d))
 state.update(budgets=budget,actual_unit_logs=unitlogs,running_units=running,updated_at=timestamp(),plan_status='执行中',execution_status='RUNNING_A_R2')
 final=LOG/'continuation_r2_02/exit.code'
 if final.exists():state.update(execution_status='EXECUTION_ENDED_NEEDS_SELF_CHECK',continuation_exit_code=int(final.read_text()));atomic_json(LOG/'status.json',state);print(json.dumps(dict(finished_at=timestamp(),budgets=budget,exit_code=int(final.read_text())),ensure_ascii=False),flush=True);break
 atomic_json(LOG/'status.json',state);print(json.dumps(dict(at=timestamp(),budgets=budget,running=running),ensure_ascii=False),flush=True);time.sleep(30)
