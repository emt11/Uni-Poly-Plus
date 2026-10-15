"""Small SP/snapshot/CPU ensemble and exclusive single-writer budget helpers."""
import fcntl,json,os
from pathlib import Path
import numpy as np
import torch
from src.training.sota_ft_contract import OUT,PLAN,REV,read

def finite(value):
 if torch.is_tensor(value):
  if not torch.isfinite(value).all():raise FloatingPointError('nonfinite tensor')
 elif isinstance(value,(float,np.floating)):
  if not np.isfinite(value):raise FloatingPointError('nonfinite scalar')
 elif isinstance(value,(dict,list,tuple)):
  for v in (value.values() if isinstance(value,dict) else value):finite(v)

def json_write(path,value):
 finite(value);path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
 fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)

def tensor_copy(state):return {k:v.detach().cpu().clone() for k,v in state.items()}
def save_tensor(path,value):
 finite(value);path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp)
 with tmp.open('rb') as f:os.fsync(f.fileno())
 os.replace(tmp,path)
 fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)

def anchors(model):return {n:p.detach().clone() for n,p in model.named_parameters() if not n.startswith('model.predictor.')}
def penalty(model,anchor,alpha):
 if alpha==0:return None
 params={n:p for n,p in model.named_parameters() if not n.startswith('model.predictor.')}
 if set(params)!=set(anchor):raise ValueError('anchor keys differ')
 return (alpha/2)*sum((p-anchor[n]).square().sum() for n,p in params.items())

class SPLoss:
 def __init__(self,model,anchor,alpha):self.model,self.anchor,self.alpha=model,anchor,alpha;self.mse=[];self.sp=[]
 def __call__(self,output,target):
  mse=torch.nn.functional.mse_loss(output,target);sp=penalty(self.model,self.anchor,self.alpha)
  self.mse.append(mse.detach());self.sp.append(mse.detach()*0 if sp is None else sp.detach())
  return mse if sp is None else mse+sp

def mean_predictions(rows):
 if not rows:raise ValueError('no completed snapshots')
 first=rows[0]
 for r in rows:
  for name in ('sample_keys','indices','y_true','scaler_mean','scaler_scale'):
   if not np.array_equal(r[name],first[name]):raise ValueError('ensemble mismatch: '+name)
  finite(r['y_pred'])
 return dict(first,y_pred=np.mean(np.stack([r['y_pred'] for r in rows]),axis=0))

def consume(folder,contract=None):
 folder=Path(folder)
 if (folder/'UNCOMMITTED').exists() or not (folder/'runtime.json').exists() or (folder/'failure.json').exists():raise ValueError('unit incomplete or failed')
 runtime=read(folder/'runtime.json');m=read(folder/'metrics.json')
 if runtime['status']!='PASS' or m['status']!='PASS' or m['outer_test']!='NOT_RUN':raise ValueError('unit not accepted development')
 if contract is not None and m['contract']!=contract:raise ValueError('consumer contract mismatch')
 return m

def reserve(folder,stage,k,epochs,updates,ledger=None,caps=None,plan=PLAN,revision=REV):
 ledger=Path(ledger or OUT/'budget.json');lock=ledger.with_suffix('.lock')
 with lock.open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  b=read(ledger) if ledger.exists() else dict(plan=plan,revision=revision,units={})
  if k in b['units']:raise ValueError('unit already reserved, no retry')
  vals=[v for v in b['units'].values() if v['stage']==stage]
  cap=caps or ((2,6,66) if stage=='smoke' else (8,560,6160))
  if len(vals)+1>cap[0] or sum(v['reserved_epochs'] for v in vals)+epochs>cap[1] or sum(v['reserved_updates'] for v in vals)+updates>cap[2]:raise ValueError('budget reservation exceeded')
  b['units'][k]=dict(stage=stage,folder=str(folder),reserved_epochs=epochs,reserved_updates=updates,attempted_epochs=0,completed_epochs=0,attempted_updates=0,completed_updates=0,status='RESERVED')
  json_write(ledger,b)

def account(k,field,ledger=None,status=None,total_cap=6226):
 ledger=Path(ledger or OUT/'budget.json')
 with ledger.with_suffix('.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);b=read(ledger);u=b['units'][k]
  if field:
   cap=u['reserved_epochs'] if field.endswith('epochs') else u['reserved_updates']
   if u[field]>=cap:raise ValueError('unit budget exceeded')
   u[field]+=1
  if status:u['status']=status
  if sum(v['attempted_updates'] for v in b['units'].values())>total_cap:raise ValueError('total update cap')
  json_write(ledger,b);json_write(Path(u['folder'])/'budget.json',u)
