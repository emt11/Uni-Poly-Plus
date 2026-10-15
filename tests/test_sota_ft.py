import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
from torch import nn
from src.training.sota_ft_contract import recipe,key,OUT,PLAN,REV,check_scope
from src.training.sota_ft_runtime import anchors,penalty,SPLoss,tensor_copy,mean_predictions,reserve,account,json_write,consume,read

class Toy(nn.Module):
 def __init__(self):
  super().__init__();self.model=nn.Module();self.model.encoder=nn.Linear(2,3);self.model.predictor=nn.Linear(3,1)
 def forward(self,x):return self.model.predictor(self.model.encoder(x))

def test_alpha_zero_identical_trajectory_rng():
 torch.manual_seed(7);a=Toy();b=Toy();b.load_state_dict(a.state_dict());anchor=anchors(b);x=torch.tensor([[1.,2.],[3.,4.]]);y=torch.tensor([[1.],[2.]])
 o=[torch.optim.AdamW(m.parameters(),lr=.001) for m in (a,b)]
 before=torch.get_rng_state().clone()
 for _ in range(3):
  for i,m in enumerate((a,b)):
   o[i].zero_grad();out=m(x);loss=nn.functional.mse_loss(out,y) if i==0 else SPLoss(m,anchor,0.)(out,y);loss.backward();o[i].step()
  assert all(torch.equal(v,b.state_dict()[n]) for n,v in a.state_dict().items())
 assert torch.equal(before,torch.get_rng_state())

def test_anchor_frozen_head_excluded_and_snapshot_no_alias():
 m=Toy();a=anchors(m);before=tensor_copy(a);snap=tensor_copy(m.state_dict());assert all('predictor' not in n and not v.requires_grad for n,v in a.items())
 with torch.no_grad():
  for p in m.parameters():p.add_(.1)
 penalty(m,a,.001).backward()
 assert all(torch.equal(v,before[n]) for n,v in a.items())
 assert all(v.data_ptr()!=m.state_dict()[n].data_ptr() for n,v in snap.items())
 assert not torch.equal(snap['model.encoder.weight'],m.model.encoder.weight)

def test_ensemble_raw_keys_scaler_guards():
 r=dict(sample_keys=np.array(['a','b']),indices=np.array([1,2]),y_true=np.array([1.,2.]),y_pred=np.array([0.,2.]),scaler_mean=np.array([3.]),scaler_scale=np.array([4.]))
 s=dict(r,y_pred=np.array([2.,4.]));np.testing.assert_array_equal(mean_predictions([r,s])['y_pred'],[1.,3.])
 for n in ('sample_keys','indices','y_true','scaler_mean','scaler_scale'):
  z=dict(s);z[n]=np.array(['c','d']) if n=='sample_keys' else s[n]+1
  with pytest.raises(ValueError,match='ensemble mismatch'):mean_predictions([r,z])

def test_scope_recipe_output_and_future_stages_reject():
 r=recipe('development','eps',0,'F0');k=key('development','eps',0,'F0');s=dict(plan=PLAN,revision=REV,status='FROZEN_A0_A3',allowed_stages=['smoke','development'],units={k:dict(recipe=r)},protected_inputs={})
 check_scope(s,'development','eps',0,'F0',OUT/k)
 with pytest.raises(ValueError):check_scope(s,'development','eps',0,'F0',OUT/'other')
 s['units'][k]['recipe']=dict(r,alpha=.01)
 with pytest.raises(ValueError):check_scope(s,'development','eps',0,'F0',OUT/k)
 with pytest.raises(ValueError):recipe('formal','eps',0,'F0')
 with pytest.raises(ValueError):recipe('development','eps',2,'F0')

def test_budget_reentry_and_update_cap(tmp_path):
 ledger=tmp_path/'ledger.json';folders=[]
 for i in range(2):
  f=tmp_path/str(i);f.mkdir();folders.append(f);reserve(f,'smoke',str(i),3,33,ledger)
 with pytest.raises(ValueError,match='already reserved'):reserve(folders[0],'smoke','0',3,33,ledger)
 with pytest.raises(ValueError,match='exceeded'):reserve(tmp_path,'smoke','2',3,33,ledger)
 b=read(ledger);b['units']['0']['attempted_updates']=33;json_write(ledger,b)
 with pytest.raises(ValueError,match='exceeded'):account('0','attempted_updates',ledger)
 assert read(ledger)['units']['0']['attempted_updates']==33

def test_failed_or_uncommitted_consumption_and_contract(tmp_path):
 json_write(tmp_path/'runtime.json',dict(status='PASS'));json_write(tmp_path/'metrics.json',dict(status='PASS',outer_test='NOT_RUN',contract={'scaler':[1.]}))
 consume(tmp_path,{'scaler':[1.]})
 with pytest.raises(ValueError):consume(tmp_path,{'scaler':[2.]})
 (tmp_path/'UNCOMMITTED').touch()
 with pytest.raises(ValueError):consume(tmp_path)
 (tmp_path/'UNCOMMITTED').unlink();json_write(tmp_path/'failure.json',dict(status='FAILED'))
 with pytest.raises(ValueError):consume(tmp_path)
