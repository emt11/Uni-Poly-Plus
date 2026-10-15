"""Sole final CPU suite for new FT dropout, gate adaptation and frozen matrix."""
import copy,json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from scripts.run_glt_ft_robustness_unit import RobustnessArm,mask_3d,add_ft_gate,strict_restore
from scripts import select_glt_ft_robustness as selection
from src.modules.glt_dual import build_dual_glt_model,mean_pool
from scripts.finetune_mcl_ph import periodic_tdl_trainability,periodic_tdl_optimizer

OPTIONS=dict(fusion_revision='glt_fusion_r1',geometry_revision='glt_periodic_seam_r1',geometry_mode='finite_incident',angle_value_enabled=True,readout='incident')

def encoded_fixture():
 return dict(atom_states=torch.randn(6,512),token_states=torch.randn(6,512),canonical_graph_index=torch.tensor([0,0,1,1,2,2]),bond_graph_index=torch.tensor([0,0,1,1,2,2]),geometry_valid=torch.tensor([True,True,False]),graph_3d=torch.randn(3,512),incidence_index=torch.stack([torch.arange(6),torch.arange(6)]))

def test_drop_cuts_both_paths_preserves_source_and_blocks_norm_bias():
 model=build_dual_glt_model('token',**OPTIONS).eval()
 with torch.no_grad():model.norm3.bias.fill_(3.)
 encoded=encoded_fixture();original=model.fuse(encoded).detach().clone()
 old_valid=encoded['geometry_valid'].clone();old_geometry=encoded['graph_3d'].clone();old_fused=encoded['fused_atom_states']
 view=mask_3d(encoded,torch.tensor([False,True,True]));actual=model.fuse(view)
 expected=model.norm2(mean_pool(encoded['atom_states'],encoded['canonical_graph_index'],3))
 torch.testing.assert_close(actual[0,:512],expected[0]);assert torch.count_nonzero(actual[[0,2],512:])==0
 torch.testing.assert_close(actual[1],original[1]);torch.testing.assert_close(actual[2,:512],expected[2])
 assert torch.equal(encoded['geometry_valid'],old_valid) and torch.equal(encoded['graph_3d'],old_geometry)
 assert encoded['fused_atom_states'] is old_fused and view['fused_atom_states'] is not old_fused

class TinyEncodedModel(nn.Module):
 def __init__(self):
  super().__init__();self.predictor=nn.Linear(1,1);self.saved=dict(geometry_valid=torch.tensor([True,True,False,True]),graph_3d=torch.ones(4,512))
 def encode(self,data):return dict(self.saved)
 def fuse(self,encoded):return encoded['geometry_valid'].clone()

def tiny_arm(fold=0):
 arm=RobustnessArm.__new__(RobustnessArm);nn.Module.__init__(arm)
 arm.model=TinyEncodedModel();arm.drop_probability=.2
 arm.mask_generator=torch.Generator().manual_seed(10000+42+fold)
 arm.counts=dict(joint_source_valid=0,joint_dropped=0,joint_effective_valid=0)
 return arm

def test_joint_only_rng_isolation_and_paired_mask_sequence():
 d=tiny_arm(2);dg=tiny_arm(2);d.train();dg.train()
 global_state=torch.get_rng_state().clone()
 for _ in range(20):assert torch.equal(d.features(None),dg.features(None))
 assert torch.equal(global_state,torch.get_rng_state())
 counts=d.dropout_snapshot();assert counts['joint_source_valid']==60 and counts['joint_dropped']>0
 assert counts['joint_effective_valid']+counts['joint_dropped']==60
 for head_only in (True,False):
  d.eval()
  if head_only:d.model.predictor.train()
  before=d.mask_generator.get_state().clone()
  assert torch.equal(d.features(None),torch.tensor([True,True,False,True]))
  assert torch.equal(before,d.mask_generator.get_state()) and d.dropout_snapshot()['joint_source_valid']==0
 assert d.model.saved['geometry_valid'].tolist()==[True,True,False,True]

def test_ft_gate_preserves_trained_common_scale_rng_and_optimizer_members():
 encoder=build_dual_glt_model('token',**OPTIONS).eval()
 with torch.no_grad():encoder.token_fusion.scale.fill_(.37)
 before={k:v.clone() for k,v in encoder.state_dict().items()};encoded=encoded_fixture();expected=encoder.fuse(dict(encoded))
 rng=torch.get_rng_state().clone();add_ft_gate(encoder)
 assert torch.equal(rng,torch.get_rng_state())
 state=encoder.state_dict();assert set(state)-set(before)=={'token_fusion.gate.weight','token_fusion.gate.bias'}
 for k,v in before.items():assert torch.equal(v,state[k])
 torch.testing.assert_close(encoder.fuse(dict(encoded)),expected)
 assert encoder.token_fusion.gate.weight.numel()+encoder.token_fusion.gate.bias.numel()==1025
 wrapper=nn.Module();wrapper.model=encoder
 for head_only in (True,False):
  periodic_tdl_trainability(wrapper,head_only=head_only)
  optimizer,_=periodic_tdl_optimizer(wrapper,'xc',head_only=head_only)
  gate=list(encoder.token_fusion.gate.parameters())
  assert all(p.requires_grad==(not head_only) for p in gate)
  backbone=[g for g in optimizer.param_groups if g['name']=='backbone']
  if head_only:assert not backbone
  else:
   assert all(any(p is q for q in backbone[0]['params']) for p in gate)
   assert backbone[0]['lr']==2e-4 and backbone[0]['weight_decay']==1e-4
 restored=build_dual_glt_model('token',**OPTIONS);add_ft_gate(restored);strict_restore(restored,state)
 with pytest.raises(ValueError):strict_restore(restored,before)
 bad=dict(state);bad['token_fusion.gate.bias']=torch.tensor([float('nan')])
 with pytest.raises(ValueError):strict_restore(restored,bad)

def test_freeze_complete_binding_and_missing_unit(tmp_path,monkeypatch):
 root=tmp_path;out=root/'result';parent=root/'parent.pt'
 monkeypatch.setattr(selection,'ROOT',root);monkeypatch.setattr(selection,'OUT',out);monkeypatch.setattr(selection,'PARENT',parent)
 monkeypatch.setattr(selection,'ft_folder',lambda a,t,f:out/'finetune'/a/t/f'fold{f}')
 def write(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
 scaler=dict(mean=[1.],scale=[2.],fit='train',transform='standard')
 for a in selection.ARMS:write(out/'initialization'/f'{a}.json',dict(arm=a,parent=str(parent)))
 for t in selection.TASKS:
  write(root/'data/splits/mips_paper5_outer'/f'{t}.json',dict(folds=[dict(train_indices=[0,1,2],validation_indices=[3],test_indices=[5,4]) for _ in range(5)]))
  for a in selection.ARMS:
   for f in range(5):
    folder=selection.ft_folder(a,t,f);n=selection.epochs(t)
    write(folder/'metrics.json',dict(status='PASS',outer_test='NOT_RUN',epochs=n,updates=dict(attempted=n,completed=n)))
    write(folder/'scaler.json',scaler)
    torch.save(dict(config=dict(arm=a,parent=str(parent)),parent=str(parent),task=t,fold=f,seed=42+f,scaler_mean=[1.],scaler_scale=[2.],best_epoch=17,state_dict={'synthetic':torch.zeros(1)}),folder/'best.pt')
 missing=selection.ft_folder('DG','xc',4)/'metrics.json';saved=missing.read_text();missing.unlink()
 with pytest.raises(FileNotFoundError):selection.freeze()
 assert not (out/'evaluation_before_test.json').exists()
 missing.write_text(saved);selection.freeze()
 frozen=json.loads((out/'evaluation_before_test.json').read_text());assert len(frozen['units'])==105
 assert all(u['test_indices']==[4,5] and u['parent']==str(parent) and u['best_epoch']==17 and u['scaler']==scaler for u in frozen['units'])
 with pytest.raises(FileExistsError):selection.freeze()

def test_conditional_effects_and_complete_macro_inputs():
 result=selection.conditional_effects(.7,.72,.73,.76)
 assert result==pytest.approx(dict(D_minus_F_CE=.02,G_minus_F_CE=.03,DG_minus_D=.04,DG_minus_G=.03,interaction=.01))
 tasks={t:[.8]*5 for t in selection.TASKS};assert selection.macro_r2(tasks)==pytest.approx(.8)
 tasks.pop('xc')
 with pytest.raises(ValueError):selection.macro_r2(tasks)
