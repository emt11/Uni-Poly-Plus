"""S1 sole final CPU suite: branch semantics, gradients, RNG and complete freeze."""
import json
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from scripts.run_glt_ft_branch_dropout_unit import BranchDropoutArm,branch_view,branch_fuse,strict_restore,PARENT
from scripts import select_glt_ft_branch_dropout as selection
from src.modules.glt_dual_opt import BondTokenFusion
from src.modules.glt_dual import mean_pool

def fixture():
 return dict(atom_states=torch.randn(6,512,requires_grad=True),token_states=torch.randn(6,512,requires_grad=True),
  canonical_graph_index=torch.tensor([0,0,1,1,2,2]),bond_graph_index=torch.tensor([0,0,1,1,2,2]),
  geometry_valid=torch.tensor([True,True,False]),graph_3d=torch.randn(3,512,requires_grad=True))

def small_model():
 return SimpleNamespace(token_fusion=BondTokenFusion(),norm2=nn.LayerNorm(512),norm3=nn.LayerNorm(512))

@pytest.mark.parametrize('arm',['F','R'])
def test_branch_outputs_gradients_invalid_and_cache(arm):
 model=small_model();model.norm3.bias.data.fill_(3.)
 encoded=fixture();encoded['fused_atom_states']=torch.full((6,512),123.)
 cached=encoded['fused_atom_states'];valid=encoded['geometry_valid'].clone()
 keep=torch.tensor([False,True,True]);view=branch_view(encoded,keep,arm)
 actual=branch_fuse(model,view)
 clean=branch_fuse(model,branch_view(encoded,torch.ones(3,dtype=torch.bool),arm))
 pure=model.norm2(mean_pool(encoded['atom_states'],encoded['canonical_graph_index'],3))
 torch.testing.assert_close(actual[1],clean[1]);torch.testing.assert_close(actual[2,:512],pure[2])
 assert torch.count_nonzero(actual[2,512:])==0
 if arm=='F':
  torch.testing.assert_close(actual[0,:512],pure[0]);torch.testing.assert_close(actual[0,512:],clean[0,512:])
 else:
  torch.testing.assert_close(actual[0,:512],clean[0,:512]);assert torch.count_nonzero(actual[0,512:])==0
 (actual[0]*torch.randn(1024)).sum().backward()
 if arm=='F':
  assert encoded['graph_3d'].grad[0].abs().sum()>0
  assert encoded['token_states'].grad is None or encoded['token_states'].grad[:2].abs().sum()==0
 else:
  assert encoded['token_states'].grad[:2].abs().sum()>0
  assert encoded['graph_3d'].grad[0].abs().sum()==0
 assert encoded['atom_states'].grad[:2].abs().sum()>0
 assert encoded['fused_atom_states'] is cached and torch.equal(encoded['geometry_valid'],valid)
 assert 'fused_atom_states' not in view and view['fusion_valid'][2]==False and view['readout_valid'][2]==False
 with pytest.raises(ValueError):branch_view(encoded,torch.ones(3),arm)

class TinyModel(nn.Module):
 def __init__(self):
  super().__init__();self.token_fusion=BondTokenFusion();self.norm2=nn.LayerNorm(512);self.norm3=nn.LayerNorm(512)
  self.predictor=nn.Linear(1024,1);self.saved=fixture()
 def encode(self,data):return dict(self.saved)

def tiny(arm,fold):
 m=BranchDropoutArm.__new__(BranchDropoutArm);nn.Module.__init__(m)
 m.model=TinyModel();m.arm=arm;m.drop_probability=.2;m.joint_enabled=False
 m.mask_generator=torch.Generator().manual_seed(10000+42+fold)
 m.counts=dict(joint_source_valid=0,fusion_closed=0,readout_closed=0)
 return m

def test_stage_rng_paired_with_D_and_global_isolation():
 f=tiny('F',2);r=tiny('R',2);d_rng=torch.Generator().manual_seed(10000+42+2)
 global_rng=torch.get_rng_state().clone()
 for m in (f,r):
  m.train();before=m.mask_generator.get_state().clone();m.features(None)
  assert torch.equal(before,m.mask_generator.get_state())
  m.joint_enabled=True
 for _ in range(20):
  keep=torch.rand(3,generator=d_rng)>=.2
  for m in (f,r):
   m.features(None);assert torch.equal(m.mask_generator.get_state(),d_rng.get_state())
  assert f.counts['fusion_closed']==r.counts['readout_closed']
 assert torch.equal(global_rng,torch.get_rng_state())
 assert f.counts['readout_closed']==r.counts['fusion_closed']==0
 for m in (f,r):
  m.eval();m.model.predictor.train();before=m.mask_generator.get_state().clone();m.features(None)
  assert torch.equal(before,m.mask_generator.get_state())
 assert f.dropout_snapshot()['joint_source_valid']==40

@pytest.mark.parametrize('arm',['F','R'])
def test_parent_restore_configuration_and_original_scale(arm):
 package=torch.load(PARENT,map_location='cpu',weights_only=False)
 model=BranchDropoutArm(package,arm,0)
 state=model.model.state_dict()
 assert set(state)-set(package['state_dict'])=={k for k in state if k.startswith('predictor.')}
 for key,value in package['state_dict'].items():assert torch.equal(state[key],value)
 strict_restore(model,model.state_dict())
 invalid=dict(package,geometry_mode='finite_internal')
 with pytest.raises(ValueError):BranchDropoutArm(invalid,arm,0)

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
 missing=selection.ft_folder('R','xc',4)/'metrics.json';saved=missing.read_text();missing.unlink()
 with pytest.raises(FileNotFoundError):selection.freeze()
 assert not (out/'evaluation_before_test.json').exists()
 missing.write_text(saved);selection.freeze()
 frozen=json.loads((out/'evaluation_before_test.json').read_text());assert len(frozen['units'])==70
 assert all(u['test_indices']==[4,5] and u['parent']==str(parent) and u['best_epoch']==17 and u['scaler']==scaler for u in frozen['units'])
 with pytest.raises(FileExistsError):selection.freeze()

def test_effects_macro_and_threshold():
 assert selection.conditional_effects(.7,.76,.72,.73)==pytest.approx(dict(F_minus_F_CE=.02,R_minus_F_CE=.03,D_minus_R=.03,D_minus_F=.04,interaction=.01))
 tasks={t:[.8]*5 for t in selection.TASKS};assert selection.macro_r2(tasks)==pytest.approx(.8)
 tasks.pop('xc')
 with pytest.raises(ValueError):selection.macro_r2(tasks)
 assert not selection.qualifies(selection.REFERENCE+.009)
 assert selection.qualifies(selection.REFERENCE+.011)
