"""F1 sole final CPU suite; no training replay or previous suite regression."""
import copy
import pytest
import torch
from scripts import run_glt_ft_graph_interaction_unit as unit
from scripts import select_glt_ft_graph_interaction as selection
from scripts import finetune_mcl_ph as ft
from scripts import run_glt_align_2d_unit as u1


def test_zero_output_init_parameter_rank_and_rng():
 torch.manual_seed(91);before=torch.get_rng_state().clone();m=unit.GraphMultiplicative()
 assert torch.equal(before,torch.get_rng_state())
 assert sum(p.numel() for p in m.parameters())==98944
 assert m.left.weight.shape==(64,512) and m.right.weight.shape==(64,512) and m.output.weight.shape==(512,64)
 assert all(x.bias is not None for x in (m.left,m.right,m.output))
 assert torch.count_nonzero(m.output.weight)==0 and torch.count_nonzero(m.output.bias)==0
 x=torch.randn(4,1024);saved=x.clone();valid=torch.tensor([True,False,True,False]);out=m(x,valid)
 torch.testing.assert_close(out,x,rtol=0,atol=0);assert torch.equal(x,saved)
 second=unit.GraphMultiplicative()
 for k,v in m.state_dict().items():assert torch.equal(v,second.state_dict()[k])
 with pytest.raises(ValueError):m(x[:,:512],valid)


def test_invalid_bias_mask_graph_isolation_and_input_immutability():
 m=unit.GraphMultiplicative()
 with torch.no_grad():m.output.weight.normal_(0,.01);m.output.bias.fill_(7)
 x=torch.randn(3,1024,requires_grad=True);saved=x.detach().clone();valid=torch.tensor([True,False,True])
 out=m(x,valid);torch.testing.assert_close(out[1],x[1],rtol=0,atol=0)
 changed=x.detach().clone();changed[2]+=50
 torch.testing.assert_close(m(changed,valid)[0],out[0],rtol=0,atol=0)
 assert torch.equal(x.detach(),saved)
 torch.testing.assert_close(out[:,512:],x[:,512:],rtol=0,atol=0)
 out[:,:512].sum().backward();assert x.grad[0,512:].abs().sum()>0 and x.grad[2,:512].abs().sum()>0
 assert x.grad[1,512:].abs().sum()==0


def test_first_joint_gradient_then_both_side_projections():
 m=unit.GraphMultiplicative();x=torch.randn(3,1024,requires_grad=True);valid=torch.ones(3,dtype=torch.bool);weight=torch.randn(3,512)
 (m(x,valid)[:,:512]*weight).sum().backward()
 assert m.output.weight.grad.abs().sum()>0 and m.output.bias.grad.abs().sum()>0
 assert m.left.weight.grad is not None and m.left.weight.grad.abs().sum()==0
 assert m.right.weight.grad is not None and m.right.weight.grad.abs().sum()==0
 # A nonzero output projection permits gradients into both product inputs.
 with torch.no_grad():m.output.weight.normal_(0,.01)
 m.zero_grad();x.grad=None
 (m(x,valid)[:,:512]*weight).sum().backward()
 for projection in (m.left,m.right):assert projection.weight.grad.abs().sum()>0 and projection.bias.grad.abs().sum()>0
 assert x.grad[:,:512].abs().sum()>0 and x.grad[:,512:].abs().sum()>0


def build():
 cfg=unit.read(unit.OUT/'initialization/F1_MUL.json')
 package=torch.load(unit.parent('F1_MUL'),map_location='cpu',weights_only=False)
 return unit.GraphInteractionArm(package,'F1_MUL',0,cfg),cfg,package


def test_public_parameters_rng_original_features_and_validity():
 cfg=unit.read(unit.OUT/'initialization/F1_MUL.json');package=torch.load(unit.parent('F1_MUL'),map_location='cpu',weights_only=False)
 torch.manual_seed(42);reference=u1.build_arm(package,'U0_FUSED',0,unit.read(unit.U1/'initialization/U0_FUSED.json'));rng=torch.get_rng_state().clone()
 torch.manual_seed(42);m=unit.GraphInteractionArm(package,'F1_MUL',0,cfg)
 assert torch.equal(rng,torch.get_rng_state())
 assert sum(p.numel() for p in m.parameters())==39975288
 for k,v in reference.model.state_dict().items():assert torch.equal(v,m.model.state_dict()[k])
 # Use native fusion with a synthetic final-invalid graph and nonzero norm3 bias.
 atoms=torch.randn(4,512);bonds=torch.randn(3,512)
 encoded=dict(atom_states=atoms,token_states=bonds,canonical_graph_index=torch.tensor([0,0,1,1]),
  bond_graph_index=torch.tensor([0,0,1]),geometry_valid=torch.tensor([True,False]),graph_3d=torch.randn(2,512),
  md200=torch.zeros(2,200),md200_valid=torch.zeros(2,dtype=torch.bool),incidence_index=torch.tensor([[0,1,2],[0,1,2]]))
 last={}
 def encode(_):
  value={k:v.clone() for k,v in encoded.items()};last.clear();last.update(value);return value
 m.model.encode=encode
 with torch.no_grad():m.model.norm3.bias.fill_(9);reference.model.norm3.bias.fill_(9)
 expected=reference.model.fuse(dict(encoded));actual=m.features(None)
 torch.testing.assert_close(actual,expected,rtol=0,atol=0)
 assert torch.count_nonzero(actual[1,512:])==0 and 'fused_atom_states' not in last
 with torch.no_grad():m.interaction.output.weight.normal_(0,.01);m.interaction.output.bias.fill_(7)
 actual=m.features(None);torch.testing.assert_close(actual[1],expected[1],rtol=0,atol=0)
 for k,v in encoded.items():assert torch.equal(v,last[k])


def test_head_only_and_joint_optimizer_membership_strict_restore():
 m,cfg,package=build()
 head=torch.load(unit.head_path('F1_MUL','eea',0),map_location='cpu',weights_only=False);unit.strict_restore(m.model.predictor,head)
 ft.configure_periodic_tdl_dropout(m,'xc',profile='uniform_u1')
 ft.periodic_tdl_trainability(m,head_only=True)
 opt,_=ft.periodic_tdl_optimizer(m,'xc',head_only=True,profile='uniform_u1')
 assert all(not p.requires_grad for p in m.interaction.parameters())
 members={id(p) for g in opt.param_groups for p in g['params']}
 assert not any(id(p) in members for p in m.interaction.parameters())
 ft.periodic_tdl_trainability(m,head_only=False)
 opt,evidence=ft.periodic_tdl_optimizer(m,'xc',head_only=False,profile='uniform_u1')
 backbone=next(g for g in opt.param_groups if g['name']=='backbone')
 assert backbone['lr']==1e-4 and backbone['weight_decay']==.02
 assert all(p.requires_grad and any(p is q for q in backbone['params']) for p in m.interaction.parameters())
 best=torch.load(unit.ft_folder('F1_MUL','eea',0)/'best.pt',map_location='cpu',weights_only=False)
 unit.validate_best(best,'F1_MUL','eea',0,cfg);unit.strict_restore(m,best['state_dict'])
 wrong=dict(best['state_dict']);wrong.pop('interaction.output.bias')
 with pytest.raises(ValueError):unit.strict_restore(m,wrong)
 wrong=dict(best['state_dict']);wrong['interaction.output.weight']=wrong['interaction.output.weight'].double()
 with pytest.raises(ValueError):unit.strict_restore(m,wrong)
 for key in unit.interaction_config():
  wrong=dict(cfg);wrong.pop(key)
  with pytest.raises(ValueError):unit.validate_config(wrong,'F1_MUL')
 for key,value in [('interaction_rank',32),('interaction_bias',False),('interaction_init_seed',42),('interaction_mask','before_output')]:
  with pytest.raises(ValueError):unit.validate_config(dict(cfg,**{key:value}),'F1_MUL')


def test_complete_freeze_binding_reject_partial(tmp_path,monkeypatch):
 cfg=unit.read(unit.OUT/'initialization/F1_MUL.json');out=tmp_path/'out'
 monkeypatch.setattr(selection,'ROOT',tmp_path);monkeypatch.setattr(selection,'OUT',out)
 monkeypatch.setattr(selection,'ft_folder',lambda a,t,f:out/'finetune'/a/t/f'fold{f}')
 unit.write(out/'initialization/F1_MUL.json',cfg);scaler=dict(mean=[1.],scale=[2.],fit='train',transform='standard')
 for task in unit.TASKS:
  unit.write(tmp_path/'data/splits/mips_paper5_outer'/f'{task}.json',dict(folds=[dict(train_indices=[0],test_indices=[3,2]) for _ in range(5)]))
  for fold in range(5):
   folder=selection.ft_folder('F1_MUL',task,fold)
   unit.write(folder/'metrics.json',dict(status='PASS',outer_test='NOT_RUN',epochs=70,updates=dict(attempted=70,completed=70)))
   unit.write(folder/'scaler.json',scaler)
   torch.save(dict(config=cfg,task=task,fold=fold,seed=42+fold,parent=str(unit.parent('F1_MUL')),scaler_mean=[1.],scaler_scale=[2.],best_epoch=7,
    graph_interaction=unit.interaction_config(),prediction_path=cfg['prediction_path'],input_dim=1024,head_initial=str(unit.head_path('F1_MUL',task,fold)),state_dict={'synthetic':torch.zeros(1)}),folder/'best.pt')
 missing=selection.ft_folder('F1_MUL','xc',4)/'metrics.json';saved=missing.read_text();missing.unlink()
 with pytest.raises(FileNotFoundError):selection.freeze()
 assert not (out/'evaluation_before_test.json').exists()
 missing.write_text(saved);selection.freeze();frozen=unit.read(out/'evaluation_before_test.json')
 assert frozen['validation_units']==35 and len(frozen['units'])==35
 for u in frozen['units']:
  assert u['test_indices']==[2,3] and u['scaler']==scaler and u['best_epoch']==7
  for k,v in unit.interaction_config().items():assert u[k]==v
 with pytest.raises(FileExistsError):selection.freeze()
