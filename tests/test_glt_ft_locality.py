"""The sole S2 final CPU suite: locality semantics, gradients and strict restore."""
import copy,json,math
from pathlib import Path
import pytest
import torch
from scripts import run_glt_ft_locality_unit as unit
from scripts import select_glt_ft_locality as selection
from src.modules.glt_dual_opt import BondTokenFusion


def inputs():
 atoms=torch.randn(6,512,requires_grad=True);bonds=torch.randn(8,512,requires_grad=True)
 ag=torch.tensor([0,0,0,1,1,2]);bg=torch.tensor([0,0,0,1,1,1,2,2])
 valid=torch.tensor([True,True,False])
 # IDs 0 and 1 stand for distinct left/right physical-image keys at the same center.
 # The repeated pair (1,0) denotes presence once, not a second bias increment.
 inc=torch.tensor([[0,1,1,2,3,4,5,6],[0,0,0,1,3,3,4,5]])
 return atoms,bonds,ag,bg,valid,inc


def reference(module,atoms,bonds,ag,bg,valid,inc=None,mode='global',beta=0.0):
 rows=[]
 for a in range(atoms.shape[0]):
  if not valid[ag[a]]:rows.append(atoms[a]);continue
  keys=torch.where(bg==ag[a])[0]
  incident=keys.new_empty(0) if inc is None else torch.unique(inc[0,inc[1]==a])
  if mode=='incident_only':keys=incident
  if keys.numel()==0:rows.append(atoms[a]);continue
  q=module.q(atoms[a]).reshape(4,1,32)
  k=module.k(bonds[keys]).reshape(-1,4,32).transpose(0,1)
  v=module.v(bonds[keys]).reshape(-1,4,32).transpose(0,1)
  score=q.float()@k.float().transpose(-1,-2)/math.sqrt(32)
  if mode=='incident_bias':score=score+beta*torch.isin(keys,incident).view(1,1,-1)
  update=(torch.softmax(score,-1).to(v.dtype)@v).transpose(0,1).reshape(128)
  rows.append(atoms[a]+module.scale*module.output(update))
 return torch.stack(rows)


def test_default_global_zero_bias_and_all_incident_equivalence():
 x=inputs();m=BondTokenFusion();original=m(*x[:5])
 torch.testing.assert_close(original,reference(m,*x[:5]))
 state={k:v.clone() for k,v in m.state_dict().items()};rng=torch.get_rng_state().clone()
 m.set_attention_mode('incident_bias',0.0)
 torch.testing.assert_close(m(*x[:5],incidence=x[5]),original,rtol=0,atol=0)
 pairs=[(int(b),int(a)) for a in range(6) for b in range(8) if x[2][a]==x[3][b]]
 all_inc=torch.tensor(pairs).T.contiguous()
 m.set_attention_mode('incident_only',0.0)
 torch.testing.assert_close(m(*x[:5],incidence=all_inc),original)
 assert torch.equal(rng,torch.get_rng_state()) and set(m.state_dict())==set(state)
 assert not list(m.named_buffers())
 for k,v in state.items():assert torch.equal(v,m.state_dict()[k])


@pytest.mark.parametrize('mode,beta',[('incident_only',0.0),('incident_bias',math.log(2.0))])
def test_local_outputs_gradients_and_graph_isolation(mode,beta):
 x=inputs();m=BondTokenFusion(attention_mode=mode,incident_bias=beta)
 actual,attention=m(*x[:5],incidence=x[5],return_attention=True)
 expected=reference(m,*x[:5],x[5],mode,beta)
 torch.testing.assert_close(actual,expected)
 weight=torch.randn_like(actual)
 parameters=[x[0],x[1],*m.parameters()]
 ga=torch.autograd.grad((actual*weight).sum(),parameters,retain_graph=True)
 gr=torch.autograd.grad((expected*weight).sum(),parameters)
 for a,b in zip(ga,gr):torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
 assert ga[1].abs().sum()>0 and ga[0].abs().sum()>0
 changed=list(x);changed[1]=x[1].clone();changed[1][3:6]+=100.
 torch.testing.assert_close(m(*changed[:5],incidence=x[5])[:3],actual[:3])
 torch.testing.assert_close(actual[5],x[0][5],rtol=0,atol=0)
 assert [a[0] for a in attention]==[0,1]


@pytest.mark.parametrize('mode,beta',[('incident_only',0.0),('incident_bias',math.log(2.0))])
def test_true_physical_candidates_duplicates_orphans_and_fixed_bias(mode,beta):
 x=inputs();m=BondTokenFusion(attention_mode=mode,incident_bias=beta)
 with torch.no_grad():
  m.q.weight.zero_();m.q.bias.zero_();m.k.weight.zero_();m.k.bias.zero_();m.output.bias.fill_(7.)
 actual,att=m(*x[:5],incidence=x[5],return_attention=True);w=att[0][3]
 if mode=='incident_only':
  torch.testing.assert_close(w[0],torch.tensor([.5,.5,0.]))
  torch.testing.assert_close(w[1],torch.tensor([0.,0.,1.]))
  assert torch.count_nonzero(w[2])==0
  torch.testing.assert_close(actual[2],x[0][2],rtol=0,atol=0)
 else:
  torch.testing.assert_close(w[0],torch.tensor([.4,.4,.2]))
  torch.testing.assert_close(w[1],torch.tensor([.25,.25,.5]))
  torch.testing.assert_close(w[2],torch.full((3,),1/3))
 # Distinct physical image IDs both participate; removing either changes the rule.
 no_right=x[5][:,x[5][0]!=1]
 other=m(*x[:5],incidence=no_right)
 assert not torch.allclose(actual[0],other[0])
 empty=torch.empty((2,0),dtype=torch.long)
 empty_actual=m(*x[:5],incidence=empty)
 if mode=='incident_only':torch.testing.assert_close(empty_actual,x[0],rtol=0,atol=0)
 else:
  m.set_attention_mode('global',0.0);torch.testing.assert_close(empty_actual,m(*x[:5]),rtol=0,atol=0)


def test_illegal_indices_schema_and_config_rejected():
 x=inputs();m=BondTokenFusion(attention_mode='incident_only')
 bad=[None,x[5].float(),x[5].reshape(-1),torch.tensor([[8],[0]]),torch.tensor([[-1],[0]]),torch.tensor([[0],[3]])]
 for inc in bad:
  with pytest.raises(ValueError):m(*x[:5],incidence=inc)
 for mode,beta in [('unknown',0.0),('incident_only',math.log(2)),('global',1.0),('incident_bias',float('nan'))]:
  with pytest.raises(ValueError):m.set_attention_mode(mode,beta)
 cfg=json.loads((unit.OUT/'initialization/L_FT.json').read_text())
 for key in ('fusion_attention_mode','incident_bias','revision','cycle','parent'):
  wrong=dict(cfg);wrong.pop(key)
  with pytest.raises(ValueError):unit.validate_config(wrong,'L_FT')
 for key,val in [('fusion_attention_mode','global'),('incident_bias',math.log(2)),('joint_3d_drop_probability',.2)]:
  with pytest.raises(ValueError):unit.validate_config(dict(cfg,**{key:val}),'L_FT')


@pytest.mark.parametrize('arm',unit.ARMS)
def test_parent_alpha_and_stage_consistency_and_strict_package_restore(arm):
 cfg=json.loads((unit.OUT/'initialization'/f'{arm}.json').read_text())
 package=torch.load(unit.PARENT,map_location='cpu',weights_only=False)
 m=unit.LocalityArm(package,arm,0,cfg);state=m.model.state_dict()
 assert sum(p.numel() for p in m.parameters())==39876344
 for k,v in package['state_dict'].items():assert torch.equal(v,state[k])
 assert torch.equal(m.model.token_fusion.scale,package['state_dict']['token_fusion.scale'])
 x=inputs();graph3=torch.randn(3,512)
 def encode(_):return dict(atom_states=x[0],token_states=x[1],canonical_graph_index=x[2],bond_graph_index=x[3],geometry_valid=x[4],incidence_index=x[5],graph_3d=graph3)
 m.model.encode=encode
 from scripts.finetune_mcl_ph import periodic_tdl_trainability
 expected=m.features(None)
 for head_only in (True,False):
  periodic_tdl_trainability(m,head_only=head_only);m.train();torch.testing.assert_close(m.features(None),expected)
  m.eval();torch.testing.assert_close(m.features(None),expected)
 torch.testing.assert_close(expected[:2,512:],m.model.norm3(graph3)[:2])
 assert torch.count_nonzero(expected[2,512:])==0
 complete={k:v.clone() for k,v in m.state_dict().items()}
 restored=unit.LocalityArm(package,arm,0,cfg);unit.strict_restore(restored,complete)
 assert restored.model.token_fusion.attention_mode==cfg['fusion_attention_mode']
 assert restored.model.token_fusion.incident_bias==cfg['incident_bias']
 with pytest.raises(ValueError):unit.strict_restore(restored,{})


def test_complete_freeze_binding_and_no_partial_matrix(tmp_path,monkeypatch):
 out=tmp_path/'out';parent=tmp_path/'parent.pt'
 monkeypatch.setattr(selection,'ROOT',tmp_path);monkeypatch.setattr(selection,'OUT',out);monkeypatch.setattr(selection,'PARENT',parent)
 monkeypatch.setattr(unit,'PARENT',parent)
 monkeypatch.setattr(selection,'ft_folder',lambda a,t,f:out/'finetune'/a/t/f'fold{f}')
 def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
 configs={a:dict(json.loads((unit.OUT/'initialization'/f'{a}.json').read_text()),parent=str(parent)) for a in unit.ARMS}
 scaler=dict(mean=[1.],scale=[2.],fit='train',transform='standard')
 for a,c in configs.items():write(out/'initialization'/f'{a}.json',c)
 for t in unit.TASKS:
  write(tmp_path/'data/splits/mips_paper5_outer'/f'{t}.json',dict(folds=[dict(train_indices=[0],test_indices=[3,2]) for _ in range(5)]))
  for a in unit.ARMS:
   for f in range(5):
    folder=selection.ft_folder(a,t,f);n=selection.epochs(t)
    write(folder/'metrics.json',dict(status='PASS',outer_test='NOT_RUN',epochs=n,updates=dict(attempted=n,completed=n)))
    write(folder/'scaler.json',scaler)
    torch.save(dict(config=configs[a],parent=str(parent),task=t,fold=f,seed=42+f,scaler_mean=[1.],scaler_scale=[2.],best_epoch=7,state_dict={'synthetic':torch.zeros(1)}),folder/'best.pt')
 missing=selection.ft_folder('B_FT','xc',4)/'metrics.json';saved=missing.read_text();missing.unlink()
 with pytest.raises(FileNotFoundError):selection.freeze()
 assert not (out/'evaluation_before_test.json').exists()
 missing.write_text(saved);selection.freeze()
 frozen=json.loads((out/'evaluation_before_test.json').read_text());assert len(frozen['units'])==70
 for u in frozen['units']:
  assert u['test_indices']==[2,3] and u['best_epoch']==7 and u['scaler']==scaler
  assert u['fusion_attention_mode']==configs[u['arm']]['fusion_attention_mode'] and u['incident_bias']==configs[u['arm']]['incident_bias']
 with pytest.raises(FileExistsError):selection.freeze()


def test_paired_effects_complete_macro_and_strict_threshold():
 assert selection.conditional_effects(.7,.72,.73)==pytest.approx(dict(L_FT_minus_F_CE=.02,B_FT_minus_F_CE=.03,B_FT_minus_L_FT=.01))
 assert selection.macro_r2({t:[.8]*5 for t in unit.TASKS})==pytest.approx(.8)
 with pytest.raises(ValueError):selection.macro_r2({'eea':[.8]*5})
 assert not selection.qualifies(selection.REFERENCE+.009)
 assert selection.qualifies(selection.REFERENCE+.011)
