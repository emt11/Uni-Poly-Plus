"""S3 unique final CPU checks, synthetic correctness plus saved initialization."""
import copy,json,types
import pytest
import torch
from scripts import run_glt_pt_locality_unit as unit
from scripts import select_glt_pt_locality as selection
from src.modules.glt_dual import build_dual_glt_model,locality_options
from src.modules.glt_dual_pretrain import DualPretrainer,deployment_package,load_deployment
from src.modules.glt_dual_opt import BondTokenFusion
from tests.test_dual_glt import sample
from src.dataset.glt_dual_pretrain import prepare_pretrain_sample,pretrain_collate

GEOMETRY=dict(geometry_revision='glt_periodic_seam_r1',geometry_mode='finite_incident',angle_value_enabled=True,readout='incident')

def model(arm):return build_dual_glt_model('token',fusion_revision='glt_fusion_r1',dropout=0,**GEOMETRY,**unit.connection_config(arm)).eval()

def batch():
 text='*COC*';top,tri,_=sample(text)
 return pretrain_collate([prepare_pretrain_sample(top,tri,text,seed=42,key='synthetic',position=0,opt_mode='GLT_FUSION_R1',geometry_mode='finite_incident',sigma=0,ratio=0.3)])


def test_endpoint_masked_initial_injection_symmetry_and_gradients():
 data,labels=batch();m=model('M_A');saved={k:v.clone() for k,v in data.items() if torch.is_tensor(v)}
 with torch.no_grad():m.endpoint_mlp[-1].weight.copy_(torch.eye(512))
 initial=[];encoded=[];endpoint_inputs=[];layer_inputs=[]
 hooks=[m.o8.atom_embedding.register_forward_hook(lambda mod,args,out:initial.append(out)),
  m.o8.register_forward_hook(lambda mod,args,out:encoded.append(out[0])),
  m.endpoint_mlp.register_forward_pre_hook(lambda mod,args:endpoint_inputs.append(args[0])),
  m.glt.layers[0].register_forward_pre_hook(lambda mod,args:layer_inputs.append(args[0]))]
 result=m.encode(data,atom_mask=labels['atom_mask'])
 for h in hooks:h.remove()
 expected=(initial[0][data.bond_canonical_a]+initial[0][data.bond_canonical_b])*.5
 torch.testing.assert_close(endpoint_inputs[0],expected)
 assert not torch.allclose(initial[0],encoded[0])
 assert labels['atom_mask'].any()
 assert (data.bond_z_a[labels['atom_mask'][data.bond_canonical_a]]==0).all()
 assert (data.bond_z_b[labels['atom_mask'][data.bond_canonical_b]]==0).all()
 # Entire feature rows are masked before the shared 138->512 projection.
 changed=data.clone();changed.mips_x[labels['atom_mask']]=torch.randn_like(data.mips_x[labels['atom_mask']])*100
 changed.mips_backbone_mask[labels['atom_mask']]=~changed.mips_backbone_mask[labels['atom_mask']]
 masked=m.o8.atom_embedding(changed,atom_mask=labels['atom_mask'])
 torch.testing.assert_close(masked,initial[0],rtol=0,atol=0)
 swapped=data.clone();swapped.bond_canonical_a=data.bond_canonical_b;swapped.bond_canonical_b=data.bond_canonical_a
 torch.testing.assert_close(m.endpoint_increment(swapped,initial[0]),m.endpoint_increment(data,initial[0]),rtol=0,atol=0)
 # At the first geometry layer the increment is already present.
 zero_inputs=[]
 h=m.glt.layers[0].register_forward_pre_hook(lambda mod,args:zero_inputs.append(args[0]))
 m.glt(data,endpoint_increment=torch.zeros_like(endpoint_inputs[0]));h.remove()
 torch.testing.assert_close(layer_inputs[0]-zero_inputs[0],m.endpoint_mlp(expected),rtol=1e-4,atol=1e-6)
 result['bond_states'].square().sum().backward()
 assert m.endpoint_mlp[0].weight.grad.abs().sum()>0 and m.endpoint_mlp[2].weight.grad.abs().sum()>0
 assert m.o8.atom_embedding.projection.bias.grad.abs().sum()>0
 for k,v in saved.items():assert torch.equal(v,data[k])
 bad=data.clone();bad.bond_canonical_a=data.bond_canonical_a.float()
 with pytest.raises(ValueError):m.endpoint_increment(bad,initial[0])


@pytest.mark.parametrize('arm',('M_G','M_L'))
def test_zero_control_stays_zero_after_optimizer(arm):
 m=model(arm);data,labels=batch();initial=m.o8.atom_embedding(data,atom_mask=labels['atom_mask'])
 with torch.no_grad():m.endpoint_mlp[-1].weight.normal_()
 opt=torch.optim.AdamW(m.endpoint_mlp.parameters(),lr=2e-4,weight_decay=0)
 increment=m.endpoint_increment(data,initial)
 assert torch.count_nonzero(increment)==0
 increment.sum().backward();opt.step()
 assert torch.count_nonzero(m.endpoint_increment(data,initial))==0
 assert all(p.grad is not None and torch.count_nonzero(p.grad)==0 for p in m.endpoint_mlp.parameters())


def test_true_local_physical_copies_orphan_fallback_and_gradient():
 atoms=torch.randn(3,512,requires_grad=True);bonds=torch.randn(3,512,requires_grad=True)
 ag=torch.zeros(3,dtype=torch.long);bg=torch.zeros(3,dtype=torch.long)
 incidence=torch.tensor([[0,1,2],[0,0,1]])
 m=BondTokenFusion(attention_mode='incident_only')
 with torch.no_grad():m.q.weight.zero_();m.q.bias.zero_();m.k.weight.zero_();m.k.bias.zero_();m.output.bias.fill_(7.)
 out,attention=m(atoms,bonds,ag,bg,torch.tensor([True]),incidence=incidence,return_attention=True)
 torch.testing.assert_close(attention[0][3][0],torch.tensor([.5,.5,0.]))
 torch.testing.assert_close(out[2],atoms[2],rtol=0,atol=0)
 assert torch.count_nonzero(attention[0][3][2])==0
 out[:2].square().sum().backward();assert bonds.grad.abs().sum()>0 and atoms.grad.abs().sum()>0
 # Invalid geometry never re-enabled, including the nonzero output bias.
 torch.testing.assert_close(m(atoms,bonds,ag,bg,torch.tensor([False]),incidence=incidence),atoms,rtol=0,atol=0)
 data,_=batch()
 pairs=list(zip(data.bond_canonical_a.tolist(),data.bond_canonical_b.tolist()))
 assert len(pairs)>len(set(pairs)) # Physical image copies were not merged.
 assert data.incidence_index[0].max()<data.bond_distance.numel()


def test_public_step0_identical_endpoint_and_independent_rng():
 common=torch.load(unit.COMMON,map_location='cpu',weights_only=False)['model'];first=None
 for arm in unit.ARMS:
  state=torch.load(unit.OUT/'initialization'/f'{arm}_step0.pt',map_location='cpu',weights_only=False)['model']
  assert set(state)-set(common)=={'encoder.endpoint_mlp.0.weight','encoder.endpoint_mlp.2.weight'}
  for k,v in common.items():assert torch.equal(v,state[k])
  endpoint={k:v for k,v in state.items() if 'endpoint_mlp' in k}
  if first is None:first=endpoint
  else:
   for k,v in first.items():assert torch.equal(v,endpoint[k])
  assert torch.count_nonzero(state['encoder.endpoint_mlp.2.weight'])==0
 torch.manual_seed(42);legacy=build_dual_glt_model('token',fusion_revision='glt_fusion_r1',dropout=0,**GEOMETRY);rng=torch.get_rng_state().clone()
 torch.manual_seed(42);new=model('M_G')
 assert torch.equal(rng,torch.get_rng_state())
 for k,v in legacy.state_dict().items():assert torch.equal(v,new.state_dict()[k])


@pytest.mark.parametrize('arm',unit.ARMS)
def test_pt_ft_formal_mode_and_strict_restore(arm):
 cfg=unit.read(unit.OUT/'initialization'/f'{arm}.json');package=torch.load(unit.parent(arm),map_location='cpu',weights_only=False)
 m=unit.LocalityArm(package,arm,0,cfg)
 assert locality_options(package)==unit.connection_config(arm)
 assert m.model.token_fusion.attention_mode==cfg['fusion_attention_mode']
 assert m.model.endpoint_input_mode==cfg['endpoint_input_mode']
 from scripts.finetune_mcl_ph import periodic_tdl_trainability
 for head_only in (True,False):
  periodic_tdl_trainability(m,head_only=head_only)
  for training in (True,False):
   m.train(training)
   assert m.model.token_fusion.attention_mode==cfg['fusion_attention_mode'] and m.model.endpoint_input_mode==cfg['endpoint_input_mode']
 restored=model(arm);load_deployment(restored,package,5000)
 for k,v in package['state_dict'].items():assert torch.equal(v,restored.state_dict()[k])
 for key in unit.connection_config(arm):
  broken=dict(package);broken.pop(key)
  with pytest.raises(ValueError):load_deployment(restored,broken,5000)
 for key,value in [('endpoint_input_mode','raw'),('fusion_attention_mode','incident_bias'),('incident_bias',1.),('endpoint_init_seed',42)]:
  with pytest.raises(ValueError):load_deployment(restored,dict(package,**{key:value}),5000)
 broken=dict(package,state_dict=dict(package['state_dict']));broken['state_dict'].pop('endpoint_mlp.0.weight')
 with pytest.raises((ValueError,RuntimeError)):load_deployment(restored,broken,5000)
 best=torch.load(unit.ft_folder(arm,'eea',0)/'best.pt',map_location='cpu',weights_only=False);unit.strict_restore(m,best['state_dict'])
 assert m.model.token_fusion.attention_mode==cfg['fusion_attention_mode']


def test_old_package_compatibility_and_new_package_construction():
 old=DualPretrainer('token',fusion_revision='glt_fusion_r1',**GEOMETRY)
 package=deployment_package(old,5000)
 assert 'endpoint_revision' not in package
 load_deployment(build_dual_glt_model('token',fusion_revision='glt_fusion_r1',**GEOMETRY),package,5000)
 new=DualPretrainer('token',fusion_revision='glt_fusion_r1',**GEOMETRY,**unit.connection_config('M_A'))
 assert locality_options(deployment_package(new,5000))==unit.connection_config('M_A')


def test_freeze_complete_binding_and_reject_partial_matrix(tmp_path,monkeypatch):
 out=tmp_path/'out';parents={a:tmp_path/f'{a}.pt' for a in unit.ARMS}
 monkeypatch.setattr(selection,'ROOT',tmp_path);monkeypatch.setattr(selection,'OUT',out)
 monkeypatch.setattr(selection,'parent',lambda a:parents[a]);monkeypatch.setattr(unit,'parent',lambda a:parents[a])
 monkeypatch.setattr(selection,'ft_folder',lambda a,t,f:out/'finetune'/a/t/f'fold{f}')
 configs={a:dict(unit.read(unit.OUT/'initialization'/f'{a}.json'),parent=str(parents[a])) for a in unit.ARMS}
 scaler=dict(mean=[1.],scale=[2.],fit='train',transform='standard')
 for a,c in configs.items():unit.write(out/'initialization'/f'{a}.json',c)
 for t in unit.TASKS:
  unit.write(tmp_path/'data/splits/mips_paper5_outer'/f'{t}.json',dict(folds=[dict(train_indices=[0],test_indices=[3,2]) for _ in range(5)]))
  for a in unit.ARMS:
   for f in range(5):
    folder=selection.ft_folder(a,t,f);n=selection.epochs(t)
    unit.write(folder/'metrics.json',dict(status='PASS',outer_test='NOT_RUN',epochs=n,updates=dict(attempted=n,completed=n)))
    unit.write(folder/'scaler.json',scaler)
    torch.save(dict(config=configs[a],parent=str(parents[a]),task=t,fold=f,seed=42+f,scaler_mean=[1.],scaler_scale=[2.],best_epoch=7,state_dict={'synthetic':torch.zeros(1)}),folder/'best.pt')
 missing=selection.ft_folder('M_A','xc',4)/'metrics.json';saved=missing.read_text();missing.unlink()
 with pytest.raises(FileNotFoundError):selection.freeze()
 assert not (out/'evaluation_before_test.json').exists()
 missing.write_text(saved);selection.freeze();frozen=unit.read(out/'evaluation_before_test.json')
 assert len(frozen['units'])==105
 for u in frozen['units']:
  assert u['test_indices']==[2,3] and u['scaler']==scaler and u['best_epoch']==7 and u['parent']==str(parents[u['arm']])
  for k,v in unit.connection_config(u['arm']).items():assert u[k]==v
 with pytest.raises(FileExistsError):selection.freeze()
 for a,c in configs.items():
  for k in ('parent',*unit.connection_config(a)):
   broken=dict(c);broken.pop(k)
   with pytest.raises(ValueError):unit.validate_config(broken,a)


def test_paired_effects_and_strict_performance_gate():
 assert selection.conditional_effects(.7,.71,.73,.74)==pytest.approx(dict(M_G_minus_F_CE=.01,M_L_minus_F_CE=.03,M_A_minus_F_CE=.04,M_L_minus_M_G=.02,M_A_minus_M_L=.01))
 assert selection.macro_r2({t:[.8]*5 for t in unit.TASKS})==pytest.approx(.8)
 assert not selection.qualifies(selection.REFERENCE+.009) and selection.qualifies(selection.REFERENCE+.011)
