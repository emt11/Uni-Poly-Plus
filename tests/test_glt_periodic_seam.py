"""One final CPU suite: synthetic local geometry, no scientific rerun."""
import copy
import json
from pathlib import Path
from contextlib import nullcontext
import pytest
import torch
from tests.test_dual_glt import sample
from src.dataset.glt_dual import build_dual_sample,dual_glt_collate
from src.dataset.glt_dual_static import build_dual_static
from src.dataset.glt_dual_pretrain import prepare_pretrain_sample,pretrain_collate
from src.modules.glt_dual_pretrain import DualPretrainer,deployment_package,load_deployment,global_objective
from src.modules.glt_dual import build_dual_glt_model
from src.modules.glt_dual_opt import weighted_graph_loss
from src.dataset.glt_periodic_geometry import geometry_options


def options(mode):
 return dict(geometry_revision='glt_periodic_seam_r1',geometry_mode=mode,angle_value_enabled=True,readout='incident')


def record(text,mode,position=0):
 top,tri,_=sample(text)
 return prepare_pretrain_sample(top,tri,text,seed=42,key='synthetic',position=position,opt_mode='GLT_FUSION_R1',geometry_mode=mode,sigma=0)


@pytest.mark.parametrize('text',('*C*','*CC*','*COC*'))
def test_seam_queries_weights_and_center_angles(text):
 top,tri,_=sample(text);data,labels=record(text,'finite_incident')
 static=build_dual_static(top,tri,text)
 pa=torch.as_tensor(static['token_pos_index_a']).long();pb=torch.as_tensor(static['token_pos_index_b']).long()
 expected=(tri.trimer_ru_offset[pa].eq(0).float()+tri.trimer_ru_offset[pb].eq(0).float())*.5
 torch.testing.assert_close(data.bond_length_weight,expected)
 distances=(tri.trimer_pos[pa]-tri.trimer_pos[pb]).norm(dim=-1)
 torch.testing.assert_close(labels['distance'],distances[expected>0])
 assert data.geometry_valid and data.incidence_index.numel()
 assert labels['angle_pairs'].numel()
 assert (labels['angle_pairs'][:,0]<labels['angle_pairs'][:,1]).all()
 triplets=torch.as_tensor(static['angle_pos_triplet'])
 for pair in labels['angle_pairs']:
  rows=torch.where((data.line_path[:,:2]==pair).all(-1)&(data.line_mask.sum(-1)==1)&~data.line_is_self)[0]
  assert rows.numel()==1 and tri.trimer_ru_offset[triplets[rows[0],0,1]]==0
 model=DualPretrainer('token',fusion_revision='glt_fusion_r1',**options('finite_incident'))
 batch,lab=pretrain_collate([(data,labels)])
 assert model(batch,lab)['counts'][1]==1


def test_periodic_self_image_paths_and_collation():
 data,_=record('*C*','periodic_template')
 assert data.periodic_keys.shape==(1,3) and data.periodic_keys[0,2]!=0
 assert set(data.periodic_line_shift.tolist())=={-2,-1,0,1,2}
 assert data.periodic_incidence_index.size(1)==2
 assert ((data.periodic_line_source==data.periodic_line_target)&(data.periodic_line_shift!=0)).any()
 second,_=record('*CC*','periodic_template')
 batch=dual_glt_collate([data,second]);count=data.periodic_keys.size(0)
 assert (batch.periodic_bond_index[data.bond_distance.numel():]>=count).all()
 assert (batch.periodic_keys[count:,:2]>=1).all()
 assert torch.equal(batch.periodic_keys[count:,2],second.periodic_keys[:,2])


@pytest.mark.parametrize('mode',('finite_incident','periodic_template'))
def test_angle_changes_single_bond_state(mode):
 data,_=record('*C*',mode);batch=dual_glt_collate([data])
 model=build_dual_glt_model('token',fusion_revision='glt_fusion_r1',**options(mode)).eval()
 changed=batch.clone()
 name='periodic_line_angle' if mode=='periodic_template' else 'line_angle'
 maskname='periodic_line_self' if mode=='periodic_template' else 'line_is_self'
 getattr(changed,name)[~getattr(changed,maskname),0]*=.7
 with torch.no_grad():a=model.glt(batch)['token_states'];b=model.glt(changed)['token_states']
 assert not torch.allclose(a,b,rtol=1e-6,atol=1e-6)


@pytest.mark.parametrize('mode',('finite_incident','periodic_template'))
@pytest.mark.parametrize('ce',('raw','fused'))
def test_ce_gradient_and_all_image_unknown(mode,ce):
 records=[record('*COC*',mode),record('*CCN*',mode,1)];data,labels=pretrain_collate(records)
 assert (data.bond_z_a[labels['atom_mask'][data.bond_canonical_a]]==0).all()
 assert (data.bond_z_b[labels['atom_mask'][data.bond_canonical_b]]==0).all()
 model=DualPretrainer('token',fusion_revision='glt_fusion_r1',atom_ce_input=ce,**options(mode))
 model(data,labels)['sums'][0].backward()
 for module in (model.encoder.token_fusion,model.encoder.glt):
  norm=sum(float(p.grad.norm()) for p in module.parameters() if p.grad is not None)
  assert norm>0 if ce=='fused' else norm==0


def test_same_physical_targets_and_unaveraged_error():
 finite,flab=record('*COC*','finite_incident');periodic,plab=record('*COC*','periodic_template')
 for name in ('distance','distance_weight','angle_cos','angle_pairs'):torch.testing.assert_close(flab[name],plab[name])
 # Unequal left/right observations retain irreducible error after quotient.
 errors=torch.tensor([1.,9.],requires_grad=True);weights=torch.tensor([.5,.5]);idx=torch.zeros(2,dtype=torch.long)
 means,valid=weighted_graph_loss(errors,weights,idx,2)
 torch.testing.assert_close(means,torch.tensor([5.,0.]));assert valid.tolist()==[True,False]
 local=torch.tensor([2.,3.,0.],requires_grad=True);counts=torch.tensor([4.,6.,0.])
 loss=global_objective(local,counts,4,(1.,1.,0.));loss.backward()
 torch.testing.assert_close(local.grad,torch.tensor([1.,2/3,0.]))


def test_frozen_template_translation_invariance():
 top,tri,_=sample('*CC*');one=build_dual_sample(top,tri,'*CC*',geometry_mode='periodic_template')
 tri2=copy.copy(tri);tri2.trimer_pos=tri.trimer_pos+torch.tensor([9.,-3.,2.])
 two=build_dual_sample(top,tri2,'*CC*',geometry_mode='periodic_template')
 for name in ('periodic_keys','periodic_bond_index','periodic_line_shift','periodic_line_path','periodic_line_angle'):
  torch.testing.assert_close(getattr(one,name),getattr(two,name),rtol=1e-5,atol=1e-5)


def test_new_package_restore_and_legacy_rejection():
 for mode in ('finite_incident','periodic_template'):
  trainer=DualPretrainer('token',fusion_revision='glt_fusion_r1',**options(mode))
  package=deployment_package(trainer,5000)
  model=build_dual_glt_model('token',fusion_revision='glt_fusion_r1',**options(mode))
  load_deployment(model,package)
  for name,value in package['state_dict'].items():torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
  broken=dict(package,geometry_mode='finite_internal')
  with pytest.raises(ValueError):load_deployment(model,broken)
  broken=copy.deepcopy(package);key=next(iter(broken['state_dict']));broken['state_dict'][key]=broken['state_dict'][key].double()
  with pytest.raises(ValueError,match='dtype'):load_deployment(model,broken)
 legacy=DualPretrainer('concat');package=deployment_package(legacy,5000)
 for name in ('geometry_revision','geometry_mode','angle_value_enabled'):package.pop(name)
 load_deployment(build_dual_glt_model('concat'),package)
 with pytest.raises(ValueError):geometry_options(None,'periodic_template',True,'incident')


def test_torsion_off_transmitted(monkeypatch):
 import src.dataset.glt_dual_pretrain as module
 top,tri,_=sample('*CC*');static=build_dual_static(top,tri,'*CC*');seen=[]
 def fields(static,z,positions,mapping,mode,table):
  seen.append(mode);return torch.empty(0,dtype=torch.long),torch.empty((0,2)),torch.empty(0,dtype=torch.bool)
 monkeypatch.setattr(module,'_torsion_fields',fields)
 build_dual_sample(top,tri,'*CC*',static=static,torsion_mode='off')
 assert seen==['off']


def test_validation_failure_restores_rng_and_mode(tmp_path,monkeypatch):
 import scripts.evaluate_glt_pt_endpoint as module
 from src.training.glt_dual_runtime import rng_state
 model=torch.nn.Linear(2,2);before={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
 monkeypatch.setattr(module,'load_endpoint_subset',lambda path:(_ for _ in ()).throw(ValueError('synthetic source failure')))
 monkeypatch.setattr(torch.distributed,'barrier',lambda:None)
 cfg=dict(cycle_root=str(tmp_path),arm='F_RAW',sample_index_artifact='synthetic')
 with pytest.raises(ValueError,match='source failure'):module.evaluate_process_validation(model,cfg,1000,torch.device('cpu'),0,4)
 assert model.training and torch.equal(torch.get_rng_state(),rng)
 for k,v in before.items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)


def test_canonical_renumber_keeps_same_frozen_physical_states():
 from src.dataset.glt_periodic_geometry import attach_periodic_template
 top,tri,_=sample('*COC*');static=build_dual_static(top,tri,'*COC*')
 one=build_dual_sample(top,tri,'*COC*',static=static,geometry_mode='periodic_template')
 other=one.clone();perm=torch.tensor([2,0,1])
 other.bond_canonical_a=perm[one.bond_canonical_a];other.bond_canonical_b=perm[one.bond_canonical_b]
 other.incidence_index[1]=perm[one.incidence_index[1]]
 pa=torch.as_tensor(static['token_pos_index_a']).long();pb=torch.as_tensor(static['token_pos_index_b']).long()
 attach_periodic_template(other,tri.trimer_ru_offset[pa],tri.trimer_ru_offset[pb],torch.as_tensor(static['angle_pos_triplet']).long(),tri.trimer_ru_offset)
 model=build_dual_glt_model('token',fusion_revision='glt_fusion_r1',**options('periodic_template')).eval()
 with torch.no_grad():
  a=model.glt(dual_glt_collate([one]))['bond_states'];b=model.glt(dual_glt_collate([other]))['bond_states']
 torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-5)


def test_fixed_validation_cpu_mock_preserves_model_and_rng(tmp_path,monkeypatch):
 import numpy as np
 from types import SimpleNamespace
 from torch_geometric.data import Data
 import scripts.evaluate_glt_pt_endpoint as module
 import src.training.glt_dual_runtime as runtime
 import src.dataset.glt_dual_pretrain as preparation
 closed=[]
 source=SimpleNamespace(entries=[{'pretrain_role':'validation'} for _ in range(1024)],samples=[(b'x'*32,'synthetic') for _ in range(1024)],close=lambda:closed.append(True))
 class Source:
  def __len__(self):return 1024
  def __getattr__(self,name):return getattr(source,name)
 monkeypatch.setattr(module,'load_endpoint_subset',lambda path:{'indices':np.arange(1024)})
 monkeypatch.setattr(runtime,'open_source',lambda *a,**kw:(Source(),None))
 monkeypatch.setattr(runtime,'pretrain_source_fields',lambda *a,**kw:(None,None,None,None,None))
 monkeypatch.setattr(preparation,'prepare_pretrain_sample',lambda *a,**kw:None)
 def collate(records):
  n=len(records)
  return Data(graph_available=torch.ones(n,dtype=torch.bool),geometry_valid=torch.ones(n,dtype=torch.bool),bond_length_weight=torch.ones(n*2),bond_distance=torch.ones(n*2)),{}
 monkeypatch.setattr(preparation,'pretrain_collate',collate)
 monkeypatch.setattr(torch.distributed,'barrier',lambda:None)
 monkeypatch.setattr(torch.distributed,'all_reduce',lambda tensor:tensor.mul_(4))
 monkeypatch.setattr(torch,'autocast',lambda *a,**kw:nullcontext())
 class Model(torch.nn.Module):
  def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(1));self.calls=0
  def forward(self,data,labels):
   assert not self.training and not torch.is_grad_enabled()
   torch.rand(7);self.calls+=1;n=data.graph_available.numel()
   return dict(sums=torch.tensor([2*n,3*n,0.]),counts=torch.tensor([n,n,0]),raw_geometry_sums=torch.tensor([n,2*n]),angle_graphs=torch.tensor(n),targets=torch.tensor([n,2*n,n,0]))
 model=Model();rng=torch.get_rng_state().clone();state=model.weight.detach().clone()
 cfg=dict(cycle_root=str(tmp_path),arm='F_RAW',sample_index_artifact='mock',noise_sigma=.03,atom_mask_ratio=.3,opt_mode='GLT_FUSION_R1',geometry_mode='finite_incident')
 module.evaluate_process_validation(model,cfg,1000,torch.device('cpu'),0,4)
 assert model.calls==4 and model.training and closed==[True]
 assert torch.equal(torch.get_rng_state(),rng) and torch.equal(model.weight,state)
 metrics=json.loads((tmp_path/'pt_validation/F_RAW/step01000/metrics.json').read_text())
 assert metrics['presented']==1024 and metrics['padding']==0 and not metrics['checkpoint_selection']
 assert metrics['components']['atom_ce']['mean']==2


@pytest.mark.parametrize('mode',('finite_incident','periodic_template'))
def test_frozen_invalid_geometry_preserves_2d_and_zero_geometry_count(mode):
 top,tri,_=sample('*COC*');tri.trimer_geometry_valid=False
 data,labels=prepare_pretrain_sample(top,tri,'*COC*',seed=42,key='synthetic',position=0,opt_mode='GLT_FUSION_R1',geometry_mode=mode)
 batch,targets=pretrain_collate([(data,labels)])
 model=DualPretrainer('token',fusion_revision='glt_fusion_r1',atom_ce_input='fused',**options(mode))
 result=model(batch,targets)
 assert result['counts'][0]==1 and result['counts'][1]==0 and result['sums'][1]==0
 assert torch.isfinite(result['sums']).all() and not model.encoder.encode(batch)['geometry_valid'].any()
