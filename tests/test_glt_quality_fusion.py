"""The sole final CPU suite: new quality, gate, restoration and selection semantics."""
from types import SimpleNamespace
import copy,json
import numpy as np
import pytest
import torch
from scripts.derive_glt_geometry_quality import classify,selected_qc,classify_with_qc,complete_persisted_quality
from scripts.select_glt_quality_fusion import macro_r2,qualifies,TASKS,REFERENCE
from src.modules.glt_dual_opt import BondTokenFusion
from src.modules.glt_dual_pretrain import DualPretrainer,deployment_package,load_deployment,global_objective
from src.modules.glt_dual import build_dual_glt_model
from src.dataset.glt_dual import build_dual_sample,dual_glt_collate
from src.dataset.glt_dual_pretrain import prepare_pretrain_sample,pretrain_collate
from src.training.glt_dual_runtime import load_sample_index_artifact
from src.utils import scale_targets
from tests.test_dual_glt import sample

OPTIONS=dict(geometry_revision='glt_periodic_seam_r1',geometry_mode='finite_incident',angle_value_enabled=True,readout='incident')

def quality_record(converged=True,valid=True):
 tri=SimpleNamespace(trimer_geometry_valid=valid,trimer_pos=torch.ones(3,3),generation_diagnostics={})
 if converged is not None:tri.selected_converged=converged
 return SimpleNamespace(trimer=tri,final_geometry_status='PASS' if valid else 'TIMEOUT',static={'geometry_valid':valid})

@pytest.mark.parametrize('status,eligible',[(0,True),(1,False),(-1,False)])
def test_complete_persisted_quality_preserves_original_evidence(status,eligible):
 old=classify(quality_record(None))
 qc=dict(energy=1.,atoms=3,diagnostics=dict(rounds=[dict(candidates=[dict(accepted=True,mmff_status=status,finite_post_mmff=True,finite_energy=True,mmff_energy_finite=True)])]))
 result=complete_persisted_quality(old,qc)
 assert result['quality_eligible']==eligible and old['reason']=='convergence_unverified'
 assert result['geometry_evidence']=='preserved_partial_ledger'
 qc['energy']=float('nan')
 with pytest.raises(ValueError):complete_persisted_quality(old,qc)

@pytest.mark.parametrize('converged,valid,eligible,reason',[(True,True,True,'selected_converged'),(False,True,False,'selected_not_converged'),(None,True,False,'convergence_unverified'),(False,False,False,'explicit_failure')])
def test_selected_quality(converged,valid,eligible,reason):
 result=classify(quality_record(converged,valid));assert result['quality_eligible']==eligible and result['reason']==reason

def test_quality_contradictions_and_corruption_stop():
 rec=quality_record();rec.static['geometry_valid']=False
 with pytest.raises(ValueError):classify(rec)
 rec=quality_record();rec.trimer.trimer_pos[0,0]=float('nan')
 with pytest.raises(ValueError):classify(rec)
 rec=quality_record();rec.trimer.trimer_conformer_round_id=0;rec.trimer.trimer_conformer_candidate_id=0
 rec.trimer.generation_diagnostics={'rounds':[{'candidates':[dict(accepted=True,round_id=0,candidate_id=0,final_valid=True,mmff_status=1)]}]}
 with pytest.raises(ValueError,match='contradiction'):classify(rec)

def test_quality_index_keeps_original_validation(tmp_path):
 original=tmp_path/'original';original.mkdir()
 np.savez(original/'pretrain_indices.npz',train_source_indices=np.array([0,1,2],dtype=np.int64),validation_source_indices=np.array([3,4],dtype=np.int64),fixed_validation_source_indices=np.array([4],dtype=np.int64))
 (original/'summary.json').write_text(json.dumps(dict(schema='mcl-ph-next-direct-index-v1',pi1m_rows=5,pretrain_train_count=3,pretrain_validation_count=2,pretrain_fixed_validation_count=1,excluded_overlap=0)))
 (tmp_path/'index.json').write_text(json.dumps(dict(schema='glt-quality-index-v1',original_index=str(original/'summary.json'))))
 np.save(tmp_path/'train_indices.npy',np.array([0,2],dtype=np.int64))
 assert load_sample_index_artifact(tmp_path/'index.json','train')['indices'].tolist()==[0,2]
 assert load_sample_index_artifact(tmp_path/'index.json','fixed_validation')['indices'].tolist()==[4]
 np.save(tmp_path/'train_indices.npy',np.array([0,3],dtype=np.int64))
 with pytest.raises(ValueError):load_sample_index_artifact(tmp_path/'index.json','train')

@pytest.mark.parametrize('text',('*C*','*CC*','*COC*'))
def test_ineligible_geometry_keeps_2d_and_disables_all_geometry_targets(text):
 top,tri,_=sample(text);top.geometry_quality_eligible=False
 data,labels=prepare_pretrain_sample(top,tri,text,seed=42,key='synthetic',position=0,opt_mode='GLT_FUSION_R1',geometry_mode='finite_incident',sigma=.03)
 assert not data.geometry_valid and data.mips_x.shape==top.mips_x.shape
 assert labels['distance'].numel()==0 and labels['angle_pairs'].numel()==0
 batch,lab=pretrain_collate([(data,labels)])
 model=DualPretrainer('token',fusion_revision='glt_fusion_r1',atom_ce_input='fused',fusion_gate='incident',align_enabled=True,**OPTIONS).eval()
 result=model(batch,lab)
 assert result['counts'][1]==0 and result['extra_counts'][0]==0
 encoded=model.encoder.encode(batch,atom_mask=lab['atom_mask']);features=model.encoder.fuse(encoded)
 assert torch.count_nonzero(features[:,512:])==0
 torch.testing.assert_close(encoded['fused_atom_states'],encoded['atom_states'])


def test_gate_initialization_graph_isolation_and_gradients():
 torch.manual_seed(42);old=BondTokenFusion();gate=BondTokenFusion('incident')
 gate.load_state_dict(dict(gate.state_dict(),**old.state_dict()))
 assert gate.gate.weight.numel()+gate.gate.bias.numel()==1025
 atoms=torch.randn(4,512,requires_grad=True);bonds=torch.randn(5,512,requires_grad=True)
 ai=torch.tensor([0,0,1,1]);bi=torch.tensor([0,0,0,1,1]);valid=torch.tensor([True,False])
 incidence=torch.tensor([[0,1,2,3,4],[0,0,1,2,3]])
 expected=old(atoms,bonds,ai,bi,valid);actual=gate(atoms,bonds,ai,bi,valid,incidence=incidence)
 torch.testing.assert_close(actual,expected);torch.testing.assert_close(actual[2:],atoms[2:])
 actual[:2].square().mean().backward()
 assert gate.gate.weight.grad.norm()>0 and bonds.grad[:3].norm()>0 and bonds.grad[3:].norm()==0
 with pytest.raises(ValueError):gate(atoms,bonds,ai,bi,valid)


def test_incident_gate_reads_real_seams_single_atom_and_multigraph():
 parts=[]
 for text in ('*C*','*CC*'):
  top,tri,_=sample(text)
  data=build_dual_sample(top,tri,text,geometry_mode='finite_incident')
  data.md200=torch.zeros((1,200))
  data.md200_valid=torch.zeros((1,),dtype=torch.bool)
  parts.append(data)
 batch=dual_glt_collate(parts)
 assert (batch.bond_length_weight==.5).any()
 model=build_dual_glt_model('token',fusion_revision='glt_fusion_r1',fusion_gate='incident',**OPTIONS).eval()
 encoded=model.encode(batch);fused=model.fuse(encoded)
 assert fused.shape==(2,1024) and torch.isfinite(fused).all()
 bond,atom=batch.incidence_index
 assert torch.equal(batch.bond_batch[bond],batch.canonical_graph_index[atom])


def test_deployment_restores_gate_and_old_default():
 for mode in ('incident','none'):
  pre=DualPretrainer('token',fusion_revision='glt_fusion_r1',fusion_gate=mode,align_enabled=True,align_weight=.01,align_warmup_steps=2000,quality_root='quality',**OPTIONS)
  package=deployment_package(pre,5000)
  model=build_dual_glt_model('token',fusion_revision='glt_fusion_r1',fusion_gate=mode,**OPTIONS)
  load_deployment(model,package)
  assert not any(n.startswith('align_proj') for n in package['state_dict'])
  if mode=='incident':
   with pytest.raises(ValueError):load_deployment(build_dual_glt_model('token',fusion_revision='glt_fusion_r1',**OPTIONS),package)
  else:
   package.pop('fusion_gate');load_deployment(model,package)


def test_weak_align_schedule_and_zero_valid_reduction():
 from scripts.pretrain_glt_dual import alignment_weight_at
 cfg={'align_enabled':True,'align_weight':.01,'align_warmup_steps':2000}
 assert alignment_weight_at(cfg,1000)==.005 and alignment_weight_at(cfg,2000)==.01 and alignment_weight_at(cfg,5000)==.01
 assert alignment_weight_at({'align_enabled':False},5000)==0
 assert alignment_weight_at({'align_enabled':True},1000)==.1
 sums=torch.tensor([2.,0.],requires_grad=True);counts=torch.tensor([2.,0.]);loss=global_objective(sums,counts,1,[.01,0.])
 assert torch.isfinite(loss) and float(loss)==pytest.approx(.01)


def test_scaler_fits_training_only_and_roundtrips_original_units():
 class Dataset:
  raw_targets=np.array([1.,3.,100.,200.]);targets=None
  def set_target_override(self,v):self.targets=v
 ds=Dataset();scaler=scale_targets(ds,'xc',train_indices=[0,1],transform_mode='standard')
 assert scaler.scaler.mean_.tolist()==[2.] and scaler.scaler.scale_.tolist()==[1.]
 np.testing.assert_allclose(scaler.inverse_transform(ds.targets).reshape(-1),ds.raw_targets)


def test_complete_macro_and_strict_absolute_threshold():
 values={t:[.5]*5 for t in TASKS};assert macro_r2(values)==.5
 assert not qualifies(.51,.5) and qualifies(.5100000001,.5)
 assert not qualifies(REFERENCE+.009) and qualifies(REFERENCE+.011)
 del values['xc']
 with pytest.raises(ValueError):macro_r2(values)
 values={t:[.5]*5 for t in TASKS};values['xc'][0]=float('nan')
 with pytest.raises(ValueError):macro_r2(values)


def test_qc_uses_only_selected_coordinate_lane():
 import sqlite3
 db=sqlite3.connect(':memory:')
 db.execute('CREATE TABLE rows (sample_key TEXT,final_status TEXT,coordinate_lane TEXT,main_detail TEXT,retry_detail TEXT)')
 def qc(status):return dict(selected_converged=status==0,round_id=1,candidate_id=2,energy=1.,atoms=3,diagnostics={'rounds':[{'candidates':[dict(accepted=True,round_id=1,candidate_id=2,final_valid=True,mmff_status=status)]}]})
 db.execute('INSERT INTO rows VALUES (?,?,?,?,?)',('identity','PASS','retry',json.dumps(qc(0)),json.dumps(qc(1))))
 chosen=selected_qc(db,'identity','PASS')
 assert not chosen['selected_converged']
 record=quality_record(None);assert not classify_with_qc(record,chosen)['quality_eligible']
 with pytest.raises(ValueError):selected_qc(db,'other','PASS')
 with pytest.raises(ValueError):selected_qc(db,'identity','TIMEOUT')
 db.close()
