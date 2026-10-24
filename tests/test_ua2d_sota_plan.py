"""One final CPU suite; synthetic correctness cases, no parent or training runs."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.training import ua2d_sota as s
from src.training import ua2d_sota_pt as pt


def test_train_bandwidth_and_query_separation():
    train=np.asarray([[1.,0.],[0.,1.],[-1.,0.]])
    _,q,bandwidth,u=s.graph_kernel(train,np.asarray([[.1,.9]]))
    _,other,other_bandwidth,_=s.graph_kernel(train,np.asarray([[500.,1.],[-2.,1.]]))
    assert bandwidth==other_bandwidth
    expected=np.median(s.distances(u,u)[~np.eye(3,dtype=bool)])
    assert bandwidth==pytest.approx(expected)
    assert np.allclose(q,np.exp(-s.distances(s.normalize([[.1,.9]]),u)/bandwidth))
    assert other.shape==(2,3)
    with pytest.raises(ValueError,match='zero norm'): s.graph_kernel([[0.,0.]],[[1.,0.]])


def test_fixed_kernel_regressor_and_inverse_transform():
    from scipy.linalg import cho_factor,cho_solve
    train=np.asarray([[1.,0.],[0.,1.],[-1.,0.]])
    kg,query,_,_=s.graph_kernel(train,[[.1,.9]])
    kf=np.asarray([[1.,.2,.1],[.2,1.,.3],[.1,.3,1.]])
    for strategy in s.STRATEGIES:
        k=s.combine(kg,kf,strategy)
        assert np.allclose(k,k.T) and np.linalg.eigvalsh(k).min()>-1e-12
        if strategy=='JOINT': assert np.allclose(k,.75*(.5*kg+.5*kf)+.25*kg*kf)
    coef=cho_solve(cho_factor(kg+.1*np.eye(3)),np.asarray([-1.,0.,1.]))
    assert np.allclose((kg+.1*np.eye(3))@coef,[-1.,0.,1.])
    raw=2.+3.*(query@coef)
    assert np.allclose((raw-2.)/3.,query@coef)
    rows=[dict(alpha=a,rmse=.5) for a in s.ALPHAS]
    assert min(rows,key=lambda x:(x['rmse'],-x['alpha']))['alpha']==10.


def test_train_only_standardization():
    from src.utils import scale_targets
    class Small:
        raw_targets=np.asarray([1.,3.,10000.])
        def set_target_override(self,v): self.targets=v
    data=Small();sc=scale_targets(data,'eea',train_indices=[0,1],transform_mode='standard')
    assert sc.scaler.mean_.tolist()==[2.]
    assert sc.scaler.scale_.tolist()==[1.]
    assert np.allclose(sc.inverse_transform(data.targets.reshape(-1,1)).reshape(-1),data.raw_targets)


def test_residual_readout_mean_equivalence_and_graph_normalization():
    from src.modules.glt_dual import mean_pool
    torch.manual_seed(17);before=torch.get_rng_state().clone();readout=s.ResidualReadout()
    assert torch.equal(before,torch.get_rng_state())
    assert sum(p.numel() for p in readout.parameters())==32898
    atoms=torch.randn(5,512);index=torch.tensor([0,0,1,1,1])
    assert torch.equal(readout(atoms,index,2),mean_pool(atoms,index,2))
    with torch.no_grad(): readout.s.fill_(.5)
    together=readout(atoms,index,2)
    assert torch.allclose(together[0],readout(atoms[:2],torch.zeros(2,dtype=torch.long),1)[0])
    assert torch.allclose(together[1],readout(atoms[2:],torch.zeros(3,dtype=torch.long),1)[0])
    together.sum().backward()
    assert readout.scorer[0].weight.grad.abs().sum()>0


def test_p_single_factors_and_fixed_initialization():
    base=s.config('UA')
    for arm,key,value in [('P_BLR','backbone_lr',3e-5),('P_HLR','head_joint_lr',3e-4),('P_WD','weight_decay',.005)]:
        candidate=s.config(arm)
        assert {k for k in base['profile'] if base['profile'][k]!=candidate['profile'][k]}=={key}
        assert candidate['profile'][key]==value
        assert candidate['parent']==base['parent'] and candidate['head_root']==base['head_root']
    assert s.profile('P_COS')['joint_scheduler']==dict(name='CosineAnnealingLR',T_max=60,eta_min=1e-6)


def test_mcl_three_arms_match_capacity_and_initial_tensors():
    torch.manual_seed(99);before=torch.get_rng_state().clone()
    branches=[pt.MCLAuxiliary(a) for a in pt.ARMS[2:]]
    assert torch.equal(before,torch.get_rng_state())
    states=[b.state_dict() for b in branches]
    assert all(set(d)==set(states[0]) for d in states)
    assert all(torch.equal(states[0][k],d[k]) for d in states[1:] for k in d)
    assert branches[0].cutoffs==(4.,4.,4.)
    assert branches[1].cutoffs==branches[2].cutoffs==(2.,3.,4.)
    assert pt.pt_config('M_MS_CONST')['mcl_descriptor']=='zero'
    assert pt.pt_config('M_MS_PH')['mcl_descriptor']=='ph'


def test_mcl_center_heavy_space_and_identity_pairing():
    def view(n,center,key,position):
        return pt.MCLView(element=torch.ones(n,dtype=torch.long),charge=torch.ones(n,dtype=torch.long),
            aromatic=torch.ones(n,dtype=torch.long),center=torch.tensor(center),edges=torch.tensor([[0],[1]]),
            distance=torch.ones(1),edge_type=torch.zeros(1,dtype=torch.long),descriptor=torch.zeros(31,5),
            valid=True,key=key,identity='identity-'+key,position=position)
    a=view(3,[2,-1],'a',7);b=view(5,[0,4],'b',8)
    packed=pt.pack_mcl([a,b])
    assert packed.center.tolist()==[2,-1,3,7]
    assert packed.edges.tolist()==[[0,3],[1,4]]
    assert packed.keys==['a','b'] and packed.identities==['identity-a','identity-b'] and packed.positions==[7,8]
    b.center=torch.tensor([5])
    with pytest.raises(ValueError,match='heavy atom'): pt.pack_mcl([a,b])


def test_new_revision_strict_and_legacy_unchanged():
    for arm in pt.ARMS:
        cfg=pt.pt_config(arm);assert pt.validate_pt(cfg)==cfg
        bad=copy.deepcopy(cfg);bad['microbatch']=42
        with pytest.raises(ValueError,match='config mismatch'): pt.validate_pt(bad)
        with pytest.raises(ValueError,match='metadata mismatch'):
            pt.restore_new(dict(sota_config=cfg,step=4999),{})
    from src.modules.glt_dual_pretrain import u1_alignment_options
    with pytest.raises(ValueError): u1_alignment_options(dict(align_revision='glt_align_2d_u1',atom_ce_input='raw'))


def test_development_does_not_read_test_labels(monkeypatch):
    from src.training import glt_dual_runtime as runtime
    seen=[]
    manifest=dict(folds=[dict(train_indices=[0,2],validation_indices=[1],test_indices=[3])])
    def fake_read(path): return manifest
    def stop_open(*args,**kwargs):
        seen.extend(kwargs['selected_indices']);raise RuntimeError('source boundary')
    monkeypatch.setattr(s,'read',fake_read);monkeypatch.setattr(runtime,'open_source',stop_open)
    with pytest.raises(RuntimeError,match='source boundary'): s.Data('eea',0)
    assert seen==[0,1,2] and 3 not in seen
    with pytest.raises(ValueError): s.Data('eat',0)


def test_campaign_root_propagation_and_legacy_separation():
    from src.training.ua2d_sota_context import campaign_paths
    out,log=campaign_paths('r1_retest1')
    assert out==s.ROOT/'results/ua2d_sota_20261018/r1_retest1'
    assert log==s.ROOT/'logs/ua2d_sota_20261018/r1_retest1'
    with pytest.raises(ValueError,match='identity mismatch'):
        campaign_paths('r1_retest1',s.ROOT/'results/ua2d_sota_20261018/r1',log)
    assert pt.OUT==s.OUT and pt.CAMPAIGN==s.CAMPAIGN
    for arm in pt.ARMS:
        cfg=pt.pt_config(arm)
        assert cfg['cycle_root']==str(s.OUT)
        assert s.config(arm)['parent']==str(s.OUT/'pretrain'/arm/'deploy_05000.pt')
    assert s.unit_path('P_BLR','eea',0).is_relative_to(s.OUT)
    assert s.unit_path('UA','eea',0).is_relative_to(s.UA)
    assert s.kernel_folder('F0')==s.OUT/'kernel/F0'


def test_worker_budget_does_not_write_shared_ledger(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'OUT',tmp_path/'results');monkeypatch.setattr(s,'LOG',tmp_path/'logs')
    monkeypatch.setenv('UA2D_SOTA_ROLE','worker')
    actual=s.refresh_budget()
    assert actual['pt_starts']==actual['ft_starts']==actual['krr_fits']==0
    assert not (s.OUT/'budget.json').exists()


def test_freeze_gate_before_formal_access(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'OUT',tmp_path)
    s.write(tmp_path/'selection_before_test.json',dict(frozen=False))
    with pytest.raises(ValueError,match='before freeze'): s.frozen_neural_unit('A','eea',0)


def test_four_gpu_pt_mapping_preserves_two_rank_groups():
    from src.training.ua2d_sota_scheduler import validate_pt_mapping
    assert validate_pt_mapping(2,'0,1','r1_retest1','four_gpu_v1')==(0,1)
    assert validate_pt_mapping(2,'2,3','r1_retest1','four_gpu_v1')==(2,3)
    for world,visible in [(4,'0,1,2,3'),(2,'0,2'),(2,'1,2'),(2,'3,2')]:
        with pytest.raises(ValueError): validate_pt_mapping(world,visible,'r1_retest1','four_gpu_v1')
    with pytest.raises(ValueError): validate_pt_mapping(2,'0,1','r1','four_gpu_v1')
    with pytest.raises(ValueError): validate_pt_mapping(2,'0,1','r1_retest1','')
    assert pt.pt_config('T_RAWCE')['global_batch']==1008
    assert pt.pt_config('T_RAWCE')['microbatch']==84
    assert pt.pt_config('T_RAWCE')['expected_world_size']==2


def test_four_gpu_attempt_classification_never_retries(tmp_path):
    from src.training.ua2d_sota_scheduler import classify_attempt
    log=tmp_path/'log';output=tmp_path/'output';metric=output/'metrics.json'
    assert classify_attempt(log,metric,output)=='pending'
    log.mkdir();(log/'command.json').write_text('{}')
    assert classify_attempt(log,metric,output)=='active'
    (log/'exit.txt').write_text('143\n')
    assert classify_attempt(log,metric,output)=='interrupted'
    output.mkdir();metric.write_text('{"status":"PASS"}')
    assert classify_attempt(log,metric,output)=='interrupted'
    (log/'exit.txt').write_text('0\n')
    assert classify_attempt(log,metric,output)=='success'


def test_four_gpu_pair_reservations_do_not_mix_ft_and_pt():
    from src.training.ua2d_sota_scheduler import occupied_gpus,available_pair
    active=[dict(kind='pt',gpus=[0,1]),dict(kind='ft',gpus=[2]),dict(kind='ft',gpus=[2])]
    pt_used,ft=occupied_gpus(active)
    assert pt_used=={0,1} and ft[2]==2 and ft[3]==0
    assert not available_pair((2,3),active,{g:0 for g in range(4)})
    assert available_pair((2,3),active[:1],{g:0 for g in range(4)})
    with pytest.raises(RuntimeError): occupied_gpus(active+[dict(kind='ft',gpus=[0])])
    with pytest.raises(RuntimeError): occupied_gpus(active+[dict(kind='ft',gpus=[2])])
